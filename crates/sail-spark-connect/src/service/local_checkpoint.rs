use std::any::Any;
use std::fs::{self, File};
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::ipc::reader::FileReader;
use datafusion::arrow::ipc::writer::FileWriter;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::{MemTable, Session, TableProvider};
use datafusion::common::{DataFusionError, Result};
use datafusion::logical_expr::{Expr, LogicalPlan, TableProviderFilterPushDown, TableType};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::common::collect;
use datafusion::physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use sail_common::spec;
use sail_common_datafusion::datasource::{SourceInfo, TableFormatRegistry};
use sail_common_datafusion::extension::SessionExtensionAccessor;
use sail_common_datafusion::session::job::{JobRunnerMode, JobService};
use sail_common_datafusion::session::remote_relation::{
    RemoteRelationBacking, RemoteRelationMaterializer,
};
use sail_execution::job_runner::ClusterJobRunner;
use sail_execution::{JobId, LocalCheckpointReadExec, LocalCheckpointWriteExec, TaskReadLocation};

use crate::executor::read_stream;

#[derive(Debug)]
pub(crate) struct SparkLocalCheckpointMaterializer;

#[async_trait]
impl RemoteRelationMaterializer for SparkLocalCheckpointMaterializer {
    async fn materialize(
        &self,
        state: &dyn Session,
        plan: &LogicalPlan,
        schema: SchemaRef,
        backing: &RemoteRelationBacking,
    ) -> Result<Arc<dyn TableProvider>> {
        let RemoteRelationBacking::LocalCache {
            storage_level,
            location,
            format,
            stream_job_id,
            ..
        } = backing
        else {
            return Err(DataFusionError::Execution(
                "local checkpoint materializer requires a local cache backing".to_string(),
            ));
        };
        let service = state.extension::<JobService>()?;

        self.cleanup(state, backing).await?;

        match service.runner().mode() {
            JobRunnerMode::Local => {
                materialize_local_checkpoint_locally(
                    state,
                    plan,
                    schema,
                    storage_level,
                    location,
                    format,
                )
                .await
            }
            JobRunnerMode::Cluster => {
                let Some(stream_job_id) = stream_job_id else {
                    return Err(DataFusionError::Execution(
                        "cluster localCheckpoint requires a checkpoint stream job id".to_string(),
                    ));
                };
                materialize_local_checkpoint_on_cluster(
                    state,
                    &service,
                    plan,
                    schema,
                    storage_level,
                    JobId::from(*stream_job_id),
                )
                .await
            }
        }
    }

    async fn cleanup(&self, state: &dyn Session, backing: &RemoteRelationBacking) -> Result<()> {
        match backing {
            RemoteRelationBacking::LocalCache {
                location,
                stream_job_id,
                ..
            } => {
                if let Some(stream_job_id) = stream_job_id {
                    cleanup_cluster_local_checkpoint(state, JobId::from(*stream_job_id)).await?;
                }
                if let Some(location) = location {
                    match fs::remove_dir_all(location) {
                        Ok(()) => {}
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                        Err(error) => return Err(error.into()),
                    }
                }
                Ok(())
            }
            _ => Err(DataFusionError::Execution(
                "local checkpoint cleanup requires a local cache backing".to_string(),
            )),
        }
    }
}

pub(crate) fn default_local_checkpoint_storage_level() -> spec::StorageLevel {
    spec::StorageLevel {
        use_disk: true,
        use_memory: true,
        use_off_heap: false,
        deserialized: false,
        replication: 1,
    }
}

pub(crate) fn validate_local_checkpoint_storage_level(
    storage_level: Option<spec::StorageLevel>,
) -> std::result::Result<spec::StorageLevel, &'static str> {
    let storage_level = storage_level.unwrap_or_else(default_local_checkpoint_storage_level);
    if storage_level.use_off_heap {
        return Err("localCheckpoint(offHeap) is not supported");
    }
    if storage_level.replication != 1 {
        return Err("localCheckpoint replication > 1 is not supported");
    }
    if !storage_level.use_memory && !storage_level.use_disk {
        return Err("localCheckpoint requires memory, disk, or both");
    }
    Ok(storage_level)
}

#[derive(Debug)]
enum LocalCheckpointMemory {
    Deserialized(Vec<Vec<RecordBatch>>),
    Serialized(Vec<Vec<u8>>),
}

#[derive(Debug)]
struct LocalCheckpointTableProvider {
    schema: SchemaRef,
    memory: Option<LocalCheckpointMemory>,
    disk_provider: Option<Arc<dyn TableProvider>>,
}

#[derive(Debug)]
struct ClusterLocalCheckpointTableProvider {
    schema: SchemaRef,
    locations: Vec<TaskReadLocation>,
}

#[async_trait]
impl TableProvider for LocalCheckpointTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match &self.memory {
            Some(LocalCheckpointMemory::Deserialized(partitions)) => Ok(Arc::new(
                MemTable::try_new(Arc::clone(&self.schema), partitions.clone())?,
            )
            .scan(state, projection, filters, limit)
            .await?),
            Some(LocalCheckpointMemory::Serialized(partitions)) => {
                let partitions = partitions
                    .iter()
                    .map(|bytes| deserialize_partition(bytes.as_slice()))
                    .collect::<Result<Vec<_>>>()?;
                Ok(
                    Arc::new(MemTable::try_new(Arc::clone(&self.schema), partitions)?)
                        .scan(state, projection, filters, limit)
                        .await?,
                )
            }
            None => {
                self.disk_provider
                    .as_ref()
                    .ok_or_else(|| {
                        DataFusionError::Execution(
                            "local checkpoint provider has no backing".to_string(),
                        )
                    })?
                    .scan(state, projection, filters, limit)
                    .await
            }
        }
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        if self.memory.is_some() {
            Ok(vec![
                TableProviderFilterPushDown::Unsupported;
                filters.len()
            ])
        } else {
            self.disk_provider
                .as_ref()
                .ok_or_else(|| {
                    DataFusionError::Execution(
                        "local checkpoint provider has no backing".to_string(),
                    )
                })?
                .supports_filters_pushdown(filters)
        }
    }
}

#[async_trait]
impl TableProvider for ClusterLocalCheckpointTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut plan: Arc<dyn ExecutionPlan> = Arc::new(LocalCheckpointReadExec::new(
            self.locations.clone(),
            Arc::clone(&self.schema),
        ));
        if let Some(projection) = projection {
            let exprs = projection
                .iter()
                .map(|index| {
                    let field = self.schema.fields().get(*index).ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "invalid local checkpoint projection index: {index}"
                        ))
                    })?;
                    Ok((
                        Arc::new(Column::new(field.name(), *index)) as Arc<dyn PhysicalExpr>,
                        field.name().to_string(),
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            plan = Arc::new(ProjectionExec::try_new(exprs, plan)?);
        }
        if let Some(limit) = limit {
            plan = Arc::new(GlobalLimitExec::new(
                Arc::new(LocalLimitExec::new(plan, limit)),
                0,
                Some(limit),
            ));
        }
        let _ = filters;
        Ok(plan)
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        Ok(vec![
            TableProviderFilterPushDown::Unsupported;
            filters.len()
        ])
    }
}

async fn materialize_local_checkpoint_locally(
    state: &dyn Session,
    plan: &LogicalPlan,
    schema: SchemaRef,
    storage_level: &spec::StorageLevel,
    location: &Option<String>,
    format: &Option<String>,
) -> Result<Arc<dyn TableProvider>> {
    if let Some(location) = location {
        fs::create_dir_all(location).map_err(DataFusionError::from)?;
    }

    let physical = state.create_physical_plan(plan).await?;
    let task_ctx = state.task_ctx();
    let partition_count = physical.output_partitioning().partition_count().max(1);
    let mut memory_partitions = Vec::with_capacity(partition_count);
    let mut serialized_partitions = Vec::with_capacity(partition_count);

    for partition in 0..partition_count {
        let stream = physical.execute(partition, Arc::clone(&task_ctx))?;
        let batches = collect(stream).await?;
        if storage_level.use_memory {
            if storage_level.deserialized {
                memory_partitions.push(batches.clone());
            } else {
                serialized_partitions.push(serialize_partition(&schema, batches.as_slice())?);
            }
        }
        if let Some(location) = location {
            let bytes = if storage_level.use_memory && !storage_level.deserialized {
                serialized_partitions.last().cloned().ok_or_else(|| {
                    DataFusionError::Execution("missing serialized partition".to_string())
                })?
            } else {
                serialize_partition(&schema, batches.as_slice())?
            };
            write_partition_file(location, partition, &bytes)?;
        }
    }

    let disk_provider =
        match (location.as_deref(), format.as_deref()) {
            (Some(location), Some(format)) => {
                Some(build_local_file_provider(state, location, format, Arc::clone(&schema)).await?)
            }
            (Some(_), None) | (None, Some(_)) => return Err(DataFusionError::Execution(
                "local checkpoint backing must include both location and format for disk storage"
                    .to_string(),
            )),
            (None, None) => None,
        };

    let memory = if storage_level.use_memory {
        Some(if storage_level.deserialized {
            LocalCheckpointMemory::Deserialized(memory_partitions)
        } else {
            LocalCheckpointMemory::Serialized(serialized_partitions)
        })
    } else {
        None
    };

    Ok(Arc::new(LocalCheckpointTableProvider {
        schema,
        memory,
        disk_provider,
    }))
}

async fn materialize_local_checkpoint_on_cluster(
    state: &dyn Session,
    service: &JobService,
    plan: &LogicalPlan,
    schema: SchemaRef,
    storage_level: &spec::StorageLevel,
    checkpoint_job_id: JobId,
) -> Result<Arc<dyn TableProvider>> {
    let Some(runner) = service.runner().as_any().downcast_ref::<ClusterJobRunner>() else {
        return Err(DataFusionError::Execution(
            "cluster localCheckpoint requires a cluster job runner".to_string(),
        ));
    };
    let physical = state.create_physical_plan(plan).await?;
    let partitions = physical.output_partitioning().partition_count().max(1);
    runner
        .begin_local_checkpoint_materialization(checkpoint_job_id, partitions)
        .await?;
    let write = Arc::new(LocalCheckpointWriteExec::new(
        physical,
        checkpoint_job_id,
        storage_level.clone(),
    ));
    let stream = service.runner().execute(state, write).await?;
    let _ = read_stream(stream)
        .await
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
    let locations = runner
        .finalize_local_checkpoint_materialization(checkpoint_job_id)
        .await?;
    Ok(Arc::new(ClusterLocalCheckpointTableProvider {
        schema,
        locations,
    }))
}

async fn cleanup_cluster_local_checkpoint(
    state: &dyn Session,
    checkpoint_job_id: JobId,
) -> Result<()> {
    let service = state.extension::<JobService>()?;
    match service.runner().mode() {
        JobRunnerMode::Local => Ok(()),
        JobRunnerMode::Cluster => {
            let Some(runner) = service.runner().as_any().downcast_ref::<ClusterJobRunner>() else {
                return Err(DataFusionError::Execution(
                    "cluster localCheckpoint cleanup requires a cluster job runner".to_string(),
                ));
            };
            runner
                .remove_local_checkpoint_materialization(checkpoint_job_id)
                .await
        }
    }
}

fn serialize_partition(schema: &SchemaRef, batches: &[RecordBatch]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut bytes, schema).map_err(DataFusionError::from)?;
        for batch in batches {
            writer.write(batch).map_err(DataFusionError::from)?;
        }
        writer.finish().map_err(DataFusionError::from)?;
    }
    Ok(bytes)
}

fn deserialize_partition(bytes: &[u8]) -> Result<Vec<RecordBatch>> {
    let reader = FileReader::try_new(Cursor::new(bytes), None).map_err(DataFusionError::from)?;
    reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(DataFusionError::from)
}

fn write_partition_file(location: &str, partition: usize, bytes: &[u8]) -> Result<()> {
    fs::create_dir_all(location).map_err(DataFusionError::from)?;
    let path = Path::new(location).join(format!("part-{partition:05}.arrow"));
    let mut file = File::create(path).map_err(DataFusionError::from)?;
    std::io::Write::write_all(&mut file, bytes).map_err(DataFusionError::from)
}

async fn build_local_file_provider(
    ctx: &dyn Session,
    location: &str,
    format: &str,
    schema: SchemaRef,
) -> Result<Arc<dyn TableProvider>> {
    let registry = ctx.extension::<TableFormatRegistry>()?;
    Ok(registry
        .get(format)?
        .create_provider(
            ctx,
            SourceInfo {
                paths: vec![location.to_string()],
                schema: Some(schema.as_ref().clone()),
                constraints: Default::default(),
                partition_by: vec![],
                bucket_by: None,
                sort_order: vec![],
                options: vec![],
            },
        )
        .await?)
}

#[cfg(test)]
mod tests {
    use sail_common::spec;

    use super::{default_local_checkpoint_storage_level, validate_local_checkpoint_storage_level};

    #[test]
    fn test_default_local_checkpoint_storage_level_uses_memory_and_disk() {
        assert_eq!(
            default_local_checkpoint_storage_level(),
            spec::StorageLevel {
                use_disk: true,
                use_memory: true,
                use_off_heap: false,
                deserialized: false,
                replication: 1,
            }
        );
    }

    #[test]
    fn test_validate_local_checkpoint_storage_level_rejects_off_heap() {
        let error = validate_local_checkpoint_storage_level(Some(spec::StorageLevel {
            use_disk: true,
            use_memory: true,
            use_off_heap: true,
            deserialized: false,
            replication: 1,
        }))
        .unwrap_err();
        assert!(error.contains("offHeap"));
    }
}
