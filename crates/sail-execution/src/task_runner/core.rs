use std::collections::HashMap;
use std::fmt::Formatter;
use std::sync::Arc;

use datafusion::arrow::datatypes::Schema;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::memory::DataSourceExec;
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::common::{exec_datafusion_err, internal_err};
use datafusion::datasource::physical_plan::{FileScanConfigBuilder, ParquetSource};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use datafusion_proto::physical_plan::AsExecutionPlan;
use datafusion_proto::protobuf::PhysicalPlanNode;
use futures::StreamExt;
use log::debug;
use prost::Message;
use sail_common::spec;
use sail_common_datafusion::error::CommonErrorCause;
use sail_delta_lake::physical_plan::DeltaPhysicalExprAdapterFactory;
use sail_python_udf::error::PyErrExtractor;
use sail_server::actor::{Actor, ActorContext};
use sail_telemetry::telemetry::global_metrics;
use sail_telemetry::{trace_execution_plan, TracingExecOptions};
use tokio::sync::oneshot;

use crate::codec::RemoteExecutionCodec;
use crate::driver::{DriverEvent, LocalCheckpointStreamOwner, TaskStatus};
use crate::error::{ExecutionError, ExecutionResult};
use crate::id::{JobId, TaskKey, TaskKeyDisplay, TaskStreamKey};
use crate::plan::{
    LocalCheckpointReadExec, LocalCheckpointWriteExec, ShuffleReadExec, ShuffleWriteExec,
    StageInputExec,
};
use crate::stream::error::TaskStreamError;
use crate::stream::writer::{
    LocalStreamStorage, TaskStreamSinkState, TaskStreamWriter, TaskWriteLocation,
};
use crate::stream_accessor::{StreamAccessor, StreamAccessorMessage};
use crate::task::definition::{TaskDefinition, TaskInput, TaskOutput};
use crate::task_runner::monitor::TaskMonitor;
use crate::task_runner::{LocalCheckpointRegistrarContext, TaskRunner, TaskRunnerMessage};

impl TaskRunner {
    pub fn new() -> Self {
        Self {
            signals: HashMap::new(),
            codec: Box::new(RemoteExecutionCodec),
        }
    }

    pub fn run_task<T: Actor>(
        &mut self,
        ctx: &mut ActorContext<T>,
        key: TaskKey,
        definition: TaskDefinition,
        context: Arc<TaskContext>,
        registrar: LocalCheckpointRegistrarContext,
    ) where
        T::Message: TaskRunnerMessage + StreamAccessorMessage,
    {
        let stream = match self.execute_plan(ctx, &key, definition, context, registrar) {
            Ok(x) => x,
            Err(e) => {
                let event = T::Message::report_task_status(
                    key,
                    TaskStatus::Failed,
                    Some(format!("failed to execute plan: {e}")),
                    Some(CommonErrorCause::new::<PyErrExtractor>(&e)),
                );
                ctx.send(event);
                return;
            }
        };
        let handle = ctx.handle().clone();
        let (tx, rx) = oneshot::channel();
        self.signals.insert(key.clone(), tx);
        let monitor = TaskMonitor::new(handle, key, stream, rx);
        ctx.spawn(monitor.run());
    }

    pub fn stop_task(&mut self, key: &TaskKey) {
        if let Some(signal) = self.signals.remove(key) {
            let _ = signal.send(());
        }
    }

    /// Deserializes and prepares a physical plan for execution on this node.
    fn execute_plan<T: Actor>(
        &mut self,
        ctx: &mut ActorContext<T>,
        key: &TaskKey,
        definition: TaskDefinition,
        context: Arc<TaskContext>,
        registrar: LocalCheckpointRegistrarContext,
    ) -> ExecutionResult<SendableRecordBatchStream>
    where
        T::Message: TaskRunnerMessage + StreamAccessorMessage,
    {
        let plan = PhysicalPlanNode::decode(definition.plan.as_ref())?;
        let plan = plan.try_into_physical_plan(&context, self.codec.as_ref())?;
        let plan = self.rewrite_parquet_adapters(plan)?;
        let plan = self.rewrite_shuffle(
            ctx,
            key,
            &definition.inputs,
            &definition.output,
            plan,
            &context,
            registrar,
        )?;
        debug!(
            "{} execution plan\n{}",
            TaskKeyDisplay(key),
            DisplayableExecutionPlan::new(plan.as_ref()).indent(true)
        );
        let options = TracingExecOptions {
            metrics: global_metrics(),
            job_id: Some(key.job_id.into()),
            stage: Some(key.stage),
            attempt: Some(key.attempt),
            operator_id: None,
        };
        let plan = trace_execution_plan(plan, options)?;
        let stream = plan.execute(key.partition, context)?;
        Ok(stream)
    }

    fn rewrite_parquet_adapters(
        &mut self,
        plan: Arc<dyn ExecutionPlan>,
    ) -> ExecutionResult<Arc<dyn ExecutionPlan>> {
        let result = plan.transform(|node| {
            if let Some(ds) = node.as_any().downcast_ref::<DataSourceExec>() {
                if let Some((base_config, _parquet)) = ds.downcast_to_file_source::<ParquetSource>()
                {
                    let adapter_factory = Arc::new(DeltaPhysicalExprAdapterFactory {});
                    let builder = FileScanConfigBuilder::from(base_config.clone())
                        .with_expr_adapter(Some(adapter_factory));
                    let new_exec = DataSourceExec::from_data_source(builder.build());
                    return Ok(Transformed::yes(new_exec as Arc<dyn ExecutionPlan>));
                }
            }
            Ok(Transformed::no(node))
        });
        Ok(result.data()?)
    }

    fn rewrite_shuffle<T: Actor>(
        &mut self,
        ctx: &mut ActorContext<T>,
        key: &TaskKey,
        inputs: &[TaskInput],
        output: &TaskOutput,
        plan: Arc<dyn ExecutionPlan>,
        context: &TaskContext,
        registrar: LocalCheckpointRegistrarContext,
    ) -> ExecutionResult<Arc<dyn ExecutionPlan>>
    where
        T::Message: TaskRunnerMessage + StreamAccessorMessage,
    {
        let handle = ctx.handle();
        let local_checkpoint_write = plan
            .as_any()
            .downcast_ref::<LocalCheckpointWriteExec>()
            .map(|checkpoint| {
                (
                    checkpoint.checkpoint_job_id(),
                    checkpoint.storage_level().clone(),
                    checkpoint.input().clone(),
                )
            });
        let plan = local_checkpoint_write
            .as_ref()
            .map(|(_, _, input)| input.clone())
            .unwrap_or(plan);
        let result = plan.transform(move |node| {
            if let Some(placeholder) = node.as_any().downcast_ref::<StageInputExec<usize>>() {
                let Some(input) = inputs.get(*placeholder.input()) else {
                    return internal_err!(
                        "stage input index {} out of bounds for {}",
                        placeholder.input(),
                        TaskKeyDisplay(key)
                    );
                };
                let locations = input.locations(key.job_id);
                let accessor = StreamAccessor::new(handle.clone());
                let shuffle = ShuffleReadExec::new(
                    locations,
                    Arc::new(accessor),
                    placeholder.properties().clone(),
                );
                Ok(Transformed::yes(Arc::new(shuffle)))
            } else if let Some(local_checkpoint) =
                node.as_any().downcast_ref::<LocalCheckpointReadExec>()
            {
                let accessor = StreamAccessor::new(handle.clone());
                let shuffle = ShuffleReadExec::new(
                    local_checkpoint
                        .locations()
                        .iter()
                        .cloned()
                        .map(|location| vec![location])
                        .collect(),
                    Arc::new(accessor),
                    local_checkpoint.properties().clone(),
                );
                Ok(Transformed::yes(Arc::new(shuffle)))
            } else {
                Ok(Transformed::no(node))
            }
        });
        let mut plan = result.data()?;
        if let Some((checkpoint_job_id, storage_level, _)) = local_checkpoint_write {
            let accessor = Arc::new(StreamAccessor::new(handle.clone()));
            plan = Arc::new(RuntimeLocalCheckpointWriteExec::new(
                plan,
                checkpoint_job_id,
                storage_level,
                accessor,
                registrar,
                key.clone(),
            ));
        }
        let schema = plan.schema();
        let accessor = StreamAccessor::new(handle.clone());
        let mut locations = vec![vec![]; plan.output_partitioning().partition_count()];
        match locations.get_mut(key.partition) {
            Some(x) => x.extend(output.locations(key)),
            None => {
                return Err(ExecutionError::InternalError(format!(
                    "invalid partition: {}",
                    TaskKeyDisplay(key)
                )));
            }
        };
        let partitioning = output.partitioning(context, &schema, self.codec.as_ref())?;
        let shuffle = ShuffleWriteExec::new(plan, locations, Arc::new(accessor), partitioning);
        Ok(Arc::new(shuffle))
    }
}

#[derive(Clone)]
struct RuntimeLocalCheckpointWriteExec {
    input: Arc<dyn ExecutionPlan>,
    checkpoint_job_id: JobId,
    storage_level: spec::StorageLevel,
    writer: Arc<dyn TaskStreamWriter>,
    registrar: LocalCheckpointRegistrarContext,
    task_key: TaskKey,
    properties: PlanProperties,
}

impl RuntimeLocalCheckpointWriteExec {
    fn new(
        input: Arc<dyn ExecutionPlan>,
        checkpoint_job_id: JobId,
        storage_level: spec::StorageLevel,
        writer: Arc<dyn TaskStreamWriter>,
        registrar: LocalCheckpointRegistrarContext,
        task_key: TaskKey,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(Arc::new(Schema::empty())),
            datafusion::physical_expr::Partitioning::UnknownPartitioning(
                input.output_partitioning().partition_count().max(1),
            ),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Self {
            input,
            checkpoint_job_id,
            storage_level,
            writer,
            registrar,
            task_key,
            properties,
        }
    }
}

impl std::fmt::Debug for RuntimeLocalCheckpointWriteExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeLocalCheckpointWriteExec")
            .field("checkpoint_job_id", &self.checkpoint_job_id)
            .field("task_key", &self.task_key)
            .field("storage_level", &self.storage_level)
            .finish()
    }
}

impl DisplayAs for RuntimeLocalCheckpointWriteExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "RuntimeLocalCheckpointWriteExec: checkpoint_job_id={}, task_partition={}, memory={}, disk={}, deserialized={}, replication={}",
            self.checkpoint_job_id,
            self.task_key.partition,
            self.storage_level.use_memory,
            self.storage_level.use_disk,
            self.storage_level.deserialized,
            self.storage_level.replication,
        )
    }
}

impl ExecutionPlan for RuntimeLocalCheckpointWriteExec {
    fn name(&self) -> &str {
        "RuntimeLocalCheckpointWriteExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let child = children.pop();
        match (child, children.is_empty()) {
            (Some(input), true) => Ok(Arc::new(Self::new(
                input,
                self.checkpoint_job_id,
                self.storage_level.clone(),
                self.writer.clone(),
                self.registrar.clone(),
                self.task_key.clone(),
            ))),
            _ => datafusion::common::plan_err!(
                "RuntimeLocalCheckpointWriteExec should have one child"
            ),
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::common::Result<SendableRecordBatchStream> {
        if partition != self.task_key.partition {
            return Err(exec_datafusion_err!(
                "local checkpoint task partition mismatch: expected {}, got {partition}",
                self.task_key.partition
            ));
        }
        let empty = RecordBatch::new_empty(self.schema());
        let writer = self.writer.clone();
        let registrar = self.registrar.clone();
        let checkpoint_job_id = self.checkpoint_job_id;
        let storage_level = self.storage_level.clone();
        let task_key = self.task_key.clone();
        let stream = self.input.execute(partition, context)?;
        let output = futures::stream::once(async move {
            local_checkpoint_write(
                writer,
                registrar,
                checkpoint_job_id,
                storage_level,
                task_key,
                stream,
            )
            .await?;
            Ok(empty)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            output,
        )))
    }
}

async fn local_checkpoint_write(
    writer: Arc<dyn TaskStreamWriter>,
    registrar: LocalCheckpointRegistrarContext,
    checkpoint_job_id: JobId,
    storage_level: spec::StorageLevel,
    task_key: TaskKey,
    mut stream: SendableRecordBatchStream,
) -> datafusion::common::Result<()> {
    let stream_key = TaskStreamKey {
        job_id: checkpoint_job_id,
        stage: 0,
        partition: task_key.partition,
        attempt: task_key.attempt,
        channel: 0,
    };
    let location = TaskWriteLocation::Local {
        storage: LocalStreamStorage::Checkpoint {
            storage_level: storage_level.clone(),
        },
        key: stream_key.clone(),
    };
    let mut sink = writer.open(&location, stream.schema()).await?;
    while let Some(batch) = stream.next().await {
        match batch {
            Ok(batch) => match sink.write(Ok(batch)).await {
                TaskStreamSinkState::Ok => {}
                TaskStreamSinkState::Error(error) => return Err(error),
                TaskStreamSinkState::Closed => {
                    return Err(exec_datafusion_err!(
                        "local checkpoint sink closed unexpectedly for {}",
                        TaskKeyDisplay(&task_key)
                    ))
                }
            },
            Err(error) => {
                let stream_error =
                    TaskStreamError::from(CommonErrorCause::new::<PyErrExtractor>(&error));
                match sink.write(Err(stream_error)).await {
                    TaskStreamSinkState::Ok | TaskStreamSinkState::Closed => {}
                    TaskStreamSinkState::Error(sink_error) => return Err(sink_error),
                }
                return Err(error);
            }
        }
    }
    sink.close().await?;
    register_local_checkpoint_partition(registrar, checkpoint_job_id, stream_key).await
}

async fn register_local_checkpoint_partition(
    registrar: LocalCheckpointRegistrarContext,
    checkpoint_job_id: JobId,
    key: TaskStreamKey,
) -> datafusion::common::Result<()> {
    match registrar {
        LocalCheckpointRegistrarContext::Driver { handle } => {
            handle
                .send(DriverEvent::RegisterLocalCheckpointPartition {
                    checkpoint_job_id,
                    key,
                    owner: LocalCheckpointStreamOwner::Driver,
                })
                .await
                .map_err(|e| exec_datafusion_err!("{e}"))?;
            Ok(())
        }
        LocalCheckpointRegistrarContext::Worker { driver, worker_id } => driver
            .report_local_checkpoint_partition(
                checkpoint_job_id,
                key.partition,
                key.attempt,
                key.channel,
                worker_id,
            )
            .await
            .map_err(|e| exec_datafusion_err!("{e}")),
    }
}
