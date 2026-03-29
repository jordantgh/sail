use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use datafusion::arrow::compute::concat_batches;
use datafusion::catalog::Session;
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::datasource::TableProvider;
use datafusion::logical_expr::{Extension, LogicalPlan};
use datafusion::prelude::SessionContext;
use fastrace::collector::SpanContext;
use fastrace::future::FutureExt;
use fastrace::Span;
use futures::{stream, StreamExt, TryStreamExt};
use log::{debug, warn};
use sail_common::spec;
use sail_common_datafusion::datasource::{SinkMode, SourceInfo, TableFormatRegistry};
use sail_common_datafusion::extension::SessionExtensionAccessor;
use sail_common_datafusion::rename::logical_plan::rename_logical_plan;
use sail_common_datafusion::rename::schema::rename_schema;
use sail_common_datafusion::session::job::{JobRunnerMode, JobService};
use sail_common_datafusion::session::remote_relation::{
    CheckpointRelation, RemoteRelationBacking, RemoteRelationCleanupPolicy,
    RemoteRelationMaterializer, RemoteRelationStore,
};
use sail_logical_plan::file_write::{FileWriteNode, FileWriteOptions};
use sail_plan::{resolve_and_execute_plan, resolve_named_plan};
use tonic::codegen::tokio_stream::wrappers::ReceiverStream;
use tonic::codegen::tokio_stream::Stream;
use tonic::Status;

use crate::error::{ProtoFieldExt, SparkError, SparkResult};
use crate::executor::{
    read_stream, to_arrow_batch, Executor, ExecutorBatch, ExecutorMetadata, ExecutorOutput,
    ExecutorOutputStream,
};
use crate::service::{validate_local_checkpoint_storage_level, SparkLocalCheckpointMaterializer};
use crate::session::SparkSession;
use crate::spark::connect::execute_plan_response::{
    ResponseType, ResultComplete, SqlCommandResult,
};
use crate::spark::connect::{
    relation, CachedRemoteRelation, CheckpointCommand, CheckpointCommandResult,
    CommonInlineUserDefinedDataSource, CommonInlineUserDefinedFunction,
    CommonInlineUserDefinedTableFunction, CreateDataFrameViewCommand, ExecutePlanResponse,
    GetResourcesCommand, LocalRelation, MergeIntoTableCommand, Relation,
    RemoveCachedRemoteRelationCommand, SqlCommand, StreamingQueryCommand,
    StreamingQueryCommandResult, StreamingQueryListenerBusCommand, StreamingQueryManagerCommand,
    StreamingQueryManagerCommandResult, WriteOperation, WriteOperationV2,
    WriteStreamOperationStart, WriteStreamOperationStartResult,
};
use crate::streaming::timeout_millis;

pub struct ExecutePlanResponseStream {
    session_id: String,
    operation_id: String,
    inner: ExecutorOutputStream,
}

impl ExecutePlanResponseStream {
    pub fn new(session_id: String, operation_id: String, inner: ExecutorOutputStream) -> Self {
        Self {
            session_id,
            operation_id,
            inner,
        }
    }
}

impl Stream for ExecutePlanResponseStream {
    type Item = Result<ExecutePlanResponse, Status>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<ExecutePlanResponse, Status>>> {
        self.inner.as_mut().poll_next(cx).map(|poll| {
            poll.map(|item| {
                let mut response = ExecutePlanResponse::default();
                response.session_id.clone_from(&self.session_id);
                response.server_side_session_id.clone_from(&self.session_id);
                response.operation_id.clone_from(&self.operation_id.clone());
                response.response_id = item.id;
                match item.batch {
                    ExecutorBatch::ArrowBatch(batch) => {
                        response.response_type = Some(ResponseType::ArrowBatch(batch));
                    }
                    ExecutorBatch::SqlCommandResult(result) => {
                        response.response_type = Some(ResponseType::SqlCommandResult(*result));
                    }
                    ExecutorBatch::WriteStreamOperationStartResult(result) => {
                        response.response_type =
                            Some(ResponseType::WriteStreamOperationStartResult(*result));
                    }
                    ExecutorBatch::StreamingQueryCommandResult(result) => {
                        response.response_type =
                            Some(ResponseType::StreamingQueryCommandResult(*result));
                    }
                    ExecutorBatch::StreamingQueryManagerCommandResult(result) => {
                        response.response_type =
                            Some(ResponseType::StreamingQueryManagerCommandResult(*result));
                    }
                    ExecutorBatch::CheckpointCommandResult(result) => {
                        response.response_type =
                            Some(ResponseType::CheckpointCommandResult(*result));
                    }
                    ExecutorBatch::Schema(schema) => {
                        response.schema = Some(*schema);
                    }
                    ExecutorBatch::Complete => {
                        response.response_type =
                            Some(ResponseType::ResultComplete(ResultComplete::default()));
                    }
                }
                debug!("{response:?}");
                Ok(response)
            })
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

enum ExecutePlanMode {
    /// Execute the plan lazily as the client reads the response stream.
    Lazy,
    /// Execute the plan eagerly and return an empty response stream.
    /// This is useful for executing command plans.
    EagerSilent,
}

async fn handle_execute_plan(
    ctx: &SessionContext,
    plan: spec::Plan,
    metadata: ExecutorMetadata,
    mode: ExecutePlanMode,
) -> SparkResult<ExecutePlanResponseStream> {
    let span = Span::root("handle_execute_plan", SpanContext::random());
    let spark = ctx.extension::<SparkSession>()?;
    let service = ctx.extension::<JobService>()?;
    let operation_id = metadata.operation_id.clone();
    let (plan, _) = resolve_and_execute_plan(ctx, spark.plan_config()?, plan).await?;
    let session_state = ctx.state();
    let stream = {
        let span = Span::enter_with_parent("JobRunner::execute", &span);
        service
            .runner()
            .execute(&session_state, plan)
            .in_span(span)
            .await?
    };
    let rx = match mode {
        ExecutePlanMode::Lazy => {
            let _guard = span.set_local_parent();
            let executor = Executor::new(
                metadata,
                stream,
                spark.options().execution_heartbeat_interval,
            );
            let rx = executor.start()?;
            spark.add_executor(executor)?;
            rx
        }
        ExecutePlanMode::EagerSilent => {
            let _ = read_stream(stream).in_span(span).await?;
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            if metadata.reattachable {
                tx.send(ExecutorOutput::complete()).await?;
            }
            ReceiverStream::new(rx)
        }
    };
    Ok(ExecutePlanResponseStream::new(
        spark.session_id().to_string(),
        operation_id,
        Box::pin(rx),
    ))
}

pub(crate) async fn handle_execute_relation(
    ctx: &SessionContext,
    relation: Relation,
    metadata: ExecutorMetadata,
) -> SparkResult<ExecutePlanResponseStream> {
    let plan = relation.try_into()?;
    handle_execute_plan(ctx, plan, metadata, ExecutePlanMode::Lazy).await
}

pub(crate) async fn handle_execute_register_function(
    ctx: &SessionContext,
    udf: CommonInlineUserDefinedFunction,
    metadata: ExecutorMetadata,
) -> SparkResult<ExecutePlanResponseStream> {
    let plan = spec::Plan::Command(spec::CommandPlan::new(spec::CommandNode::RegisterFunction(
        udf.try_into()?,
    )));
    handle_execute_plan(ctx, plan, metadata, ExecutePlanMode::EagerSilent).await
}

pub(crate) async fn handle_execute_write_operation(
    ctx: &SessionContext,
    write: WriteOperation,
    metadata: ExecutorMetadata,
) -> SparkResult<ExecutePlanResponseStream> {
    let plan = spec::Plan::Command(spec::CommandPlan::new(spec::CommandNode::Write(
        write.try_into()?,
    )));
    handle_execute_plan(ctx, plan, metadata, ExecutePlanMode::EagerSilent).await
}

pub(crate) async fn handle_execute_create_dataframe_view(
    ctx: &SessionContext,
    view: CreateDataFrameViewCommand,
    metadata: ExecutorMetadata,
) -> SparkResult<ExecutePlanResponseStream> {
    let plan = spec::Plan::Command(spec::CommandPlan::new(view.try_into()?));
    handle_execute_plan(ctx, plan, metadata, ExecutePlanMode::EagerSilent).await
}

pub(crate) async fn handle_execute_write_operation_v2(
    ctx: &SessionContext,
    write: WriteOperationV2,
    metadata: ExecutorMetadata,
) -> SparkResult<ExecutePlanResponseStream> {
    let plan = spec::Plan::Command(spec::CommandPlan::new(spec::CommandNode::WriteTo(
        write.try_into()?,
    )));
    handle_execute_plan(ctx, plan, metadata, ExecutePlanMode::EagerSilent).await
}

pub(crate) async fn handle_execute_merge_into_table_command(
    ctx: &SessionContext,
    command: MergeIntoTableCommand,
    metadata: ExecutorMetadata,
) -> SparkResult<ExecutePlanResponseStream> {
    let plan = spec::Plan::Command(spec::CommandPlan::new(command.try_into()?));
    handle_execute_plan(ctx, plan, metadata, ExecutePlanMode::EagerSilent).await
}

/// Handles execution of a SQL command.
/// If a string is sent over we convert it to a relation then convert it to a plan, then execute it.
pub(crate) async fn handle_execute_sql_command(
    ctx: &SessionContext,
    sql: SqlCommand,
    metadata: ExecutorMetadata,
) -> SparkResult<ExecutePlanResponseStream> {
    let spark = ctx.extension::<SparkSession>()?;
    let service = ctx.extension::<JobService>()?;
    let relation = if let Some(input) = sql.input {
        input
    } else {
        Relation {
            common: None,
            #[expect(deprecated)]
            rel_type: Some(relation::RelType::Sql(crate::spark::connect::Sql {
                query: sql.sql,
                args: sql.args,
                pos_args: sql.pos_args,
                named_arguments: sql.named_arguments,
                pos_arguments: sql.pos_arguments,
            })),
        }
    };
    let plan: spec::Plan = relation.clone().try_into()?;
    let relation = match plan {
        spec::Plan::Query(_) => relation,
        command @ spec::Plan::Command(_) => {
            let (plan, _) = resolve_and_execute_plan(ctx, spark.plan_config()?, command).await?;
            let session_state = ctx.state();
            let stream = service.runner().execute(&session_state, plan).await?;
            let schema = stream.schema();
            let data = read_stream(stream).await?;
            let data = concat_batches(&schema, data.iter())?;
            Relation {
                common: None,
                rel_type: Some(relation::RelType::LocalRelation(LocalRelation {
                    data: Some(to_arrow_batch(&data)?.data),
                    schema: None,
                })),
            }
        }
    };
    let result = ExecutorBatch::SqlCommandResult(Box::new(SqlCommandResult {
        relation: Some(relation),
    }));
    let mut output = vec![ExecutorOutput::new(result)];
    if metadata.reattachable {
        output.push(ExecutorOutput::complete());
    }
    Ok(ExecutePlanResponseStream::new(
        spark.session_id().to_string(),
        metadata.operation_id,
        Box::pin(stream::iter(output)),
    ))
}

#[cfg(test)]
mod tests {
    use datafusion::logical_expr::{LogicalPlan, LogicalPlanBuilder};
    use sail_common_datafusion::datasource::SinkMode;
    use sail_common_datafusion::session::remote_relation::{
        RemoteRelationBacking, RemoteRelationCleanupPolicy,
    };
    use sail_logical_plan::file_write::FileWriteNode;

    use super::build_checkpoint_write_plan;

    #[test]
    fn test_build_checkpoint_write_plan_uses_parquet_file_write_node() {
        let plan = LogicalPlanBuilder::empty(false).build().unwrap();
        let schema = plan.schema().as_arrow().clone();
        let backing = RemoteRelationBacking::Files {
            location: "file:///tmp/checkpoints/relation-a".to_string(),
            format: "parquet".to_string(),
            cleanup_policy: RemoteRelationCleanupPolicy::RetainOnRemove,
        };

        let checkpoint = build_checkpoint_write_plan(plan, schema.as_ref(), &backing).unwrap();
        let LogicalPlan::Extension(extension) = checkpoint else {
            panic!("checkpoint write plan should be a logical extension node");
        };
        let node = extension
            .node
            .as_any()
            .downcast_ref::<FileWriteNode>()
            .expect("checkpoint write plan should use FileWriteNode");
        let options = node.options();

        assert_eq!(options.format, "parquet");
        assert_eq!(options.mode, SinkMode::Overwrite);
        assert!(options.partition_by.is_empty());
        assert!(options.sort_by.is_empty());
        assert!(options.bucket_by.is_none());
        assert!(options.table_properties.is_empty());
        assert_eq!(
            options.options,
            vec![vec![(
                "path".to_string(),
                "file:///tmp/checkpoints/relation-a".to_string(),
            )]]
        );
    }
}

pub(crate) async fn handle_execute_write_stream_operation_start(
    ctx: &SessionContext,
    start: WriteStreamOperationStart,
    metadata: ExecutorMetadata,
) -> SparkResult<ExecutePlanResponseStream> {
    let spark = ctx.extension::<SparkSession>()?;
    let service = ctx.extension::<JobService>()?;
    let operation_id = metadata.operation_id.clone();
    let reattachable = metadata.reattachable;
    let query_name = start.query_name.clone();
    let plan = spec::Plan::Command(spec::CommandPlan::new(start.try_into()?));
    let (plan, info) = resolve_and_execute_plan(ctx, spark.plan_config()?, plan).await?;
    let session_state = ctx.state();
    let stream = service.runner().execute(&session_state, plan).await?;
    let id = spark.start_streaming_query(query_name.clone(), info, stream)?;
    let result = WriteStreamOperationStartResult {
        query_id: Some(id.into()),
        name: query_name,
        // The event is for the client-side listener, which is not supported yet.
        query_started_event_json: None,
    };
    let mut output = vec![ExecutorOutput::new(
        ExecutorBatch::WriteStreamOperationStartResult(Box::new(result)),
    )];
    if reattachable {
        output.push(ExecutorOutput::complete());
    }
    Ok(ExecutePlanResponseStream::new(
        spark.session_id().to_string(),
        operation_id,
        Box::pin(stream::iter(output)),
    ))
}

pub(crate) async fn handle_execute_streaming_query_command(
    ctx: &SessionContext,
    stream: StreamingQueryCommand,
    metadata: ExecutorMetadata,
) -> SparkResult<ExecutePlanResponseStream> {
    use crate::spark::connect::streaming_query_command::{
        AwaitTerminationCommand, Command, ExplainCommand,
    };
    use crate::spark::connect::streaming_query_command_result::{
        AwaitTerminationResult, ExceptionResult, ExplainResult, RecentProgressResult, ResultType,
        StatusResult,
    };

    let spark = ctx.extension::<SparkSession>()?;
    let StreamingQueryCommand { query_id, command } = stream;
    let query_id = query_id.required("streaming query ID")?;
    let command = command.required("streaming query command")?;
    let result_type = match command {
        Command::Status(true) => {
            let status = spark.get_streaming_query_status(&query_id.clone().into())?;
            Some(ResultType::Status(StatusResult {
                status_message: status.message,
                is_data_available: true,
                is_trigger_active: true,
                is_active: status.is_active,
            }))
        }
        Command::LastProgress(true) | Command::RecentProgress(true) => {
            Some(ResultType::RecentProgress(RecentProgressResult {
                recent_progress_json: vec![],
            }))
        }
        Command::Stop(true) => {
            spark.stop_streaming_query(&query_id.clone().into())?;
            None
        }
        Command::ProcessAllAvailable(true) => None,
        Command::Explain(ExplainCommand { extended }) => {
            let mut result = spark.explain_streaming_query(&query_id.clone().into(), extended)?;
            while result.ends_with('\n') {
                result.pop();
            }
            Some(ResultType::Explain(ExplainResult { result }))
        }
        Command::Exception(true) => {
            let (message, class) = if let Some(throwable) =
                spark.get_streaming_query_exception(&query_id.clone().into())?
            {
                (
                    Some(throwable.message().to_string()),
                    Some(throwable.class_name().to_string()),
                )
            } else {
                (None, None)
            };
            Some(ResultType::Exception(ExceptionResult {
                exception_message: message,
                error_class: class,
                stack_trace: None,
            }))
        }
        Command::AwaitTermination(AwaitTerminationCommand { timeout_ms }) => {
            let timeout = timeout_ms.map(timeout_millis).transpose()?;
            let handle = spark.await_streaming_query(&query_id.clone().into())?;
            let terminated = if let Some(handle) = handle {
                handle.terminated(timeout).await?
            } else {
                true
            };
            Some(ResultType::AwaitTermination(AwaitTerminationResult {
                terminated,
            }))
        }
        Command::Status(false)
        | Command::LastProgress(false)
        | Command::RecentProgress(false)
        | Command::Stop(false)
        | Command::ProcessAllAvailable(false)
        | Command::Exception(false) => {
            return Err(SparkError::invalid(format!(
                "invalid streaming query command: {command:?}"
            )))
        }
    };
    let result = StreamingQueryCommandResult {
        query_id: Some(query_id),
        result_type,
    };
    let mut output = vec![ExecutorOutput::new(
        ExecutorBatch::StreamingQueryCommandResult(Box::new(result)),
    )];
    if metadata.reattachable {
        output.push(ExecutorOutput::complete());
    }
    Ok(ExecutePlanResponseStream::new(
        spark.session_id().to_string(),
        metadata.operation_id,
        Box::pin(stream::iter(output)),
    ))
}

pub(crate) async fn handle_execute_get_resources_command(
    _ctx: &SessionContext,
    _resource: GetResourcesCommand,
    _metadata: ExecutorMetadata,
) -> SparkResult<ExecutePlanResponseStream> {
    Err(SparkError::todo("get resources command"))
}

pub(crate) async fn handle_execute_streaming_query_manager_command(
    ctx: &SessionContext,
    command: StreamingQueryManagerCommand,
    metadata: ExecutorMetadata,
) -> SparkResult<ExecutePlanResponseStream> {
    use crate::spark::connect::streaming_query_manager_command::{
        AwaitAnyTerminationCommand, Command,
    };
    use crate::spark::connect::streaming_query_manager_command_result::{
        ActiveResult, AwaitAnyTerminationResult, ResultType, StreamingQueryInstance,
    };

    let spark = ctx.extension::<SparkSession>()?;
    let StreamingQueryManagerCommand { command } = command;
    let command = command.required("streaming query manager command")?;
    let result_type = match command {
        Command::Active(true) => {
            let active_queries = spark
                .list_active_streaming_queries()?
                .into_iter()
                .map(|(id, status)| StreamingQueryInstance {
                    id: Some(id.into()),
                    name: Some(status.name),
                })
                .collect();
            Some(ResultType::Active(ActiveResult { active_queries }))
        }
        Command::GetQuery(id) => {
            let (id, status) = spark.find_streaming_query_by_query_id(&id)?;
            Some(ResultType::Query(StreamingQueryInstance {
                id: Some(id.into()),
                name: Some(status.name),
            }))
        }
        Command::AwaitAnyTermination(AwaitAnyTerminationCommand { timeout_ms }) => {
            let timeout = timeout_ms.map(timeout_millis).transpose()?;
            let handles = spark.await_streaming_queries()?;
            let terminated = handles.any_terminated(timeout).await?;
            Some(ResultType::AwaitAnyTermination(AwaitAnyTerminationResult {
                terminated,
            }))
        }
        Command::ResetTerminated(true) => {
            spark.reset_terminated_streaming_queries()?;
            Some(ResultType::ResetTerminated(true))
        }
        Command::AddListener(_) => {
            return Err(SparkError::NotImplemented("add listener".to_string()))
        }
        Command::RemoveListener(_) => {
            return Err(SparkError::NotImplemented("remove listener".to_string()))
        }
        Command::ListListeners(_) => {
            return Err(SparkError::NotImplemented("list listeners".to_string()))
        }
        Command::Active(false) | Command::ResetTerminated(false) => {
            return Err(SparkError::invalid(format!(
                "invalid streaming query manager command: {command:?}"
            )))
        }
    };
    let result = StreamingQueryManagerCommandResult { result_type };
    let mut output = vec![ExecutorOutput::new(
        ExecutorBatch::StreamingQueryManagerCommandResult(Box::new(result)),
    )];
    if metadata.reattachable {
        output.push(ExecutorOutput::complete());
    }
    Ok(ExecutePlanResponseStream::new(
        spark.session_id().to_string(),
        metadata.operation_id,
        Box::pin(stream::iter(output)),
    ))
}

pub(crate) async fn handle_execute_register_table_function(
    ctx: &SessionContext,
    udtf: CommonInlineUserDefinedTableFunction,
    metadata: ExecutorMetadata,
) -> SparkResult<ExecutePlanResponseStream> {
    let plan = spec::Plan::Command(spec::CommandPlan::new(
        spec::CommandNode::RegisterTableFunction(udtf.try_into()?),
    ));
    handle_execute_plan(ctx, plan, metadata, ExecutePlanMode::EagerSilent).await
}

pub(crate) async fn handle_execute_streaming_query_listener_bus_command(
    _ctx: &SessionContext,
    _command: StreamingQueryListenerBusCommand,
    _metadata: ExecutorMetadata,
) -> SparkResult<ExecutePlanResponseStream> {
    Err(SparkError::NotImplemented(
        "streaming query listener bus".to_string(),
    ))
}

pub(crate) async fn handle_execute_checkpoint_command(
    ctx: &SessionContext,
    checkpoint: CheckpointCommand,
    metadata: ExecutorMetadata,
) -> SparkResult<ExecutePlanResponseStream> {
    let spark = ctx.extension::<SparkSession>()?;
    let service = ctx.extension::<JobService>()?;
    let store = ctx.extension::<RemoteRelationStore>()?;
    let CheckpointCommand {
        relation,
        local,
        eager,
        storage_level,
    } = checkpoint;

    let relation = relation.required("checkpoint relation")?;
    let plan: spec::Plan = relation.try_into()?;
    let query = match plan {
        spec::Plan::Query(plan) => plan,
        spec::Plan::Command(_) => {
            return Err(SparkError::invalid(
                "checkpoint relation must resolve to a query plan",
            ))
        }
    };
    let relation_id = uuid::Uuid::new_v4().to_string();
    let storage_id = uuid::Uuid::new_v4().to_string();
    let runner_mode = service.runner().mode();
    let plan_config = spark.plan_config()?;
    let sail_plan::resolver::plan::NamedPlan { plan, fields } =
        resolve_named_plan(ctx, plan_config, spec::Plan::Query(query)).await?;
    let fields = fields
        .ok_or_else(|| SparkError::invalid("checkpoint relation must resolve to a query plan"))?;
    let schema = rename_schema(plan.schema().as_arrow(), &fields)?;
    let backing = if local {
        let storage_level = storage_level.map(TryInto::try_into).transpose()?;
        let storage_level = validate_local_checkpoint_storage_level(storage_level)
            .map_err(SparkError::unsupported)?;
        RemoteRelationBacking::LocalCache {
            storage_level: storage_level.clone(),
            location: (runner_mode == JobRunnerMode::Local && storage_level.use_disk)
                .then(|| spark.checkpoint_location(true, &storage_id))
                .transpose()?,
            format: (runner_mode == JobRunnerMode::Local && storage_level.use_disk)
                .then(|| "arrow".to_string()),
            stream_job_id: (runner_mode == JobRunnerMode::Cluster)
                .then_some(generate_local_checkpoint_job_id()),
            cleanup_policy: RemoteRelationCleanupPolicy::DeleteOnRemove,
        }
    } else {
        let location = spark.checkpoint_location(false, &storage_id)?;
        RemoteRelationBacking::Files {
            location,
            format: "parquet".to_string(),
            cleanup_policy: RemoteRelationCleanupPolicy::RetainOnRemove,
        }
    };
    let relation = Arc::new(CheckpointRelation::new(
        relation_id.clone(),
        schema,
        plan,
        backing,
        if local {
            Arc::new(SparkLocalCheckpointMaterializer) as Arc<dyn RemoteRelationMaterializer>
        } else {
            Arc::new(SparkCheckpointMaterializer) as Arc<dyn RemoteRelationMaterializer>
        },
    ));
    if eager {
        let session_state = ctx.state();
        let _ = relation.ensure_materialized(&session_state).await?;
    }
    let _ = store.insert(relation_id.clone(), relation)?;

    let result = CheckpointCommandResult {
        relation: Some(CachedRemoteRelation { relation_id }),
    };
    let mut output = vec![ExecutorOutput::new(ExecutorBatch::CheckpointCommandResult(
        Box::new(result),
    ))];
    if metadata.reattachable {
        output.push(ExecutorOutput::complete());
    }
    Ok(ExecutePlanResponseStream::new(
        spark.session_id().to_string(),
        metadata.operation_id,
        Box::pin(stream::iter(output)),
    ))
}

fn generate_local_checkpoint_job_id() -> u64 {
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&uuid::Uuid::new_v4().into_bytes()[..8]);
    u64::from_le_bytes(bytes) | (1_u64 << 63)
}

pub(crate) async fn handle_execute_remove_cached_remote_relation_command(
    ctx: &SessionContext,
    command: RemoveCachedRemoteRelationCommand,
    metadata: ExecutorMetadata,
) -> SparkResult<ExecutePlanResponseStream> {
    let spark = ctx.extension::<SparkSession>()?;
    let store = ctx.extension::<RemoteRelationStore>()?;
    let relation = command.relation.required("cached remote relation")?;
    if let Some(entry) = store.remove(&relation.relation_id)? {
        let session_state = ctx.state();
        entry.remove(&session_state).await?;
    }
    let mut output = vec![];
    if metadata.reattachable {
        output.push(ExecutorOutput::complete());
    }
    Ok(ExecutePlanResponseStream::new(
        spark.session_id().to_string(),
        metadata.operation_id,
        Box::pin(stream::iter(output)),
    ))
}

#[derive(Debug)]
struct SparkCheckpointMaterializer;

#[async_trait]
impl RemoteRelationMaterializer for SparkCheckpointMaterializer {
    async fn materialize(
        &self,
        state: &dyn Session,
        plan: &LogicalPlan,
        schema: Arc<datafusion::arrow::datatypes::Schema>,
        backing: &RemoteRelationBacking,
    ) -> DataFusionResult<Arc<dyn TableProvider>> {
        self.cleanup(state, backing).await?;
        let write = build_checkpoint_write_plan(plan.clone(), schema.as_ref(), backing)?;
        let physical = state.create_physical_plan(&write).await?;
        let service = state.extension::<JobService>()?;
        let stream = service.runner().execute(state, physical).await?;
        let _ = read_stream(stream)
            .await
            .map_err(|error| DataFusionError::External(Box::new(error)))?;
        build_checkpoint_provider(state, backing, schema.as_ref().clone()).await
    }

    async fn cleanup(
        &self,
        state: &dyn Session,
        backing: &RemoteRelationBacking,
    ) -> DataFusionResult<()> {
        let RemoteRelationBacking::Files { location, .. } = backing else {
            return Err(DataFusionError::Execution(
                "file-backed checkpoint cleanup requires a file backing".to_string(),
            ));
        };
        let parsed = ListingTableUrl::parse(location)?;
        let store = state
            .runtime_env()
            .object_store_registry
            .get_store(parsed.as_ref())?;
        let prefix = parsed.prefix().clone();
        let to_delete = store
            .list(Some(&prefix))
            .map_ok(move |meta| meta.location)
            .boxed();
        let _ = store
            .delete_stream(to_delete)
            .try_collect::<Vec<_>>()
            .await?;
        if parsed.scheme() == "file" {
            let url: &url::Url = parsed.as_ref();
            if let Ok(path) = url.to_file_path() {
                match std::fs::remove_dir_all(path) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error.into()),
                }
            }
        }
        Ok(())
    }
}

async fn build_checkpoint_provider(
    ctx: &dyn Session,
    backing: &RemoteRelationBacking,
    schema: datafusion::arrow::datatypes::Schema,
) -> DataFusionResult<Arc<dyn TableProvider>> {
    let registry = ctx.extension::<TableFormatRegistry>()?;
    let RemoteRelationBacking::Files {
        location, format, ..
    } = backing
    else {
        return Err(DataFusionError::Execution(
            "file-backed checkpoint provider requires a file backing".to_string(),
        ));
    };
    Ok(registry
        .get(format.as_str())?
        .create_provider(
            ctx,
            SourceInfo {
                paths: vec![location.clone()],
                schema: Some(schema),
                constraints: Default::default(),
                partition_by: vec![],
                bucket_by: None,
                sort_order: vec![],
                options: vec![],
            },
        )
        .await?)
}

fn build_checkpoint_write_plan(
    plan: LogicalPlan,
    schema: &datafusion::arrow::datatypes::Schema,
    backing: &RemoteRelationBacking,
) -> DataFusionResult<LogicalPlan> {
    let RemoteRelationBacking::Files {
        location, format, ..
    } = backing
    else {
        return Err(DataFusionError::Execution(
            "checkpoint write plans require a file backing".to_string(),
        ));
    };
    let names = schema
        .fields()
        .iter()
        .map(|field| field.name().to_string())
        .collect::<Vec<_>>();
    let plan = rename_logical_plan(plan, &names)?;
    Ok(LogicalPlan::Extension(Extension {
        node: Arc::new(FileWriteNode::new(
            Arc::new(plan),
            FileWriteOptions {
                format: format.clone(),
                mode: SinkMode::Overwrite,
                partition_by: vec![],
                sort_by: vec![],
                bucket_by: None,
                table_properties: vec![],
                options: vec![vec![("path".to_string(), location.clone())]],
            },
        )),
    }))
}

pub(crate) async fn handle_interrupt_all(ctx: &SessionContext) -> SparkResult<Vec<String>> {
    let spark = ctx.extension::<SparkSession>()?;
    let mut results = vec![];
    for executor in spark.remove_all_executors()? {
        executor.pause_if_running().await?;
        results.push(executor.metadata.operation_id.clone());
    }
    Ok(results)
}

pub(crate) async fn handle_interrupt_tag(
    ctx: &SessionContext,
    tag: String,
) -> SparkResult<Vec<String>> {
    let spark = ctx.extension::<SparkSession>()?;
    let mut results = vec![];
    for executor in spark.remove_executors_by_tag(tag.as_str())? {
        executor.pause_if_running().await?;
        results.push(executor.metadata.operation_id.clone());
    }
    Ok(results)
}

pub(crate) async fn handle_interrupt_operation_id(
    ctx: &SessionContext,
    operation_id: String,
) -> SparkResult<Vec<String>> {
    let spark = ctx.extension::<SparkSession>()?;
    match spark.remove_executor(operation_id.as_str())? {
        Some(executor) => {
            executor.pause_if_running().await?;
            Ok(vec![executor.metadata.operation_id.clone()])
        }
        None => Ok(vec![]),
    }
}

pub(crate) async fn handle_reattach_execute(
    ctx: &SessionContext,
    operation_id: String,
    response_id: Option<String>,
) -> SparkResult<ExecutePlanResponseStream> {
    let spark = ctx.extension::<SparkSession>()?;
    let executor = spark
        .get_executor(operation_id.as_str())?
        .ok_or_else(|| SparkError::invalid(format!("operation not found: {operation_id}")))?;
    if !executor.metadata.reattachable {
        return Err(SparkError::invalid(format!(
            "operation not reattachable: {operation_id}"
        )));
    }
    executor.pause_if_running().await?;
    executor.release(response_id)?;
    let rx = executor.start()?;
    Ok(ExecutePlanResponseStream::new(
        spark.session_id().to_string(),
        operation_id,
        Box::pin(rx),
    ))
}

pub(crate) async fn handle_release_execute(
    ctx: &SessionContext,
    operation_id: String,
    response_id: Option<String>,
) -> SparkResult<()> {
    let spark = ctx.extension::<SparkSession>()?;
    // Some operations may not have an executor (e.g. DDL statements),
    // so it is a no-op if the executor is not found.
    if let Some(executor) = spark.get_executor(operation_id.as_str())? {
        executor.release(response_id)?;
    }
    Ok(())
}

pub(crate) async fn handle_execute_register_datasource(
    ctx: &SessionContext,
    datasource: CommonInlineUserDefinedDataSource,
    metadata: ExecutorMetadata,
) -> SparkResult<ExecutePlanResponseStream> {
    use crate::spark::connect::common_inline_user_defined_data_source::DataSource;

    log::info!(
        "RegisterDataSource handler called for datasource: {}",
        datasource.name
    );

    let spark = ctx.extension::<SparkSession>()?;
    let name = datasource.name.clone();

    // Extract the pickled Python datasource class
    let command = match datasource.data_source {
        Some(DataSource::PythonDataSource(pds)) => pds.command,
        None => {
            return Err(SparkError::invalid(
                "RegisterDataSource requires a python_data_source",
            ))
        }
    };

    // Register in the session-scoped TableFormatRegistry with embedded pickled bytes
    {
        use std::sync::Arc;

        use sail_common_datafusion::datasource::TableFormatRegistry;
        use sail_data_source::formats::python::PythonTableFormat;

        // Register format in session's TableFormatRegistry with embedded pickled class
        // This provides session isolation - the format is only visible to this session
        if let Ok(registry) = ctx.extension::<TableFormatRegistry>() {
            let format = Arc::new(PythonTableFormat::with_pickled_class(name.clone(), command));
            // Ignore error if already registered (allows re-registration to update)
            if let Err(e) = registry.register(format) {
                warn!("Failed to register python datasource {}: {}", name, e);
            }
            log::info!("Registered session-scoped datasource: {}", name);
        } else {
            return Err(SparkError::internal(
                "TableFormatRegistry not found in session context",
            ));
        }
    }

    // Return empty success response
    let mut output = vec![];
    if metadata.reattachable {
        output.push(ExecutorOutput::complete());
    }
    Ok(ExecutePlanResponseStream::new(
        spark.session_id().to_string(),
        metadata.operation_id,
        Box::pin(stream::iter(output)),
    ))
}
