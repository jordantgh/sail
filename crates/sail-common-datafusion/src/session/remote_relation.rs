use std::any::Any;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::{Session, TableProvider};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion_common::{exec_datafusion_err, internal_datafusion_err, Result};
use datafusion_expr::LogicalPlan;

use crate::extension::SessionExtension;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RemoteRelationCleanupPolicy {
    RetainOnRemove,
    DeleteOnRemove,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RemoteRelationBacking {
    Files {
        location: String,
        format: String,
        cleanup_policy: RemoteRelationCleanupPolicy,
    },
}

impl RemoteRelationBacking {
    pub fn cleanup_policy(&self) -> RemoteRelationCleanupPolicy {
        match self {
            Self::Files { cleanup_policy, .. } => *cleanup_policy,
        }
    }
}

#[async_trait]
pub trait RemoteRelationMaterializer: Debug + Send + Sync + 'static {
    async fn materialize(
        &self,
        state: &dyn Session,
        plan: &LogicalPlan,
        schema: SchemaRef,
        backing: &RemoteRelationBacking,
    ) -> Result<Arc<dyn TableProvider>>;

    async fn cleanup(&self, state: &dyn Session, backing: &RemoteRelationBacking) -> Result<()>;
}

#[async_trait]
pub trait RemoteRelationHandle: Debug + Send + Sync + 'static {
    fn provider(self: Arc<Self>) -> Arc<dyn TableProvider>;

    async fn remove(&self, state: &dyn Session) -> Result<()>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CheckpointRelationState {
    Pending,
    Materializing,
    Materialized,
    Failed { message: String, retryable: bool },
    Removed,
}

/// Opaque session-scoped checkpoint relation backed by a one-time materialization flow.
pub struct CheckpointRelation {
    relation_id: String,
    schema: SchemaRef,
    plan: LogicalPlan,
    backing: RemoteRelationBacking,
    materializer: Arc<dyn RemoteRelationMaterializer>,
    state: Mutex<CheckpointRelationState>,
    provider: Mutex<Option<Arc<dyn TableProvider>>>,
    notify: tokio::sync::Notify,
}

impl CheckpointRelation {
    pub fn new(
        relation_id: String,
        schema: SchemaRef,
        plan: LogicalPlan,
        backing: RemoteRelationBacking,
        materializer: Arc<dyn RemoteRelationMaterializer>,
    ) -> Self {
        Self {
            relation_id,
            schema,
            plan,
            backing,
            materializer,
            state: Mutex::new(CheckpointRelationState::Pending),
            provider: Mutex::new(None),
            notify: tokio::sync::Notify::new(),
        }
    }

    pub fn relation_id(&self) -> &str {
        &self.relation_id
    }

    pub fn backing(&self) -> &RemoteRelationBacking {
        &self.backing
    }

    pub async fn ensure_materialized(&self, state: &dyn Session) -> Result<Arc<dyn TableProvider>> {
        enum NextStep {
            Wait,
            Materialize,
        }

        loop {
            let next = {
                let mut status = self
                    .state
                    .lock()
                    .map_err(|e| internal_datafusion_err!("{e}"))?;
                match &*status {
                    CheckpointRelationState::Pending => {
                        *status = CheckpointRelationState::Materializing;
                        NextStep::Materialize
                    }
                    CheckpointRelationState::Materializing => NextStep::Wait,
                    CheckpointRelationState::Materialized => {
                        let provider = self
                            .provider
                            .lock()
                            .map_err(|e| internal_datafusion_err!("{e}"))?
                            .clone()
                            .ok_or_else(|| {
                                internal_datafusion_err!(
                                    "materialized checkpoint is missing its table provider"
                                )
                            })?;
                        return Ok(provider);
                    }
                    CheckpointRelationState::Failed { message, retryable } => {
                        if *retryable {
                            *status = CheckpointRelationState::Materializing;
                            NextStep::Materialize
                        } else {
                            return Err(exec_datafusion_err!("{message}"));
                        }
                    }
                    CheckpointRelationState::Removed => {
                        return Err(exec_datafusion_err!(
                            "cached relation has been removed: {}",
                            self.relation_id
                        ));
                    }
                }
            };

            match next {
                NextStep::Wait => self.notify.notified().await,
                NextStep::Materialize => {
                    let materialized = self
                        .materializer
                        .materialize(state, &self.plan, Arc::clone(&self.schema), &self.backing)
                        .await;
                    match materialized {
                        Ok(provider) => {
                            let cleanup = {
                                let mut status = self
                                    .state
                                    .lock()
                                    .map_err(|e| internal_datafusion_err!("{e}"))?;
                                match &*status {
                                    CheckpointRelationState::Removed => true,
                                    _ => {
                                        *status = CheckpointRelationState::Materialized;
                                        let mut current = self
                                            .provider
                                            .lock()
                                            .map_err(|e| internal_datafusion_err!("{e}"))?;
                                        *current = Some(provider.clone());
                                        false
                                    }
                                }
                            };
                            self.notify.notify_waiters();
                            if cleanup {
                                self.materializer.cleanup(state, &self.backing).await?;
                                return Err(exec_datafusion_err!(
                                    "cached relation has been removed: {}",
                                    self.relation_id
                                ));
                            }
                            return Ok(provider);
                        }
                        Err(error) => {
                            self.materializer.cleanup(state, &self.backing).await?;
                            {
                                let mut provider = self
                                    .provider
                                    .lock()
                                    .map_err(|e| internal_datafusion_err!("{e}"))?;
                                *provider = None;
                            }
                            {
                                let mut status = self
                                    .state
                                    .lock()
                                    .map_err(|e| internal_datafusion_err!("{e}"))?;
                                if !matches!(*status, CheckpointRelationState::Removed) {
                                    *status = CheckpointRelationState::Failed {
                                        message: error.to_string(),
                                        retryable: true,
                                    };
                                }
                            }
                            self.notify.notify_waiters();
                            return Err(error);
                        }
                    }
                }
            }
        }
    }

    fn materialized_provider(&self) -> Result<Option<Arc<dyn TableProvider>>> {
        let state = self
            .state
            .lock()
            .map_err(|e| internal_datafusion_err!("{e}"))?;
        if !matches!(*state, CheckpointRelationState::Materialized) {
            return Ok(None);
        }
        let provider = self
            .provider
            .lock()
            .map_err(|e| internal_datafusion_err!("{e}"))?;
        Ok(provider.clone())
    }
}

impl Debug for CheckpointRelation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointRelation")
            .field("relation_id", &self.relation_id)
            .field("schema", &self.schema)
            .field("backing", &self.backing)
            .finish()
    }
}

#[async_trait]
impl RemoteRelationHandle for CheckpointRelation {
    fn provider(self: Arc<Self>) -> Arc<dyn TableProvider> {
        Arc::new(CheckpointRelationProvider { relation: self })
    }

    async fn remove(&self, state: &dyn Session) -> Result<()> {
        let cleanup_on_remove = {
            let mut status = self
                .state
                .lock()
                .map_err(|e| internal_datafusion_err!("{e}"))?;
            match &*status {
                CheckpointRelationState::Removed => false,
                CheckpointRelationState::Pending
                | CheckpointRelationState::Materializing
                | CheckpointRelationState::Failed { .. } => {
                    *status = CheckpointRelationState::Removed;
                    false
                }
                CheckpointRelationState::Materialized => {
                    *status = CheckpointRelationState::Removed;
                    matches!(
                        self.backing.cleanup_policy(),
                        RemoteRelationCleanupPolicy::DeleteOnRemove
                    )
                }
            }
        };
        self.notify.notify_waiters();
        if cleanup_on_remove {
            self.materializer.cleanup(state, &self.backing).await?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct CheckpointRelationProvider {
    relation: Arc<CheckpointRelation>,
}

#[async_trait]
impl TableProvider for CheckpointRelationProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.relation.schema)
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
    ) -> Result<Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
        let provider = self.relation.ensure_materialized(state).await?;
        provider.scan(state, projection, filters, limit).await
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        if let Some(provider) = self.relation.materialized_provider()? {
            provider.supports_filters_pushdown(filters)
        } else {
            Ok(vec![
                TableProviderFilterPushDown::Unsupported;
                filters.len()
            ])
        }
    }
}

/// Session-scoped storage for server-side relations referenced by Spark Connect handles.
pub struct RemoteRelationStore {
    relations: Mutex<HashMap<String, Arc<dyn RemoteRelationHandle>>>,
}

impl Default for RemoteRelationStore {
    fn default() -> Self {
        Self::new()
    }
}

impl RemoteRelationStore {
    pub fn new() -> Self {
        Self {
            relations: Mutex::new(HashMap::new()),
        }
    }

    pub fn insert(
        &self,
        relation_id: String,
        relation: Arc<dyn RemoteRelationHandle>,
    ) -> Result<Option<Arc<dyn RemoteRelationHandle>>> {
        let mut relations = self
            .relations
            .lock()
            .map_err(|e| internal_datafusion_err!("{e}"))?;
        Ok(relations.insert(relation_id, relation))
    }

    pub fn get(&self, relation_id: &str) -> Result<Option<Arc<dyn RemoteRelationHandle>>> {
        let relations = self
            .relations
            .lock()
            .map_err(|e| internal_datafusion_err!("{e}"))?;
        Ok(relations.get(relation_id).cloned())
    }

    pub fn remove(&self, relation_id: &str) -> Result<Option<Arc<dyn RemoteRelationHandle>>> {
        let mut relations = self
            .relations
            .lock()
            .map_err(|e| internal_datafusion_err!("{e}"))?;
        Ok(relations.remove(relation_id))
    }
}

impl SessionExtension for RemoteRelationStore {
    fn name() -> &'static str {
        "RemoteRelationStore"
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use async_trait::async_trait;
    use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use datafusion::catalog::{Session, TableProvider};
    use datafusion::datasource::empty::EmptyTable;
    use datafusion::logical_expr::{col, lit, Expr, TableProviderFilterPushDown, TableType};
    use datafusion::physical_plan::empty::EmptyExec;
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::prelude::SessionContext;
    use datafusion_common::{internal_datafusion_err, DFSchema, DataFusionError, Result};
    use datafusion_expr::{EmptyRelation, LogicalPlan};

    use super::{
        CheckpointRelation, RemoteRelationBacking, RemoteRelationCleanupPolicy,
        RemoteRelationHandle, RemoteRelationMaterializer,
    };

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]))
    }

    fn test_plan() -> LogicalPlan {
        LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(DFSchema::empty()),
        })
    }

    fn test_backing(policy: RemoteRelationCleanupPolicy) -> RemoteRelationBacking {
        RemoteRelationBacking::Files {
            location: "file:///tmp/checkpoints/relation-a".to_string(),
            format: "parquet".to_string(),
            cleanup_policy: policy,
        }
    }

    #[derive(Debug)]
    struct CountingMaterializer {
        materialize_calls: AtomicUsize,
        cleanup_calls: AtomicUsize,
    }

    #[async_trait]
    impl RemoteRelationMaterializer for CountingMaterializer {
        async fn materialize(
            &self,
            _state: &dyn Session,
            _plan: &LogicalPlan,
            schema: SchemaRef,
            _backing: &RemoteRelationBacking,
        ) -> Result<Arc<dyn TableProvider>> {
            self.materialize_calls.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(25)).await;
            Ok(Arc::new(EmptyTable::new(schema)))
        }

        async fn cleanup(
            &self,
            _state: &dyn Session,
            _backing: &RemoteRelationBacking,
        ) -> Result<()> {
            self.cleanup_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[derive(Debug)]
    struct BlockingMaterializer {
        materialize_calls: AtomicUsize,
        cleanup_calls: AtomicUsize,
        started: tokio::sync::Notify,
        release: tokio::sync::Notify,
    }

    #[async_trait]
    impl RemoteRelationMaterializer for BlockingMaterializer {
        async fn materialize(
            &self,
            _state: &dyn Session,
            _plan: &LogicalPlan,
            schema: SchemaRef,
            _backing: &RemoteRelationBacking,
        ) -> Result<Arc<dyn TableProvider>> {
            self.materialize_calls.fetch_add(1, Ordering::SeqCst);
            self.started.notify_waiters();
            self.release.notified().await;
            Ok(Arc::new(EmptyTable::new(schema)))
        }

        async fn cleanup(
            &self,
            _state: &dyn Session,
            _backing: &RemoteRelationBacking,
        ) -> Result<()> {
            self.cleanup_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[derive(Debug)]
    struct PushdownMaterializer;

    #[derive(Debug)]
    struct PushdownTableProvider {
        schema: SchemaRef,
    }

    #[async_trait]
    impl TableProvider for PushdownTableProvider {
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
            _projection: Option<&Vec<usize>>,
            _filters: &[Expr],
            _limit: Option<usize>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(EmptyExec::new(Arc::clone(&self.schema))))
        }

        fn supports_filters_pushdown(
            &self,
            filters: &[&Expr],
        ) -> Result<Vec<TableProviderFilterPushDown>> {
            Ok(vec![TableProviderFilterPushDown::Exact; filters.len()])
        }
    }

    #[async_trait]
    impl RemoteRelationMaterializer for PushdownMaterializer {
        async fn materialize(
            &self,
            _state: &dyn Session,
            _plan: &LogicalPlan,
            schema: SchemaRef,
            _backing: &RemoteRelationBacking,
        ) -> Result<Arc<dyn TableProvider>> {
            Ok(Arc::new(PushdownTableProvider { schema }))
        }

        async fn cleanup(
            &self,
            _state: &dyn Session,
            _backing: &RemoteRelationBacking,
        ) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_checkpoint_relation_materializes_once_for_concurrent_access() -> Result<()> {
        let ctx = SessionContext::new();
        let state = ctx.state();
        let materializer = Arc::new(CountingMaterializer {
            materialize_calls: AtomicUsize::new(0),
            cleanup_calls: AtomicUsize::new(0),
        });
        let relation = Arc::new(CheckpointRelation::new(
            "relation-1".to_string(),
            test_schema(),
            test_plan(),
            test_backing(RemoteRelationCleanupPolicy::RetainOnRemove),
            materializer.clone(),
        ));

        let first = {
            let relation = relation.clone();
            let state = state.clone();
            tokio::spawn(async move { relation.ensure_materialized(&state).await })
        };
        let second = {
            let relation = relation.clone();
            let state = state.clone();
            tokio::spawn(async move { relation.ensure_materialized(&state).await })
        };

        let first = first.await.map_err(|e| internal_datafusion_err!("{e}"))??;
        let second = second
            .await
            .map_err(|e| internal_datafusion_err!("{e}"))??;

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(materializer.materialize_calls.load(Ordering::SeqCst), 1);
        assert_eq!(materializer.cleanup_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_checkpoint_relation_cleanup_on_remove_during_materialization() -> Result<()> {
        let ctx = SessionContext::new();
        let state = ctx.state();
        let materializer = Arc::new(BlockingMaterializer {
            materialize_calls: AtomicUsize::new(0),
            cleanup_calls: AtomicUsize::new(0),
            started: tokio::sync::Notify::new(),
            release: tokio::sync::Notify::new(),
        });
        let relation = Arc::new(CheckpointRelation::new(
            "relation-2".to_string(),
            test_schema(),
            test_plan(),
            test_backing(RemoteRelationCleanupPolicy::DeleteOnRemove),
            materializer.clone(),
        ));

        let materializing = {
            let relation = relation.clone();
            let state = state.clone();
            tokio::spawn(async move { relation.ensure_materialized(&state).await })
        };

        materializer.started.notified().await;
        relation.remove(&state).await?;
        materializer.release.notify_waiters();

        let error: DataFusionError = match materializing
            .await
            .map_err(|e| internal_datafusion_err!("{e}"))?
        {
            Ok(_) => {
                return Err(internal_datafusion_err!(
                    "removed relation should not publish a provider"
                ))
            }
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("cached relation has been removed"));
        assert_eq!(materializer.materialize_calls.load(Ordering::SeqCst), 1);
        assert_eq!(materializer.cleanup_calls.load(Ordering::SeqCst), 1);

        let removed = match relation.ensure_materialized(&state).await {
            Ok(_) => {
                return Err(internal_datafusion_err!(
                    "subsequent access should continue to fail for a removed relation"
                ))
            }
            Err(error) => error,
        };
        assert!(removed
            .to_string()
            .contains("cached relation has been removed"));
        Ok(())
    }

    #[tokio::test]
    async fn test_checkpoint_relation_blocks_filter_pushdown_until_materialized() -> Result<()> {
        let ctx = SessionContext::new();
        let state = ctx.state();
        let relation = Arc::new(CheckpointRelation::new(
            "relation-3".to_string(),
            test_schema(),
            test_plan(),
            test_backing(RemoteRelationCleanupPolicy::RetainOnRemove),
            Arc::new(PushdownMaterializer),
        ));

        let provider = relation.clone().provider();
        let filter = col("value").eq(lit(1_i64));
        let before = provider.supports_filters_pushdown(&[&filter])?;
        assert_eq!(before, vec![TableProviderFilterPushDown::Unsupported]);

        let _ = relation.ensure_materialized(&state).await?;

        let after = relation.provider().supports_filters_pushdown(&[&filter])?;
        assert_eq!(after, vec![TableProviderFilterPushDown::Exact]);
        Ok(())
    }
}
