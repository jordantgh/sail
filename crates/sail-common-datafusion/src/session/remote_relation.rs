use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use datafusion::datasource::TableProvider;
use datafusion_common::internal_datafusion_err;
use datafusion_expr::LogicalPlan;

use crate::extension::SessionExtension;

#[derive(Clone)]
pub enum RemoteRelationEntry {
    Materialized(Arc<dyn TableProvider>),
    Deferred {
        plan: LogicalPlan,
        fields: Vec<String>,
    },
}

/// Session-scoped storage for server-side relations referenced by Spark Connect handles.
pub struct RemoteRelationStore {
    relations: Mutex<HashMap<String, RemoteRelationEntry>>,
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
        relation: RemoteRelationEntry,
    ) -> datafusion_common::Result<Option<RemoteRelationEntry>> {
        let mut relations = self
            .relations
            .lock()
            .map_err(|e| internal_datafusion_err!("{e}"))?;
        Ok(relations.insert(relation_id, relation))
    }

    pub fn get(&self, relation_id: &str) -> datafusion_common::Result<Option<RemoteRelationEntry>> {
        let relations = self
            .relations
            .lock()
            .map_err(|e| internal_datafusion_err!("{e}"))?;
        Ok(relations.get(relation_id).cloned())
    }

    pub fn remove(
        &self,
        relation_id: &str,
    ) -> datafusion_common::Result<Option<RemoteRelationEntry>> {
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
