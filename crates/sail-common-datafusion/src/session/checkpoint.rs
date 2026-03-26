use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use datafusion::datasource::TableProvider;
use datafusion_common::internal_datafusion_err;

use crate::extension::SessionExtension;

/// Session-scoped storage for server-side cached relations such as Spark Connect checkpoints.
pub struct CheckpointStore {
    relations: Mutex<HashMap<String, Arc<dyn TableProvider>>>,
}

impl Default for CheckpointStore {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckpointStore {
    pub fn new() -> Self {
        Self {
            relations: Mutex::new(HashMap::new()),
        }
    }

    pub fn insert(
        &self,
        relation_id: String,
        relation: Arc<dyn TableProvider>,
    ) -> datafusion_common::Result<Option<Arc<dyn TableProvider>>> {
        let mut relations = self
            .relations
            .lock()
            .map_err(|e| internal_datafusion_err!("{e}"))?;
        Ok(relations.insert(relation_id, relation))
    }

    pub fn get(
        &self,
        relation_id: &str,
    ) -> datafusion_common::Result<Option<Arc<dyn TableProvider>>> {
        let relations = self
            .relations
            .lock()
            .map_err(|e| internal_datafusion_err!("{e}"))?;
        Ok(relations.get(relation_id).cloned())
    }

    pub fn remove(
        &self,
        relation_id: &str,
    ) -> datafusion_common::Result<Option<Arc<dyn TableProvider>>> {
        let mut relations = self
            .relations
            .lock()
            .map_err(|e| internal_datafusion_err!("{e}"))?;
        Ok(relations.remove(relation_id))
    }
}

impl SessionExtension for CheckpointStore {
    fn name() -> &'static str {
        "CheckpointStore"
    }
}
