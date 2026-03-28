use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;

use datafusion::arrow::datatypes::Schema;
use datafusion::common::{plan_err, Result};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use sail_common::spec;

use crate::id::JobId;

/// A placeholder execution plan that materializes a relation into persistent local streams.
#[derive(Debug, Clone)]
pub struct LocalCheckpointWriteExec {
    input: Arc<dyn ExecutionPlan>,
    checkpoint_job_id: JobId,
    storage_level: spec::StorageLevel,
    properties: PlanProperties,
}

impl LocalCheckpointWriteExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        checkpoint_job_id: JobId,
        storage_level: spec::StorageLevel,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(Arc::new(Schema::empty())),
            Partitioning::UnknownPartitioning(input.output_partitioning().partition_count().max(1)),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Self {
            input,
            checkpoint_job_id,
            storage_level,
            properties,
        }
    }

    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    pub fn checkpoint_job_id(&self) -> JobId {
        self.checkpoint_job_id
    }

    pub fn storage_level(&self) -> &spec::StorageLevel {
        &self.storage_level
    }
}

impl DisplayAs for LocalCheckpointWriteExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "LocalCheckpointWriteExec: checkpoint_job_id={}, input_partitions={}, memory={}, disk={}, deserialized={}, replication={}",
            self.checkpoint_job_id,
            self.input.output_partitioning().partition_count(),
            self.storage_level.use_memory,
            self.storage_level.use_disk,
            self.storage_level.deserialized,
            self.storage_level.replication,
        )
    }
}

impl ExecutionPlan for LocalCheckpointWriteExec {
    fn name(&self) -> &str {
        "LocalCheckpointWriteExec"
    }

    fn as_any(&self) -> &dyn Any {
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
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let child = children.pop();
        match (child, children.is_empty()) {
            (Some(input), true) => Ok(Arc::new(Self::new(
                input,
                self.checkpoint_job_id,
                self.storage_level.clone(),
            ))),
            _ => plan_err!("LocalCheckpointWriteExec should have one child"),
        }
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        datafusion::common::internal_err!("{} should be resolved before execution", self.name())
    }
}
