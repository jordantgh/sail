use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::{internal_err, Result};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

use crate::stream::reader::TaskReadLocation;

/// A placeholder execution plan that reads a materialized local checkpoint relation.
#[derive(Debug, Clone)]
pub struct LocalCheckpointReadExec {
    locations: Vec<TaskReadLocation>,
    properties: PlanProperties,
}

impl LocalCheckpointReadExec {
    pub fn new(locations: Vec<TaskReadLocation>, schema: SchemaRef) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema),
            datafusion::physical_expr::Partitioning::UnknownPartitioning(locations.len().max(1)),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Self {
            locations,
            properties,
        }
    }

    pub fn locations(&self) -> &[TaskReadLocation] {
        &self.locations
    }
}

impl DisplayAs for LocalCheckpointReadExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "LocalCheckpointReadExec: partitioning={}, partitions={}",
            self.properties.output_partitioning(),
            self.locations.len(),
        )
    }
}

impl ExecutionPlan for LocalCheckpointReadExec {
    fn name(&self) -> &str {
        "LocalCheckpointReadExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return internal_err!("{} does not accept children", self.name());
        }
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        internal_err!("{} should be resolved before execution", self.name())
    }
}
