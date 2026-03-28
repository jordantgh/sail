use std::collections::HashMap;

use crate::error::{ExecutionError, ExecutionResult};
use crate::id::{JobId, TaskStreamKey, WorkerId};
use crate::stream::reader::TaskReadLocation;

#[derive(Debug, Clone, Copy)]
pub(crate) enum LocalCheckpointStreamOwner {
    Driver,
    Worker { worker_id: WorkerId },
}

#[derive(Default)]
pub(crate) struct LocalCheckpointRegistry {
    checkpoints: HashMap<JobId, LocalCheckpointMaterialization>,
}

impl LocalCheckpointRegistry {
    pub fn begin(&mut self, checkpoint_job_id: JobId, partitions: usize) -> ExecutionResult<()> {
        if partitions == 0 {
            return Err(ExecutionError::InvalidArgument(
                "local checkpoint materialization requires at least one partition".to_string(),
            ));
        }
        if self
            .checkpoints
            .insert(
                checkpoint_job_id,
                LocalCheckpointMaterialization {
                    partitions,
                    locations: HashMap::new(),
                },
            )
            .is_some()
        {
            return Err(ExecutionError::InternalError(format!(
                "local checkpoint {} is already being materialized",
                checkpoint_job_id
            )));
        }
        Ok(())
    }

    pub fn register(
        &mut self,
        checkpoint_job_id: JobId,
        key: TaskStreamKey,
        owner: LocalCheckpointStreamOwner,
    ) -> ExecutionResult<()> {
        let Some(materialization) = self.checkpoints.get_mut(&checkpoint_job_id) else {
            return Err(ExecutionError::InvalidArgument(format!(
                "local checkpoint {} is not registered",
                checkpoint_job_id
            )));
        };
        if key.job_id != checkpoint_job_id {
            return Err(ExecutionError::InvalidArgument(format!(
                "local checkpoint stream {} does not belong to checkpoint {}",
                key.job_id, checkpoint_job_id
            )));
        }
        if key.partition >= materialization.partitions {
            return Err(ExecutionError::InvalidArgument(format!(
                "local checkpoint partition {} is out of bounds for {} partition(s)",
                key.partition, materialization.partitions
            )));
        }
        let partition = key.partition;
        let location = match owner {
            LocalCheckpointStreamOwner::Driver => TaskReadLocation::Driver { key },
            LocalCheckpointStreamOwner::Worker { worker_id } => {
                TaskReadLocation::Worker { worker_id, key }
            }
        };
        materialization.locations.insert(partition, location);
        Ok(())
    }

    pub fn finalize(&self, checkpoint_job_id: JobId) -> ExecutionResult<Vec<TaskReadLocation>> {
        let Some(materialization) = self.checkpoints.get(&checkpoint_job_id) else {
            return Err(ExecutionError::InvalidArgument(format!(
                "local checkpoint {} is not registered",
                checkpoint_job_id
            )));
        };
        let mut locations = Vec::with_capacity(materialization.partitions);
        for partition in 0..materialization.partitions {
            let location = materialization
                .locations
                .get(&partition)
                .cloned()
                .ok_or_else(|| {
                    ExecutionError::InternalError(format!(
                        "local checkpoint {} is missing partition {}",
                        checkpoint_job_id, partition
                    ))
                })?;
            locations.push(location);
        }
        Ok(locations)
    }

    pub fn remove(&mut self, checkpoint_job_id: JobId) {
        self.checkpoints.remove(&checkpoint_job_id);
    }
}

struct LocalCheckpointMaterialization {
    partitions: usize,
    locations: HashMap<usize, TaskReadLocation>,
}
