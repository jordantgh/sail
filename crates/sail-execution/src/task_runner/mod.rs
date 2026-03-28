mod core;
mod monitor;

use std::collections::HashMap;

use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use sail_common_datafusion::error::CommonErrorCause;
use sail_server::actor::ActorHandle;
use tokio::sync::oneshot;

use crate::driver::{DriverActor, DriverClient, DriverEvent, TaskStatus};
use crate::id::{TaskKey, WorkerId};
use crate::worker::WorkerEvent;

pub struct TaskRunner {
    signals: HashMap<TaskKey, oneshot::Sender<()>>,
    codec: Box<dyn PhysicalExtensionCodec>,
}

#[derive(Clone)]
pub enum LocalCheckpointRegistrarContext {
    Driver {
        handle: ActorHandle<DriverActor>,
    },
    Worker {
        driver: DriverClient,
        worker_id: WorkerId,
    },
}

pub trait TaskRunnerMessage {
    fn report_task_status(
        key: TaskKey,
        status: TaskStatus,
        message: Option<String>,
        cause: Option<CommonErrorCause>,
    ) -> Self;
}

impl TaskRunnerMessage for DriverEvent {
    fn report_task_status(
        key: TaskKey,
        status: TaskStatus,
        message: Option<String>,
        cause: Option<CommonErrorCause>,
    ) -> Self {
        DriverEvent::UpdateTask {
            key,
            status,
            message,
            cause,
            sequence: None,
        }
    }
}

impl TaskRunnerMessage for WorkerEvent {
    fn report_task_status(
        key: TaskKey,
        status: TaskStatus,
        message: Option<String>,
        cause: Option<CommonErrorCause>,
    ) -> Self {
        WorkerEvent::ReportTaskStatus {
            key,
            status,
            message,
            cause,
        }
    }
}
