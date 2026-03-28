use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::Cursor;
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::ipc::reader::FileReader;
use datafusion::arrow::ipc::writer::FileWriter;
use datafusion::common::{internal_datafusion_err, Result};
use futures::stream;
use log::debug;
use sail_common::spec;
use tokio::sync::mpsc;
use tonic::codegen::tokio_stream::wrappers::ReceiverStream;

use crate::error::{ExecutionError, ExecutionResult};
use crate::id::TaskStreamKey;
use crate::stream::error::TaskStreamResult;
use crate::stream::reader::TaskStreamSource;
use crate::stream::writer::{TaskStreamSink, TaskStreamSinkState};

pub trait LocalStream: Send {
    fn publish(&mut self) -> ExecutionResult<Box<dyn TaskStreamSink>>;
    fn subscribe(&mut self) -> ExecutionResult<TaskStreamSource>;
}

/// A memory stream that can be read multiple times.
/// It maintains multiple replicas of the stream internally.
/// Since [`Arc`] is used inside the record batch, it is relatively cheap
/// to clone the data in multiple replicas.
pub(crate) struct MemoryStream {
    sender: Option<MemoryStreamReplicaSender>,
    receivers: Vec<mpsc::Receiver<TaskStreamResult<RecordBatch>>>,
}

impl MemoryStream {
    pub fn new(
        buffer: usize,
        replicas: usize,
        senders: Vec<mpsc::Sender<TaskStreamResult<RecordBatch>>>,
    ) -> Self {
        let replicas = replicas.max(senders.len());
        let diff = replicas - senders.len();
        let mut senders = senders.into_iter().map(Some).collect::<Vec<_>>();
        senders.reserve(diff);
        let mut receivers = Vec::with_capacity(diff);
        for _ in 0..diff {
            let (tx, rx) = mpsc::channel(buffer);
            senders.push(Some(tx));
            receivers.push(rx);
        }
        let overflow = vec![VecDeque::new(); senders.len()];
        Self {
            sender: Some(MemoryStreamReplicaSender { senders, overflow }),
            receivers,
        }
    }
}

impl LocalStream for MemoryStream {
    fn publish(&mut self) -> ExecutionResult<Box<dyn TaskStreamSink>> {
        let sender = self.sender.take().ok_or_else(|| {
            ExecutionError::InternalError("memory stream can only be written once".to_string())
        })?;
        Ok(Box::new(sender))
    }

    fn subscribe(&mut self) -> ExecutionResult<TaskStreamSource> {
        let rx = self.receivers.pop().ok_or_else(|| {
            ExecutionError::InternalError("memory stream has exhausted all replica(s)".to_string())
        })?;
        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}

struct MemoryStreamReplicaSender {
    senders: Vec<Option<mpsc::Sender<TaskStreamResult<RecordBatch>>>>,
    /// An overflow buffer for each sender to avoid blocking sending for slow senders.
    /// This also avoids deadlock situations where the task stream buffer size is small.
    // TODO: More investigation is needed to understand why deadlocks might happen among stages
    //   when the task stream buffer is of a limited size.
    overflow: Vec<VecDeque<TaskStreamResult<RecordBatch>>>,
}

#[tonic::async_trait]
impl TaskStreamSink for MemoryStreamReplicaSender {
    async fn write(&mut self, batch: TaskStreamResult<RecordBatch>) -> TaskStreamSinkState {
        let mut active = false;
        for (i, sender) in self.senders.iter_mut().enumerate() {
            if sender.is_none() {
                continue;
            }

            let overflow = &mut self.overflow[i];
            let mut dropped = false;

            if let Some(tx) = sender.as_ref() {
                // Try to flush overflow first
                while let Some(item) = overflow.pop_front() {
                    match tx.try_send(item) {
                        Ok(_) => {}
                        Err(mpsc::error::TrySendError::Full(x)) => {
                            overflow.push_front(x);
                            break;
                        }
                        Err(mpsc::error::TrySendError::Closed(_)) => {
                            dropped = true;
                            break;
                        }
                    }
                }
            }

            // A dropped receiver can happen under normal operation when the receiver no longer
            // needs more data (e.g., after a LIMIT operator has received enough rows).

            if dropped {
                debug!("memory stream replica receiver has been dropped");
                *sender = None;
                overflow.clear();
                continue;
            }

            if let Some(tx) = sender.as_ref() {
                if overflow.is_empty() {
                    match tx.try_send(batch.clone()) {
                        Ok(_) => {}
                        Err(mpsc::error::TrySendError::Full(x)) => {
                            overflow.push_back(x);
                        }
                        Err(mpsc::error::TrySendError::Closed(_)) => {
                            dropped = true;
                        }
                    }
                } else {
                    overflow.push_back(batch.clone());
                }
            }

            if dropped {
                debug!("memory stream replica receiver has been dropped");
                *sender = None;
                overflow.clear();
            } else {
                active = true;
            }
        }
        if active {
            TaskStreamSinkState::Ok
        } else {
            TaskStreamSinkState::Closed
        }
    }

    async fn close(mut self: Box<Self>) -> Result<()> {
        for (i, sender) in self.senders.iter_mut().enumerate() {
            if sender.is_none() {
                continue;
            }

            let overflow = &mut self.overflow[i];
            let mut dropped = false;
            while let Some(item) = overflow.pop_front() {
                if let Some(tx) = sender.as_ref() {
                    // TODO: `send` here is blocking and may introduce deadlocks among tasks.
                    //   This is low-risk empirically though.
                    if tx.send(item).await.is_err() {
                        dropped = true;
                        break;
                    }
                }
            }

            if dropped {
                *sender = None;
                overflow.clear();
            }
        }
        Ok(())
    }
}

pub(crate) struct PersistentLocalCheckpointStream {
    state: Arc<PersistentLocalCheckpointState>,
    publisher_open: bool,
}

impl PersistentLocalCheckpointStream {
    pub fn new(
        key: &TaskStreamKey,
        schema: SchemaRef,
        storage_level: spec::StorageLevel,
    ) -> ExecutionResult<Self> {
        let disk_path = storage_level
            .use_disk
            .then(|| checkpoint_stream_path(key))
            .transpose()?;
        let disk_writer = match &disk_path {
            Some(path) => {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                Some(
                    FileWriter::try_new(File::create(path)?, &schema).map_err(|e| {
                        ExecutionError::from(datafusion::error::DataFusionError::from(e))
                    })?,
                )
            }
            None => None,
        };
        Ok(Self {
            state: Arc::new(PersistentLocalCheckpointState {
                schema,
                storage_level,
                disk_path,
                inner: Mutex::new(PersistentLocalCheckpointInner::Writing {
                    memory_batches: Vec::new(),
                    disk_writer,
                }),
            }),
            publisher_open: true,
        })
    }
}

impl LocalStream for PersistentLocalCheckpointStream {
    fn publish(&mut self) -> ExecutionResult<Box<dyn TaskStreamSink>> {
        if !self.publisher_open {
            return Err(ExecutionError::InternalError(
                "persistent checkpoint stream can only be written once".to_string(),
            ));
        }
        self.publisher_open = false;
        Ok(Box::new(PersistentLocalCheckpointSink {
            state: Arc::clone(&self.state),
        }))
    }

    fn subscribe(&mut self) -> ExecutionResult<TaskStreamSource> {
        self.state.subscribe()
    }
}

struct PersistentLocalCheckpointState {
    schema: SchemaRef,
    storage_level: spec::StorageLevel,
    disk_path: Option<PathBuf>,
    inner: Mutex<PersistentLocalCheckpointInner>,
}

enum PersistentLocalCheckpointInner {
    Writing {
        memory_batches: Vec<RecordBatch>,
        disk_writer: Option<FileWriter<File>>,
    },
    Ready {
        memory: Option<PersistentLocalCheckpointMemory>,
    },
    Failed {
        message: String,
    },
}

enum PersistentLocalCheckpointMemory {
    Deserialized(Vec<RecordBatch>),
    Serialized(Vec<u8>),
}

impl PersistentLocalCheckpointState {
    fn subscribe(&self) -> ExecutionResult<TaskStreamSource> {
        let batches = {
            let inner = self
                .inner
                .lock()
                .map_err(|error| ExecutionError::InternalError(error.to_string()))?;
            match &*inner {
                PersistentLocalCheckpointInner::Writing { .. } => {
                    return Err(ExecutionError::InternalError(
                        "persistent checkpoint stream is not ready for reading".to_string(),
                    ));
                }
                PersistentLocalCheckpointInner::Ready { memory } => {
                    if let Some(memory) = memory {
                        match memory {
                            PersistentLocalCheckpointMemory::Deserialized(batches) => {
                                batches.clone()
                            }
                            PersistentLocalCheckpointMemory::Serialized(bytes) => {
                                deserialize_batches(bytes.as_slice())?
                            }
                        }
                    } else {
                        let path = self.disk_path.as_ref().ok_or_else(|| {
                            ExecutionError::InternalError(
                                "persistent checkpoint stream is missing its disk backing"
                                    .to_string(),
                            )
                        })?;
                        read_batches_from_disk(path)?
                    }
                }
                PersistentLocalCheckpointInner::Failed { message } => {
                    return Err(ExecutionError::InternalError(message.clone()));
                }
            }
        };
        let stream = stream::iter(batches.into_iter().map(Ok));
        Ok(Box::pin(stream))
    }
}

impl Drop for PersistentLocalCheckpointState {
    fn drop(&mut self) {
        let Some(path) = self.disk_path.take() else {
            return;
        };
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => debug!(
                "failed to remove checkpoint stream file {}: {error}",
                path.display()
            ),
        }
        remove_empty_checkpoint_parents(path.parent());
    }
}

struct PersistentLocalCheckpointSink {
    state: Arc<PersistentLocalCheckpointState>,
}

#[tonic::async_trait]
impl TaskStreamSink for PersistentLocalCheckpointSink {
    async fn write(&mut self, batch: TaskStreamResult<RecordBatch>) -> TaskStreamSinkState {
        let batch = match batch {
            Ok(batch) => batch,
            Err(error) => {
                let message = error.to_string();
                if let Ok(mut inner) = self.state.inner.lock() {
                    *inner = PersistentLocalCheckpointInner::Failed {
                        message: message.clone(),
                    };
                }
                return TaskStreamSinkState::Error(datafusion::error::DataFusionError::External(
                    Box::new(error),
                ));
            }
        };

        let mut inner = match self.state.inner.lock() {
            Ok(inner) => inner,
            Err(error) => {
                return TaskStreamSinkState::Error(internal_datafusion_err!("{error}"));
            }
        };
        let PersistentLocalCheckpointInner::Writing {
            memory_batches,
            disk_writer,
        } = &mut *inner
        else {
            return TaskStreamSinkState::Error(internal_datafusion_err!(
                "persistent checkpoint stream is no longer writable"
            ));
        };

        if self.state.storage_level.use_memory {
            memory_batches.push(batch.clone());
        }
        if let Some(writer) = disk_writer.as_mut() {
            if let Err(error) = writer.write(&batch) {
                *inner = PersistentLocalCheckpointInner::Failed {
                    message: error.to_string(),
                };
                return TaskStreamSinkState::Error(error.into());
            }
        }
        TaskStreamSinkState::Ok
    }

    async fn close(self: Box<Self>) -> Result<()> {
        let mut inner = self
            .state
            .inner
            .lock()
            .map_err(|error| internal_datafusion_err!("{error}"))?;
        let next = match mem::replace(
            &mut *inner,
            PersistentLocalCheckpointInner::Failed {
                message: "persistent checkpoint stream closed unexpectedly".to_string(),
            },
        ) {
            PersistentLocalCheckpointInner::Writing {
                memory_batches,
                mut disk_writer,
            } => {
                if let Some(writer) = disk_writer.as_mut() {
                    writer.finish()?;
                }
                let memory = if self.state.storage_level.use_memory {
                    Some(if self.state.storage_level.deserialized {
                        PersistentLocalCheckpointMemory::Deserialized(memory_batches)
                    } else {
                        PersistentLocalCheckpointMemory::Serialized(serialize_batches(
                            &self.state.schema,
                            &memory_batches,
                        )?)
                    })
                } else {
                    None
                };
                PersistentLocalCheckpointInner::Ready { memory }
            }
            PersistentLocalCheckpointInner::Ready { memory } => {
                *inner = PersistentLocalCheckpointInner::Ready { memory };
                return Err(internal_datafusion_err!(
                    "persistent checkpoint stream has already been closed"
                ));
            }
            PersistentLocalCheckpointInner::Failed { message } => {
                *inner = PersistentLocalCheckpointInner::Failed { message };
                return Err(internal_datafusion_err!(
                    "persistent checkpoint stream failed before close"
                ));
            }
        };
        *inner = next;
        Ok(())
    }
}

fn serialize_batches(schema: &SchemaRef, batches: &[RecordBatch]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut bytes, schema)?;
        for batch in batches {
            writer.write(batch)?;
        }
        writer.finish()?;
    }
    Ok(bytes)
}

fn deserialize_batches(bytes: &[u8]) -> ExecutionResult<Vec<RecordBatch>> {
    let reader = FileReader::try_new(Cursor::new(bytes), None)
        .map_err(|error| ExecutionError::InternalError(error.to_string()))?;
    reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| ExecutionError::InternalError(error.to_string()))
}

fn read_batches_from_disk(path: &Path) -> ExecutionResult<Vec<RecordBatch>> {
    let reader = FileReader::try_new(File::open(path)?, None)
        .map_err(|error| ExecutionError::InternalError(error.to_string()))?;
    reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| ExecutionError::InternalError(error.to_string()))
}

fn checkpoint_stream_path(key: &TaskStreamKey) -> ExecutionResult<PathBuf> {
    let root = std::env::temp_dir()
        .join("sail-local-checkpoint-streams")
        .join(key.job_id.to_string())
        .join(key.stage.to_string())
        .join(key.partition.to_string());
    Ok(root.join(format!(
        "attempt-{}-channel-{}.arrow",
        key.attempt, key.channel
    )))
}

fn remove_empty_checkpoint_parents(path: Option<&Path>) {
    let mut current = path.map(Path::to_path_buf);
    while let Some(path) = current {
        match fs::remove_dir(&path) {
            Ok(()) => {
                current = path.parent().map(Path::to_path_buf);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                current = path.parent().map(Path::to_path_buf);
            }
            Err(error) if error.kind() == std::io::ErrorKind::DirectoryNotEmpty => {
                break;
            }
            Err(error) => {
                debug!(
                    "failed to remove checkpoint stream directory {}: {error}",
                    path.display()
                );
                break;
            }
        }
    }
}
