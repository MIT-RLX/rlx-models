use super::duplex::{DuplexStreamEngine, StreamStepOutput};
use super::sync::{StreamCommand, StreamEvent, StreamStats};
use crate::session::{GenerationConfig, MoshiSession};
use anyhow::Result;

pub type TokioStreamEvent = StreamEvent;
pub type TokioStreamCommand = StreamCommand;

/// Tokio mpsc handle for duplex streaming.
pub struct TokioStreamHandle {
    pub cmd_tx: tokio::sync::mpsc::Sender<StreamCommand>,
    pub event_rx: tokio::sync::mpsc::Receiver<StreamEvent>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl TokioStreamHandle {
    pub fn stop(mut self) {
        let _ = self.cmd_tx.blocking_send(StreamCommand::Stop);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

/// Worker on std thread; command/event channels are tokio mpsc (blocking_send/recv).
pub fn spawn_duplex_tokio(
    session: MoshiSession,
    prompt: &str,
    run_cfg: GenerationConfig,
    channel_capacity: usize,
) -> Result<TokioStreamHandle> {
    let (cmd_tx, mut cmd_rx) = tokio::sync::mpsc::channel::<StreamCommand>(channel_capacity);
    let (event_tx, event_rx) = tokio::sync::mpsc::channel::<StreamEvent>(channel_capacity);
    let prompt = prompt.to_string();
    let join = std::thread::spawn(move || {
        let worker = || -> Result<()> {
            let _ = event_tx.blocking_send(StreamEvent::Ready);
            let mut engine = DuplexStreamEngine::from_session(session, &prompt, &run_cfg)?;
            while let Some(cmd) = cmd_rx.blocking_recv() {
                match cmd {
                    StreamCommand::Pcm(pcm) => {
                        for step in engine.feed_pcm(&pcm)? {
                            emit_step_events(&event_tx, step)?;
                        }
                    }
                    StreamCommand::Finish => {
                        for step in engine.finish()? {
                            emit_step_events(&event_tx, step)?;
                        }
                        let _ = event_tx.blocking_send(StreamEvent::Finished(StreamStats {
                            steps: engine.steps_done(),
                            output_samples: 0,
                            device: engine.device(),
                        }));
                        break;
                    }
                    StreamCommand::Stop => break,
                }
            }
            Ok(())
        };
        if let Err(e) = worker() {
            let _ = event_tx.blocking_send(StreamEvent::Error(e.to_string()));
        }
    });
    Ok(TokioStreamHandle {
        cmd_tx,
        event_rx,
        join: Some(join),
    })
}

fn emit_step_events(
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
    step: StreamStepOutput,
) -> Result<()> {
    if let Some(text) = step.transcript_delta.clone() {
        let _ = tx.blocking_send(StreamEvent::Text {
            step: step.step,
            text,
        });
    }
    if !step.moshi_pcm.is_empty() {
        let _ = tx.blocking_send(StreamEvent::OutputPcm {
            step: step.step,
            samples: step.moshi_pcm.clone(),
        });
    }
    let _ = tx.blocking_send(StreamEvent::Step(step));
    Ok(())
}
