use super::duplex::{DuplexStreamEngine, StreamStepOutput};
use crate::session::{GenerationConfig, MoshiSession};
use anyhow::Result;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

/// Events emitted by the duplex worker thread.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Ready,
    Step(StreamStepOutput),
    OutputPcm { step: usize, samples: Vec<f32> },
    Text { step: usize, text: String },
    Finished(StreamStats),
    Error(String),
}

#[derive(Debug, Clone)]
pub struct StreamStats {
    pub steps: usize,
    pub output_samples: usize,
    pub device: rlx_runtime::Device,
}

/// Commands sent to the duplex worker.
#[derive(Debug)]
pub enum StreamCommand {
    Pcm(Vec<f32>),
    Finish,
    Stop,
}

/// Handle to a running duplex stream (std::sync::mpsc).
pub struct StreamHandle {
    pub cmd_tx: Sender<StreamCommand>,
    pub event_rx: Receiver<StreamEvent>,
    join: Option<JoinHandle<()>>,
}

impl StreamHandle {
    pub fn stop(mut self) {
        let _ = self.cmd_tx.send(StreamCommand::Stop);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

/// Spawn duplex streaming on a dedicated worker thread (LM + Mimi).
pub fn spawn_duplex_stream(
    session: MoshiSession,
    prompt: &str,
    run_cfg: GenerationConfig,
) -> Result<StreamHandle> {
    let (cmd_tx, cmd_rx) = mpsc::channel::<StreamCommand>();
    let (event_tx, event_rx) = mpsc::channel::<StreamEvent>();
    let prompt = prompt.to_string();
    let join = thread::spawn(move || {
        let worker = || -> Result<()> {
            event_tx.send(StreamEvent::Ready)?;
            let mut engine = DuplexStreamEngine::from_session(session, &prompt, &run_cfg)?;
            loop {
                match cmd_rx.recv() {
                    Ok(StreamCommand::Pcm(pcm)) => {
                        for step in engine.feed_pcm(&pcm)? {
                            emit_step_events(&event_tx, step)?;
                        }
                    }
                    Ok(StreamCommand::Finish) => {
                        for step in engine.finish()? {
                            emit_step_events(&event_tx, step)?;
                        }
                        let _ = event_tx.send(StreamEvent::Finished(StreamStats {
                            steps: engine.steps_done(),
                            output_samples: 0,
                            device: engine.device(),
                        }));
                        break;
                    }
                    Ok(StreamCommand::Stop) | Err(_) => break,
                }
            }
            Ok(())
        };
        if let Err(e) = worker() {
            let _ = event_tx.send(StreamEvent::Error(e.to_string()));
        }
    });
    Ok(StreamHandle {
        cmd_tx,
        event_rx,
        join: Some(join),
    })
}

pub(crate) fn emit_step_events(tx: &Sender<StreamEvent>, step: StreamStepOutput) -> Result<()> {
    if let Some(text) = step.transcript_delta.clone() {
        let _ = tx.send(StreamEvent::Text {
            step: step.step,
            text,
        });
    }
    if !step.moshi_pcm.is_empty() {
        let _ = tx.send(StreamEvent::OutputPcm {
            step: step.step,
            samples: step.moshi_pcm.clone(),
        });
    }
    let _ = tx.send(StreamEvent::Step(step));
    Ok(())
}
