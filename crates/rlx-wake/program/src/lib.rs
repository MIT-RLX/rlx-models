// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! The wake CNN's host logic as an RLX program, answering `detect` (and
//! `run`): what the engines' `push_pcm` does over the audio pushed `chunk`
//! samples at a time. A frame is `win` samples every `hop`, so the frames
//! after each push are known from the samples so far: the mel of all of
//! them is one run (`mel`). With the wake CNN, each push that added a frame
//! and holds `least` scores the last `window` (`score_{n}`), the rest score
//! 0; with openWakeWord's (`embed` in the config), each `window` frames
//! every `step` are embedded (`embed`) and a push holding `history`
//! embeddings scores the last ones (`phrase`). A score at
//! `threshold` fires unless the last fire is within `cooldown_ms`. Returns
//! `scores` (a tensor and the body's), `fired`, each push's `t_ms`, and the
//! `detections` (the fires' `t_ms`).

use rlx_program::{Message, Program, Tensor, run_body};
use serde_json::{Value, json};

/// What scores a push.
enum Scoring {
    /// The wake CNN over the last `window` frames (`score_{n}`), once there
    /// are `least`.
    Cnn { window: usize, least: usize },
    /// openWakeWord: an embedding (`embed`) of each `window` frames every
    /// `step`, and the phrase head (`phrase`) over the last `history`.
    Embed {
        window: usize,
        step: usize,
        history: usize,
        dim: usize,
    },
}

struct Wake {
    sample_rate: usize,
    hop: usize,
    win: usize,
    n_mels: usize,
    chunk: usize,
    scoring: Scoring,
    threshold: f64,
    cooldown_ms: f64,
}

/// The first value of the output `n`.
fn first(outs: &[Tensor], n: &str) -> Result<f32, String> {
    Ok(outs
        .iter()
        .find(|t| t.name == n)
        .ok_or(format!("no {n}"))?
        .to_f32()[0])
}

impl Wake {
    /// Frames after `n` samples.
    fn frames(&self, n: usize) -> usize {
        if n < self.win {
            0
        } else {
            (n - self.win) / self.hop + 1
        }
    }
}

impl Program for Wake {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| {
            config[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or(format!("config has no {k}"))
        };
        let f = |k: &str| config[k].as_f64().ok_or(format!("config has no {k}"));
        Ok(Wake {
            sample_rate: u("sample_rate")?,
            hop: u("hop")?,
            win: u("win")?,
            n_mels: u("n_mels")?,
            chunk: u("chunk")?,
            scoring: match config.get("embed") {
                Some(e) => {
                    let e = |k: &str| {
                        e[k].as_u64()
                            .map(|v| v as usize)
                            .ok_or(format!("config.embed has no {k}"))
                    };
                    Scoring::Embed {
                        window: e("window")?,
                        step: e("step")?,
                        history: e("history")?,
                        dim: e("dim")?,
                    }
                }
                None => Scoring::Cnn {
                    window: u("window")?,
                    least: u("least")?,
                },
            },
            threshold: f("threshold")?,
            cooldown_ms: f("cooldown_ms")?,
        })
    }

    fn call(&mut self, _task: &str, req: Message) -> Result<Message, String> {
        let rate = req.body["sample_rate"]
            .as_u64()
            .map_or(self.sample_rate, |v| v as usize);
        if rate != self.sample_rate {
            return Err(format!(
                "the audio is at {rate} Hz; this model takes {} Hz",
                self.sample_rate
            ));
        }
        let pcm = req.tensor("audio").ok_or("detect takes `audio`")?.to_f32();
        let o = &req.body["options"];
        let threshold = o["threshold"].as_f64().unwrap_or(self.threshold);
        let cooldown = o["cooldown_ms"].as_f64().unwrap_or(self.cooldown_ms);

        // Every frame's mel, at once.
        let total = self.frames(pcm.len());
        let mut mel = Vec::new();
        if total > 0 {
            let r = self.win.div_ceil(self.hop);
            let len = self.hop * (total + r - 1);
            let mut signal = vec![0f32; len];
            let used = (self.hop * (total - 1) + self.win).min(pcm.len());
            signal[..used].copy_from_slice(&pcm[..used]);
            let outs = run_body(
                "mel",
                vec![Tensor::f32("signal", vec![len], &signal)],
                json!({}),
            )?;
            mel = outs
                .iter()
                .find(|t| t.name == "mel")
                .ok_or("no mel")?
                .to_f32();
        }

        // Each push's score.
        let (mut scores, mut fired, mut times, mut detections) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut last_fire = -1.0e9f64;
        let (mut seen, mut before) = (0usize, 0usize);
        let mut embeds: Vec<f32> = Vec::new();
        while seen < pcm.len() {
            seen = (seen + self.chunk).min(pcm.len());
            let after = self.frames(seen);
            let score = match self.scoring {
                Scoring::Cnn { window, least } => {
                    let w = after.min(window);
                    if after > before && w >= least {
                        let rows = &mel[(after - w) * self.n_mels..after * self.n_mels];
                        let outs = run_body(
                            &format!("score_{w}"),
                            vec![Tensor::f32("mel", vec![w, self.n_mels], rows)],
                            json!({}),
                        )?;
                        first(&outs, "score")?
                    } else {
                        0.0
                    }
                }
                Scoring::Embed {
                    window,
                    step,
                    history,
                    dim,
                } => {
                    let count = if after < window {
                        0
                    } else {
                        (after - window) / step + 1
                    };
                    while embeds.len() / dim.max(1) < count {
                        let k = embeds.len() / dim;
                        let rows = &mel[k * step * self.n_mels..(k * step + window) * self.n_mels];
                        let outs = run_body(
                            "embed",
                            vec![Tensor::f32("mel", vec![window, self.n_mels], rows)],
                            json!({}),
                        )?;
                        embeds.extend(
                            outs.iter()
                                .find(|t| t.name == "embedding")
                                .ok_or("no embedding")?
                                .to_f32(),
                        );
                    }
                    if count >= history {
                        let rows = &embeds[(count - history) * dim..count * dim];
                        let outs = run_body(
                            "phrase",
                            vec![Tensor::f32("embeddings", vec![history, dim], rows)],
                            json!({}),
                        )?;
                        first(&outs, "score")?
                    } else {
                        0.0
                    }
                }
            };
            before = after;
            let t_ms = seen as f64 * 1000.0 / self.sample_rate as f64;
            let fire = f64::from(score) >= threshold && t_ms - last_fire >= cooldown;
            if fire {
                last_fire = t_ms;
                detections.push(t_ms);
            }
            scores.push(score);
            fired.push(fire);
            times.push(t_ms);
        }
        let n = scores.len();
        Ok(Message::new(
            json!({"scores": scores, "fired": fired, "t_ms": times, "detections": detections}),
            vec![Tensor::f32("scores", vec![n], &scores)],
        ))
    }
}

rlx_program::program!(Wake);
