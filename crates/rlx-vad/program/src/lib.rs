// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! A frame-recurrent VAD's host logic as an RLX program, answering `run`:
//! what `speech_segments_silero` / `speech_segments_earshot` do around the
//! network — the audio in frames (the last zero-padded), each scored by the
//! package's `step` graph (after the previous frame's last `context`
//! samples, for Silero), its recurrent state carried from frame to frame —
//! and the speech segments from the probabilities (the crate's own
//! `segments.rs`, with its parameters for the `algorithm`).
//!
//! `step` takes `frame` and each of `states` (zeros at the start) and
//! returns `prob` then their next values, in order.
//!
//! In: `audio`, `f32 [samples]` at 16 kHz. Out: `probs` (`[frames]`) and
//! `segments` (`i32 [n, 2]`, start and end samples). Options: `threshold`,
//! `min_speech_ms`, `min_silence_ms`, `speech_pad_ms`.

#[allow(dead_code)]
#[path = "../../src/segments.rs"]
mod segments;

use rlx_program::{Message, Program, Tensor, run};
use serde_json::{Value, json};

/// As `rlx_vad::SpeechSegment`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechSegment {
    pub start: usize,
    pub end: usize,
}

struct Vad {
    frame: usize,
    context: usize,
    /// The step's recurrent inputs: name and length.
    states: Vec<(String, usize)>,
    algorithm: String,
    sample_rate: u64,
}

impl Program for Vad {
    fn init(config: Value) -> Result<Self, String> {
        let n = |k: &str| {
            config[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or(format!("config has no {k}"))
        };
        let states = config["states"]
            .as_array()
            .ok_or("config has no states")?
            .iter()
            .filter_map(|s| {
                Some((
                    s["name"].as_str()?.to_string(),
                    s["size"].as_u64()? as usize,
                ))
            })
            .collect();
        Ok(Vad {
            frame: n("frame")?,
            context: config["context"].as_u64().unwrap_or(0) as usize,
            states,
            algorithm: config["algorithm"].as_str().unwrap_or("silero").to_string(),
            sample_rate: config["sample_rate"].as_u64().unwrap_or(16_000),
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "run" {
            return Err(format!("the VAD answers run, not {task}"));
        }
        let audio = req.tensor("audio").ok_or("run takes `audio`")?.to_f32();
        let o = &req.body["options"];
        let mut params = segments::SegmentParams::for_algorithm(&self.algorithm);
        let ms = |k: &str, d: usize| {
            o[k].as_u64()
                .map_or(d, |v| v as usize * self.sample_rate as usize / 1000)
        };
        if let Some(t) = o["threshold"].as_f64() {
            params.threshold = t as f32;
        }
        params.min_speech_samples = ms("min_speech_ms", params.min_speech_samples);
        params.min_silence_samples = ms("min_silence_ms", params.min_silence_samples);
        params.speech_pad_samples = ms("speech_pad_ms", params.speech_pad_samples);

        let (f, ctx) = (self.frame, self.context);
        let mut context = vec![0f32; ctx];
        let mut states: Vec<Vec<f32>> = self.states.iter().map(|(_, n)| vec![0f32; *n]).collect();
        let mut probs = Vec::new();
        let mut input = vec![0f32; ctx + f];
        for chunk in audio.chunks(f) {
            input[..ctx].copy_from_slice(&context);
            input[ctx..].fill(0.0);
            input[ctx..ctx + chunk.len()].copy_from_slice(chunk);
            let mut inputs = vec![Tensor::f32("frame", vec![ctx + f], &input)];
            for ((name, n), v) in self.states.iter().zip(&states) {
                inputs.push(Tensor::f32(name.clone(), vec![*n], v));
            }
            let out = run("step", inputs)?;
            if out.len() != 1 + states.len() {
                return Err(format!(
                    "the step returns {} tensors, not prob and {} states",
                    out.len(),
                    states.len()
                ));
            }
            probs.push(out[0].to_f32()[0]);
            for (s, t) in states.iter_mut().zip(&out[1..]) {
                *s = t.to_f32();
            }
            // The context is the frame's own last samples (the padded one's).
            context.copy_from_slice(&input[f..]);
        }
        let segs = segments::speech_segments_from_probs(audio.len(), f, &params, &probs);
        let flat: Vec<i32> = segs
            .iter()
            .flat_map(|s| [s.start as i32, s.end as i32])
            .collect();
        Ok(Message::new(
            json!({"segments": segs.iter().map(|s| [s.start, s.end]).collect::<Vec<_>>()}),
            vec![
                Tensor::f32("probs", vec![probs.len()], &probs),
                Tensor::i32("segments", vec![segs.len(), 2], &flat),
            ],
        ))
    }
}

rlx_program::program!(Vad);
