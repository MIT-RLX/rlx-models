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

//! OpenVoice's host logic as an RLX program, answering `synthesize`: what
//! `OpenVoice::synthesize` does — MeloTTS speaks the text (rlx-tiny-tts'
//! program, `voice.rs`), both voices are resampled to 22.05 kHz and
//! `center`-padded, and the tone converter carries the prompt's timbre over.
//! The spectrograms, the tone extractor and the converter run in rlx.
//!
//! The voice to clone is `options.prompt`: `{audio: [samples], sample_rate}`.

use rlx_program::{Message, Program, Tensor, run};
use serde_json::{Value, json};

#[allow(dead_code, clippy::all)]
#[path = "../../../rlx-tiny-tts/program/src/frontend/mod.rs"]
mod frontend;
#[allow(dead_code, clippy::all)]
#[path = "../../../rlx-tiny-tts/src/glue.rs"]
mod glue;
#[path = "../../../rlx-tiny-tts/program/src/voice.rs"]
mod voice;

const N_FFT: usize = 1024;
const HOP: usize = 256;
const N_FREQ: usize = N_FFT / 2 + 1;
const TONE: usize = 256;

struct OpenVoice {
    tts: voice::TinyTts,
    rate: u32,
    tau: f32,
}

/// Linear resampling, the runner's.
fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let n = (x.len() as u64 * to as u64 / from as u64).max(1) as usize;
    (0..n)
        .map(|i| {
            let s = i as f64 * from as f64 / to as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let a = x[idx.min(x.len() - 1)];
            let b = x[(idx + 1).min(x.len() - 1)];
            a + (b - a) * f
        })
        .collect()
}

/// `wav`'s magnitude spectrogram `[t, N_FREQ]` (time-major), and `t`: the
/// signal reflected `(N_FFT − HOP) / 2` each end, then the package's graph.
fn spectrogram(wav: &[f32]) -> Result<(Vec<f32>, usize), String> {
    let pad = (N_FFT - HOP) / 2;
    if wav.len() <= pad + 1 {
        return Err("the audio is too short".into());
    }
    let n = wav.len();
    let mut padded = Vec::with_capacity(n + 2 * pad);
    for i in 0..pad {
        padded.push(wav[pad - i]);
    }
    padded.extend_from_slice(wav);
    for i in 0..pad {
        padded.push(wav[n - 2 - i]);
    }
    let t = (padded.len() - N_FFT) / HOP + 1;
    let signal = &padded[..HOP * (t + N_FFT / HOP - 1)];
    let out = run(
        "spectrogram",
        vec![Tensor::f32("signal", vec![signal.len()], signal)],
    )?;
    Ok((out.first().ok_or("no spectrogram")?.to_f32(), t))
}

impl OpenVoice {
    /// The tone-color embedding of `wav` (at the converter's rate).
    fn tone(&self, wav: &[f32]) -> Result<Vec<f32>, String> {
        let (spec, t) = spectrogram(wav)?;
        let out = run(
            "tone_extract",
            vec![Tensor::f32("input", vec![1, t, N_FREQ], &spec)],
        )?;
        let e = out
            .first()
            .ok_or("the tone extractor gave no output")?
            .to_f32();
        if e.len() < TONE {
            return Err(format!("a tone embedding of {}", e.len()));
        }
        Ok(e[..TONE].to_vec())
    }
}

impl Program for OpenVoice {
    fn init(config: Value) -> Result<Self, String> {
        Ok(OpenVoice {
            tts: voice::TinyTts::load(&config)?,
            rate: config["tone_rate"]
                .as_u64()
                .ok_or("config has no tone_rate")? as u32,
            tau: config["tau"].as_f64().ok_or("config has no tau")? as f32,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "synthesize" {
            return Err(format!("OpenVoice answers synthesize, not {task}"));
        }
        let text = req.body["text"].as_str().ok_or("synthesize takes text")?;
        let o = &req.body["options"];
        let prompt = &o["prompt"];
        let reference: Vec<f32> = prompt["audio"]
            .as_array()
            .ok_or("OpenVoice clones a voice: options.prompt.audio, its samples")?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        let ref_sr = prompt["sample_rate"]
            .as_u64()
            .ok_or("options.prompt.sample_rate")? as u32;
        let tau = o["tau"].as_f64().map_or(self.tau, |v| v as f32);
        let tts = &self.tts;
        let base = tts.speak(text, tts.length_scale, tts.noise_scale, tts.seed)?;
        let base = resample(&base, tts.sample_rate, self.rate);
        let reference = resample(&reference, ref_sr, self.rate);
        let src_tone = self.tone(&base)?;
        let dest_tone = self.tone(&reference)?;
        let (spec, t) = spectrogram(&base)?;
        // The converter takes it frequency-major.
        let mut freq_major = vec![0f32; N_FREQ * t];
        for frame in 0..t {
            for f in 0..N_FREQ {
                freq_major[f * t + frame] = spec[frame * N_FREQ + f];
            }
        }
        let out = run(
            "tone_color",
            vec![
                Tensor::f32("audio", vec![1, N_FREQ, t], &freq_major),
                Tensor::new(
                    "audio_length",
                    "i64",
                    vec![1],
                    (t as i64).to_le_bytes().to_vec(),
                ),
                Tensor::f32("src_tone", vec![1, TONE, 1], &src_tone),
                Tensor::f32("dest_tone", vec![1, TONE, 1], &dest_tone),
                Tensor::f32("tau", vec![1], &[tau]),
            ],
        )?;
        let wav = out.first().ok_or("the converter gave no output")?.to_f32();
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(OpenVoice);
