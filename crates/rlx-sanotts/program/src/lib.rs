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

//! sanoTTS's host logic as an RLX program, answering `synthesize`: what
//! `Synthesizer::synthesize_with` does around its graphs — espeak-ng's
//! phonemizer mode (the voice's language, else its `-us` / `-gb` variants),
//! the IPA NFD-decomposed and mapped through the voice's table framed `^ _
//! (p _)* $`, each component's ids clamped to its vocabulary (schwa past
//! it); the durations (`duration`: `exp`, at least 1, times the length
//! scale, rounded half to even, clamped); the token stack (`tokens`) with
//! each token's log-duration share; its expansion to frames with the frame
//! and token positions and each frame's place in its token; the frame
//! stack and decoder (`frames`), the waveform clamped to [−1, 1].
//! `options.length_scale` overrides the voice's.

use std::collections::HashMap;

use rlx_program::{Message, Program, Tensor, run_body};
use serde_json::{Value, json};
use unicode_normalization::UnicodeNormalization;

const PAD: i32 = 0;
const BOS: i32 = 1;
const EOS: i32 = 2;

struct Sano {
    table: HashMap<char, i32>,
    espeak_voice: String,
    espeak_dir: String,
    duration_vocab: usize,
    acoustic_vocab: usize,
    fallback: i32,
    max_tokens: usize,
    max_duration: usize,
    hidden: usize,
    length_scale: f32,
    hop: usize,
    sample_rate: u32,
}

fn linspace01(n: usize) -> Vec<f32> {
    match n {
        0 => Vec::new(),
        1 => vec![0.0],
        _ => {
            let last = (n - 1) as f64;
            (0..n).map(|i| (i as f64 / last) as f32).collect()
        }
    }
}

/// numpy's `round`: half to even.
fn round_half_even(v: f32) -> f32 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - v.signum()
    } else {
        r
    }
}

impl Sano {
    fn phonemize(&self, text: &str) -> Result<String, String> {
        let mut candidates = vec![self.espeak_voice.clone()];
        if !self.espeak_voice.contains('-') {
            candidates.push(format!("{}-us", self.espeak_voice));
            candidates.push(format!("{}-gb", self.espeak_voice));
        }
        let mut last = String::from("no espeak language");
        for lang in &candidates {
            let engine = match espeak_ng::EspeakNg::with_data_dir(
                lang,
                std::path::Path::new(&self.espeak_dir),
            ) {
                Ok(e) => e,
                Err(e) => {
                    last = format!("espeak-ng init for '{lang}': {e}");
                    continue;
                }
            };
            match engine.text_to_phonemes_phonemizer(text) {
                Ok(ipa) => return Ok(ipa.trim_end().to_string()),
                Err(e) => last = format!("espeak-ng phonemize ({lang}): {e}"),
            }
        }
        Err(last)
    }

    fn ids(&self, phonemes: &str) -> Vec<i32> {
        let mut ids = vec![BOS, PAD];
        for ch in phonemes.nfd() {
            if let Some(&id) = self.table.get(&ch) {
                ids.push(id);
                ids.push(PAD);
            }
        }
        ids.push(EOS);
        ids
    }

    fn clamp(&self, ids: &[i32], vocab: usize) -> Vec<i32> {
        let fallback = if (self.fallback as usize) < vocab {
            self.fallback
        } else {
            0
        };
        ids.iter()
            .map(|&i| {
                if i < 0 || i as usize >= vocab {
                    fallback
                } else {
                    i
                }
            })
            .collect()
    }

    fn speak(&self, ids: &[i32], length_scale: f32) -> Result<Vec<f32>, String> {
        let n = ids.len();
        let ones = vec![1f32; n];
        // Durations.
        let mut features = linspace01(n);
        let hint = ((n as f64).ln_1p() / (self.max_tokens as f64).ln_1p()) as f32;
        features.extend(std::iter::repeat_n(hint, n));
        features.extend(std::iter::repeat_n(1.0, n));
        let outs = run_body(
            "duration",
            vec![
                Tensor::i32("ids", vec![n], &self.clamp(ids, self.duration_vocab)),
                Tensor::f32("features", vec![1, 3, n, 1], &features),
                Tensor::f32("mask0", vec![1, 1, n, 1], &ones),
            ],
            json!({}),
        )?;
        let log_d = outs.first().ok_or("no log durations")?.to_f32();
        let durations: Vec<usize> = log_d
            .iter()
            .map(|&v| {
                let d = v.exp().max(1.0);
                round_half_even(d * length_scale).clamp(1.0, self.max_duration as f32) as usize
            })
            .collect();
        let frames: usize = durations.iter().sum();

        // The token stack.
        let mut features = linspace01(n);
        let ln_max = (durations.iter().copied().max().unwrap_or(1).max(1) as f64).ln_1p();
        features.extend(
            durations
                .iter()
                .map(|&d| ((d as f64).ln_1p() / ln_max) as f32),
        );
        let outs = run_body(
            "tokens",
            vec![
                Tensor::i32("ids", vec![n], &self.clamp(ids, self.acoustic_vocab)),
                Tensor::f32("features", vec![1, 2, n, 1], &features),
                Tensor::f32("mask0", vec![1, 1, n, 1], &ones),
            ],
            json!({}),
        )?;
        let tokens = outs.first().ok_or("no token stack")?.to_f32();

        // Expanded to frames, with the frame position, the token position
        // and each frame's place in its token.
        let h = self.hidden;
        let mut input = vec![0f32; (h + 3) * frames];
        for c in 0..h {
            let dst = &mut input[c * frames..(c + 1) * frames];
            let mut pos = 0;
            for (t, &d) in durations.iter().enumerate() {
                dst[pos..pos + d].fill(tokens[c * n + t]);
                pos += d;
            }
        }
        input[h * frames..(h + 1) * frames].copy_from_slice(&linspace01(frames));
        let token_count = n.saturating_sub(1).max(1) as f32;
        let mut pos = 0;
        for (t, &d) in durations.iter().enumerate() {
            let tp = t as f32 / token_count;
            for j in 0..d {
                input[(h + 1) * frames + pos + j] = tp;
                input[(h + 2) * frames + pos + j] = if d == 1 {
                    0.0
                } else {
                    (j as f64 / (d - 1) as f64) as f32
                };
            }
            pos += d;
        }
        let mut inputs = vec![Tensor::f32(
            "frame_input",
            vec![1, h + 3, frames, 1],
            &input,
        )];
        for (level, f) in [1usize, 8, 64, self.hop].iter().enumerate() {
            let len = frames * f;
            inputs.push(Tensor::f32(
                format!("mask{level}"),
                vec![1, 1, len, 1],
                &vec![1f32; len],
            ));
        }
        let outs = run_body("frames", inputs, json!({}))?;
        let audio = outs.last().ok_or("no audio")?.to_f32();
        Ok(audio.into_iter().map(|s| s.clamp(-1.0, 1.0)).collect())
    }
}

impl Program for Sano {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| {
            config[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or(format!("config has no {k}"))
        };
        let s = |k: &str| {
            config[k]
                .as_str()
                .map(String::from)
                .ok_or(format!("config has no {k}"))
        };
        let mut table = HashMap::new();
        for (k, v) in config["phoneme_id_map"]
            .as_object()
            .ok_or("config has no phoneme_id_map")?
        {
            let mut chars = k.chars();
            if let (Some(c), None, Some(id)) = (
                chars.next(),
                chars.next(),
                v.as_array().and_then(|a| a.first()).and_then(Value::as_i64),
            ) {
                table.insert(c, id as i32);
            }
        }
        Ok(Sano {
            table,
            espeak_voice: s("espeak_voice")?,
            espeak_dir: s("espeak")?,
            duration_vocab: u("duration_vocab")?,
            acoustic_vocab: u("acoustic_vocab")?,
            fallback: u("fallback")? as i32,
            max_tokens: u("max_tokens")?,
            max_duration: u("max_duration")?,
            hidden: u("hidden")?,
            length_scale: config["length_scale"].as_f64().unwrap_or(1.0) as f32,
            hop: u("hop")?,
            sample_rate: u("sample_rate")? as u32,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "synthesize" {
            return Err(format!("sanoTTS answers synthesize, not {task}"));
        }
        let text = req.body["text"].as_str().ok_or("synthesize takes `text`")?;
        let o = &req.body["options"];
        let length_scale = o["length_scale"]
            .as_f64()
            .map_or(self.length_scale, |v| v as f32);
        if !(length_scale > 0.0) {
            return Err(format!("length_scale must be positive, got {length_scale}"));
        }
        let clean = text.trim();
        if clean.is_empty() {
            return Err("the text is empty".into());
        }
        let ids = self.ids(&self.phonemize(clean)?);
        if ids.len() <= 3 {
            return Err(format!(
                "phonemization produced no usable phonemes for {text:?}"
            ));
        }
        let wav = self.speak(&ids, length_scale)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Sano);
