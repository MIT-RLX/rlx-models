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

//! F5-TTS's host logic as an RLX program, answering `synthesize`: what
//! `F5Native::synthesize` does around its graphs — the reference audio's
//! trim, the reference text's normalization and the character tokenizer
//! (`rlx-f5tts`' own modules, the vocabulary read from the package), the
//! duration estimate, the chunking and the crossfade. The preprocessor,
//! the transformer (each flow step) and the decoder run in rlx; the
//! tensors between them travel in f16, as the runner's do.
//!
//! The voice to clone is `options.prompt`: `{audio: [samples], sample_rate:
//! 24000, text}`.

use std::path::Path;

use half::f16;
use rlx_program::{Message, Program, Tensor, run, run_sized};
use serde_json::{Value, json};

#[allow(dead_code, clippy::all)]
#[path = "../../src/config.rs"]
mod config;
#[allow(dead_code, clippy::all)]
#[path = "../../src/dsp.rs"]
mod dsp;
#[allow(dead_code, clippy::all)]
#[path = "../../src/tokenize.rs"]
mod tokenize;

use config::{HOP_LENGTH, SAMPLE_RATE};

struct F5 {
    vocab: config::Vocab,
    nfe: usize,
    max_duration: usize,
}

/// `x` as an f16 tensor (the runner's edges between graphs).
fn half(name: &str, shape: Vec<usize>, x: &[f32]) -> Tensor {
    let data = x
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect();
    Tensor::new(name, "f16", shape, data)
}

fn estimate_duration(ref_audio_len: f64, ref_tl: f64, gen_text: &str, speed: f32) -> usize {
    let gen_tl = tokenize::text_len(gen_text) as f64;
    (ref_audio_len + (ref_audio_len / ref_tl.max(1.0) * gen_tl / speed.max(0.1) as f64)) as usize
}

/// `gen_text` split so each piece's estimated frames stay within `cap`:
/// by sentence, an oversized one by word (the runner's).
fn chunk_gen_text(
    gen_text: &str,
    ref_audio_len: f64,
    ref_tl: f64,
    speed: f32,
    cap: usize,
) -> Vec<String> {
    let trimmed = gen_text.trim();
    if trimmed.is_empty() {
        return vec![String::new()];
    }
    if estimate_duration(ref_audio_len, ref_tl, trimmed, speed) <= cap {
        return vec![trimmed.to_string()];
    }
    let mut sentences: Vec<&str> = Vec::new();
    let mut start = 0;
    for (i, &b) in trimmed.as_bytes().iter().enumerate() {
        if matches!(b, b'.' | b'!' | b'?') {
            let end = (i + 1).min(trimmed.len());
            let s = trimmed[start..end].trim();
            if !s.is_empty() {
                sentences.push(s);
            }
            start = end;
        }
    }
    if start < trimmed.len() {
        let s = trimmed[start..].trim();
        if !s.is_empty() {
            sentences.push(s);
        }
    }
    if sentences.is_empty() {
        sentences.push(trimmed);
    }
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for s in sentences {
        let candidate = if cur.is_empty() {
            s.to_string()
        } else {
            format!("{cur} {s}")
        };
        if estimate_duration(ref_audio_len, ref_tl, &candidate, speed) <= cap {
            cur = candidate;
            continue;
        }
        if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        if estimate_duration(ref_audio_len, ref_tl, s, speed) <= cap {
            cur = s.to_string();
        } else {
            let mut piece = String::new();
            for w in s.split_whitespace() {
                let candidate = if piece.is_empty() {
                    w.to_string()
                } else {
                    format!("{piece} {w}")
                };
                if estimate_duration(ref_audio_len, ref_tl, &candidate, speed) <= cap {
                    piece = candidate;
                } else {
                    if !piece.is_empty() {
                        out.push(std::mem::take(&mut piece));
                    }
                    piece = w.to_string();
                }
            }
            if !piece.is_empty() {
                out.push(piece);
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    if out.is_empty() {
        out.push(trimmed.to_string());
    }
    out
}

fn crossfade_append(dst: &mut Vec<f32>, src: &[f32], fade: usize) {
    let fade = fade.min(dst.len()).min(src.len());
    if fade == 0 {
        dst.extend_from_slice(src);
        return;
    }
    let d0 = dst.len() - fade;
    for i in 0..fade {
        let a = (i as f32 + 1.0) / (fade as f32 + 1.0);
        dst[d0 + i] = dst[d0 + i] * (1.0 - a) + src[i] * a;
    }
    dst.extend_from_slice(&src[fade..]);
}

impl F5 {
    fn speak_one(
        &self,
        gen_text: &str,
        ref_audio: &[f32],
        ref_text: &str,
        speed: f32,
    ) -> Result<Vec<f32>, String> {
        let ids = tokenize::encode(ref_text, gen_text, &self.vocab);
        if ids.is_empty() {
            return Err("empty text".into());
        }
        let n = ref_audio.len();
        let ref_audio_len = (n / HOP_LENGTH + 1) as f64;
        let ref_tl = tokenize::text_len(ref_text).max(1) as f64;
        let gen_tl = tokenize::text_len(gen_text) as f64;
        let d = ((ref_audio_len + (ref_audio_len / ref_tl * gen_tl / speed as f64)) as usize)
            .min(self.max_duration);
        if d == 0 {
            return Err("zero duration".into());
        }
        let t = ids.len();
        let pre = run_sized(
            "F5_Preprocess",
            vec![
                half("audio", vec![1, 1, n], ref_audio),
                Tensor::i32("text_ids", vec![1, t], &ids),
            ],
            &[("max_duration", d)],
        )?;
        if pre.len() < 7 {
            return Err(format!("the preprocessor gave {} outputs", pre.len()));
        }
        let feed = |i: usize, name: &str| half(name, pre[i].shape.clone(), &pre[i].to_f32());
        let rest = [
            feed(1, "rope_cos"),
            feed(2, "rope_sin"),
            feed(3, "cat_mel_text"),
            feed(4, "cat_mel_text_drop"),
            feed(5, "qk_rotated_empty"),
        ];
        let ref_signal_len = pre[6].to_i64().first().copied().unwrap_or(0);
        let mut noise = feed(0, "noise");
        for step in 0..self.nfe.saturating_sub(1).max(1) {
            let mut inputs = vec![noise.clone()];
            inputs.extend(rest.iter().cloned());
            inputs.push(Tensor::i32("time_step", vec![1], &[step as i32]));
            let out = run("F5_Transformer", inputs)?;
            let out = out.first().ok_or("the transformer gave no output")?;
            noise = half("noise", out.shape.clone(), &out.to_f32());
        }
        let mut denoised = noise;
        denoised.name = "denoised".into();
        let dec = run("F5_Decode", vec![denoised])?;
        let mut wav = dec.first().ok_or("the decoder gave no output")?.to_f32();
        // The reference's frames off the front, as the runner crops them.
        let skip = (ref_signal_len.max(0) as usize).saturating_mul(HOP_LENGTH);
        if wav.len() > skip + HOP_LENGTH {
            wav = wav[skip..].to_vec();
        }
        dsp::soft_peak_limit(&mut wav, 0.95);
        Ok(wav)
    }
}

impl Program for F5 {
    fn init(config: Value) -> Result<Self, String> {
        Ok(F5 {
            vocab: config::Vocab::load(Path::new("/package")).map_err(|e| e.to_string())?,
            nfe: config["nfe"].as_u64().ok_or("config has no nfe")? as usize,
            max_duration: config["max_duration"]
                .as_u64()
                .ok_or("config has no max_duration")? as usize,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "synthesize" {
            return Err(format!("F5-TTS answers synthesize, not {task}"));
        }
        let text = req.body["text"].as_str().ok_or("synthesize takes text")?;
        let o = &req.body["options"];
        let prompt = &o["prompt"];
        let audio: Vec<f32> = prompt["audio"]
            .as_array()
            .ok_or("F5-TTS clones a voice: options.prompt.audio, its samples")?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        let sr = prompt["sample_rate"]
            .as_u64()
            .unwrap_or(u64::from(SAMPLE_RATE)) as u32;
        if sr != SAMPLE_RATE {
            return Err(format!("the prompt must be {SAMPLE_RATE} Hz"));
        }
        let ref_text = prompt["text"]
            .as_str()
            .ok_or("options.prompt.text, its transcript")?;
        let speed = o["speed"].as_f64().unwrap_or(1.0) as f32;
        if let Some(n) = o["nfe"].as_u64() {
            self.nfe = n as usize;
        }

        let ref_text = tokenize::normalize_ref_text(ref_text);
        let ref_audio = dsp::preprocess_ref_audio(&audio, sr);
        let ref_audio_len = (ref_audio.len() / HOP_LENGTH + 1) as f64;
        let ref_tl = tokenize::text_len(&ref_text).max(1) as f64;
        let chunks = chunk_gen_text(text, ref_audio_len, ref_tl, speed, self.max_duration);
        let n_chunks = chunks.len();
        let mut out = Vec::new();
        for chunk in &chunks {
            let pcm = self.speak_one(chunk, &ref_audio, &ref_text, speed)?;
            let peak = pcm.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            if peak < 1e-3 {
                if n_chunks == 1 {
                    return Err(format!("the audio is silent (peak {peak:.2e})"));
                }
                continue;
            }
            if out.is_empty() {
                out = pcm;
            } else {
                crossfade_append(&mut out, &pcm, (SAMPLE_RATE / 50) as usize);
            }
        }
        if out.is_empty() {
            return Err("every chunk was silent".into());
        }
        let n = out.len();
        Ok(Message::new(
            json!({"sample_rate": SAMPLE_RATE}),
            vec![Tensor::f32("audio", vec![n], &out)],
        ))
    }
}

rlx_program::program!(F5);
