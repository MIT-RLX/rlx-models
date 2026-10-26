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

//! Kokoro's host logic as an RLX program, answering `synthesize`: what
//! `NativeKokoro::generate_from_text` does around its two graphs, the same
//! code — the text preprocessor (rlx-kittentts'), espeak-ng (its data read
//! from the package), the phoneme vocabulary, the voice's style row, the
//! length regulator, the ISTFT normalization and the soft peak limit. The
//! encoder and decoder run in rlx (`rlx_program::run`).

use std::collections::HashMap;

use rlx_program::{Message, Program, Tensor, host, run};
use serde_json::{Value, json};

// rlx-kittentts' own file, linted there.
#[allow(dead_code, clippy::all)]
#[path = "../../../rlx-kittentts/src/preprocess.rs"]
mod preprocess;

const SAMPLE_RATE: u32 = 24_000;
const PAD_ID: i64 = 0;
const MAX_PHONEME_LEN: usize = 510;
const STYLE_DIM: usize = 256;
/// ISTFT overlap-add: n_fft/hop, and the n_fft/2 crop each end.
const OVERLAP_FACTOR: f32 = 4.0;
const ISTFT_CROP: usize = 10;

struct Kokoro {
    vocab: HashMap<char, i64>,
    voices: HashMap<String, String>,
    default_voice: String,
    window_sum: Vec<f32>,
    espeak_dir: String,
}

fn file(name: &str) -> Result<Vec<u8>, String> {
    let m = host("file", Message::body(json!({"name": name})))?;
    Ok(m.tensor("bytes").ok_or("no bytes")?.data.clone())
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// The espeak language of a voice: its prefix (`a…` American, `b…` British).
fn voice_lang(voice: &str) -> &'static str {
    match voice.chars().next() {
        Some('b') => "en-gb",
        _ => "en-us",
    }
}

impl Kokoro {
    fn phonemize(&self, lang: &str, text: &str) -> Result<String, String> {
        if text.is_empty() {
            return Ok(String::new());
        }
        let engine =
            espeak_ng::EspeakNg::with_data_dir(lang, std::path::Path::new(&self.espeak_dir))
                .map_err(|e| format!("espeak-ng init for '{lang}': {e}"))?;
        // Punctuation kept where it was (phonemizer's preserve_punctuation),
        // the spans between phonemized — as rlx-kittentts' frontend does.
        const PUNCT: &[char] = &['.', ',', '!', '?', ';', ':'];
        let mut out = String::new();
        let mut span = String::new();
        let flush = |span: &mut String, out: &mut String| -> Result<(), String> {
            if span.trim().is_empty() {
                span.clear();
                return Ok(());
            }
            let ipa = engine
                .text_to_phonemes(span.trim())
                .map_err(|e| format!("espeak-ng phonemize ({lang}): {e}"))?;
            let ipa = ipa.replace('\n', " ");
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
            out.push_str(ipa.trim());
            span.clear();
            Ok(())
        };
        for ch in text.chars() {
            if PUNCT.contains(&ch) {
                flush(&mut span, &mut out)?;
                out.push(ch);
            } else {
                span.push(ch);
            }
        }
        flush(&mut span, &mut out)?;
        Ok(out.trim().to_owned())
    }

    /// Phonemes → content ids (no pads), whitespace runs one space, unknowns
    /// dropped.
    fn content_ids(&self, phonemes: &str) -> Vec<i64> {
        let mut ids = Vec::new();
        let mut prev_space = false;
        for c in phonemes.trim().chars() {
            if c.is_whitespace() {
                if prev_space {
                    continue;
                }
                prev_space = true;
                if let Some(id) = self.vocab.get(&' ') {
                    ids.push(*id);
                }
                continue;
            }
            prev_space = false;
            if let Some(&id) = self.vocab.get(&c) {
                ids.push(id);
            }
        }
        ids
    }

    fn speak(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>, String> {
        let processed = preprocess::TextPreprocessor::new().process(text);
        let ipa = self
            .phonemize(voice_lang(voice), &processed)
            .or_else(|_| self.phonemize("en-us", &processed))?;
        let mut content = self.content_ids(&ipa);
        if content.is_empty() {
            return Err(format!("no in-vocabulary phonemes in {ipa:?}"));
        }
        content.truncate(MAX_PHONEME_LEN);
        let content_len = content.len();
        let mut ids = vec![PAD_ID];
        ids.extend(content);
        ids.push(PAD_ID);
        // The voice's style row for this many phonemes (the last past its end).
        let path = self
            .voices
            .get(voice)
            .ok_or_else(|| format!("no voice '{voice}' (have {:?})", self.voices.keys()))?;
        let v = f32s(&file(path)?);
        let rows = v.len() / STYLE_DIM;
        let r = content_len.min(rows.saturating_sub(1));
        let style = v[r * STYLE_DIM..(r + 1) * STYLE_DIM].to_vec();

        let seq = ids.len();
        let idb: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        let enc = run(
            "encoder",
            vec![
                Tensor::new("input_ids", "i64", vec![1, seq], idb),
                Tensor::f32("style", vec![1, STYLE_DIM], &style),
                Tensor::f32("speed", vec![1], &[speed]),
            ],
        )?;
        if enc.len() < 3 {
            return Err(format!("the encoder gave {} outputs", enc.len()));
        }
        let prosody = enc[0].to_f32();
        let txt = enc[1].to_f32();
        let dur: Vec<usize> = match enc[2].dtype.as_str() {
            "f32" | "f16" | "bf16" => enc[2]
                .to_f32()
                .iter()
                .map(|&x| x.round().max(0.0) as usize)
                .collect(),
            _ => enc[2].to_i64().iter().map(|&x| x.max(0) as usize).collect(),
        };
        if dur.len() != seq {
            return Err(format!("{} durations for {seq} phonemes", dur.len()));
        }
        let frames: usize = dur.iter().sum();
        if frames == 0 {
            return Err("the predicted duration is zero".into());
        }
        let en = length_regulate(&prosody, 640, seq, &dur, frames);
        let asr = length_regulate(&txt, 512, seq, &dur, frames);
        let dec = run(
            "decoder",
            vec![
                Tensor::f32("/encoder/MatMul_output_0", vec![1, 640, frames], &en),
                Tensor::f32("/encoder/MatMul_1_output_0", vec![1, 512, frames], &asr),
                Tensor::f32("style", vec![1, STYLE_DIM], &style),
            ],
        )?;
        let raw: Vec<f32> = dec
            .first()
            .ok_or("the decoder gave no output")?
            .to_f32()
            .into_iter()
            .map(|v| if v.is_finite() { v } else { 0.0 })
            .collect();
        let mut wav = istft_normalize(&raw, &self.window_sum);
        soft_peak_limit(&mut wav, 0.95);
        Ok(wav)
    }
}

/// Each column of `x` (`[1, c, seq]`) repeated `dur[i]` times → `[1, c, frames]`.
fn length_regulate(x: &[f32], c: usize, seq: usize, dur: &[usize], frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * frames];
    for ch in 0..c {
        let src = &x[ch * seq..ch * seq + seq];
        let dst = &mut out[ch * frames..ch * frames + frames];
        let mut f = 0;
        for (i, &d) in dur.iter().enumerate() {
            for _ in 0..d {
                dst[f] = src[i];
                f += 1;
            }
        }
    }
    out
}

/// The edges divided by the window's overlap, scaled by n_fft/hop, cropped.
fn istft_normalize(raw: &[f32], window_sum: &[f32]) -> Vec<f32> {
    let l = raw.len();
    let mut scat = raw.to_vec();
    for (j, &w) in window_sum.iter().enumerate().take(l) {
        if w > 1.18e-38 {
            scat[j] = raw[j] / w;
        }
    }
    let end = l.saturating_sub(ISTFT_CROP);
    scat[ISTFT_CROP.min(l)..end.max(ISTFT_CROP.min(l))]
        .iter()
        .map(|&s| s * OVERLAP_FACTOR)
        .collect()
}

fn soft_peak_limit(wav: &mut [f32], ceiling: f32) {
    let peak = wav.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    if peak > ceiling && peak.is_finite() && peak > 0.0 {
        let s = ceiling / peak;
        wav.iter_mut().for_each(|x| *x *= s);
    }
}

impl Program for Kokoro {
    fn init(config: Value) -> Result<Self, String> {
        let s = |k: &str| {
            config[k]
                .as_str()
                .map(String::from)
                .ok_or(format!("config has no {k}"))
        };
        let tok: Value = serde_json::from_slice(&file(&s("vocab")?)?)
            .map_err(|e| format!("tokenizer.json: {e}"))?;
        let vocab = tok["model"]["vocab"]
            .as_object()
            .ok_or("tokenizer.json has no model.vocab")?
            .iter()
            .filter_map(|(k, v)| {
                let mut c = k.chars();
                let ch = c.next()?;
                c.next().is_none().then_some((ch, v.as_i64()?))
            })
            .collect();
        let voices = config["voices"]
            .as_object()
            .ok_or("config has no voices")?
            .iter()
            .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string())))
            .collect();
        Ok(Kokoro {
            vocab,
            voices,
            default_voice: s("default_voice")?,
            window_sum: f32s(&file(&s("window_sum")?)?),
            espeak_dir: s("espeak")?,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "synthesize" {
            return Err(format!("Kokoro answers synthesize, not {task}"));
        }
        let text = req.body["text"].as_str().ok_or("synthesize takes text")?;
        let o = &req.body["options"];
        let voice = o["voice"]
            .as_str()
            .unwrap_or(&self.default_voice)
            .to_string();
        let speed = o["speed"].as_f64().unwrap_or(1.0) as f32;
        let wav = self.speak(text, &voice, speed)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": SAMPLE_RATE}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Kokoro);
