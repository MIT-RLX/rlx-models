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

//! A Piper voice's host logic as an RLX program, answering `synthesize`:
//! what `NativeVits::synthesize` does around its graphs — espeak-ng (its
//! data read from the package), the `phoneme_id_map` tokenizer, the flow
//! noise (the runner's seeded generator) and the length regulator. The
//! encoder, the duration predictor, the prior and the decoder run in rlx
//! (`rlx_program::run`).

use std::collections::HashMap;

use rlx_program::{Message, Program, Tensor, run};
use serde_json::{Value, json};

const CHANNELS: usize = 192;

struct Piper {
    ids: HashMap<String, Vec<i64>>,
    espeak_voice: String,
    espeak_dir: String,
    sample_rate: u32,
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
    /// The most ids an `encoder_<n>` of its own takes.
    short_encoders: usize,
}

impl Piper {
    /// `text` phonemized by espeak-ng in the voice's language (else
    /// `en-us`, `en`), punctuation kept where it was.
    fn phonemize(&self, text: &str) -> Result<String, String> {
        let mut langs = vec![self.espeak_voice.as_str()];
        for f in ["en-us", "en"] {
            if !langs.contains(&f) {
                langs.push(f);
            }
        }
        let mut last = String::from("no espeak language");
        for lang in langs {
            match self.phonemize_lang(lang, text) {
                Ok(ipa) => return Ok(ipa),
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    fn phonemize_lang(&self, lang: &str, text: &str) -> Result<String, String> {
        if text.is_empty() {
            return Ok(String::new());
        }
        let engine =
            espeak_ng::EspeakNg::with_data_dir(lang, std::path::Path::new(&self.espeak_dir))
                .map_err(|e| format!("espeak-ng init for '{lang}': {e}"))?;
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

    fn id_of(&self, sym: &str) -> Option<i64> {
        self.ids.get(sym).and_then(|v| v.first().copied())
    }

    /// Phonemes → ids: `^`, each mapped phoneme and a `_` pad, `$`.
    fn tokenize(&self, phonemes: &str) -> Vec<i64> {
        let pad = self.id_of("_").unwrap_or(0);
        let mut ids = Vec::new();
        ids.extend(self.id_of("^"));
        for c in phonemes.chars() {
            if let Some(mapped) = self.ids.get(&c.to_string()) {
                ids.extend_from_slice(mapped);
                ids.push(pad);
            }
        }
        ids.extend(self.id_of("$"));
        ids
    }

    fn speak(
        &self,
        ids: &[i64],
        length_scale: f32,
        deterministic: bool,
    ) -> Result<Vec<f32>, String> {
        let seq = ids.len();
        let (noise_w, noise_scale) = if deterministic {
            (0.0, 0.0)
        } else {
            (self.noise_w, self.noise_scale)
        };
        let idb: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        let encoder = if seq <= self.short_encoders {
            format!("encoder_{seq}")
        } else {
            "encoder".into()
        };
        let enc = run(
            &encoder,
            vec![
                Tensor::new("input", "i64", vec![1, seq], idb),
                Tensor::new(
                    "input_lengths",
                    "i64",
                    vec![1],
                    (seq as i64).to_le_bytes().to_vec(),
                ),
            ],
        )?;
        if enc.len() < 3 {
            return Err(format!("the encoder gave {} outputs", enc.len()));
        }
        let (m_p, logs_p, dp_in) = (enc[0].to_f32(), enc[1].to_f32(), enc[2].to_f32());
        if m_p.len() != CHANNELS * seq {
            return Err(format!("m_p holds {}, not 192·{seq}", m_p.len()));
        }

        let noise = gaussian(2 * seq, noise_w, 0x5eed_1234);
        let dur = run(
            "durations",
            vec![
                Tensor::f32("dp_in", vec![1, CHANNELS, seq], &dp_in),
                Tensor::f32("noise", vec![2, seq], &noise),
                Tensor::f32("length_scale", vec![1], &[length_scale]),
            ],
        )?;
        let dur: Vec<usize> = dur
            .first()
            .ok_or("the duration predictor gave no output")?
            .to_f32()
            .iter()
            .map(|d| *d as usize)
            .collect();
        let frames: usize = dur.iter().sum();
        if frames == 0 {
            return Err("the predicted duration is zero".into());
        }

        let m = length_regulate(&m_p, CHANNELS, seq, &dur, frames);
        let logs = length_regulate(&logs_p, CHANNELS, seq, &dur, frames);
        let noise = gaussian(CHANNELS * frames, 1.0, 0xabcd_ef01);
        let shape = vec![1, CHANNELS, frames];
        let z = run(
            "prior",
            vec![
                Tensor::f32("m", shape.clone(), &m),
                Tensor::f32("logs", shape.clone(), &logs),
                Tensor::f32("noise", shape.clone(), &noise),
                Tensor::f32("noise_scale", vec![1], &[noise_scale]),
            ],
        )?;
        let z = z.first().ok_or("the prior gave no output")?.to_f32();
        let audio = run(
            "decoder",
            vec![
                Tensor::f32("/Add_output_0", shape, &z),
                Tensor::f32("/Cast_2_output_0", vec![1, 1, frames], &vec![1.0; frames]),
            ],
        )?;
        let audio = audio.first().ok_or("the decoder gave no output")?.to_f32();
        let peak = audio
            .iter()
            .filter(|s| s.is_finite())
            .fold(0f32, |a, s| a.max(s.abs()));
        if peak < 1e-3 {
            return Err(format!("the audio is silent (peak {peak:.2e})"));
        }
        Ok(audio)
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

/// N(0, `scale`²) samples, the runner's: a seeded LCG and Box-Muller.
fn gaussian(n: usize, scale: f32, seed: u64) -> Vec<f32> {
    if scale == 0.0 {
        return vec![0.0f32; n];
    }
    let mut state = seed | 1;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) as f32
    };
    let mut out = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let u1 = next().max(1e-7);
        let u2 = next();
        let r = (-2.0 * u1.ln()).sqrt();
        out[i] = r * (std::f32::consts::TAU * u2).cos() * scale;
        if i + 1 < n {
            out[i + 1] = r * (std::f32::consts::TAU * u2).sin() * scale;
        }
        i += 2;
    }
    out
}

impl Program for Piper {
    fn init(config: Value) -> Result<Self, String> {
        let s = |k: &str| {
            config[k]
                .as_str()
                .map(String::from)
                .ok_or(format!("config has no {k}"))
        };
        let f = |k: &str| {
            config[k]
                .as_f64()
                .map(|v| v as f32)
                .ok_or(format!("config has no {k}"))
        };
        let ids = config["phoneme_id_map"]
            .as_object()
            .ok_or("config has no phoneme_id_map")?
            .iter()
            .map(|(k, v)| {
                let ids = v
                    .as_array()
                    .map(|a| a.iter().filter_map(Value::as_i64).collect());
                (k.clone(), ids.unwrap_or_default())
            })
            .collect();
        Ok(Piper {
            ids,
            espeak_voice: s("espeak_voice")?,
            espeak_dir: s("espeak")?,
            sample_rate: config["sample_rate"]
                .as_u64()
                .ok_or("config has no sample_rate")? as u32,
            noise_scale: f("noise_scale")?,
            length_scale: f("length_scale")?,
            noise_w: f("noise_w")?,
            short_encoders: config["short_encoders"].as_u64().unwrap_or(0) as usize,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "synthesize" {
            return Err(format!("Piper answers synthesize, not {task}"));
        }
        let text = req.body["text"].as_str().ok_or("synthesize takes text")?;
        let o = &req.body["options"];
        let length_scale = o["length_scale"]
            .as_f64()
            .map(|v| v as f32)
            .unwrap_or(self.length_scale);
        let deterministic = o["deterministic"].as_bool().unwrap_or(false);
        let ids = self.tokenize(&self.phonemize(text)?);
        if ids.len() <= 2 {
            return Err("the text has no phonemes".into());
        }
        let wav = self.speak(&ids, length_scale, deterministic)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Piper);
