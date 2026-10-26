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

//! MioTTS's host logic as an RLX program, answering `synthesize`: what
//! `MioSession::synthesize` does — the chat prompt (the host's tokenizer),
//! the language model's ids (the package's `generate` entry, sampled as the
//! runner samples: temperature 0.8, a stream per step), the speech codes
//! among them, the preset voice — and the codec (`decoder_body`, its ISTFT)
//! in rlx.

use rlx_program::{Message, Program, Tensor, host, run};
use serde_json::{Value, json};

const TOKENIZER: &str = "sidecar:tokenizer.json";

struct Mio {
    speech_base: u32,
    codebook: u32,
    eos: u32,
    speech_len: usize,
    max_new_tokens: usize,
    temperature: f64,
    seed: u64,
    preset: String,
    sample_rate: u32,
}

fn file(name: &str) -> Result<Vec<u8>, String> {
    let m = host("file", Message::body(json!({"name": name})))?;
    Ok(m.tensor("bytes").ok_or("no bytes")?.data.clone())
}

impl Mio {
    /// The speech codes the language model speaks for `text`.
    fn codes(&self, text: &str, seed: u64, max_new: usize) -> Result<Vec<i64>, String> {
        let prompt = format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n");
        let m = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": TOKENIZER, "text": prompt})),
        )?;
        let ids: Vec<i32> = m.body["ids"]
            .as_array()
            .ok_or("no ids")?
            .iter()
            .filter_map(|v| v.as_i64().map(|v| v as i32))
            .collect();
        let n = ids.len();
        let out = host(
            "entry.run",
            Message::new(
                json!({"entry": "generate", "options": {
                    "max_tokens": max_new,
                    "temperature": self.temperature,
                    "top_p": 1.0,
                    "seed": seed,
                    "seed_per_step": true,
                    "stop_ids": [self.eos],
                }}),
                vec![Tensor::i32("input_ids", vec![1, n], &ids)],
            ),
        )?;
        let generated = out
            .tensors
            .first()
            .ok_or("the language model gave no ids")?
            .to_i64();
        let mut codes: Vec<i64> = Vec::new();
        for id in generated {
            let id = id as u32;
            if id == self.eos {
                break;
            }
            if (self.speech_base..self.speech_base + self.codebook).contains(&id) {
                codes.push((id - self.speech_base) as i64);
            }
        }
        Ok(codes)
    }

    /// `codes` through the codec in the voice `preset`.
    fn decode(&self, codes: &[i64], preset: &str) -> Result<Vec<f32>, String> {
        let mut codes = codes.to_vec();
        codes.resize(self.speech_len, 0);
        let emb: Vec<f32> = file(&format!("sidecar:presets/{preset}.f32"))
            .map_err(|_| format!("no preset voice {preset}"))?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let body = run(
            "decoder_body",
            vec![
                Tensor::new(
                    "content_token_indices",
                    "i64",
                    vec![1, self.speech_len],
                    codes.iter().flat_map(|c| c.to_le_bytes()).collect(),
                ),
                Tensor::f32("global_embedding", vec![1, emb.len()], &emb),
            ],
        )?;
        if body.len() < 2 {
            return Err("the codec gave no magnitude and phase".into());
        }
        let (mut mag, mut phase) = (body[0].clone(), body[1].clone());
        mag.name = "mag".into();
        phase.name = "phase".into();
        let audio = run("istft", vec![mag, phase])?;
        Ok(audio.first().ok_or("the ISTFT gave no audio")?.to_f32())
    }
}

impl Program for Mio {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        Ok(Mio {
            speech_base: u("speech_base")? as u32,
            codebook: u("codebook")? as u32,
            eos: u("eos")? as u32,
            speech_len: u("speech_len")? as usize,
            max_new_tokens: u("max_new_tokens")? as usize,
            temperature: config["temperature"]
                .as_f64()
                .ok_or("config has no temperature")?,
            seed: u("seed")?,
            preset: config["preset"]
                .as_str()
                .ok_or("config has no preset")?
                .into(),
            sample_rate: u("sample_rate")? as u32,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "synthesize" {
            return Err(format!("MioTTS answers synthesize, not {task}"));
        }
        let text = req.body["text"].as_str().ok_or("synthesize takes text")?;
        let o = &req.body["options"];
        let preset = o["voice"].as_str().unwrap_or(&self.preset);
        if preset.contains(['/', '\\']) || preset.starts_with('.') {
            return Err(format!("no preset voice {preset}"));
        }
        let seed = o["seed"].as_u64().unwrap_or(self.seed);
        let max_new = o["max_new_tokens"]
            .as_u64()
            .map_or(self.max_new_tokens, |v| v as usize);
        // `options.codes`: speech codes to decode, the language model skipped.
        let codes = match o["codes"].as_array() {
            Some(c) => c.iter().filter_map(Value::as_i64).collect(),
            None => self.codes(text, seed, max_new)?,
        };
        let wav = self.decode(&codes, preset)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Mio);
