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

//! MiraTTS's host logic as an RLX program, answering `synthesize`: what
//! `MiraTts::synthesize` does — the voice's 32 global tokens
//! (`options.context`, or a `reference` clip at 16 kHz: volume-normalized,
//! tiled to 6 s, `s_mel`, `s_encoder`), the prompt around them and the
//! text (the host's tokenizer), the Qwen2 LM a token at a time (the
//! package's `generate` entry; `penalty^count` over the ids it generated,
//! then the host's `sample`, each token from the stream `seed + step`), its
//! acoustic codes, BiCodec's `detokenizer`. `run` answers the global
//! tokens of a `reference`, or the acoustic codes of `options.text`.

use std::collections::HashMap;

use rlx_program::{Message, Program, Tensor, host, run};
use serde_json::{Value, json};

const TOKENIZER: &str = "sidecar:tokenizer.json";
/// FastBiCodec's reference volume.
const VOL_COEFF: f32 = 0.2;

struct Mira {
    task_tts: u32,
    start_text: u32,
    end_text: u32,
    context_audio_start: u32,
    context_audio_end: u32,
    prompt_speech_start: u32,
    context_base: u32,
    speech_base: u32,
    speech_codebook: u32,
    stops: Vec<u32>,
    speech_len: usize,
    context_len: usize,
    ref_samples: usize,
    max_new_tokens: usize,
    temperature: f64,
    top_k: u64,
    top_p: f64,
    min_p: f64,
    repetition_penalty: f32,
    sample_rate: u32,
}

fn ints(v: &Value) -> Vec<u32> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|x| x as u32))
                .collect()
        })
        .unwrap_or_default()
}

/// Pad with 0 / truncate to exactly `n`.
fn fit(codes: &[u32], n: usize) -> Vec<u32> {
    let mut v = codes.to_vec();
    v.resize(n, 0);
    v
}

impl Mira {
    /// A 16 kHz reference → its 32 global tokens.
    fn globals(&self, pcm: &[f32]) -> Result<Vec<u32>, String> {
        let mut x = if pcm.is_empty() {
            vec![0.0]
        } else {
            pcm.to_vec()
        };
        let peak = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-8);
        let scale = VOL_COEFF / peak;
        for v in &mut x {
            *v *= scale;
        }
        if x.len() < self.ref_samples {
            let n = x.len();
            while x.len() < self.ref_samples {
                let take = (self.ref_samples - x.len()).min(n);
                let chunk = x[..take].to_vec();
                x.extend_from_slice(&chunk);
            }
        }
        x.truncate(self.ref_samples);
        let mel = run(
            "s_mel",
            vec![Tensor::f32("pcm", vec![self.ref_samples], &x)],
        )?;
        let mut mel = mel.into_iter().next().ok_or("s_mel gave nothing")?;
        mel.name = "mel_spectrogram".into();
        let out = run("s_encoder", vec![mel])?;
        let t = out.first().ok_or("s_encoder gave nothing")?;
        let tokens: Vec<u32> = match t.dtype.as_str() {
            "i64" => t.to_i64().into_iter().map(|v| v as u32).collect(),
            _ => t.to_f32().into_iter().map(|v| v as u32).collect(),
        };
        Ok(fit(&tokens, self.context_len))
    }

    fn prompt(&self, text: &str, context: &[u32]) -> Result<Vec<u32>, String> {
        let m = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": TOKENIZER, "text": text, "special": false})),
        )?;
        let mut p = vec![self.task_tts, self.start_text];
        p.extend(ints(&m.body["ids"]));
        p.push(self.end_text);
        p.push(self.context_audio_start);
        p.extend(context.iter().map(|&c| self.context_base + c));
        p.push(self.context_audio_end);
        p.push(self.prompt_speech_start);
        Ok(p)
    }

    /// `MiraLm::generate_speech_codes`.
    fn codes(&self, prompt: &[u32], seed: u64, max_new: usize) -> Result<Vec<u32>, String> {
        let mut context = prompt.to_vec();
        let mut counts: HashMap<u32, i32> = HashMap::new();
        let mut codes = Vec::new();
        for step in 0..max_new {
            let ids: Vec<i32> = context.iter().map(|&t| t as i32).collect();
            let out = host(
                "entry.run",
                Message::new(
                    json!({"entry": "generate"}),
                    vec![Tensor::i32("input_ids", vec![1, ids.len()], &ids)],
                ),
            )?;
            let mut logits = out.tensors.first().ok_or("the LM gave no logits")?.to_f32();
            for (&t, &n) in &counts {
                if let Some(l) = logits.get_mut(t as usize) {
                    let f = self.repetition_penalty.powi(n);
                    *l = if *l > 0.0 { *l / f } else { *l * f };
                }
            }
            let n = logits.len();
            let opts = json!({"seed": seed, "step": step, "temperature": self.temperature,
                              "top_k": self.top_k, "top_p": self.top_p, "min_p": self.min_p});
            let next = host(
                "sample",
                Message::new(opts, vec![Tensor::f32("logits", vec![n], &logits)]),
            )?
            .body["id"]
                .as_u64()
                .ok_or("the host sampled no id")? as u32;
            if self.stops.contains(&next) {
                break;
            }
            if (self.speech_base..self.speech_base + self.speech_codebook).contains(&next) {
                codes.push(next - self.speech_base);
            }
            *counts.entry(next).or_insert(0) += 1;
            context.push(next);
        }
        Ok(codes)
    }

    /// Acoustic codes and global tokens → 16 kHz, `detokenizer`.
    fn decode(&self, codes: &[u32], globals: &[u32]) -> Result<Vec<f32>, String> {
        let speech: Vec<i64> = fit(codes, self.speech_len)
            .into_iter()
            .map(i64::from)
            .collect();
        let context: Vec<i32> = fit(globals, self.context_len)
            .into_iter()
            .map(|c| c as i32)
            .collect();
        let out = run(
            "detokenizer",
            vec![
                Tensor::new(
                    "speech_tokens",
                    "i64",
                    vec![1, self.speech_len],
                    speech.iter().flat_map(|x| x.to_le_bytes()).collect(),
                ),
                Tensor::i32("context_tokens", vec![1, self.context_len], &context),
            ],
        )?;
        Ok(out.first().ok_or("the detokenizer gave nothing")?.to_f32())
    }

    /// The voice: `options.context`, else a `reference` tensor's (or
    /// `options.reference`'s) global tokens.
    fn voice(&self, req: &Message) -> Result<Vec<u32>, String> {
        let o = &req.body["options"];
        if o["context"].is_array() {
            return Ok(fit(&ints(&o["context"]), self.context_len));
        }
        let pcm: Vec<f32> = match req.tensor("reference") {
            Some(t) => t.to_f32(),
            None => o["reference"]
                .as_array()
                .ok_or("a voice: options.context, or a 16 kHz reference")?
                .iter()
                .filter_map(|v| v.as_f64().map(|v| v as f32))
                .collect(),
        };
        self.globals(&pcm)
    }
}

impl Program for Mira {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        let f = |k: &str| config[k].as_f64().ok_or(format!("config has no {k}"));
        Ok(Mira {
            task_tts: u("task_tts")? as u32,
            start_text: u("start_text")? as u32,
            end_text: u("end_text")? as u32,
            context_audio_start: u("context_audio_start")? as u32,
            context_audio_end: u("context_audio_end")? as u32,
            prompt_speech_start: u("prompt_speech_start")? as u32,
            context_base: u("context_base")? as u32,
            speech_base: u("speech_base")? as u32,
            speech_codebook: u("speech_codebook")? as u32,
            stops: ints(&config["stops"]),
            speech_len: u("speech_len")? as usize,
            context_len: u("context_len")? as usize,
            ref_samples: u("ref_samples")? as usize,
            max_new_tokens: u("max_new_tokens")? as usize,
            temperature: f("temperature")?,
            top_k: u("top_k")?,
            top_p: f("top_p")?,
            min_p: f("min_p")?,
            repetition_penalty: f("repetition_penalty")? as f32,
            sample_rate: u("sample_rate")? as u32,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        let o = req.body["options"].clone();
        let seed = o["seed"].as_u64().unwrap_or(0);
        let max_new = o["max_new_tokens"]
            .as_u64()
            .map_or(self.max_new_tokens, |v| v as usize)
            .min(self.speech_len);
        match task {
            // `run`: a reference's global tokens, or `options.text`'s codes.
            "run" if !o["text"].is_string() => {
                let g = self.voice(&req)?;
                let g: Vec<i32> = g.iter().map(|&x| x as i32).collect();
                let n = g.len();
                Ok(Message::tensors(vec![Tensor::i32("context", vec![n], &g)]))
            }
            "run" | "synthesize" => {
                let text = if task == "run" {
                    o["text"].as_str().unwrap_or("")
                } else {
                    req.body["text"].as_str().ok_or("synthesize takes text")?
                };
                let globals = self.voice(&req)?;
                // `options.codes`: acoustic codes to decode, the LM skipped.
                let codes = match o["codes"].as_array() {
                    Some(_) => ints(&o["codes"]),
                    None => self.codes(&self.prompt(text, &globals)?, seed, max_new)?,
                };
                if task == "run" {
                    let c: Vec<i32> = codes.iter().map(|&x| x as i32).collect();
                    let n = c.len();
                    return Ok(Message::tensors(vec![Tensor::i32("codes", vec![n], &c)]));
                }
                if codes.is_empty() {
                    return Err(format!("the LM produced no acoustic codes for {text:?}"));
                }
                let wav = self.decode(&codes, &globals)?;
                let n = wav.len();
                Ok(Message::new(
                    json!({"sample_rate": self.sample_rate}),
                    vec![Tensor::f32("audio", vec![n], &wav)],
                ))
            }
            _ => Err(format!("MiraTTS answers synthesize and run, not {task}")),
        }
    }
}

rlx_program::program!(Mira);
