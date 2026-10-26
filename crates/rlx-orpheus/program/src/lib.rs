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

//! Orpheus's host logic as an RLX program, answering `synthesize`: what
//! `OrpheusTts::synthesize` does — the prompt (the host's tokenizer, the
//! start and end tokens around it), the language model a token at a time
//! (the package's `generate` entry; its logits penalized by how often each
//! id came before, then sampled by the host from the runner's stream per
//! step), the seven SNAC codes of each frame among its ids — and SNAC
//! (`snac_latent`, `snac_decode`) in rlx. `run` answers the codes alone.

use std::collections::HashMap;

use rlx_program::{Message, Program, Tensor, host, run};
use serde_json::{Value, json};

const TOKENIZER: &str = "sidecar:tokenizer.json";

struct Orpheus {
    template: String,
    voice: Option<String>,
    start: u32,
    bos: u32,
    end: Vec<u32>,
    stop: u32,
    custom_base: u32,
    snac_offset: u32,
    max_new_tokens: usize,
    temperature: f64,
    top_p: f64,
    top_k: u64,
    repetition_penalty: f32,
    seed: u64,
    greedy: bool,
    sample_rate: u32,
}

impl Orpheus {
    /// The prompt's ids: the start token, BOS, the body, the end tokens.
    fn prompt(&self, text: &str, voice: Option<&str>) -> Result<Vec<u32>, String> {
        let body = match voice {
            Some(v) => self.template.replace("{voice}", v).replace("{text}", text),
            None => text.to_string(),
        };
        let m = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": TOKENIZER, "text": body})),
        )?;
        let ids: Vec<u32> = m.body["ids"]
            .as_array()
            .ok_or("no ids")?
            .iter()
            .filter_map(|v| v.as_u64().map(|v| v as u32))
            .collect();
        let mut out = vec![self.start];
        if ids.first() != Some(&self.bos) {
            out.push(self.bos);
        }
        out.extend(ids);
        out.extend(&self.end);
        Ok(out)
    }

    /// The SNAC codes the language model speaks after `prompt`: each id in
    /// its frame slot's range (`snac_offset + slot · 4096 + code`) a code,
    /// the slot moving on only past a nonzero one — as the runner reads its
    /// stream.
    fn codes(&self, prompt: &[u32], seed: u64, max_new: usize) -> Result<Vec<i32>, String> {
        let mut context = prompt.to_vec();
        let mut counts: HashMap<u32, i32> = HashMap::new();
        let mut codes = Vec::new();
        let mut slot = 0usize;
        for step in 0..max_new {
            let ids: Vec<i32> = context.iter().map(|&t| t as i32).collect();
            let out = host(
                "entry.run",
                Message::new(
                    json!({"entry": "generate"}),
                    vec![Tensor::i32("input_ids", vec![1, ids.len()], &ids)],
                ),
            )?;
            let mut logits = out
                .tensors
                .first()
                .ok_or("the language model gave no logits")?
                .to_f32();
            // The runner's penalty: `penalty^count` for each id generated.
            if self.repetition_penalty > 1.0 {
                for (&t, &n) in &counts {
                    if let Some(l) = logits.get_mut(t as usize) {
                        let f = self.repetition_penalty.powi(n);
                        *l = if *l > 0.0 { *l / f } else { *l * f };
                    }
                }
            }
            let mut opts = json!({"seed": seed, "step": step});
            if !self.greedy {
                opts["temperature"] = json!(self.temperature);
                opts["top_p"] = json!(self.top_p);
                if self.top_k > 0 {
                    opts["top_k"] = json!(self.top_k);
                }
            }
            let n = logits.len();
            let next = host(
                "sample",
                Message::new(opts, vec![Tensor::f32("logits", vec![n], &logits)]),
            )?
            .body["id"]
                .as_u64()
                .ok_or("the host sampled no id")? as u32;
            if next == self.stop {
                break;
            }
            if next >= self.custom_base {
                let lo = self.snac_offset + (slot % 7) as u32 * 4096;
                if (lo..lo + 4096).contains(&next) {
                    let code = (next - lo) as i32;
                    if code > 0 {
                        codes.push(code);
                        slot += 1;
                    }
                }
            }
            *counts.entry(next).or_insert(0) += 1;
            context.push(next);
        }
        Ok(codes)
    }

    /// Whole frames of `codes` through SNAC: each frame's seven codes to
    /// its three levels (1, 2 and 4 of them), then the lookup and the
    /// decoder; a quiet result raised toward full scale, as the runner's.
    fn decode(&self, codes: &[i32]) -> Result<Vec<f32>, String> {
        let frames = codes.len() / 7;
        if frames == 0 {
            return Ok(Vec::new());
        }
        let (mut c0, mut c1, mut c2) = (Vec::new(), Vec::new(), Vec::new());
        for f in codes[..frames * 7].chunks_exact(7) {
            c0.push(f[0]);
            c1.extend([f[1], f[4]]);
            c2.extend([f[2], f[3], f[5], f[6]]);
        }
        if !c0
            .iter()
            .chain(&c1)
            .chain(&c2)
            .all(|c| (0..4096).contains(c))
        {
            return Err("a SNAC code outside 0..4096".into());
        }
        let latent = run(
            "snac_latent",
            vec![
                Tensor::i32("codes_0", vec![frames], &c0),
                Tensor::i32("codes_1", vec![2 * frames], &c1),
                Tensor::i32("codes_2", vec![4 * frames], &c2),
            ],
        )?;
        let mut z = latent.into_iter().next().ok_or("SNAC gave no latent")?;
        z.name = "z".into();
        let audio = run("snac_decode", vec![z])?;
        let mut wav = audio.first().ok_or("SNAC gave no audio")?.to_f32();
        let peak = wav.iter().fold(0f32, |m, s| m.max(s.abs()));
        if peak > 1e-6 && peak < 0.25 {
            let gain = 0.95 / peak;
            for s in &mut wav {
                *s = (*s * gain).clamp(-1.0, 1.0);
            }
        }
        Ok(wav)
    }
}

impl Program for Orpheus {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        let f = |k: &str| config[k].as_f64().ok_or(format!("config has no {k}"));
        Ok(Orpheus {
            template: config["template"]
                .as_str()
                .ok_or("config has no template")?
                .into(),
            voice: config["voice"].as_str().map(String::from),
            start: u("start")? as u32,
            bos: u("bos")? as u32,
            end: config["end"]
                .as_array()
                .ok_or("config has no end")?
                .iter()
                .filter_map(|v| v.as_u64().map(|v| v as u32))
                .collect(),
            stop: u("stop")? as u32,
            custom_base: u("custom_base")? as u32,
            snac_offset: u("snac_offset")? as u32,
            max_new_tokens: u("max_new_tokens")? as usize,
            temperature: f("temperature")?,
            top_p: f("top_p")?,
            top_k: u("top_k")?,
            repetition_penalty: f("repetition_penalty")? as f32,
            seed: u("seed")?,
            greedy: config["greedy"].as_bool().unwrap_or(false),
            sample_rate: u("sample_rate")? as u32,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        let o = &req.body["options"];
        // `run`: the SNAC codes for `options.text`, as `[n]` i32, no audio.
        let text = match task {
            "synthesize" => req.body["text"].as_str().ok_or("synthesize takes text")?,
            "run" => o["text"].as_str().ok_or("run takes options.text")?,
            _ => return Err(format!("Orpheus answers synthesize and run, not {task}")),
        };
        let voice = o["voice"].as_str().or(self.voice.as_deref());
        let seed = o["seed"].as_u64().unwrap_or(self.seed);
        let max_new = o["max_new_tokens"]
            .as_u64()
            .map_or(self.max_new_tokens, |v| v as usize);
        // `options.codes`: SNAC codes to decode, the language model skipped.
        let codes: Vec<i32> = match o["codes"].as_array() {
            Some(c) => c
                .iter()
                .filter_map(|v| v.as_i64().map(|v| v as i32))
                .collect(),
            None => {
                let prompt = self.prompt(text, voice)?;
                self.codes(&prompt, seed, max_new)?
            }
        };
        if task == "run" {
            let n = codes.len();
            return Ok(Message::tensors(vec![Tensor::i32(
                "codes",
                vec![n],
                &codes,
            )]));
        }
        let wav = self.decode(&codes)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Orpheus);
