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

//! Parler-TTS's host logic as an RLX program, answering `synthesize`: what
//! `NativeParler::synthesize` does around its graphs — the T5 ids (the
//! host's tokenizer), the delayed-codebook sampling loop (its seeded
//! sampler, the runner's), the un-delay — the description through the
//! encoder, each step through the decoder (its logits kept host-side, the
//! step's row picked out), the codes through DAC, all in rlx.
//!
//! `options.description` is the voice; `options.greedy`, `temperature`,
//! `top_k`, `seed` the sampling.

use rlx_program::{Message, Program, Tensor, host, run, run_with};
use serde_json::{Value, json};

const TOKENIZER: &str = "sidecar:tokenizer.json";
const EOS: i64 = 1;

struct Parler {
    k: usize,
    bos: i64,
    pad: i64,
    vocab: usize,
    dac_max: i64,
    steps: usize,
    sample_rate: u32,
    description: String,
    temperature: f32,
    top_k: usize,
    seed: u64,
}

/// FNV-1a 64 (the runner's seed mix).
fn hash64(s: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// The runner's splitmix64.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E3779B97F4A7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

/// Temperature and top-k sampling over logits (the runner's).
fn sample(row: &[f32], temperature: f32, top_k: usize, rng: &mut Rng) -> usize {
    let temp = temperature.max(1e-4);
    let mut idx: Vec<usize> = (0..row.len()).collect();
    let k = top_k.clamp(1, row.len());
    idx.select_nth_unstable_by(k - 1, |&a, &b| row[b].partial_cmp(&row[a]).unwrap());
    idx.truncate(k);
    let maxl = idx
        .iter()
        .map(|&i| row[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = idx
        .iter()
        .map(|&i| ((row[i] - maxl) / temp).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }
    let r = rng.next_f32();
    let mut acc = 0.0;
    for (j, &p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return idx[j];
        }
    }
    idx[k - 1]
}

fn i64s(name: &str, shape: Vec<usize>, v: &[i64]) -> Tensor {
    Tensor::new(
        name,
        "i64",
        shape,
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
}

/// The T5 ids of `text`, ending in `</s>`.
fn tokenize(text: &str) -> Result<Vec<i64>, String> {
    let m = host(
        "tokenizer.encode",
        Message::body(json!({"tokenizer": TOKENIZER, "text": text})),
    )?;
    let mut ids: Vec<i64> = m.body["ids"]
        .as_array()
        .ok_or("no ids")?
        .iter()
        .filter_map(Value::as_i64)
        .collect();
    if ids.last() != Some(&EOS) {
        ids.push(EOS);
    }
    Ok(ids)
}

impl Parler {
    fn speak(
        &self,
        text: &str,
        description: &str,
        greedy: bool,
        temperature: f32,
        top_k: usize,
        seed: u64,
    ) -> Result<Vec<f32>, String> {
        let (k, t) = (self.k, self.steps);
        let ids = tokenize(description)?;
        let n = ids.len();
        let enc = run(
            "text_encoder",
            vec![
                i64s("input_ids", vec![1, n], &ids),
                i64s("attention_mask", vec![1, n], &vec![1; n]),
            ],
        )?;
        let mut hs = enc.into_iter().next().ok_or("the encoder gave no output")?;
        hs.name = "encoder_hidden_states".into();
        let prompt = tokenize(text.trim())?;
        let pt = prompt.len();

        let mut dids = vec![self.pad; k * t];
        for c in 0..k {
            dids[c * t] = self.bos;
        }
        let mut rng = Rng::new(seed ^ hash64(description));
        let mut steps: Vec<Vec<i64>> = Vec::with_capacity(t);
        let mut seq = 0usize;
        for step in 0..t {
            let inputs = vec![
                i64s("decoder_input_ids", vec![1, k, t], &dids),
                hs.clone(),
                i64s("encoder_attention_mask", vec![1, n], &vec![1; n]),
                i64s("prompt_input_ids", vec![1, pt], &prompt),
            ];
            // The logits stay host-side; only the step's row comes back.
            if step == 0 {
                let out = run_with("decoder", inputs, &[], &[])?;
                let logits = out.first().ok_or("the decoder gave no output")?;
                seq = logits.data.len() / 4 / (k * self.vocab);
                host(
                    "slot.set",
                    Message::new(
                        json!({"slot": "logits"}),
                        vec![{
                            let mut l = logits.clone();
                            l.name = "logits".into();
                            l
                        }],
                    ),
                )?;
            } else {
                run_with("decoder", inputs, &[], &[("logits", "logits")])?;
            }
            if seq < t {
                return Err(format!("the decoder's sequence {seq} is shorter than {t}"));
            }
            let at = (seq - t) + step;
            let row = run_with(
                "pick",
                vec![i64s("position", vec![1], &[at as i64])],
                &[("logits", "logits")],
                &[],
            )?;
            let rows = row.first().ok_or("pick gave no output")?.to_f32();
            let mut nxt = vec![self.pad; k];
            for c in 0..k {
                if step < c {
                    nxt[c] = self.bos;
                    continue;
                }
                let r = &rows[c * self.vocab..(c + 1) * self.vocab];
                nxt[c] = if greedy {
                    argmax(r)
                } else {
                    sample(r, temperature, top_k, &mut rng)
                } as i64;
            }
            let done = step >= k && nxt.iter().all(|&c| c == self.pad);
            steps.push(nxt.clone());
            if done {
                break;
            }
            if step + 1 < t {
                for c in 0..k {
                    dids[c * t + step + 1] = nxt[c];
                }
            }
        }

        // Un-delay, and keep the frames where every codebook is a DAC code.
        let n_steps = steps.len();
        let rows: Vec<Vec<i64>> = (0..k)
            .map(|c| {
                (0..n_steps.saturating_sub(c))
                    .map(|p| steps[p + c][c])
                    .collect()
            })
            .collect();
        let min_len = rows.iter().map(Vec::len).min().unwrap_or(0);
        let valid = (0..min_len)
            .take_while(|&f| rows.iter().all(|r| r[f] < self.dac_max))
            .count();
        if valid == 0 {
            return Err(format!(
                "no valid DAC frames (all codes ≥ {})",
                self.dac_max
            ));
        }
        let codes: Vec<i32> = rows
            .iter()
            .flat_map(|r| r[..valid].iter().map(|&c| c as i32))
            .collect();
        let z = run("lookup", vec![Tensor::i32("codes", vec![k, valid], &codes)])?;
        let mut z = z
            .into_iter()
            .next()
            .ok_or("the DAC lookup gave no output")?;
        z.name = "z".into();
        let audio = run("decode", vec![z])?;
        Ok(audio.first().ok_or("DAC gave no audio")?.to_f32())
    }
}

impl Program for Parler {
    fn init(config: Value) -> Result<Self, String> {
        let u = |key: &str| config[key].as_u64().ok_or(format!("config has no {key}"));
        Ok(Parler {
            k: u("codebooks")? as usize,
            bos: u("bos")? as i64,
            pad: u("pad")? as i64,
            vocab: u("vocab")? as usize,
            dac_max: u("dac_max")? as i64,
            steps: u("max_steps")? as usize,
            sample_rate: u("sample_rate")? as u32,
            description: config["description"]
                .as_str()
                .ok_or("config has no description")?
                .into(),
            temperature: config["temperature"].as_f64().unwrap_or(1.0) as f32,
            top_k: u("top_k")? as usize,
            seed: u("seed")?,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "synthesize" {
            return Err(format!("Parler-TTS answers synthesize, not {task}"));
        }
        let text = req.body["text"].as_str().ok_or("synthesize takes text")?;
        if text.trim().is_empty() {
            return Err("empty transcript".into());
        }
        let o = &req.body["options"];
        let description = o["description"]
            .as_str()
            .unwrap_or(&self.description)
            .to_string();
        let greedy = o["greedy"].as_bool().unwrap_or(false);
        let temperature = o["temperature"]
            .as_f64()
            .map_or(self.temperature, |v| v as f32);
        let top_k = o["top_k"].as_u64().map_or(self.top_k, |v| v as usize);
        let seed = o["seed"].as_u64().unwrap_or(self.seed);
        let wav = self.speak(text, &description, greedy, temperature, top_k, seed)?;
        let _ = host("slot.drop", Message::body(json!({"slots": ["logits"]})));
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Parler);
