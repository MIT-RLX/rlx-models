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

//! Soprano's host logic as an RLX program, answering `synthesize`: what
//! `NativeSoprano::synthesize` does — the prompt (`[STOP][TEXT]…[START]`,
//! the host's tokenizer), the backbone over the whole context each token
//! (`soprano_backbone`), sampled as the runner samples (temperature, the
//! repetition penalty over every id seen, top-k, top-p, its xorshift
//! stream), the last hidden state of each audio token, the decoder
//! (`soprano_decoder`) and the offline crop; text past the prompt limit
//! chunked at sentences, then commas, then words. `run` answers the audio
//! token ids.

use rlx_program::{Message, Program, Tensor, host, run};
use serde_json::{Value, json};

struct Soprano {
    tokenizer: String,
    eos: i64,
    vocab: usize,
    hidden: usize,
    token_size: usize,
    max_tokens: usize,
    max_prompt_tokens: usize,
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repetition_penalty: f32,
    seed: u64,
    greedy: bool,
    sample_rate: u32,
}

/// The runner's sampling options for one request.
#[derive(Clone, Copy)]
struct Opts {
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repetition_penalty: f32,
    seed: u64,
    greedy: bool,
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as f64 / u64::MAX as f64) as f32
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// The runner's sampler: temperature, the repetition penalty on every id
/// seen, the top `top_k`, the nucleus `top_p`, one draw.
fn sample_top_p(logits: &[f32], o: &Opts, seen: &[bool], rng: &mut Rng) -> usize {
    let temp = o.temperature.max(1e-5);
    let mut scores: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let mut s = s / temp;
            if seen[i] && o.repetition_penalty != 1.0 {
                s = if s < 0.0 {
                    s * o.repetition_penalty
                } else {
                    s / o.repetition_penalty
                };
            }
            (i, s)
        })
        .collect();
    scores.sort_unstable_by(|a, b| match (a.1.is_finite(), b.1.is_finite()) {
        (true, true) => b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal),
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => std::cmp::Ordering::Equal,
    });
    let k = if o.top_k == 0 {
        scores.len()
    } else {
        o.top_k.min(scores.len())
    };
    scores.truncate(k);
    if scores.is_empty() || !scores[0].1.is_finite() {
        return argmax(logits);
    }
    let mx = scores[0].1;
    let mut probs: Vec<(usize, f32)> = scores.iter().map(|&(i, s)| (i, (s - mx).exp())).collect();
    let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    for p in &mut probs {
        p.1 /= sum.max(1e-20);
    }
    let cutoff = o.top_p.clamp(0.0, 1.0);
    let mut cum = 0.0f32;
    let mut last = 0usize;
    for (ki, (_, p)) in probs.iter().enumerate() {
        cum += *p;
        last = ki;
        if cum >= cutoff {
            break;
        }
    }
    let target = rng.next_f32() * cum.min(1.0);
    let mut running = 0.0f32;
    for &(i, p) in probs.iter().take(last + 1) {
        running += p;
        if running >= target {
            return i;
        }
    }
    probs[last].0
}

/// `text` in chunks whose prompts stay within `max` tokens: sentences,
/// then commas, then words — the runner's `split_utterances`.
fn split_utterances(
    text: &str,
    max: usize,
    tokens: &dyn Fn(&str) -> Result<usize, String>,
) -> Result<Vec<String>, String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut buf = String::new();
    for ch in text.chars() {
        buf.push(ch);
        if matches!(ch, '.' | '?' | '!') {
            let t = buf.trim();
            if !t.is_empty() {
                sentences.push(t.to_string());
            }
            buf.clear();
        }
    }
    let t = buf.trim();
    if !t.is_empty() {
        sentences.push(t.to_string());
    }
    if sentences.is_empty() {
        sentences.push(text.trim().to_string());
    }
    let mut out: Vec<String> = Vec::new();
    for sent in sentences {
        if tokens(&sent)? <= max {
            out.push(sent);
            continue;
        }
        let clauses: Vec<&str> = sent
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        let mut cur = String::new();
        for clause in clauses {
            let trial = if cur.is_empty() {
                clause.to_string()
            } else {
                format!("{cur}, {clause}")
            };
            if !cur.is_empty() && tokens(&trial)? > max {
                out.push(std::mem::take(&mut cur));
                cur = clause.to_string();
            } else {
                cur = trial;
            }
            if tokens(&cur)? > max {
                let words: Vec<String> = cur.split_whitespace().map(str::to_string).collect();
                cur.clear();
                let mut pack = String::new();
                for w in words {
                    let trial = if pack.is_empty() {
                        w.clone()
                    } else {
                        format!("{pack} {w}")
                    };
                    if !pack.is_empty() && tokens(&trial)? > max {
                        out.push(std::mem::take(&mut pack));
                        pack = w;
                    } else {
                        pack = trial;
                    }
                }
                cur = pack;
            }
        }
        let t = cur.trim();
        if !t.is_empty() {
            out.push(t.to_string());
        }
    }
    Ok(out.into_iter().filter(|s| !s.trim().is_empty()).collect())
}

fn i64_tensor(name: &str, v: &[i64]) -> Tensor {
    let data = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    Tensor::new(name, "i64", vec![1, v.len()], data)
}

impl Soprano {
    /// The prompt's ids: `[STOP][TEXT]{text}[START]`, tokenized.
    fn prompt(&self, text: &str) -> Result<Vec<i64>, String> {
        let m = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": self.tokenizer,
                                 "text": format!("[STOP][TEXT]{}[START]", text.trim())})),
        )?;
        Ok(m.body["ids"]
            .as_array()
            .ok_or("no ids")?
            .iter()
            .filter_map(Value::as_i64)
            .collect())
    }

    /// One backbone pass over `ids`: the next id and the last hidden state.
    fn step(
        &self,
        ids: &[i64],
        o: &Opts,
        seen: &[bool],
        rng: &mut Rng,
    ) -> Result<(i64, Vec<f32>), String> {
        let n = ids.len();
        let positions: Vec<i64> = (0..n as i64).collect();
        let outs = run(
            "soprano_backbone",
            vec![
                i64_tensor("input_ids", ids),
                i64_tensor("attention_mask", &vec![1; n]),
                i64_tensor("position_ids", &positions),
            ],
        )?;
        let logits = outs.first().ok_or("the backbone gave no logits")?.to_f32();
        let hidden = outs
            .get(1)
            .ok_or("the backbone gave no hidden states")?
            .to_f32();
        let row = &logits[(n - 1) * self.vocab..n * self.vocab];
        if !row.iter().any(|x| x.is_finite()) {
            return Err(format!("backbone logits all non-finite at {n} tokens"));
        }
        let next = if o.greedy {
            argmax(row)
        } else {
            sample_top_p(row, o, seen, rng)
        } as i64;
        Ok((
            next,
            hidden[(n - 1) * self.hidden..n * self.hidden].to_vec(),
        ))
    }

    /// The runner's `generate_latents`: the audio token ids, and the hidden
    /// state of each but the first and the end.
    fn latents(&self, text: &str, o: &Opts) -> Result<(Vec<Vec<f32>>, Vec<i64>), String> {
        let prompt = self.prompt(text)?;
        if prompt.is_empty() {
            return Err("empty tokenization".into());
        }
        let mut seen = vec![false; self.vocab];
        let mark = |seen: &mut Vec<bool>, id: i64| {
            if (0..self.vocab as i64).contains(&id) {
                seen[id as usize] = true;
            }
        };
        for &id in &prompt {
            mark(&mut seen, id);
        }
        let mut rng = Rng::new(o.seed);
        let mut latents = Vec::new();
        let mut toks = Vec::new();
        let mut ids = prompt;
        let (next, _) = self.step(&ids, o, &seen, &mut rng)?;
        toks.push(next);
        if next == self.eos {
            return Ok((latents, toks));
        }
        mark(&mut seen, next);
        ids.push(next);
        for _ in 0..o.max_new_tokens {
            if ids.len() >= self.max_tokens {
                break;
            }
            let (tok, hidden) = self.step(&ids, o, &seen, &mut rng)?;
            toks.push(tok);
            if tok != self.eos {
                latents.push(hidden);
            }
            mark(&mut seen, tok);
            ids.push(tok);
            if tok == self.eos {
                break;
            }
        }
        Ok((latents, toks))
    }

    /// Latents `[t][hidden]` → PCM through the decoder, non-finite samples
    /// zero, the last `(t − 1) · 2048` kept.
    fn decode(&self, latents: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        let t = latents.len();
        let mut buf = vec![0f32; self.hidden * t];
        for (w, h) in latents.iter().enumerate() {
            for d in 0..self.hidden {
                buf[d * t + w] = h[d];
            }
        }
        let outs = run(
            "soprano_decoder",
            vec![Tensor::f32("hidden_states", vec![1, self.hidden, t], &buf)],
        )?;
        let mut pcm = outs.first().ok_or("the decoder gave no audio")?.to_f32();
        for s in &mut pcm {
            if !s.is_finite() {
                *s = 0.0;
            }
        }
        let keep = t.saturating_sub(1) * self.token_size;
        if keep > 0 && pcm.len() > keep {
            pcm = pcm[pcm.len() - keep..].to_vec();
        }
        Ok(pcm)
    }

    fn opts(&self, o: &Value) -> Opts {
        let f = |k: &str, d: f32| o[k].as_f64().map_or(d, |v| v as f32);
        let u = |k: &str, d: usize| o[k].as_u64().map_or(d, |v| v as usize);
        Opts {
            max_new_tokens: u("max_new_tokens", self.max_new_tokens),
            temperature: f("temperature", self.temperature),
            top_p: f("top_p", self.top_p),
            top_k: u("top_k", self.top_k),
            repetition_penalty: f("repetition_penalty", self.repetition_penalty),
            seed: o["seed"].as_u64().unwrap_or(self.seed),
            greedy: o["greedy"].as_bool().unwrap_or(self.greedy),
        }
    }

    fn synthesize(&self, text: &str, o: &Opts) -> Result<Vec<f32>, String> {
        let speak = |chunk: &str| -> Result<Vec<f32>, String> {
            let (latents, _) = self.latents(chunk, o)?;
            if latents.is_empty() {
                return Err(format!("no audio latents produced for {chunk:?}"));
            }
            self.decode(&latents)
        };
        if self.prompt(text)?.len() <= self.max_prompt_tokens {
            return speak(text);
        }
        let count = |s: &str| self.prompt(s).map(|ids| ids.len());
        let chunks = split_utterances(text, self.max_prompt_tokens, &count)?;
        let gap = self.sample_rate as usize / 20;
        let mut pcm = Vec::new();
        for chunk in &chunks {
            let part = speak(chunk)?;
            if !pcm.is_empty() {
                pcm.extend(std::iter::repeat_n(0.0f32, gap));
            }
            pcm.extend(part);
        }
        Ok(pcm)
    }
}

impl Program for Soprano {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        let f = |k: &str| {
            config[k]
                .as_f64()
                .map(|v| v as f32)
                .ok_or(format!("config has no {k}"))
        };
        Ok(Soprano {
            tokenizer: config["tokenizer"]
                .as_str()
                .ok_or("config has no tokenizer")?
                .into(),
            eos: u("eos")? as i64,
            vocab: u("vocab")? as usize,
            hidden: u("hidden")? as usize,
            token_size: u("token_size")? as usize,
            max_tokens: u("max_tokens")? as usize,
            max_prompt_tokens: u("max_prompt_tokens")? as usize,
            max_new_tokens: u("max_new_tokens")? as usize,
            temperature: f("temperature")?,
            top_p: f("top_p")?,
            top_k: u("top_k")? as usize,
            repetition_penalty: f("repetition_penalty")?,
            seed: u("seed")?,
            greedy: config["greedy"].as_bool().unwrap_or(false),
            sample_rate: u("sample_rate")? as u32,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        let o = &req.body["options"];
        let opts = self.opts(o);
        let text = match task {
            "synthesize" => req.body["text"].as_str().ok_or("synthesize takes text")?,
            "run" => o["text"].as_str().ok_or("run takes options.text")?,
            _ => return Err(format!("Soprano answers synthesize and run, not {task}")),
        };
        let text = text.trim();
        if text.is_empty() {
            return Err("empty text".into());
        }
        // `run`: the audio token ids of one prompt, as `[n]` i32.
        if task == "run" {
            let (_, toks) = self.latents(text, &opts)?;
            let toks: Vec<i32> = toks.iter().map(|&t| t as i32).collect();
            let n = toks.len();
            return Ok(Message::tensors(vec![Tensor::i32(
                "tokens",
                vec![n],
                &toks,
            )]));
        }
        let wav = self.synthesize(text, &opts)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Soprano);
