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

//! Sesame CSM's host logic as an RLX program, answering `synthesize`: what
//! `SesameSession::synthesize` does — the prompt's text frames
//! (`<bos>[speaker]text<eos>`, the host's tokenizer) embedded
//! (`frame_embed`); each frame the backbone over the frames so far (its
//! last hidden state), codebook 0 from `c0_head`, codebooks 1 … 31 from the
//! depth decoder over the projected state and the codes so far
//! (`depth_project`, `depth_embed`, `depth`, `codebook_head`), each sampled
//! top-k from the runner's `fastrand` stream; the frame fed back, until one
//! of all zeros; Mimi (`mimi_lookup`, `mimi_decode`). `run` answers the
//! frames' codes.

use rlx_program::{Message, Program, Tensor, host, run};
use serde_json::{Value, json};

struct Csm {
    tokenizer: String,
    bos: u32,
    eos: u32,
    k: usize,
    depth_len: usize,
    hidden: usize,
    depth_hidden: usize,
    max_audio_frames: usize,
    temperature: f32,
    topk: usize,
    seed: u64,
    sample_rate: u32,
}

/// Top-k sampling, the runner's `sample_topk`.
fn sample_topk(logits: &[f32], topk: usize, temperature: f32, rng: &mut fastrand::Rng) -> u32 {
    let temp = temperature.max(1e-5);
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v / temp))
        .collect();
    let k = topk.min(indexed.len()).max(1);
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed.truncate(k);
    let max_v = indexed
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_v).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum.max(1e-9);
    }
    let r = rng.f32();
    let mut acc = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        acc += *p;
        if r <= acc {
            return indexed[i].0 as u32;
        }
    }
    indexed[k - 1].0 as u32
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn one(name: &str, v: u32) -> Tensor {
    Tensor::i32(name, vec![1], &[v as i32])
}

fn first(outs: Vec<Tensor>, what: &str) -> Result<Vec<f32>, String> {
    Ok(outs
        .into_iter()
        .next()
        .ok_or_else(|| format!("{what} gave nothing"))?
        .to_f32())
}

impl Csm {
    /// Frames (`[n][k + 1]` tokens, masks) → their embeddings `[n · hidden]`.
    fn embed(&self, frames: &[(Vec<u32>, Vec<f32>)]) -> Result<Vec<f32>, String> {
        let n = frames.len();
        let tokens: Vec<i32> = frames
            .iter()
            .flat_map(|(t, _)| t.iter().map(|&x| x as i32))
            .collect();
        let mask: Vec<f32> = frames.iter().flat_map(|(_, m)| m.clone()).collect();
        first(
            run(
                "frame_embed",
                vec![
                    Tensor::i32("tokens", vec![n, self.k + 1], &tokens),
                    Tensor::f32("mask", vec![n, self.k + 1], &mask),
                ],
            )?,
            "frame_embed",
        )
    }

    fn text_frame(&self, token: u32) -> (Vec<u32>, Vec<f32>) {
        let mut t = vec![0; self.k + 1];
        let mut m = vec![0.0; self.k + 1];
        t[self.k] = token;
        m[self.k] = 1.0;
        (t, m)
    }

    fn audio_frame(&self, codes: &[u32]) -> (Vec<u32>, Vec<f32>) {
        let mut t = vec![0; self.k + 1];
        let mut m = vec![0.0; self.k + 1];
        t[..self.k].copy_from_slice(codes);
        for v in m.iter_mut().take(self.k) {
            *v = 1.0;
        }
        (t, m)
    }

    /// The last row of a causal model over `rows` (`[n · width]`), padded
    /// to `len` rows: one length compiled per request, not per step.
    fn last_row(
        &self,
        function: &str,
        rows: &[f32],
        width: usize,
        len: usize,
    ) -> Result<Vec<f32>, String> {
        let n = rows.len() / width;
        let mut x = rows.to_vec();
        x.resize(len * width, 0.0);
        let h = first(
            run(
                function,
                vec![Tensor::f32("inputs_embeds", vec![1, len, width], &x)],
            )?,
            function,
        )?;
        Ok(h[(n - 1) * width..n * width].to_vec())
    }

    /// `generate_codes`: the frames' codes.
    fn frames(&self, text: &str, speaker: u32, o: &Value) -> Result<Vec<Vec<u32>>, String> {
        let m = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": self.tokenizer,
                                 "text": format!("[{speaker}]{text}"), "special": false})),
        )?;
        let ids: Vec<u32> = m.body["ids"]
            .as_array()
            .ok_or("no ids")?
            .iter()
            .filter_map(|v| v.as_u64().map(|v| v as u32))
            .collect();
        let mut prompt = vec![self.text_frame(self.bos)];
        prompt.extend(ids.iter().map(|&id| self.text_frame(id)));
        prompt.push(self.text_frame(self.eos));
        let mut embeds = self.embed(&prompt)?;

        let max_frames = o["max_audio_frames"]
            .as_u64()
            .map_or(self.max_audio_frames, |v| v as usize);
        let temperature = o["temperature"]
            .as_f64()
            .map_or(self.temperature, |v| v as f32);
        let topk = o["topk"].as_u64().map_or(self.topk, |v| v as usize);
        let greedy = o["greedy"].as_bool().unwrap_or(false);
        let mut rng = fastrand::Rng::with_seed(o["seed"].as_u64().unwrap_or(self.seed));
        let mut pick = |logits: &[f32], rng: &mut fastrand::Rng| {
            if greedy {
                argmax(logits)
            } else {
                sample_topk(logits, topk, temperature, rng)
            }
        };
        let context = prompt.len() + max_frames;
        let mut out = Vec::new();
        for _ in 0..max_frames {
            let h = self.last_row("backbone", &embeds, self.hidden, context)?;
            let c0 = pick(
                &first(
                    run("c0_head", vec![Tensor::f32("h", vec![self.hidden], &h)])?,
                    "c0_head",
                )?,
                &mut rng,
            );
            let mut codes = vec![c0];
            let mut depth = first(
                run(
                    "depth_project",
                    vec![Tensor::f32("h", vec![self.hidden], &h)],
                )?,
                "depth_project",
            )?;
            depth.extend(first(
                run("depth_embed", vec![one("codebook", 0), one("code", c0)])?,
                "depth_embed",
            )?);
            for cb in 1..self.k {
                let hd = self.last_row("depth", &depth, self.depth_hidden, self.depth_len)?;
                let logits = first(
                    run(
                        "codebook_head",
                        vec![
                            Tensor::f32("h", vec![self.depth_hidden], &hd),
                            one("codebook", cb as u32),
                        ],
                    )?,
                    "codebook_head",
                )?;
                let ci = pick(&logits, &mut rng);
                codes.push(ci);
                if cb + 1 < self.k {
                    depth.extend(first(
                        run(
                            "depth_embed",
                            vec![one("codebook", cb as u32), one("code", ci)],
                        )?,
                        "depth_embed",
                    )?);
                }
            }
            if codes.iter().all(|&c| c == 0) {
                break;
            }
            embeds.extend(self.embed(&[self.audio_frame(&codes)])?);
            out.push(codes);
        }
        Ok(out)
    }

    /// Frames → 24 kHz audio, Mimi.
    fn decode(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>, String> {
        let (k, t) = (self.k, frames.len());
        let mut codes = vec![0i32; k * t];
        for (f, row) in frames.iter().enumerate() {
            for (c, &v) in row.iter().enumerate().take(k) {
                codes[c * t + f] = v as i32;
            }
        }
        let z = run(
            "mimi_lookup",
            vec![Tensor::i32("codes", vec![k, t], &codes)],
        )?;
        let mut z = z.into_iter().next().ok_or("Mimi gave no latent")?;
        z.name = "z".into();
        first(run("mimi_decode", vec![z])?, "mimi_decode")
    }
}

impl Program for Csm {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        let f = |k: &str| {
            config[k]
                .as_f64()
                .map(|v| v as f32)
                .ok_or(format!("config has no {k}"))
        };
        Ok(Csm {
            tokenizer: config["tokenizer"]
                .as_str()
                .ok_or("config has no tokenizer")?
                .into(),
            bos: u("bos")? as u32,
            eos: u("eos")? as u32,
            k: u("num_codebooks")? as usize,
            depth_len: u("depth_len")? as usize,
            hidden: u("hidden")? as usize,
            depth_hidden: u("depth_hidden")? as usize,
            max_audio_frames: u("max_audio_frames")? as usize,
            temperature: f("temperature")?,
            topk: u("topk")? as usize,
            seed: u("seed")?,
            sample_rate: u("sample_rate")? as u32,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        let o = &req.body["options"];
        let speaker = o["speaker"].as_u64().unwrap_or(0) as u32;
        let text = match task {
            "synthesize" => req.body["text"].as_str().ok_or("synthesize takes text")?,
            "run" => o["text"].as_str().ok_or("run takes options.text")?,
            _ => return Err(format!("Sesame answers synthesize and run, not {task}")),
        };
        // `options.frames`: codes `[n][32]` to decode, the LM skipped.
        let frames = match o["frames"].as_array() {
            Some(rows) => rows
                .iter()
                .map(|r| {
                    r.as_array()
                        .map(|c| {
                            c.iter()
                                .filter_map(|v| v.as_u64().map(|v| v as u32))
                                .collect()
                        })
                        .unwrap_or_default()
                })
                .collect(),
            None => self.frames(text, speaker, o)?,
        };
        if frames.is_empty() {
            return Err("CSM produced zero audio frames".into());
        }
        if task == "run" {
            let flat: Vec<i32> = frames.iter().flatten().map(|&c| c as i32).collect();
            let n = frames.len();
            return Ok(Message::tensors(vec![Tensor::i32(
                "frames",
                vec![n, self.k],
                &flat,
            )]));
        }
        let wav = self.decode(&frames)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Csm);
