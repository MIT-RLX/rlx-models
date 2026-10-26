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

//! Fish Audio S2-Pro's host logic as an RLX program, answering
//! `synthesize`: what `s2::synthesize_codes` and the codec do — the text
//! split by speaker into batches, each a user turn of the growing
//! conversation (`prompt.rs`; the host's tokenizer, special tokens inline);
//! the slow stack over its rows (`embed`, `prefill`, its caches kept in
//! slots, one length a request); each frame the main token sampled from the
//! band logits (`sampling.rs`: top-p over the softmax, top-k, temperature;
//! redrawn when it repeats one of the last ten), the fast decoder's nine
//! codes after it (`fast`, `fast_embed`), the frame's row fed back
//! (`step`); until `<|im_end|>`; then the codes dequantized (`dequant`) and
//! decoded (`codec`). `run` answers the codes (`options.codes` decodes
//! given ones).

#[allow(dead_code)]
#[path = "../../src/prompt.rs"]
mod prompt;
#[allow(dead_code)]
#[path = "../../src/sampling.rs"]
mod sampling;

use prompt::{Row, Segment, TextEncoder};
use rlx_program::{Message, Program, Tensor, host, run, run_body, run_with};
use sampling::{Flavor, Rng, SampleParams};
use serde_json::{Value, json};

const TOKENIZER: &str = "sidecar:tokenizer.json";
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>\n";
/// Repetition-aware sampling: a main token among the last ten is redrawn.
const RAS_WINDOW: usize = 10;
const RAS: SampleParams = SampleParams {
    temperature: 1.0,
    top_p: 0.9,
    top_k: 30,
};
const DECODE_BUCKET: usize = 256;

/// The host's tokenizer.
struct HostTokenizer;

impl TextEncoder for HostTokenizer {
    fn encode_text(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let m = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": TOKENIZER, "text": text, "special": false})),
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        Ok(m.body["ids"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("no ids"))?
            .iter()
            .filter_map(|v| v.as_u64().map(|v| v as u32))
            .collect())
    }

    fn special_id(&self, token: &str) -> Option<u32> {
        match self.encode_text(token).ok()?.as_slice() {
            [id] => Some(*id),
            _ => None,
        }
    }
}

struct Fish {
    slow_layers: usize,
    fast_layers: usize,
    fast_kv: usize,
    fast_dim: usize,
    nc: usize,
    codebook_size: usize,
    semantic_begin: u32,
    semantic_end: u32,
    eos: u32,
    max_seq_len: usize,
    max_rows: usize,
    max_new_tokens: usize,
    chunk_length: usize,
    params: SampleParams,
    seed: u64,
    semantic_size: usize,
    residual_size: usize,
    sample_rate: u32,
}

fn first(outs: Vec<Tensor>, what: &str) -> Result<Tensor, String> {
    outs.into_iter()
        .next()
        .ok_or_else(|| format!("{what} gave nothing"))
}

fn named(outs: &[Tensor], n: &str) -> Result<Vec<f32>, String> {
    outs.iter()
        .find(|t| t.name == n)
        .map(Tensor::to_f32)
        .ok_or_else(|| format!("no output {n}"))
}

fn pairs(v: &[(String, String)]) -> Vec<(&str, &str)> {
    v.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect()
}

/// `(input, slot)` pairs.
type Slots = Vec<(String, String)>;

/// Each layer's cache inputs and slots, and their `.next` outputs'.
fn cache_names(prefix: &str, layers: usize) -> (Slots, Slots) {
    let names: Vec<(String, String)> = (0..layers)
        .flat_map(|l| [format!("past_k_{l}"), format!("past_v_{l}")])
        .map(|n| (n.clone(), format!("{prefix}.{n}")))
        .collect();
    let nexts = names
        .iter()
        .map(|(n, s)| (format!("{n}.next"), s.clone()))
        .collect();
    (names, nexts)
}

/// S2's conversation for one utterance (`s2::Conversation`, no references).
struct Conversation {
    segments: Vec<Segment>,
}

impl Conversation {
    fn new() -> Self {
        Self {
            segments: vec![
                Segment::Text(format!("{IM_START}system\n")),
                Segment::Text("convert the provided text to speech".into()),
                Segment::Text(IM_END.into()),
            ],
        }
    }

    fn push_user(&mut self, text: &str) {
        self.segments
            .push(Segment::Text(format!("{IM_START}user\n")));
        self.segments.push(Segment::Text(text.into()));
        self.segments.push(Segment::Text(IM_END.into()));
    }

    fn pop_user(&mut self) {
        let n = self.segments.len();
        self.segments.truncate(n - 3);
    }

    fn push_assistant(&mut self, codes: Vec<Vec<u32>>) {
        self.segments
            .push(Segment::Text(format!("{IM_START}assistant\n<|voice|>")));
        self.segments.push(Segment::Codes(codes));
        self.segments.push(Segment::Text(IM_END.into()));
    }

    fn prompt_rows(&self, f: &Fish) -> Result<Vec<Row>, String> {
        let mut segs = self.segments.clone();
        segs.push(Segment::Text(format!("{IM_START}assistant\n<|voice|>")));
        prompt::rows(&segs, &HostTokenizer, f.nc, f.semantic_begin).map_err(|e| e.to_string())
    }
}

impl Fish {
    fn is_semantic(&self, t: u32) -> bool {
        (self.semantic_begin..=self.semantic_end).contains(&t)
    }

    fn band_token(&self, idx: usize) -> u32 {
        if idx < self.codebook_size {
            self.semantic_begin + idx as u32
        } else {
            self.eos
        }
    }

    /// The rows' embeddings `[1, t, dim]`.
    fn embed(&self, rows: &[Row]) -> Result<Tensor, String> {
        let t = rows.len();
        let mut tokens = Vec::with_capacity(t);
        let mut codes = Vec::with_capacity(t * self.nc);
        let mut audio = Vec::with_capacity(t);
        for r in rows {
            tokens.push(r[0] as i32);
            let a = self.is_semantic(r[0]);
            audio.push(if a { 1.0 } else { 0.0 });
            for i in 0..self.nc {
                codes.push(if a {
                    (i * self.codebook_size) as i32 + r[1 + i] as i32
                } else {
                    0
                });
            }
        }
        let mut x = first(
            run(
                "embed",
                vec![
                    Tensor::i32("tokens", vec![t], &tokens),
                    Tensor::i32("codes", vec![t, self.nc], &codes),
                    Tensor::f32("audio", vec![t, 1], &audio),
                ],
            )?,
            "embed",
        )?;
        x.name = "embeds".into();
        Ok(x)
    }

    /// One frame's codes after the slow stack chose `main` (`fast_frame`).
    fn fast_frame(&self, hidden: &[f32], main: u32, rng: &mut Rng) -> Result<Vec<u32>, String> {
        let nc = self.nc;
        let (names, nexts) = cache_names("fish.fast", self.fast_layers);
        for (n, s) in &names {
            host(
                "slot.set",
                Message::new(
                    json!({ "slot": s }),
                    vec![Tensor::f32(
                        n.clone(),
                        vec![1, nc, self.fast_kv],
                        &vec![0.0; nc * self.fast_kv],
                    )],
                ),
            )?;
        }
        let c0 = main
            .saturating_sub(self.semantic_begin)
            .min(self.codebook_size as u32 - 1);
        let mut codes = vec![c0];
        let mut x = Tensor::f32("x", vec![1, 1, self.fast_dim], hidden);
        for pos in 0..nc {
            let mut mask = vec![0f32; nc + 1];
            mask[..pos].fill(1.0);
            mask[nc] = 1.0;
            let outs = run_with(
                "fast",
                vec![
                    x.clone(),
                    Tensor::f32("pos", vec![1], &[pos as f32]),
                    Tensor::f32("mask", vec![1, nc + 1], &mask),
                ],
                &pairs(&names),
                &pairs(&nexts),
            )?;
            if pos > 0 {
                let logits = named(&outs, "logits")?;
                codes.push(sampling::sample(&logits, self.params, Flavor::S2, rng) as u32);
            }
            if pos + 1 < nc {
                let prev = *codes.last().expect("a code") as i32;
                x = first(
                    run("fast_embed", vec![Tensor::i32("id", vec![1], &[prev])])?,
                    "fast_embed",
                )?;
                x.name = "x".into();
            }
        }
        Ok(codes)
    }

    /// Frames after `rows` until `<|im_end|>` or `max_new` (`generate`).
    fn generate(
        &self,
        rows: &[Row],
        max_new: usize,
        rng: &mut Rng,
    ) -> Result<Vec<Vec<u32>>, String> {
        let t = rows.len();
        if t + max_new > self.max_seq_len {
            return Err(format!("prompt {t} + {max_new} frames exceeds the context"));
        }
        let cap = (t + max_new)
            .div_ceil(DECODE_BUCKET)
            .saturating_mul(DECODE_BUCKET)
            .min(self.max_seq_len)
            .min(self.max_rows);
        let (names, nexts) = cache_names("fish.slow", self.slow_layers);
        let keep: serde_json::Map<String, Value> = names
            .iter()
            .map(|(a, b)| (a.clone(), Value::String(b.clone())))
            .collect();
        let outs = run_body(
            "prefill",
            vec![self.embed(rows)?],
            json!({"keep": keep, "dims": {"cap": cap}}),
        )?;
        let (mut hidden, mut logits) = (named(&outs, "hidden")?, named(&outs, "logits")?);
        let mut len = t;
        let mut window: Vec<u32> = Vec::with_capacity(RAS_WINDOW);
        let mut frames = Vec::new();
        let mut mask = vec![0f32; cap + 1];
        mask[..t].fill(1.0);
        mask[cap] = 1.0;
        for step in 0..max_new {
            let first_draw =
                self.band_token(sampling::sample(&logits, self.params, Flavor::S2, rng));
            let mut main = first_draw;
            if step > 0 {
                let ras = SampleParams {
                    top_k: self.params.top_k,
                    ..RAS
                };
                let retry = self.band_token(sampling::sample(&logits, ras, Flavor::S2, rng));
                if self.is_semantic(first_draw) && window.contains(&first_draw) {
                    main = retry;
                }
            }
            if main == self.eos {
                break;
            }
            let codes = self.fast_frame(&hidden, main, rng)?;
            if step > 0 {
                if window.len() == RAS_WINDOW {
                    window.remove(0);
                }
                window.push(main);
            }
            if step + 1 < max_new {
                if len >= cap {
                    return Err(format!("KV cache full ({cap} rows)"));
                }
                let mut row = vec![main];
                row.extend(&codes);
                let outs = run_with(
                    "step",
                    vec![
                        self.embed(&[row])?,
                        Tensor::f32("pos", vec![1], &[len as f32]),
                        Tensor::f32("mask", vec![1, cap + 1], &mask),
                    ],
                    &pairs(&names),
                    &pairs(&nexts),
                )?;
                hidden = named(&outs, "hidden")?;
                logits = named(&outs, "logits")?;
                mask[len] = 1.0;
                len += 1;
            }
            frames.push(codes);
        }
        let slots: Vec<&str> = names.iter().map(|(_, s)| s.as_str()).collect();
        host("slot.drop", Message::body(json!({ "slots": slots })))?;
        Ok(frames)
    }

    /// `synthesize_codes`: codes `[num_codebooks][frames]`.
    fn codes(&self, text: &str, o: &Value) -> Result<Vec<Vec<u32>>, String> {
        let max_new = o["max_new_tokens"]
            .as_u64()
            .map_or(self.max_new_tokens, |v| v as usize);
        let turns = prompt::split_by_speaker(text);
        let batches = if turns.is_empty() {
            vec![text.to_string()]
        } else {
            prompt::group_turns(&turns, 5, self.chunk_length)
        };
        let mut conv = Conversation::new();
        let mut rng = Rng::new(o["seed"].as_u64().unwrap_or(self.seed));
        let mut all = vec![Vec::new(); self.nc];
        for batch in batches {
            conv.push_user(&batch);
            let rows = conv.prompt_rows(self)?;
            if rows.len() + 2048 > self.max_seq_len {
                return Err(format!("prompt is too long: {} rows", rows.len()));
            }
            let frames = self.generate(&rows, max_new, &mut rng)?;
            if frames.is_empty() {
                conv.pop_user();
                continue;
            }
            let codes: Vec<Vec<u32>> = (0..self.nc)
                .map(|i| frames.iter().map(|f| f[i]).collect())
                .collect();
            for (dst, src) in all.iter_mut().zip(&codes) {
                dst.extend(src);
            }
            conv.push_assistant(codes);
        }
        Ok(all)
    }

    /// Codes `[num_codebooks][frames]` → 44.1 kHz audio.
    fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>, String> {
        let f = codes[0].len();
        let mut flat = Vec::with_capacity(f * self.nc);
        for t in 0..f {
            for (i, book) in codes.iter().enumerate() {
                let limit = if i == 0 {
                    self.semantic_size
                } else {
                    self.residual_size
                };
                flat.push((book[t] as usize).min(limit - 1) as i32);
            }
        }
        let mut z = first(
            run(
                "dequant",
                vec![Tensor::i32("codes", vec![f, self.nc], &flat)],
            )?,
            "dequant",
        )?;
        z.name = "z".into();
        Ok(first(run("codec", vec![z])?, "codec")?.to_f32())
    }
}

impl Program for Fish {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        let f = |k: &str| {
            config[k]
                .as_f64()
                .map(|v| v as f32)
                .ok_or(format!("config has no {k}"))
        };
        Ok(Fish {
            slow_layers: u("slow_layers")? as usize,
            fast_layers: u("fast_layers")? as usize,
            fast_kv: u("fast_kv")? as usize,
            fast_dim: u("fast_dim")? as usize,
            nc: u("num_codebooks")? as usize,
            codebook_size: u("codebook_size")? as usize,
            semantic_begin: u("semantic_begin")? as u32,
            semantic_end: u("semantic_end")? as u32,
            eos: u("eos")? as u32,
            max_seq_len: u("max_seq_len")? as usize,
            max_rows: u("max_rows")? as usize,
            max_new_tokens: u("max_new_tokens")? as usize,
            chunk_length: u("chunk_length")? as usize,
            params: SampleParams {
                temperature: f("temperature")?,
                top_p: f("top_p")?,
                top_k: u("top_k")? as usize,
            },
            seed: u("seed")?,
            semantic_size: u("semantic_size")? as usize,
            residual_size: u("residual_size")? as usize,
            sample_rate: u("sample_rate")? as u32,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        let o = &req.body["options"];
        let text = match task {
            "synthesize" => req.body["text"].as_str().ok_or("synthesize takes text")?,
            "run" => o["text"].as_str().unwrap_or(""),
            _ => return Err(format!("Fish S2 answers synthesize and run, not {task}")),
        };
        let mut me = Fish { ..*self };
        if let Some(t) = o["temperature"].as_f64() {
            me.params.temperature = t as f32;
        }
        if let Some(p) = o["top_p"].as_f64() {
            me.params.top_p = p as f32;
        }
        if let Some(k) = o["top_k"].as_u64() {
            me.params.top_k = k as usize;
        }
        // `options.codes`: codes `[10][frames]` to decode, the stacks skipped.
        let codes: Vec<Vec<u32>> = match o["codes"].as_array() {
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
            None => me.codes(text, o)?,
        };
        if codes.is_empty() || codes[0].is_empty() {
            return Err("Fish produced zero frames".into());
        }
        if task == "run" {
            let f = codes[0].len();
            let flat: Vec<i32> = codes.iter().flatten().map(|&c| c as i32).collect();
            return Ok(Message::tensors(vec![Tensor::i32(
                "codes",
                vec![codes.len(), f],
                &flat,
            )]));
        }
        let wav = me.decode(&codes)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": me.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Fish);
