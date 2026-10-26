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

//! Fish Audio Drama 3's host logic as an RLX program, answering
//! `synthesize`: what `drama3::synthesize_codes` and the codec do — the
//! text split into chunks, each prompted with the first chunk and the last
//! one before it (`drama3_text.rs`, `prompt.rs`; the host's tokenizer); the
//! backbone over the prompt's rows layer by layer (`layer{l}.prefill`, its
//! caches kept in slots), each layer's router logits routed here (softmax,
//! top 8, renormalized) and its experts applied eight at a time
//! (`experts`, their FP8 weights read for the run from the checkpoint);
//! the tower teacher-forced over the prompt (`tower{i}.prefill`); each frame
//! the main token sampled from `final`'s logits (`sampling.rs`; redrawn when
//! it repeats one of the last 16), the tower's nine codes after it
//! (`tower{i}.step`), the frame's row fed back (`embed`, `layer{l}.step`);
//! until `<|im_end|>`; then the codes dequantized (`dequant`) and decoded
//! (`codec`). `run` answers the codes (`options.codes` decodes given ones).

#[allow(dead_code)]
#[path = "../../src/drama3_text.rs"]
mod drama3_text;
#[allow(dead_code)]
#[path = "../../src/prompt.rs"]
mod prompt;
#[allow(dead_code)]
#[path = "../../src/sampling.rs"]
mod sampling;

use std::collections::BTreeMap;

use prompt::{Row, TextEncoder};
use rlx_program::{Message, Program, Tensor, host, run_body};
use sampling::{Flavor, Rng, SampleParams};
use serde_json::{Value, json};

const TOKENIZER: &str = "sidecar:tokenizer.json";
/// Repetition-aware sampling: a main token among the last 16 is redrawn.
const RAS_WINDOW: usize = 16;
const RAS: SampleParams = SampleParams {
    temperature: 1.0,
    top_p: 0.95,
    top_k: 30,
};
const DECODE_BUCKET: usize = 256;
/// Experts an `experts` run applies.
const PER_RUN: usize = 8;

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

struct Drama {
    /// Each backbone layer: full attention (`true`) or a DeltaNet.
    full: Vec<bool>,
    hidden: usize,
    conv_kernel: usize,
    conv_dim: usize,
    lin_v_heads: usize,
    lin_k_dim: usize,
    lin_v_dim: usize,
    num_experts: usize,
    top_k: usize,
    norm_topk: bool,
    experts: String,
    tower_layers: usize,
    tower_shared: usize,
    tower_vocab: usize,
    nc: usize,
    backbone_codebooks: usize,
    semantic_begin: u32,
    semantic_end: u32,
    eos: u32,
    max_rows: usize,
    max_prompt_rows: usize,
    max_new_tokens: usize,
    params: SampleParams,
    seed: u64,
    semantic_size: usize,
    residual_size: usize,
    sample_rate: u32,
}

fn named(outs: &[Tensor], n: &str) -> Result<Vec<f32>, String> {
    outs.iter()
        .find(|t| t.name == n)
        .map(Tensor::to_f32)
        .ok_or_else(|| format!("no output {n}"))
}

fn map(pairs: &[(&str, String)]) -> Value {
    Value::Object(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), Value::String(v.clone())))
            .collect(),
    )
}

fn set_slot(slot: &str, t: Tensor) -> Result<(), String> {
    host("slot.set", Message::new(json!({ "slot": slot }), vec![t]))?;
    Ok(())
}

impl Drama {
    fn is_semantic(&self, t: u32) -> bool {
        (self.semantic_begin..=self.semantic_end).contains(&t)
    }

    /// Candidate logit index → token id (`0` is `<|im_end|>`).
    fn candidate_token(&self, idx: usize) -> u32 {
        if idx == 0 {
            self.eos
        } else {
            self.semantic_begin + idx as u32 - 1
        }
    }

    /// The rows' embeddings, kept in the slot `d3.x`.
    fn embed(&self, rows: &[Row]) -> Result<(), String> {
        let t = rows.len();
        let k = self.backbone_codebooks;
        let vocab = self.tower_vocab;
        let mut tokens = Vec::with_capacity(t);
        let mut codes = Vec::with_capacity(t * k);
        let mut audio = Vec::with_capacity(t);
        for r in rows {
            tokens.push(r[0] as i32);
            let a = k > 0 && self.is_semantic(r[0]);
            audio.push(if a { 1.0 } else { 0.0 });
            for cb in 1..=k {
                codes.push(if a {
                    (cb * vocab) as i32 + r[1 + cb] as i32
                } else {
                    0
                });
            }
        }
        run_body(
            "embed",
            vec![
                Tensor::i32("tokens", vec![t], &tokens),
                Tensor::i32("codes", vec![t, k], &codes),
                Tensor::f32("audio", vec![t, 1], &audio),
            ],
            json!({"keep": {"embeds": "d3.x"}}),
        )?;
        Ok(())
    }

    /// A row's experts: `(expert, weight)`, as `moe::ExpertBank::route`.
    fn route(&self, logits: &[f32]) -> Vec<(usize, f32)> {
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
        let s: f32 = e.iter().sum();
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| e[b].total_cmp(&e[a]).then(a.cmp(&b)));
        let mut picks: Vec<(usize, f32)> =
            order[..self.top_k].iter().map(|&i| (i, e[i] / s)).collect();
        if self.norm_topk {
            let t: f32 = picks.iter().map(|p| p.1).sum();
            for p in &mut picks {
                p.1 /= t;
            }
        }
        picks
    }

    /// Layer `l`'s routed experts on its `rows` rows (`d3.xn`, router
    /// logits `router`), added to `d3.h` into `d3.x`.
    fn experts(&self, l: usize, router: &[f32], rows: usize) -> Result<(), String> {
        let n = self.num_experts;
        let mut by: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
        for r in 0..rows {
            for (e, w) in self.route(&router[r * n..(r + 1) * n]) {
                by.entry(e).or_default().push((r, w));
            }
        }
        let chosen: Vec<(usize, Vec<(usize, f32)>)> = by.into_iter().collect();
        set_slot(
            "d3.acc",
            Tensor::f32(
                "acc",
                vec![rows, self.hidden],
                &vec![0.0; rows * self.hidden],
            ),
        )?;
        for run in chosen.chunks(PER_RUN) {
            // Picks per expert, a power of two: few shapes to compile.
            let most = run.iter().map(|(_, p)| p.len()).max().unwrap_or(1);
            let q = most.next_power_of_two();
            let mut src = vec![0i32; PER_RUN * q];
            let mut dest = vec![0f32; PER_RUN * q];
            let mut prob = vec![0f32; PER_RUN * q];
            let mut weights = serde_json::Map::new();
            for (input, proj) in [
                ("gate", "gate_proj"),
                ("up", "up_proj"),
                ("down", "down_proj"),
            ] {
                let mut w = Vec::new();
                let mut s = Vec::new();
                for slot in 0..PER_RUN {
                    // An unused slot reads the run's first expert at weight 0.
                    let e = run.get(slot).unwrap_or(&run[0]).0;
                    let key = format!("{}{l}.mlp.experts.{e}.{proj}.weight", self.experts);
                    s.push(Value::String(format!("{key}_scale_inv")));
                    w.push(Value::String(key));
                }
                weights.insert(input.into(), Value::Array(w));
                weights.insert(format!("{input}_s"), Value::Array(s));
            }
            for (slot, (_, picks)) in run.iter().enumerate() {
                for (i, &(r, w)) in picks.iter().enumerate() {
                    src[slot * q + i] = r as i32;
                    dest[slot * q + i] = r as f32;
                    prob[slot * q + i] = w;
                }
            }
            run_body(
                "experts",
                vec![
                    Tensor::i32("src", vec![PER_RUN * q], &src),
                    Tensor::f32("dest", vec![PER_RUN * q], &dest),
                    Tensor::f32("prob", vec![PER_RUN, q, 1], &prob),
                ],
                json!({
                    "from": {"x": "d3.xn", "h": "d3.h", "acc": "d3.acc"},
                    "keep": {"acc.next": "d3.acc", "out": "d3.x"},
                    "weights": weights,
                }),
            )?;
        }
        Ok(())
    }

    fn zeros_state(&self) -> (Tensor, Tensor) {
        let hist = self.conv_kernel - 1;
        let (hv, dk, dv) = (self.lin_v_heads, self.lin_k_dim, self.lin_v_dim);
        (
            Tensor::f32(
                "conv_state",
                vec![1, hist, self.conv_dim],
                &vec![0.0; hist * self.conv_dim],
            ),
            Tensor::f32("ssm_state", vec![1, hv, dk, dv], &vec![0.0; hv * dk * dv]),
        )
    }

    /// The backbone over the prompt (`d3.x` its embeddings): the caches in
    /// their slots, every row's final norm in `d3.hidden`, the last's in
    /// `d3.last`; the main token's logits.
    fn prefill(&self, t: usize, cap: usize) -> Result<Vec<f32>, String> {
        let hv = self.lin_v_heads;
        let (conv, ssm) = self.zeros_state();
        let valid = Tensor::f32("valid", vec![1, t, hv], &vec![1.0; t * hv]);
        for (l, &full) in self.full.iter().enumerate() {
            let (inputs, keep, body) = if full {
                (
                    vec![],
                    map(&[
                        ("h", "d3.h".into()),
                        ("xn", "d3.xn".into()),
                        ("past_k", format!("d3.k{l}")),
                        ("past_v", format!("d3.v{l}")),
                    ]),
                    json!({"cap": cap}),
                )
            } else {
                (
                    vec![conv.clone(), ssm.clone(), valid.clone()],
                    map(&[
                        ("h", "d3.h".into()),
                        ("xn", "d3.xn".into()),
                        ("conv", format!("d3.conv{l}")),
                        ("ssm", format!("d3.ssm{l}")),
                    ]),
                    json!({}),
                )
            };
            let outs = run_body(
                &format!("layer{l}.prefill"),
                inputs,
                json!({"from": {"x": "d3.x"}, "keep": keep, "dims": body}),
            )?;
            self.experts(l, &named(&outs, "router")?, t)?;
        }
        let outs = run_body(
            "final",
            vec![],
            json!({"from": {"x": "d3.x"}, "keep": {"hidden": "d3.hidden", "last": "d3.last"}}),
        )?;
        named(&outs, "logits")
    }

    /// One row through the backbone at `pos` (`d3.x` its embedding): the
    /// main token's logits, its final norm in `d3.last`.
    fn step(&self, pos: usize, cap: usize, mask: &[f32]) -> Result<Vec<f32>, String> {
        let hv = self.lin_v_heads;
        for (l, &full) in self.full.iter().enumerate() {
            let (inputs, from, keep) = if full {
                (
                    vec![
                        Tensor::f32("pos", vec![1], &[pos as f32]),
                        Tensor::f32("mask", vec![1, cap + 1], mask),
                    ],
                    map(&[
                        ("x", "d3.x".into()),
                        ("past_k", format!("d3.k{l}")),
                        ("past_v", format!("d3.v{l}")),
                    ]),
                    map(&[
                        ("h", "d3.h".into()),
                        ("xn", "d3.xn".into()),
                        ("past_k.next", format!("d3.k{l}")),
                        ("past_v.next", format!("d3.v{l}")),
                    ]),
                )
            } else {
                (
                    vec![Tensor::f32("valid", vec![1, 1, hv], &vec![1.0; hv])],
                    map(&[
                        ("x", "d3.x".into()),
                        ("conv_state", format!("d3.conv{l}")),
                        ("ssm_state", format!("d3.ssm{l}")),
                    ]),
                    map(&[
                        ("h", "d3.h".into()),
                        ("xn", "d3.xn".into()),
                        ("conv.next", format!("d3.conv{l}")),
                        ("ssm.next", format!("d3.ssm{l}")),
                    ]),
                )
            };
            let outs = run_body(
                &format!("layer{l}.step"),
                inputs,
                json!({"from": from, "keep": keep}),
            )?;
            self.experts(l, &named(&outs, "router")?, 1)?;
        }
        let outs = run_body(
            "final",
            vec![],
            json!({"from": {"x": "d3.x"}, "keep": {"hidden": "d3.drop", "last": "d3.last"}}),
        )?;
        named(&outs, "logits")
    }

    /// Tower layer `i`'s code inputs for rows labelled `labels` whose own
    /// frames are `curs`.
    fn tower_codes(&self, i: usize, labels: &[&Row], curs: &[&Row]) -> Vec<Tensor> {
        let s = labels.len();
        let ns = self.tower_shared;
        let vocab = self.tower_vocab;
        let mut code = Vec::with_capacity(s);
        let mut on = Vec::with_capacity(s);
        for label in labels {
            if self.is_semantic(label[0]) {
                let c = if i < ns { label[1] } else { label[1 + i - ns] };
                code.push(c.min(vocab as u32 - 1) as i32);
                on.push(1.0);
            } else {
                code.push(0);
                on.push(0.0);
            }
        }
        let mut out = vec![
            Tensor::i32("code", vec![s], &code),
            Tensor::f32("on", vec![s, 1], &on),
        ];
        let first = 1 + self.backbone_codebooks;
        let extra = self.nc - first;
        if i == 0 && extra > 0 {
            let mut codes = Vec::with_capacity(s * extra);
            let mut cur_on = Vec::with_capacity(s);
            for cur in curs {
                let a = self.is_semantic(cur[0]);
                cur_on.push(if a { 1.0 } else { 0.0 });
                for cb in first..self.nc {
                    codes.push((cb * vocab) as i32 + if a { cur[1 + cb] as i32 } else { 0 });
                }
            }
            out.push(Tensor::i32("codes", vec![s, extra], &codes));
            out.push(Tensor::f32("cur_on", vec![s, 1], &cur_on));
        }
        out
    }

    /// Teacher-force the tower over rows `0..t-1`, row `p` labelled with
    /// row `p + 1`, filling its caches.
    fn tower_prefill(&self, rows: &[Row], cap: usize) -> Result<(), String> {
        let m = rows.len() - 1;
        run_body(
            "rows",
            vec![],
            json!({"from": {"x": "d3.hidden"}, "keep": {"rows": "d3.t"}}),
        )?;
        let labels: Vec<&Row> = rows[1..].iter().collect();
        let curs: Vec<&Row> = rows[..m].iter().collect();
        for i in 0..self.tower_layers {
            run_body(
                &format!("tower{i}.prefill"),
                self.tower_codes(i, &labels, &curs),
                json!({
                    "from": {"hidden": "d3.t"},
                    "keep": {"out": "d3.t", "past_k": format!("d3.tk{i}"), "past_v": format!("d3.tv{i}")},
                    "dims": {"cap": cap},
                }),
            )?;
        }
        Ok(())
    }

    /// One tower step at `pos`: the frame after `cur` (`tower_step`).
    #[allow(clippy::too_many_arguments)]
    fn tower_step(
        &self,
        pos: usize,
        cap: usize,
        logits: &[f32],
        cur: &Row,
        history: &[u32],
        params: SampleParams,
        rng: &mut Rng,
    ) -> Result<Row, String> {
        let ns = self.tower_shared;
        let first = self.candidate_token(sampling::sample(logits, params, Flavor::Drama3, rng));
        let retry = self.candidate_token(sampling::sample(logits, RAS, Flavor::Drama3, rng));
        let window = &history[history.len().saturating_sub(RAS_WINDOW)..];
        let main = if self.is_semantic(first) && window.contains(&first) {
            retry
        } else {
            first
        };
        let c0 = main
            .saturating_sub(self.semantic_begin)
            .min(self.tower_vocab as u32 - 1);
        let mut frame = vec![0u32; 1 + self.nc];
        frame[0] = main;
        frame[1] = c0;
        if main == self.eos {
            return Ok(frame);
        }
        let mut mask = vec![0f32; cap + 1];
        mask[..pos].fill(1.0);
        mask[cap] = 1.0;
        for i in 0..self.tower_layers {
            let mut inputs = self.tower_codes(i, &[&frame], &[cur]);
            inputs.push(Tensor::f32("pos", vec![1], &[pos as f32]));
            inputs.push(Tensor::f32("mask", vec![1, cap + 1], &mask));
            let hidden = if i == 0 { "d3.last" } else { "d3.t" };
            let outs = run_body(
                &format!("tower{i}.step"),
                inputs,
                json!({
                    "from": {"hidden": hidden, "past_k": format!("d3.tk{i}"), "past_v": format!("d3.tv{i}")},
                    "keep": {"out": "d3.t", "past_k.next": format!("d3.tk{i}"), "past_v.next": format!("d3.tv{i}")},
                }),
            )?;
            if i >= ns {
                let l = named(&outs, "logits")?;
                frame[2 + i - ns] = sampling::sample(&l, params, Flavor::Drama3, rng) as u32;
            }
        }
        Ok(frame)
    }

    /// Frames after `rows` until `<|im_end|>` or `max_new` (`generate`).
    fn generate(
        &self,
        rows: &[Row],
        max_new: usize,
        params: SampleParams,
        rng: &mut Rng,
    ) -> Result<Vec<Row>, String> {
        let t = rows.len();
        let cap = (t + max_new).div_ceil(DECODE_BUCKET) * DECODE_BUCKET;
        if cap > self.max_rows {
            return Err(format!(
                "prompt {t} + {max_new} frames exceeds {} rows",
                self.max_rows
            ));
        }
        self.embed(rows)?;
        let mut logits = self.prefill(t, cap)?;
        self.tower_prefill(rows, cap)?;
        let mut history: Vec<u32> = rows.iter().map(|r| r[0]).collect();
        let mut cur = rows[t - 1].clone();
        let mut mask = vec![0f32; cap + 1];
        mask[..t].fill(1.0);
        mask[cap] = 1.0;
        let mut out = Vec::new();
        for step in 0..max_new {
            let pos = t - 1 + step;
            let frame = self.tower_step(pos, cap, &logits, &cur, &history, params, rng)?;
            if frame[0] == self.eos {
                break;
            }
            history.push(frame[0]);
            out.push(frame.clone());
            if step + 1 < max_new {
                let len = t + step;
                self.embed(std::slice::from_ref(&frame))?;
                logits = self.step(len, cap, &mask)?;
                mask[len] = 1.0;
            }
            cur = frame;
        }
        let mut slots: Vec<String> = vec![
            "d3.x",
            "d3.h",
            "d3.xn",
            "d3.acc",
            "d3.hidden",
            "d3.last",
            "d3.drop",
            "d3.t",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        for l in 0..self.full.len() {
            slots.extend([
                format!("d3.k{l}"),
                format!("d3.v{l}"),
                format!("d3.conv{l}"),
                format!("d3.ssm{l}"),
            ]);
        }
        for i in 0..self.tower_layers {
            slots.extend([format!("d3.tk{i}"), format!("d3.tv{i}")]);
        }
        host("slot.drop", Message::body(json!({ "slots": slots })))?;
        Ok(out)
    }

    /// `synthesize_codes`: codes `[num_codebooks][frames]`.
    fn codes(&self, text: &str, o: &Value) -> Result<Vec<Vec<u32>>, String> {
        let max_new = o["max_new_tokens"]
            .as_u64()
            .map_or(self.max_new_tokens, |v| v as usize);
        drama3_text::check_tokenizer(
            &HostTokenizer,
            self.semantic_begin,
            self.semantic_end,
            self.eos,
            self.tower_vocab,
        )
        .map_err(|e| e.to_string())?;
        let chunks = if o["single_chunk"].as_bool().unwrap_or(false) {
            vec![text.to_string()]
        } else {
            let v = drama3_text::split_text(
                text,
                drama3_text::CHUNK_BYTES,
                drama3_text::CHUNK_MAX_SPEAKERS,
            );
            if v.is_empty() {
                vec![text.to_string()]
            } else {
                v
            }
        };
        let mut rng = Rng::new(o["seed"].as_u64().unwrap_or(self.seed));
        let mut history = drama3_text::History::default();
        let mut all = vec![Vec::new(); self.nc];
        for chunk in &chunks {
            let mut rows = None;
            for turns in history.candidates() {
                let segs =
                    drama3_text::prompt_segments(chunk, &[], &turns).map_err(|e| e.to_string())?;
                let r = prompt::rows(&segs, &HostTokenizer, self.nc, self.semantic_begin)
                    .map_err(|e| e.to_string())?;
                if r.len() <= self.max_prompt_rows {
                    rows = Some(r);
                    break;
                }
            }
            let rows = rows.ok_or_else(|| {
                format!(
                    "chunk prompt exceeds {} rows even without history",
                    self.max_prompt_rows
                )
            })?;
            let frames = self.generate(&rows, max_new, self.params, &mut rng)?;
            if frames.is_empty() {
                continue;
            }
            let codes: Vec<Vec<u32>> = (0..self.nc)
                .map(|i| frames.iter().map(|f| f[1 + i]).collect())
                .collect();
            for (dst, src) in all.iter_mut().zip(&codes) {
                dst.extend(src);
            }
            history.push(drama3_text::Turn {
                text: chunk.clone(),
                codes,
            });
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
        let outs = run_body(
            "dequant",
            vec![Tensor::i32("codes", vec![f, self.nc], &flat)],
            json!({"keep": {"z": "d3.z"}}),
        )?;
        debug_assert!(outs.is_empty());
        let outs = run_body("codec", vec![], json!({"from": {"z": "d3.z"}}))?;
        host("slot.drop", Message::body(json!({ "slots": ["d3.z"] })))?;
        named(&outs, "audio")
    }
}

impl Program for Drama {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        let f = |k: &str| {
            config[k]
                .as_f64()
                .map(|v| v as f32)
                .ok_or(format!("config has no {k}"))
        };
        let full = config["layers"]
            .as_array()
            .ok_or("config has no layers")?
            .iter()
            .map(|k| k.as_str() == Some("full"))
            .collect();
        Ok(Drama {
            full,
            hidden: u("hidden")? as usize,
            conv_kernel: u("conv_kernel")? as usize,
            conv_dim: u("conv_dim")? as usize,
            lin_v_heads: u("lin_v_heads")? as usize,
            lin_k_dim: u("lin_k_dim")? as usize,
            lin_v_dim: u("lin_v_dim")? as usize,
            num_experts: u("num_experts")? as usize,
            top_k: u("top_k")? as usize,
            norm_topk: config["norm_topk"].as_bool().unwrap_or(true),
            experts: config["experts"]
                .as_str()
                .ok_or("config has no experts")?
                .to_string(),
            tower_layers: u("tower_layers")? as usize,
            tower_shared: u("tower_shared")? as usize,
            tower_vocab: u("tower_vocab")? as usize,
            nc: u("num_codebooks")? as usize,
            backbone_codebooks: u("backbone_codebooks")? as usize,
            semantic_begin: u("semantic_begin")? as u32,
            semantic_end: u("semantic_end")? as u32,
            eos: u("eos")? as u32,
            max_rows: u("max_rows")? as usize,
            max_prompt_rows: u("max_prompt_rows")? as usize,
            max_new_tokens: u("max_new_tokens")? as usize,
            params: SampleParams {
                temperature: f("temperature")?,
                top_p: f("top_p")?,
                top_k: u("top_k_sample")? as usize,
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
            _ => return Err(format!("Drama 3 answers synthesize and run, not {task}")),
        };
        let mut params = self.params;
        if let Some(t) = o["temperature"].as_f64() {
            params.temperature = t as f32;
        }
        if let Some(p) = o["top_p"].as_f64() {
            params.top_p = p as f32;
        }
        if let Some(k) = o["top_k"].as_u64() {
            params.top_k = k as usize;
        }
        let saved = self.params;
        self.params = params;
        // `options.codes`: codes `[10][frames]` to decode, the stacks skipped.
        let codes: Result<Vec<Vec<u32>>, String> = match o["codes"].as_array() {
            Some(rows) => Ok(rows
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
                .collect()),
            None => self.codes(text, o),
        };
        self.params = saved;
        let codes = codes?;
        if codes.is_empty() || codes[0].is_empty() {
            return Err("Drama 3 produced zero frames".into());
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
        let wav = self.decode(&codes)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Drama);
