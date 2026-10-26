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

//! Export Moshi (`kyutai/moshiko-candle-bf16`) one-way as an `.rlxp`
//! package rlx runs alone: a text prompt in, Moshi's speech out.
//!
//! Its graphs read the checkpoint's tensors by their own names (referenced,
//! none copied): `embed` (the text token and the delayed audio codes),
//! `step` (the temporal transformer one row at `position`, under the mask
//! the program gives; the text logits, the hidden state before `out_norm`,
//! each cache with the row written in), symbolic in the caches' rows;
//! `dep_{k}` (the DepFormer's slice `k`, its own weights, attending to the
//! frame's earlier slices); and Mimi's `mimi_lookup` and `mimi_decode`
//! ([`rlx_mimi::export::mimi_functions`]). The program (`program/`) is the
//! runner's one-way loop ([`crate::rlx_gen::RlxGenerateState::step`]): the
//! prompt's tokens one a frame, the delayed codes, the text and audio
//! samplers (the crate's `sampling.rs`, its seeds), the forced pad tokens
//! of the first frames, and each frame past the acoustic delay to Mimi.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::export::graph::symbolic_in;
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use serde_json::json;

use crate::config::{GenerateConfig, LmConfig};

/// The caches' rows (Mimi's lookup and decode take 0 and 1).
const CAP: u32 = 2;
const EPS: f32 = 1e-8;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// A graph over the checkpoint's tensors, each param named and shaped as
/// the checkpoint stores it.
struct B<'a> {
    g: Graph,
    shapes: &'a HashMap<String, Vec<usize>>,
}

impl B<'_> {
    fn new<'a>(name: &str, shapes: &'a HashMap<String, Vec<usize>>) -> B<'a> {
        B {
            g: Graph::new(name),
            shapes,
        }
    }

    fn p(&mut self, key: &str) -> Result<NodeId> {
        let d = self
            .shapes
            .get(key)
            .with_context(|| format!("no tensor {key}"))?;
        Ok(self.g.param(key, f32s(d)))
    }

    fn pt(&mut self, key: &str) -> Result<NodeId> {
        let w = self.p(key)?;
        Ok(self.g.transpose_(w, vec![1, 0]))
    }

    fn rms(&mut self, x: NodeId, key: &str, d: usize) -> Result<NodeId> {
        let a = self.p(key)?;
        let a = self.g.reshape_(a, vec![d as i64]);
        let shape = self.g.node(x).shape.clone();
        let zero = self.g.add_node(
            Op::Constant {
                data: 0f32.to_le_bytes().to_vec(),
            },
            vec![],
            f32s(&[1]),
        );
        let beta = self.g.add_node(
            Op::Expand {
                target_shape: vec![d as i64],
            },
            vec![zero],
            f32s(&[d]),
        );
        Ok(self
            .g
            .add_node(Op::RmsNorm { axis: -1, eps: EPS }, vec![x, a, beta], shape))
    }

    /// `x · Wᵀ` for the rows `[start, start + len)` of the stored `key`.
    fn rows_linear(&mut self, x: NodeId, key: &str, start: usize, len: usize) -> Result<NodeId> {
        let w = self.p(key)?;
        let w = self.g.narrow_(w, 0, start, len);
        let w = self.g.transpose_(w, vec![1, 0]);
        Ok(self.g.mm(x, w))
    }

    fn name(&mut self, id: NodeId, n: &str) {
        self.g.node_mut(id).name = Some(n.into());
    }

    /// SwiGLU: `linear_out(silu(gate) · up)`, `gate`/`up` the halves of
    /// `linear_in`.
    fn swiglu(&mut self, x: NodeId, pre: &str) -> Result<NodeId> {
        let gi = format!("{pre}.gating.linear_in.weight");
        let ffn = self.shapes[&gi][0] / 2;
        let gate = self.rows_linear(x, &gi, 0, ffn)?;
        let up = self.rows_linear(x, &gi, ffn, ffn)?;
        let gate = self.g.silu(gate);
        let h = self.g.mul(gate, up);
        let go = self.pt(&format!("{pre}.gating.linear_out.weight"))?;
        Ok(self.g.mm(h, go))
    }
}

/// Half-split RoPE (`rlx_lm::apply_rope`) on `x` `[1, 1, nh, hd]`.
fn rope(g: &mut Graph, x: NodeId, half: usize, cos: NodeId, sin: NodeId) -> NodeId {
    let x1 = g.narrow_(x, 3, 0, half);
    let x2 = g.narrow_(x, 3, half, half);
    let a = g.mul(x1, cos);
    let b = g.mul(x2, sin);
    let c = g.mul(x2, cos);
    let d = g.mul(x1, sin);
    let r1 = g.sub(a, b);
    let r2 = g.add(c, d);
    g.concat_(vec![r1, r2], 3)
}

/// `embed`: `text` `[1]` i32 and `audio` `[n]` i32 (the delayed codes) →
/// `inputs_embeds` `[1, 1, d]`, their embeddings summed.
fn embed(d: usize, books: usize, shapes: &HashMap<String, Vec<usize>>) -> Result<Graph> {
    let mut b = B::new("embed", shapes);
    let text = b.g.input("text", Shape::new(&[1], DType::I32));
    let audio = b.g.input("audio", Shape::new(&[books], DType::I32));
    let t = b.p("text_emb.weight")?;
    let mut x = b.g.gather_(t, text, 0);
    for q in 0..books {
        let table = b.p(&format!("emb.{q}.weight"))?;
        let at = b.g.narrow_(audio, 0, q, 1);
        let e = b.g.gather_(table, at, 0);
        x = b.g.add(x, e);
    }
    let x = b.g.reshape_(x, vec![1, 1, d as i64]);
    b.name(x, "inputs_embeds");
    b.g.set_outputs(vec![x]);
    Ok(b.g)
}

/// `step` at `cap` cache rows: `inputs_embeds` `[1, 1, d]`, `position`,
/// `rope_cos` / `rope_sin` `[1, hd / 2]`, `mask` `[1, cap + 1]`,
/// `past_k_{l}` / `past_v_{l}` `[1, cap, nh, hd]` → `logits`, `hidden` (the
/// residual stream before `out_norm`) and each cache with the row written at
/// `position`.
fn step(lm: &LmConfig, shapes: &HashMap<String, Vec<usize>>, cap: usize) -> Result<Graph> {
    let t = &lm.transformer;
    let (d, nh, layers) = (t.d_model, t.num_heads, t.num_layers);
    let hd = d / nh;
    let half = hd / 2;
    let mut b = B::new("step", shapes);
    let mut x = b.g.input("inputs_embeds", f32s(&[1, 1, d]));
    let pos = b.g.input("position", f32s(&[1]));
    let cos = b.g.input("rope_cos", f32s(&[1, half]));
    let sin = b.g.input("rope_sin", f32s(&[1, half]));
    let cos = b.g.reshape_(cos, vec![1, 1, 1, half as i64]);
    let sin = b.g.reshape_(sin, vec![1, 1, 1, half as i64]);
    let mask = b.g.input("mask", f32s(&[1, cap + 1]));
    let heads = vec![1i64, 1, nh as i64, hd as i64];
    let mut outs = Vec::new();
    for l in 0..layers {
        let pre = format!("transformer.layers.{l}");
        let n1 = b.rms(x, &format!("{pre}.norm1.alpha"), d)?;
        let inp = format!("{pre}.self_attn.in_proj_weight");
        let q = b.rows_linear(n1, &inp, 0, d)?;
        let k = b.rows_linear(n1, &inp, d, d)?;
        let v = b.rows_linear(n1, &inp, 2 * d, d)?;
        let q = b.g.reshape_(q, heads.clone());
        let k = b.g.reshape_(k, heads.clone());
        let v = b.g.reshape_(v, heads.clone());
        let q = rope(&mut b.g, q, half, cos, sin);
        let k = rope(&mut b.g, k, half, cos, sin);
        let kv_shape = f32s(&[1, cap, nh, hd]);
        let pk = b.g.input(format!("past_k_{l}"), kv_shape.clone());
        let pv = b.g.input(format!("past_v_{l}"), kv_shape.clone());
        let kf = b.g.concat_(vec![pk, k], 1);
        let vf = b.g.concat_(vec![pv, v], 1);
        let a = b.g.attention_(q, kf, vf, mask, nh, hd);
        let a = b.g.reshape_(a, vec![1, 1, d as i64]);
        let ow = b.pt(&format!("{pre}.self_attn.out_proj.weight"))?;
        let a = b.g.mm(a, ow);
        x = b.g.add(x, a);
        for (kind, row, cache) in [("k", k, pk), ("v", v, pv)] {
            let next = b.g.add_node(
                Op::DynamicUpdateSlice { axis: 1 },
                vec![cache, row, pos],
                kv_shape.clone(),
            );
            b.name(next, &format!("past_{kind}_{l}.next"));
            outs.push(next);
        }
        let n2 = b.rms(x, &format!("{pre}.norm2.alpha"), d)?;
        let m = b.swiglu(n2, &pre)?;
        x = b.g.add(x, m);
    }
    let xn = b.rms(x, "out_norm.alpha", d)?;
    let tl = b.pt("text_linear.weight")?;
    let logits = b.g.mm(xn, tl);
    b.name(logits, "logits");
    // The residual stream, kept apart from what the head reads.
    let one = b.g.add_node(
        Op::Constant {
            data: 1f32.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    );
    let hidden = b.g.mul(x, one);
    b.name(hidden, "hidden");
    let mut all = vec![logits, hidden];
    all.extend(outs);
    b.g.set_outputs(all);
    Ok(b.g)
}

/// `dep_{k}`: `hidden` `[1, 1, d]`, `token` `[1]` i32 (the text token, or
/// the previous slice's code), the earlier slices' `k_{l}` / `v_{l}` `[1,
/// k, nh, hd]` (none at `k = 0`) → `logits` `[1, card]` and each with this
/// slice's row (`k_{l}.next`, `v_{l}.next`).
fn dep(lm: &LmConfig, shapes: &HashMap<String, Vec<usize>>, k: usize) -> Result<Graph> {
    let dc = lm.depformer.as_ref().context("no DepFormer")?;
    let t = &dc.transformer;
    let (dd, nh) = (t.d_model, t.num_heads);
    let hd = dd / nh;
    let d = lm.transformer.d_model;
    let pre = format!("depformer.{k}");
    let mut b = B::new(&format!("dep_{k}"), shapes);
    let hidden = b.g.input("hidden", f32s(&[1, 1, d]));
    let hidden = b.g.reshape_(hidden, vec![1, d as i64]);
    let token = b.g.input("token", Shape::new(&[1], DType::I32));
    let w_in = b.pt(&format!("{pre}.linear_in.weight"))?;
    let x = b.g.mm(hidden, w_in);
    let table = b.p(&format!("{pre}.emb.weight"))?;
    let e = b.g.gather_(table, token, 0);
    let mut x = b.g.add(x, e);
    let heads = vec![1i64, 1, nh as i64, hd as i64];
    let mut outs = Vec::new();
    for l in 0..t.num_layers {
        let lp = format!("{pre}.transformer.layers.{l}");
        let n1 = b.rms(x, &format!("{lp}.norm1.alpha"), dd)?;
        let inp = format!("{lp}.self_attn.in_proj_weight");
        let q = b.rows_linear(n1, &inp, 0, dd)?;
        let kk = b.rows_linear(n1, &inp, dd, dd)?;
        let v = b.rows_linear(n1, &inp, 2 * dd, dd)?;
        let q = b.g.reshape_(q, heads.clone());
        let kk = b.g.reshape_(kk, heads.clone());
        let v = b.g.reshape_(v, heads.clone());
        let (kf, vf) = if k == 0 {
            (kk, v)
        } else {
            let pk = b.g.input(format!("k_{l}"), f32s(&[1, k, nh, hd]));
            let pv = b.g.input(format!("v_{l}"), f32s(&[1, k, nh, hd]));
            (b.g.concat_(vec![pk, kk], 1), b.g.concat_(vec![pv, v], 1))
        };
        let one = b.g.add_node(
            Op::Constant {
                data: 1f32.to_le_bytes().to_vec(),
            },
            vec![],
            f32s(&[1]),
        );
        let ko = b.g.mul(kf, one);
        let vo = b.g.mul(vf, one);
        b.name(ko, &format!("k_{l}.next"));
        b.name(vo, &format!("v_{l}.next"));
        outs.push(ko);
        outs.push(vo);
        // One query over every row: the causal mask admits them all.
        let a = b.g.attention_kind(
            q,
            kf,
            vf,
            nh,
            hd,
            rlx_ir::op::MaskKind::None,
            f32s(&[1, 1, nh, hd]),
        );
        let a = b.g.reshape_(a, vec![1, dd as i64]);
        let ow = b.pt(&format!("{lp}.self_attn.out_proj.weight"))?;
        let a = b.g.mm(a, ow);
        x = b.g.add(x, a);
        let n2 = b.rms(x, &format!("{lp}.norm2.alpha"), dd)?;
        let m = b.swiglu(n2, &lp)?;
        x = b.g.add(x, m);
    }
    let lo = b.pt(&format!("{pre}.linear_out.weight"))?;
    let logits = b.g.mm(x, lo);
    b.name(logits, "logits");
    let mut all = vec![logits];
    all.extend(outs);
    b.g.set_outputs(all);
    Ok(b.g)
}

fn shapes_of(path: &Path) -> Result<HashMap<String, Vec<usize>>> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    Ok(st
        .tensors()
        .into_iter()
        .map(|(n, v)| (n, v.shape().to_vec()))
        .collect())
}

/// Export the Moshiko directory `dir` (`model.safetensors`,
/// `tokenizer_spm_32k_3.model`) with Mimi from `RLX_MIMI_DIR` as the
/// package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    export_with(dir, &rlx_mimi::default_mimi_dir(), out, opts)
}

/// [`export_rlxp`] with the Mimi directory `mimi_dir` (Hugging Face
/// `kyutai/mimi`) given.
pub fn export_with(
    dir: &Path,
    mimi_dir: &Path,
    out: &Path,
    opts: &ExportOptions,
) -> Result<ExportReport> {
    let gen_cfg = GenerateConfig::v0_1_one_way();
    let lm = LmConfig::v0_1();
    let st = dir.join("model.safetensors");
    let shapes = shapes_of(&st)?;
    let books = gen_cfg.total_audio_codebooks();
    let slices = lm.depformer.as_ref().context("no DepFormer")?.num_slices;
    let mut pkg = Package::new(
        "moshiko-one-way",
        &opts.version,
        "moshi",
        "text-to-speech",
        "rlx-moshi",
    );
    pkg.model["license"] = json!("cc-by-4.0");
    // The checkpoint's bf16 weights kept bf16 (`reference_named` records
    // them): half the memory, the values the runner reads.
    pkg.weights_as_stored();
    pkg.function(embed(lm.transformer.d_model, books, &shapes)?);
    pkg.function(symbolic_in(CAP, (37, 53, 20), |c| step(&lm, &shapes, c))?);
    for k in 0..slices {
        pkg.function(dep(&lm, &shapes, k)?);
    }
    let (graphs, held, mimi_cfg, _) = rlx_mimi::export::mimi_functions(mimi_dir)
        .with_context(|| format!("Mimi from {}", mimi_dir.display()))?;
    for mut g in graphs {
        g.name = format!("mimi_{}", g.name);
        pkg.function(g);
    }
    pkg.dims = vec![
        ("frames".into(), 0),
        ("codebooks".into(), 1),
        ("cap".into(), CAP),
    ];
    let dims = pkg.param_dims();
    let mut count = 0;
    for n in dims.keys() {
        if held.contains_key(n) {
            continue;
        }
        if !shapes.contains_key(n) {
            bail!("param {n} is neither a checkpoint tensor nor held");
        }
        count += 1;
    }
    pkg.reference_named("safetensors", &st, out, count)?;
    let mut names: Vec<&String> = held.keys().collect();
    names.sort();
    for n in names {
        let d = dims.get(n).cloned().unwrap_or_else(|| vec![held[n].len()]);
        pkg.hold_f32(n.clone(), d, &held[n]);
    }

    let spm = dir.join("tokenizer_spm_32k_3.model");
    pkg.sidecar(
        "tokenizer.json",
        "application/json",
        rlx_core::export::spm::tokenizer_json(&std::fs::read(&spm)?)?.into_bytes(),
    );
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))
        .context("build Moshi's program")?;
    let defaults = crate::session::GenerationConfig::default();
    let t = &lm.transformer;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize", "run"],
        "text-to-speech",
        json!({
            "tokenizer": "sidecar:tokenizer.json",
            "layers": t.num_layers,
            "heads": t.num_heads,
            "head_dim": t.d_model / t.num_heads,
            "rope_theta": t.max_period,
            "dep_layers": lm.depformer.as_ref().map_or(0, |d| d.transformer.num_layers),
            "slices": slices,
            "books": books,
            "generated": gen_cfg.generated_audio_codebooks,
            "acoustic_delay": gen_cfg.acoustic_delay,
            "audio_pad": gen_cfg.audio_pad_token(),
            "text_pad": gen_cfg.text_pad_token,
            "text_start": gen_cfg.text_start_token,
            "max_steps": defaults.max_steps,
            "text_temperature": defaults.text_temperature,
            "text_top_k": defaults.text_top_k,
            "audio_temperature": defaults.audio_temperature,
            "audio_top_k": defaults.audio_top_k,
            "text_seed": defaults.text_seed,
            "audio_seed": defaults.audio_seed,
            "mimi_codebooks": defaults.mimi_codebooks,
            "sample_rate": mimi_cfg.sampling_rate,
        }),
    );
    pkg.metadata("sample_rate", json!(mimi_cfg.sampling_rate));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family moshi`).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("moshi", "text-to-speech", export_rlxp));
}
