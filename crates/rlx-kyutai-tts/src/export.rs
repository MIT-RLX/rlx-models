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

//! Export Kyutai TTS (`kyutai/tts-1.6b-en_fr`) as an `.rlxp` package rlx
//! runs alone: text in, 24 kHz speech in a voice out.
//!
//! Its graphs read the checkpoint's tensors by their own names (referenced,
//! none copied): `embed` (the multiplexed text token, the delayed audio
//! codes that are not the zero token, and the conditioning sum for the CFG
//! bin); `speaker` (a voice's `speaker_wavs` projected, the learnt padding
//! to five slots, the sinusoidal positions: the cross-attention context);
//! `step` (the temporal transformer one row at `position`, under the mask
//! the program gives, its cross-attention over the context; the text
//! logits, the normed hidden state, each cache with the row written in),
//! symbolic in the caches' rows; `dep_{k}` (the DepFormer's step for
//! codebook `k`: its weights the schedule's, the codebook's embedding of the
//! previous one's token, attention over the frame's earlier codebooks; the
//! codebook's logits); and Mimi's `mimi_lookup` and `mimi_decode`
//! ([`rlx_mimi::export::mimi_functions`]). The program (`program/`) is the
//! crate's own `generate.rs`, `state_machine.rs`, `delays.rs` and
//! `sampling.rs` (the runner's `StdRng` stream) over those graphs.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::export::graph::symbolic_in;
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use serde_json::json;

use crate::config::KyutaiTtsConfig;
use crate::model::{MAX_SPEAKER_CROSS_FRAMES, SPEAKER_FRAMES_PER_SLOT};

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

    /// `[out, in]` stored, `[in, out]` read.
    fn pt(&mut self, key: &str) -> Result<NodeId> {
        let w = self.p(key)?;
        Ok(self.g.transpose_(w, vec![1, 0]))
    }

    /// A `[1, 1, d]` (or `[d]`) norm scale as `[d]`.
    fn alpha(&mut self, key: &str, d: usize) -> Result<NodeId> {
        let a = self.p(key)?;
        Ok(self.g.reshape_(a, vec![d as i64]))
    }

    fn constant(&mut self, v: f32) -> NodeId {
        self.g.add_node(
            Op::Constant {
                data: v.to_le_bytes().to_vec(),
            },
            vec![],
            f32s(&[1]),
        )
    }

    fn rms(&mut self, x: NodeId, key: &str, d: usize) -> Result<NodeId> {
        let a = self.alpha(key, d)?;
        let shape = self.g.node(x).shape.clone();
        let zero = self.constant(0.0);
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

    /// `x · Wᵀ` for the rows `[start, start + len)` of the stored `[rows,
    /// in]` weight `key`.
    fn rows_linear(&mut self, x: NodeId, key: &str, start: usize, len: usize) -> Result<NodeId> {
        let w = self.p(key)?;
        let w = self.g.narrow_(w, 0, start, len);
        let w = self.g.transpose_(w, vec![1, 0]);
        Ok(self.g.mm(x, w))
    }

    fn name(&mut self, id: NodeId, n: &str) {
        self.g.node_mut(id).name = Some(n.into());
    }
}

/// Moshi's interleaved RoPE on `x` `[1, 1, nh, hd]` (pairs `(2i, 2i+1)`).
fn rope(g: &mut Graph, x: NodeId, nh: usize, half: usize, cos: NodeId, sin: NodeId) -> NodeId {
    let x5 = g.reshape_(x, vec![1, 1, nh as i64, half as i64, 2]);
    let re = g.narrow_(x5, 4, 0, 1);
    let im = g.narrow_(x5, 4, 1, 1);
    let re = g.reshape_(re, vec![1, 1, nh as i64, half as i64]);
    let im = g.reshape_(im, vec![1, 1, nh as i64, half as i64]);
    let a = g.mul(re, cos);
    let b = g.mul(im, sin);
    let re2 = g.sub(a, b);
    let c = g.mul(re, sin);
    let d = g.mul(im, cos);
    let im2 = g.add(c, d);
    let re2 = g.reshape_(re2, vec![1, 1, nh as i64, half as i64, 1]);
    let im2 = g.reshape_(im2, vec![1, 1, nh as i64, half as i64, 1]);
    let x = g.concat_(vec![re2, im2], 4);
    g.reshape_(x, vec![1, 1, nh as i64, (2 * half) as i64])
}

/// `embed`: `text` `[2]` i32 (the multiplexed token's main and second ids,
/// each `card + 1`-strided), `text_on` `[2]` (1 where each is present),
/// `audio` `[n_q]` i32 and `audio_on` `[n_q]` (0 for the zero token), `cfg`
/// `[1]` i32 (the CFG bin) → `inputs_embeds` `[1, 1, d]`.
fn embed(cfg: &KyutaiTtsConfig, shapes: &HashMap<String, Vec<usize>>) -> Result<Graph> {
    let d = cfg.dim;
    let mut b = B::new("embed", shapes);
    let text = b.g.input("text", Shape::new(&[2], DType::I32));
    let text_on = b.g.input("text_on", f32s(&[2]));
    let audio = b.g.input("audio", Shape::new(&[cfg.n_q], DType::I32));
    let audio_on = b.g.input("audio_on", f32s(&[cfg.n_q]));
    let bin = b.g.input("cfg", Shape::new(&[1], DType::I32));
    // The text token: its main id through `out1`, its second through `out2`.
    let table = b.p("text_emb.weight")?;
    let rows = b.g.gather_(table, text, 0);
    let main = b.g.narrow_(rows, 0, 0, 1);
    let second = b.g.narrow_(rows, 0, 1, 1);
    let o1 = b.pt("text_emb.out1.weight")?;
    let o2 = b.pt("text_emb.out2.weight")?;
    let main = b.g.mm(main, o1);
    let second = b.g.mm(second, o2);
    let on_main = b.g.narrow_(text_on, 0, 0, 1);
    let on_second = b.g.narrow_(text_on, 0, 1, 1);
    let main = b.g.mul(main, on_main);
    let second = b.g.mul(second, on_second);
    let mut x = b.g.add(main, second);
    for q in 0..cfg.n_q {
        let t = b.p(&format!("emb.{q}.weight"))?;
        let at = b.g.narrow_(audio, 0, q, 1);
        let e = b.g.gather_(t, at, 0);
        let on = b.g.narrow_(audio_on, 0, q, 1);
        let e = b.g.mul(e, on);
        x = b.g.add(x, e);
    }
    let sums = b.g.param("cond.sum", f32s(&[cfg_bins(cfg), d]));
    let s = b.g.gather_(sums, bin, 0);
    let x = b.g.add(x, s);
    let x = b.g.reshape_(x, vec![1, 1, d as i64]);
    b.name(x, "inputs_embeds");
    b.g.set_outputs(vec![x]);
    Ok(b.g)
}

fn cfg_bins(cfg: &KyutaiTtsConfig) -> usize {
    match cfg.conditioners.get("cfg") {
        Some(crate::config::ConditionerKind::Lut { lut }) => lut.possible_values.len(),
        _ => 1,
    }
}

fn cfg_values(cfg: &KyutaiTtsConfig) -> Vec<String> {
    match cfg.conditioners.get("cfg") {
        Some(crate::config::ConditionerKind::Lut { lut }) => lut.possible_values.clone(),
        _ => vec!["1.0".into()],
    }
}

/// `speaker`: `speaker_wavs` `[125, 512]` (a voice, zero-padded) →
/// `cross_ctx` `[1, 625, d]`: its projection, the learnt padding for the
/// other four slots, the sinusoidal positions (`cond.positions`).
fn speaker(cfg: &KyutaiTtsConfig, shapes: &HashMap<String, Vec<usize>>) -> Result<Graph> {
    let d = cfg.dim;
    let pfx = "condition_provider.conditioners.speaker_wavs.";
    let mut b = B::new("speaker", shapes);
    let wavs =
        b.g.input("speaker_wavs", f32s(&[SPEAKER_FRAMES_PER_SLOT, 512]));
    let w = b.pt(&format!("{pfx}output_proj.weight"))?;
    let proj = b.g.mm(wavs, w);
    let pad = b.p(&format!("{pfx}learnt_padding"))?;
    let pad = b.g.reshape_(pad, vec![1, d as i64]);
    let rest = MAX_SPEAKER_CROSS_FRAMES - SPEAKER_FRAMES_PER_SLOT;
    let pad = b.g.add_node(
        Op::Expand {
            target_shape: vec![rest as i64, d as i64],
        },
        vec![pad],
        f32s(&[rest, d]),
    );
    let ctx = b.g.concat_(vec![proj, pad], 0);
    let ctx = if cfg.fuser.cross_attention_pos_emb {
        let pe =
            b.g.param("cond.positions", f32s(&[MAX_SPEAKER_CROSS_FRAMES, d]));
        b.g.add(ctx, pe)
    } else {
        ctx
    };
    let ctx =
        b.g.reshape_(ctx, vec![1, MAX_SPEAKER_CROSS_FRAMES as i64, d as i64]);
    b.name(ctx, "cross_ctx");
    b.g.set_outputs(vec![ctx]);
    Ok(b.g)
}

/// `step` at `cap` cache rows: `inputs_embeds` `[1, 1, d]`, `position`
/// `[1]`, `rope_cos` / `rope_sin` `[1, hd / 2]`, `mask` `[1, cap + 1]` (1
/// on the cached rows and the last, the row's own), `cross_ctx` `[1, 625,
/// d]`, `cross_mask` `[1, 625]`, `past_k_{l}` / `past_v_{l}` `[1, cap, nh,
/// hd]` → `logits` `[1, 1, text_card]`, `hidden` `[1, 1, d]` and each cache
/// with the row written at `position` (`past_k_{l}.next`, …).
fn step(cfg: &KyutaiTtsConfig, shapes: &HashMap<String, Vec<usize>>, cap: usize) -> Result<Graph> {
    let (d, nh, layers) = (cfg.dim, cfg.num_heads, cfg.num_layers);
    let hd = d / nh;
    let half = hd / 2;
    let t_cross = MAX_SPEAKER_CROSS_FRAMES;
    let mut b = B::new("step", shapes);
    let mut x = b.g.input("inputs_embeds", f32s(&[1, 1, d]));
    let pos = b.g.input("position", f32s(&[1]));
    let cos = b.g.input("rope_cos", f32s(&[1, half]));
    let sin = b.g.input("rope_sin", f32s(&[1, half]));
    let cos = b.g.reshape_(cos, vec![1, 1, 1, half as i64]);
    let sin = b.g.reshape_(sin, vec![1, 1, 1, half as i64]);
    let mask = b.g.input("mask", f32s(&[1, cap + 1]));
    let ctx = b.g.input("cross_ctx", f32s(&[1, t_cross, d]));
    let cmask = b.g.input("cross_mask", f32s(&[1, t_cross]));
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
        let q = rope(&mut b.g, q, nh, half, cos, sin);
        let k = rope(&mut b.g, k, nh, half, cos, sin);
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
        for (kind, row) in [("k", k), ("v", v)] {
            let cache = if kind == "k" { pk } else { pv };
            let next = b.g.add_node(
                Op::DynamicUpdateSlice { axis: 1 },
                vec![cache, row, pos],
                kv_shape.clone(),
            );
            b.name(next, &format!("past_{kind}_{l}.next"));
            outs.push(next);
        }
        // Cross-attention over the conditioning.
        let nw = b.p(&format!("{pre}.norm_cross.weight"))?;
        let nb = b.p(&format!("{pre}.norm_cross.bias"))?;
        let n = b.g.ln(x, nw, nb, EPS);
        let cin = format!("{pre}.cross_attention.in_proj_weight");
        let n2 = b.g.reshape_(n, vec![1, d as i64]);
        let cq = b.rows_linear(n2, &cin, 0, d)?;
        let cq = b.g.reshape_(cq, heads.clone());
        let c2 = b.g.reshape_(ctx, vec![t_cross as i64, d as i64]);
        let ck = b.rows_linear(c2, &cin, d, d)?;
        let cv = b.rows_linear(c2, &cin, 2 * d, d)?;
        let ch = vec![1i64, t_cross as i64, nh as i64, hd as i64];
        let ck = b.g.reshape_(ck, ch.clone());
        let cv = b.g.reshape_(cv, ch);
        let ca = b.g.attention_(cq, ck, cv, cmask, nh, hd);
        let ca = b.g.reshape_(ca, vec![1, 1, d as i64]);
        let co = b.pt(&format!("{pre}.cross_attention.out_proj.weight"))?;
        let ca = b.g.mm(ca, co);
        x = b.g.add(x, ca);
        // SwiGLU.
        let n2 = b.rms(x, &format!("{pre}.norm2.alpha"), d)?;
        let gi = format!("{pre}.gating.linear_in.weight");
        let ffn = shapes[&gi][0] / 2;
        let gate = b.rows_linear(n2, &gi, 0, ffn)?;
        let up = b.rows_linear(n2, &gi, ffn, ffn)?;
        let gate = b.g.silu(gate);
        let h = b.g.mul(gate, up);
        let go = b.pt(&format!("{pre}.gating.linear_out.weight"))?;
        let m = b.g.mm(h, go);
        x = b.g.add(x, m);
    }
    let xn = b.rms(x, "out_norm.alpha", d)?;
    let tl = b.pt("text_linear.weight")?;
    let logits = b.g.mm(xn, tl);
    b.name(logits, "logits");
    b.name(xn, "hidden");
    let mut all = vec![logits, xn];
    all.extend(outs);
    b.g.set_outputs(all);
    Ok(b.g)
}

/// `dep_{k}`: `hidden` `[1, 1, d]`, the previous codebook's token (`k = 0`:
/// the text token's demultiplexed ids `text` `[2]` i32 and `text_on` `[2]`;
/// else `token` `[1]` i32), the frame's earlier codebooks' keys and values
/// `k_{l}` / `v_{l}` `[nh, k, hd]` (none at `k = 0`) → `logits` `[1,
/// card]` and each with this codebook's row (`k_{l}.next`, `v_{l}.next`).
fn dep(cfg: &KyutaiTtsConfig, shapes: &HashMap<String, Vec<usize>>, k: usize) -> Result<Graph> {
    let df = &cfg.depformer;
    let (dd, nh) = (df.dim, df.num_heads);
    let hd = dd / nh;
    let head = df.weights_per_step_schedule[k];
    let sets = df
        .weights_per_step_schedule
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        + 1;
    let mut b = B::new(&format!("dep_{k}"), shapes);
    let hidden = b.g.input("hidden", f32s(&[1, 1, cfg.dim]));
    let hidden = b.g.reshape_(hidden, vec![1, cfg.dim as i64]);
    let w_in = b.pt(&format!("depformer_in.{head}.weight"))?;
    let mut x = b.g.mm(hidden, w_in);
    let emb = if k == 0 {
        let text = b.g.input("text", Shape::new(&[2], DType::I32));
        let on = b.g.input("text_on", f32s(&[2]));
        let t = b.p("depformer_text_emb.weight")?;
        let rows = b.g.gather_(t, text, 0);
        let l = b.g.narrow_(rows, 0, 0, 1);
        let r = b.g.narrow_(rows, 0, 1, 1);
        let o1 = b.pt("depformer_text_emb.out1.weight")?;
        let o2 = b.pt("depformer_text_emb.out2.weight")?;
        let l = b.g.mm(l, o1);
        let r = b.g.mm(r, o2);
        let l_on = b.g.narrow_(on, 0, 0, 1);
        let r_on = b.g.narrow_(on, 0, 1, 1);
        let l = b.g.mul(l, l_on);
        let r = b.g.mul(r, r_on);
        b.g.add(l, r)
    } else {
        let token = b.g.input("token", Shape::new(&[1], DType::I32));
        let a = b.p(&format!("depformer_emb.{}.weight", k - 1))?;
        let row = b.g.gather_(a, token, 0);
        let key = format!("depformer_emb.{}.low_rank.weight", k - 1);
        // `[dim, rank]` stored: `row · Wᵀ`.
        let w = b.pt(&key)?;
        b.g.mm(row, w)
    };
    x = b.g.add(x, emb);
    let scale = b.constant(1.0 / (hd as f32).sqrt());
    let mut outs = Vec::new();
    for l in 0..df.num_layers {
        let pre = format!("depformer.layers.{l}");
        let n1 = b.rms(x, &format!("{pre}.norm1.alpha"), dd)?;
        let inp = format!("{pre}.self_attn.in_proj_weight");
        let base = head * 3 * dd;
        let q = b.rows_linear(n1, &inp, base, dd)?;
        let kk = b.rows_linear(n1, &inp, base + dd, dd)?;
        let v = b.rows_linear(n1, &inp, base + 2 * dd, dd)?;
        let to_heads = |g: &mut Graph, t: NodeId| g.reshape_(t, vec![nh as i64, 1, hd as i64]);
        let q = to_heads(&mut b.g, q);
        let kk = to_heads(&mut b.g, kk);
        let v = to_heads(&mut b.g, v);
        let (kf, vf) = if k == 0 {
            (kk, v)
        } else {
            let pk = b.g.input(format!("k_{l}"), f32s(&[nh, k, hd]));
            let pv = b.g.input(format!("v_{l}"), f32s(&[nh, k, hd]));
            (b.g.concat_(vec![pk, kk], 1), b.g.concat_(vec![pv, v], 1))
        };
        b.name(kf, &format!("k_{l}.next"));
        b.name(vf, &format!("v_{l}.next"));
        outs.push(kf);
        outs.push(vf);
        let kt = b.g.transpose_(kf, vec![0, 2, 1]);
        let s = b.g.mm(q, kt);
        let s = b.g.mul(s, scale);
        let p = b.g.sm(s, -1);
        let a = b.g.mm(p, vf);
        let a = b.g.reshape_(a, vec![1, dd as i64]);
        let out = format!("{pre}.self_attn.out_proj.weight");
        if shapes[&out][0] != sets * dd {
            bail!("{out} is {:?}: not {sets} stacked steps", shapes[&out]);
        }
        let a = b.rows_linear(a, &out, head * dd, dd)?;
        x = b.g.add(x, a);
        let n2 = b.rms(x, &format!("{pre}.norm2.alpha"), dd)?;
        let gi = format!("{pre}.gating.{head}.linear_in.weight");
        let ffn = shapes[&gi][0] / 2;
        let gate = b.rows_linear(n2, &gi, 0, ffn)?;
        let up = b.rows_linear(n2, &gi, ffn, ffn)?;
        let gate = b.g.silu(gate);
        let h = b.g.mul(gate, up);
        let go = b.pt(&format!("{pre}.gating.{head}.linear_out.weight"))?;
        let m = b.g.mm(h, go);
        x = b.g.add(x, m);
    }
    let lw = b.pt(&format!("linears.{k}.weight"))?;
    let logits = b.g.mm(x, lw);
    b.name(logits, "logits");
    let mut all = vec![logits];
    all.extend(outs);
    b.g.set_outputs(all);
    Ok(b.g)
}

/// Each tensor's shape, from the checkpoint's header.
fn shapes_of(path: &Path) -> Result<HashMap<String, Vec<usize>>> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    Ok(st
        .tensors()
        .into_iter()
        .map(|(n, v)| (n, v.shape().to_vec()))
        .collect())
}

/// The conditioning sums, one row per CFG bin: `control`'s projection plus
/// the bin's `cfg` projection (the fuser's order).
fn cond_sums(cfg: &KyutaiTtsConfig, weights: &crate::weights::WeightMap) -> Result<Vec<f32>> {
    let bundle = crate::model::ConditionerBundle::load(cfg, weights)?;
    let mut out = Vec::new();
    for v in cfg_values(cfg) {
        let fused = bundle.fused(cfg, &v, None)?;
        let s = fused.sum.vector.context("no conditioning sum")?;
        out.extend(s.iter());
    }
    Ok(out)
}

/// Export the Kyutai TTS directory `dir` (`dsm_tts_*.safetensors`,
/// `tokenizer_spm_*.model`, `config.json`) with the Mimi directory
/// `RLX_MIMI_DIR` and the voice found under `RLX_KYUTAI_TTS_VOICES_DIR`
/// as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let voice = crate::download::voice_embedding_path(
        &crate::download::default_voices_dir(),
        crate::checkpoint::KyutaiTtsCheckpoint::V1_6bEnFr,
        crate::download::DEFAULT_VOICE_NAME,
    );
    export_with(dir, &crate::download::default_mimi_dir(), &voice, out, opts)
}

/// [`export_rlxp`] with the Mimi directory `mimi_dir` (Hugging Face
/// `kyutai/mimi`) and the default voice `voice` (a `kyutai/tts-voices`
/// embedding) given.
pub fn export_with(
    dir: &Path,
    mimi_dir: &Path,
    voice: &Path,
    out: &Path,
    opts: &ExportOptions,
) -> Result<ExportReport> {
    let cfg = KyutaiTtsConfig::v1_6b_en_fr();
    let st = dir.join(crate::download::TTS_WEIGHTS_FILE);
    let shapes = shapes_of(&st)?;
    let mut pkg = Package::new(
        "kyutai-tts-1.6b-en-fr",
        &opts.version,
        "kyutai-tts",
        "text-to-speech",
        "rlx-kyutai-tts",
    );
    pkg.model["license"] = json!("cc-by-4.0");
    let mut held: HashMap<String, Vec<f32>> = HashMap::new();

    pkg.function(embed(&cfg, &shapes)?);
    pkg.function(speaker(&cfg, &shapes)?);
    pkg.function(symbolic_in(CAP, (37, 53, 20), |c| step(&cfg, &shapes, c))?);
    for k in 0..cfg.dep_q {
        pkg.function(dep(&cfg, &shapes, k)?);
    }
    {
        let weights = crate::weights::load_weight_map(&st)?;
        held.insert("cond.sum".into(), cond_sums(&cfg, &weights)?);
    }
    let pe = crate::nn::sin_pos_embed(MAX_SPEAKER_CROSS_FRAMES, cfg.dim, cfg.max_period as f32);
    let scale = cfg.fuser.cross_attention_pos_emb_scale;
    held.insert(
        "cond.positions".into(),
        pe.iter().map(|v| v * scale).collect(),
    );

    // Mimi.
    let (graphs, mimi_params, mimi_cfg, _) = rlx_mimi::export::mimi_functions(mimi_dir)
        .with_context(|| format!("Mimi from {}", mimi_dir.display()))?;
    for mut g in graphs {
        g.name = format!("mimi_{}", g.name);
        pkg.function(g);
    }
    held.extend(mimi_params);
    pkg.dims = vec![
        ("frames".into(), 0),
        ("codebooks".into(), 1),
        ("cap".into(), CAP),
    ];

    // The default voice, held, for `speaker`.
    let wavs = crate::model::load_voice_speaker_wavs(voice)
        .with_context(|| format!("the default voice {}", voice.display()))?;
    let mut padded = vec![0f32; SPEAKER_FRAMES_PER_SLOT * 512];
    for t in 0..wavs.nrows().min(SPEAKER_FRAMES_PER_SLOT) {
        for c in 0..wavs.ncols().min(512) {
            padded[t * 512 + c] = wavs[[t, c]];
        }
    }
    held.insert("voice.default".into(), padded);

    // The checkpoint's tensors by their names; the rest held.
    let dims = pkg.param_dims();
    let mut count = 0;
    let mut names: Vec<&String> = dims.keys().collect();
    names.sort();
    for n in names {
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
        let d = match n.as_str() {
            "voice.default" => vec![SPEAKER_FRAMES_PER_SLOT, 512],
            _ => dims.get(n).cloned().unwrap_or_else(|| vec![held[n].len()]),
        };
        pkg.hold_f32(n.clone(), d, &held[n]);
    }

    // The tokenizer, and the program.
    let spm = dir.join(crate::download::SPM_TOKENIZER_FILE);
    pkg.sidecar(
        "tokenizer.json",
        "application/json",
        rlx_core::export::spm::tokenizer_json(&std::fs::read(&spm)?)?.into_bytes(),
    );
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))
        .context("build Kyutai TTS' program")?;
    let defaults = crate::session::GenerationConfig::default();
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize", "run"],
        "text-to-speech",
        json!({
            "tokenizer": "sidecar:tokenizer.json",
            "cfg_values": cfg_values(&cfg),
            "cfg_alpha": defaults.cfg_alpha,
            "max_steps": defaults.max_steps,
            "text_temperature": defaults.text_temperature,
            "audio_temperature": defaults.audio_temperature,
            "seed": defaults.seed,
            "rope_theta": cfg.max_period,
            "sample_rate": mimi_cfg.sampling_rate,
        }),
    );
    pkg.metadata("sample_rate", json!(mimi_cfg.sampling_rate));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family kyutai-tts`).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("kyutai-tts", "text-to-speech", export_rlxp));
}
