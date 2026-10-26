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

//! The FLUX 3 DiT as compiled graphs.
//!
//! Structure (BFL `flux3/dit.py`): each mode — image, reference image,
//! text — first runs its own `depth` blocks, attending only within the
//! mode; then all tokens of a sample run `depth_single` shared blocks over
//! the joint sequence `[image | references | text]`, modulated per token by
//! the token's mode. Every block is the same parallel layout: one
//! LayerNorm-modulated input feeds q/k/v and a SwiGLU MLP, and
//! `x + gate · (attn_out(attn) + mlp_out(silu(g) · u))` closes it. No biases.
//!
//! Text and reference tokens always sit at flow time 0, and their early
//! blocks never see the image, so for the CFG checkpoint their early output
//! is computed once per prompt ([`ModeStack`]). Each step then runs one
//! chain ([`MainChain`]): image embedding and early blocks, the join, the
//! single blocks and the final layer, cut into chunks so the host holds at
//! most one chunk of weights while building.
//!
//! CFG runs the unconditional and conditional samples as a batch of two.
//! Their texts differ in length (each padded to a multiple of 80), so the
//! shorter one is zero-filled to the longer and a key-padding mask removes
//! the filler — the samples never attend to each other in the reference
//! either, so this is exact.

use anyhow::{Context, Result, ensure};
use rlx_flow::{BuiltModel, Emit, ModelFlow, WeightStorage};
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use rlx_runtime::{CompiledGraph, Device};

use crate::config::{Flux3Config, Mode, NORM_EPS};
use crate::embed::Conditioning;
use crate::rope::rope_tables;
use crate::weights::DitSource;

/// One unit of the per-step chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitPart {
    /// `emb_in.image` (graph input `img`).
    Embed,
    /// Early image block `i`.
    Early(usize),
    /// Join `[image | references | text]` into the batched joint stream.
    Join,
    /// Single-stream block `i`.
    Single(usize),
    /// `final_layer.image` over the image tokens.
    Final,
}

/// Split the chain into chunks of at most `blocks` transformer blocks.
pub fn plan_chunks(cfg: &Flux3Config, blocks: usize) -> Vec<Vec<DitPart>> {
    let blocks = blocks.max(1);
    let mut all: Vec<DitPart> = vec![DitPart::Embed];
    all.extend((0..cfg.depth).map(DitPart::Early));
    all.push(DitPart::Join);
    all.extend((0..cfg.depth_single).map(DitPart::Single));
    all.push(DitPart::Final);
    let mut chunks: Vec<Vec<DitPart>> = vec![Vec::new()];
    let mut n = 0;
    for p in all {
        let is_block = matches!(p, DitPart::Early(_) | DitPart::Single(_));
        if is_block && n == blocks {
            chunks.push(Vec::new());
            n = 0;
        }
        n += usize::from(is_block);
        chunks.last_mut().expect("non-empty").push(p);
    }
    chunks
}

/// Sequence geometry the per-step chain is compiled for.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainGeometry {
    /// Samples run together (2 for CFG).
    pub batch: usize,
    /// Latent grid (`H/16 × W/16`).
    pub grid_h: usize,
    pub grid_w: usize,
    /// Reference tokens (0 for text-to-image).
    pub cond_len: usize,
    /// Text tokens per sample, after padding to the longest.
    pub txt_len: usize,
    /// Whether some sample's text is shorter than `txt_len` (needs a mask).
    pub masked: bool,
    /// Position ids of the joint sequence `[image | references | text]`.
    pub ids: Vec<[f32; 4]>,
}

impl ChainGeometry {
    pub fn img_len(&self) -> usize {
        self.grid_h * self.grid_w
    }
    pub fn total(&self) -> usize {
        self.img_len() + self.cond_len + self.txt_len
    }
}

/// Graph-building context shared by the chain and the mode stacks.
struct Dit<'e, 'a> {
    emit: &'e mut Emit<'a>,
    cfg: &'e Flux3Config,
    ln_one: HirNodeId,
    ln_zero: HirNodeId,
    head_zero: HirNodeId,
}

/// Modulation operands of a block, each broadcastable to the stream.
#[derive(Clone, Copy)]
struct Mods {
    shift: HirNodeId,
    scale: HirNodeId,
    gate: HirNodeId,
}

impl<'e, 'a> Dit<'e, 'a> {
    fn new(emit: &'e mut Emit<'a>, cfg: &'e Flux3Config) -> Self {
        let f = DType::F32;
        let (d, hd) = (cfg.hidden, cfg.head_dim);
        let ln_one = emit.synth_param("flux3.ln_one", vec![1.0; d], Shape::new(&[d], f));
        let ln_zero = emit.synth_param("flux3.ln_zero", vec![0.0; d], Shape::new(&[d], f));
        let head_zero = emit.synth_param("flux3.head_zero", vec![0.0; hd], Shape::new(&[hd], f));
        Self {
            emit,
            cfg,
            ln_one,
            ln_zero,
            head_zero,
        }
    }

    fn g(&mut self) -> HirMut<'_> {
        HirMut::new(self.emit.hir())
    }

    fn dims(&mut self, x: HirNodeId) -> Vec<i64> {
        self.g()
            .shape(x)
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect()
    }

    /// `(cos, sin)` tables `[ids.len(), head_dim / 2]`.
    fn rope(&mut self, name: &str, ids: &[[f32; 4]]) -> (HirNodeId, HirNodeId) {
        let (cos, sin) = rope_tables(ids, self.cfg.axes_dim, self.cfg.theta);
        let shape = Shape::new(&[ids.len(), self.cfg.head_dim / 2], DType::F32);
        (
            self.emit
                .synth_param(&format!("{name}.cos"), to_f32(&cos), shape.clone()),
            self.emit
                .synth_param(&format!("{name}.sin"), to_f32(&sin), shape),
        )
    }

    /// `x @ Wᵀ` for `key.weight` (packed when the source quantizes).
    fn linear(&mut self, x: HirNodeId, key: &str) -> Result<HirNodeId> {
        let wk = format!("{key}.weight");
        if let Some(p) = self.emit.weights.take_packed(&wk)? {
            let qkey = format!("{wk}\0q");
            let id = self
                .emit
                .hir()
                .param(&qkey, Shape::new(&[p.w_q.len()], DType::U8));
            self.emit.state.typed_params.push((qkey, p.w_q, DType::U8));
            let s = self.g().shape(x).clone();
            let last = s.rank() - 1;
            let shape = s.with_dim(last, rlx_ir::Dim::Static(p.out_dim));
            return Ok(self.emit.hir().dequant_matmul(
                x,
                id,
                None,
                None,
                p.scheme,
                shape.with_dtype(DType::F32),
            ));
        }
        let w = self.emit.load_matrix(&wk, true)?;
        Ok(self.g().mm(x, w))
    }

    /// `LayerNorm(x) · (1 + scale) + shift` (no affine).
    fn modulate(&mut self, x: HirNodeId, shift: HirNodeId, scale: HirNodeId) -> HirNodeId {
        let (one, zero) = (self.ln_one, self.ln_zero);
        let dims = self.dims(x);
        let mut g = self.g();
        let n = g.ln(x, one, zero, NORM_EPS);
        let scale = g.expand_(scale, dims.clone());
        let shift = g.expand_(shift, dims);
        let ns = g.mul(n, scale);
        let n = g.add(n, ns);
        g.add(n, shift)
    }

    /// Per-head RMSNorm with the (permuted) `scale`.
    fn qk_norm(&mut self, x: HirNodeId, scale_key: &str) -> Result<HirNodeId> {
        let scale = self.emit.load_param(scale_key, false)?;
        let dims = self.dims(x);
        let (h, hd) = (self.cfg.num_heads as i64, self.cfg.head_dim as i64);
        let zero = self.head_zero;
        let mut g = self.g();
        let x4 = g.reshape_(x, vec![dims[0], dims[1], h, hd]);
        let n = g.rms_norm(x4, scale, zero, NORM_EPS);
        Ok(g.reshape_(n, dims))
    }

    /// One block over `x` (`[B, S, D]`), RoPE tables with `S` rows, an
    /// optional key mask `[B, S]`.
    fn block(
        &mut self,
        prefix: &str,
        x: HirNodeId,
        m: Mods,
        rope: (HirNodeId, HirNodeId),
        mask: Option<HirNodeId>,
    ) -> Result<HirNodeId> {
        let (h, hd, mh) = (self.cfg.num_heads, self.cfg.head_dim, self.cfg.mlp_hidden);
        let xm = self.modulate(x, m.shift, m.scale);
        let q = self.linear(xm, &format!("{prefix}.q_proj"))?;
        let k = self.linear(xm, &format!("{prefix}.k_proj"))?;
        let v = self.linear(xm, &format!("{prefix}.v_proj"))?;
        let q = self.qk_norm(q, &format!("{prefix}.norm.query_norm.scale"))?;
        let k = self.qk_norm(k, &format!("{prefix}.norm.key_norm.scale"))?;
        let attn = {
            let (cos, sin) = rope;
            let mut g = self.g();
            let q = g.rope(q, cos, sin, hd);
            let k = g.rope(k, cos, sin, hd);
            let shape = g.shape(v).clone();
            match mask {
                Some(mask) => g.attention_(q, k, v, mask, h, hd),
                None => g.attention_kind(q, k, v, h, hd, MaskKind::None, shape),
            }
        };
        let a = self.linear(attn, &format!("{prefix}.attn_out"))?;
        let gu = self.linear(xm, &format!("{prefix}.mlp_in"))?;
        let act = {
            let mut g = self.g();
            let gate = g.narrow_(gu, 2, 0, mh);
            let up = g.narrow_(gu, 2, mh, mh);
            let s = g.silu(gate);
            g.mul(s, up)
        };
        let y = self.linear(act, &format!("{prefix}.mlp_out"))?;
        let dims = self.dims(x);
        let mut g = self.g();
        let y = g.add(a, y);
        let gate = g.expand_(m.gate, dims);
        let gy = g.mul(gate, y);
        Ok(g.add(x, gy))
    }

    /// `[shift, scale, gate]` rows `row0..row0+3` of a `[n, D]` input, as `[1, 1, D]`.
    fn mod_rows(&mut self, mods: HirNodeId, row0: usize) -> Mods {
        let d = self.cfg.hidden as i64;
        let mut g = self.g();
        let mut r = (0..3).map(|i| {
            let c = g.narrow_(mods, 0, row0 + i, 1);
            g.reshape_(c, vec![1, 1, d])
        });
        Mods {
            shift: r.next().expect("3 rows"),
            scale: r.next().expect("3 rows"),
            gate: r.next().expect("3 rows"),
        }
    }
}

fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

fn wrap_out(dit: &mut Dit<'_, '_>, x: HirNodeId) -> rlx_flow::FlowValue {
    let shape = dit.g().shape(x).clone();
    dit.emit.wrap(x, shape)
}

/// Embed + early blocks of one mode over `n` tokens (batch 1), inputs `x`
/// (`[1, n, in]`) and `mod` (`[3, D]` early modulation), output `[1, n, D]`.
pub fn build_mode_stack(
    cfg: &Flux3Config,
    mode: Mode,
    ids: &[[f32; 4]],
    weights: &mut DitSource<'_>,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let n = ids.len();
    let in_dim = match mode {
        Mode::Text => cfg.context_dim,
        _ => cfg.in_channels,
    };
    let cfg_c = cfg.clone();
    let ids = ids.to_vec();
    ModelFlow::new("flux3_mode_stack")
        .with_weight_storage(WeightStorage::Auto)
        .input("x", Shape::new(&[1, n, in_dim], f))
        .input("mod", Shape::new(&[3, cfg.hidden], f))
        .plugin_named("flux3.mode_stack", move |emit, _| {
            let x = emit.flow_input("x")?.hir_id();
            let mods = emit.flow_input("mod")?.hir_id();
            let mut g = Dit::new(emit, &cfg_c);
            let rope = g.rope("flux3.rope", &ids);
            let m = g.mod_rows(mods, 0);
            let mut x = g.linear(x, mode.embed_key())?;
            for i in 0..cfg_c.depth {
                x = g.block(&format!("{}.{i}", mode.block_prefix()), x, m, rope, None)?;
            }
            Ok(Some(wrap_out(&mut g, x)))
        })
        .output("out")
        .build(weights)
}

/// Build one chunk of the per-step chain.
pub fn build_chain_chunk(
    cfg: &Flux3Config,
    geo: &ChainGeometry,
    parts: &[DitPart],
    weights: &mut DitSource<'_>,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let (b, si, sc, st, d) = (
        geo.batch,
        geo.img_len(),
        geo.cond_len,
        geo.txt_len,
        cfg.hidden,
    );
    let s = geo.total();
    let joined = |p: &DitPart| matches!(p, DitPart::Single(_) | DitPart::Final);
    let has = |want: fn(&DitPart) -> bool| parts.iter().any(want);
    let mut flow = ModelFlow::new("flux3_chain_chunk").with_weight_storage(WeightStorage::Auto);
    match parts.first() {
        Some(DitPart::Embed) => {
            flow = flow.input("img", Shape::new(&[1, si, cfg.in_channels], f));
        }
        Some(p) if joined(p) => flow = flow.input("x", Shape::new(&[b, s, d], f)),
        _ => flow = flow.input("h", Shape::new(&[1, si, d], f)),
    }
    if has(|p| matches!(p, DitPart::Early(_))) {
        flow = flow.input("mod_early", Shape::new(&[3, d], f));
    }
    if has(|p| matches!(p, DitPart::Join)) {
        if sc > 0 {
            flow = flow.input("cond", Shape::new(&[1, sc, d], f));
        }
        flow = flow.input("txt", Shape::new(&[b, st, d], f));
    }
    let singles = has(|p| matches!(p, DitPart::Single(_)));
    if singles {
        flow = flow.input("mod_single", Shape::new(&[9, d], f));
        if geo.masked {
            flow = flow.input("mask", Shape::new(&[b, s], f));
        }
    }
    if has(|p| matches!(p, DitPart::Final)) {
        flow = flow.input("mod_final", Shape::new(&[2, d], f));
    }
    let cfg_c = cfg.clone();
    let geo_c = geo.clone();
    let parts_c = parts.to_vec();
    flow = flow.plugin_named("flux3.chain", move |emit, _| {
        let first = match parts_c.first() {
            Some(DitPart::Embed) => emit.flow_input("img")?.hir_id(),
            Some(p) if joined(p) => emit.flow_input("x")?.hir_id(),
            _ => emit.flow_input("h")?.hir_id(),
        };
        let early_mods = emit.flow_input("mod_early").ok().map(|v| v.hir_id());
        let cond = emit.flow_input("cond").ok().map(|v| v.hir_id());
        let txt = emit.flow_input("txt").ok().map(|v| v.hir_id());
        let single_mods = emit.flow_input("mod_single").ok().map(|v| v.hir_id());
        let mask = emit.flow_input("mask").ok().map(|v| v.hir_id());
        let final_mods = emit.flow_input("mod_final").ok().map(|v| v.hir_id());
        let mut g = Dit::new(emit, &cfg_c);
        let joint = g.rope("flux3.rope", &geo_c.ids);
        let image_rope = {
            let mut h = g.g();
            (h.narrow_(joint.0, 0, 0, si), h.narrow_(joint.1, 0, 0, si))
        };
        let early = early_mods.map(|m| g.mod_rows(m, 0));
        // Per-token single-stream modulation `[1, S, D]`: shared by every
        // single block, so built once per chunk.
        let single = match single_mods {
            Some(m) => {
                let per_mode = [g.mod_rows(m, 0), g.mod_rows(m, 3), g.mod_rows(m, 6)];
                let d = d as i64;
                let lens = [si, sc, st];
                let mut h = g.g();
                let mut tok = |pick: fn(&Mods) -> HirNodeId| {
                    let pieces: Vec<HirNodeId> = per_mode
                        .iter()
                        .zip(lens)
                        .filter(|(_, n)| *n > 0)
                        .map(|(mm, n)| h.expand_(pick(mm), vec![1, n as i64, d]))
                        .collect();
                    h.concat_(pieces, 1)
                };
                Some(Mods {
                    shift: tok(|m| m.shift),
                    scale: tok(|m| m.scale),
                    gate: tok(|m| m.gate),
                })
            }
            None => None,
        };
        let mut x = first;
        for part in &parts_c {
            x = match *part {
                DitPart::Embed => g.linear(x, Mode::Image.embed_key())?,
                DitPart::Early(i) => g.block(
                    &format!("{}.{i}", Mode::Image.block_prefix()),
                    x,
                    early.context("early modulation")?,
                    image_rope,
                    None,
                )?,
                DitPart::Join => {
                    let bl = b as i64;
                    let d = d as i64;
                    let mut h = g.g();
                    let mut pieces = vec![h.expand_(x, vec![bl, si as i64, d])];
                    if let Some(c) = cond {
                        pieces.push(h.expand_(c, vec![bl, sc as i64, d]));
                    }
                    pieces.push(txt.context("txt input")?);
                    h.concat_(pieces, 1)
                }
                DitPart::Single(i) => g.block(
                    &format!("single_blocks.{i}"),
                    x,
                    single.context("single modulation")?,
                    joint,
                    mask,
                )?,
                DitPart::Final => {
                    let fm = final_mods.context("final modulation")?;
                    let dd = d as i64;
                    let (img, shift, scale) = {
                        let mut h = g.g();
                        let img = h.narrow_(x, 1, 0, si);
                        let sh = h.narrow_(fm, 0, 0, 1);
                        let sh = h.reshape_(sh, vec![1, 1, dd]);
                        let sc = h.narrow_(fm, 0, 1, 1);
                        let sc = h.reshape_(sc, vec![1, 1, dd]);
                        (img, sh, sc)
                    };
                    let y = g.modulate(img, shift, scale);
                    g.linear(y, "final_layer.image.linear")?
                }
            };
        }
        Ok(Some(wrap_out(&mut g, x)))
    });
    flow.output("out").build(weights)
}

/// A compiled [`build_mode_stack`].
pub struct ModeStack {
    pub mode: Mode,
    pub len: usize,
    graph: CompiledGraph,
}

impl ModeStack {
    pub fn compile(
        cfg: &Flux3Config,
        mode: Mode,
        ids: &[[f32; 4]],
        weights: &mut DitSource<'_>,
        device: Device,
    ) -> Result<Self> {
        let built = build_mode_stack(cfg, mode, ids, weights)?;
        Ok(Self {
            mode,
            len: ids.len(),
            graph: rlx_core::flow_util::compile_built(built, device)?,
        })
    }

    /// `[n, in]` tokens and `[3·D]` early modulation → `[n, D]`.
    pub fn run(&mut self, tokens: &[f32], mods: &[f32]) -> Result<Vec<f32>> {
        let out = self.graph.run(&[("x", tokens), ("mod", mods)]);
        out.into_iter()
            .next()
            .context("mode stack produced no output")
    }
}

/// Inputs of one step of the chain.
pub struct StepInputs<'a> {
    /// `[Si, in]` latents.
    pub img: &'a [f32],
    /// `[Sc, D]` reference early output (empty for text-to-image).
    pub cond: &'a [f32],
    /// `[B, St, D]` text early output, zero-filled per sample.
    pub txt: &'a [f32],
    /// `[B, S]` key mask (when [`ChainGeometry::masked`]).
    pub mask: &'a [f32],
    pub mod_early: &'a [f32],
    /// `[9, D]`: image, reference and text single-stream modulation.
    pub mod_single: &'a [f32],
    pub mod_final: &'a [f32],
}

/// The compiled per-step chain.
pub struct MainChain {
    pub geo: ChainGeometry,
    chunks: Vec<(Vec<DitPart>, CompiledGraph)>,
}

impl MainChain {
    /// Build + compile every chunk; each chunk's host weights are dropped
    /// once uploaded.
    pub fn compile(
        cfg: &Flux3Config,
        geo: ChainGeometry,
        weights: &mut DitSource<'_>,
        device: Device,
        blocks_per_graph: usize,
    ) -> Result<Self> {
        ensure!(
            geo.ids.len() == geo.total(),
            "position ids / geometry mismatch"
        );
        let plan = plan_chunks(cfg, blocks_per_graph);
        let mut chunks = Vec::with_capacity(plan.len());
        for (n, parts) in plan.into_iter().enumerate() {
            let t0 = std::time::Instant::now();
            let built = build_chain_chunk(cfg, &geo, &parts, weights)?;
            let graph = rlx_core::flow_util::compile_built(built, device)?;
            if rlx_ir::env::flag("RLX_FLUX3_VERBOSE") {
                eprintln!(
                    "[rlx-flux3] chunk {} {:?}..{:?} compiled in {:.1}s",
                    n + 1,
                    parts.first(),
                    parts.last(),
                    t0.elapsed().as_secs_f32()
                );
            }
            chunks.push((parts, graph));
        }
        Ok(Self { geo, chunks })
    }

    /// Velocity `[B, Si, in]`.
    pub fn run(&mut self, inp: &StepInputs<'_>) -> Result<Vec<f32>> {
        let mut x: Vec<f32> = Vec::new();
        let masked = self.geo.masked;
        let has_cond = self.geo.cond_len > 0;
        for (n, (parts, graph)) in self.chunks.iter_mut().enumerate() {
            let prev = std::mem::take(&mut x);
            let mut feeds: Vec<(&str, &[f32])> = Vec::new();
            let joined = |p: &DitPart| matches!(p, DitPart::Single(_) | DitPart::Final);
            match parts.first() {
                Some(DitPart::Embed) => feeds.push(("img", inp.img)),
                Some(p) if joined(p) => feeds.push(("x", &prev)),
                _ => feeds.push(("h", &prev)),
            }
            if parts.iter().any(|p| matches!(p, DitPart::Early(_))) {
                feeds.push(("mod_early", inp.mod_early));
            }
            if parts.contains(&DitPart::Join) {
                if has_cond {
                    feeds.push(("cond", inp.cond));
                }
                feeds.push(("txt", inp.txt));
            }
            if parts.iter().any(|p| matches!(p, DitPart::Single(_))) {
                feeds.push(("mod_single", inp.mod_single));
                if masked {
                    feeds.push(("mask", inp.mask));
                }
            }
            if parts.contains(&DitPart::Final) {
                feeds.push(("mod_final", inp.mod_final));
            }
            let out = graph.run(&feeds);
            x = out
                .into_iter()
                .next()
                .with_context(|| format!("chain chunk {n} produced no output"))?;
        }
        Ok(x)
    }
}

/// Host-side per-step modulation inputs for the chain, from [`Conditioning`].
pub struct StepMods {
    pub early: Vec<f32>,
    pub single: Vec<f32>,
    pub final_: Vec<f32>,
}

/// The chain's modulations at flow time `t` (`t_ref` for a flow map).
/// References and text sit at time 0.
pub fn step_mods(c: &Conditioning, t: f32, t_ref: Option<f32>) -> StepMods {
    let vi = c.vec(t, t_ref);
    let v0 = c.vec(0.0, t_ref);
    let img = c.mode_mods(Mode::Image, &vi);
    let cond = c.mode_mods(Mode::Cond, &v0);
    let txt = c.mode_mods(Mode::Text, &v0);
    let mut single = img.single;
    single.extend(cond.single);
    single.extend(txt.single);
    StepMods {
        early: img.early,
        single,
        final_: c.final_mods(&vi),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cut_on_block_count() {
        let cfg = Flux3Config::klein();
        let plan = plan_chunks(&cfg, 8);
        let flat: Vec<DitPart> = plan.iter().flatten().copied().collect();
        assert_eq!(flat.len(), 1 + 5 + 1 + 28 + 1);
        assert_eq!(plan[0][0], DitPart::Embed);
        // 5 early + 3 single in the first chunk, the join between them.
        assert_eq!(plan[0].len(), 1 + 5 + 1 + 3);
        assert_eq!(*plan.last().unwrap().last().unwrap(), DitPart::Final);
        for c in &plan {
            let blocks = c
                .iter()
                .filter(|p| matches!(p, DitPart::Early(_) | DitPart::Single(_)))
                .count();
            assert!(blocks <= 8);
        }
    }
}
