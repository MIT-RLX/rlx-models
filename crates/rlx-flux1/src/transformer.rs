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

//! The FLUX.1 MMDiT denoiser as a chain of compiled graphs.
//!
//! One graph per *chunk* of blocks: building a 12B model as a single graph
//! would hold every parameter as host f32 (~48 GB) until compile. Each chunk
//! instead materializes only its own blocks, uploads them (bf16 stays bf16
//! under [`WeightStorage::Auto`]) and frees the host copy before the next is
//! built. Chunks hand the joint stream `x = [txt | img]` (`[1, St+Si, D]`)
//! to each other; the conditioning vector `vec` is an input to every chunk.
//!
//! Block structure follows BFL `flux/model.py` + `flux/modules/layers.py`:
//! adaLN-Zero modulation from `silu(vec)`, QK-RMSNorm, 3-axis RoPE over the
//! joint sequence, tanh-GELU MLPs; single blocks fuse qkv+MLP-in (`linear1`)
//! and attn-out+MLP-out (`linear2`).

use anyhow::{Context, Result, ensure};
use rlx_flow::{BuiltModel, Emit, ModelFlow, WeightStorage};
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use rlx_runtime::{CompiledGraph, Device};

use crate::config::{Flux1Config, NORM_EPS};
use crate::embed::VecEmbedder;
use crate::rope::{position_ids, rope_tables};
use crate::weights::{DitWeights, LORA_A, LORA_B};
use std::collections::HashSet;
use std::sync::Arc;

/// One unit of the denoiser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitPart {
    /// `img_in` / `txt_in` projections (graph inputs `img`, `txt`).
    Embed,
    Double(usize),
    Single(usize),
    /// `final_layer` over the image tokens (output `[1, Si, in_channels]`).
    Final,
}

/// Split the denoiser into chunks of at most `doubles` double blocks /
/// `singles` single blocks. The embed rides in the first chunk and the final
/// layer in the last.
pub fn plan_chunks(cfg: &Flux1Config, doubles: usize, singles: usize) -> Vec<Vec<DitPart>> {
    let (doubles, singles) = (doubles.max(1), singles.max(1));
    let mut chunks: Vec<Vec<DitPart>> = Vec::new();
    for start in (0..cfg.depth).step_by(doubles) {
        chunks.push(
            (start..(start + doubles).min(cfg.depth))
                .map(DitPart::Double)
                .collect(),
        );
    }
    for start in (0..cfg.depth_single).step_by(singles) {
        chunks.push(
            (start..(start + singles).min(cfg.depth_single))
                .map(DitPart::Single)
                .collect(),
        );
    }
    chunks[0].insert(0, DitPart::Embed);
    chunks
        .last_mut()
        .expect("at least one chunk")
        .push(DitPart::Final);
    chunks
}

/// Sequence geometry a set of chunk graphs is compiled for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DitGeometry {
    /// Text tokens (T5 sequence length).
    pub txt_len: usize,
    /// Packed latent grid (`latent_h / 2`, `latent_w / 2`).
    pub grid_h: usize,
    pub grid_w: usize,
}

impl DitGeometry {
    pub fn img_len(&self) -> usize {
        self.grid_h * self.grid_w
    }
    pub fn total(&self) -> usize {
        self.txt_len + self.img_len()
    }
}

/// Build the graph for one chunk.
pub fn build_chunk(
    cfg: &Flux1Config,
    geo: DitGeometry,
    parts: &[DitPart],
    weights: &mut dyn DitWeights,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let (st, si, d) = (geo.txt_len, geo.img_len(), cfg.hidden);
    let s = geo.total();
    let mut flow = ModelFlow::new("flux1_dit_chunk").with_weight_storage(WeightStorage::Auto);
    if parts.first() == Some(&DitPart::Embed) {
        flow = flow
            .input("img", Shape::new(&[1, si, cfg.in_channels], f))
            .input("txt", Shape::new(&[1, st, cfg.context_dim], f));
    } else {
        flow = flow.input("x", Shape::new(&[1, s, d], f));
    }
    flow = flow.input("vec", Shape::new(&[1, d], f));
    let cfg_c = cfg.clone();
    let parts_c = parts.to_vec();
    let lora: Arc<HashSet<String>> = Arc::new(weights.lora_targets());
    flow = flow.plugin_named("flux1.chunk", move |emit, _| {
        let mut g = Dit::new(emit, &cfg_c, geo, lora.clone())?;
        let mut x = if parts_c.first() == Some(&DitPart::Embed) {
            None
        } else {
            Some(g.emit.flow_input("x")?.hir_id())
        };
        for part in &parts_c {
            x = Some(match *part {
                DitPart::Embed => g.embed()?,
                DitPart::Double(i) => g.double_block(i, x.context("no stream")?)?,
                DitPart::Single(i) => g.single_block(i, x.context("no stream")?)?,
                DitPart::Final => g.final_layer(x.context("no stream")?)?,
            });
        }
        let x = x.context("empty chunk")?;
        let shape = g.shape(x);
        Ok(Some(g.emit.wrap(x, shape)))
    });
    flow.output("out")
        .build(weights as &mut dyn rlx_flow::WeightSource)
}

/// Graph-building context for one chunk.
struct Dit<'e, 'a> {
    emit: &'e mut Emit<'a>,
    cfg: &'e Flux1Config,
    geo: DitGeometry,
    vec_silu: HirNodeId,
    cos: HirNodeId,
    sin: HirNodeId,
    ln_one: HirNodeId,
    ln_zero: HirNodeId,
    head_zero: HirNodeId,
    /// Weight keys carrying a LoRA branch.
    lora: Arc<HashSet<String>>,
}

impl<'e, 'a> Dit<'e, 'a> {
    fn new(
        emit: &'e mut Emit<'a>,
        cfg: &'e Flux1Config,
        geo: DitGeometry,
        lora: Arc<HashSet<String>>,
    ) -> Result<Self> {
        let f = DType::F32;
        let (d, hd) = (cfg.hidden, cfg.head_dim);
        let ids = position_ids(geo.txt_len, geo.grid_h, geo.grid_w);
        let (cos, sin) = rope_tables(&ids, cfg.axes_dim, cfg.theta);
        let cos = emit.synth_param("flux1.rope_cos", cos, Shape::new(&[geo.total(), hd / 2], f));
        let sin = emit.synth_param("flux1.rope_sin", sin, Shape::new(&[geo.total(), hd / 2], f));
        let ln_one = emit.synth_param("flux1.ln_one", vec![1.0; d], Shape::new(&[d], f));
        let ln_zero = emit.synth_param("flux1.ln_zero", vec![0.0; d], Shape::new(&[d], f));
        let head_zero = emit.synth_param("flux1.head_zero", vec![0.0; hd], Shape::new(&[hd], f));
        let vec = emit.flow_input("vec")?.hir_id();
        let vec_silu = HirMut::new(emit.hir()).silu(vec);
        Ok(Self {
            emit,
            cfg,
            geo,
            vec_silu,
            cos,
            sin,
            ln_one,
            ln_zero,
            head_zero,
            lora,
        })
    }

    fn g(&mut self) -> HirMut<'_> {
        HirMut::new(self.emit.hir())
    }

    fn shape(&mut self, x: HirNodeId) -> Shape {
        self.g().shape(x).clone()
    }

    /// `x @ Wᵀ + b` for BFL `prefix.{weight,bias}`, plus `(x Aᵀ) Bᵀ` when a
    /// LoRA targets it.
    fn linear(&mut self, x: HirNodeId, prefix: &str) -> Result<HirNodeId> {
        let wk = format!("{prefix}.weight");
        let packed = match self.emit.weights.take_packed(&wk)? {
            Some(p) => {
                let key = format!("{wk}\0q");
                let id = self
                    .emit
                    .hir()
                    .param(&key, Shape::new(&[p.w_q.len()], DType::U8));
                self.emit.state.typed_params.push((key, p.w_q, DType::U8));
                Some((id, p.out_dim, p.scheme))
            }
            None => None,
        };
        let w = match packed {
            Some(_) => None,
            None => Some(self.emit.load_matrix(&wk, true)?),
        };
        let b = self.emit.load_param(&format!("{prefix}.bias"), false)?;
        // A packed projection has its LoRA merged in (`DitSource::take_packed`).
        let lora = if packed.is_none() && self.lora.contains(&wk) {
            Some((
                self.emit.load_param(&format!("{wk}{LORA_A}"), true)?,
                self.emit.load_param(&format!("{wk}{LORA_B}"), true)?,
            ))
        } else {
            None
        };
        let y = match (packed, w) {
            (Some((id, out, scheme)), _) => {
                let s = self.shape(x);
                let last = s.rank() - 1;
                let shape = s.with_dim(last, rlx_ir::Dim::Static(out));
                self.emit.hir().dequant_matmul(
                    x,
                    id,
                    None,
                    None,
                    scheme,
                    shape.with_dtype(DType::F32),
                )
            }
            (None, Some(w)) => self.g().mm(x, w),
            (None, None) => unreachable!("dense weight loaded above"),
        };
        let mut g = self.g();
        let y = match lora {
            Some((a, bb)) => {
                let h = g.mm(x, a);
                let d = g.mm(h, bb);
                g.add(y, d)
            }
            None => y,
        };
        Ok(g.add(y, b))
    }

    /// `n` modulation vectors of width D from `lin(silu(vec))`, each `[1, 1, D]`.
    fn modulation(&mut self, prefix: &str, n: usize) -> Result<Vec<HirNodeId>> {
        let v = self.vec_silu;
        let m = self.linear(v, prefix)?;
        let d = self.cfg.hidden as i64;
        let mut g = self.g();
        Ok((0..n)
            .map(|i| {
                let c = g.narrow_(m, 1, i * d as usize, d as usize);
                g.reshape_(c, vec![1, 1, d])
            })
            .collect())
    }

    /// `LayerNorm(x) · (1 + scale) + shift` (no affine).
    fn modulate(&mut self, x: HirNodeId, shift: HirNodeId, scale: HirNodeId) -> HirNodeId {
        let (one, zero) = (self.ln_one, self.ln_zero);
        let mut g = self.g();
        let dims: Vec<i64> = g
            .shape(x)
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        let n = g.ln(x, one, zero, NORM_EPS);
        let scale = g.expand_(scale, dims.clone());
        let shift = g.expand_(shift, dims);
        let ns = g.mul(n, scale);
        let n = g.add(n, ns);
        g.add(n, shift)
    }

    /// `x + gate · y`.
    fn gated(&mut self, x: HirNodeId, gate: HirNodeId, y: HirNodeId) -> HirNodeId {
        let mut g = self.g();
        let dims: Vec<i64> = g
            .shape(x)
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        let gate = g.expand_(gate, dims);
        let gy = g.mul(gate, y);
        g.add(x, gy)
    }

    /// Per-head RMSNorm with the (de-interleaved) `scale`.
    fn qk_norm(&mut self, x: HirNodeId, seq: usize, scale_key: &str) -> Result<HirNodeId> {
        let scale = self.emit.load_param(scale_key, false)?;
        let (h, hd, d) = (
            self.cfg.num_heads as i64,
            self.cfg.head_dim as i64,
            self.cfg.hidden as i64,
        );
        let zero = self.head_zero;
        let mut g = self.g();
        let x4 = g.reshape_(x, vec![1, seq as i64, h, hd]);
        let n = g.rms_norm(x4, scale, zero, NORM_EPS);
        Ok(g.reshape_(n, vec![1, seq as i64, d]))
    }

    fn rope(&mut self, x: HirNodeId) -> HirNodeId {
        let (cos, sin, hd) = (self.cos, self.sin, self.cfg.head_dim);
        self.g().rope(x, cos, sin, hd)
    }

    fn attention(&mut self, q: HirNodeId, k: HirNodeId, v: HirNodeId) -> HirNodeId {
        let (h, hd) = (self.cfg.num_heads, self.cfg.head_dim);
        let q = self.rope(q);
        let k = self.rope(k);
        let shape = self.shape(v);
        self.g()
            .attention_kind(q, k, v, h, hd, MaskKind::None, shape)
    }

    fn mlp(&mut self, x: HirNodeId, prefix: &str) -> Result<HirNodeId> {
        let h = self.linear(x, &format!("{prefix}.0"))?;
        let h = self.g().gelu_approx(h);
        self.linear(h, &format!("{prefix}.2"))
    }

    fn embed(&mut self) -> Result<HirNodeId> {
        let img = self.emit.flow_input("img")?.hir_id();
        let txt = self.emit.flow_input("txt")?.hir_id();
        let img = self.linear(img, "img_in")?;
        let txt = self.linear(txt, "txt_in")?;
        Ok(self.g().concat_(vec![txt, img], 1))
    }

    /// Dual-stream block: separate weights per stream, joint attention.
    fn double_block(&mut self, i: usize, x: HirNodeId) -> Result<HirNodeId> {
        let (st, si, d) = (self.geo.txt_len, self.geo.img_len(), self.cfg.hidden);
        let p = format!("double_blocks.{i}");
        let (txt, img) = {
            let mut g = self.g();
            (g.narrow_(x, 1, 0, st), g.narrow_(x, 1, st, si))
        };
        let mut qkv = Vec::new();
        let mut mods = Vec::new();
        for (stream, seq, h) in [("txt", st, txt), ("img", si, img)] {
            let m = self.modulation(&format!("{p}.{stream}_mod.lin"), 6)?;
            let xm = self.modulate(h, m[0], m[1]);
            let fused = self.linear(xm, &format!("{p}.{stream}_attn.qkv"))?;
            let (q, k, v) = {
                let mut g = self.g();
                (
                    g.narrow_(fused, 2, 0, d),
                    g.narrow_(fused, 2, d, d),
                    g.narrow_(fused, 2, 2 * d, d),
                )
            };
            let q = self.qk_norm(q, seq, &format!("{p}.{stream}_attn.norm.query_norm.scale"))?;
            let k = self.qk_norm(k, seq, &format!("{p}.{stream}_attn.norm.key_norm.scale"))?;
            qkv.push((q, k, v));
            mods.push(m);
        }
        let (q, k, v) = {
            let mut g = self.g();
            (
                g.concat_(vec![qkv[0].0, qkv[1].0], 1),
                g.concat_(vec![qkv[0].1, qkv[1].1], 1),
                g.concat_(vec![qkv[0].2, qkv[1].2], 1),
            )
        };
        let attn = self.attention(q, k, v);
        let (attn_t, attn_i) = {
            let mut g = self.g();
            (g.narrow_(attn, 1, 0, st), g.narrow_(attn, 1, st, si))
        };
        let mut outs = Vec::new();
        for ((stream, h, a), m) in [("txt", txt, attn_t), ("img", img, attn_i)]
            .into_iter()
            .zip(&mods)
        {
            let proj = self.linear(a, &format!("{p}.{stream}_attn.proj"))?;
            let h = self.gated(h, m[2], proj);
            let hm = self.modulate(h, m[3], m[4]);
            let y = self.mlp(hm, &format!("{p}.{stream}_mlp"))?;
            outs.push(self.gated(h, m[5], y));
        }
        Ok(self.g().concat_(outs, 1))
    }

    /// Single-stream block over the joint sequence.
    fn single_block(&mut self, i: usize, x: HirNodeId) -> Result<HirNodeId> {
        let (s, d, mh) = (self.geo.total(), self.cfg.hidden, self.cfg.mlp_hidden);
        let p = format!("single_blocks.{i}");
        let m = self.modulation(&format!("{p}.modulation.lin"), 3)?;
        let xm = self.modulate(x, m[0], m[1]);
        let fused = self.linear(xm, &format!("{p}.linear1"))?;
        let (q, k, v, mlp) = {
            let mut g = self.g();
            (
                g.narrow_(fused, 2, 0, d),
                g.narrow_(fused, 2, d, d),
                g.narrow_(fused, 2, 2 * d, d),
                g.narrow_(fused, 2, 3 * d, mh),
            )
        };
        let q = self.qk_norm(q, s, &format!("{p}.norm.query_norm.scale"))?;
        let k = self.qk_norm(k, s, &format!("{p}.norm.key_norm.scale"))?;
        let attn = self.attention(q, k, v);
        let cat = {
            let mut g = self.g();
            let act = g.gelu_approx(mlp);
            g.concat_(vec![attn, act], 2)
        };
        let out = self.linear(cat, &format!("{p}.linear2"))?;
        Ok(self.gated(x, m[2], out))
    }

    /// `final_layer`: image tokens only, BFL order `(shift, scale)`.
    fn final_layer(&mut self, x: HirNodeId) -> Result<HirNodeId> {
        let (st, si) = (self.geo.txt_len, self.geo.img_len());
        let img = self.g().narrow_(x, 1, st, si);
        let m = self.modulation("final_layer.adaLN_modulation.1", 2)?;
        let y = self.modulate(img, m[0], m[1]);
        self.linear(y, "final_layer.linear")
    }
}

/// The compiled denoiser for one geometry.
pub struct Flux1Dit {
    pub cfg: Flux1Config,
    pub geo: DitGeometry,
    vec_embed: VecEmbedder,
    chunks: Vec<CompiledGraph>,
}

impl Flux1Dit {
    /// Build + compile every chunk. Each chunk's host weights are dropped once
    /// uploaded, so peak host memory is one chunk, not the model.
    pub fn compile(
        cfg: &Flux1Config,
        geo: DitGeometry,
        weights: &mut dyn DitWeights,
        device: Device,
        doubles_per_graph: usize,
        singles_per_graph: usize,
    ) -> Result<Self> {
        let vec_embed = VecEmbedder::load(weights, cfg)?;
        let plan = plan_chunks(cfg, doubles_per_graph, singles_per_graph);
        let mut chunks = Vec::with_capacity(plan.len());
        for (n, parts) in plan.iter().enumerate() {
            let t0 = std::time::Instant::now();
            let built = build_chunk(cfg, geo, parts, weights)?;
            chunks.push(rlx_core::flow_util::compile_built(built, device)?);
            if rlx_ir::env::flag("RLX_FLUX1_VERBOSE") {
                eprintln!(
                    "[rlx-flux1] chunk {}/{} {:?}..{:?} compiled in {:.1}s",
                    n + 1,
                    plan.len(),
                    parts.first(),
                    parts.last(),
                    t0.elapsed().as_secs_f32()
                );
            }
        }
        Ok(Self {
            cfg: cfg.clone(),
            geo,
            vec_embed,
            chunks,
        })
    }

    /// Velocity prediction `[Si, in_channels]` for packed latents `img`
    /// (`[Si, in_channels]`), T5 states `txt` (`[St, context_dim]`), pooled
    /// CLIP `pooled`, flow time `t` and distilled guidance `guidance`.
    pub fn forward(
        &mut self,
        img: &[f32],
        txt: &[f32],
        pooled: &[f32],
        t: f32,
        guidance: f32,
    ) -> Result<Vec<f32>> {
        ensure!(
            img.len() == self.geo.img_len() * self.cfg.in_channels,
            "img len {}",
            img.len()
        );
        ensure!(
            txt.len() == self.geo.txt_len * self.cfg.context_dim,
            "txt len {}",
            txt.len()
        );
        let vec = self.vec_embed.forward(t, guidance, pooled);
        let mut x: Vec<f32> = Vec::new();
        for (n, chunk) in self.chunks.iter_mut().enumerate() {
            let out = if n == 0 {
                chunk.run(&[("img", img), ("txt", txt), ("vec", &vec)])
            } else {
                chunk.run(&[("x", &x), ("vec", &vec)])
            };
            x = out.into_iter().next().context("chunk produced no output")?;
        }
        Ok(x)
    }
}
