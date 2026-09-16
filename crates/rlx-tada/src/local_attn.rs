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

//! `LocalAttentionEncoder` — the 6-layer transformer both codec halves run
//! over the 50 Hz frame grid (`tada.modules.encoder`).
//!
//! Three details differ from a stock encoder block and all three are
//! load-bearing:
//!
//! * **Post-norm, twice.** Attention output is added to the *input* and then
//!   LayerNormed inside `LocalSelfAttention`; the FFN residual is normed again
//!   by the layer. There is no pre-norm anywhere.
//! * **Interleaved RoPE** over adjacent channel pairs, with the cos/sin table
//!   shipped in the checkpoint rather than derived — so it is read from the
//!   weights instead of recomputed.
//! * **Exact (erf) GELU** in the FFN, not the tanh approximation.
//!
//! The name says "local", but the shipped `_precomputed_mask` is all-true and
//! the caller always passes an explicit segment mask, so attention is full
//! within whatever the segment mask allows. That mask is the real locality.

use crate::builder::{Ctx, F32};
use crate::weights::TensorStore;
use anyhow::{Context, Result, bail};
use rlx_ir::hir::HirNodeId;
use rlx_ir::{HirGraphExt, Shape};
use std::sync::Arc;

/// One `LocalAttentionEncoderLayer` — key prefix plus the one shape that is
/// not derivable from the stack's own dimensions.
pub struct Layer {
    prefix: String,
    ffn_dim: usize,
}

/// A full `LocalAttentionEncoder` stack plus its RoPE table.
///
/// Weights stay in the mapped checkpoint and are read once, at graph-build
/// time, straight into the graph's params. Two of these stacks are live at
/// once (codec encoder and decoder); holding their weights on the heap as well
/// as in the arenas doubled the codec's footprint for no benefit.
pub struct LocalAttnStack {
    store: Arc<TensorStore>,
    prefix: String,
    pub layers: Vec<Layer>,
    /// `[max_seq, head_dim / 2]`, split out of the checkpoint's `rope_freqs`.
    cos: Vec<f32>,
    sin: Vec<f32>,
    pub max_seq: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub eps: f32,
}

impl LocalAttnStack {
    /// Load `num_layers` layers from `{prefix}.layers.{i}` plus
    /// `{prefix}.final_norm`.
    pub fn load(
        store: Arc<TensorStore>,
        prefix: &str,
        num_layers: usize,
        num_heads: usize,
        eps: f32,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let p = format!("{prefix}.layers.{i}");
            let ffn_dim = store.shape(&format!("{p}.ffn.0.weight"))?[0];
            layers.push(Layer { prefix: p, ffn_dim });
        }
        let first = layers
            .first()
            .context("local attention stack has no layers")?;
        let d_model = store.shape(&format!("{}.self_attn.out_proj.weight", first.prefix))?[0];
        if !d_model.is_multiple_of(num_heads) {
            bail!("{prefix}: d_model {d_model} is not divisible by {num_heads} heads");
        }

        // Every layer registers an identical `rope_freqs` buffer; read one.
        let key = format!("{prefix}.layers.0.self_attn.rope_freqs");
        let shape = store.shape(&key)?.to_vec();
        if shape.len() != 3 || shape[2] != 2 {
            bail!("{key}: expected [max_seq, head_dim/2, 2], got {shape:?}");
        }
        let half = d_model / num_heads / 2;
        if shape[1] != half {
            bail!(
                "{key}: table half-width {} does not match head_dim/2 = {half}",
                shape[1]
            );
        }
        let flat = store.get(&key)?;
        let max_seq = shape[0];
        let mut cos = Vec::with_capacity(max_seq * half);
        let mut sin = Vec::with_capacity(max_seq * half);
        for chunk in flat.chunks_exact(2) {
            cos.push(chunk[0]);
            sin.push(chunk[1]);
        }

        Ok(Self {
            store,
            prefix: prefix.to_string(),
            layers,
            cos,
            sin,
            max_seq,
            d_model,
            num_heads,
            eps,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    /// Emit the stack over `x` (`[1, seq, d_model]`) under an additive
    /// attention `bias` (`[1, num_heads, seq, seq]`).
    pub fn build(
        &self,
        ctx: &mut Ctx,
        x: HirNodeId,
        bias: HirNodeId,
        seq: usize,
    ) -> Result<HirNodeId> {
        if seq > self.max_seq {
            bail!(
                "sequence of {seq} frames exceeds the checkpoint's {} RoPE positions",
                self.max_seq
            );
        }
        let d = self.d_model;
        let h = self.num_heads;
        let hd = self.head_dim();
        let half = hd / 2;

        let cos = ctx.param(self.cos[..seq * half].to_vec(), &[seq, half]);
        let sin = ctx.param(self.sin[..seq * half].to_vec(), &[seq, half]);

        let mut cur = x;
        for layer in &self.layers {
            let lp = &layer.prefix;
            let qkv =
                ctx.store_linear(&self.store, cur, &format!("{lp}.self_attn.qkv"), 3 * d, d)?;
            let mut parts = Vec::with_capacity(3);
            for slot in 0..3 {
                let p = ctx.g.narrow_(qkv, 2, slot * d, d);
                let p = ctx.g.reshape_(p, vec![1, seq as i64, h as i64, hd as i64]);
                parts.push(p);
            }
            let q = ctx.rope_interleaved(parts[0], cos, sin, 1, seq, h, hd);
            let k = ctx.rope_interleaved(parts[1], cos, sin, 1, seq, h, hd);
            let attn = ctx.g.attention_bias(
                q,
                k,
                parts[2],
                bias,
                h,
                hd,
                Shape::new(&[1, seq, h, hd], F32),
            );
            let attn = ctx.g.reshape_(attn, vec![1, seq as i64, d as i64]);
            let proj =
                ctx.store_linear(&self.store, attn, &format!("{lp}.self_attn.out_proj"), d, d)?;
            let res = ctx.g.add(cur, proj);
            let (ng, nb) = ctx.store_norm(&self.store, &format!("{lp}.self_attn.layer_norm"), d)?;
            cur = ctx.g.ln(res, ng, nb, self.eps);

            let ff = layer.ffn_dim;
            let hgt = ctx.store_linear(&self.store, cur, &format!("{lp}.ffn.0"), ff, d)?;
            let hgt = ctx.g.gelu(hgt);
            let hgt = ctx.store_linear(&self.store, hgt, &format!("{lp}.ffn.3"), d, ff)?;
            let res = ctx.g.add(cur, hgt);
            let (ng, nb) = ctx.store_norm(&self.store, &format!("{lp}.norm"), d)?;
            cur = ctx.g.ln(res, ng, nb, self.eps);
        }

        let (fg, fb) = ctx.store_norm(&self.store, &format!("{}.final_norm", self.prefix), d)?;
        Ok(ctx.g.ln(cur, fg, fb, self.eps))
    }
}
