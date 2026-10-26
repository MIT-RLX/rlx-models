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

//! T5 v1.1 encoder (FLUX.1's `text_encoder_2`), HF `T5EncoderModel` layout.
//!
//! Pre-RMSNorm blocks; attention has **no** `1/√d` scaling and adds a learned
//! relative-position bias (bucketed, bidirectional, owned by block 0 and
//! shared by every layer); the FFN is gated tanh-GELU. FLUX pads the prompt to
//! a fixed length and passes **no** attention mask, so the pad tokens are
//! attended — reproduced here.
//!
//! T5-XXL is 4.7B parameters, so — as for the denoiser — layers are built in
//! chunks (see [`T5Encoder::compile`]) and stored at their checkpoint width.

use anyhow::{Context, Result, ensure};
use rlx_flow::{BuiltModel, Emit, ModelFlow, WeightStorage};
use rlx_ir::hir::HirMut;
use rlx_ir::op::{MaskKind, Op};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use rlx_runtime::{CompiledGraph, Device};

use crate::config::T5Config;
use crate::weights::{PlainSource, TensorFile};

/// HF `T5Attention._relative_position_bucket(bidirectional=True)`, f32 math as
/// torch does it.
pub fn relative_position_bucket(rel: i64, num_buckets: usize, max_distance: usize) -> usize {
    let half = (num_buckets / 2) as i64;
    let mut bucket = if rel > 0 { half } else { 0 };
    let n = rel.abs();
    let max_exact = half / 2;
    if n < max_exact {
        bucket += n;
    } else {
        let ratio = (n as f32 / max_exact as f32).ln()
            / ((max_distance as f64 / max_exact as f64).ln() as f32)
            * (half - max_exact) as f32;
        bucket += (max_exact + ratio as i64).min(half - 1);
    }
    bucket as usize
}

/// Position bias `[1, heads, len, len]` from block 0's
/// `relative_attention_bias.weight` (`[num_buckets, heads]`).
pub fn position_bias(cfg: &T5Config, table: &[f32], len: usize) -> Vec<f32> {
    let h = cfg.num_heads;
    let mut out = vec![0.0f32; h * len * len];
    for i in 0..len {
        for j in 0..len {
            let b = relative_position_bucket(
                j as i64 - i as i64,
                cfg.relative_attention_num_buckets,
                cfg.relative_attention_max_distance,
            );
            for head in 0..h {
                out[(head * len + i) * len + j] = table[b * h + head];
            }
        }
    }
    out
}

fn emit_layer(
    emit: &mut Emit<'_>,
    cfg: &T5Config,
    len: usize,
    i: usize,
    x: HirNodeId,
    bias: HirNodeId,
    zero: HirNodeId,
) -> Result<HirNodeId> {
    let p = format!("encoder.block.{i}.layer");
    let eps = cfg.layer_norm_epsilon as f32;
    let ln0 = emit.load_param(&format!("{p}.0.layer_norm.weight"), false)?;
    let ln1 = emit.load_param(&format!("{p}.1.layer_norm.weight"), false)?;
    let inner = cfg.inner_dim();
    let h = HirMut::new(emit.hir()).rms_norm(x, ln0, zero, eps);
    let q = linear(emit, h, &format!("{p}.0.SelfAttention.q.weight"))?;
    let k = linear(emit, h, &format!("{p}.0.SelfAttention.k.weight"))?;
    let v = linear(emit, h, &format!("{p}.0.SelfAttention.v.weight"))?;
    let mut g = HirMut::new(emit.hir());
    let attn = g.add_node(
        Op::Attention {
            num_heads: cfg.num_heads,
            head_dim: cfg.d_kv,
            v_head_dim: None,
            mask_kind: MaskKind::Bias,
            score_scale: Some(1.0),
            attn_logit_softcap: None,
        },
        vec![q, k, v, bias],
        Shape::new(&[1, len, inner], DType::F32),
    );
    let o = linear(emit, attn, &format!("{p}.0.SelfAttention.o.weight"))?;
    let mut g = HirMut::new(emit.hir());
    let x = g.add(x, o);
    let h = g.rms_norm(x, ln1, zero, eps);
    let a = linear(emit, h, &format!("{p}.1.DenseReluDense.wi_0.weight"))?;
    let a = HirMut::new(emit.hir()).gelu_approx(a);
    let b = linear(emit, h, &format!("{p}.1.DenseReluDense.wi_1.weight"))?;
    let ab = HirMut::new(emit.hir()).mul(a, b);
    let y = linear(emit, ab, &format!("{p}.1.DenseReluDense.wo.weight"))?;
    Ok(HirMut::new(emit.hir()).add(x, y))
}

/// `x · Wᵀ` for the `[out, in]` checkpoint matrix `key`: packed (dequantized in
/// the matmul) when the source quantizes, else loaded transposed.
fn linear(emit: &mut Emit<'_>, x: HirNodeId, key: &str) -> Result<HirNodeId> {
    if let Some(p) = emit.weights.take_packed(key)? {
        let qkey = format!("{key}\0q");
        let id = emit
            .hir()
            .param(&qkey, Shape::new(&[p.w_q.len()], DType::U8));
        emit.state.typed_params.push((qkey, p.w_q, DType::U8));
        let s = HirMut::new(emit.hir()).shape(x).clone();
        let last = s.rank() - 1;
        let shape = s.with_dim(last, rlx_ir::Dim::Static(p.out_dim));
        return Ok(emit.hir().dequant_matmul(
            x,
            id,
            None,
            None,
            p.scheme,
            shape.with_dtype(DType::F32),
        ));
    }
    let w = emit.load_matrix(key, true)?;
    Ok(HirMut::new(emit.hir()).mm(x, w))
}

/// Build the graph for encoder layers `layers` (+ the final norm when `last`).
pub fn build_t5_chunk(
    cfg: &T5Config,
    len: usize,
    layers: std::ops::Range<usize>,
    last: bool,
    weights: &mut dyn rlx_flow::WeightSource,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let d = cfg.d_model;
    let cfg_c = cfg.clone();
    ModelFlow::new("flux1_t5_chunk")
        .with_weight_storage(WeightStorage::Auto)
        .input("x", Shape::new(&[1, len, d], f))
        .input("bias", Shape::new(&[1, cfg.num_heads, len, len], f))
        .plugin_named("t5.layers", move |emit, _| {
            let mut x = emit.flow_input("x")?.hir_id();
            let bias = emit.flow_input("bias")?.hir_id();
            let zero = emit.synth_param("t5.zero", vec![0.0; d], Shape::new(&[d], f));
            for i in layers.clone() {
                x = emit_layer(emit, &cfg_c, len, i, x, bias, zero)?;
            }
            if last {
                let w = emit.load_param("encoder.final_layer_norm.weight", false)?;
                x = HirMut::new(emit.hir()).rms_norm(x, w, zero, cfg_c.layer_norm_epsilon as f32);
            }
            Ok(Some(emit.wrap(x, Shape::new(&[1, len, d], f))))
        })
        .output("hidden")
        .build(weights)
}

/// A compiled T5 encoder at a fixed sequence length.
pub struct T5Encoder {
    pub cfg: T5Config,
    pub len: usize,
    file: TensorFile,
    embed_key: String,
    bias: Vec<f32>,
    chunks: Vec<CompiledGraph>,
}

impl T5Encoder {
    /// Load + compile from an HF T5 encoder directory or `.safetensors` file.
    pub fn compile(
        path: &std::path::Path,
        cfg: T5Config,
        len: usize,
        device: Device,
        layers_per_graph: usize,
    ) -> Result<Self> {
        Self::compile_quant(path, cfg, len, device, layers_per_graph, None)
    }

    /// [`Self::compile`] with the projection weights quantized at load.
    pub fn compile_quant(
        path: &std::path::Path,
        cfg: T5Config,
        len: usize,
        device: Device,
        layers_per_graph: usize,
        quant: Option<crate::weights::WeightQuant>,
    ) -> Result<Self> {
        let file = TensorFile::open(path)?;
        let embed_key = ["shared.weight", "encoder.embed_tokens.weight"]
            .into_iter()
            .find(|k| file.contains(k))
            .context("T5 checkpoint has no token embedding")?
            .to_string();
        let (table, _) =
            file.load("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")?;
        let bias = position_bias(&cfg, &table, len);
        let mut chunks = Vec::new();
        let step = layers_per_graph.max(1);
        for start in (0..cfg.num_layers).step_by(step) {
            let end = (start + step).min(cfg.num_layers);
            let mut src = PlainSource {
                file: &file,
                prefix: "",
                quant,
            };
            let built = build_t5_chunk(&cfg, len, start..end, end == cfg.num_layers, &mut src)?;
            chunks.push(rlx_core::flow_util::compile_built(built, device)?);
        }
        Ok(Self {
            cfg,
            len,
            file,
            embed_key,
            bias,
            chunks,
        })
    }

    /// Encode exactly `len` token ids → `[len, d_model]`.
    pub fn encode(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        ensure!(
            ids.len() == self.len,
            "T5 expects {} ids, got {}",
            self.len,
            ids.len()
        );
        let mut x = self.file.load_rows(&self.embed_key, ids)?;
        for chunk in &mut self.chunks {
            let out = chunk.run(&[("x", &x), ("bias", &self.bias)]);
            x = out
                .into_iter()
                .next()
                .context("T5 chunk produced no output")?;
        }
        Ok(x)
    }
}

/// T5 ids for FLUX: tokenize, truncate to `len - 1`, append `</s>` (1), pad
/// with `<pad>` (0) to `len`.
pub fn t5_ids(tok: &tokenizers::Tokenizer, text: &str, len: usize) -> Result<Vec<u32>> {
    let enc = tok
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("T5 tokenize: {e}"))?;
    let mut ids: Vec<u32> = enc.get_ids().to_vec();
    if ids.last() != Some(&1) {
        ids.push(1);
    }
    if ids.len() > len {
        ids.truncate(len - 1);
        ids.push(1);
    }
    ids.resize(len, 0);
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_match_hf_table() {
        // Values from HF `_relative_position_bucket(bidirectional=True, 32, 128)`.
        let want = [
            (0, 0),
            (1, 17),
            (-1, 1),
            (7, 23),
            (8, 24),
            (-8, 8),
            (15, 25),
            (16, 26),
            (-16, 10),
            (100, 31),
            (-127, 15),
            (300, 31),
        ];
        for (rel, b) in want {
            assert_eq!(relative_position_bucket(rel, 32, 128), b, "rel {rel}");
        }
    }
}
