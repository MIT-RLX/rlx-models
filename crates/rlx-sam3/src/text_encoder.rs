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

//! Native SAM3 text encoder (`VETextEncoder`).
//!
//! Architecture (matches `facebookresearch/sam3.model.text_encoder_ve`):
//!
//!   - `token_embedding`         : `[49408, 1024]`
//!   - `positional_embedding`    : `[32, 1024]`
//!   - 24 × ResidualAttentionBlock(width=1024, heads=16, mlp_ratio=4)
//!     using `nn.MultiheadAttention` (`in_proj_weight [3*W, W]`,
//!     `in_proj_bias [3*W]`, `out_proj.weight [W, W]`, `out_proj.bias [W]`)
//!     and a 32×32 upper-triangular `-inf` causal mask.
//!   - `ln_final`                : LayerNorm 1024
//!   - `resizer = Linear(1024, 256)` outside the encoder.
//!
//! Output: `text_memory_resized` of shape `[seq_len, batch, 256]`.
//!
//! Token IDs are accepted as input — the BPE tokenizer port is deferred to
//! a follow-up that ships an embedded BPE vocab.

use super::config::Sam3TextConfig;
use super::tensor::{layer_norm, matmul, matmul_bt, softmax_rows};
use rlx_core::weight_map::WeightMap;
use rlx_flow::GgufPackedParams;

use crate::packed_gguf::{linear_maybe_gguf, take_or_gguf, take_transposed_with_gguf_key};
use anyhow::{Result, ensure};

#[derive(Clone)]
pub struct Sam3TextBlock {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub qkv_w_t: Vec<f32>,
    pub qkv_b: Vec<f32>,
    pub proj_w_t: Vec<f32>,
    pub proj_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub mlp_fc_w_t: Vec<f32>,
    pub mlp_fc_b: Vec<f32>,
    pub mlp_proj_w_t: Vec<f32>,
    pub mlp_proj_b: Vec<f32>,
    pub qkv_gguf_key: Option<String>,
    pub proj_gguf_key: Option<String>,
    pub mlp_fc_gguf_key: Option<String>,
    pub mlp_proj_gguf_key: Option<String>,
}

#[derive(Clone, Default)]
pub struct Sam3TextEncoderWeights {
    pub loaded: bool,
    pub width: usize,
    pub heads: usize,
    pub context_length: usize,
    pub d_model: usize,
    pub vocab_size: usize,
    pub token_embedding: Vec<f32>,
    pub positional_embedding: Vec<f32>,
    pub ln_final_w: Vec<f32>,
    pub ln_final_b: Vec<f32>,
    pub blocks: Vec<Sam3TextBlock>,
    pub resizer_w_t: Vec<f32>,
    pub resizer_b: Vec<f32>,
    pub resizer_gguf_key: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Sam3TextEncoded {
    /// `[batch, seq]` byte mask (1 = PAD token).
    pub attention_mask: Vec<u8>,
    /// `[seq, batch, d_model]` resized text memory.
    pub text_memory_resized: Vec<f32>,
    /// `[seq, batch, width]` raw token embeddings.
    pub inputs_embeds: Vec<f32>,
    pub seq_len: usize,
    pub batch: usize,
    pub d_model: usize,
    pub width: usize,
}

pub fn extract_text_encoder_weights(
    weights: &mut WeightMap,
    cfg: &Sam3TextConfig,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Sam3TextEncoderWeights> {
    let width = cfg.width;
    let heads = cfg.heads;
    let layers = cfg.layers;
    let d_model = cfg.d_model;
    let context_length = 32usize;
    let vocab_size = 49408usize;
    let _mlp_width = width * 4;

    let prefixes = [
        "detector.backbone.language_backbone",
        "backbone.language_backbone",
        "language_backbone",
    ];
    let enc_prefix = {
        let mut found = None;
        for p in prefixes {
            let key = format!("{p}.encoder.token_embedding.weight");
            if weights.has(&key) {
                found = Some(p);
                break;
            }
        }
        found.ok_or_else(|| anyhow::anyhow!("SAM3 language_backbone not found"))?
    };

    let (token_embedding, te_shape) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{enc_prefix}.encoder.token_embedding.weight"),
    )?;
    ensure!(
        te_shape == vec![vocab_size, width],
        "token_embedding shape {te_shape:?}"
    );
    let (positional_embedding, pe_shape) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{enc_prefix}.encoder.positional_embedding"),
    )?;
    ensure!(
        pe_shape == vec![context_length, width],
        "positional_embedding shape {pe_shape:?}"
    );
    let (ln_final_w, _) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{enc_prefix}.encoder.ln_final.weight"),
    )?;
    let (ln_final_b, _) = take_or_gguf(
        weights,
        gguf_packed,
        &format!("{enc_prefix}.encoder.ln_final.bias"),
    )?;

    // text_projection is unused by VETextEncoder.forward (returns the
    // per-token sequence, not pooled) — drop it without checking shape.
    let _ = weights.take(&format!("{enc_prefix}.encoder.text_projection"));

    let mut blocks = Vec::with_capacity(layers);
    for i in 0..layers {
        let bp = format!("{enc_prefix}.encoder.transformer.resblocks.{i}");
        let (ln1_w, _) = take_or_gguf(weights, gguf_packed, &format!("{bp}.ln_1.weight"))?;
        let (ln1_b, _) = take_or_gguf(weights, gguf_packed, &format!("{bp}.ln_1.bias"))?;
        let (qkv_w_t, qkv_gguf_key) = take_transposed_with_gguf_key(
            weights,
            gguf_packed,
            &format!("{bp}.attn.in_proj_weight"),
        )?;
        let (qkv_b, _) = take_or_gguf(weights, gguf_packed, &format!("{bp}.attn.in_proj_bias"))?;
        let (proj_w_t, proj_gguf_key) = take_transposed_with_gguf_key(
            weights,
            gguf_packed,
            &format!("{bp}.attn.out_proj.weight"),
        )?;
        let (proj_b, _) = take_or_gguf(weights, gguf_packed, &format!("{bp}.attn.out_proj.bias"))?;
        let (ln2_w, _) = take_or_gguf(weights, gguf_packed, &format!("{bp}.ln_2.weight"))?;
        let (ln2_b, _) = take_or_gguf(weights, gguf_packed, &format!("{bp}.ln_2.bias"))?;
        let (mlp_fc_w_t, mlp_fc_gguf_key) =
            take_transposed_with_gguf_key(weights, gguf_packed, &format!("{bp}.mlp.c_fc.weight"))?;
        let (mlp_fc_b, _) = take_or_gguf(weights, gguf_packed, &format!("{bp}.mlp.c_fc.bias"))?;
        let (mlp_proj_w_t, mlp_proj_gguf_key) = take_transposed_with_gguf_key(
            weights,
            gguf_packed,
            &format!("{bp}.mlp.c_proj.weight"),
        )?;
        let (mlp_proj_b, _) = take_or_gguf(weights, gguf_packed, &format!("{bp}.mlp.c_proj.bias"))?;
        blocks.push(Sam3TextBlock {
            ln1_w,
            ln1_b,
            qkv_w_t,
            qkv_b,
            proj_w_t,
            proj_b,
            ln2_w,
            ln2_b,
            mlp_fc_w_t,
            mlp_fc_b,
            mlp_proj_w_t,
            mlp_proj_b,
            qkv_gguf_key,
            proj_gguf_key,
            mlp_fc_gguf_key,
            mlp_proj_gguf_key,
        });
    }

    let (resizer_w_t, resizer_gguf_key) = take_transposed_with_gguf_key(
        weights,
        gguf_packed,
        &format!("{enc_prefix}.resizer.weight"),
    )?;
    let (resizer_b, _) = take_or_gguf(weights, gguf_packed, &format!("{enc_prefix}.resizer.bias"))?;

    Ok(Sam3TextEncoderWeights {
        loaded: true,
        width,
        heads,
        context_length,
        d_model,
        vocab_size,
        token_embedding,
        positional_embedding,
        ln_final_w,
        ln_final_b,
        blocks,
        resizer_w_t,
        resizer_b,
        resizer_gguf_key,
    })
}

/// Run the text encoder on already-tokenized inputs (`[batch, seq_len]`).
pub fn encode_tokens(
    weights: &Sam3TextEncoderWeights,
    tokens: &[u32],
    batch: usize,
    seq_len: usize,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Sam3TextEncoded> {
    ensure!(weights.loaded, "SAM3 text encoder weights not loaded");
    ensure!(
        tokens.len() == batch * seq_len,
        "expected {} tokens, got {}",
        batch * seq_len,
        tokens.len()
    );
    ensure!(
        seq_len <= weights.context_length,
        "seq_len {seq_len} exceeds context_length {}",
        weights.context_length
    );
    let w = weights.width;
    let h = weights.heads;
    let head_dim = w / h;
    ensure!(head_dim * h == w, "width {w} not divisible by heads {h}");

    let mut x = vec![0f32; batch * seq_len * w];
    let mut inputs_embeds = vec![0f32; batch * seq_len * w];
    for b in 0..batch {
        for l in 0..seq_len {
            let tok = tokens[b * seq_len + l] as usize;
            ensure!(tok < weights.vocab_size, "token id {tok} out of vocab");
            let src = &weights.token_embedding[tok * w..(tok + 1) * w];
            let dst_x = &mut x[(b * seq_len + l) * w..(b * seq_len + l + 1) * w];
            let dst_emb = &mut inputs_embeds[(b * seq_len + l) * w..(b * seq_len + l + 1) * w];
            dst_emb.copy_from_slice(src);
            let pos = &weights.positional_embedding[l * w..(l + 1) * w];
            for k in 0..w {
                dst_x[k] = src[k] + pos[k];
            }
        }
    }

    // Causal additive mask [seq_len, seq_len].
    let neg_inf = f32::NEG_INFINITY;
    let mut mask = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i * seq_len + j] = neg_inf;
        }
    }

    for block in &weights.blocks {
        x = block_forward(
            &x,
            block,
            batch,
            seq_len,
            w,
            h,
            head_dim,
            &mask,
            gguf_packed,
        )?;
    }
    x = layer_norm(&x, &weights.ln_final_w, &weights.ln_final_b, w, 1e-5)?;

    // Reorder [B, L, W] → [L, B, W] (sequence-first), then resize.
    let mut text_memory_seq_first = vec![0f32; seq_len * batch * w];
    for b in 0..batch {
        for l in 0..seq_len {
            let src = &x[(b * seq_len + l) * w..(b * seq_len + l + 1) * w];
            let dst = &mut text_memory_seq_first[(l * batch + b) * w..(l * batch + b + 1) * w];
            dst.copy_from_slice(src);
        }
    }
    let mut inputs_embeds_seq_first = vec![0f32; seq_len * batch * w];
    for b in 0..batch {
        for l in 0..seq_len {
            let src = &inputs_embeds[(b * seq_len + l) * w..(b * seq_len + l + 1) * w];
            let dst = &mut inputs_embeds_seq_first[(l * batch + b) * w..(l * batch + b + 1) * w];
            dst.copy_from_slice(src);
        }
    }

    let text_memory_resized = linear_maybe_gguf(
        &text_memory_seq_first,
        seq_len * batch,
        w,
        &weights.resizer_w_t,
        weights.resizer_gguf_key.as_deref(),
        gguf_packed,
        weights.d_model,
        &weights.resizer_b,
    )?;

    let mut attention_mask = vec![0u8; batch * seq_len];
    for i in 0..batch * seq_len {
        attention_mask[i] = if tokens[i] == 0 { 1 } else { 0 };
    }

    Ok(Sam3TextEncoded {
        attention_mask,
        text_memory_resized,
        inputs_embeds: inputs_embeds_seq_first,
        seq_len,
        batch,
        d_model: weights.d_model,
        width: w,
    })
}

fn block_forward(
    x_in: &[f32],
    block: &Sam3TextBlock,
    batch: usize,
    seq_len: usize,
    width: usize,
    heads: usize,
    head_dim: usize,
    mask: &[f32],
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Vec<f32>> {
    let rows = batch * seq_len;
    let n1 = layer_norm(x_in, &block.ln1_w, &block.ln1_b, width, 1e-5)?;
    let qkv = linear_maybe_gguf(
        &n1,
        rows,
        width,
        &block.qkv_w_t,
        block.qkv_gguf_key.as_deref(),
        gguf_packed,
        3 * width,
        &block.qkv_b,
    )?;

    let bh = batch * heads;
    let mut q = vec![0f32; bh * seq_len * head_dim];
    let mut k = vec![0f32; bh * seq_len * head_dim];
    let mut v = vec![0f32; bh * seq_len * head_dim];
    for b in 0..batch {
        for l in 0..seq_len {
            let src = (b * seq_len + l) * 3 * width;
            for hd in 0..heads {
                let qd_src = src + hd * head_dim;
                let kd_src = src + width + hd * head_dim;
                let vd_src = src + 2 * width + hd * head_dim;
                let dst = ((b * heads + hd) * seq_len + l) * head_dim;
                q[dst..dst + head_dim].copy_from_slice(&qkv[qd_src..qd_src + head_dim]);
                k[dst..dst + head_dim].copy_from_slice(&qkv[kd_src..kd_src + head_dim]);
                v[dst..dst + head_dim].copy_from_slice(&qkv[vd_src..vd_src + head_dim]);
            }
        }
    }

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0f32; bh * seq_len * head_dim];
    let mut scores = vec![0f32; seq_len * seq_len];
    for bhi in 0..bh {
        let q_h = &q[bhi * seq_len * head_dim..(bhi + 1) * seq_len * head_dim];
        let k_h = &k[bhi * seq_len * head_dim..(bhi + 1) * seq_len * head_dim];
        let v_h = &v[bhi * seq_len * head_dim..(bhi + 1) * seq_len * head_dim];
        matmul_bt(q_h, k_h, &mut scores, seq_len, head_dim, seq_len, scale);
        for r in 0..seq_len {
            for c in 0..seq_len {
                scores[r * seq_len + c] += mask[r * seq_len + c];
            }
        }
        softmax_rows(&mut scores, seq_len, seq_len);
        let out_h = &mut attn_out[bhi * seq_len * head_dim..(bhi + 1) * seq_len * head_dim];
        matmul(&scores, v_h, out_h, seq_len, seq_len, head_dim);
    }

    let mut packed = vec![0f32; rows * width];
    for b in 0..batch {
        for l in 0..seq_len {
            for hd in 0..heads {
                let src = ((b * heads + hd) * seq_len + l) * head_dim;
                let dst = (b * seq_len + l) * width + hd * head_dim;
                packed[dst..dst + head_dim].copy_from_slice(&attn_out[src..src + head_dim]);
            }
        }
    }
    let attn_proj = linear_maybe_gguf(
        &packed,
        rows,
        width,
        &block.proj_w_t,
        block.proj_gguf_key.as_deref(),
        gguf_packed,
        width,
        &block.proj_b,
    )?;

    let mut x = x_in.to_vec();
    for i in 0..x.len() {
        x[i] += attn_proj[i];
    }
    let n2 = layer_norm(&x, &block.ln2_w, &block.ln2_b, width, 1e-5)?;
    let mlp_hidden = block.mlp_fc_b.len();
    let mut mlp = linear_maybe_gguf(
        &n2,
        rows,
        width,
        &block.mlp_fc_w_t,
        block.mlp_fc_gguf_key.as_deref(),
        gguf_packed,
        mlp_hidden,
        &block.mlp_fc_b,
    )?;
    gelu_exact_inplace(&mut mlp);
    let ffn = linear_maybe_gguf(
        &mlp,
        rows,
        mlp_hidden,
        &block.mlp_proj_w_t,
        block.mlp_proj_gguf_key.as_deref(),
        gguf_packed,
        width,
        &block.mlp_proj_b,
    )?;
    for i in 0..x.len() {
        x[i] += ffn[i];
    }
    Ok(x)
}

fn gelu_exact_inplace(x: &mut [f32]) {
    let inv_sqrt2 = 1.0f32 / std::f32::consts::SQRT_2;
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + erf_approx(*v * inv_sqrt2));
    }
}

fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0f32 } else { 1.0 };
    let ax = x.abs();
    let p = 0.3275911f32;
    let a1 = 0.2548296f32;
    let a2 = -0.2844967f32;
    let a3 = 1.4214138f32;
    let a4 = -1.4531521f32;
    let a5 = 1.0614054f32;
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

/// Legacy shim for `Sam3::predict_image`. Returns an empty/PAD encoding —
/// the BPE tokenizer port is a separate task; for now real prompts must go
/// through `encode_tokens` with externally-tokenized inputs.
pub fn encode_text_native(
    weights: &Sam3TextEncoderWeights,
    cfg: &Sam3TextConfig,
    _prompt: Option<&str>,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Sam3TextEncoded> {
    if !weights.loaded {
        return Ok(Sam3TextEncoded {
            d_model: cfg.d_model,
            width: cfg.width,
            ..Default::default()
        });
    }
    let seq_len = weights.context_length;
    let tokens = vec![0u32; seq_len];
    encode_tokens(weights, &tokens, 1, seq_len, gguf_packed)
}
