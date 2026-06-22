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

//! V-JEPA2 attentive pooler + optional classifier (finetuned checkpoints).

use super::config::Vjepa2Config;
use super::layers::{attention_plain, cross_attention};
use super::weights::{Vjepa2PoolerCrossWeights, Vjepa2PoolerSelfBlockWeights, Vjepa2PoolerWeights};
use anyhow::Result;
use rlx_core::host_kernels::{gelu_tanh, layer_norm, linear};

pub struct Vjepa2PoolerOutput {
    pub embedding: Vec<f32>,
    pub logits: Option<Vec<f32>>,
}

/// Pool encoder tokens `[batch, seq, hidden]` → `[batch, hidden]` embedding.
pub fn pool_native(
    encoder_tokens: &[f32],
    weights: &Vjepa2PoolerWeights,
    cfg: &Vjepa2Config,
    batch: usize,
    seq: usize,
) -> Result<Vjepa2PoolerOutput> {
    let e = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let head_dim = cfg.head_dim();
    let hidden = cfg.pooler_intermediate_size();
    let eps = cfg.layer_norm_eps as f32;

    let mut per_batch = Vec::with_capacity(batch * e);

    for bi in 0..batch {
        let mut x = encoder_tokens[bi * seq * e..(bi + 1) * seq * e].to_vec();

        for block in &weights.self_blocks {
            pooler_self_block(&mut x, block, 1, seq, e, nh, head_dim, hidden, eps)?;
        }

        let mut q = weights.query_tokens.clone();
        cross_block(
            &mut q,
            &x,
            &weights.cross,
            1,
            1,
            seq,
            e,
            nh,
            head_dim,
            hidden,
            eps,
        )?;
        per_batch.extend_from_slice(&q[..e]);
    }

    let logits = match (&weights.classifier_w_t, &weights.classifier_b) {
        (Some(w), Some(b)) => {
            let nc = b.len();
            Some(linear(&per_batch, batch, e, w, nc, b)?)
        }
        _ => None,
    };

    Ok(Vjepa2PoolerOutput {
        embedding: per_batch,
        logits,
    })
}

#[allow(clippy::too_many_arguments)]
fn pooler_self_block(
    x: &mut [f32],
    block: &Vjepa2PoolerSelfBlockWeights,
    batch: usize,
    seq: usize,
    e: usize,
    nh: usize,
    head_dim: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    let rows = batch * seq;
    let n1 = layer_norm(x, &block.norm1_w, &block.norm1_b, e, eps)?;
    let attn = attention_plain(
        &n1,
        batch,
        seq,
        e,
        nh,
        head_dim,
        &block.q_w_t,
        &block.q_b,
        &block.k_w_t,
        &block.k_b,
        &block.v_w_t,
        &block.v_b,
        &block.out_w_t,
        &block.out_b,
    )?;
    for i in 0..x.len() {
        x[i] += attn[i];
    }

    let n2 = layer_norm(x, &block.norm2_w, &block.norm2_b, e, eps)?;
    let mut mlp = linear(&n2, rows, e, &block.mlp_fc1_w_t, hidden, &block.mlp_fc1_b)?;
    gelu_tanh(&mut mlp);
    let ffn = linear(&mlp, rows, hidden, &block.mlp_fc2_w_t, e, &block.mlp_fc2_b)?;
    for i in 0..x.len() {
        x[i] += ffn[i];
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cross_block(
    queries: &mut [f32],
    context: &[f32],
    block: &Vjepa2PoolerCrossWeights,
    batch: usize,
    l_q: usize,
    l_kv: usize,
    e: usize,
    nh: usize,
    head_dim: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    let residual = queries.to_vec();
    let ctx_norm = layer_norm(context, &block.norm1_w, &block.norm1_b, e, eps)?;
    let attn = cross_attention(
        queries,
        &ctx_norm,
        batch,
        l_q,
        l_kv,
        e,
        nh,
        head_dim,
        &block.q_w_t,
        &block.q_b,
        &block.k_w_t,
        &block.k_b,
        &block.v_w_t,
        &block.v_b,
    )?;
    for i in 0..queries.len() {
        queries[i] = residual[i] + attn[i];
    }

    let n2 = layer_norm(queries, &block.norm2_w, &block.norm2_b, e, eps)?;
    let mut mlp = linear(
        &n2,
        batch * l_q,
        e,
        &block.mlp_fc1_w_t,
        hidden,
        &block.mlp_fc1_b,
    )?;
    gelu_tanh(&mut mlp);
    let ffn = linear(
        &mlp,
        batch * l_q,
        hidden,
        &block.mlp_fc2_w_t,
        e,
        &block.mlp_fc2_b,
    )?;
    for i in 0..queries.len() {
        queries[i] += ffn[i];
    }
    Ok(())
}
