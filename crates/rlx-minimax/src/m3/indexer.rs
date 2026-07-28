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

//! MiniMax-M3 MSA lightning indexer (`self_attn.index_*`).
//!
//! Produces the per-query, per-head block-sparse **additive attention bias**
//! `[1, num_heads, seq, seq]` (`0` where a key is attendable, a large negative
//! elsewhere) that the main attention adds to `QKᵀ·scale` before softmax.
//!
//! Per indexer head (one per GQA group):
//! ```text
//!   score  = idx_q_h · idx_kᵀ                      [seq_q, seq_k]  (+ causal −BIG)
//!   pad k to a block multiple with −BIG, reshape → [seq_q, n_blk, block]
//!   blk    = max over block                        [seq_q, n_blk]
//!   blk   += local_boost (query's own block +BIG)  [seq_q, n_blk]
//!   thr    = k-th largest of blk per row           [seq_q, 1]
//!   keep   = blk >= thr                             (bool → 0 / −BIG)
//!   expand blocks→keys, crop to seq, + causal      [seq_q, seq_k]
//! ```
//! The indexer's own scores are exp-free; top-k is realized as a threshold
//! comparison against the k-th largest block score (ties keep a few extra
//! blocks — harmless, since the final causal mask still applies). Indexer heads
//! are repeat-interleaved by the GQA group size up to `num_heads`.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{CmpOp, Op, PadMode, ReduceOp};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use super::ops::per_head_gemma_rmsnorm;
use super::{ROPE_COS, ROPE_SIN};

/// Large finite magnitude used in place of ±∞ so mask arithmetic never produces
/// NaNs (softmax treats `-BIG` as ~zero weight).
const BIG: f32 = 1e30;

/// Dimensions for the MSA lightning indexer of one sparse layer.
#[derive(Debug, Clone, Copy)]
pub struct IndexerDims {
    /// Residual-stream width.
    pub hidden: usize,
    /// Query heads of the main attention (bias is expanded up to this count).
    pub num_heads: usize,
    /// Indexer heads (one per GQA group; `== num_key_value_heads`).
    pub index_n_heads: usize,
    /// Indexer projection dim per head.
    pub index_head_dim: usize,
    /// Leading indexer dims that receive RoPE.
    pub n_rot: usize,
    /// Keys per block for max-pool selection.
    pub block_size: usize,
    /// Top-k blocks kept per query.
    pub topk_blocks: usize,
    /// Blocks ending at the query's own block that are always visible.
    pub local_blocks: usize,
    /// RMSNorm epsilon (Gemma indexer norm).
    pub eps: f32,
    /// Sequence length.
    pub seq: usize,
}

/// Emit the MSA additive bias `[1, num_heads, seq, seq]` for `{prefix}` (= the
/// layer's `self_attn`), consuming the layer-normed hidden `[1, seq, hidden]`.
pub fn emit_msa_bias(
    emit: &mut Emit<'_>,
    prefix: &str,
    normed: HirNodeId,
    d: IndexerDims,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let seq = d.seq;
    let ih = d.index_n_heads;
    let id = d.index_head_dim;
    let block = d.block_size;
    let n_blk = seq.div_ceil(block);
    let kpad = n_blk * block;
    let eff_k = d.topk_blocks.min(n_blk).max(1);
    let group = d.num_heads / ih;

    // --- Host-precomputed masks (fixed for a given prefill seq) ---
    let causal_neg = emit.synth_param(
        &format!("{prefix}.m3_causal_neg"),
        causal_neg_data(seq),
        Shape::new(&[seq, seq], f),
    );
    let local_boost = emit.synth_param(
        &format!("{prefix}.m3_local_boost"),
        local_boost_data(seq, block, n_blk, d.local_blocks),
        Shape::new(&[seq, n_blk], f),
    );
    let zeros = emit.synth_param(
        &format!("{prefix}.m3_keep_zeros"),
        vec![0.0f32; seq * n_blk],
        Shape::new(&[seq, n_blk], f),
    );
    let negbig = emit.synth_param(
        &format!("{prefix}.m3_keep_negbig"),
        vec![-BIG; seq * n_blk],
        Shape::new(&[seq, n_blk], f),
    );

    // --- Indexer projections + per-head Gemma norm + partial RoPE ---
    let iq = {
        let w = emit.load_param(&format!("{prefix}.index_q_proj.weight"), true)?;
        let mut gb = HirMut::new(emit.hir());
        gb.mm(normed, w)
    };
    let ik = {
        let w = emit.load_param(&format!("{prefix}.index_k_proj.weight"), true)?;
        let mut gb = HirMut::new(emit.hir());
        gb.mm(normed, w)
    };
    let iq = per_head_gemma_rmsnorm(
        emit,
        &format!("{prefix}.index_q_norm"),
        iq,
        1,
        seq,
        ih,
        id,
        d.eps,
    )?;
    let ik = per_head_gemma_rmsnorm(
        emit,
        &format!("{prefix}.index_k_norm"),
        ik,
        1,
        seq,
        1,
        id,
        d.eps,
    )?;
    let cos = emit.flow_input(ROPE_COS)?.hir_id();
    let sin = emit.flow_input(ROPE_SIN)?.hir_id();

    let mut gb = HirMut::new(emit.hir());
    let iq = gb.rope_n(iq, cos, sin, id, d.n_rot);
    let ik = gb.rope_n(ik, cos, sin, id, d.n_rot);
    // ik → [id, seq] for the score matmul.
    let ik2d = gb.reshape_(ik, vec![seq as i64, id as i64]);
    let ik_t = gb.transpose_(ik2d, vec![1, 0]); // [id, seq]

    let mut head_biases: Vec<HirNodeId> = Vec::with_capacity(d.num_heads);
    for h in 0..ih {
        let iq_h = gb.narrow_(iq, 2, h * id, id); // [1, seq, id]
        let iq_h2d = gb.reshape_(iq_h, vec![seq as i64, id as i64]);
        let score = gb.mm(iq_h2d, ik_t); // [seq, seq]
        let score = gb.add(score, causal_neg);
        // Pad key axis up to a block multiple with -BIG, then block-max-pool.
        let score_p = if kpad > seq {
            gb.pad_(
                score,
                vec![[0, 0], [0, kpad - seq]],
                PadMode::Constant(-BIG),
            )
        } else {
            score
        };
        let blk = gb.reshape_(score_p, vec![seq as i64, n_blk as i64, block as i64]);
        let blkmax = gb.add_node(
            Op::Reduce {
                op: ReduceOp::Max,
                axes: vec![2],
                keep_dim: false,
            },
            vec![blk],
            Shape::new(&[seq, n_blk], f),
        );
        let blkmax = gb.add(blkmax, local_boost);
        // Top-k as a threshold: keep blocks whose score ≥ the k-th largest.
        let top_idx = gb.add_node(
            Op::TopK { k: eff_k },
            vec![blkmax],
            Shape::new(&[seq, eff_k], f),
        );
        let top_val = gb.add_node(
            Op::GatherElements { axis: 1 },
            vec![blkmax, top_idx],
            Shape::new(&[seq, eff_k], f),
        ); // [seq, eff_k]
        let thresh = gb.narrow_(top_val, 1, eff_k - 1, 1); // [seq, 1]
        // Expand to full shape: Metal's Compare kernel rejects non-scalar broadcast.
        let thresh = gb.add_node(
            Op::Expand {
                target_shape: vec![seq as i64, n_blk as i64],
            },
            vec![thresh],
            Shape::new(&[seq, n_blk], f),
        );
        let keep = gb.add_node(
            Op::Compare(CmpOp::Ge),
            vec![blkmax, thresh],
            Shape::new(&[seq, n_blk], DType::Bool),
        );
        let keep_add = gb.add_node(
            Op::Where,
            vec![keep, zeros, negbig],
            Shape::new(&[seq, n_blk], f),
        );
        // Expand block verdict back onto every key, crop pad, add causal.
        let ka3 = gb.reshape_(keep_add, vec![seq as i64, n_blk as i64, 1]);
        let ka3 = gb.add_node(
            Op::Expand {
                target_shape: vec![seq as i64, n_blk as i64, block as i64],
            },
            vec![ka3],
            Shape::new(&[seq, n_blk, block], f),
        );
        let kt = gb.reshape_(ka3, vec![seq as i64, kpad as i64]);
        let kt = if kpad > seq {
            gb.narrow_(kt, 1, 0, seq)
        } else {
            kt
        };
        let head_bias = gb.add(kt, causal_neg); // [seq, seq]
        let head_bias4 = gb.reshape_(head_bias, vec![1, 1, seq as i64, seq as i64]);
        for _ in 0..group {
            head_biases.push(head_bias4);
        }
    }

    Ok(gb.concat_(head_biases, 1)) // [1, num_heads, seq, seq]
}

/// `causal_neg[q,k] = 0` if `k ≤ q` else `-BIG`.
fn causal_neg_data(seq: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; seq * seq];
    for q in 0..seq {
        for k in 0..seq {
            if k > q {
                v[q * seq + k] = -BIG;
            }
        }
    }
    v
}

/// `local_boost[q,b] = +BIG` for the `local_blocks` blocks ending at the query's
/// own block (`q / block_size`), else `0` — forces them always-visible.
fn local_boost_data(seq: usize, block: usize, n_blk: usize, local: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; seq * n_blk];
    for q in 0..seq {
        let qb = q / block;
        for l in 0..local {
            let b = qb.saturating_sub(l);
            v[q * n_blk + b] = BIG;
        }
    }
    v
}
