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

//! QSA — `Qwen4ExpTextQSAIndexer`, the block indexer in front of every
//! full-attention layer.
//!
//! A cheap side network (4 heads × 128, against the main path's 24 × 256)
//! decides which keys each query may see. Candidates are compressed first:
//! `indexer_compress_ratio = 4` consecutive key vectors average into one
//! block key, so the `indexer_budget = 2048` token budget buys
//! `2048 / 4 = 512` *blocks* — plus each query's own incomplete tail.
//!
//! ```text
//!   q         = RMSNorm(x @ q_projᵀ)          [s, 4, 128]   then M-RoPE at q's position
//!   k_raw     = x @ k_projᵀ                   [s, 128]      no norm, no rotation
//!   pooled_p  = RMSNorm(mean_j k_raw[4p+j])   [P, 128]      then M-RoPE at position 4p
//!   score     = Σ_h relu(q_h · pooled_pᵀ) / √128            [s, P]
//!   selected  = topk(score masked to complete visible blocks)
//! ```
//!
//! ## Pooling happens before the norm and the rotation
//!
//! The cache stores the *raw* key; `k_layernorm` and the rotation are applied
//! to the block mean, and the rotation uses the position of the block's
//! **first** member. Normalizing or rotating per token first and then averaging
//! is a different (and wrong) function — llama.cpp is explicit about the order
//! for exactly this reason.
//!
//! ## The tail is per query, not a global remainder
//!
//! Query `q` has seen `q + 1` tokens, so `(q + 1) % ratio` of them trail its
//! last complete block, and those are kept verbatim. "Every complete block at
//! or before `q`" ∪ "that tail" is exactly `0..=q` — which is why an
//! unbudgeted selection reproduces the causal mask, and why no query (not even
//! query 0, whose first block is not yet complete) ends up seeing nothing.
//!
//! ## When this is a no-op
//!
//! `select_k = min(budget / ratio, P)` with `P = floor(seq / ratio)`. For
//! `seq / ratio <= budget / ratio` there are never more complete blocks than
//! the budget, so every visible one is selected and the mask is exactly causal.
//! [`QsaDims::is_dense`] tests this and [`crate::attention`] then uses
//! `MaskKind::Causal` and the fused kernel rather than materializing an
//! `[s, s]` bias already known to be causal. The short-circuit is exact, not
//! an approximation — which also means that below the budget this whole module
//! is dead code at runtime, and the `[s, s]` constants it would allocate are
//! never built.
//!
//! ## What this emitter assumes
//!
//! A dense, unpadded, batch-1 prefill: all `seq` tokens are real. That makes
//! the reference's `local_visible_indices` the identity `0..=q`, so block
//! membership, block causality and the per-query tail are compile-time
//! constants. Padded batches and cached decode need the reference's dynamic
//! per-query `nonzero` walk and are not emitted here.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, BinaryOp, Op, ScatterNdReduction};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::common::{linear, rms_gamma, scalar};
use crate::config::Qwen4ExpConfig;

/// Additive penalty standing in for `-inf`; finite so that an all-masked row
/// cannot produce NaN, large enough to vanish under softmax.
const MASK_PENALTY: f32 = -1.0e30;

#[derive(Debug, Clone, Copy)]
pub struct QsaDims {
    pub hidden: usize,
    /// `indexer_n_heads`.
    pub n_heads: usize,
    /// `indexer_head_dim`.
    pub head_dim: usize,
    /// `indexer_budget` — tokens kept from complete blocks.
    pub budget: usize,
    /// `indexer_compress_ratio` — keys per block.
    pub ratio: usize,
    /// Rotary width (`rope_dim_count`), which must fit the index head.
    pub n_rot: usize,
    pub eps: f32,
    pub norm_offset: bool,
    pub seq: usize,
    /// Emit the selection even when [`Self::is_dense`] proves it is the
    /// identity.
    ///
    /// Exists because the short-circuit is otherwise untestable: selection is
    /// the identity *exactly when* `is_dense` holds, so there is no
    /// configuration in which the machinery runs and provably selects
    /// everything. Setting this runs it anyway, which is what lets a test
    /// compare the two paths and check they agree. Production leaves it
    /// `false`.
    pub force_emit: bool,
}

impl QsaDims {
    /// Derive from the model config, for a step of `seq` positions.
    ///
    /// There is exactly one statement of how a config becomes these dimensions, so
    /// prefill, chunked prefill, the decode step, the MTP block and the tests
    /// cannot disagree about what model they are building. They used to hold four
    /// copies of it.
    /// `force_emit` defaults off — set it to keep the selection machinery in
    /// the graph below the budget, which only a test wants.
    pub fn from_config(cfg: &Qwen4ExpConfig, seq: usize) -> Self {
        QsaDims {
            hidden: cfg.hidden_size,
            n_heads: cfg.indexer_n_heads,
            head_dim: cfg.indexer_head_dim,
            budget: cfg.indexer_budget,
            ratio: cfg.indexer_compress_ratio,
            n_rot: cfg.rope_dim_count,
            eps: cfg.rms_norm_eps,
            norm_offset: cfg.rms_norm_offset,
            seq,
            force_emit: false,
        }
    }
    /// Complete blocks in a `seq`-token prefill. The trailing `seq % ratio`
    /// keys never form a block; they reach a query through the tail.
    pub fn blocks(&self) -> usize {
        self.seq / self.ratio.max(1)
    }

    /// `min(budget / ratio, blocks)` — blocks that survive selection.
    pub fn select_k(&self) -> usize {
        (self.budget / self.ratio.max(1)).min(self.blocks())
    }

    /// Whether selection is the identity, i.e. the mask is just causal.
    pub fn is_dense(&self) -> bool {
        self.select_k() >= self.blocks()
    }
}

/// Emit the additive attention bias implied by QSA selection: `0` where a key
/// is selected and visible, `MASK_PENALTY` where it is not.
///
/// * `x` — `[1, seq, hidden]`, the same block input q/k/v read.
/// * `cos` / `sin` — the shared M-RoPE tables, `[seq, table_width]`.
/// * `attn_heads` — the main path's head count; the bias is expanded to it
///   because `MaskKind::Bias` indexes `[batch, num_heads, q, k]` and reads a
///   `[1, 1, s, s]` operand past its end.
///
/// Returns `Ok(None)` when [`QsaDims::is_dense`] — the caller should then use
/// `MaskKind::Causal` instead of a bias tensor.
pub fn emit_qsa_bias(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    cos: HirNodeId,
    sin: HirNodeId,
    d: QsaDims,
    attn_heads: usize,
) -> Result<Option<HirNodeId>> {
    if d.is_dense() && !d.force_emit {
        return Ok(None);
    }
    let f = DType::F32;
    let (s, p, r) = (d.seq, d.blocks(), d.ratio);
    let (nh, hd) = (d.n_heads, d.head_dim);
    let (si, pi, ri, nhi, hdi) = (s as i64, p as i64, r as i64, nh as i64, hd as i64);
    let sel = d.select_k();

    let x2d = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(x, vec![si, d.hidden as i64])
    };

    // ── projections ──
    // The converter split the reference's single `index_qk_proj` into these
    // two (the q half first, `indexer_n_heads * head_dim` rows of it).
    let q = linear(emit, &format!("{prefix}.indexer.q_proj.weight"), x2d)?;
    let k_raw = linear(emit, &format!("{prefix}.indexer.k_proj.weight"), x2d)?;
    let q_gain = rms_gamma(
        emit,
        &format!("{prefix}.indexer.q_norm.weight"),
        hd,
        d.norm_offset,
    )?;
    let k_gain = rms_gamma(
        emit,
        &format!("{prefix}.indexer.k_norm.weight"),
        hd,
        d.norm_offset,
    )?;
    let q_zb = emit.synth_zeros(&format!("{prefix}.indexer.q_norm.zb"), hd);
    let k_zb = emit.synth_zeros(&format!("{prefix}.indexer.k_norm.zb"), hd);

    // ── constants, hoisted before the graph builder borrows `emit` ──
    let block_visible = emit.synth_param(
        &format!("{prefix}.qsa.block_causal"),
        block_causal_mask(s, p, r),
        Shape::new(&[s, p], f),
    );
    let tail = emit.synth_param(
        &format!("{prefix}.qsa.tail"),
        tail_mask(s, r),
        Shape::new(&[s, s], f),
    );
    let zeros = {
        let z = emit.synth_zeros(&format!("{prefix}.qsa.zeros"), s * s);
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(z, vec![si, si])
    };
    let one = scalar(emit, &format!("{prefix}.qsa.one"), 1.0);
    let penalty = scalar(emit, &format!("{prefix}.qsa.neg"), MASK_PENALTY);
    let qk_scale = scalar(emit, &format!("{prefix}.qsa.scale"), (hd as f32).powf(-0.5));
    let inv_ratio = scalar(emit, &format!("{prefix}.qsa.inv_ratio"), 1.0 / r as f32);
    let ratio_c = scalar(emit, &format!("{prefix}.qsa.ratio"), r as f32);
    // `offsets[j] = j` turns a block id into its member token ids.
    let offsets = emit.synth_param(
        &format!("{prefix}.qsa.offsets"),
        (0..r).map(|j| j as f32).collect(),
        Shape::new(&[r], f),
    );
    // Row `p` of the M-RoPE table a block's pooled key rotates at — the
    // position of its FIRST member, `r·p`.
    let block_start = emit.synth_param(
        &format!("{prefix}.qsa.block_start"),
        (0..p).map(|i| (i * r) as f32).collect(),
        Shape::new(&[p], f),
    );

    let mut gb = HirMut::new(emit.hir());

    // ── query: per-head RMSNorm, then rotate at its own position ──
    let q4 = gb.reshape_(q, vec![1, si, nhi, hdi]);
    let q4 = gb.rms_norm(q4, q_gain, q_zb, d.eps);
    let q3 = gb.reshape_(q4, vec![1, si, (nh * hd) as i64]);
    let q3 = gb.rope_n(q3, cos, sin, hd, d.n_rot);

    // ── block keys: mean of the raw keys, THEN norm, THEN rotate ──
    let trimmed = gb.narrow_(k_raw, 0, 0, p * r);
    let grouped = gb.reshape_(trimmed, vec![pi, ri, hdi]);
    let summed = gb.sum(grouped, vec![1], false); // [P, hd]
    let pooled = gb.mul(summed, inv_ratio);
    let pooled = gb.rms_norm(pooled, k_gain, k_zb, d.eps);
    let pooled3 = gb.reshape_(pooled, vec![1, pi, hdi]);
    // Gathering the table rather than rebuilding it keeps the block keys on
    // exactly the positions the queries use, including a multimodal M-RoPE
    // layout the caller fed in.
    let cos_b = gb.gather_(cos, block_start, 0);
    let sin_b = gb.gather_(sin, block_start, 0);
    let pooled3 = gb.rope_n(pooled3, cos_b, sin_b, hd, d.n_rot);
    let pooled2 = gb.reshape_(pooled3, vec![pi, hdi]);

    // ── score every query against every block, per indexer head ──
    let qh = gb.reshape_(q3, vec![si, nhi, hdi]);
    let qh = gb.transpose_(qh, vec![1, 0, 2]); // [nh, s, hd]
    let pk_t = gb.transpose_(pooled2, vec![1, 0]); // [hd, P]
    let scores = gb.mm(qh, pk_t); // [nh, s, P]
    let scores = gb.add_node(
        Op::Activation(Activation::Relu),
        vec![scores],
        Shape::new(&[nh, s, p], f),
    );
    // The reference sums the rectified head scores UNWEIGHTED and scales once
    // by `head_dim^-0.5` — there is no per-head weight projection here, unlike
    // DeepSeek's lightning indexer.
    let index_scores = gb.sum(scores, vec![0], false); // [s, P]
    let index_scores = gb.mul(index_scores, qk_scale);

    // ── mask incomplete/invisible blocks, take the top `select_k` ──
    let inv = gb.sub(one, block_visible);
    let masked = {
        let pen = gb.mul(inv, penalty);
        gb.add(index_scores, pen)
    };
    let chosen = gb.add_node(Op::TopK { k: sel }, vec![masked], Shape::new(&[s, sel], f));

    // **Top-k does not filter.** It returns `select_k` indices whatever the
    // scores are, so a query with fewer visible blocks than the budget — every
    // query early in the sequence — gets blocks whose tokens are in its
    // future. Scattering those unmasked would let a query attend forwards.
    //
    // Re-read each selection's visibility and use it as the scatter's update,
    // so an invalid pick writes 0.0 instead of 1.0. `Max` reduction means a
    // token reached by both a valid and an invalid pick stays visible.
    let sel_valid = gb.add_node(
        Op::GatherElements { axis: 1 },
        vec![block_visible, chosen],
        Shape::new(&[s, sel], f),
    );

    // ── expand selected blocks to token ids and scatter visibility ──
    let base = gb.mul(chosen, ratio_c); // [s, sel]
    let mut visible = zeros;
    for j in 0..r {
        let off = gb.narrow_(offsets, 0, j, 1);
        let idx = gb.add(base, off);
        visible = gb.add_node(
            Op::ScatterElements {
                axis: 1,
                reduction: ScatterNdReduction::Max,
            },
            vec![visible, idx, sel_valid],
            Shape::new(&[s, s], f),
        );
    }

    // ── the query's own incomplete tail is always visible ──
    visible = gb.add_node(
        Op::Binary(BinaryOp::Max),
        vec![visible, tail],
        Shape::new(&[s, s], f),
    );

    // ── visibility → additive bias, per attention head ──
    let hidden_m = gb.sub(one, visible);
    let bias = gb.mul(hidden_m, penalty);
    let bias = gb.reshape_(bias, vec![1, 1, si, si]);
    Ok(Some(gb.expand_(bias, vec![1, attn_heads as i64, si, si])))
}

/// `[seq, blocks]`, 1.0 where block `p` is *complete* as of query `q` — its
/// last member `ratio·p + ratio − 1` is at or before `q`.
///
/// Mirrors the reference's `num_complete_blocks = len(visible) // ratio` for an
/// unpadded prefill, where `visible` is `0..=q`.
fn block_causal_mask(seq: usize, blocks: usize, ratio: usize) -> Vec<f32> {
    let mut m = vec![0.0; seq * blocks];
    for q in 0..seq {
        for p in 0..blocks {
            if p * ratio + ratio - 1 <= q {
                m[q * blocks + p] = 1.0;
            }
        }
    }
    m
}

/// `[seq, seq]`, 1.0 on the tokens query `q` has seen since its last *complete*
/// block — the reference's
/// `local_visible_indices[num_complete_blocks * ratio:]`.
fn tail_mask(seq: usize, ratio: usize) -> Vec<f32> {
    let mut m = vec![0.0; seq * seq];
    for q in 0..seq {
        let complete = (q + 1) / ratio;
        for t in (complete * ratio)..=q {
            m[q * seq + t] = 1.0;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_causality_needs_a_complete_block() {
        // seq 8, ratio 4 → blocks {0: tokens 0..3, 1: tokens 4..7}.
        let m = block_causal_mask(8, 2, 4);
        let at = |q: usize, p: usize| m[q * 2 + p];
        assert_eq!(at(2, 0), 0.0, "block 0 is not complete until token 3");
        assert_eq!(at(3, 0), 1.0);
        assert_eq!(at(3, 1), 0.0);
        assert_eq!(at(7, 1), 1.0);
    }

    #[test]
    fn the_tail_covers_the_queries_own_remainder() {
        let m = tail_mask(8, 4);
        let at = |q: usize, t: usize| m[q * 8 + t];
        // Query 0's first block is incomplete, so it reaches itself via the tail.
        assert_eq!(at(0, 0), 1.0);
        // Query 3 completes block 0 — nothing trails it.
        for t in 0..8 {
            assert_eq!(at(3, t), 0.0, "query 3 has an empty tail");
        }
        // Query 5 has block 0 complete and tokens 4,5 trailing.
        assert_eq!(at(5, 4), 1.0);
        assert_eq!(at(5, 5), 1.0);
        assert_eq!(at(5, 3), 0.0, "token 3 belongs to a complete block");
        assert_eq!(at(5, 6), 0.0, "the tail stays causal");
    }

    /// The union of "every complete visible block" and "the query's tail" is
    /// exactly `0..=q`. This is what makes an unbudgeted selection identical to
    /// dense causal attention, and therefore what licenses the short-circuit.
    #[test]
    fn blocks_plus_tail_reconstruct_the_causal_mask() {
        for &(seq, ratio) in &[(8usize, 4usize), (10, 4), (7, 3), (16, 4), (13, 2)] {
            let blocks = seq / ratio;
            let bm = block_causal_mask(seq, blocks, ratio);
            let tm = tail_mask(seq, ratio);
            for q in 0..seq {
                let mut seen = vec![false; seq];
                for p in 0..blocks {
                    if bm[q * blocks + p] == 1.0 {
                        for j in 0..ratio {
                            seen[p * ratio + j] = true;
                        }
                    }
                }
                for (t, s) in seen.iter_mut().enumerate() {
                    if tm[q * seq + t] == 1.0 {
                        *s = true;
                    }
                }
                for (t, s) in seen.iter().enumerate() {
                    assert_eq!(
                        *s,
                        t <= q,
                        "seq={seq} ratio={ratio} q={q} t={t}: visibility must be causal"
                    );
                }
            }
        }
    }

    fn dims(seq: usize, budget: usize) -> QsaDims {
        QsaDims {
            hidden: 32,
            n_heads: 2,
            head_dim: 8,
            budget,
            ratio: 4,
            n_rot: 4,
            eps: 1e-6,
            norm_offset: false,
            seq,
            force_emit: false,
        }
    }

    #[test]
    fn selection_saturates_below_the_budget() {
        let d = dims(64, 2048);
        assert_eq!(d.blocks(), 16);
        assert_eq!(d.select_k(), 16);
        assert!(d.is_dense());
    }

    #[test]
    fn selection_engages_past_the_budget() {
        let d = dims(64, 8);
        assert_eq!(d.select_k(), 2);
        assert!(!d.is_dense());
    }
}
