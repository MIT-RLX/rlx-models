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

//! `Glm5NextTextIndexer` — the DeepSeek-sparse-attention (DSA) lightning
//! indexer, with GLM-5.3-Flash's k-pool compression.
//!
//! The indexer is a cheap side network (32 heads × 128, versus the main path's
//! 64 × 256) that decides which keys each query may see. GLM compresses
//! candidates first: `index_kpool = 4` consecutive tokens pool into one
//! candidate under a learned per-channel softmax over `gate + ape`, so the
//! top-k budget buys `index_topk / index_kpool = 512` *pools* — 2048 tokens —
//! plus each query's own incomplete tail.
//!
//! ```text
//!   q        = wq_b(q_resid)                            [s, 32, 128]
//!   k        = LayerNorm(wk(x))                         [s, 128]
//!   gate     = kpool_compress_gate(x)                   [s, 128]
//!
//!   pooled_p = Σ_j softmax_j(gate[4p+j] + ape[j]) · k[4p+j]
//!   score    = relu(q · pooled_pᵀ · 128^-0.5)           [32, s, P]
//!   index    = Σ_h weights_proj(x)[h] · 32^-0.5 · score [s, P]
//!   selected = topk(index masked to complete visible pools)
//! ```
//!
//! ## The tail is per query, not a global remainder
//!
//! `append_visible_tail` recomputes the incomplete pool *for each query*:
//! query `q` has seen `q + 1` tokens, so `(q + 1) % kpool` of them trail the
//! last complete pool, and those are appended verbatim. Together, "every
//! complete pool at or before `q`" plus "that tail" is exactly `0..=q` — which
//! is why an unbudgeted selection reproduces the causal mask, and why no query
//! (not even query 0, whose first pool is not yet complete) ends up seeing
//! nothing.
//!
//! ## When this is a no-op
//!
//! `select_k = min(index_topk / index_kpool, P)` with `P = floor(seq / kpool)`.
//! For `seq <= index_topk` there are never more complete pools than the budget,
//! so every visible pool is selected and the mask is exactly causal.
//! [`IndexerDims::is_dense`] tests this and the MLA layer then skips the
//! indexer entirely, using `MaskKind::Causal` and the fused attention kernel
//! rather than materializing an `[s, s]` bias already known to be causal. The
//! short-circuit is exact, not an approximation.
//!
//! ## What this emitter assumes
//!
//! A dense, unpadded, batch-1 prefill: all `seq` tokens are real. That makes
//! the reference's `first_key` zero and its `valid_keys` all true, so pool
//! membership, pool causality and the per-query tail are all compile-time
//! constants. Padded batches and cached decode need the reference's dynamic
//! `first_key` walk and are not emitted here.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, BinaryOp, Op, ScatterNdReduction};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::common::linear;

/// LayerNorm epsilon inside the indexer's `k_norm` — hardcoded `1e-6`
/// upstream, *not* `config.rms_norm_eps`.
const K_NORM_EPS: f32 = 1e-6;

/// Additive penalty standing in for `-inf`; finite so that an all-masked row
/// cannot produce NaN, large enough to vanish under softmax.
const MASK_PENALTY: f32 = -1.0e30;

#[derive(Debug, Clone, Copy)]
pub struct IndexerDims {
    pub hidden: usize,
    pub q_lora_rank: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub topk: usize,
    pub kpool: usize,
    pub always_select_tail: bool,
    pub seq: usize,
    /// Emit the selection even when [`Self::is_dense`] proves it is the
    /// identity.
    ///
    /// Exists because the short-circuit is otherwise untestable: selection is
    /// the identity *exactly when* `is_dense` holds, so there is no
    /// configuration in which the machinery runs and provably selects
    /// everything. Setting this runs it anyway, which is what lets
    /// `dsa_selection_is_the_identity_below_the_budget` compare the two paths
    /// and check they agree. Production code leaves it `false`.
    pub force_emit: bool,
}

impl IndexerDims {
    /// Complete pools in a `seq`-token prefill. The trailing `seq % kpool`
    /// tokens never form a pool candidate; they reach a query through the tail.
    pub fn pools(&self) -> usize {
        self.seq / self.kpool
    }

    /// `min(topk / kpool, pools)` — how many pools survive selection.
    pub fn select_k(&self) -> usize {
        (self.topk / self.kpool).min(self.pools())
    }

    /// Whether selection is the identity, i.e. the mask is just causal.
    pub fn is_dense(&self) -> bool {
        self.select_k() >= self.pools()
    }
}

/// Emit the additive attention bias `[1, 1, seq, seq]` implied by the indexer's
/// selection: `0` where a key is selected and visible, `MASK_PENALTY` where
/// it is not.
///
/// Returns `Ok(None)` when [`IndexerDims::is_dense`] — the caller should then
/// use `MaskKind::Causal` instead of a bias tensor.
pub fn emit_dsa_bias(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,  // [1, seq, hidden]
    q_resid: HirNodeId, // [seq, q_lora_rank]
    d: IndexerDims,
) -> Result<Option<HirNodeId>> {
    if d.is_dense() && !d.force_emit {
        return Ok(None);
    }
    let f = DType::F32;
    let (s, p, k) = (d.seq, d.pools(), d.kpool);
    let (nh, hd) = (d.n_heads, d.head_dim);
    let (si, pi, ki, nhi, hdi) = (s as i64, p as i64, k as i64, nh as i64, hd as i64);
    let sel = d.select_k();

    let x2d = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(hidden, vec![si, d.hidden as i64])
    };

    // ── indexer projections ──
    let q = linear(emit, &format!("{prefix}.indexer.attn_q_b.weight"), q_resid)?; // [s, nh*hd]
    let k_raw = linear(emit, &format!("{prefix}.indexer.attn_k.weight"), x2d)?; // [s, hd]
    let k_gain = emit.load_param(&format!("{prefix}.indexer.k_norm.weight"), false)?;
    let k_bias = emit.load_param(&format!("{prefix}.indexer.k_norm.bias"), false)?;
    let gate = linear(
        emit,
        &format!("{prefix}.indexer_compressor_gate.weight"),
        x2d,
    )?; // [s, hd]
    let ape = emit.load_param(&format!("{prefix}.indexer_compressor_ape.weight"), false)?; // [kpool, hd]
    let head_w = linear(emit, &format!("{prefix}.indexer.proj.weight"), x2d)?; // [s, nh]

    // ── every constant this block needs, hoisted before the graph builder
    //    borrows `emit` ──
    let pool_visible = emit.synth_param(
        &format!("{prefix}.dsa.pool_causal"),
        pool_causal_mask(s, p, k),
        Shape::new(&[s, p], f),
    );
    let tail = d.always_select_tail.then(|| {
        emit.synth_param(
            &format!("{prefix}.dsa.tail"),
            tail_mask(s, k),
            Shape::new(&[s, s], f),
        )
    });
    let zeros = {
        let z = emit.synth_zeros(&format!("{prefix}.dsa.zeros"), s * s);
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(z, vec![si, si])
    };
    let one = crate::common::scalar(emit, &format!("{prefix}.dsa.one"), 1.0);
    let penalty_c = crate::common::scalar(emit, &format!("{prefix}.dsa.neg"), MASK_PENALTY);
    let qk_scale =
        crate::common::scalar(emit, &format!("{prefix}.dsa.scale"), (hd as f32).powf(-0.5));
    let head_scale = crate::common::scalar(
        emit,
        &format!("{prefix}.dsa.hscale"),
        (nh as f32).powf(-0.5),
    );
    // `offsets[j] = j·1.0` turns a pool id into its member token ids.
    let offsets = emit.synth_param(
        &format!("{prefix}.dsa.offsets"),
        (0..k).map(|j| j as f32).collect(),
        Shape::new(&[k], f),
    );
    let kpool_c = crate::common::scalar(emit, &format!("{prefix}.dsa.kpool"), k as f32);

    let mut gb = HirMut::new(emit.hir());

    // `nn.LayerNorm`, not RMSNorm — it subtracts the mean.
    let keys = gb.ln(k_raw, k_gain, k_bias, K_NORM_EPS);

    // ── pool the complete pools → [P, hd] ──
    let trimmed_k = gb.narrow_(keys, 0, 0, p * k);
    let grouped_k = gb.reshape_(trimmed_k, vec![pi, ki, hdi]);
    let trimmed_g = gb.narrow_(gate, 0, 0, p * k);
    let grouped_g = gb.reshape_(trimmed_g, vec![pi, ki, hdi]);
    let ape3 = gb.reshape_(ape, vec![1, ki, hdi]);
    let logits = gb.add(grouped_g, ape3);
    // Softmax runs over the token-within-pool axis, independently per channel.
    let probs = gb.sm(logits, 1);
    let weighted = gb.mul(probs, grouped_k);
    let pool_keys = gb.sum(weighted, vec![1], false); // [P, hd]

    // ── score every query against every pool, per indexer head ──
    let q3 = gb.reshape_(q, vec![si, nhi, hdi]);
    let q3 = gb.transpose_(q3, vec![1, 0, 2]); // [nh, s, hd]
    let pk_t = gb.transpose_(pool_keys, vec![1, 0]); // [hd, P]
    let scores = gb.mm(q3, pk_t); // [nh, s, P]
    let scores = gb.mul(scores, qk_scale);
    let scores = gb.add_node(
        Op::Activation(Activation::Relu),
        vec![scores],
        Shape::new(&[nh, s, p], f),
    );

    // ── weight per head, sum across heads ──
    let w = gb.mul(head_w, head_scale); // [s, nh]
    let w = gb.transpose_(w, vec![1, 0]); // [nh, s]
    let w = gb.reshape_(w, vec![nhi, si, 1]);
    let contrib = gb.mul(scores, w);
    let index_scores = gb.sum(contrib, vec![0], false); // [s, P]

    // ── mask incomplete/invisible pools, take the top `select_k` ──
    let inv = gb.sub(one, pool_visible);
    let masked = {
        let pen = gb.mul(inv, penalty_c);
        gb.add(index_scores, pen)
    };
    let chosen = gb.add_node(Op::TopK { k: sel }, vec![masked], Shape::new(&[s, sel], f));

    // **Top-k does not filter.** It returns `select_k` indices whatever the
    // scores are, so a query with fewer visible pools than the budget — every
    // query early in the sequence, and all of them for a short prompt — gets
    // pools whose tokens are in its future. Scattering those unmasked would let
    // a query attend forwards.
    //
    // This is the reference's `selected_valid` step. Re-read each selection's
    // visibility and use it as the scatter's update, so an invalid pick writes
    // 0.0 instead of 1.0. `Max` reduction means a token reached by both a valid
    // and an invalid pick still ends up visible.
    let sel_valid = gb.add_node(
        Op::GatherElements { axis: 1 },
        vec![pool_visible, chosen],
        Shape::new(&[s, sel], f),
    );

    // ── expand selected pools back to token ids and scatter visibility ──
    // A *valid* selected pool is complete and ends at or before `q`, so every
    // token it contributes is causal; `sel_valid` is what makes that hold.
    let base = gb.mul(chosen, kpool_c); // [s, sel]
    let mut visible = zeros;
    for j in 0..k {
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
    if let Some(tail) = tail {
        visible = gb.add_node(
            Op::Binary(BinaryOp::Max),
            vec![visible, tail],
            Shape::new(&[s, s], f),
        );
    }

    // ── visibility → additive bias ──
    let hidden_m = gb.sub(one, visible);
    let bias = gb.mul(hidden_m, penalty_c);
    Ok(Some(gb.reshape_(bias, vec![1, 1, si, si])))
}

/// `[seq, pools]`, 1.0 where pool `p` is *complete* as of query `q` — its last
/// token `kpool·p + kpool − 1` is at or before `q`. Mirrors the reference's
/// `pool_visible & pool_valid` for an unpadded prefill.
fn pool_causal_mask(seq: usize, pools: usize, kpool: usize) -> Vec<f32> {
    let mut m = vec![0.0; seq * pools];
    for q in 0..seq {
        for p in 0..pools {
            if p * kpool + kpool - 1 <= q {
                m[q * pools + p] = 1.0;
            }
        }
    }
    m
}

/// `[seq, seq]`, 1.0 on the tokens query `q` has seen since its last *complete*
/// pool — `append_visible_tail` with `first_key = 0` and no padding.
fn tail_mask(seq: usize, kpool: usize) -> Vec<f32> {
    let mut m = vec![0.0; seq * seq];
    for q in 0..seq {
        let complete = (q + 1) / kpool;
        for t in (complete * kpool)..=q {
            m[q * seq + t] = 1.0;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_causality_needs_a_complete_pool() {
        // seq 8, kpool 4 → pools {0: tokens 0..3, 1: tokens 4..7}.
        let m = pool_causal_mask(8, 2, 4);
        let at = |q: usize, p: usize| m[q * 2 + p];
        assert_eq!(at(2, 0), 0.0, "pool 0 is not complete until token 3");
        assert_eq!(at(3, 0), 1.0);
        assert_eq!(at(3, 1), 0.0);
        assert_eq!(at(7, 1), 1.0);
    }

    #[test]
    fn tail_covers_the_queries_own_remainder() {
        let m = tail_mask(8, 4);
        let at = |q: usize, t: usize| m[q * 8 + t];
        // Query 0's first pool is incomplete, so it reaches itself via the tail.
        assert_eq!(at(0, 0), 1.0);
        // Query 3 completes pool 0 — nothing trails it.
        for t in 0..8 {
            assert_eq!(at(3, t), 0.0, "query 3 has an empty tail");
        }
        // Query 5 has pool 0 complete and tokens 4,5 trailing.
        assert_eq!(at(5, 4), 1.0);
        assert_eq!(at(5, 5), 1.0);
        assert_eq!(at(5, 3), 0.0, "token 3 belongs to a complete pool");
        assert_eq!(at(5, 6), 0.0, "the tail stays causal");
    }

    /// The union of "every complete visible pool" and "the query's tail" is
    /// exactly `0..=q`. This is what makes an unbudgeted selection identical to
    /// dense causal attention.
    #[test]
    fn pools_plus_tail_reconstruct_the_causal_mask() {
        for &(seq, kpool) in &[(8usize, 4usize), (10, 4), (7, 3), (16, 4), (5, 1)] {
            let pools = seq / kpool;
            let pm = pool_causal_mask(seq, pools, kpool);
            let tm = tail_mask(seq, kpool);
            for q in 0..seq {
                let mut seen = vec![false; seq];
                for p in 0..pools {
                    if pm[q * pools + p] == 1.0 {
                        for j in 0..kpool {
                            seen[p * kpool + j] = true;
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
                        "seq={seq} kpool={kpool} q={q} t={t}: visibility must be causal"
                    );
                }
            }
        }
    }

    #[test]
    fn select_k_saturates_below_the_topk_budget() {
        let d = IndexerDims {
            hidden: 16,
            q_lora_rank: 8,
            n_heads: 2,
            head_dim: 4,
            topk: 2048,
            kpool: 4,
            always_select_tail: true,
            seq: 64,
            force_emit: false,
        };
        assert_eq!(d.pools(), 16);
        assert_eq!(d.select_k(), 16);
        assert!(d.is_dense());
    }

    #[test]
    fn selection_engages_past_the_budget() {
        let d = IndexerDims {
            hidden: 16,
            q_lora_rank: 8,
            n_heads: 2,
            head_dim: 4,
            topk: 8,
            kpool: 4,
            always_select_tail: true,
            seq: 64,
            force_emit: false,
        };
        assert_eq!(d.select_k(), 2);
        assert!(!d.is_dense());
    }
}
