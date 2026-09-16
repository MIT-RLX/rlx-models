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

//! DFlash2 candidate selector — a low-rank bilinear lattice over adjacent
//! block positions.
//!
//! ```text
//! S_t(a, b) = U_t(b) + ⟨A[a] ⊙ H(h_t), B[b]⟩
//! ```
//!
//! A block drafter picks each position's token independently, so it will
//! happily emit a locally-plausible pair that reads as nonsense together. The
//! selector keeps the top `k` candidates per position and scores every
//! *adjacent pair* — `A` and `B` are rank-256 codebooks for the predecessor and
//! successor, `H(h_t)` is a context gate deciding which codebook components
//! matter here, and `U_t` is the drafter's own logit for the candidate. A cheap
//! chain walk over that lattice then picks a coherent path.
//!
//! ## Two outputs, not one packed row
//!
//! The reference concatenates ids and scores into an `n_embd`-wide row and
//! pads, because the lattice has to ride the fixed `h_nextn` embeddings
//! channel back to the host — which is where its load-time constraint
//! `n_embd >= top_k * (top_k + 1)` comes from. rlx graphs declare their own
//! outputs, so this emits `cand` and `scores` as separate tensors and the
//! constraint disappears.
//!
//! What it keeps is the part that actually matters: **full-vocab logits never
//! leave the device.** The host sees `k + k²` floats per position instead of
//! `n_vocab`.

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Philox4x32, Shape};

use crate::config::Dflash2Config;

/// One block position's proposal distribution, over the selector's slate.
///
/// Mirrors `rlx_runtime::spec_decode::SparseDist`, which is what the verifier
/// consumes. It is restated here so this crate keeps building against the
/// published rlx; convert at the call site once that type is released.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CandidateDist {
    pub ids: Vec<u32>,
    pub probs: Vec<f32>,
}

/// How the chain walk resolves each step.
#[derive(Debug, Clone, Copy)]
pub enum SelectorSampling {
    /// Take the highest-scoring successor. No proposal distributions are
    /// produced, so verification falls back to the greedy prefix rule.
    Greedy,
    /// Sample `softmax(scores / temperature)`, recording the distribution so
    /// the target can verify by maximal coupling and stay exact.
    Temperature(f32),
}

/// Host-side view of the selector's two output tensors, for one batch of
/// blocks.
#[derive(Debug, Clone)]
pub struct SelectorLattice {
    /// Blocks in flight.
    pub batch: usize,
    /// Tokens per block, slot 0 being the anchor.
    pub block: usize,
    /// Candidates per position.
    pub top_k: usize,
    /// `[batch, block, top_k]` — token ids, as read back from `Op::TopK`
    /// (f32-encoded, rounded here).
    pub cand_ids: Vec<u32>,
    /// `[batch, block - 1, top_k, top_k]` — `scores[.., pos-1, a, b]` is the
    /// score of successor `b` at `pos` given predecessor `a` at `pos - 1`.
    pub scores: Vec<f32>,
}

/// One block's drafted continuation.
#[derive(Debug, Clone, Default)]
pub struct DraftBlock {
    /// `block - 1` proposed tokens (the anchor is already committed).
    pub tokens: Vec<u32>,
    /// One distribution per proposed token; empty under [`SelectorSampling::Greedy`].
    pub dists: Vec<CandidateDist>,
}

/// Emit the selector lattice.
///
/// * `logits`     — `[batch, block, vocab]`, the drafter's own logits.
/// * `hgate`      — `H(h_t)`, `[batch, block, rank]`: the hidden states already
///   projected through the selector's `H`. Taking the *projection* rather than
///   the weight keeps this emitter out of the quantization business — in the
///   released checkpoints `selector_hidden.weight` ships as Q4_K, so it reaches
///   the graph as a packed U8 blob that only `Op::DequantMatMul` can consume.
/// * `anchor`     — `[batch, 1]`, f32-encoded id of each block's committed
///   last token. It seeds position 1's predecessor, which is what ties the
///   drafted block to what the target already accepted.
/// * `sel_prev`   — `A`, `[vocab, rank]`, dequantized.
/// * `sel_next`   — `B`, `[vocab, rank]`, dequantized.
///
/// Returns `(cand, scores)` with shapes `[batch, block, k]` and
/// `[batch, block - 1, k, k]`. Position 1 has a single predecessor (the
/// anchor); its scores are broadcast across all `k` predecessor rows so
/// [`walk_lattice`] can index every position identically.
#[allow(clippy::too_many_arguments)]
pub fn emit_selector_lattice(
    g: &mut Graph,
    logits: NodeId,
    hgate: NodeId,
    anchor: NodeId,
    sel_prev: NodeId,
    sel_next: NodeId,
    cfg: &Dflash2Config,
) -> (NodeId, NodeId) {
    let ls = g.shape(logits).clone();
    assert_eq!(
        ls.rank(),
        3,
        "selector expects logits [batch, block, vocab]"
    );
    let (b, s) = (ls.dim(0).unwrap_static(), ls.dim(1).unwrap_static());
    assert!(
        s >= 2,
        "a selector lattice needs an anchor plus at least one draft slot"
    );
    let k = cfg.selector_top_k;
    let rank = cfg.selector_rank;
    let f = DType::F32;

    // Top-k slate per position, and the drafter's own logit for each — the
    // unary term. `Op::TopK` yields f32-encoded indices, which both `Gather`
    // and `GatherElements` accept directly.
    let cand = g.add_node(Op::TopK { k }, vec![logits], Shape::new(&[b, s, k], f));
    let unary = g.gather_elements(logits, cand, 2); // [B, S, k]

    let expand = |g: &mut Graph, x: NodeId, to: Vec<usize>| -> NodeId {
        let target: Vec<i64> = to.iter().map(|d| *d as i64).collect();
        g.add_node(
            Op::Expand {
                target_shape: target,
            },
            vec![x],
            Shape::new(&to, f),
        )
    };

    let mut rows: Vec<NodeId> = Vec::with_capacity(s - 1);
    for pos in 1..s {
        // Successors: this position's slate.
        let ids = g.narrow_(cand, 1, pos, 1);
        let ids = g.reshape_(ids, vec![b as i64, k as i64]);
        let brows = g.gather_(sel_next, ids, 0); // [B, k, rank]
        let brows = g.transpose_(brows, vec![0, 2, 1]); // [B, rank, k]

        // Predecessors: the anchor at pos 1, the previous slate after that.
        let (prev, kp) = if pos == 1 {
            (anchor, 1)
        } else {
            let p = g.narrow_(cand, 1, pos - 1, 1);
            (g.reshape_(p, vec![b as i64, k as i64]), k)
        };
        let arows = g.gather_(sel_prev, prev, 0); // [B, kp, rank]

        // A[a] ⊙ H(h_t): the context gate masks the codebook per position.
        let hp = g.narrow_(hgate, 1, pos, 1);
        let hp = g.reshape_(hp, vec![b as i64, 1, rank as i64]);
        let hp = expand(g, hp, vec![b, kp, rank]);
        let cond = g.mul(arows, hp); // [B, kp, rank]

        // ⟨·, B[b]⟩ + U_t(b)
        let sc = g.mm(cond, brows); // [B, kp, k]
        let un = g.narrow_(unary, 1, pos, 1);
        let un = g.reshape_(un, vec![b as i64, 1, k as i64]);
        let un = expand(g, un, vec![b, kp, k]);
        let sc = g.add(sc, un);

        let sc = if kp == k {
            sc
        } else {
            expand(g, sc, vec![b, k, k])
        };
        rows.push(g.reshape_(sc, vec![b as i64, 1, k as i64, k as i64]));
    }

    let scores = if rows.len() == 1 {
        rows[0]
    } else {
        g.concat_(rows, 1)
    };
    (cand, scores)
}

/// Walk the lattice, one path per block.
///
/// Greedy takes the best successor at each step; `Temperature` samples and
/// records the slate's distribution so the target can verify by maximal
/// coupling. Either way the predecessor index carries forward, which is the
/// whole point — position `t + 1` is scored against the token actually chosen
/// at `t`, not against `t`'s independent argmax.
///
/// `n_min` drops a block whose path came out shorter than the caller considers
/// worth verifying; the reference does the same, since a 1-token draft costs a
/// target forward and repays almost nothing.
pub fn walk_lattice(
    lattice: &SelectorLattice,
    mode: SelectorSampling,
    n_min: usize,
    rng: &mut Philox4x32,
) -> Vec<DraftBlock> {
    let (bs, block, k) = (lattice.batch, lattice.block, lattice.top_k);
    assert_eq!(
        lattice.cand_ids.len(),
        bs * block * k,
        "cand_ids must be [batch, block, top_k]"
    );
    assert_eq!(
        lattice.scores.len(),
        bs * (block - 1) * k * k,
        "scores must be [batch, block - 1, top_k, top_k]"
    );

    let mut out = Vec::with_capacity(bs);
    for bi in 0..bs {
        let mut blk = DraftBlock::default();
        let mut predecessor = 0usize;

        for pos in 1..block {
            let row =
                &lattice.scores[((bi * (block - 1)) + (pos - 1)) * k * k + predecessor * k..][..k];
            let ids = &lattice.cand_ids[(bi * block + pos) * k..][..k];

            match mode {
                SelectorSampling::Greedy => {
                    predecessor = argmax(row);
                }
                SelectorSampling::Temperature(t) => {
                    let probs = softmax(row, t);
                    predecessor = sample(&probs, rng);
                    blk.dists.push(CandidateDist {
                        ids: ids.to_vec(),
                        probs,
                    });
                }
            }
            blk.tokens.push(ids[predecessor]);
        }

        if blk.tokens.len() < n_min {
            blk = DraftBlock::default();
        }
        out.push(blk);
    }
    out
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, v) in xs.iter().enumerate() {
        if *v > xs[best] {
            best = i;
        }
    }
    best
}

/// `softmax(x / t)`, max-shifted. A non-positive temperature would divide by
/// zero, so it degenerates to a one-hot on the argmax — the greedy answer.
fn softmax(xs: &[f32], t: f32) -> Vec<f32> {
    if t <= 0.0 {
        let mut p = vec![0f32; xs.len()];
        p[argmax(xs)] = 1.0;
        return p;
    }
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = xs.iter().map(|v| ((v - max) / t).exp()).collect();
    let sum: f32 = p.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in p.iter_mut() {
            *v *= inv;
        }
    }
    p
}

fn sample(probs: &[f32], rng: &mut Philox4x32) -> usize {
    let r = rng.next_f32();
    let mut acc = 0f32;
    for (i, p) in probs.iter().enumerate() {
        acc += *p;
        if r <= acc {
            return i;
        }
    }
    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_runtime::{Device, Session};

    const V: usize = 8;

    fn cfg() -> Dflash2Config {
        Dflash2Config {
            conv_kernel_size: 2,
            conv_group_size: 2,
            selector_rank: 3,
            selector_top_k: 2,
        }
    }

    fn deterministic(n: usize, seed: u64) -> Vec<f32> {
        let mut st = seed;
        (0..n)
            .map(|_| {
                st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                (((st >> 33) as f64 / (1u64 << 30) as f64) - 1.0) as f32 * 0.5
            })
            .collect()
    }

    /// `S_t(a,b) = U_t(b) + ⟨A[a] ⊙ H(h_t), B[b]⟩`, written straight from the
    /// paper rather than from the emitter.
    #[allow(clippy::too_many_arguments)]
    fn oracle(
        logits: &[f32],
        embd: &[f32],
        anchor: &[f32],
        a_cb: &[f32],
        b_cb: &[f32],
        h_w: &[f32],
        b: usize,
        s: usize,
        h: usize,
        c: &Dflash2Config,
    ) -> (Vec<u32>, Vec<f32>) {
        let (k, rank) = (c.selector_top_k, c.selector_rank);
        let mut cand = vec![0u32; b * s * k];
        for bi in 0..b {
            for p in 0..s {
                let row = &logits[(bi * s + p) * V..][..V];
                let mut order: Vec<usize> = (0..V).collect();
                // Ties break towards the smaller index, matching Op::TopK.
                order.sort_by(|x, y| row[*y].partial_cmp(&row[*x]).unwrap().then(x.cmp(y)));
                for j in 0..k {
                    cand[(bi * s + p) * k + j] = order[j] as u32;
                }
            }
        }

        let mut scores = vec![0f32; b * (s - 1) * k * k];
        for bi in 0..b {
            for p in 1..s {
                // H(h_t) = embd[p] @ h_w
                let mut hg = vec![0f32; rank];
                for r in 0..rank {
                    for j in 0..h {
                        hg[r] += embd[(bi * s + p) * h + j] * h_w[j * rank + r];
                    }
                }
                let preds: Vec<u32> = if p == 1 {
                    vec![anchor[bi] as u32]
                } else {
                    cand[(bi * s + p - 1) * k..][..k].to_vec()
                };
                for a in 0..k {
                    // Position 1's single anchor row is repeated across a.
                    let pid = preds[if p == 1 { 0 } else { a }] as usize;
                    for bb in 0..k {
                        let sid = cand[(bi * s + p) * k + bb] as usize;
                        let mut dot = 0f32;
                        for r in 0..rank {
                            dot += a_cb[pid * rank + r] * hg[r] * b_cb[sid * rank + r];
                        }
                        let unary = logits[(bi * s + p) * V + sid];
                        scores[((bi * (s - 1)) + (p - 1)) * k * k + a * k + bb] = dot + unary;
                    }
                }
            }
        }
        (cand, scores)
    }

    #[test]
    fn lattice_matches_oracle() {
        let c = cfg();
        let (b, s, h) = (2usize, 4usize, 4usize);
        let (k, rank) = (c.selector_top_k, c.selector_rank);

        let logits = deterministic(b * s * V, 0x243f_6a88_85a3_08d3);
        let embd = deterministic(b * s * h, 0x1319_8a2e_0370_7344);
        let a_cb = deterministic(V * rank, 0xa409_3822_299f_31d0);
        let b_cb = deterministic(V * rank, 0x082e_fa98_ec4e_6c89);
        let h_w = deterministic(h * rank, 0x4528_21e6_38d0_1377);
        let anchor: Vec<f32> = (0..b).map(|i| (i * 3 + 1) as f32).collect();

        let mut g = Graph::new("selector_test");
        let li = g.input("logits", Shape::new(&[b, s, V], DType::F32));
        let ei = g.input("embd", Shape::new(&[b, s, h], DType::F32));
        let ai = g.input("anchor", Shape::new(&[b, 1], DType::F32));
        let ap = g.param("A", Shape::new(&[V, rank], DType::F32));
        let bp = g.param("B", Shape::new(&[V, rank], DType::F32));
        let hp = g.param("H", Shape::new(&[h, rank], DType::F32));
        let hgate = g.mm(ei, hp);
        let (cand, scores) = emit_selector_lattice(&mut g, li, hgate, ai, ap, bp, &c);
        g.set_outputs(vec![cand, scores]);

        let mut compiled = Session::new(Device::Cpu).compile(g);
        compiled.set_param("A", &a_cb);
        compiled.set_param("B", &b_cb);
        compiled.set_param("H", &h_w);
        let out = compiled.run(&[
            ("logits", logits.as_slice()),
            ("embd", embd.as_slice()),
            ("anchor", anchor.as_slice()),
        ]);

        let (want_cand, want_scores) =
            oracle(&logits, &embd, &anchor, &a_cb, &b_cb, &h_w, b, s, h, &c);

        assert_eq!(out[0].len(), b * s * k);
        for (i, (got, exp)) in out[0].iter().zip(&want_cand).enumerate() {
            assert_eq!(*got as u32, *exp, "candidate {i}");
        }
        assert_eq!(out[1].len(), b * (s - 1) * k * k);
        for (i, (got, exp)) in out[1].iter().zip(&want_scores).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "score {i}: graph {got} vs oracle {exp}"
            );
        }
    }

    /// The chain must follow the predecessor it actually chose. A walk that
    /// re-read row 0 every step would pick `[1, 1]` here instead of `[1, 3]`.
    #[test]
    fn walk_conditions_on_the_chosen_predecessor() {
        // block = 3 (anchor + 2 draft slots), k = 2.
        let lattice = SelectorLattice {
            batch: 1,
            block: 3,
            top_k: 2,
            cand_ids: vec![
                0, 0, // anchor slot, unused
                1, 2, // pos 1 slate
                3, 4, // pos 2 slate
            ],
            scores: vec![
                // pos 1: predecessor rows identical (anchor), successor 0 wins
                9.0, 1.0, 9.0, 1.0, //
                // pos 2: row 0 prefers successor 0 (id 3), row 1 prefers 1 (id 4)
                5.0, 0.0, 0.0, 5.0,
            ],
        };
        let mut rng = Philox4x32::new(1);
        let got = walk_lattice(&lattice, SelectorSampling::Greedy, 0, &mut rng);
        assert_eq!(got[0].tokens, vec![1, 3]);
        assert!(got[0].dists.is_empty(), "greedy records no distributions");
    }

    /// Sampling must record the slate it sampled from, or the verifier cannot
    /// run maximal coupling and the draft stops being distribution-exact.
    #[test]
    fn temperature_walk_records_slate_distributions() {
        let lattice = SelectorLattice {
            batch: 1,
            block: 3,
            top_k: 2,
            cand_ids: vec![0, 0, 1, 2, 3, 4],
            // Equal scores → uniform over the slate at every step.
            scores: vec![0.0; 2 * 2 * 2],
        };
        let mut rng = Philox4x32::new(7);
        let got = walk_lattice(&lattice, SelectorSampling::Temperature(1.0), 0, &mut rng);
        assert_eq!(got[0].tokens.len(), 2);
        assert_eq!(got[0].dists.len(), 2);
        assert_eq!(got[0].dists[0].ids, vec![1, 2]);
        for p in &got[0].dists[0].probs {
            assert!((p - 0.5).abs() < 1e-6, "uniform slate expected, got {p}");
        }
        // Every emitted token must come from its own position's slate.
        assert!(got[0].dists[1].ids.contains(&got[0].tokens[1]));
    }

    #[test]
    fn n_min_drops_a_draft_too_short_to_pay_for_itself() {
        let lattice = SelectorLattice {
            batch: 1,
            block: 3,
            top_k: 2,
            cand_ids: vec![0, 0, 1, 2, 3, 4],
            scores: vec![0.0; 2 * 2 * 2],
        };
        let mut rng = Philox4x32::new(1);
        let got = walk_lattice(&lattice, SelectorSampling::Greedy, 5, &mut rng);
        assert!(got[0].tokens.is_empty());
    }
}
