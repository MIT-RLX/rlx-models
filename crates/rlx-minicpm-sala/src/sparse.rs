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

//! InfLLM-v2 block-sparse attention, the `minicpm4` layers' mode once the
//! keys reach `sparse_config.dense_len`.
//!
//! What OpenBMB's `infllm_v2` CUDA kernels compute, per KV head and query:
//!
//! 1. **Compressed keys**: the mean of every 32-key window, stride 16 —
//!    window `j` covers keys `16j .. 16j+32`.
//! 2. **Stage 1**, the kernels' compressed-LSE softmax: per query head,
//!    `exp(q·ck_j/√d − lse)`, where `lse` is the log-sum-exp over the coarse
//!    128-key windows (stride 64) the query sees — not over the 32-key ones
//!    being scored — summed in f32 over the 16 query heads of the group and
//!    rounded to bf16 (the kernel's output type). In prefill the query at
//!    position `i` sees the first `A(i)` windows
//!    (`A(i) = ⌊(i−15)/16⌋ + k_len − ⌊(N−15)/16⌋`, clamped, for an `N`-token
//!    prompt; the coarse windows likewise with 64 for 16); a decode step sees
//!    every complete window.
//! 3. **Pooling** to 64-key blocks: block `k` takes the max over windows
//!    `4k−1 .. 4k+3`. The first `init_blocks` blocks and the `local_blocks`
//!    blocks up to the query's own are forced in (+∞).
//! 4. **Top-k** (`topk` = 64 + `local_blocks`) blocks, ties to the lower
//!    block index (the bf16 scores tie often), those after the query's block
//!    dropped.
//! 5. **Stage 2**: attention over the keys of the chosen blocks (causal in
//!    prefill).
//!
//! In the step graph this is an extra additive bias on the attention: the
//! host feeds the index maps and masks that depend only on positions
//! ([`SparseFeeds`]), the graph computes the scores, pools, ranks and
//! expands the choice to keys. Top-k is a rank count (a block is kept when
//! fewer than `topk` blocks outrank it: a higher score, or an equal one at a
//! lower index) — compare-and-sum, so every backend runs it.
//!
//! Checked against the kernels on a 12,696-token prompt
//! (`scripts/infllm_selection_check.py`): this rule reproduces the chosen
//! blocks of all 25,392 (head, query) rows; dropping the coarse-window
//! normalizer, the bf16 rounding or the tie order each changes 0.5–1 % of
//! them.
//!
//! Below `dense_len` keys the same graph runs dense: the host forces every
//! past block in.

use rlx_fish::nn::Builder;
use rlx_ir::infer::GraphExt;
use rlx_ir::number_format::{FloatFormat, FormatScaling};
use rlx_ir::op::{Op, PadMode, ReduceOp};
use rlx_ir::{DType, NodeId, Shape};
use serde_json::Value;

const F32: DType = DType::F32;
const BIG: f32 = 1e30;

pub const IN_SP_ABS2ROW: &str = "sparse_abs2row";
pub const IN_SP_ROW2ABS: &str = "sparse_row2abs";
pub const IN_SP_STAGE1: &str = "sparse_stage1_mask";
pub const IN_SP_COARSE: &str = "sparse_coarse_mask";
pub const IN_SP_ROWVALID: &str = "sparse_coarse_valid";
pub const IN_SP_BLOCKBIAS: &str = "sparse_block_bias";
pub const IN_SP_NOTFUTURE: &str = "sparse_not_future";

/// `sparse_config` of the checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseParams {
    pub kernel_size: usize,
    pub kernel_stride: usize,
    pub block_size: usize,
    pub init_blocks: usize,
    /// `window_size / block_size`.
    pub local_blocks: usize,
    /// Blocks kept per query: `topk + local_blocks`.
    pub topk: usize,
    pub dense_len: usize,
}

impl SparseParams {
    pub fn from_value(sc: &Value) -> Option<Self> {
        let u = |k: &str, d: usize| sc.get(k).and_then(Value::as_u64).map_or(d, |n| n as usize);
        let block_size = u("block_size", 64);
        let local_blocks = u("window_size", 2048) / block_size;
        let p = Self {
            kernel_size: u("kernel_size", 32),
            kernel_stride: u("kernel_stride", 16),
            block_size,
            init_blocks: u("init_blocks", 1),
            local_blocks,
            topk: u("topk", 64) + local_blocks,
            dense_len: sc.get("dense_len")?.as_u64()? as usize,
        };
        // The graph's window and pooling arithmetic is written for the
        // released geometry (32/16 windows, 64-key blocks).
        (p.kernel_size == 2 * p.kernel_stride && p.block_size == 4 * p.kernel_stride).then_some(p)
    }

    /// Compressed windows over `cap` keys (the graph's fixed width).
    pub fn windows(&self, cap: usize) -> usize {
        cap / self.kernel_stride - 1
    }

    /// Coarse windows (4× the size and stride) over `cap` keys.
    pub fn coarse_windows(&self, cap: usize) -> usize {
        cap / (4 * self.kernel_stride) - 1
    }

    /// Complete coarse windows over `len` keys.
    pub fn complete_coarse_windows(&self, len: usize) -> usize {
        let (size, stride) = (4 * self.kernel_size, 4 * self.kernel_stride);
        if len < size {
            0
        } else {
            (len - size) / stride + 1
        }
    }

    pub fn blocks(&self, cap: usize) -> usize {
        cap / self.block_size
    }

    /// Complete windows over `len` keys.
    pub fn complete_windows(&self, len: usize) -> usize {
        if len < self.kernel_size {
            0
        } else {
            (len - self.kernel_size) / self.kernel_stride + 1
        }
    }
}

/// How the queries of a step choose blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseMode {
    /// Every past block (below `dense_len`).
    Dense,
    /// Part of an `n_prompt`-token prompt fed from position 0.
    Prefill { n_prompt: usize },
    /// Each query on its own over the keys so far (decode steps).
    Decode,
}

/// Host feeds of one step (shared by the sparse layers).
#[derive(Debug, Clone)]
pub struct SparseFeeds {
    pub abs2row: Vec<f32>,
    pub row2abs: Vec<f32>,
    pub stage1: Vec<f32>,
    pub coarse: Vec<f32>,
    pub row_valid: Vec<f32>,
    pub block_bias: Vec<f32>,
    pub not_future: Vec<f32>,
}

impl SparseFeeds {
    /// Feeds for a chunk of `t` rows (`n` real) at `past` cached keys in a
    /// graph of capacity `cap`.
    pub fn new(
        sp: &SparseParams,
        t: usize,
        cap: usize,
        past: usize,
        n: usize,
        mode: SparseMode,
    ) -> Self {
        let c = sp.windows(cap);
        let c2 = sp.coarse_windows(cap);
        let nb = sp.blocks(cap);
        // Key rows of `[cache | chunk]` in position order, and back.
        let abs2row: Vec<f32> = (0..cap)
            .map(|p| {
                if p < past {
                    p as f32
                } else if p < past + t {
                    (cap + p - past) as f32
                } else {
                    0.0
                }
            })
            .collect();
        let row2abs: Vec<f32> = (0..cap + t)
            .map(|r| if r < cap { r } else { (past + r - cap).min(cap - 1) } as f32)
            .collect();

        let mut stage1 = vec![-BIG; t * c];
        let mut coarse = vec![-BIG; t * c2];
        let mut row_valid = vec![0f32; t];
        let mut block_bias = vec![-BIG; t * nb];
        let mut not_future = vec![0f32; t * nb];
        for u in 0..n {
            let i = past + u;
            let bq = i / sp.block_size;
            // Windows this query scores against, and the coarse ones its
            // normalizer runs over.
            let (seen, seen2) = match mode {
                SparseMode::Dense => (0, 0),
                SparseMode::Prefill { n_prompt } => {
                    let visible = |k_len: usize, stride: i64| {
                        let k_len = k_len as i64;
                        let shift = k_len - (n_prompt as i64 - (stride - 1)).div_euclid(stride);
                        ((i as i64 - (stride - 1)).div_euclid(stride) + shift).clamp(0, k_len)
                            as usize
                    };
                    let s = sp.kernel_stride as i64;
                    (
                        visible(sp.complete_windows(n_prompt), s),
                        visible(sp.complete_coarse_windows(n_prompt), 4 * s),
                    )
                }
                SparseMode::Decode => (
                    sp.complete_windows(i + 1),
                    sp.complete_coarse_windows(i + 1),
                ),
            };
            let (seen, seen2) = (seen.min(c), seen2.min(c2));
            stage1[u * c..u * c + seen].fill(0.0);
            coarse[u * c2..u * c2 + seen2].fill(0.0);
            row_valid[u] = if seen2 > 0 { 1.0 } else { 0.0 };
            for k in 0..nb.min(bq + 1) {
                let forced =
                    mode == SparseMode::Dense || k < sp.init_blocks || bq <= k + sp.local_blocks;
                block_bias[u * nb + k] = if forced { BIG } else { 0.0 };
                not_future[u * nb + k] = 1.0;
            }
        }
        Self {
            abs2row,
            row2abs,
            stage1,
            coarse,
            row_valid,
            block_bias,
            not_future,
        }
    }

    pub fn inputs(&self) -> [(&'static str, &[f32]); 7] {
        [
            (IN_SP_ABS2ROW, &self.abs2row),
            (IN_SP_ROW2ABS, &self.row2abs),
            (IN_SP_STAGE1, &self.stage1),
            (IN_SP_COARSE, &self.coarse),
            (IN_SP_ROWVALID, &self.row_valid),
            (IN_SP_BLOCKBIAS, &self.block_bias),
            (IN_SP_NOTFUTURE, &self.not_future),
        ]
    }
}

/// Graph inputs of the sparse path.
#[derive(Debug, Clone, Copy)]
pub struct SparseNodes {
    abs2row: NodeId,
    row2abs: NodeId,
    stage1: NodeId,
    coarse: NodeId,
    row_valid: NodeId,
    block_bias: NodeId,
    not_future: NodeId,
    /// `[nb, nb]`: 1 where the row's block index is below the column's
    /// (ties go to the lower index).
    lower: NodeId,
}

impl SparseNodes {
    pub fn declare(b: &mut Builder, sp: &SparseParams, t: usize, cap: usize) -> Self {
        let c = sp.windows(cap);
        let c2 = sp.coarse_windows(cap);
        let nb = sp.blocks(cap);
        let lower: Vec<f32> = (0..nb * nb)
            .map(|x| if x / nb < x % nb { 1.0 } else { 0.0 })
            .collect();
        Self {
            abs2row: b.input(IN_SP_ABS2ROW, &[cap]),
            row2abs: b.input(IN_SP_ROW2ABS, &[cap + t]),
            stage1: b.input(IN_SP_STAGE1, &[1, t, 1, c]),
            coarse: b.input(IN_SP_COARSE, &[1, t, 1, c2]),
            row_valid: b.input(IN_SP_ROWVALID, &[1, t, 1, 1]),
            block_bias: b.input(IN_SP_BLOCKBIAS, &[1, t, nb]),
            not_future: b.input(IN_SP_NOTFUTURE, &[1, t, nb]),
            lower: b.data("sparse.lower_index", lower, &[1, 1, nb, nb]),
        }
    }
}

/// The additive bias `[1, nh, t, cap + t]` that keeps each query to its
/// chosen blocks, and the choice itself `[nkv, t, nb]` (1 = kept).
///
/// `q` is `[1, t, nh·hd]`; `kk` the keys `[1, cap + t, nkv·hd]` as
/// `[cache | chunk]`, before the heads are repeated.
#[allow(clippy::too_many_arguments)]
pub fn sparse_bias(
    b: &mut Builder,
    sp: &SparseParams,
    nodes: &SparseNodes,
    q: NodeId,
    kk: NodeId,
    t: usize,
    cap: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
) -> (NodeId, NodeId) {
    let c = sp.windows(cap);
    let nb = sp.blocks(cap);
    let group = nh / nkv;
    let s16 = sp.kernel_stride;
    let w = nkv * hd;
    let g = &mut b.g;

    // 1. Keys in position order, 16-key means, then 32-key windows.
    let kflat = g.reshape_(kk, vec![(cap + t) as i64, w as i64]);
    let kabs = g.gather_(kflat, nodes.abs2row, 0); // [cap, w]
    let k16 = g.reshape_(kabs, vec![(cap / s16) as i64, s16 as i64, w as i64]);
    let m16 = g.mean(k16, vec![1], false); // [cap/16, w]
    let lo = g.narrow_(m16, 0, 0, c);
    let hi = g.narrow_(m16, 0, 1, c);
    let sum = g.add(lo, hi);
    let half = g.constant(0.5, F32);
    let ck = g.mul(sum, half); // [c, w]
    let ck = g.reshape_(ck, vec![c as i64, nkv as i64, hd as i64]);
    let ck = g.transpose_(ck, vec![1, 2, 0]); // [nkv, hd, c]
    // Coarse windows: 128 keys, stride 64 — pairs of 64-key means.
    let c2 = sp.coarse_windows(cap);
    let k64 = g.reshape_(m16, vec![(cap / (4 * s16)) as i64, 4, w as i64]);
    let m64 = g.mean(k64, vec![1], false); // [cap/64, w]
    let lo2 = g.narrow_(m64, 0, 0, c2);
    let hi2 = g.narrow_(m64, 0, 1, c2);
    let sum2 = g.add(lo2, hi2);
    let ck2 = g.mul(sum2, half);
    let ck2 = g.reshape_(ck2, vec![c2 as i64, nkv as i64, hd as i64]);
    let ck2 = g.transpose_(ck2, vec![1, 2, 0]); // [nkv, hd, c2]

    // 2. Stage 1 (compressed LSE): exp(q·ck_j/√d − lse_coarse) per head,
    //    summed per group in f32, rounded to bf16.
    let qg = g.reshape_(q, vec![t as i64, nkv as i64, group as i64, hd as i64]);
    let qg = g.transpose_(qg, vec![1, 0, 2, 3]); // [nkv, t, group, hd]
    let qg = g.reshape_(qg, vec![nkv as i64, (t * group) as i64, hd as i64]);
    let scale = g.constant((hd as f64).powf(-0.5), F32);
    let s2 = g.mm(qg, ck2); // [nkv, t·group, c2]
    let s2 = g.mul(s2, scale);
    let s2 = g.reshape_(s2, vec![nkv as i64, t as i64, group as i64, c2 as i64]);
    let s2 = g.add(s2, nodes.coarse);
    let m2 = g.reduce(
        s2,
        ReduceOp::Max,
        vec![3],
        true,
        Shape::new(&[nkv, t, group, 1], F32),
    );
    let e2 = g.sub(s2, m2);
    let e2 = g.exp(e2);
    let l2 = g.sum(e2, vec![3], true);
    let l2 = g.log(l2);
    let lse = g.add(l2, m2); // [nkv, t, group, 1]
    // A query with no coarse window yet (the first 127 keys) would have
    // lse = −∞; its blocks are all forced in, so any finite value does.
    let one = g.constant(1.0, F32);
    let none = g.sub(one, nodes.row_valid);
    let big = g.constant(BIG as f64, F32);
    let lift = g.mul(none, big);
    let lse = g.add(lse, lift);
    let s = g.mm(qg, ck); // [nkv, t·group, c]
    let s = g.mul(s, scale);
    let s = g.reshape_(s, vec![nkv as i64, t as i64, group as i64, c as i64]);
    let s = g.add(s, nodes.stage1);
    let s = g.sub(s, lse);
    let p = g.exp(s);
    let p = g.sum(p, vec![2], false); // [nkv, t, c]
    // Round to bf16 explicitly: a cast there and back is folded away as a
    // no-op, and without the rounding 0.6 % of the chosen blocks change.
    let p = g.add_node(
        Op::RoundToFormat {
            format: FloatFormat::BF16,
            scaling: FormatScaling::None,
        },
        vec![p],
        Shape::new(&[nkv, t, c], F32),
    );

    // 3. Pooling: block k = max over windows 4k-1 .. 4k+3 (scores are ≥ 0, so
    //    zero padding never wins over a present window).
    let per = sp.block_size / s16; // 4
    let padded = g.pad_(
        p,
        vec![[0, 0], [0, 0], [1, per * nb + per - 1 - c]],
        PadMode::Constant(0.0),
    ); // [nkv, t, per·nb + per]
    let body = g.narrow_(padded, 2, 0, per * nb);
    let body = g.reshape_(body, vec![nkv as i64, t as i64, nb as i64, per as i64]);
    let next = g.slice_(padded, 2, per, nb, per as i64);
    let next = g.reshape_(next, vec![nkv as i64, t as i64, nb as i64, 1]);
    let win = g.concat_(vec![body, next], 3);
    let blk = g.reduce(
        win,
        ReduceOp::Max,
        vec![3],
        false,
        Shape::new(&[nkv, t, nb], F32),
    );
    let blk = g.add(blk, nodes.block_bias);

    // 4. Top-k by rank: keep a block when fewer than `topk` outrank it — a
    //    higher score, or the same score at a lower index.
    let bi = g.reshape_(blk, vec![nkv as i64, t as i64, nb as i64, 1]);
    let bj = g.reshape_(blk, vec![nkv as i64, t as i64, 1, nb as i64]);
    let higher = g.lt(bi, bj);
    let higher = g.cast(higher, F32);
    let same = g.eq(bi, bj);
    let same = g.cast(same, F32);
    let lower_t = g.transpose_(nodes.lower, vec![0, 1, 3, 2]);
    let earlier = g.mul(same, lower_t);
    let outrank = g.add(higher, earlier);
    let rank = g.sum(outrank, vec![3], false); // [nkv, t, nb]
    let limit = g.constant(sp.topk as f64, F32);
    let kept = g.lt(rank, limit);
    let kept = g.cast(kept, F32);
    let kept = g.mul(kept, nodes.not_future); // [nkv, t, nb]

    // 5. Blocks → keys in position order → `[cache | chunk]` rows.
    let keys = g.reshape_(kept, vec![nkv as i64, t as i64, nb as i64, 1]);
    let keys = b.expand(keys, &[nkv, t, nb, sp.block_size]);
    let g = &mut b.g;
    let keys = g.reshape_(keys, vec![nkv as i64, t as i64, cap as i64]);
    let rows = g.gather_(keys, nodes.row2abs, 2); // [nkv, t, cap + t]
    let one = g.constant(1.0, F32);
    let closed = g.sub(rows, one);
    let big = g.constant(BIG as f64, F32);
    let bias = g.mul(closed, big);
    let bias = g.reshape_(bias, vec![1, nkv as i64, 1, t as i64, (cap + t) as i64]);
    let bias = b.expand(bias, &[1, nkv, group, t, cap + t]);
    let bias =
        b.g.reshape_(bias, vec![1, nh as i64, t as i64, (cap + t) as i64]);
    (bias, kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SparseParams {
        SparseParams::from_value(&serde_json::json!({
            "kernel_size": 32, "kernel_stride": 16, "init_blocks": 1, "block_size": 64,
            "window_size": 2048, "topk": 64, "dense_len": 8192
        }))
        .unwrap()
    }

    #[test]
    fn release_geometry() {
        let sp = params();
        assert_eq!((sp.local_blocks, sp.topk), (32, 96));
        assert_eq!(sp.complete_windows(31), 0);
        assert_eq!(sp.complete_windows(32), 1);
        assert_eq!(sp.complete_windows(12696), 792);
    }

    #[test]
    fn prefill_window_rule_matches_the_kernel() {
        // `naive_attention`'s mask for a 12696-token prompt: query i sees
        // windows j < ((i - 15) // 16) + k_len - (q_len - 16 + 1) // 16.
        let sp = params();
        let n = 12696;
        let f = SparseFeeds::new(&sp, 1, 12736, 9000, 1, SparseMode::Prefill { n_prompt: n });
        let c = sp.windows(12736);
        let seen = f.stage1[..c].iter().filter(|v| **v == 0.0).count();
        let k_len = 792i64;
        let want = ((9000i64 - 15).div_euclid(16) + k_len - (n as i64 - 15).div_euclid(16))
            .clamp(0, k_len);
        assert_eq!(seen as i64, want);
    }

    #[test]
    fn forced_blocks_are_init_and_local() {
        let sp = params();
        let f = SparseFeeds::new(&sp, 1, 12736, 12000, 1, SparseMode::Decode);
        let nb = sp.blocks(12736);
        let bq = 12000 / 64; // 187
        let forced: Vec<usize> = (0..nb).filter(|&k| f.block_bias[k] > 1.0).collect();
        assert_eq!(forced[0], 0);
        assert_eq!(&forced[1..], &(bq - 32..=bq).collect::<Vec<_>>()[..]);
        assert!((bq + 1..nb).all(|k| f.not_future[k] == 0.0));
    }
}
