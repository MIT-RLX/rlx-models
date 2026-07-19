// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Distributed (data-parallel) training: gradient reduction + weight sync
//! across nodes.
//!
//! In data-parallel SGD every rank holds the **same** weights and a
//! **different** data shard. Each step it computes a gradient on its shard,
//! the gradients are **all-reduced (averaged)** across ranks, and every rank
//! applies the same averaged update — so weights stay identical without ever
//! moving them. Averaging equal-size per-shard mean-gradients is exactly the
//! full-batch gradient (`mean(mean₀, mean₁) = mean(all)`), so N-way data
//! parallel is numerically equivalent to single-node training on the union of
//! the shards.
//!
//! [`GradComm`] abstracts the collective; [`RdmaGradComm`] implements it over
//! any [`SymmetricTransport`] (one-sided RDMA) — `LocalTransport` in-process,
//! `NetTransport` over TCP, or MLX's jaccl over Thunderbolt RDMA on a real
//! cluster, unchanged. [`ProcessGroupGradComm`] implements it over rlx-driver's
//! higher-level [`ProcessGroup`] (bandwidth-optimal ring all-reduce), and
//! [`from_env`] wires one up straight from `RANK`/`WORLD` env vars so the same
//! binary goes data-parallel with **no code change** — distributed out of the
//! box.

use anyhow::{Context, Result};
use rlx_driver::{
    Node, ProcessGroup, Rank, ReduceKind, SymmetricBuffer, SymmetricTransport, all_gather,
    all_reduce, ring_all_reduce,
};
use rlx_ir::DType;
use std::sync::Arc;

/// Wire precision for gradient all-reduce.
///
/// `Bf16` halves the bytes moved on production transports (the reduction still
/// accumulates in f64, so dynamic range isn't lost mid-sum) at bfloat16
/// gradient precision; `F32` is the exact default. The emulated and native
/// bf16 paths agree on values because both round inputs *and* the averaged
/// output to bf16.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ReduceDtype {
    #[default]
    F32,
    Bf16,
}

/// f32 → bfloat16 bits, round-to-nearest-even (the top 16 bits of the f32, with
/// rounding). NaNs pass through unchanged.
#[inline]
pub(crate) fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return (bits >> 16) as u16 | 0x0040; // keep it a NaN
    }
    let rounding_bias = 0x0000_7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// Round an f32 through bfloat16 precision (used by the emulated typed reduce
/// so it matches the native bf16 path bit-for-bit).
#[inline]
pub(crate) fn bf16_round(x: f32) -> f32 {
    f32::from_bits((f32_to_bf16_bits(x) as u32) << 16)
}

/// The collective surface a distributed trainer needs. Beyond the core
/// gradient-average + weight-broadcast, it exposes the primitives that
/// optimizer-state **sharding** (ZeRO-1) and **mixed-precision** reduction
/// build on. Everything above [`all_reduce_mean`](GradComm::all_reduce_mean),
/// [`broadcast`](GradComm::broadcast) and [`world_size`](GradComm::world_size)
/// has a portable default, so a minimal transport still works — production
/// impls override for bandwidth.
///
/// `Send + Sync` so a trainer can drive the collective from a background thread
/// (overlapping communication with the optimizer step).
pub trait GradComm: Send + Sync {
    /// Number of participating ranks.
    fn world_size(&self) -> u32;
    /// This process's rank in `0..world_size`. Selects which shard of a fused
    /// bucket this rank owns under optimizer-state sharding.
    fn rank(&self) -> u32 {
        0
    }
    /// In-place **mean** all-reduce of `v` across all ranks.
    fn all_reduce_mean(&self, v: &mut [f32]);
    /// Broadcast `v` from `root` to every rank (in place on non-roots).
    fn broadcast(&self, root: u32, v: &mut [f32]);

    /// Mean all-reduce at a chosen wire precision ([`ReduceDtype`]). The
    /// default emulates `Bf16` over the f32 collective (rounding inputs and the
    /// averaged result to bf16) so its values match a native bf16 reduce;
    /// bandwidth-saving impls override it.
    fn all_reduce_mean_typed(&self, v: &mut [f32], dtype: ReduceDtype) {
        match dtype {
            ReduceDtype::F32 => self.all_reduce_mean(v),
            ReduceDtype::Bf16 => {
                for x in v.iter_mut() {
                    *x = bf16_round(*x);
                }
                self.all_reduce_mean(v);
                for x in v.iter_mut() {
                    *x = bf16_round(*x);
                }
            }
        }
    }

    /// Reduce-scatter (mean): `full` (length `world_size * shard`, identical on
    /// every rank) is averaged across ranks and this rank keeps only its
    /// contiguous shard `[rank*shard, (rank+1)*shard)` in `shard_out`. The
    /// default is a full all-reduce then a slice — memory-optimal ZeRO-1 (each
    /// rank still stores optimizer state for its shard only) even without a
    /// native reduce-scatter.
    fn reduce_scatter_mean(&self, full: &[f32], shard_out: &mut [f32]) {
        let n = self.world_size().max(1) as usize;
        let chunk = shard_out.len();
        debug_assert_eq!(full.len(), n * chunk, "reduce_scatter: full != world*shard");
        if n <= 1 {
            shard_out.copy_from_slice(full);
            return;
        }
        let mut tmp = full.to_vec();
        self.all_reduce_mean(&mut tmp);
        let base = self.rank() as usize * chunk;
        shard_out.copy_from_slice(&tmp[base..base + chunk]);
    }

    /// All-gather equal-size shards: every rank contributes `shard` (identical
    /// length everywhere) and `out` (length `world_size * shard`) receives the
    /// rank-ordered concatenation. Reconstructs the full parameter bucket after
    /// each rank has updated only its own shard. The default is correct for a
    /// single rank; multi-rank impls override.
    fn all_gather_into(&self, shard: &[f32], out: &mut [f32]) {
        debug_assert!(self.world_size() <= 1, "all_gather_into needs a real impl");
        out[..shard.len()].copy_from_slice(shard);
    }
}

/// [`GradComm`] over a one-sided RDMA [`SymmetricTransport`]. The transport's
/// symmetric heap (offset 0) is used as the reduce/broadcast staging area, so
/// it must hold at least the largest tensor (`v.len() * 4` bytes).
pub struct RdmaGradComm<T: SymmetricTransport> {
    transport: T,
}

impl<T: SymmetricTransport> RdmaGradComm<T> {
    pub fn new(transport: T) -> Self {
        Self { transport }
    }
    pub fn transport(&self) -> &T {
        &self.transport
    }
}

impl<T: SymmetricTransport + Send + Sync> GradComm for RdmaGradComm<T> {
    fn world_size(&self) -> u32 {
        self.transport.num_ranks()
    }

    fn rank(&self) -> u32 {
        self.transport.this_rank().0
    }

    fn all_gather_into(&self, shard: &[f32], out: &mut [f32]) {
        let n = self.transport.num_ranks() as usize;
        if n <= 1 {
            out[..shard.len()].copy_from_slice(shard);
            return;
        }
        let buf = SymmetricBuffer {
            rank: self.transport.this_rank(),
            offset: 0,
            len: shard.len() * 4,
        };
        all_gather(&self.transport, buf, shard, out).expect("rdma all_gather");
        // Trailing barrier: stop a rank's next heap use racing a peer's read.
        self.transport
            .barrier()
            .expect("all_gather trailing barrier");
    }

    fn all_reduce_mean(&self, v: &mut [f32]) {
        let n = self.transport.num_ranks();
        if n <= 1 || v.is_empty() {
            return;
        }
        if v.len().is_multiple_of(n as usize) {
            // Bandwidth-optimal ring (the RDMA collective); its per-step
            // barriers also serialize heap reuse. Mailbox at heap offset 0.
            ring_all_reduce(&self.transport, 0, v, ReduceKind::Mean).expect("ring all_reduce_mean");
        } else {
            // Fallback for sizes not divisible by the rank count.
            let buf = SymmetricBuffer {
                rank: self.transport.this_rank(),
                offset: 0,
                len: v.len() * 4,
            };
            all_reduce(&self.transport, buf, v, ReduceKind::Mean).expect("rdma all_reduce_mean");
            // `all_reduce` syncs after the put but not after the get-loop;
            // a trailing barrier stops a rank's next reduction overwriting the
            // shared heap while a peer is still reading this one.
            self.transport
                .barrier()
                .expect("all_reduce_mean trailing barrier");
        }
    }

    fn broadcast(&self, root: u32, v: &mut [f32]) {
        if self.transport.num_ranks() <= 1 || v.is_empty() {
            return;
        }
        let len = v.len() * 4;
        let buf = SymmetricBuffer {
            rank: Rank(root),
            offset: 0,
            len,
        };
        let me = self.transport.this_rank();
        if me == Rank(root) {
            let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, len) };
            self.transport.put(buf, bytes).expect("broadcast put");
        }
        self.transport.barrier().expect("broadcast barrier");
        if me != Rank(root) {
            let mut bytes = vec![0u8; len];
            self.transport.get(buf, &mut bytes).expect("broadcast get");
            let f = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, v.len()) };
            v.copy_from_slice(f);
        }
        self.transport.barrier().expect("broadcast barrier2");
    }
}

/// [`GradComm`] over an rlx-driver [`ProcessGroup`] — the mesh [`from_env`]
/// hands back (TCP over the network, Thunderbolt on Apple Silicon, or an
/// in-process channel). Gradient averaging rides the group's
/// **bandwidth-optimal ring all-reduce**, so a fused bucket moves only
/// `~2·(n-1)/n · len` floats per rank regardless of world size, and — unlike
/// the [`RdmaGradComm`] symmetric-heap path — there is no per-tensor
/// divisibility special case or staging-heap to size.
pub struct ProcessGroupGradComm {
    group: Arc<ProcessGroup>,
}

impl ProcessGroupGradComm {
    pub fn new(group: Arc<ProcessGroup>) -> Self {
        Self { group }
    }
    /// The underlying process group (rank, world size, other collectives).
    pub fn group(&self) -> &Arc<ProcessGroup> {
        &self.group
    }
}

impl GradComm for ProcessGroupGradComm {
    fn world_size(&self) -> u32 {
        self.group.world_size()
    }

    fn rank(&self) -> u32 {
        self.group.rank()
    }

    fn all_reduce_mean(&self, v: &mut [f32]) {
        if self.group.world_size() <= 1 || v.is_empty() {
            return;
        }
        self.group
            .all_reduce(v, ReduceKind::Mean)
            .expect("process-group all_reduce_mean");
    }

    /// Native mixed-precision reduce: `Bf16` packs the bucket to bf16 and rides
    /// the group's typed ring all-reduce, so only **half** the bytes cross the
    /// wire (the reduction still accumulates in f64). Values match the emulated
    /// default because both round to bf16.
    fn all_reduce_mean_typed(&self, v: &mut [f32], dtype: ReduceDtype) {
        match dtype {
            ReduceDtype::F32 => self.all_reduce_mean(v),
            ReduceDtype::Bf16 => {
                if self.group.world_size() <= 1 || v.is_empty() {
                    return;
                }
                let mut bytes = Vec::with_capacity(v.len() * 2);
                for &x in v.iter() {
                    bytes.extend_from_slice(&f32_to_bf16_bits(x).to_le_bytes());
                }
                self.group
                    .all_reduce_typed(&mut bytes, DType::BF16, ReduceKind::Mean)
                    .expect("process-group bf16 all_reduce");
                for (x, c) in v.iter_mut().zip(bytes.chunks_exact(2)) {
                    let bf = u16::from_le_bytes([c[0], c[1]]);
                    *x = f32::from_bits((bf as u32) << 16);
                }
            }
        }
    }

    fn all_gather_into(&self, shard: &[f32], out: &mut [f32]) {
        let n = self.group.world_size() as usize;
        if n <= 1 {
            out[..shard.len()].copy_from_slice(shard);
            return;
        }
        let gathered = self
            .group
            .all_gather(shard)
            .expect("process-group all_gather");
        out.copy_from_slice(&gathered);
    }

    fn broadcast(&self, root: u32, v: &mut [f32]) {
        if self.group.world_size() <= 1 || v.is_empty() {
            return;
        }
        self.group
            .broadcast(root, v)
            .expect("process-group broadcast");
    }
}

/// Zero-config data-parallel bring-up from the environment.
///
/// Reads `RANK` / `WORLD` and the peer wiring (`PEERS=host:port,…`, or
/// `DISCOVER=1` for UDP auto-discovery; `TOPOLOGY=mesh|star`) via rlx-driver's
/// [`Node::from_env`], connects the process group, and hands back a boxed
/// [`GradComm`] ready to pass to [`crate::trainer::train`].
///
/// Returns `Ok(None)` when `WORLD <= 1` (the default) — so a plain
/// single-process run needs no environment at all and pays nothing, while the
/// **same binary** goes data-parallel the moment it is launched with
/// `RANK`/`WORLD` set (torchrun / mlx-lm style):
///
/// ```no_run
/// # fn main() -> anyhow::Result<()> {
/// let comm = rlx_tune::from_env()?;                 // Some(..) iff WORLD > 1
/// // rlx_tune::train(graph, &wrt, &mut params, &inputs, &mut opt, steps, comm.as_deref())?;
/// let _ = comm;
/// # Ok(()) }
/// ```
pub fn from_env() -> Result<Option<Box<dyn GradComm>>> {
    let node = Node::from_env().map_err(|e| anyhow::anyhow!("Node::from_env: {e}"))?;
    if node.world() <= 1 {
        return Ok(None);
    }
    let group = node
        .connect()
        .context("connecting the data-parallel process group")?;
    Ok(Some(Box::new(ProcessGroupGradComm::new(group))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::{
        Adam, AdamConfig, Checkpoint, DpConfig, ParamSlot, Trainer, lora_linear, train, train_dp,
        train_dp_with,
    };
    use rlx_driver::LocalTransport;
    use rlx_ir::infer::GraphExt;
    use rlx_ir::{DType, Graph, NodeId, Shape};
    use std::collections::HashMap;

    fn pseudo(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / u32::MAX as f32 - 0.5) * 0.2
            })
            .collect()
    }

    fn host_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for t in 0..k {
                    acc += a[i * k + t] * b[t * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    /// `y = (x·A)·B` (frozen zero base) with MSE-to-target loss; returns the
    /// graph + the `a`/`b` nodes. Same shape regardless of batch `m`.
    fn build_fit_graph(m: usize, k: usize, n: usize, r: usize) -> (Graph, NodeId, NodeId) {
        let f = DType::F32;
        let mut g = Graph::new("dp_fit");
        let x = g.input("x", Shape::new(&[m, k], f));
        let w = g.param("w", Shape::new(&[k, n], f));
        let a = g.param("a", Shape::new(&[k, r], f));
        let b = g.param("b", Shape::new(&[r, n], f));
        let y = lora_linear(&mut g, x, w, a, b, m, n, r);
        let t = g.input("t", Shape::new(&[m, n], f));
        let diff = g.sub(y, t);
        let sq = g.mul(diff, diff);
        let flat = g.reshape_(sq, vec![(m * n) as i64]);
        let loss = g.mean(flat, vec![0], false);
        g.set_outputs(vec![loss]);
        (g, a, b)
    }

    fn run_fit(
        m: usize,
        x: Vec<f32>,
        t: Vec<f32>,
        a0: Vec<f32>,
        b0: Vec<f32>,
        steps: usize,
        comm: Option<&dyn GradComm>,
        k: usize,
        n: usize,
        r: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let (g, a, b) = build_fit_graph(m, k, n, r);
        let mut params = HashMap::new();
        params.insert("w".to_string(), vec![0.0; k * n]);
        params.insert("a".to_string(), a0);
        params.insert("b".to_string(), b0);
        let wrt = vec![
            ParamSlot {
                name: "a".into(),
                node: a,
            },
            ParamSlot {
                name: "b".into(),
                node: b,
            },
        ];
        let inputs = vec![("x".to_string(), x), ("t".to_string(), t)];
        let mut opt = Adam::new(0.05);
        train(g, &wrt, &mut params, &inputs, &mut opt, steps, comm).unwrap();
        (params["a"].clone(), params["b"].clone())
    }

    // ---- train_dp harness: sharding / overlap / reduce / metrics ----------

    /// Synthetic LoRA-fit problem: `(x, target=x·M, a0, b0)`.
    fn synth(
        rows: usize,
        k: usize,
        n: usize,
        r: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let xfull = pseudo(rows * k, 1);
        let m_true = pseudo(k * n, 2);
        let tfull = host_matmul(&xfull, &m_true, rows, k, n);
        (xfull, tfull, pseudo(k * r, 3), pseudo(r * n, 4))
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
        assert_eq!(got.len(), want.len(), "{label} length");
        for (x, y) in got.iter().zip(want) {
            assert!((x - y).abs() < tol, "{label} mismatch: {x} vs {y}");
        }
    }

    /// Like `run_fit`, but drives `train_dp` with a `DpConfig`. Returns the
    /// fitted `a`, `b`, and the per-step loss.
    #[allow(clippy::too_many_arguments)]
    fn run_fit_dp(
        m: usize,
        x: Vec<f32>,
        t: Vec<f32>,
        a0: Vec<f32>,
        b0: Vec<f32>,
        steps: usize,
        comm: Option<&dyn GradComm>,
        cfg: &DpConfig,
        k: usize,
        n: usize,
        r: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (g, a, b) = build_fit_graph(m, k, n, r);
        let mut params = HashMap::new();
        params.insert("w".to_string(), vec![0.0; k * n]);
        params.insert("a".to_string(), a0);
        params.insert("b".to_string(), b0);
        let wrt = vec![
            ParamSlot {
                name: "a".into(),
                node: a,
            },
            ParamSlot {
                name: "b".into(),
                node: b,
            },
        ];
        let inputs = vec![("x".to_string(), x), ("t".to_string(), t)];
        let losses = train_dp(g, &wrt, &mut params, &inputs, steps, comm, cfg, |_| {}).unwrap();
        (params["a"].clone(), params["b"].clone(), losses)
    }

    /// 2-rank in-process data-parallel fit with `cfg`, rows sharded across
    /// ranks; returns each rank's fitted `(a, b)`.
    #[allow(clippy::too_many_arguments)]
    fn run_2rank_dp(
        cfg: DpConfig,
        xf: &[f32],
        tf: &[f32],
        a0: &[f32],
        b0: &[f32],
        steps: usize,
        k: usize,
        n: usize,
        r: usize,
    ) -> Vec<(Vec<f32>, Vec<f32>)> {
        let handles: Vec<_> = LocalTransport::fan_out(2, 1 << 16)
            .into_iter()
            .enumerate()
            .map(|(rank, tpt)| {
                let xshard = xf[rank * 2 * k..(rank * 2 + 2) * k].to_vec();
                let tshard = tf[rank * 2 * n..(rank * 2 + 2) * n].to_vec();
                let (a0, b0) = (a0.to_vec(), b0.to_vec());
                std::thread::spawn(move || {
                    let comm = RdmaGradComm::new(tpt);
                    let (a, b, _) =
                        run_fit_dp(2, xshard, tshard, a0, b0, steps, Some(&comm), &cfg, k, n, r);
                    (a, b)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    }

    #[test]
    fn train_dp_matches_single_process() {
        let (k, n, r, rows, steps) = (3, 2, 3, 4, 80);
        let (xf, tf, a0, b0) = synth(rows, k, n, r);
        let cfg = DpConfig {
            adam: AdamConfig::new(0.05),
            ..Default::default()
        };
        let (ref_a, ref_b, _) = run_fit_dp(
            rows,
            xf.clone(),
            tf.clone(),
            a0.clone(),
            b0.clone(),
            steps,
            None,
            &cfg,
            k,
            n,
            r,
        );
        for (a, b) in run_2rank_dp(cfg, &xf, &tf, &a0, &b0, steps, k, n, r) {
            assert_close(&a, &ref_a, 1e-4, "a");
            assert_close(&b, &ref_b, 1e-4, "b");
        }
    }

    #[test]
    fn train_dp_sharded_matches_single_process() {
        // ZeRO-1 optimizer-state sharding is numerically identical to DDP.
        let (k, n, r, rows, steps) = (3, 2, 3, 4, 80);
        let (xf, tf, a0, b0) = synth(rows, k, n, r);
        let base = DpConfig {
            adam: AdamConfig::new(0.05),
            ..Default::default()
        };
        let (ref_a, ref_b, _) = run_fit_dp(
            rows,
            xf.clone(),
            tf.clone(),
            a0.clone(),
            b0.clone(),
            steps,
            None,
            &base,
            k,
            n,
            r,
        );
        let cfg = DpConfig {
            shard_optimizer: true,
            ..base
        };
        for (a, b) in run_2rank_dp(cfg, &xf, &tf, &a0, &b0, steps, k, n, r) {
            assert_close(&a, &ref_a, 1e-4, "sharded a");
            assert_close(&b, &ref_b, 1e-4, "sharded b");
        }
    }

    #[test]
    fn train_dp_overlap_matches_single_process() {
        // Overlapping the reduce with the optimizer step is numerically exact.
        let (k, n, r, rows, steps) = (3, 2, 3, 4, 80);
        let (xf, tf, a0, b0) = synth(rows, k, n, r);
        let base = DpConfig {
            adam: AdamConfig::new(0.05),
            ..Default::default()
        };
        let (ref_a, ref_b, _) = run_fit_dp(
            rows,
            xf.clone(),
            tf.clone(),
            a0.clone(),
            b0.clone(),
            steps,
            None,
            &base,
            k,
            n,
            r,
        );
        let cfg = DpConfig {
            overlap: true,
            chunks: 4,
            ..base
        };
        for (a, b) in run_2rank_dp(cfg, &xf, &tf, &a0, &b0, steps, k, n, r) {
            assert_close(&a, &ref_a, 1e-4, "overlap a");
            assert_close(&b, &ref_b, 1e-4, "overlap b");
        }
    }

    #[test]
    fn train_dp_sharded_overlap_matches_and_lockstep() {
        // Overlapped ZeRO-1 (block-cyclic, pipelined reduce-scatter) matches
        // single-process at short horizon, and ranks stay bit-identical.
        let (k, n, r, rows) = (3, 2, 3, 4);
        let (xf, tf, a0, b0) = synth(rows, k, n, r);
        let base = DpConfig {
            adam: AdamConfig::new(0.05),
            ..Default::default()
        };
        let cfg = DpConfig {
            shard_optimizer: true,
            overlap: true,
            chunks: 3,
            ..base
        };
        let (sa, sb, _) = run_fit_dp(
            rows,
            xf.clone(),
            tf.clone(),
            a0.clone(),
            b0.clone(),
            5,
            None,
            &base,
            k,
            n,
            r,
        );
        for (a, b) in run_2rank_dp(cfg, &xf, &tf, &a0, &b0, 5, k, n, r) {
            assert_close(&a, &sa, 1e-4, "sh-overlap a");
            assert_close(&b, &sb, 1e-4, "sh-overlap b");
        }
        let ranks = run_2rank_dp(cfg, &xf, &tf, &a0, &b0, 60, k, n, r);
        assert_eq!(ranks[0], ranks[1], "sharded-overlap ranks drifted");
    }

    #[test]
    fn grad_accum_constant_data_matches_no_accum() {
        // Accumulating the same batch G times = one batch (mean of identical
        // grads), so grad_accum=G equals grad_accum=1 for constant data.
        let (k, n, r, rows, steps) = (3, 2, 3, 4, 40);
        let (xf, tf, a0, b0) = synth(rows, k, n, r);
        let cfg1 = DpConfig {
            adam: AdamConfig::new(0.05),
            ..Default::default()
        };
        let (ref_a, ref_b, _) = run_fit_dp(
            rows,
            xf.clone(),
            tf.clone(),
            a0.clone(),
            b0.clone(),
            steps,
            None,
            &cfg1,
            k,
            n,
            r,
        );

        let cfg_g = DpConfig {
            grad_accum: 3,
            ..cfg1
        };
        let (g, a, b) = build_fit_graph(rows, k, n, r);
        let mut params = HashMap::new();
        params.insert("w".to_string(), vec![0.0; k * n]);
        params.insert("a".to_string(), a0.clone());
        params.insert("b".to_string(), b0.clone());
        let wrt = vec![
            ParamSlot {
                name: "a".into(),
                node: a,
            },
            ParamSlot {
                name: "b".into(),
                node: b,
            },
        ];
        let batch = vec![("x".to_string(), xf.clone()), ("t".to_string(), tf.clone())];
        let losses = train_dp_with(
            g,
            &wrt,
            &mut params,
            steps,
            None,
            &cfg_g,
            |_, _| batch.clone(),
            |_| {},
        )
        .unwrap();
        assert_close(&params["a"], &ref_a, 1e-4, "accum a");
        assert_close(&params["b"], &ref_b, 1e-4, "accum b");
        assert!(*losses.last().unwrap() < losses[0], "accum did not train");
    }

    #[test]
    fn grad_accum_two_micro_equals_full_batch() {
        // 2 micro-batches of 2 rows (m=2 graph) accumulate to the same gradient
        // as one 4-row batch (m=4 graph) → matching training.
        let (k, n, r) = (3, 2, 3);
        let (xf, tf, a0, b0) = synth(4, k, n, r);
        let cfg1 = DpConfig {
            adam: AdamConfig::new(0.05),
            ..Default::default()
        };
        let (ref_a, ref_b, _) = run_fit_dp(
            4,
            xf.clone(),
            tf.clone(),
            a0.clone(),
            b0.clone(),
            10,
            None,
            &cfg1,
            k,
            n,
            r,
        );

        let (g, a, b) = build_fit_graph(2, k, n, r);
        let mut params = HashMap::new();
        params.insert("w".to_string(), vec![0.0; k * n]);
        params.insert("a".to_string(), a0.clone());
        params.insert("b".to_string(), b0.clone());
        let wrt = vec![
            ParamSlot {
                name: "a".into(),
                node: a,
            },
            ParamSlot {
                name: "b".into(),
                node: b,
            },
        ];
        let micro = [
            (xf[0..2 * k].to_vec(), tf[0..2 * n].to_vec()),
            (xf[2 * k..4 * k].to_vec(), tf[2 * n..4 * n].to_vec()),
        ];
        let cfg_a = DpConfig {
            grad_accum: 2,
            ..cfg1
        };
        train_dp_with(
            g,
            &wrt,
            &mut params,
            10,
            None,
            &cfg_a,
            |_step, m| {
                let (x, t) = &micro[m];
                vec![("x".to_string(), x.clone()), ("t".to_string(), t.clone())]
            },
            |_| {},
        )
        .unwrap();
        assert_close(&params["a"], &ref_a, 1e-4, "accum-2micro a");
        assert_close(&params["b"], &ref_b, 1e-4, "accum-2micro b");
    }

    #[test]
    fn prefetch_matches_run() {
        // Background prefetch is numerically identical to run (same batches,
        // same order) — exercised with gradient accumulation.
        let (k, n, r, rows, steps) = (3, 2, 3, 4, 40);
        let (xf, tf, a0, b0) = synth(rows, k, n, r);
        let cfg = DpConfig::new(0.05).grad_accum(2);
        let batch = vec![("x".to_string(), xf), ("t".to_string(), tf)];
        let build = || {
            let (g, a, b) = build_fit_graph(rows, k, n, r);
            let params = fit_params(&a0, &b0, k, n);
            let wrt = vec![
                ParamSlot {
                    name: "a".into(),
                    node: a,
                },
                ParamSlot {
                    name: "b".into(),
                    node: b,
                },
            ];
            Trainer::new(g, &wrt, &params, steps, None, &cfg).unwrap()
        };
        let mut t_run = build();
        t_run.run(|_, _| batch.clone(), |_| {}).unwrap();
        let mut t_pf = build();
        t_pf.run_prefetched(|_, _| batch.clone(), |_| {}).unwrap();
        let (a, b) = (t_run.params(), t_pf.params());
        assert_eq!(a["a"], b["a"], "prefetch a");
        assert_eq!(a["b"], b["b"], "prefetch b");
    }

    // ---- checkpoint save/resume ------------------------------------------

    fn fit_params(a0: &[f32], b0: &[f32], k: usize, n: usize) -> HashMap<String, Vec<f32>> {
        let mut p = HashMap::new();
        p.insert("w".to_string(), vec![0.0; k * n]);
        p.insert("a".to_string(), a0.to_vec());
        p.insert("b".to_string(), b0.to_vec());
        p
    }

    #[allow(clippy::too_many_arguments)]
    fn ckpt_single(
        cfg: DpConfig,
        xf: &[f32],
        tf: &[f32],
        a0: &[f32],
        b0: &[f32],
        steps: usize,
        k: usize,
        n: usize,
        r: usize,
        rows: usize,
    ) -> Checkpoint {
        let (g, a, b) = build_fit_graph(rows, k, n, r);
        let params = fit_params(a0, b0, k, n);
        let wrt = vec![
            ParamSlot {
                name: "a".into(),
                node: a,
            },
            ParamSlot {
                name: "b".into(),
                node: b,
            },
        ];
        let batch = vec![
            ("x".to_string(), xf.to_vec()),
            ("t".to_string(), tf.to_vec()),
        ];
        let mut tr = Trainer::new(g, &wrt, &params, steps, None, &cfg).unwrap();
        tr.run(|_, _| batch.clone(), |_| {}).unwrap();
        tr.checkpoint()
    }

    #[allow(clippy::too_many_arguments)]
    fn ckpt_sharded_2rank(
        cfg: DpConfig,
        xf: &[f32],
        tf: &[f32],
        a0: &[f32],
        b0: &[f32],
        steps: usize,
        k: usize,
        n: usize,
        r: usize,
        rows: usize,
    ) -> Checkpoint {
        let per = rows / 2;
        let handles: Vec<_> = LocalTransport::fan_out(2, 1 << 16)
            .into_iter()
            .enumerate()
            .map(|(rank, tpt)| {
                let x = xf[rank * per * k..(rank + 1) * per * k].to_vec();
                let t = tf[rank * per * n..(rank + 1) * per * n].to_vec();
                let (a0, b0) = (a0.to_vec(), b0.to_vec());
                std::thread::spawn(move || {
                    let comm = RdmaGradComm::new(tpt);
                    let (g, a, b) = build_fit_graph(per, k, n, r);
                    let params = fit_params(&a0, &b0, k, n);
                    let wrt = vec![
                        ParamSlot {
                            name: "a".into(),
                            node: a,
                        },
                        ParamSlot {
                            name: "b".into(),
                            node: b,
                        },
                    ];
                    let batch = vec![("x".to_string(), x), ("t".to_string(), t)];
                    let mut tr = Trainer::new(g, &wrt, &params, steps, Some(&comm), &cfg).unwrap();
                    tr.run(|_, _| batch.clone(), |_| {}).unwrap();
                    tr.checkpoint() // collective — every rank participates
                })
            })
            .collect();
        // All ranks produce the identical (gathered) checkpoint; take rank 0's.
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .next()
            .unwrap()
    }

    #[test]
    fn checkpoint_resume_is_exact() {
        // Save at step 10 → file → fresh Trainer → restore → run to 30 equals
        // 30 steps straight (constant LR ⇒ exact resume). Also covers the
        // binary save/load round-trip.
        let (k, n, r, rows) = (3, 2, 3, 4);
        let (xf, tf, a0, b0) = synth(rows, k, n, r);
        let cfg = DpConfig {
            adam: AdamConfig::new(0.05),
            ..Default::default()
        };
        let batch = vec![("x".to_string(), xf.clone()), ("t".to_string(), tf.clone())];

        let (rg, ra, rb) = build_fit_graph(rows, k, n, r);
        let wrt = vec![
            ParamSlot {
                name: "a".into(),
                node: ra,
            },
            ParamSlot {
                name: "b".into(),
                node: rb,
            },
        ];
        let mut rt = Trainer::new(rg, &wrt, &fit_params(&a0, &b0, k, n), 30, None, &cfg).unwrap();
        rt.run(|_, _| batch.clone(), |_| {}).unwrap();
        let want = rt.params();

        let (g1, a1, b1) = build_fit_graph(rows, k, n, r);
        let wrt1 = vec![
            ParamSlot {
                name: "a".into(),
                node: a1,
            },
            ParamSlot {
                name: "b".into(),
                node: b1,
            },
        ];
        let mut t1 = Trainer::new(g1, &wrt1, &fit_params(&a0, &b0, k, n), 10, None, &cfg).unwrap();
        t1.run(|_, _| batch.clone(), |_| {}).unwrap();
        let path = std::env::temp_dir().join("rlx_tune_ckpt_roundtrip.bin");
        t1.checkpoint().save(&path).unwrap();
        let ck = Checkpoint::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(ck.step, 10);

        let (g2, a2, b2) = build_fit_graph(rows, k, n, r);
        let wrt2 = vec![
            ParamSlot {
                name: "a".into(),
                node: a2,
            },
            ParamSlot {
                name: "b".into(),
                node: b2,
            },
        ];
        let mut t2 = Trainer::new(g2, &wrt2, &fit_params(&a0, &b0, k, n), 30, None, &cfg).unwrap();
        t2.restore(&ck);
        t2.run(|_, _| batch.clone(), |_| {}).unwrap();
        let got = t2.params();
        assert_close(&got["a"], &want["a"], 1e-6, "resume a");
        assert_close(&got["b"], &want["b"], 1e-6, "resume b");
    }

    #[test]
    fn checkpoint_sharded_gather_matches_single() {
        // A sharded run's checkpoint gathers the full optimizer state; it should
        // match single-process at short horizon — for both contiguous and
        // block-cyclic (overlapped) sharding.
        let (k, n, r, rows, steps) = (3, 2, 3, 4, 6);
        let (xf, tf, a0, b0) = synth(rows, k, n, r);
        let single = DpConfig {
            adam: AdamConfig::new(0.05),
            ..Default::default()
        };
        let ref_ck = ckpt_single(single, &xf, &tf, &a0, &b0, steps, k, n, r, rows);

        let sh = DpConfig {
            shard_optimizer: true,
            ..single
        };
        let sh_ck = ckpt_sharded_2rank(sh, &xf, &tf, &a0, &b0, steps, k, n, r, rows);
        assert_close(&sh_ck.m, &ref_ck.m, 1e-4, "sharded ckpt m");
        assert_close(&sh_ck.v, &ref_ck.v, 1e-4, "sharded ckpt v");

        let sho = DpConfig {
            shard_optimizer: true,
            overlap: true,
            chunks: 3,
            ..single
        };
        let sho_ck = ckpt_sharded_2rank(sho, &xf, &tf, &a0, &b0, steps, k, n, r, rows);
        assert_close(&sho_ck.m, &ref_ck.m, 1e-4, "block-cyclic ckpt m");

        let find = |ck: &Checkpoint, name: &str| {
            ck.params
                .iter()
                .find(|(nm, _)| nm == name)
                .unwrap()
                .1
                .clone()
        };
        assert_close(
            &find(&sh_ck, "a"),
            &find(&ref_ck, "a"),
            1e-4,
            "sharded ckpt a",
        );
    }

    #[test]
    fn train_dp_bf16_reduce_trains() {
        // bf16 gradient reduction still drives the loss down (emulated typed path).
        let (k, n, r, rows, steps) = (3, 2, 3, 4, 120);
        let (xf, tf, a0, b0) = synth(rows, k, n, r);
        let cfg = DpConfig {
            adam: AdamConfig::new(0.05),
            reduce_dtype: ReduceDtype::Bf16,
            ..Default::default()
        };
        let handles: Vec<_> = LocalTransport::fan_out(2, 1 << 16)
            .into_iter()
            .enumerate()
            .map(|(rank, tpt)| {
                let xshard = xf[rank * 2 * k..(rank * 2 + 2) * k].to_vec();
                let tshard = tf[rank * 2 * n..(rank * 2 + 2) * n].to_vec();
                let (a0, b0) = (a0.clone(), b0.clone());
                std::thread::spawn(move || {
                    let comm = RdmaGradComm::new(tpt);
                    let (_, _, losses) =
                        run_fit_dp(2, xshard, tshard, a0, b0, steps, Some(&comm), &cfg, k, n, r);
                    losses
                })
            })
            .collect();
        for h in handles {
            let losses = h.join().unwrap();
            let (first, last) = (losses[0], *losses.last().unwrap());
            assert!(first > 1e-4, "initial loss should be nonzero: {first}");
            assert!(
                last < first * 0.1,
                "bf16 reduce did not train: {first} -> {last}"
            );
        }
    }

    #[test]
    fn train_dp_reports_metrics() {
        let (k, n, r, rows, steps) = (3, 2, 3, 4, 25);
        let (xf, tf, a0, b0) = synth(rows, k, n, r);
        let (g, a, b) = build_fit_graph(rows, k, n, r);
        let mut params = HashMap::new();
        params.insert("w".to_string(), vec![0.0; k * n]);
        params.insert("a".to_string(), a0);
        params.insert("b".to_string(), b0);
        let wrt = vec![
            ParamSlot {
                name: "a".into(),
                node: a,
            },
            ParamSlot {
                name: "b".into(),
                node: b,
            },
        ];
        let inputs = vec![("x".to_string(), xf), ("t".to_string(), tf)];
        let cfg = DpConfig {
            adam: AdamConfig::new(0.05),
            log_every: 10,
            ..Default::default()
        };
        let mut steps_seen = Vec::new();
        train_dp(g, &wrt, &mut params, &inputs, steps, None, &cfg, |m| {
            assert!(m.compute_ms >= 0.0 && m.comm_ms >= 0.0 && m.step_ms >= 0.0);
            assert_eq!(m.world_size, 1);
            assert_eq!(m.reduced_elems, k * r + r * n);
            assert!((m.lr - 0.05).abs() < 1e-6, "constant lr reported: {}", m.lr);
            steps_seen.push(m.step);
        })
        .unwrap();
        // log_every 10 over 25 steps → steps 9, 19, and always the last (24).
        assert_eq!(steps_seen, vec![9, 19, 24]);
    }

    #[test]
    fn train_dp_one_step_matches_single() {
        // One step of 2-rank DP equals single-process on the union of the rows
        // (the gradient-averaging identity, before convergence can mask it).
        let (k, n, r, rows) = (3, 2, 3, 4);
        let (xf, tf, a0, b0) = synth(rows, k, n, r);
        let cfg = DpConfig {
            adam: AdamConfig::new(0.1),
            ..Default::default()
        };
        let (sa, sb, _) = run_fit_dp(
            rows,
            xf.clone(),
            tf.clone(),
            a0.clone(),
            b0.clone(),
            1,
            None,
            &cfg,
            k,
            n,
            r,
        );
        for (a, b) in run_2rank_dp(cfg, &xf, &tf, &a0, &b0, 1, k, n, r) {
            assert_close(&a, &sa, 1e-6, "1step a");
            assert_close(&b, &sb, 1e-6, "1step b");
        }
    }

    #[test]
    fn train_dp_clipping_is_lockstep_and_active() {
        // Global-norm clipping is distributed-correct: every rank derives the
        // SAME scale from the globally-reduced gradient, so ranks never drift —
        // even at a long horizon where the stiff-clip + scale-invariant-Adam
        // dynamics are chaotic. (Ranks bit-identical; cross-reduction-path
        // parity is only asserted at a short horizon, before chaos amplifies
        // fp reduction-order noise.)
        let (k, n, r, rows) = (3, 2, 3, 4);
        let (xf, tf, a0, b0) = synth(rows, k, n, r);
        let noclip = DpConfig {
            adam: AdamConfig::new(0.1),
            ..Default::default()
        };
        let clip = DpConfig {
            max_grad_norm: Some(1e-4),
            ..noclip
        };

        // Short horizon: DP-with-clip matches single-process-with-clip.
        let (sa, sb, _) = run_fit_dp(
            rows,
            xf.clone(),
            tf.clone(),
            a0.clone(),
            b0.clone(),
            5,
            None,
            &clip,
            k,
            n,
            r,
        );
        for (a, b) in run_2rank_dp(clip, &xf, &tf, &a0, &b0, 5, k, n, r) {
            assert_close(&a, &sa, 1e-5, "clip dp a");
            assert_close(&b, &sb, 1e-5, "clip dp b");
        }

        // Any horizon: ranks stay bit-identical (no drift) — unsharded…
        let un = run_2rank_dp(clip, &xf, &tf, &a0, &b0, 60, k, n, r);
        assert_eq!(un[0], un[1], "unsharded clip drifted across ranks");
        // …and sharded (each rank all-reduces its shard's sum-of-squares).
        let sharded = DpConfig {
            shard_optimizer: true,
            ..clip
        };
        let sh = run_2rank_dp(sharded, &xf, &tf, &a0, &b0, 60, k, n, r);
        assert_eq!(sh[0], sh[1], "sharded clip drifted across ranks");

        // Non-vacuous: clipping changed the outcome vs no clip.
        let (nc_a, _, _) = run_fit_dp(
            rows,
            xf.clone(),
            tf.clone(),
            a0.clone(),
            b0.clone(),
            5,
            None,
            &noclip,
            k,
            n,
            r,
        );
        assert!(
            sa.iter().zip(&nc_a).any(|(x, y)| (x - y).abs() > 1e-5),
            "clipping did not engage"
        );
    }

    #[test]
    fn data_parallel_matches_single_process() {
        // 2-way data parallel over in-process RDMA must equal single-process
        // training on the union of the shards (gradient-averaging identity).
        let (k, n, r) = (3usize, 2usize, 3usize);
        let steps = 100usize;
        let rows = 4usize;

        let xfull = pseudo(rows * k, 1);
        let m_true = pseudo(k * n, 2);
        let tfull = host_matmul(&xfull, &m_true, rows, k, n);
        let a0 = pseudo(k * r, 3);
        let b0 = pseudo(r * n, 4);

        // Reference: single process, full batch.
        let (ref_a, ref_b) = run_fit(
            rows,
            xfull.clone(),
            tfull.clone(),
            a0.clone(),
            b0.clone(),
            steps,
            None,
            k,
            n,
            r,
        );

        // Data-parallel: 2 ranks, 2 rows each, sharing one RDMA heap + barrier.
        let ts = LocalTransport::fan_out(2, 256);
        let handles: Vec<_> = ts
            .into_iter()
            .enumerate()
            .map(|(rank, t)| {
                let xshard = xfull[rank * 2 * k..(rank * 2 + 2) * k].to_vec();
                let tshard = tfull[rank * 2 * n..(rank * 2 + 2) * n].to_vec();
                let (a0, b0) = (a0.clone(), b0.clone());
                std::thread::spawn(move || {
                    let comm = RdmaGradComm::new(t);
                    run_fit(2, xshard, tshard, a0, b0, steps, Some(&comm), k, n, r)
                })
            })
            .collect();

        for h in handles {
            let (a, b) = h.join().unwrap();
            for (x, y) in a.iter().zip(&ref_a) {
                assert!((x - y).abs() < 1e-3, "a mismatch: {x} vs {y}");
            }
            for (x, y) in b.iter().zip(&ref_b) {
                assert!((x - y).abs() < 1e-3, "b mismatch: {x} vs {y}");
            }
        }
    }

    #[test]
    fn single_rank_comm_is_noop() {
        // world_size 1 → all_reduce/broadcast are no-ops (training unchanged).
        let comm = RdmaGradComm::new(LocalTransport::new(1, 64, rlx_driver::Rank(0)));
        assert_eq!(comm.world_size(), 1);
        let mut v = vec![1.0, 2.0, 3.0];
        comm.all_reduce_mean(&mut v);
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn fused_bucket_reduce_equals_per_param() {
        // The trainer's bucketing relies on: averaging a concatenation of
        // gradients == concatenating the per-gradient averages. Verify it over
        // the real transport (two ranks, in-process RDMA).
        let ga = [[1.0f32, 2.0, 3.0], [3.0, 2.0, 1.0]]; // rank 0, rank 1
        let gb = [[10.0f32, 20.0], [30.0, 40.0]];
        let handles: Vec<_> = LocalTransport::fan_out(2, 4096)
            .into_iter()
            .enumerate()
            .map(|(rank, t)| {
                let (a, b) = (ga[rank].to_vec(), gb[rank].to_vec());
                std::thread::spawn(move || {
                    let comm = RdmaGradComm::new(t);
                    // Per-parameter: one reduce each (call order identical on
                    // every rank, so the shared-heap barriers stay aligned).
                    let (mut a1, mut b1) = (a.clone(), b.clone());
                    comm.all_reduce_mean(&mut a1);
                    comm.all_reduce_mean(&mut b1);
                    // Fused: one reduce over the concatenation.
                    let mut fused: Vec<f32> = a.iter().chain(&b).copied().collect();
                    comm.all_reduce_mean(&mut fused);
                    (a1, b1, fused)
                })
            })
            .collect();
        for h in handles {
            let (a1, b1, fused) = h.join().unwrap();
            assert_eq!(a1, vec![2.0, 2.0, 2.0]); // mean([1,2,3],[3,2,1])
            assert_eq!(b1, vec![20.0, 30.0]); // mean([10,20],[30,40])
            let expect: Vec<f32> = a1.iter().chain(&b1).copied().collect();
            for (x, y) in fused.iter().zip(&expect) {
                assert!((x - y).abs() < 1e-6, "fused {x} vs per-param {y}");
            }
        }
    }

    #[test]
    fn from_env_single_rank_is_none() {
        // With no RANK/WORLD in the environment, WORLD defaults to 1 → no
        // process group is connected and `from_env` returns None, so plain
        // single-process training needs no env and opens no sockets.
        assert!(std::env::var("WORLD").is_err());
        assert!(from_env().unwrap().is_none());
    }
}
