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
//! cluster, unchanged.

use rlx_driver::{
    Rank, ReduceKind, SymmetricBuffer, SymmetricTransport, all_reduce, ring_all_reduce,
};

/// The collective surface a distributed trainer needs: average a gradient
/// across ranks, and broadcast a weight from a root rank.
pub trait GradComm {
    /// Number of participating ranks.
    fn world_size(&self) -> u32;
    /// In-place **mean** all-reduce of `v` across all ranks.
    fn all_reduce_mean(&self, v: &mut [f32]);
    /// Broadcast `v` from `root` to every rank (in place on non-roots).
    fn broadcast(&self, root: u32, v: &mut [f32]);
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

impl<T: SymmetricTransport> GradComm for RdmaGradComm<T> {
    fn world_size(&self) -> u32 {
        self.transport.num_ranks()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::{Adam, ParamSlot, lora_linear, train};
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
}
