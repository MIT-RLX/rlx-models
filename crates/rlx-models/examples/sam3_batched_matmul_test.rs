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

// Test the IR's 4D batched matmul gives the same result as per-batch
// 2D matmul on the host. Layer-0 of the SAM3 decoder hits a [B,H,L,D] @
// [B,H,D,L_k] matmul that produces wrong output; this isolates whether
// the executor's MatMul handler correctly batches over outer dims.

use anyhow::Result;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::Device;

fn main() -> Result<()> {
    let (b, h, m, n, k) = (1, 8, 201, 5184, 32);
    let mut q = vec![0f32; b * h * m * k];
    let mut kt = vec![0f32; b * h * k * n];
    for i in 0..q.len() {
        q[i] = ((i as f32 + 1.0) * 0.0001).sin();
    }
    for i in 0..kt.len() {
        kt[i] = ((i as f32 + 1.0) * 0.00007).cos();
    }

    // ── Reference: per-(b,h) 2D sgemm ────────────────────────────
    let mut ref_out = vec![0f32; b * h * m * n];
    for bi in 0..b {
        for hi in 0..h {
            let q_off = (bi * h + hi) * m * k;
            let k_off = (bi * h + hi) * k * n;
            let o_off = (bi * h + hi) * m * n;
            rlx_cpu::blas::sgemm(
                &q[q_off..q_off + m * k],
                &kt[k_off..k_off + k * n],
                &mut ref_out[o_off..o_off + m * n],
                m,
                k,
                n,
            );
        }
    }

    // ── IR: g.matmul on 4D ───────────────────────────────────────
    let mut g = Graph::new("bmm");
    let f = DType::F32;
    let q_in = g.input("q", Shape::new(&[b, h, m, k], f));
    let k_in = g.input("k", Shape::new(&[b, h, k, n], f));
    let out = g.matmul(q_in, k_in, Shape::new(&[b, h, m, n], f));
    g.set_outputs(vec![out]);
    let mut compiled = rlx_models::flow_bridge::compile_graph_encoder(Device::Cpu, g)?;
    let outputs = compiled.run(&[("q", q.as_slice()), ("k", kt.as_slice())]);
    let ir_out = outputs.into_iter().next().unwrap();

    // ── Compare ──────────────────────────────────────────────────
    let n_elem = ref_out.len();
    let mut mad = 0f32;
    let mut idx = 0;
    for i in 0..n_elem {
        let d = (ref_out[i] - ir_out[i]).abs();
        if d > mad {
            mad = d;
            idx = i;
        }
    }
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..n_elem {
        let a = ref_out[i] as f64;
        let b = ir_out[i] as f64;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    let cos = 1.0 - dot / (na * nb).sqrt();
    println!("4D batched matmul: len={n_elem} mad={mad:.3e} cos_dist={cos:.3e} idx={idx}");
    println!("ref [0..4] = {:?}", &ref_out[..4]);
    println!("ir  [0..4] = {:?}", &ir_out[..4]);
    println!("ref [last 4] = {:?}", &ref_out[n_elem - 4..]);
    println!("ir  [last 4] = {:?}", &ir_out[n_elem - 4..]);
    Ok(())
}
