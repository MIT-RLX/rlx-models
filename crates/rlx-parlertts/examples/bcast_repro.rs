// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0
//! Minimal repro: does rlx-cpu broadcast a rank-4 mask `[1,1,1,K]` correctly over
//! scores `[1,H,Q,K]`? (The T5 padding-mask add pattern.) Fill the mask with
//! -3.4e38 in the last 2 columns; every `[.,.,.,K-2..]` cell of the output must be
//! -3.4e38 and everything else 0. Any other placement = the broadcast bug behind
//! the Parler native encoder's 0.47 cosine.

use anyhow::Result;
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, HirGraphExt, Op, Shape};
use rlx_runtime::{AotCache, CompileOptions, Device};

fn main() -> Result<()> {
    let le = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let cache = AotCache::new(std::env::temp_dir().join("rlx_bcast_repro"));

    // --- TEST 0: REAL-shape rank-4 broadcast Add [1,16,128,128] + [1,1,1,128] ---
    // (262144 elems → the rayon-parallel BinaryFull path; the tiny test used serial.)
    {
        let (h, q, kk) = (16usize, 128usize, 128usize);
        let mut hir = HirModule::new("bcast_big");
        let a = hir.input("a", Shape::new(&[1, h, q, kk], DType::F32));
        let b = hir.input("b", Shape::new(&[1, 1, 1, kk], DType::F32));
        let c = {
            let mut m = HirMut::new(&mut hir);
            m.add_node(
                Op::Binary(BinaryOp::Add),
                vec![a, b],
                Shape::new(&[1, h, q, kk], DType::F32),
            )
        };
        hir.set_outputs(vec![c]);
        let mut g = cache
            .compile_hir_cached("bcast_big", Device::Cpu, hir, &CompileOptions::default())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let a_data = vec![0f32; h * q * kk];
        let mut b_data = vec![0f32; kk];
        b_data[kk - 2] = -3.4028235e38;
        b_data[kk - 1] = -3.4028235e38;
        let out = g.run_typed(&[
            ("a", &le(&a_data), DType::F32),
            ("b", &le(&b_data), DType::F32),
        ]);
        let o: Vec<f32> = out[0]
            .0
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut bad = 0;
        for head in 0..h {
            for row in 0..q {
                for col in 0..kk {
                    let v = o[(head * q + row) * kk + col];
                    if (col >= kk - 2) != (v < -1e30) {
                        bad += 1;
                    }
                }
            }
        }
        println!(
            "[BIG {}x{}x{}] mask-broadcast bad cells: {bad}  row0[..4]={:?} row0[last2]={:?}",
            h,
            q,
            kk,
            &o[..4],
            &o[kk - 2..kk]
        );
        println!(
            "{}",
            if bad == 0 {
                "  ✅ big broadcast CORRECT"
            } else {
                "  ❌ big broadcast WRONG — the rayon-parallel path is the Parler bug"
            }
        );
    }

    let k = 8usize;

    // --- TEST: scalar (rank-0 `[]`) Mul — the T5 mask `(1-mask) * -3.4e38` ---
    // input = (1-mask) = [0,0,0,0,0,0,1,1]; scalar = -3.4e38.
    // Expect [0,0,0,0,0,0,-3.4e38,-3.4e38].
    let mut hir = HirModule::new("scalar_mul");
    let x = hir.input("x", Shape::new(&[1, 1, 1, k], DType::F32));
    let s = hir.param("s", Shape::new(&[], DType::F32)); // rank-0 scalar constant
    let y = {
        let mut m = HirMut::new(&mut hir);
        m.add_node(
            Op::Binary(BinaryOp::Mul),
            vec![x, s],
            Shape::new(&[1, 1, 1, k], DType::F32),
        )
    };
    hir.set_outputs(vec![y]);
    let mut g = cache
        .compile_hir_cached("scalar_mul", Device::Cpu, hir, &CompileOptions::default())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    g.set_param("s", &[-3.4028235e38]);
    g.finalize_params();
    let mut xin = vec![0f32; k];
    xin[k - 2] = 1.0;
    xin[k - 1] = 1.0;
    let out = g.run_typed(&[("x", &le(&xin), DType::F32)]);
    let o: Vec<f32> = out[0]
        .0
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    println!("scalar-Mul out: {:?}", o);
    let ok = o[k - 1] < -1e30 && o[k - 2] < -1e30 && o[0] == 0.0;
    if ok {
        println!("✅ scalar-[] Mul CORRECT — bug is the Slice (or Sub)");
    } else {
        println!(
            "❌ scalar-[] Mul WRONG — THIS is the Parler encoder mask bug (rank-0 broadcast drops the scalar)"
        );
    }
    Ok(())
}
