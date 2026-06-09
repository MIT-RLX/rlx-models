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

//! Probe whether Accelerate's sgemm uses Apple AMX more aggressively as N grows.
//!
//! If sgemm(M, K, N=16) costs not-much-more than sgemm(M, K, N=1), we have a
//! free 16× per-call FLOPs upgrade — and Option B (batching substeps) becomes
//! viable IF we can find a way to feed it ≥16 independent inputs.
//!
//! Conversely, if N=16 sgemm scales linearly with N, AMX isn't kicking in and
//! batching doesn't pay.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

#[derive(Clone, Copy)]
struct S {
    m: usize,
    k: usize,
    label: &'static str,
}

fn bench(shape: S, iters: usize) {
    let S { m, k, label } = shape;
    let mut rng = StdRng::seed_from_u64(7);
    let w: Vec<f32> = (0..m * k).map(|_| rng.r#gen::<f32>() - 0.5).collect();

    // Try N = 1, 2, 4, 8, 16, 32 to see when (if) AMX kicks in.
    let ns = [1usize, 2, 4, 8, 16, 32, 64];
    print!("{label:>22}  M={m:>5} K={k:>5}  ");
    let mut t1 = 0f64;
    for &n in &ns {
        let x: Vec<f32> = (0..k * n).map(|_| rng.r#gen::<f32>() - 0.5).collect();
        let mut out = vec![0f32; m * n];
        for _ in 0..3 {
            rlx_cpu::blas::sgemm(&w, &x, &mut out, m, k, n);
        }
        let t = Instant::now();
        for _ in 0..iters {
            rlx_cpu::blas::sgemm(&w, &x, &mut out, m, k, n);
        }
        let elapsed = t.elapsed().as_secs_f64() / iters as f64;
        if n == 1 {
            t1 = elapsed;
        }
        let per_col = elapsed / n as f64;
        let scaling = elapsed / t1;
        print!(
            "N={n}:{:.1}µs ({:.1}µs/col, {:.1}×)  ",
            elapsed * 1e6,
            per_col * 1e6,
            scaling
        );
    }
    println!();
}

fn main() {
    println!("=== Accelerate sgemm scaling with N (does AMX kick in?) ===");
    println!("    If per-col cost stays flat as N grows, AMX is active → batching is profitable.");
    println!("    If per-col cost is constant, N=16 takes ~16× longer than N=1 → no AMX win.\n");
    let shapes = [
        S {
            m: 2048,
            k: 768,
            label: "CP wqkv",
        },
        S {
            m: 768,
            k: 1024,
            label: "CP wo",
        },
        S {
            m: 6144,
            k: 768,
            label: "CP gate+up",
        },
        S {
            m: 768,
            k: 3072,
            label: "CP down",
        },
        S {
            m: 8192,
            k: 1024,
            label: "Talker gate+up",
        },
    ];
    let iters = 200;
    for s in shapes {
        bench(s, iters);
    }
}
