//! `profile_moe` — thunk-level time breakdown of one LatentMoE block at
//! realistic per-expert dims (hidden 7168, latent 3584, moe_inter 3072, top_k
//! 8) with few experts so the weights fit RAM. Confirms whether the `situ`
//! activation (div/tanh/sigmoid/mul) is worth a fused kernel vs the grouped
//! matmuls. Run with RLX_PROFILE_THUNKS to get the per-kind breakdown:
//!   RLX_PROFILE_THUNKS=1 cargo run -p rlx-kimi-k3 --example profile_moe --release -- 64

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::moe::{MoeDims, MoeWeights, build_latent_moe};
use rlx_runtime::Device;
use std::collections::HashMap;

fn fill(n: usize, s: u64) -> Vec<f32> {
    let mut x = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((x >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.05
        })
        .collect()
}

fn main() {
    let seq: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    // Real per-expert dims; small expert count to fit RAM.
    let d = MoeDims {
        hidden: 7168,
        latent: 3584,
        moe_inter: 3072,
        num_experts: 8,
        top_k: 8,
        num_shared: 2,
        routed_scaling: 1.0,
        eps: 1e-5,
        situ_beta: 4.0,
        situ_linear_beta: Some(4.0),
        batch: 1,
        seq,
    };
    let (h, ll, i, e, s) = (d.hidden, d.latent, d.moe_inter, d.num_experts, d.num_shared);
    let w = MoeWeights {
        router: fill(h * e, 1),
        e_score_bias: vec![0.0; e],
        down_latent: fill(h * ll, 2),
        up_latent: fill(ll * h, 3),
        routed_norm: vec![1.0; ll],
        experts_gate_up: fill(e * ll * 2 * i, 4),
        experts_down: fill(e * i * ll, 5),
        shared_gate: fill(h * s * i, 6),
        shared_up: fill(h * s * i, 7),
        shared_down: fill(s * i * h, 8),
    };
    let mut hir = HirModule::new("moe");
    let mut g = HirMut::new(&mut hir);
    let hin = g.input("h", Shape::new(&[1, seq, h], DType::F32));
    let mut p = HashMap::new();
    let out = build_latent_moe(&mut g, &mut p, "moe", hin, &w, d).unwrap();
    g.set_outputs(vec![out]);
    let mut c = compile_built(built_from_hir(hir, p).unwrap(), Device::Cpu).unwrap();
    let feed = fill(seq * h, 100);
    // warmup + timed
    let _ = c.run(&[("h", feed.as_slice())]);
    let t = std::time::Instant::now();
    for _ in 0..iters {
        let _ = c.run(&[("h", feed.as_slice())]);
    }
    eprintln!(
        "LatentMoE seq={seq} experts={e} top_k={} : {:.3} ms/iter  (set RLX_PROFILE_THUNKS=1 for kind breakdown)",
        d.top_k,
        t.elapsed().as_secs_f64() * 1e3 / iters as f64
    );
}
