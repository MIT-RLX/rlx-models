//! `profile_mla` — thunk breakdown of one MLA layer at DeepSeek/Kimi-style dims
//! (num_heads 128, qk_head_dim 192 = nope 128 + rope 64, v_head_dim 128). MLA
//! zero-pads V from 128→192 to fit the single-head_dim attention op, then slices
//! the output back — this bounds the win from a real `v_head_dim` on Op::Attention.
//!   RLX_PROFILE_THUNKS=1 cargo run -p rlx-kimi-k3 --example profile_mla --release -- 256

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::mla::{MlaDims, MlaWeights, build_mla_layer};
use rlx_runtime::Device;
use std::collections::HashMap;

fn fill(n: usize, s: u64) -> Vec<f32> {
    let mut x = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((x >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.03
        })
        .collect()
}

fn main() {
    let seq: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let d = MlaDims {
        hidden: 7168,
        num_heads: 128,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
        eps: 1e-5,
        batch: 1,
        seq,
    };
    let nh = d.num_heads;
    let qk = d.qk_nope_head_dim + d.qk_rope_head_dim;
    let w = MlaWeights {
        q_a_proj: fill(d.hidden * d.q_lora_rank, 1),
        q_a_layernorm: vec![1.0; d.q_lora_rank],
        q_b_proj: fill(d.q_lora_rank * nh * qk, 2),
        kv_a_proj_with_mqa: fill(d.hidden * (d.kv_lora_rank + d.qk_rope_head_dim), 3),
        kv_a_layernorm: vec![1.0; d.kv_lora_rank],
        kv_b_proj: fill(d.kv_lora_rank * nh * (d.qk_nope_head_dim + d.v_head_dim), 4),
        g_proj: fill(d.hidden * nh * d.v_head_dim, 5),
        o_proj: fill(nh * d.v_head_dim * d.hidden, 6),
    };
    let mut hir = HirModule::new("mla");
    let mut g = HirMut::new(&mut hir);
    let hin = g.input("h", Shape::new(&[1, seq, d.hidden], DType::F32));
    let mut p = HashMap::new();
    let out = build_mla_layer(&mut g, &mut p, "mla", hin, &w, d).unwrap();
    g.set_outputs(vec![out]);
    let mut c = compile_built(built_from_hir(hir, p).unwrap(), Device::Cpu).unwrap();
    let feed = fill(seq * d.hidden, 100);
    let _ = c.run(&[("h", feed.as_slice())]);
    let t = std::time::Instant::now();
    for _ in 0..iters {
        let _ = c.run(&[("h", feed.as_slice())]);
    }
    eprintln!(
        "MLA seq={seq} heads={nh} qk={qk} v={} : {:.3} ms/iter",
        d.v_head_dim,
        t.elapsed().as_secs_f64() * 1e3 / iters as f64
    );
}
