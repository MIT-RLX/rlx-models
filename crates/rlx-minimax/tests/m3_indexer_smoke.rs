// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! MSA lightning-indexer value checks. Builds a mini-flow that outputs the
//! block-sparse additive bias `[1, num_heads, seq, seq]` and asserts its
//! deterministic structure (independent of the random indexer weights):
//!   * causal — every `k > q` entry is masked (large negative);
//!   * local block always visible — the diagonal `k == q` is kept (0);
//!   * when `topk_blocks >= n_blocks`, the bias collapses to a pure causal mask.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};
use rlx_minimax::m3::indexer::{IndexerDims, emit_msa_bias};
use rlx_minimax::m3::{ROPE_COS, ROPE_SIN, rope_tables};
use rlx_runtime::Device;
use std::collections::HashMap;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.4
        })
        .collect()
}

fn run_bias(seq: usize, block: usize, topk: usize) -> (Vec<f32>, usize) {
    let hidden = 8usize;
    let num_heads = 4usize;
    let index_n_heads = 2usize;
    let index_head_dim = 4usize;
    let n_rot = 2usize;

    let dims = IndexerDims {
        hidden,
        num_heads,
        index_n_heads,
        index_head_dim,
        n_rot,
        block_size: block,
        topk_blocks: topk,
        local_blocks: 1,
        eps: 1e-6,
        seq,
    };

    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1u64;
    let mut put = |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: &str, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        seed += 3;
        t.insert(k.to_string(), (fill(n, seed), shape));
    };
    put(
        &mut t,
        "sa.index_q_proj.weight",
        vec![index_n_heads * index_head_dim, hidden],
    );
    put(
        &mut t,
        "sa.index_k_proj.weight",
        vec![index_head_dim, hidden],
    );
    put(&mut t, "sa.index_q_norm.weight", vec![index_head_dim]);
    put(&mut t, "sa.index_k_norm.weight", vec![index_head_dim]);
    let mut wm = WeightMap::from_tensors(t);

    let f = DType::F32;
    let half = n_rot / 2;
    let flow = ModelFlow::new("m3_indexer")
        .with_profile(CompileProfile::llama32_prefill())
        .input("hidden", Shape::new(&[1, seq, hidden], f))
        .input(ROPE_COS, Shape::new(&[seq, half], f))
        .input(ROPE_SIN, Shape::new(&[seq, half], f))
        .plugin_named("idx", move |emit, _prev| {
            let h = emit.flow_input("hidden")?.hir_id();
            let bias = emit_msa_bias(emit, "sa", h, dims)?;
            Ok(Some(
                emit.wrap(bias, Shape::new(&[1, num_heads, seq, seq], f)),
            ))
        })
        .output("bias");

    let built = flow
        .build_with(&mut rlx_core::flow_util::WeightMapSource(&mut wm), None)
        .expect("build indexer flow");
    let mut compiled = compile_built(built, Device::Cpu).expect("compile indexer flow");

    let hidden_data = fill(seq * hidden, 99);
    let (cos, sin) = rope_tables(seq, n_rot, 10000.0);
    let out = compiled
        .run(&[
            ("hidden", hidden_data.as_slice()),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("indexer forward returned output");
    (out, num_heads)
}

const MASKED: f32 = -1e29;

#[test]
fn msa_bias_is_causal_and_local_visible() {
    let seq = 6;
    let block = 2;
    let topk = 2; // < n_blocks(=3): genuinely sparse
    let (bias, heads) = run_bias(seq, block, topk);
    assert_eq!(bias.len(), heads * seq * seq);
    for h in 0..heads {
        for q in 0..seq {
            for k in 0..seq {
                let v = bias[(h * seq + q) * seq + k];
                assert!(v.is_finite(), "bias must be finite");
                if k > q {
                    assert!(
                        v <= MASKED,
                        "future key q={q} k={k} must be masked, got {v}"
                    );
                }
            }
            // Local block (contains the diagonal) is always kept → k==q is 0.
            let diag = bias[(h * seq + q) * seq + q];
            assert!(
                diag.abs() < 1.0,
                "diagonal q={q} must be kept (~0), got {diag}"
            );
        }
    }
}

#[test]
fn msa_bias_collapses_to_causal_when_all_blocks_selected() {
    let seq = 6;
    let block = 2;
    let topk = 8; // >= n_blocks(=3): every block selected → pure causal
    let (bias, heads) = run_bias(seq, block, topk);
    for h in 0..heads {
        for q in 0..seq {
            for k in 0..seq {
                let v = bias[(h * seq + q) * seq + k];
                if k <= q {
                    assert!(v.abs() < 1.0, "kept (q={q},k={k}) must be ~0, got {v}");
                } else {
                    assert!(v <= MASKED, "future (q={q},k={k}) must be masked, got {v}");
                }
            }
        }
    }
}
