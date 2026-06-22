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

//! End-to-end `propose(n=N)` bench using the HIR-compiled draft step
//! on real RedHatAI weights. Maintains KV cache across N steps; each
//! step uses a separately compiled graph (one per `past_seq`).
//!
//! Headline comparison:
//! - llama.cpp b9606 EAGLE3 full pipeline: **6.1 tok/s** (measured)
//! - rlx-eagle3 scalar reference: **2.3 tok/s** (measured)
//! - rlx-eagle3 HIR: this bench
//!
//! Run with:
//! ```bash
//! cargo run -p rlx-eagle3 --release --features "metal mlx" \
//!     --example bench_propose_e2e -- \
//!     /Users/Shared/rlx-models/.eagle3-bench/weights/draft
//! ```

use anyhow::{Context, Result};
use rlx_eagle3::config::Eagle3Config;
use rlx_eagle3::draft::{DraftGeom, DraftWeightRefs, Eagle3DraftReference};
use rlx_eagle3::hir_draft::{build_draft_step_graph, input_names as I, tensor_names as T};
use rlx_eagle3::weights::Eagle3DraftWeights;
use rlx_runtime::{CompiledGraph, Device, Session, is_available};
use std::path::PathBuf;
use std::time::Instant;

const SPECULATIVE_TOKENS: usize = 3;
const WARMUP_PROPOSES: usize = 3;
const ITERS: usize = 20;

/// Transpose `[rows, cols]` row-major → `[cols, rows]`.
fn transpose_rc(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Pre-transposed weight cache so each backend compile doesn't redo
/// it. Saves several seconds of host work per backend.
struct TransposedWeights {
    q_t: Vec<f32>,
    k_t: Vec<f32>,
    v_t: Vec<f32>,
    o_t: Vec<f32>,
    gate_t: Vec<f32>,
    up_t: Vec<f32>,
    down_t: Vec<f32>,
    lm_t: Vec<f32>,
    zero_beta: Vec<f32>,
}

impl TransposedWeights {
    fn from_weights(weights: &Eagle3DraftWeights, geom: DraftGeom) -> Result<Self> {
        let get = |name: &str| -> Result<&[f32]> {
            weights
                .get(name)
                .map(|t| t.data.as_slice())
                .with_context(|| format!("missing {name}"))
        };
        let q_dim = geom.n_heads * geom.head_dim;
        let kv_dim = geom.n_kv_heads * geom.head_dim;
        let two_h = 2 * geom.h_draft;
        Ok(Self {
            q_t: transpose_rc(get("decoder.self_attn.q_proj.weight")?, q_dim, two_h),
            k_t: transpose_rc(get("decoder.self_attn.k_proj.weight")?, kv_dim, two_h),
            v_t: transpose_rc(get("decoder.self_attn.v_proj.weight")?, kv_dim, two_h),
            o_t: transpose_rc(get("decoder.self_attn.o_proj.weight")?, geom.h_draft, q_dim),
            gate_t: transpose_rc(
                get("decoder.mlp.gate_proj.weight")?,
                geom.intermediate,
                geom.h_draft,
            ),
            up_t: transpose_rc(
                get("decoder.mlp.up_proj.weight")?,
                geom.intermediate,
                geom.h_draft,
            ),
            down_t: transpose_rc(
                get("decoder.mlp.down_proj.weight")?,
                geom.h_draft,
                geom.intermediate,
            ),
            lm_t: transpose_rc(get("lm_head.weight")?, geom.draft_vocab, geom.h_draft),
            zero_beta: vec![0.0f32; geom.h_draft],
        })
    }
}

fn set_all_params(
    compiled: &mut CompiledGraph,
    weights: &Eagle3DraftWeights,
    tr: &TransposedWeights,
) -> Result<()> {
    let get = |name: &str| -> Result<&[f32]> {
        weights
            .get(name)
            .map(|t| t.data.as_slice())
            .with_context(|| format!("missing {name}"))
    };
    compiled.set_param(T::INPUT_LAYERNORM, get("decoder.input_layernorm.weight")?);
    compiled.set_param(T::HIDDEN_NORM, get("decoder.hidden_norm.weight")?);
    compiled.set_param(
        T::POST_ATTN_LN,
        get("decoder.post_attention_layernorm.weight")?,
    );
    compiled.set_param(T::NORM, get("norm.weight")?);
    compiled.set_param(T::Q_PROJ, &tr.q_t);
    compiled.set_param(T::K_PROJ, &tr.k_t);
    compiled.set_param(T::V_PROJ, &tr.v_t);
    compiled.set_param(T::O_PROJ, &tr.o_t);
    compiled.set_param(T::GATE_PROJ, &tr.gate_t);
    compiled.set_param(T::UP_PROJ, &tr.up_t);
    compiled.set_param(T::DOWN_PROJ, &tr.down_t);
    compiled.set_param(T::LM_HEAD, &tr.lm_t);
    compiled.set_param(T::ZERO_BETA, &tr.zero_beta);
    Ok(())
}

/// Compute RoPE (cos, sin) row for a single position.
fn rope_row(position: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for k in 0..half {
        let exp = -(2.0 * k as f64) / (head_dim as f64);
        let freq = (theta as f64).powf(exp);
        let angle = (position as f64) * freq;
        cos[k] = angle.cos() as f32;
        sin[k] = angle.sin() as f32;
    }
    (cos, sin)
}

/// One full propose(n=N): N steps with growing past_seq.
/// Reuses compiled graphs from `graphs[past_seq]`. Returns the N
/// proposed target-token ids + cumulative wall time.
#[allow(clippy::too_many_arguments)]
fn propose_once(
    graphs: &mut [CompiledGraph],
    weights: &Eagle3DraftWeights,
    geom: DraftGeom,
    aux: &[Vec<f32>],
    initial_target_token: u32,
    n: usize,
) -> Result<Vec<u32>> {
    // First-step hidden via fc fusion (reuse the scalar reference's
    // `init_hidden` — it's cheap and shared).
    let refs = DraftWeightRefs::from_weights(
        weights,
        &Eagle3Config::from_bytes(
            // Minimal cfg recoverable from geom for the reference helper.
            format!(
                r#"{{
            "draft_vocab_size": {dv},
            "target_hidden_size": {h},
            "norm_before_residual": true,
            "transformer_layer_config": {{
                "model_type": "llama",
                "hidden_size": {h}, "intermediate_size": {i},
                "num_hidden_layers": 1, "num_attention_heads": {nh},
                "num_key_value_heads": {nk}, "head_dim": {hd},
                "vocab_size": {tv}, "rms_norm_eps": {eps}
            }}
        }}"#,
                dv = geom.draft_vocab,
                h = geom.h_draft,
                i = geom.intermediate,
                nh = geom.n_heads,
                nk = geom.n_kv_heads,
                hd = geom.head_dim,
                tv = geom.target_vocab,
                eps = geom.rms_eps
            )
            .as_bytes(),
        )?,
    )?;
    let cfg_for_ref = Eagle3Config::from_bytes(
        format!(
            r#"{{
            "draft_vocab_size": {dv},
            "target_hidden_size": {h},
            "norm_before_residual": true,
            "transformer_layer_config": {{
                "model_type": "llama",
                "hidden_size": {h}, "intermediate_size": {i},
                "num_hidden_layers": 1, "num_attention_heads": {nh},
                "num_key_value_heads": {nk}, "head_dim": {hd},
                "vocab_size": {tv}, "rms_norm_eps": {eps}
            }}
        }}"#,
            dv = geom.draft_vocab,
            h = geom.h_draft,
            i = geom.intermediate,
            nh = geom.n_heads,
            nk = geom.n_kv_heads,
            hd = geom.head_dim,
            tv = geom.target_vocab,
            eps = geom.rms_eps
        )
        .as_bytes(),
    )?;
    let draft_ref = Eagle3DraftReference::new(&cfg_for_ref, refs);
    let h0 = draft_ref.init_hidden(aux);
    let _ = draft_ref; // we only used it for init_hidden

    // KV cache across steps, host-side.
    let kv_dim = geom.n_kv_heads * geom.head_dim;
    let mut past_k: Vec<f32> = Vec::new();
    let mut past_v: Vec<f32> = Vec::new();
    let embed_tokens = weights
        .get("embed_tokens.weight")
        .context("missing embed_tokens.weight")?;
    let embed_tokens = &embed_tokens.data;

    let mut prev_hidden = h0;
    let mut prev_target_token = initial_target_token;
    let mut proposed: Vec<u32> = Vec::with_capacity(n);

    for (step, compiled) in graphs.iter_mut().enumerate().take(n) {
        // Host-side embed lookup for the previous target token.
        let row_off = (prev_target_token as usize) * geom.h_draft;
        let prev_embed = &embed_tokens[row_off..row_off + geom.h_draft];
        let (rope_cos, rope_sin) = rope_row(step, geom.head_dim, geom.rope_theta);

        let outs = compiled.run(&[
            (I::PREV_EMBED, prev_embed),
            (I::PREV_HIDDEN, prev_hidden.as_slice()),
            (I::PAST_K, past_k.as_slice()),
            (I::PAST_V, past_v.as_slice()),
            (I::ROPE_COS, rope_cos.as_slice()),
            (I::ROPE_SIN, rope_sin.as_slice()),
        ]);
        let logits = &outs[0];
        let new_hidden = outs[1].clone();
        let new_k = &outs[2];
        let new_v = &outs[3];

        // Greedy argmax → draft id → target id via d2t (offset).
        let mut best = 0u32;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i as u32;
            }
        }
        let target_id = best + weights.d2t()[best as usize];
        proposed.push(target_id);

        // Advance state.
        prev_hidden = new_hidden;
        prev_target_token = target_id;
        // Append new k/v to the host-side past KV cache (just the
        // last row — the graph already concatenated past + new and
        // returned the full new_k/new_v with cur_seq rows).
        let cur_seq = step + 1;
        debug_assert_eq!(new_k.len(), cur_seq * kv_dim);
        past_k = new_k.to_vec();
        past_v = new_v.to_vec();
    }

    Ok(proposed)
}

fn bench_device(
    device: Device,
    label: &str,
    weights: &Eagle3DraftWeights,
    geom: DraftGeom,
    tr: &TransposedWeights,
) -> Result<f64> {
    if !is_available(device) {
        println!("   [{label}] not available — skipped");
        return Ok(f64::NAN);
    }
    // Compile one graph per past_seq value (0, 1, ..., N-1).
    let session = Session::new(device);
    let mut graphs: Vec<CompiledGraph> = Vec::with_capacity(SPECULATIVE_TOKENS);
    for past_seq in 0..SPECULATIVE_TOKENS {
        let g = build_draft_step_graph(geom, past_seq);
        let mut compiled = session.compile(g);
        set_all_params(&mut compiled, weights, tr)?;
        graphs.push(compiled);
    }

    // Synthesize a stable verifier-aux input set (3 layers, h_target).
    let aux: Vec<Vec<f32>> = (0..3)
        .map(|l| {
            (0..geom.h_target)
                .map(|d| ((d as f32) * 0.001 - (l as f32) * 0.0007).sin())
                .collect()
        })
        .collect();
    let initial_token: u32 = 1;

    // Warmup
    for _ in 0..WARMUP_PROPOSES {
        let _ = propose_once(
            &mut graphs,
            weights,
            geom,
            &aux,
            initial_token,
            SPECULATIVE_TOKENS,
        )?;
    }

    let t0 = Instant::now();
    let mut last: Vec<u32> = Vec::new();
    for _ in 0..ITERS {
        last = propose_once(
            &mut graphs,
            weights,
            geom,
            &aux,
            initial_token,
            SPECULATIVE_TOKENS,
        )?;
    }
    let total = t0.elapsed().as_secs_f64();
    let per_propose_ms = total * 1000.0 / ITERS as f64;
    let tokens_per_s = (ITERS * SPECULATIVE_TOKENS) as f64 / total;

    println!(
        "   [{label:6}] {per_propose_ms:7.2} ms/propose(n={SPECULATIVE_TOKENS}) · {tokens_per_s:6.1} tok/s · sample tokens={last:?}"
    );
    Ok(tokens_per_s)
}

fn main() -> Result<()> {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: bench_propose_e2e <draft-dir>")?;

    println!("→ Loading config + weights from {:?}", dir);
    let cfg = Eagle3Config::from_file(dir.join("config.json"))?;
    let geom = DraftGeom::from_cfg(&cfg);
    let t0 = Instant::now();
    let weights = Eagle3DraftWeights::open(dir.join("model.safetensors"))?;
    println!(
        "   loaded {} tensors in {:.2}s",
        weights.len(),
        t0.elapsed().as_secs_f64()
    );

    println!("\n→ Pre-transposing 8 weights once for all backends...");
    let t0 = Instant::now();
    let tr = TransposedWeights::from_weights(&weights, geom)?;
    println!("   {:.2}s", t0.elapsed().as_secs_f64());

    println!(
        "\n→ Benching propose(n={SPECULATIVE_TOKENS}) end-to-end across backends ({WARMUP_PROPOSES} warmup + {ITERS} timed)\n"
    );

    let mut results: Vec<(&str, f64)> = Vec::new();
    for (device, label) in [
        (Device::Cpu, "CPU"),
        (Device::Metal, "Metal"),
        (Device::Mlx, "MLX"),
    ] {
        match bench_device(device, label, &weights, geom, &tr) {
            Ok(tps) if !tps.is_nan() => results.push((label, tps)),
            Ok(_) => {}
            Err(e) => println!("   [{label}] FAILED: {e}"),
        }
    }

    println!("\n→ Headline comparison:");
    println!("   {:<15} {:>10}", "Pipeline", "tok/s");
    println!("   {:-<15} {:->10}", "", "");
    println!("   {:<15} {:>10.2}", "llama.cpp b9606", 6.1);
    println!("   {:<15} {:>10.2}", "rlx scalar ref", 2.3);
    for (label, tps) in &results {
        println!("   {:<15} {:>10.2}", format!("rlx HIR {label}"), tps);
    }
    println!("\n✓ DONE.");
    Ok(())
}
