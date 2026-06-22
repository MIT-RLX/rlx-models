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

//! Multi-op submission bench for the EAGLE3 draft step.
//!
//! This is what we hoped the single-`lm_head` micro-bench would tell
//! us, except now MLX gets the full graph (~10 ops) in one compiled
//! trace — exactly the regime MLX is designed for. We expect MLX's
//! per-call floor to amortize across 10 ops instead of pay-per-op.
//!
//! Run with:
//! ```bash
//! cargo run -p rlx-eagle3 --release --features "mlx metal" \
//!     --example bench_draft_step_backends -- \
//!     /Users/Shared/rlx-models/.eagle3-bench/weights/draft
//! ```

use anyhow::{Context, Result};
use rlx_eagle3::config::Eagle3Config;
use rlx_eagle3::draft::DraftGeom;
use rlx_eagle3::hir_draft::{build_draft_step_graph, input_names as I, tensor_names as T};
use rlx_eagle3::weights::Eagle3DraftWeights;
use rlx_runtime::{Device, Session, is_available};
use std::path::PathBuf;
use std::time::Instant;

const WARMUP: usize = 3;
const ITERS: usize = 50;
/// Use past_seq=1 so the KV-concat inputs are non-empty (Metal's
/// MPSNDArray crashes on zero-length descriptors).
const PAST_SEQ: usize = 1;

/// Transpose `[rows, cols]` row-major → `[cols, rows]` row-major.
/// Called once per param at startup.
fn transpose_rc(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

fn bench_device(
    device: Device,
    label: &str,
    geom: DraftGeom,
    weights: &Eagle3DraftWeights,
) -> Result<f64> {
    if !is_available(device) {
        println!("   [{label}] not available — skipped");
        return Ok(f64::NAN);
    }

    let graph = build_draft_step_graph(geom, PAST_SEQ);
    let session = Session::new(device);
    let mut compiled = session.compile(graph);

    let get = |name: &str| -> Result<&[f32]> {
        weights
            .get(name)
            .map(|t| t.data.as_slice())
            .with_context(|| format!("missing tensor {name}"))
    };

    // ── Load params (transposed where the mm convention requires) ──
    // embed_tokens stays on host — prev_embed is gathered before run().
    compiled.set_param(T::INPUT_LAYERNORM, get("decoder.input_layernorm.weight")?);
    compiled.set_param(T::HIDDEN_NORM, get("decoder.hidden_norm.weight")?);
    compiled.set_param(
        T::POST_ATTN_LN,
        get("decoder.post_attention_layernorm.weight")?,
    );
    compiled.set_param(T::NORM, get("norm.weight")?);
    // q/k/v_proj on disk: [q_dim or kv_dim, 2*H]. mm wants [2*H, out].
    let q_dim = geom.n_heads * geom.head_dim;
    let kv_dim = geom.n_kv_heads * geom.head_dim;
    let two_h = 2 * geom.h_draft;
    let q_t = transpose_rc(get("decoder.self_attn.q_proj.weight")?, q_dim, two_h);
    let k_t = transpose_rc(get("decoder.self_attn.k_proj.weight")?, kv_dim, two_h);
    let v_t = transpose_rc(get("decoder.self_attn.v_proj.weight")?, kv_dim, two_h);
    compiled.set_param(T::Q_PROJ, &q_t);
    compiled.set_param(T::K_PROJ, &k_t);
    compiled.set_param(T::V_PROJ, &v_t);
    // o_proj on disk: [H, q_dim]. mm wants [q_dim, H].
    let o_t = transpose_rc(get("decoder.self_attn.o_proj.weight")?, geom.h_draft, q_dim);
    compiled.set_param(T::O_PROJ, &o_t);
    // gate/up_proj on disk: [I, H]. mm wants [H, I].
    let gate_t = transpose_rc(
        get("decoder.mlp.gate_proj.weight")?,
        geom.intermediate,
        geom.h_draft,
    );
    let up_t = transpose_rc(
        get("decoder.mlp.up_proj.weight")?,
        geom.intermediate,
        geom.h_draft,
    );
    compiled.set_param(T::GATE_PROJ, &gate_t);
    compiled.set_param(T::UP_PROJ, &up_t);
    // down_proj on disk: [H, I]. mm wants [I, H].
    let down_t = transpose_rc(
        get("decoder.mlp.down_proj.weight")?,
        geom.h_draft,
        geom.intermediate,
    );
    compiled.set_param(T::DOWN_PROJ, &down_t);
    // lm_head on disk: [V_draft, H]. mm wants [H, V_draft].
    let lm_t = transpose_rc(get("lm_head.weight")?, geom.draft_vocab, geom.h_draft);
    compiled.set_param(T::LM_HEAD, &lm_t);
    // zero_beta: all zeros, length h_draft.
    let zero_beta = vec![0.0f32; geom.h_draft];
    compiled.set_param(T::ZERO_BETA, &zero_beta);

    // ── Per-call inputs ───────────────────────────────────────────
    let prev_hidden: Vec<f32> = (0..geom.h_draft)
        .map(|i| ((i as f32) * 0.001).sin())
        .collect();
    // Host-side embed lookup: pull row `tok` from embed_tokens.
    let embed_tokens = get("embed_tokens.weight")?;
    let tok: usize = 1;
    let prev_embed: Vec<f32> = embed_tokens[tok * geom.h_draft..(tok + 1) * geom.h_draft].to_vec();
    // past_seq=PAST_SEQ — fill with synthetic but finite data.
    let past_k: Vec<f32> = (0..PAST_SEQ * kv_dim)
        .map(|i| ((i as f32) * 0.0001).cos())
        .collect();
    let past_v: Vec<f32> = (0..PAST_SEQ * kv_dim)
        .map(|i| ((i as f32) * 0.0001).sin())
        .collect();
    // rope at position 0: cos=1, sin=0.
    let half = geom.head_dim / 2;
    let rope_cos: Vec<f32> = vec![1.0; half];
    let rope_sin: Vec<f32> = vec![0.0; half];

    let inputs = vec![
        (I::PREV_EMBED, prev_embed.as_slice()),
        (I::PREV_HIDDEN, prev_hidden.as_slice()),
        (I::PAST_K, past_k.as_slice()),
        (I::PAST_V, past_v.as_slice()),
        (I::ROPE_COS, rope_cos.as_slice()),
        (I::ROPE_SIN, rope_sin.as_slice()),
    ];

    // Warmup
    for _ in 0..WARMUP {
        let _ = compiled.run(&inputs);
    }

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = compiled.run(&inputs);
    }
    let total = t0.elapsed().as_secs_f64();
    let per_call_ms = total * 1000.0 / ITERS as f64;

    // Sanity-check the output shapes + finiteness.
    let outs = compiled.run(&inputs);
    let logits = outs.first().context("logits missing")?;
    let new_hidden = outs.get(1).context("new_hidden missing")?;
    if logits.len() != geom.draft_vocab {
        anyhow::bail!(
            "[{label}] logits len {} != draft_vocab {}",
            logits.len(),
            geom.draft_vocab
        );
    }
    if new_hidden.len() != geom.h_draft {
        anyhow::bail!(
            "[{label}] new_hidden len {} != h_draft {}",
            new_hidden.len(),
            geom.h_draft
        );
    }
    if !logits.iter().all(|v| v.is_finite()) {
        anyhow::bail!("[{label}] non-finite logits");
    }
    if !new_hidden.iter().all(|v| v.is_finite()) {
        anyhow::bail!("[{label}] non-finite new_hidden");
    }

    println!("   [{label:6}] {per_call_ms:7.3} ms/step (ITERS={ITERS})");
    Ok(per_call_ms)
}

fn main() -> Result<()> {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: bench_draft_step_backends <draft-dir>")?;

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
    println!(
        "   geom: h_draft={}, n_heads={}, n_kv={}, head_dim={}, intermediate={}, draft_vocab={}, target_vocab={}",
        geom.h_draft,
        geom.n_heads,
        geom.n_kv_heads,
        geom.head_dim,
        geom.intermediate,
        geom.draft_vocab,
        geom.target_vocab,
    );

    println!(
        "\n→ Benching one draft step (past_seq={PAST_SEQ}) — ~10 ops fused into a single compiled graph\n"
    );

    let mut results: Vec<(&str, f64)> = Vec::new();
    for (device, label) in [
        (Device::Cpu, "CPU"),
        (Device::Metal, "Metal"),
        (Device::Mlx, "MLX"),
    ] {
        let ms = bench_device(device, label, geom, &weights)?;
        if !ms.is_nan() {
            results.push((label, ms));
        }
    }

    println!("\n→ Speedup vs CPU:");
    let cpu_ms = results.iter().find(|(n, _)| *n == "CPU").map(|(_, m)| *m);
    if let Some(cpu) = cpu_ms {
        for (n, m) in &results {
            if *n == "CPU" {
                continue;
            }
            println!("   {n:6}: {:.2}× faster than CPU ({m:.3} ms)", cpu / m);
        }
    }

    println!(
        "\n✓ DONE. If MLX's gap to Metal narrowed vs the single-op bench,\n  \
         the per-call floor is amortizing as expected for multi-op submissions."
    );
    Ok(())
}
