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

//! Loads the RedHatAI/gemma-4-31B-it-speculator.eagle3 draft from
//! disk and exercises the full `Eagle3Speculator::propose` pipeline
//! against deterministic synthetic verifier hidden states.
//!
//! This is *not* parity-vs-llama.cpp. It is:
//!   - "do we parse the real config.json?" — yes if it runs
//!   - "do we load the real model.safetensors?" — yes if it loads
//!   - "do all expected tensors exist with expected shapes?" — yes if no panic
//!   - "does the pure-Rust forward produce finite outputs?" — yes if asserts hold
//!
//! Run with:
//! ```bash
//! cargo run -p rlx-eagle3 --release --example load_real_draft -- \
//!     /Users/Shared/rlx-models/.eagle3-bench/weights/draft
//! ```

use anyhow::{Context, Result};
use rlx_eagle3::config::Eagle3Config;
use rlx_eagle3::draft::DraftGeom;
use rlx_eagle3::speculator::{Eagle3Speculator, VerifierHiddenSource};
use rlx_eagle3::weights::Eagle3DraftWeights;
use rlx_runtime::spec_decode::Speculator;
use std::path::PathBuf;
use std::time::Instant;

/// Synthetic verifier hidden source — deterministic, just enough to
/// drive `propose()` end-to-end. Real verification needs the rlx-gemma
/// runner glue (TaskList #8).
struct SynthHidden {
    target_hidden: usize,
    layers: usize,
}
impl VerifierHiddenSource for SynthHidden {
    fn aux_hidden_states(&self) -> Result<Vec<Vec<f32>>> {
        Ok((0..self.layers)
            .map(|l| {
                (0..self.target_hidden)
                    .map(|d| ((d as f32) * 0.001 - (l as f32) * 0.0007).sin())
                    .collect()
            })
            .collect())
    }
    fn target_hidden_size(&self) -> usize {
        self.target_hidden
    }
    fn num_aux_layers(&self) -> usize {
        self.layers
    }
}

fn main() -> Result<()> {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: load_real_draft <draft-dir-with-config.json+model.safetensors>")?;
    let config_path = dir.join("config.json");
    let model_path = dir.join("model.safetensors");

    println!("→ Loading config from {:?}", config_path);
    let cfg = Eagle3Config::from_file(&config_path)?;
    let g = DraftGeom::from_cfg(&cfg);
    println!(
        "   draft_vocab={}  target_vocab={}",
        g.draft_vocab, g.target_vocab,
    );
    println!(
        "   h_draft={}  h_target={}  n_heads={}  n_kv_heads={}  head_dim={}",
        g.h_draft, g.h_target, g.n_heads, g.n_kv_heads, g.head_dim,
    );
    println!(
        "   intermediate={}  rope_theta={}  rms_eps={}",
        g.intermediate, g.rope_theta, g.rms_eps,
    );
    println!(
        "   norm_before_residual={}  norm_before_fc={}  speculative_tokens={}",
        g.norm_before_residual, g.norm_before_fc, cfg.speculative_tokens,
    );
    let aux_ids = cfg
        .eagle_aux_hidden_state_layer_ids
        .clone()
        .unwrap_or_default();
    println!("   eagle_aux_hidden_state_layer_ids={:?}", aux_ids);

    println!("\n→ Loading model.safetensors from {:?}", model_path);
    let t0 = Instant::now();
    let weights = Eagle3DraftWeights::open(&model_path)?;
    let load_secs = t0.elapsed().as_secs_f32();
    println!(
        "   {} f32 tensors + d2t LUT loaded in {:.2}s",
        weights.len(),
        load_secs
    );

    // Dump every tensor name + shape so the on-disk layout is visible.
    println!("\n→ All loaded tensors:");
    let mut names: Vec<&str> = weights.tensor_names().collect();
    names.sort();
    for n in &names {
        let t = weights.get(n).unwrap();
        println!("   {:50}  shape={:?}", n, t.shape);
    }

    // Spot-check tensor shapes against architecture pin from
    // speculators/models/eagle3/model_definitions.py.
    //
    // Critical: q_proj output dim is n_heads * head_dim = 8192, NOT
    // h_draft = 5376. The vLLM source explicitly constructs
    //   q_proj = Linear(2*hidden, num_heads * head_dim)
    // (see `_patch_eagle3_projections`). For Llama-style models
    // where n_heads*head_dim == hidden_size this coincidence
    // disappears, but for Gemma 4 it doesn't — o_proj then maps
    // q_dim → h_draft back into the residual.
    //
    // `verifier_norm.weight` is in speculators'
    // `_keys_to_ignore_on_load_missing` — it's filled from the
    // verifier at runtime, not stored on disk for inference.
    let q_dim = g.n_heads * g.head_dim;
    let kv_dim = g.n_kv_heads * g.head_dim;
    let expectations: &[(&str, Vec<usize>)] = &[
        ("fc.weight", vec![g.h_draft, 3 * g.h_target]),
        ("embed_tokens.weight", vec![g.target_vocab, g.h_draft]),
        ("lm_head.weight", vec![g.draft_vocab, g.h_draft]),
        ("norm.weight", vec![g.h_draft]),
        ("decoder.input_layernorm.weight", vec![g.h_draft]),
        ("decoder.hidden_norm.weight", vec![g.h_draft]),
        ("decoder.post_attention_layernorm.weight", vec![g.h_draft]),
        (
            "decoder.self_attn.q_proj.weight",
            vec![q_dim, 2 * g.h_draft],
        ),
        (
            "decoder.self_attn.k_proj.weight",
            vec![kv_dim, 2 * g.h_draft],
        ),
        (
            "decoder.self_attn.v_proj.weight",
            vec![kv_dim, 2 * g.h_draft],
        ),
        ("decoder.self_attn.o_proj.weight", vec![g.h_draft, q_dim]),
        (
            "decoder.mlp.gate_proj.weight",
            vec![g.intermediate, g.h_draft],
        ),
        (
            "decoder.mlp.up_proj.weight",
            vec![g.intermediate, g.h_draft],
        ),
        (
            "decoder.mlp.down_proj.weight",
            vec![g.h_draft, g.intermediate],
        ),
    ];
    println!("\n→ Spot-checking critical tensor shapes:");
    let mut missing = Vec::new();
    let mut mismatched = Vec::new();
    for (name, expected) in expectations {
        match weights.get(name) {
            Some(t) => {
                if &t.shape != expected {
                    mismatched.push(format!(
                        "   ✗ {} shape={:?} expected={:?}",
                        name, t.shape, expected
                    ));
                } else {
                    println!("   ✓ {:50}  shape={:?}", name, t.shape);
                }
            }
            None => missing.push(format!("   ✗ MISSING: {}", name)),
        }
    }
    for m in &missing {
        println!("{m}");
    }
    for m in &mismatched {
        println!("{m}");
    }
    if !missing.is_empty() || !mismatched.is_empty() {
        anyhow::bail!(
            "{} missing, {} mismatched tensor(s)",
            missing.len(),
            mismatched.len()
        );
    }
    println!(
        "\n   d2t len={}  offsets[0..8]={:?}",
        weights.d2t().len(),
        &weights.d2t()[..weights.d2t().len().min(8)],
    );

    // Finite-value check on every loaded tensor.
    println!("\n→ Checking that every f32 weight is finite...");
    let mut nonfinite_count = 0usize;
    let mut nonfinite_first: Option<(String, usize, f32)> = None;
    for name in weights.tensor_names() {
        let t = weights.get(name).unwrap();
        for (i, &v) in t.data.iter().enumerate() {
            if !v.is_finite() {
                nonfinite_count += 1;
                if nonfinite_first.is_none() {
                    nonfinite_first = Some((name.to_string(), i, v));
                }
                break; // count once per tensor
            }
        }
    }
    if nonfinite_count > 0 {
        let (n, i, v) = nonfinite_first.unwrap();
        eprintln!(
            "   ✗ {nonfinite_count} tensor(s) contain non-finite values; first: {n}[{i}] = {v}"
        );
        anyhow::bail!("non-finite weights");
    }
    println!("   ✓ all {} f32 tensors are finite", weights.len());

    // Drive the full propose pipeline.
    println!(
        "\n→ Running Eagle3Speculator::propose with n={} on synthetic verifier hidden states",
        cfg.speculative_tokens
    );
    let n_aux = aux_ids.len();
    let h = SynthHidden {
        target_hidden: g.h_target,
        layers: n_aux,
    };
    let n = cfg.speculative_tokens;
    let mut spec = Eagle3Speculator::new(cfg, weights, h)?;
    let context: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let t0 = Instant::now();
    let proposal = spec.propose(&context, n);
    let secs = t0.elapsed().as_secs_f32();
    println!(
        "   {} tokens proposed in {:.2}s ({:.1} tok/s in pure-Rust reference)",
        proposal.tokens.len(),
        secs,
        proposal.tokens.len() as f32 / secs.max(1e-6),
    );
    println!("   tokens={:?}", proposal.tokens);
    println!(
        "   probs row[0] len={} sum={:.4}",
        proposal.probs[0].len(),
        proposal.probs[0].iter().sum::<f32>(),
    );

    println!("\n✓ DONE. RedHatAI/gemma-4-31B-it-speculator.eagle3 loads cleanly into rlx-eagle3.");
    println!(
        "  Real-weight parity vs llama.cpp b9606 still requires the runner glue\n  \
         (rlx-gemma → VerifierHiddenSource); see crates/rlx-eagle3/PLAN.md."
    );
    Ok(())
}
