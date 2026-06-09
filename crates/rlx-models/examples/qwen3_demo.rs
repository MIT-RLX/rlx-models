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

//! End-to-end Qwen3 demo — load weights, prefill a prompt, generate
//! tokens via the cached KV-cache path, time both naive vs cached.
//!
//! ## Usage
//!
//! With a real checkpoint:
//!
//! ```bash
//! cargo run -p rlx-models --release --example qwen3_demo -- \
//!     --config /path/to/qwen3/config.json \
//!     --weights /path/to/qwen3/model.safetensors \
//!     --prompt-ids 1,2,3,4 \
//!     --new-tokens 16
//! ```
//!
//! Without any arguments, runs a synthetic-weights basic test using
//! the same tiny config as the unit tests. This is what `cargo run
//! --example qwen3_demo` does — a self-check that the public API
//! works.
//!
//! GGUF works the same way: `--weights model.gguf` (extension
//! auto-detected via `rlx_models::load_from_path`).

use anyhow::{Context, Result, anyhow};
use rlx_models::qwen3::{Qwen3Config, Qwen3Generator, SampleOpts};
use rlx_runtime::Device;
use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::time::Instant;

struct Args {
    config: Option<String>,
    weights: Option<String>,
    prompt_ids: Vec<u32>,
    new_tokens: usize,
}

fn parse_args() -> Result<Args> {
    let mut config = None;
    let mut weights = None;
    let mut prompt_ids = vec![1u32, 2, 3, 4];
    let mut new_tokens = 8usize;
    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--config" => config = Some(it.next().ok_or_else(|| anyhow!("--config needs value"))?),
            "--weights" => {
                weights = Some(it.next().ok_or_else(|| anyhow!("--weights needs value"))?)
            }
            "--prompt-ids" => {
                let v = it
                    .next()
                    .ok_or_else(|| anyhow!("--prompt-ids needs value"))?;
                prompt_ids = v
                    .split(',')
                    .map(|s| s.trim().parse::<u32>().context("bad token id"))
                    .collect::<Result<_>>()?;
            }
            "--new-tokens" => {
                new_tokens = it
                    .next()
                    .ok_or_else(|| anyhow!("--new-tokens needs value"))?
                    .parse()?;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }
    Ok(Args {
        config,
        weights,
        prompt_ids,
        new_tokens,
    })
}

fn print_help() {
    eprintln!("qwen3_demo — run a Qwen3 generation loop on CPU");
    eprintln!();
    eprintln!("Options:");
    eprintln!(
        "  --config <path>       HF Qwen3 config.json (optional; uses tiny synthetic config if absent)"
    );
    eprintln!(
        "  --weights <path>      .safetensors or .gguf weights (optional; uses synthetic weights if absent)"
    );
    eprintln!("  --prompt-ids 1,2,3,4  Comma-separated input token ids (default: 1,2,3,4)");
    eprintln!("  --new-tokens 8        Number of tokens to generate (default: 8)");
}

fn main() -> Result<()> {
    let args = parse_args()?;

    let (cfg, mode) = match args.config {
        Some(path) => {
            let c = Qwen3Config::from_file(Path::new(&path))?;
            (c, "real-config")
        }
        None => (tiny_synthetic_cfg(), "synthetic-config"),
    };

    eprintln!("[qwen3_demo] {mode}");
    eprintln!(
        "  hidden={} layers={} q_heads={} kv_heads={} head_dim={} vocab={} rope_theta={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.vocab_size,
        cfg.rope_theta,
    );

    let mut gn = match args.weights {
        Some(path) => Qwen3Generator::from_path(cfg.clone(), &path, Device::Cpu)?,
        None => {
            let mut wm = synthetic_weights(&cfg);
            Qwen3Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu)?
        }
    };

    // Validate prompt ids fit in the vocab.
    for &t in &args.prompt_ids {
        if (t as usize) >= cfg.vocab_size {
            anyhow::bail!("prompt id {t} >= vocab_size {}", cfg.vocab_size);
        }
    }
    eprintln!(
        "  prompt = {:?} ({} tokens) → generating {} new tokens, greedy",
        args.prompt_ids,
        args.prompt_ids.len(),
        args.new_tokens
    );

    // ── Cached path ───────────────────────────────────────────────
    gn.prefill(&args.prompt_ids);
    let t0 = Instant::now();
    let cached_new = gn.generate_cached(args.new_tokens, SampleOpts::greedy())?;
    let cached_ms = t0.elapsed().as_secs_f64() * 1e3;

    eprintln!(
        "[cached] {} tokens in {:.1} ms ({:.1} tok/s) → {:?}",
        cached_new.len(),
        cached_ms,
        cached_new.len() as f64 / (cached_ms / 1000.0),
        cached_new,
    );

    // ── Naive path (recompute-each-step) for comparison ───────────
    gn.prefill(&args.prompt_ids);
    let t0 = Instant::now();
    let naive_new = gn.generate(args.new_tokens, SampleOpts::greedy())?;
    let naive_ms = t0.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "[naive ] {} tokens in {:.1} ms ({:.1} tok/s) → {:?}",
        naive_new.len(),
        naive_ms,
        naive_new.len() as f64 / (naive_ms / 1000.0),
        naive_new,
    );

    if cached_new != naive_new {
        anyhow::bail!(
            "cached vs naive sequences diverged: {:?} vs {:?}",
            cached_new,
            naive_new
        );
    }
    eprintln!(
        "[ok    ] cached == naive ({}× speedup)",
        if cached_ms > 0.0 {
            naive_ms / cached_ms
        } else {
            0.0
        }
    );

    Ok(())
}

fn tiny_synthetic_cfg() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 16,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        max_position_embeddings: 32,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
        attention_bias: false,
        qk_norm: true,
        sliding_window: None,
        max_window_layers: usize::MAX,
        use_sliding_window: false,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    }
}

/// Same deterministic pattern as the unit tests so synthetic-mode
/// output is reproducible across runs.
fn synthetic_weights(cfg: &Qwen3Config) -> rlx_models::weight_map::WeightMap {
    let h = cfg.hidden_size;
    let q_dim = cfg.q_proj_dim();
    let kv_dim = cfg.kv_proj_dim();
    let int_dim = cfg.intermediate_size;
    let dh = cfg.head_dim;
    let pat = |n: usize, salt: u32| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(salt)) >> 8;
                (x as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    };
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert(
        "model.embed_tokens.weight".into(),
        (pat(cfg.vocab_size * h, 1), vec![cfg.vocab_size, h]),
    );
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        t.insert(
            format!("{lp}.input_layernorm.weight"),
            (pat(h, 100 + i as u32), vec![h]),
        );
        t.insert(
            format!("{lp}.post_attention_layernorm.weight"),
            (pat(h, 200 + i as u32), vec![h]),
        );
        t.insert(
            format!("{lp}.self_attn.q_proj.weight"),
            (pat(q_dim * h, 300 + i as u32), vec![q_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (pat(kv_dim * h, 400 + i as u32), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (pat(kv_dim * h, 500 + i as u32), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.o_proj.weight"),
            (pat(h * q_dim, 600 + i as u32), vec![h, q_dim]),
        );
        t.insert(
            format!("{lp}.self_attn.q_norm.weight"),
            (pat(dh, 700 + i as u32), vec![dh]),
        );
        t.insert(
            format!("{lp}.self_attn.k_norm.weight"),
            (pat(dh, 800 + i as u32), vec![dh]),
        );
        t.insert(
            format!("{lp}.mlp.gate_proj.weight"),
            (pat(int_dim * h, 900 + i as u32), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.up_proj.weight"),
            (pat(int_dim * h, 1000 + i as u32), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.down_proj.weight"),
            (pat(h * int_dim, 1100 + i as u32), vec![h, int_dim]),
        );
    }
    t.insert("model.norm.weight".into(), (pat(h, 2000), vec![h]));
    t.insert(
        "lm_head.weight".into(),
        (pat(cfg.vocab_size * h, 3000), vec![cfg.vocab_size, h]),
    );
    rlx_models::weight_map::WeightMap::from_tensors(t)
}
