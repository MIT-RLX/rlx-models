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

//! `rlx-laguna` CLI — synth, GGUF sniff, packed mmap generate, optional `--serve`.

use crate::chat::LagunaChat;
use crate::config::{FAMILY, HF_GGUF_REPO, HF_MODEL_ID, LagunaConfig};
use crate::gguf_layout::{self, LAYOUT_NOTES};
use crate::packed::process_rss_bytes;
use crate::runner::{LagunaPackedRunner, LagunaRunner};
use crate::synth::{synthetic_text_weights, tiny_cfg};
use crate::weights::expected_hf_keys;
use anyhow::{Context, Result, bail};
use rlx_text::chat::ChatMessage;
use std::path::PathBuf;

fn flag_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn parse_prompt_ids(args: &[String], default: &[u32]) -> Result<Vec<u32>> {
    Ok(flag_value(args, "--prompt-ids")
        .map(|s| {
            s.split(',')
                .filter(|p| !p.is_empty())
                .map(|p| p.trim().parse::<u32>())
                .collect::<std::result::Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_else(|| default.to_vec()))
}

fn parse_max_tokens(args: &[String]) -> Result<Option<usize>> {
    flag_value(args, "--max-tokens")
        .map(|s| s.parse::<usize>().context("--max-tokens"))
        .transpose()
}

fn parse_u16(args: &[String], name: &str, default: u16) -> Result<u16> {
    Ok(flag_value(args, name)
        .map(|s| s.parse::<u16>().with_context(|| name.to_string()))
        .transpose()?
        .unwrap_or(default))
}

fn resolve_prompt_ids(args: &[String], chat: Option<&LagunaChat>, default: &[u32]) -> Result<Vec<u32>> {
    if let Some(text) = flag_value(args, "--prompt") {
        let chat = chat.ok_or_else(|| {
            anyhow::anyhow!("--prompt requires --tokenizer-dir DIR (with tokenizer.json)")
        })?;
        let mut msgs = Vec::new();
        if let Some(sys) = flag_value(args, "--system") {
            msgs.push(ChatMessage {
                role: "system".into(),
                content: sys,
            });
        }
        msgs.push(ChatMessage::user(text));
        return chat.encode_chat(&msgs, false);
    }
    parse_prompt_ids(args, default)
}

pub fn run(args: &[String]) -> Result<()> {
    if has_flag(args, "--help") || has_flag(args, "-h") {
        print_help();
        return Ok(());
    }

    if has_flag(args, "--allow-f32-expand") {
        // SAFETY: CLI startup before any Laguna load; mirrors env opt-in.
        unsafe { std::env::set_var("RLX_LAGUNA_ALLOW_F32_EXPAND", "1") };
        eprintln!(
            "rlx-laguna: F32 expand ENABLED (RLX_LAGUNA_ALLOW_F32_EXPAND=1) — \
             check `--weights` sniff for F32-expand≈ before full drain"
        );
    }

    if has_flag(args, "--serve") {
        return run_serve(args);
    }

    if has_flag(args, "--synth") {
        return run_eager_synth(args);
    }

    if let Some(path) = flag_value(args, "--weights") {
        if has_flag(args, "--packed-load") {
            return packed_load_gguf(PathBuf::from(path), args);
        }
        return sniff_gguf(PathBuf::from(path));
    }

    if has_flag(args, "--probe-gguf-remote") {
        return run_probe_gguf_remote(args);
    }

    if let Some(path) = flag_value(args, "--config") {
        let cfg = LagunaConfig::from_json_path(path)?;
        print_config_summary(&cfg);
        return Ok(());
    }

    if has_flag(args, "--list-hf-keys") {
        let cfg = if let Some(path) = flag_value(args, "--config") {
            LagunaConfig::from_json_path(path)?
        } else {
            LagunaConfig::production_s21()
        };
        for k in expected_hf_keys(&cfg) {
            println!("{k}");
        }
        return Ok(());
    }

    bail!(
        "rlx-laguna: pass --synth, --weights GGUF [--packed-load], --serve, --config FILE, or --list-hf-keys (see --help)"
    );
}

fn load_chat(args: &[String]) -> Result<Option<LagunaChat>> {
    let Some(dir) = flag_value(args, "--tokenizer-dir") else {
        return Ok(None);
    };
    Ok(Some(LagunaChat::from_dir(dir)?))
}

fn run_serve(args: &[String]) -> Result<()> {
    #[cfg(not(feature = "serve"))]
    {
        let _ = args;
        bail!("rebuild with --features serve for --serve (OpenAI HTTP)");
    }
    #[cfg(feature = "serve")]
    {
        let weights = flag_value(args, "--weights")
            .ok_or_else(|| anyhow::anyhow!("--serve requires --weights PATH.gguf"))?;
        let tok_dir = flag_value(args, "--tokenizer-dir").ok_or_else(|| {
            anyhow::anyhow!("--serve requires --tokenizer-dir DIR (tokenizer.json + template)")
        })?;
        let host = flag_value(args, "--host").unwrap_or_else(|| "127.0.0.1".into());
        let port = parse_u16(args, "--port", 8080)?;
        let model_id = flag_value(args, "--model-id").unwrap_or_else(|| "laguna".into());
        let default_max = parse_max_tokens(args)?.unwrap_or(256);
        let device_s = flag_value(args, "--device").unwrap_or_else(|| "cpu".into());

        let runner = LagunaPackedRunner::from_gguf_packed(weights)?;
        let chat = LagunaChat::from_dir(tok_dir)?;
        if chat.used_fallback_template {
            eprintln!("rlx-laguna: using in-crate chat template fallback");
        }
        let accel = match crate::device_matmul::parse_device(&device_s)? {
            Some(d) => Some(crate::device_matmul::DeviceMatmul::try_new(d)?),
            None => None,
        };
        let engine = crate::serve::LagunaEngine::new(runner, chat, accel, model_id);
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(crate::serve::serve(engine, &host, port, default_max))
    }
}

fn run_eager_synth(args: &[String]) -> Result<()> {
    let cfg = tiny_cfg();
    let w = synthetic_text_weights(&cfg);
    let runner = LagunaRunner::new(cfg.clone(), w);
    let ids = parse_prompt_ids(args, &[1, 2, 3])?;
    let max_tokens = parse_max_tokens(args)?.unwrap_or(0);
    let logits = runner.predict_logits(&ids)?;
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("{FAMILY} RLX eager text forward ok");
    println!(
        "  layers={} hidden={} experts={}/{} top_k={} seq={}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_experts,
        cfg.shared_expert_intermediate_size,
        cfg.num_experts_per_tok,
        ids.len()
    );
    println!("  logits[{}] min={min:.4} max={max:.4}", logits.len());
    if max_tokens > 0 {
        let mut new_toks = Vec::new();
        let out = runner.generate(&ids, max_tokens, |t| new_toks.push(t))?;
        println!("  generate -> {out:?} (new={new_toks:?})");
    }
    Ok(())
}

fn packed_load_gguf(path: PathBuf, args: &[String]) -> Result<()> {
    let rss_before = process_rss_bytes();
    let runner = LagunaPackedRunner::from_gguf_packed(&path)?;
    let rss_after = process_rss_bytes();
    let w = runner.weights();
    let cfg = runner.config();
    let chat = load_chat(args)?;

    println!(
        "{FAMILY} packed mmap load ok (no quant F32 expand): {}",
        path.display()
    );
    print_config_summary(cfg);
    println!();
    println!("{}", crate::memory::PACKED_ONLY_POLICY);
    println!(
        "  packed_tensors={}  native_f32_side≈{:.2} MB  packed_params={}",
        w.packed_tensor_count,
        w.estimate_resident_bytes() as f64 / (1024.0 * 1024.0),
        w.packed_params.len()
    );
    println!("  layers_loaded={}", w.layers.len());
    println!("  F32 expand: {} (default off; --allow-f32-expand / RLX_LAGUNA_ALLOW_F32_EXPAND=1)",
        if crate::memory::allow_f32_expand() {
            "ENABLED"
        } else {
            "disabled"
        }
    );
    if let (Some(b), Some(a)) = (rss_before, rss_after) {
        println!(
            "  process_rss: before≈{:.1} MB  after≈{:.1} MB  Δ≈{:.1} MB",
            b as f64 / (1024.0 * 1024.0),
            a as f64 / (1024.0 * 1024.0),
            a.saturating_sub(b) as f64 / (1024.0 * 1024.0)
        );
    }

    let max_tokens = parse_max_tokens(args)?.unwrap_or(0);
    if max_tokens > 0 {
        let ids = resolve_prompt_ids(args, chat.as_ref(), &[cfg.bos_token_id.max(1)])?;
        let device_s = flag_value(args, "--device").unwrap_or_else(|| "cpu".into());
        let mut accel = match crate::device_matmul::parse_device(&device_s)? {
            Some(d) => {
                let a = crate::device_matmul::DeviceMatmul::try_new(d)?;
                println!(
                    "  packed generate device={} exec={:?} prompt_len={} max_tokens={max_tokens}",
                    format!("{d:?}"),
                    a.exec(),
                    ids.len()
                );
                Some(a)
            }
            None => {
                println!(
                    "  packed generate device=HostKernel prompt_len={} max_tokens={max_tokens}",
                    ids.len()
                );
                None
            }
        };
        let rss_g0 = process_rss_bytes();
        let t0 = std::time::Instant::now();
        let mut new_toks = Vec::new();
        let out = runner.generate_with_device(
            &ids,
            max_tokens,
            accel.as_mut(),
            &mut |t| {
                if let Some(ref c) = chat {
                    let piece = c.decode_token(t);
                    print!("{piece}");
                } else {
                    print!("{t} ");
                }
                let _ = std::io::Write::flush(&mut std::io::stdout());
                new_toks.push(t);
            },
        )?;
        let elapsed = t0.elapsed();
        println!();
        let rss_g1 = process_rss_bytes();
        if chat.is_some() {
            if let Some(ref c) = chat {
                let decoded = c.decode(&new_toks, true).unwrap_or_default();
                println!("  generate decoded: {decoded}");
            }
        } else {
            println!("  generate -> {out:?} (new={new_toks:?})");
        }
        println!(
            "  wall_time={:.2}s  tokens_new={}  ms/token={:.1}",
            elapsed.as_secs_f64(),
            new_toks.len(),
            if new_toks.is_empty() {
                0.0
            } else {
                elapsed.as_secs_f64() * 1000.0 / new_toks.len() as f64
            }
        );
        if let (Some(b), Some(a)) = (rss_g0, rss_g1) {
            println!(
                "  process_rss after generate≈{:.1} MB (Δ generate≈{:.1} MB)",
                a as f64 / (1024.0 * 1024.0),
                a.saturating_sub(b) as f64 / (1024.0 * 1024.0)
            );
        }
    } else {
        println!("  (pass --max-tokens N to run packed greedy generate)");
    }
    Ok(())
}

fn sniff_gguf(path: PathBuf) -> Result<()> {
    // Packed-only: header metadata + tensor table. No Q4→F32 expand, no payload RSS.
    let raw = crate::memory::open_gguf_header_only(&path)?;
    let cfg = LagunaConfig::from_gguf(&raw)?;
    let est = crate::memory::estimate_ram(&raw);

    println!("{FAMILY} GGUF header ok (packed-only, no tensor payload): {}", path.display());
    print_config_summary(&cfg);
    println!();
    println!("{}", crate::memory::PACKED_ONLY_POLICY);
    if est.tensor_count > 0 {
        println!(
            "  tensors={}  packed≈{:.2} GB  F32-expand≈{:.2} GB  (×{:.1} if fully widened)",
            est.tensor_count,
            est.packed_gb(),
            est.f32_gb(),
            est.expand_ratio()
        );
        println!("  F32 expand: {} (default off; --allow-f32-expand / RLX_LAGUNA_ALLOW_F32_EXPAND=1)",
        if crate::memory::allow_f32_expand() {
            "ENABLED"
        } else {
            "disabled"
        }
    );
    } else {
        println!(
            "  tensors=0 in this shard (split meta-only?) — siblings hold weight payloads"
        );
    }
    println!();
    println!("{LAYOUT_NOTES}");
    let mut names: Vec<_> = raw.tensors.keys().cloned().collect();
    names.sort();
    println!("tensors in this shard: {}", names.len());
    for n in names.iter().take(12) {
        let mapped = gguf_layout::gguf_to_eager_key(n)
            .map(|k| format!(" -> {k}"))
            .unwrap_or_else(|| " (expert pack / other)".into());
        println!("  {n}{mapped}");
    }
    if names.len() > 12 {
        println!("  … {} more", names.len() - 12);
    }
    Ok(())
}

fn print_config_summary(cfg: &LagunaConfig) {
    println!(
        "{FAMILY} {}  layers={} hidden={} heads={}/{} head_dim={} experts={}/{} top_k={} swa={} gated={:?} ctx={}",
        cfg.variant().name(),
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.num_experts,
        1,
        cfg.num_experts_per_tok,
        cfg.sliding_window,
        cfg.gating,
        cfg.max_position_embeddings,
    );
    println!(
        "  dense_lead={} full_layers={} sliding_layers={} routed_scale={}",
        cfg.dense_lead_count(),
        cfg.layer_types
            .iter()
            .filter(|t| **t == crate::config::AttnLayerType::Full)
            .count(),
        cfg.layer_types
            .iter()
            .filter(|t| **t == crate::config::AttnLayerType::Sliding)
            .count(),
        cfg.moe_routed_scaling_factor,
    );
    println!(
        "  HF={}  GGUF={}",
        cfg.variant().hf_model_id(),
        cfg.variant().hf_gguf_repo()
    );
}

fn run_probe_gguf_remote(args: &[String]) -> Result<()> {
    #[cfg(not(feature = "hf-probe"))]
    {
        let _ = args;
        bail!("rebuild with --features hf-probe for --probe-gguf-remote");
    }
    #[cfg(feature = "hf-probe")]
    {
        let quant = flag_value(args, "--quant").unwrap_or_else(|| "UD-Q4_K_XL".into());
        let report = crate::gguf_probe::probe_remote(&quant)?;
        println!("{report}");
        Ok(())
    }
}

fn print_help() {
    println!(
        "rlx-laguna — Poolside Laguna MoE on RLX

Usage:
  rlx-laguna --synth [--prompt-ids 1,2,3] [--max-tokens N]
  rlx-laguna --config PATH/config.json
  rlx-laguna --weights PATH.gguf                 # header-only sniff
  rlx-laguna --weights PATH.gguf --packed-load [--device metal|mlx|cpu|auto]
             [--prompt-ids … | --tokenizer-dir DIR --prompt TEXT [--system …]]
             [--max-tokens N]   # KV-cached packed generate
  rlx-laguna --allow-f32-expand …   # opt in to quant→F32 (off by default; see sniff F32-expand≈)
  rlx-laguna --serve --weights PATH.gguf --tokenizer-dir DIR
             [--device metal|mlx|cpu|auto] [--host 127.0.0.1] [--port 8080]
             [--model-id laguna] [--max-tokens 256]
             # needs --features serve; prefer: rlx-openai --engine laguna …
  rlx-laguna --list-hf-keys [--config …]
  rlx-laguna --probe-gguf-remote [--quant UD-Q4_K_XL]   # needs --features hf-probe

Metal/MLX: build with --features apple-silicon (or metal / mlx).
F32 expand: off by default; RLX_LAGUNA_ALLOW_F32_EXPAND=1 or --allow-f32-expand.
OpenAI: prefer `rlx-openai` (just openai-serve).

HF:    {HF_MODEL_ID}
GGUF:  {HF_GGUF_REPO}
"
    );
}
