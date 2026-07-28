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
use rlx_cli::{WeightsResolveCli, resolve_weights_cli};
use rlx_text::chat::ChatMessage;
use std::path::{Path, PathBuf};

fn flag_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

/// Cap Rayon width for Laguna MoE unless the user already set `RLX_WORKERS`.
///
/// Expert-parallel decode + attention-head parallel oversubscribe easily on
/// full logical CPU counts; 4–8 workers measured fastest on Apple Silicon.
fn ensure_moe_worker_default() {
    if std::env::var_os("RLX_WORKERS").is_some() {
        return;
    }
    let hint = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let w = (hint / 2).clamp(4, 8);
    // SAFETY: CLI entry before any rlx-cpu Rayon pool init.
    unsafe { std::env::set_var("RLX_WORKERS", w.to_string()) };
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

/// Resolve `--weights` (file or dir) with optional `--prefer` / `--gguf-index`.
/// Unsloth Laguna-S trees nest quants (`UD-Q4_K_M/*-00001-of-*.gguf`); prefer
/// substring picks the first split shard under one child dir.
fn resolve_weights_arg(args: &[String], path: &str) -> Result<PathBuf> {
    let mut resolve = WeightsResolveCli::default();
    if let Some(pref) = flag_value(args, "--prefer")
        .or_else(|| flag_value(args, "--prefer-quant"))
        .or_else(|| flag_value(args, "-p"))
    {
        resolve.prefer_gguf = Some(pref);
    }
    if let Some(idx) = flag_value(args, "--gguf-index") {
        resolve.gguf_index = Some(idx.parse().context("--gguf-index")?);
    }
    resolve_weights_cli(Path::new(path), &resolve)
}

#[allow(dead_code)] // used by OpenAI serve / future HTTP flags
fn parse_u16(args: &[String], name: &str, default: u16) -> Result<u16> {
    Ok(flag_value(args, name)
        .map(|s| s.parse::<u16>().with_context(|| name.to_string()))
        .transpose()?
        .unwrap_or(default))
}

fn resolve_prompt_ids(
    args: &[String],
    chat: Option<&LagunaChat>,
    default: &[u32],
) -> Result<Vec<u32>> {
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
    ensure_moe_worker_default();
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
        // mlx-community directory (HF config.json + affine safetensors) → native
        // affine load + KV-cached generate (no GGUF). Detected before GGUF resolve.
        // Accept either the dir itself or a `.safetensors` shard inside it (the
        // auto-dispatch / resolver may hand us the single-file form).
        let raw = Path::new(&path);
        let mlx_dir: Option<PathBuf> = if raw.is_dir() && raw.join("config.json").exists() {
            Some(raw.to_path_buf())
        } else if raw.is_file()
            && raw.extension().and_then(|e| e.to_str()) == Some("safetensors")
            && raw.parent().is_some_and(|p| p.join("config.json").exists())
        {
            raw.parent().map(Path::to_path_buf)
        } else {
            None
        };
        if let Some(dir) = mlx_dir {
            return run_mlx_dir(dir, args);
        }
        let path = resolve_weights_arg(args, &path)?;
        if has_flag(args, "--packed-load") {
            return packed_load_gguf(path, args);
        }
        return sniff_gguf(path);
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
        "rlx-laguna: pass --synth, --weights GGUF [--packed-load] | MLX_DIR, --serve, --config FILE, or --list-hf-keys (see --help)"
    );
}

/// Load an **mlx-community** Laguna directory (HF `config.json` + affine
/// safetensors) natively — no GGUF — and KV-decode. Mirrors the GGUF
/// [`packed_load_gguf`] generate loop but sources packed weights from
/// [`LagunaPackedRunner::from_mlx_dir`] (mlx-affine dequant) and reuses the
/// same tokenizer/chat-template plumbing. This is the reference wiring for
/// dispatching a dedicated-crate model straight from an mlx-community dir.
fn run_mlx_dir(dir: PathBuf, args: &[String]) -> Result<()> {
    println!("{FAMILY} mlx-community affine load: {}", dir.display());
    let t0 = std::time::Instant::now();
    let runner = LagunaPackedRunner::from_mlx_dir(&dir)?;
    println!("  loaded in {:.1?}", t0.elapsed());
    print_config_summary(runner.config());

    let max_tokens = parse_max_tokens(args)?.unwrap_or(0);
    if max_tokens == 0 {
        println!("  (pass --max-tokens N to run mlx-affine greedy generate)");
        return Ok(());
    }

    // Tokenizer/chat template: --tokenizer-dir, else the mlx dir itself
    // (mlx-community ships tokenizer.json alongside config.json).
    let chat = load_chat(args)?.or_else(|| LagunaChat::from_dir(&dir).ok());
    let bos = runner.config().bos_token_id.max(1);
    let ids = resolve_prompt_ids(args, chat.as_ref(), &[bos])?;

    let t1 = std::time::Instant::now();
    let mut new_toks = Vec::new();
    let out = runner.generate(&ids, max_tokens, |t| {
        if let Some(ref c) = chat {
            print!("{}", c.decode_token(t));
        } else {
            print!("{t} ");
        }
        let _ = std::io::Write::flush(&mut std::io::stdout());
        new_toks.push(t);
    })?;
    let elapsed = t1.elapsed();
    println!();
    if let Some(ref c) = chat {
        println!(
            "  decoded: {}",
            c.decode(&new_toks, true).unwrap_or_default()
        );
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
    Ok(())
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
        let weights = resolve_weights_arg(args, &weights)?;
        let tok_dir = flag_value(args, "--tokenizer-dir").ok_or_else(|| {
            anyhow::anyhow!("--serve requires --tokenizer-dir DIR (tokenizer.json + template)")
        })?;
        let host = flag_value(args, "--host").unwrap_or_else(|| "127.0.0.1".into());
        let port = parse_u16(args, "--port", 8080)?;
        let model_id = flag_value(args, "--model-id").unwrap_or_else(|| "laguna".into());
        let default_max = parse_max_tokens(args)?.unwrap_or(256);
        let device_s = flag_value(args, "--device").unwrap_or_else(|| "cpu".into());

        let runner = LagunaPackedRunner::from_gguf_packed(&weights)?;
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
    println!(
        "  F32 expand: {} (default off; --allow-f32-expand / RLX_LAGUNA_ALLOW_F32_EXPAND=1)",
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
        let force_device = has_flag(args, "--force-device")
            || matches!(
                std::env::var("RLX_LAGUNA_FORCE_DEVICE").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes")
            );
        if has_flag(args, "--batched-moe") {
            unsafe { std::env::set_var("RLX_LAGUNA_BATCHED_MOE", "1") };
        }
        if has_flag(args, "--device-moe") {
            unsafe { std::env::set_var("RLX_LAGUNA_DEVICE_MOE", "1") };
        }
        if has_flag(args, "--no-device-moe") {
            unsafe { std::env::set_var("RLX_LAGUNA_DEVICE_MOE_DISABLE", "1") };
        }
        let batched_host = matches!(
            std::env::var("RLX_LAGUNA_BATCHED_MOE").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes")
        );
        let device_moe = matches!(
            std::env::var("RLX_LAGUNA_DEVICE_MOE").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes")
        ) && !matches!(
            std::env::var("RLX_LAGUNA_DEVICE_MOE_DISABLE").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes")
        );
        let mut accel = match crate::device_matmul::parse_device(&device_s)? {
            Some(d) if force_device || ids.len() >= crate::packed_forward::DEVICE_MATMUL_MIN_M => {
                let a = crate::device_matmul::DeviceMatmul::try_new(d)?;
                let moe = if device_moe {
                    "grouped-moe-resident"
                } else {
                    "attn/shared-only"
                };
                println!(
                    "  packed generate device={d:?} exec={:?} moe={moe} prompt_len={} max_tokens={max_tokens}{}",
                    a.exec(),
                    ids.len(),
                    if force_device { " (force-device)" } else { "" }
                );
                Some(a)
            }
            Some(d) => {
                // Opening Metal/MLX still costs (GPU power + sync) even when
                // short-seq MoE never launches DequantMatMul — stay on host.
                println!(
                    "  packed generate device=HostKernel ({d:?} skipped: prompt_len={} < min_m={}; MoE chat/decode is faster on host; pass --force-device to override) max_tokens={max_tokens}",
                    ids.len(),
                    crate::packed_forward::DEVICE_MATMUL_MIN_M,
                );
                None
            }
            None => {
                let moe = if batched_host {
                    "batched-host"
                } else {
                    "per-expert-int8"
                };
                println!(
                    "  packed generate device=HostKernel moe={moe} prompt_len={} max_tokens={max_tokens}",
                    ids.len()
                );
                None
            }
        };
        let rss_g0 = process_rss_bytes();
        let t0 = std::time::Instant::now();
        let mut new_toks = Vec::new();
        let out = runner.generate_with_device(&ids, max_tokens, accel.as_mut(), &mut |t| {
            if let Some(ref c) = chat {
                let piece = c.decode_token(t);
                print!("{piece}");
            } else {
                print!("{t} ");
            }
            let _ = std::io::Write::flush(&mut std::io::stdout());
            new_toks.push(t);
        })?;
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

    println!(
        "{FAMILY} GGUF header ok (packed-only, no tensor payload): {}",
        path.display()
    );
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
        println!(
            "  F32 expand: {} (default off; --allow-f32-expand / RLX_LAGUNA_ALLOW_F32_EXPAND=1)",
            if crate::memory::allow_f32_expand() {
                "ENABLED"
            } else {
                "disabled"
            }
        );
    } else {
        println!("  tensors=0 in this shard (split meta-only?) — siblings hold weight payloads");
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
  rlx-laguna --weights PATH.gguf|DIR                 # header-only sniff
  rlx-laguna --weights MLX_DIR [--tokenizer-dir DIR] --prompt TEXT --max-tokens N
             # mlx-community affine dir (config.json + safetensors) → native load + KV-decode
  rlx-laguna --weights PATH.gguf|DIR --packed-load [--device metal|mlx|gpu|wgpu|coreml|cpu|auto]
             [--prefer Q4_K_M] [--gguf-index N] [--force-device]
             [--batched-moe] [--device-moe] [--no-device-moe]
             [--prompt-ids … | --tokenizer-dir DIR --prompt TEXT [--system …]]
             [--max-tokens N]   # KV-cached packed generate
  rlx-laguna --allow-f32-expand …   # opt in to quant→F32 (off by default; see sniff F32-expand≈)
  rlx-laguna --serve --weights PATH.gguf|DIR --tokenizer-dir DIR
             [--prefer Q4_K_M] [--device metal|mlx|gpu|wgpu|coreml|cpu|auto]
             [--host 127.0.0.1] [--port 8080] [--model-id laguna] [--max-tokens 256]
             # needs --features serve; prefer: rlx-openai --engine laguna …
  rlx-laguna --list-hf-keys [--config …]
  rlx-laguna --probe-gguf-remote [--quant UD-Q4_K_XL]   # needs --features hf-probe

Weights dirs: Unsloth nests quants (e.g. UD-Q4_K_M/*-00001-of-00003.gguf).
  `--prefer Q4_K_M` (alias --prefer-quant) picks the first split shard under a
  matching child dir; default prefer is Q4_K_M when multiple .gguf match.
Device: short MoE chat/decode stays on host fused kernels (faster); pass
  `--force-device` (or RLX_LAGUNA_FORCE_DEVICE=1) to always open Metal/MLX/GPU/CoreML
  for attn/shared mats. Opt in to resident MoE with `--device-moe` /
  RLX_LAGUNA_DEVICE_MOE=1 (expert stacks stay on GPU after first upload per layer).
  Host batched MoE: `--batched-moe` / RLX_LAGUNA_BATCHED_MOE=1.
Metal/MLX/wgpu/CoreML: build with --features all-backends (or apple-silicon).
F32 expand: off by default; RLX_LAGUNA_ALLOW_F32_EXPAND=1 or --allow-f32-expand.
OpenAI: prefer `rlx-openai` (just openai-serve).

HF:    {HF_MODEL_ID}
GGUF:  {HF_GGUF_REPO}
"
    );
}
