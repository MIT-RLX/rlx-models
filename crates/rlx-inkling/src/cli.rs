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

//! `rlx-inkling` CLI — RLX eager text path + HF / GGUF inspect.

use crate::chat::{self, ReasoningEffort};
use crate::config::{FAMILY, HF_GGUF_REPO, HF_MODEL_ID, InklingConfig, InklingTextConfig};
use crate::fixture::{self, load_meta, load_text_weights};
#[cfg(feature = "hf-probe")]
use crate::gguf_probe::{self, DEFAULT_QUANT};
use crate::probe;
use crate::runner::InklingRunner;
use crate::synth::{synthetic_text_weights, tiny_cfg};
use crate::weights::expected_text_hf_keys;
use anyhow::{Context, Result, bail};
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

pub fn run(args: &[String]) -> Result<()> {
    if has_flag(args, "--help") || has_flag(args, "-h") {
        print_help();
        return Ok(());
    }

    if has_flag(args, "--synth") {
        return run_eager_synth(args);
    }

    if has_flag(args, "--fixture") {
        return run_eager_fixture(args);
    }

    if let Some(path) = flag_value(args, "--weights") {
        return sniff_gguf(PathBuf::from(path));
    }

    if has_flag(args, "--probe-gguf-remote") {
        return run_probe_gguf_remote(args);
    }

    if has_flag(args, "--probe-remote") {
        return run_probe_remote();
    }

    if let Some(dir) = flag_value(args, "--model-dir") {
        let probe = has_flag(args, "--probe");
        return inspect_model_dir(PathBuf::from(dir), probe);
    }

    if let Some(path) = flag_value(args, "--config") {
        let cfg = InklingConfig::from_json_path(path)?;
        print_config_summary(&cfg);
        return Ok(());
    }

    if has_flag(args, "--chat-demo") {
        let effort = flag_value(args, "--effort")
            .as_deref()
            .map(ReasoningEffort::parse)
            .transpose()?
            .unwrap_or(ReasoningEffort::High);
        let user = flag_value(args, "--prompt").unwrap_or_else(|| "Hello!".into());
        println!("{}", chat::format_user_turn(&user, effort));
        return Ok(());
    }

    bail!(
        "rlx-inkling: pass --synth, --fixture, --weights GGUF, --model-dir DIR, \
         --config FILE, or --chat-demo (see --help)"
    );
}

fn run_eager(runner: &InklingRunner, ids: &[u32], max_tokens: Option<usize>) -> Result<()> {
    let cfg = runner.config();
    let logits = runner.predict_logits(ids)?;
    let next = crate::eager::greedy_next(cfg, &runner.weights, ids)?;
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("{FAMILY} RLX eager text forward ok");
    println!(
        "  layers={} hidden={} experts={}/{} top_k={} seq={}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.n_routed_experts,
        cfg.n_shared_experts,
        cfg.num_experts_per_tok,
        ids.len()
    );
    println!("  logits: len={} min={min:.4} max={max:.4}", logits.len());
    println!("  greedy_next={next}");
    if let Some(n) = max_tokens {
        let mut printed = Vec::new();
        let out = runner.generate(ids, n, |tok| printed.push(tok))?;
        println!("  generate(+{n}): {:?}", printed);
        println!("  full_ids: {out:?}");
    }
    Ok(())
}

fn run_eager_synth(args: &[String]) -> Result<()> {
    let cfg = tiny_cfg();
    let w = synthetic_text_weights(&cfg);
    let ids = parse_prompt_ids(args, &[1, 2, 3, 4])?;
    let max_tokens = parse_max_tokens(args)?;
    let runner = InklingRunner::new(cfg, w);
    run_eager(&runner, &ids, max_tokens)
}

fn run_eager_fixture(args: &[String]) -> Result<()> {
    let dir = flag_value(args, "--fixture-dir")
        .map(PathBuf::from)
        .unwrap_or_else(fixture::fixture_dir);
    let cfg = InklingConfig::from_json_path(dir.join("config.json"))?.text;
    let weights = load_text_weights(&dir)?;
    let meta = load_meta(&dir)?;
    let ids = parse_prompt_ids(args, &meta.input_ids)?;
    let max_tokens = parse_max_tokens(args)?;
    let runner = InklingRunner::new(cfg, weights);
    run_eager(&runner, &ids, max_tokens)
}

fn sniff_gguf(path: PathBuf) -> Result<()> {
    let cfg = InklingTextConfig::from_gguf_path(&path)?;
    println!("{FAMILY} GGUF metadata ({})", path.display());
    println!("  arch=inkling  Hub quant repo={HF_GGUF_REPO}");
    println!(
        "  layers={} hidden={} vocab={} heads={}/{} hd={}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.vocab_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim
    );
    println!(
        "  swa_kv={} window={} d_rel={} rel_extent={} sconv_k={}",
        cfg.swa_num_key_value_heads,
        cfg.sliding_window_size,
        cfg.d_rel,
        cfg.rel_extent,
        cfg.conv_kernel_size
    );
    println!(
        "  moe: routed={} shared={} top_k={} dense_layers={} moe_ff={} dense_ff={}",
        cfg.n_routed_experts,
        cfg.n_shared_experts,
        cfg.num_experts_per_tok,
        cfg.dense_mlp_idx,
        cfg.moe_intermediate_size,
        cfg.dense_intermediate_size
    );
    println!(
        "  ctx={} eos={} (RLX eager generate needs dequant loader — sniff only)",
        cfg.max_position_embeddings, cfg.eos_token_id
    );
    Ok(())
}

fn inspect_model_dir(dir: PathBuf, probe: bool) -> Result<()> {
    let cfg = InklingConfig::from_model_dir(&dir)?;
    print_config_summary(&cfg);
    let keys = expected_text_hf_keys(&cfg.text);
    println!(
        "expected text HF keys (no vision/audio/MTP): {}",
        keys.len()
    );
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        println!("index: {}", index.display());
    } else {
        println!(
            "note: no model.safetensors.index.json under {}",
            dir.display()
        );
    }
    let tok = dir.join("tokenizer.json");
    if tok.is_file() {
        println!("tokenizer: {}", tok.display());
    }
    if probe {
        let report = probe::validate_model_dir(&dir)?;
        report.print();
        report.assert_ok()?;
    } else {
        println!("tip: add --probe to shape-check safetensors headers (no full shard slurp)");
    }
    println!("HF hub: {HF_MODEL_ID}");
    Ok(())
}

fn run_probe_remote() -> Result<()> {
    #[cfg(feature = "hf-probe")]
    {
        let report = probe::probe_remote(None)?;
        report.print();
        report.assert_ok()?;
        return Ok(());
    }
    #[cfg(not(feature = "hf-probe"))]
    {
        bail!(
            "rlx-inkling: --probe-remote needs `--features hf-probe` \
             (downloads config+index only, then HTTP Range-reads shard headers)"
        );
    }
}

fn run_probe_gguf_remote(args: &[String]) -> Result<()> {
    #[cfg(feature = "hf-probe")]
    {
        let quant = flag_value(args, "--quant").unwrap_or_else(|| DEFAULT_QUANT.into());
        let repo = flag_value(args, "--repo");
        let report = gguf_probe::probe_remote_gguf(repo.as_deref(), Some(quant.as_str()), None)?;
        report.print();
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/unsloth_gguf_sniff.json");
        report.write_compact_fixture(&fixture)?;
        println!("  wrote compact fixture {}", fixture.display());
        if let Some(full) = flag_value(args, "--write-json") {
            report.write_json(&full)?;
            println!("  wrote full tensor sniff {full}");
        }
        return Ok(());
    }
    #[cfg(not(feature = "hf-probe"))]
    {
        let _ = args;
        bail!(
            "rlx-inkling: --probe-gguf-remote needs `--features hf-probe` \
             (Hub API + HTTP Range on GGUF prefixes only — no weight payload)"
        );
    }
}

fn print_config_summary(cfg: &InklingConfig) {
    let t = &cfg.text;
    println!("{FAMILY} ({})", cfg.model_type);
    println!(
        "  text: layers={} hidden={} vocab={} (unpadded={:?})",
        t.num_hidden_layers, t.hidden_size, t.vocab_size, t.unpadded_vocab_size
    );
    println!(
        "  attn: heads={}/{} hd={} swa_kv={} window={} d_rel={} rel_extent={}",
        t.num_attention_heads,
        t.num_key_value_heads,
        t.head_dim,
        t.swa_num_key_value_heads,
        t.sliding_window_size,
        t.d_rel,
        t.rel_extent
    );
    println!(
        "  moe: routed={} shared={} top_k={} dense_layers={} moe_ff={} dense_ff={}",
        t.n_routed_experts,
        t.n_shared_experts,
        t.num_experts_per_tok,
        t.dense_mlp_idx,
        t.moe_intermediate_size,
        t.dense_intermediate_size
    );
    println!(
        "  ctx={} sconv_k={} mtp={} eos={}",
        t.max_position_embeddings, t.conv_kernel_size, t.num_mtp_layers, t.eos_token_id
    );
    println!(
        "  vision: {} patches {}×{} (T={}) layers={}",
        cfg.vision.vision_encoder_type,
        cfg.vision.patch_size,
        cfg.vision.patch_size,
        cfg.vision.temporal_patch_size,
        cfg.vision.n_layers
    );
    println!(
        "  audio: mode={} mel_bins={} codebook={}",
        cfg.audio.audio_mode, cfg.audio.n_mel_bins, cfg.audio.mel_vocab_size
    );
}

fn print_help() {
    eprintln!(
        "\
rlx-inkling — {FAMILY} on RLX ({HF_MODEL_ID})

USAGE:
  rlx-inkling --synth [--prompt-ids 1,2,3] [--max-tokens N]
      Tiny RLX eager CPU text forward; generate only if --max-tokens is set
  rlx-inkling --fixture [--fixture-dir DIR] [--prompt-ids …] [--max-tokens N]
      Same path on the HF tiny parity fixture (tests/fixtures/hf_tiny_parity)
  rlx-inkling --weights PATH.gguf
      Sniff local GGUF metadata (header-only; no dequant yet)
  rlx-inkling --probe-gguf-remote [--quant UD-IQ1_S] [--write-json PATH]
      Hub Range-sniff Unsloth GGUF (meta + shard 00002 headers only; needs hf-probe)
  rlx-inkling --model-dir DIR [--probe]
  rlx-inkling --config FILE
  rlx-inkling --probe-remote          (needs --features hf-probe)
  rlx-inkling --chat-demo [--prompt TEXT] [--effort high]

Weights: BF16 Hub is ~1.9TB. Prefer {HF_GGUF_REPO} for RLX (not llama.cpp runtime)."
    );
}
