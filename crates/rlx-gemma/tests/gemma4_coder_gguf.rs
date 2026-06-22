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

//! Gemma 4 12B Coder GGUF (Composer 2.5 × Fable 5 fine-tune).
//!
//! ```bash
//! just fetch-gemma4-12b-coder-gguf
//! RLX_GEMMA4_CODER_FIXTURE=.cache/gemma4-12b-coder \
//!   cargo test -p rlx-gemma --release --features apple-silicon \
//!   --test gemma4_coder_gguf -- --nocapture
//! ```

mod gemma4_bench_common;

use anyhow::Result;
use gemma4_bench_common::{bench_device_from_env, resolve_gemma4_config, resolve_gemma4_gguf};
use rlx_gemma::{
    GemmaArch, GemmaRunner, decode_ids_auto, encode_chat_prompt_auto, gemma_cfg_from_gguf,
};
use rlx_gguf::GgufFile;
use rlx_qwen3::SampleOpts;
use std::path::PathBuf;

const CODER_PROMPT: &str =
    "Write a Python function that returns True when a string is a palindrome.";

fn coder_fixture_dir() -> Option<PathBuf> {
    std::env::var_os("RLX_GEMMA4_CODER_FIXTURE")
        .or_else(|| std::env::var_os("RLX_GEMMA4_FIXTURE"))
        .map(PathBuf::from)
}

#[test]
fn gemma4_coder_gguf_config_has_256k_context() -> Result<()> {
    let Some(dir) = coder_fixture_dir() else {
        eprintln!("[gemma4 coder] RLX_GEMMA4_CODER_FIXTURE unset — skip");
        return Ok(());
    };
    let Some(gguf) = resolve_gemma4_gguf(&dir) else {
        eprintln!("[gemma4 coder] no GGUF in {dir:?} — skip");
        return Ok(());
    };
    let raw = GgufFile::from_path(&gguf)?;
    let cfg = gemma_cfg_from_gguf(&raw)?;
    assert_eq!(cfg.arch, GemmaArch::Gemma4);
    assert_eq!(cfg.hidden_size, 3840);
    assert_eq!(cfg.num_hidden_layers, 48);
    assert_eq!(
        cfg.max_position_embeddings, 262_144,
        "coder GGUF must expose full 256K context (re-patched metadata)"
    );
    assert_eq!(
        cfg.layer_n_rot(5),
        128,
        "full-attention layer must use p-RoPE n_rot=128"
    );
    assert_eq!(cfg.layer_n_rot(0), 256, "swa layer n_rot=256");
    eprintln!(
        "[gemma4 coder] cfg ok: layers={} ctx={} vocab={}",
        cfg.num_hidden_layers, cfg.max_position_embeddings, cfg.vocab_size
    );
    Ok(())
}

#[test]
fn gemma4_coder_chat_template_opens_thought_channel() -> Result<()> {
    let Some(dir) = coder_fixture_dir() else {
        eprintln!("[gemma4 coder] RLX_GEMMA4_CODER_FIXTURE unset — skip");
        return Ok(());
    };
    let Some(gguf) = resolve_gemma4_gguf(&dir) else {
        eprintln!("[gemma4 coder] no GGUF — skip");
        return Ok(());
    };
    let ids = encode_chat_prompt_auto(&gguf, None, None, CODER_PROMPT, true)?;
    assert!(
        ids.len() >= 8,
        "chat template should produce a non-trivial prompt (got {} ids)",
        ids.len()
    );
    let rendered = decode_ids_auto(&gguf, None, &ids, false)?;
    assert!(
        rendered.contains("thought") || rendered.contains("<|channel>"),
        "expected Gemma 4 thinking channel in rendered prompt: {rendered:?}"
    );
    eprintln!("[gemma4 coder] chat prompt: {} ids", ids.len());
    Ok(())
}

#[test]
fn gemma4_coder_packed_runner_generates() -> Result<()> {
    let Some(dir) = coder_fixture_dir() else {
        eprintln!("[gemma4 coder generate] fixture unset — skip");
        return Ok(());
    };
    let Some(gguf) = resolve_gemma4_gguf(&dir) else {
        eprintln!("[gemma4 coder generate] no GGUF — skip");
        return Ok(());
    };
    let _cfg = resolve_gemma4_config(&dir, &gguf)?;

    let device = bench_device_from_env();
    let n_new = std::env::var("RLX_GEMMA4_CODER_DECODE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let max_seq = std::env::var("RLX_GEMMA4_CODER_MAX_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    let ids = encode_chat_prompt_auto(&gguf, None, None, CODER_PROMPT, true)?;
    let mut runner = GemmaRunner::builder()
        .weights(&gguf)
        .device(device)
        .max_seq(max_seq)
        .packed_weights(true)
        .sample(SampleOpts {
            temperature: 1.0,
            top_p: 0.95,
            ..SampleOpts::greedy()
        })
        .build()?;

    let mut out = Vec::new();
    runner.generate(&ids, n_new, |tok| out.push(tok))?;
    assert_eq!(out.len(), n_new);
    let text = decode_ids_auto(&gguf, None, &out, false).unwrap_or_default();
    eprintln!("[gemma4 coder] {device:?} generated {n_new} tokens: {text:?}");
    Ok(())
}
