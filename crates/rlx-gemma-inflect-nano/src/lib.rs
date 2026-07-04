// RLX — versatile ML compiler + runtime. GPLv3.
//! Gemma 3 270M chat + Inflect-Nano English TTS — unpublished pairing demo.
//!
//! Two examples:
//! - `chat` — interactive streaming voice REPL (type → Gemma → spoken reply,
//!   sentences vocoded as they're generated). Metal by default.
//!   `cargo run --release -p rlx-gemma-inflect-nano --features metal --example chat`
//! - `speak` — one-shot: prompt in, WAV out.
//!   `cargo run -p rlx-gemma-inflect-nano --example speak -- --user "…"`
//!
//! See this crate's `README.md` and the repo-root `TTS.md` for the full writeup
//! (design, perf, and the Metal / tokenizer / vocoder-cache fixes).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_cli::{ChatMessage, auto_chat_template, parse_gemma_device};
use rlx_inflect_nano::InflectNano;
use rlx_qwen35::{encode_prompt_auto, resolve_tokenizer_path};
use rlx_runtime::{Device, is_available};

/// Default Gemma 3 270M IT GGUF (`just fetch-gemma3-270m`).
pub fn default_gemma_gguf() -> PathBuf {
    std::env::var_os("RLX_GEMMA3_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/rlx-weights/gemma-3-270m.gguf"))
}

/// HuggingFace `tokenizer.json` for Gemma 3 270M (sibling of the GGUF by default).
pub fn default_gemma_tokenizer() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("RLX_GEMMA3_TOKENIZER") {
        return Some(PathBuf::from(p));
    }
    resolve_tokenizer_path(&default_gemma_gguf(), None)
}

/// Default Inflect-Nano RLX bundle (`scripts/export_inflect_nano.py`).
pub fn default_inflect_data_dir() -> PathBuf {
    std::env::var_os("RLX_INFLECT_NANO_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("weights/inflect-nano-rlx"))
}

/// Trim and flatten model output into a single spoken line for TTS.
pub fn speech_text_from_reply(reply: &str) -> String {
    let mut out = String::new();
    for line in reply.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(line);
        if line.ends_with('.') || line.ends_with('!') || line.ends_with('?') {
            break;
        }
    }
    let trimmed = out.trim();
    if trimmed.is_empty() {
        reply.trim().replace('\n', " ")
    } else {
        trimmed.to_string()
    }
}

/// Check that default weight paths exist; returns `(gemma, inflect)` paths.
pub fn resolve_default_paths() -> (PathBuf, PathBuf) {
    (default_gemma_gguf(), default_inflect_data_dir())
}

/// Ensure the Gemma GGUF, tokenizer, and Inflect bundle exist.
pub fn ensure_paths_exist(
    gemma: &Path,
    tokenizer: Option<&Path>,
    inflect: &Path,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        gemma.is_file(),
        "missing Gemma GGUF at {gemma:?} — run `just fetch-gemma3-270m` or set RLX_GEMMA3_GGUF"
    );
    anyhow::ensure!(
        resolve_tokenizer_path(gemma, tokenizer).is_some(),
        "missing Gemma tokenizer.json next to {gemma:?} — run `just fetch-gemma3-270m` \
         or set RLX_GEMMA3_TOKENIZER"
    );
    anyhow::ensure!(
        inflect.join("config.json").is_file(),
        "missing Inflect bundle at {inflect:?} — export with scripts/export_inflect_nano.py \
         (see crates/rlx-inflect-nano/README.md) or set RLX_INFLECT_NANO_DATA"
    );
    Ok(())
}

/// Render a full multi-turn transcript through the GGUF chat template and
/// tokenize it. Unlike [`rlx_gemma::encode_chat_prompt_auto`] (a single user
/// turn), this keeps every prior turn so Gemma sees the conversation so far —
/// the difference between a one-shot `speak` and an actual `chat`.
///
/// Mirrors the `<bos>` / duplicate-`107,2` cleanup that
/// `rlx_gemma::encode_chat_prompt_auto` applies, so tokens match the runner's
/// expectations exactly.
pub fn encode_chat_turns(
    weights: &Path,
    tokenizer: Option<&Path>,
    messages: &[ChatMessage],
) -> Result<Vec<u32>> {
    let tmpl = auto_chat_template(weights)
        .context("Gemma GGUF has no embedded chat template; multi-turn chat needs one")?;
    let text = tmpl.render(messages, true)?;
    let mut ids = if text.starts_with("<bos>") {
        let body = text.strip_prefix("<bos>").unwrap_or(&text);
        let mut v = encode_prompt_auto(weights, tokenizer, body)?;
        if v.first() != Some(&2) {
            v.insert(0, 2);
        }
        v
    } else {
        encode_prompt_auto(weights, tokenizer, &text)?
    };
    if ids.len() >= 2 && ids[0] == 107 && ids[1] == 2 {
        ids.remove(0);
    }
    Ok(ids)
}

/// Resolve the Inflect vocoder device from a CLI spec (`auto`, `cpu`, `metal`, …).
///
/// `auto` prefers the fastest detected accelerator (Metal → MLX → wgpu), then
/// falls back to the LM device if it is available, else CPU.
pub fn resolve_tts_device(spec: &str, lm_device: Device) -> Result<Device> {
    if spec.eq_ignore_ascii_case("auto") {
        if let Some(d) = InflectNano::preferred_accelerator() {
            return Ok(d);
        }
        return Ok(if is_available(lm_device) {
            lm_device
        } else {
            Device::Cpu
        });
    }
    parse_gemma_device(spec)
}

/// Play mono f32 PCM at `sample_rate` through the system default output,
/// blocking until playback finishes. Writes a temporary WAV and shells out to
/// macOS `afplay`; other platforms get an error that names the written file so
/// the caller can fall back gracefully.
pub fn play_samples(samples: &[f32], sample_rate: u32) -> Result<()> {
    let path = std::env::temp_dir().join("rlx-gemma-inflect-chat.wav");
    rlx_inflect_nano::audio::write_wav(&path, samples, sample_rate)?;
    play_wav_file(&path)
}

/// Play an existing WAV file through the system player, blocking until done.
pub fn play_wav_file(path: &Path) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let status = std::process::Command::new("afplay")
            .arg(path)
            .status()
            .context("failed to launch macOS `afplay`")?;
        anyhow::ensure!(status.success(), "afplay exited with status {status}");
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        anyhow::bail!(
            "live playback is wired for macOS (afplay) only; WAV written to {}",
            path.display()
        )
    }
}
