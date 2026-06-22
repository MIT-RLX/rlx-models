//! Whisper round-trip validation for Kyutai TTS output.
//!
//! Two opt-in modes:
//!
//! 1. **Static WAV mode** — point `RLX_KYUTAI_TTS_VALIDATE_WAV` at a WAV that
//!    was synthesised against `kyutai/tts-1.6b-en_fr` (e.g. produced by the
//!    upstream `moshi` Python pipeline, or by a future RLX generation path).
//!    The test loads it, runs `rlx-whisper` greedy ASR, and asserts the
//!    transcript is non-empty (and optionally contains words from
//!    `RLX_KYUTAI_TTS_VALIDATE_PROMPT`). This is the immediately-useful path:
//!    it lets you validate any synthesised audio against the same Whisper
//!    pipeline the rest of the workspace uses for parity.
//!
//! 2. **End-to-end mode** (placeholder) — when `RLX_KYUTAI_TTS_E2E=1` and
//!    weights are on disk, would drive `KyutaiTtsSession` → Mimi WAV → Whisper.
//!    Currently skipped: `KyutaiTtsSession::generate` returns "not yet wired"
//!    because the depth-multiplexed Kyutai TTS architecture is not yet
//!    implemented in the eager backbone (see crate docs).
//!
//! Both modes skip silently when their preconditions aren't met so CI stays
//! green without weights.

use anyhow::{Context, Result, ensure};
use rlx_kyutai_tts::KyutaiTtsConfig;
use rlx_mimi::audio::load_wav_mono;
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner, normalize_transcript};
use std::path::{Path, PathBuf};

const MIN_OUTPUT_SAMPLES: usize = 4_800; // 0.3 s @ 16 kHz
const MIN_PEAK: f32 = 1e-3;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        let p = PathBuf::from(d);
        if p.join("model.safetensors").is_file() {
            return Some(p);
        }
    }
    let cache = repo_root().join(".cache");
    for name in [
        "whisper-base.en",
        "whisper-base",
        "whisper-tiny.en",
        "whisper-tiny",
    ] {
        let p = cache.join(name);
        if p.join("model.safetensors").is_file() {
            return Some(p);
        }
    }
    None
}

fn peak(pcm: &[f32]) -> f32 {
    pcm.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
}

fn build_whisper(dir: &Path, device: Device) -> Result<WhisperRunner> {
    WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(device)
        .language("en")
        .build()
}

fn transcribe_wav(wav_path: &Path) -> Result<String> {
    let whisper_dir = whisper_dir()
        .context("missing Whisper weights — set RLX_WHISPER_DIR or place under .cache/whisper-*")?;
    // Kyutai TTS output is 24 kHz; `load_wav_mono` from rlx-mimi resamples
    // to Whisper's 16 kHz target on the fly.
    let pcm_16k = load_wav_mono(wav_path, WHISPER_RATE as u32)
        .with_context(|| format!("load wav {}", wav_path.display()))?;
    ensure!(
        pcm_16k.len() >= MIN_OUTPUT_SAMPLES,
        "wav too short ({} samples; need ≥ {})",
        pcm_16k.len(),
        MIN_OUTPUT_SAMPLES
    );
    ensure!(
        peak(&pcm_16k) >= MIN_PEAK,
        "wav near silence (peak {})",
        peak(&pcm_16k)
    );

    let mut whisper = build_whisper(&whisper_dir, Device::Cpu)?;
    whisper.transcribe_greedy(&pcm_16k)
}

/// Mode 1: validate a pre-existing TTS WAV.
#[test]
fn whisper_validates_static_kyutai_tts_wav() -> Result<()> {
    let wav = match std::env::var("RLX_KYUTAI_TTS_VALIDATE_WAV").ok() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!(
                "skip: set RLX_KYUTAI_TTS_VALIDATE_WAV=/path/to/output.wav to validate \
                 a Kyutai TTS-synthesised WAV against rlx-whisper"
            );
            return Ok(());
        }
    };
    ensure!(
        wav.is_file(),
        "RLX_KYUTAI_TTS_VALIDATE_WAV missing: {}",
        wav.display()
    );

    if whisper_dir().is_none() {
        eprintln!("skip: set RLX_WHISPER_DIR or fetch whisper weights into .cache/whisper-*");
        return Ok(());
    }

    let text = transcribe_wav(&wav)?;
    let norm = normalize_transcript(&text);
    eprintln!("kyutai-tts → whisper: {norm}");
    ensure!(
        !norm.trim().is_empty(),
        "whisper returned an empty transcript"
    );

    if let Ok(prompt) = std::env::var("RLX_KYUTAI_TTS_VALIDATE_PROMPT") {
        let prompt_norm = normalize_transcript(&prompt);
        let hit = prompt_norm
            .split_whitespace()
            .filter(|w| w.len() >= 4)
            .filter(|w| norm.contains(*w))
            .count();
        let total = prompt_norm
            .split_whitespace()
            .filter(|w| w.len() >= 4)
            .count();
        eprintln!("prompt word coverage: {hit}/{total}");
        ensure!(
            hit * 2 >= total.max(1),
            "transcript {norm:?} covered < 50% of prompt content words from {prompt:?}"
        );
    }
    Ok(())
}

/// Mode 2: end-to-end roundtrip (placeholder, currently always skipped).
#[test]
fn kyutai_tts_to_whisper_e2e_roundtrip() -> Result<()> {
    if std::env::var("RLX_KYUTAI_TTS_E2E").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_KYUTAI_TTS_E2E=1 to drive Kyutai TTS → Whisper");
        return Ok(());
    }
    // Confirm config still loads (defensive against drift between this test
    // and `KyutaiTtsConfig`).
    let cfg = KyutaiTtsConfig::v1_6b_en_fr();
    ensure!(cfg.dim == 2048, "config drift: dim {}", cfg.dim);
    ensure!(cfg.n_q == 32, "config drift: n_q {}", cfg.n_q);

    // `KyutaiTtsSession::generate` returns "not yet wired" — the
    // depth-multiplexed architecture (per-step DepFormer + cross-attn
    // conditioners + demuxed second stream) is not yet implemented.
    // Once that lands, drive it here and feed the Mimi-decoded PCM through
    // `transcribe_wav` (via a temp WAV) the same way Mode 1 does.
    eprintln!(
        "skip: KyutaiTtsSession::generate not yet wired — see crate docs. \
         Use Mode 1 (RLX_KYUTAI_TTS_VALIDATE_WAV) with WAVs from the upstream \
         moshi pipeline in the meantime."
    );
    Ok(())
}
