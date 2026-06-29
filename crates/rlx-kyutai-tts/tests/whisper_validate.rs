//! Whisper round-trip validation for Kyutai TTS output.
//!
//! Two opt-in modes:
//!
//! 1. **Static WAV mode** — point `RLX_KYUTAI_TTS_VALIDATE_WAV` at a WAV that
//!    was synthesised against `kyutai/tts-1.6b-en_fr`. Runs `rlx-whisper` greedy
//!    ASR and asserts the transcript is non-empty (optionally checks prompt words
//!    via `RLX_KYUTAI_TTS_VALIDATE_PROMPT`).
//!
//! 2. **End-to-end mode** — `RLX_KYUTAI_TTS_E2E=1` with weights on disk drives
//!    `KyutaiTtsSession::generate` → Mimi WAV → Whisper.
//!
//! Both modes skip when preconditions aren't met so CI stays green without weights.

use anyhow::{Context, Result, ensure};
use rlx_kyutai_tts::{
    GenerationConfig, KyutaiTtsConfig, KyutaiTtsSession,
    checkpoint::KyutaiTtsVoice,
    device::resolve_kyutai_tts_device,
    download::{DEFAULT_VOICE_NAME, default_kyutai_tts_dir, default_mimi_dir},
};
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

/// Mode 2: synthesise with native Kyutai TTS, then transcribe with Whisper.
#[test]
fn kyutai_tts_to_whisper_e2e_roundtrip() -> Result<()> {
    if std::env::var("RLX_KYUTAI_TTS_E2E").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_KYUTAI_TTS_E2E=1 to drive Kyutai TTS → Whisper");
        return Ok(());
    }

    let cfg = KyutaiTtsConfig::v1_6b_en_fr();
    ensure!(cfg.dim == 2048, "config drift: dim {}", cfg.dim);
    ensure!(cfg.n_q == 32, "config drift: n_q {}", cfg.n_q);

    let model_dir = std::env::var("RLX_KYUTAI_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_kyutai_tts_dir());
    let weights = model_dir.join("dsm_tts_1e68beda@240.safetensors");
    if !weights.is_file() {
        eprintln!(
            "skip: missing LM weights at {} (fetch with --fetch)",
            weights.display()
        );
        return Ok(());
    }
    let mimi_dir = std::env::var("RLX_MIMI_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_mimi_dir());
    if !mimi_dir.join("model.safetensors").is_file()
        && !mimi_dir
            .join("tokenizer-e351c8d8-checkpoint125.safetensors")
            .is_file()
    {
        eprintln!(
            "skip: missing Mimi sidecar in {} (fetch with --fetch)",
            mimi_dir.display()
        );
        return Ok(());
    }
    if whisper_dir().is_none() {
        eprintln!("skip: set RLX_WHISPER_DIR or fetch whisper weights for e2e");
        return Ok(());
    }

    let prompt =
        std::env::var("RLX_KYUTAI_TTS_VALIDATE_PROMPT").unwrap_or_else(|_| "Hello.".into());
    // Eager CPU LM is the default when device is Cpu (avoids GPU requirement in e2e rigs).
    let device = std::env::var("RLX_KYUTAI_TTS_E2E_DEVICE")
        .ok()
        .map(|s| resolve_kyutai_tts_device(&s))
        .transpose()?
        .unwrap_or_else(|| resolve_kyutai_tts_device("auto").unwrap());
    let mut session = KyutaiTtsSession::open_on(&model_dir, &mimi_dir, device)?;
    session.set_voice(KyutaiTtsVoice::new(
        std::env::var("RLX_KYUTAI_TTS_VOICE").unwrap_or_else(|_| DEFAULT_VOICE_NAME.into()),
    ));
    let gen_cfg = GenerationConfig {
        max_steps: 80,
        ..GenerationConfig::default()
    };
    eprintln!("e2e: synthesising {:?} on {device:?} …", prompt);
    let result = session.generate(&prompt, &gen_cfg)?;
    ensure!(
        !result.samples.is_empty(),
        "generation returned empty PCM ({} frames)",
        result.audio_frames.len()
    );
    let tmp = std::env::temp_dir().join("rlx-kyutai-tts-e2e.wav");
    rlx_mimi::audio::write_wav_mono(&tmp, &result.samples, result.sample_rate)?;
    eprintln!(
        "e2e: wrote {} samples to {}",
        result.samples.len(),
        tmp.display()
    );
    let text = transcribe_wav(&tmp)?;
    let norm = normalize_transcript(&text);
    eprintln!("e2e transcript: {norm}");
    ensure!(!norm.trim().is_empty(), "whisper returned empty transcript");
    Ok(())
}
