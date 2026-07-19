//! End-to-end Kyutai TTS generation on each available backend (real weights).
//!
//! ```bash
//! RLX_KYUTAI_TTS_E2E=1 cargo test -p rlx-kyutai-tts --test rlx_e2e_backends --features all-backends -- --nocapture
//! ```

mod backend_common;

use anyhow::{Context, Result, ensure};
use rlx_kyutai_tts::{
    GenerationConfig, KyutaiTtsSession, KyutaiTtsVoice,
    download::{DEFAULT_VOICE_NAME, default_kyutai_tts_dir, default_mimi_dir},
};
use rlx_mimi::audio::{load_wav_mono, write_wav_mono};
use rlx_runtime::{Device, is_available};
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner, normalize_transcript};
use std::path::{Path, PathBuf};

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

fn mimi_ready(mimi_dir: &Path) -> bool {
    mimi_dir.join("model.safetensors").is_file()
        || mimi_dir
            .join("tokenizer-e351c8d8-checkpoint125.safetensors")
            .is_file()
}

fn weights_ready(model_dir: &Path, mimi_dir: &Path) -> bool {
    model_dir.join("dsm_tts_1e68beda@240.safetensors").is_file() && mimi_ready(mimi_dir)
}

fn transcribe(wav: &Path) -> Result<String> {
    let dir = whisper_dir().context("missing Whisper weights")?;
    let pcm = load_wav_mono(wav, WHISPER_RATE as u32)?;
    ensure!(pcm.len() >= 4800, "wav too short");
    let mut w = WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()?;
    w.transcribe_greedy(&pcm)
}

pub fn e2e_on_device(device: Device, label: &str) -> Result<bool> {
    if device != Device::Cpu && !is_available(device) {
        eprintln!("{label}: skipped (not available)");
        return Ok(false);
    }
    let model_dir = std::env::var("RLX_KYUTAI_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_kyutai_tts_dir());
    let mimi_dir = std::env::var("RLX_MIMI_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_mimi_dir());
    if !weights_ready(&model_dir, &mimi_dir) {
        eprintln!("{label}: skipped (missing LM or Mimi weights)");
        return Ok(false);
    }
    if whisper_dir().is_none() {
        eprintln!("{label}: skipped (no Whisper weights)");
        return Ok(false);
    }

    let prompt =
        std::env::var("RLX_KYUTAI_TTS_VALIDATE_PROMPT").unwrap_or_else(|_| "Hello.".into());
    eprintln!("{label}: synthesising {prompt:?} on {device:?} …");
    let mut session = KyutaiTtsSession::open_on(&model_dir, &mimi_dir, device)?;
    session.set_voice(KyutaiTtsVoice::new(DEFAULT_VOICE_NAME));
    let result = session.generate(
        &prompt,
        &GenerationConfig {
            max_steps: 60,
            ..GenerationConfig::default()
        },
    )?;
    ensure!(!result.samples.is_empty(), "{label}: empty PCM");
    let wav = std::env::temp_dir().join(format!(
        "rlx-kyutai-e2e-{}-{}.wav",
        label.replace(['/', ' '], "_"),
        std::process::id()
    ));
    write_wav_mono(&wav, &result.samples, result.sample_rate)?;
    let text = transcribe(&wav)?;
    let norm = normalize_transcript(&text);
    eprintln!("{label}: {norm:?} ({} samples)", result.samples.len());
    ensure!(!norm.trim().is_empty(), "{label}: empty whisper transcript");
    Ok(true)
}

macro_rules! e2e_backend_test {
    ($name:ident, $dev:expr, $label:literal) => {
        #[test]
        fn $name() -> Result<()> {
            if std::env::var("RLX_KYUTAI_TTS_E2E").ok().as_deref() != Some("1") {
                eprintln!(concat!("skip ", $label, ": set RLX_KYUTAI_TTS_E2E=1"));
                return Ok(());
            }
            e2e_on_device($dev, $label)?;
            Ok(())
        }
    };
}

e2e_backend_test!(e2e_cpu, Device::Cpu, "CPU");
e2e_backend_test!(e2e_metal, Device::Metal, "Metal");
e2e_backend_test!(e2e_mlx, Device::Mlx, "MLX");
e2e_backend_test!(e2e_cuda, Device::Cuda, "CUDA");
e2e_backend_test!(e2e_rocm, Device::Rocm, "ROCm");
e2e_backend_test!(e2e_wgpu, Device::Gpu, "wgpu/Gpu");
e2e_backend_test!(e2e_vulkan, Device::Vulkan, "Vulkan");

#[test]
fn e2e_all_available_backends() -> Result<()> {
    if std::env::var("RLX_KYUTAI_TTS_E2E").ok().as_deref() != Some("1") {
        eprintln!("skip e2e_all: set RLX_KYUTAI_TTS_E2E=1");
        return Ok(());
    }
    let mut ran = 0usize;
    for &(dev, label) in backend_common::BACKENDS {
        if dev == Device::Ane {
            eprintln!("{label}: skipped (ANE not on Kyutai session path)");
            continue;
        }
        if dev != Device::Cpu && !is_available(dev) {
            eprintln!("{label}: skipped (not available)");
            continue;
        }
        if e2e_on_device(dev, label)? {
            ran += 1;
        }
    }
    eprintln!("e2e: completed {ran} backend run(s)");
    Ok(())
}
