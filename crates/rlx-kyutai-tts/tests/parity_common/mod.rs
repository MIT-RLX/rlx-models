#![allow(dead_code)]

//! Shared helpers for Kyutai TTS code-frame parity (Moshi ref, eager CPU, RLX backends).

use anyhow::{Result, bail, ensure};
use rlx_kyutai_tts::checkpoint::KyutaiTtsCheckpoint;
use rlx_kyutai_tts::config::KyutaiTtsConfig;
use rlx_kyutai_tts::download::{
    default_kyutai_tts_dir, default_voices_dir, ensure_voice_embedding, tokenizer_path,
};
use rlx_kyutai_tts::generate::{GenerateConfig, generate_codes};
use rlx_kyutai_tts::model::{KyutaiTtsModel, load_voice_speaker_wavs};
use rlx_kyutai_tts::rlx_model::RlxKyutaiTtsModel;
use rlx_kyutai_tts::tokenizer::KyutaiTokenizer;
use rlx_runtime::Device;
use serde::Deserialize;
use std::path::{Path, PathBuf};

pub const PARITY_PROMPT: &str = "Hello world, this is a test of the Kyutai text to speech system.";
pub const PARITY_VOICE: &str = "alba-mackenna/casual.wav";

#[derive(Debug, Deserialize)]
pub struct RefExport {
    pub delay: usize,
    pub end: Option<usize>,
    pub trimmed: Vec<Vec<u32>>,
}

pub fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var("RLX_KYUTAI_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_kyutai_tts_dir());
    if dir.join("dsm_tts_1e68beda@240.safetensors").is_file() {
        Some(dir)
    } else {
        eprintln!("skip: missing weights in {}", dir.display());
        None
    }
}

pub fn mimi_codes_ref_path() -> PathBuf {
    std::env::var("RLX_KYUTAI_MIMI_CODES_REF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/py_mimi_codes.json"))
}

pub fn parity_gen_cfg() -> GenerateConfig {
    GenerateConfig {
        max_steps: 200,
        n_q: 32,
        cfg_alpha: 2.0,
        text_temperature: 0.0,
        audio_temperature: 0.0,
        seed: 42,
    }
}

pub fn short_gen_cfg() -> GenerateConfig {
    GenerateConfig {
        max_steps: 25,
        n_q: 8,
        cfg_alpha: 2.0,
        text_temperature: 0.0,
        audio_temperature: 0.0,
        seed: 7,
    }
}

pub fn load_speaker(_dir: &Path) -> Result<ndarray::Array2<f32>> {
    let voice = ensure_voice_embedding(
        &default_voices_dir(),
        KyutaiTtsCheckpoint::V1_6bEnFr,
        PARITY_VOICE,
    )?;
    load_voice_speaker_wavs(&voice)
}

pub fn eager_codes(dir: &Path, cfg: &GenerateConfig, prompt: &str) -> Result<Vec<Vec<u32>>> {
    let model_cfg = KyutaiTtsConfig::v1_6b_en_fr();
    let tokenizer = KyutaiTokenizer::load(tokenizer_path(dir))?;
    let mut m = KyutaiTtsModel::open(dir, model_cfg, Device::Cpu)?;
    let spk = load_speaker(dir)?;
    generate_codes(&mut m, &tokenizer, prompt, cfg.clone(), Some(&spk)).map(|(f, _)| f)
}

pub fn rlx_codes(
    dir: &Path,
    device: Device,
    cfg: &GenerateConfig,
    prompt: &str,
) -> Result<Vec<Vec<u32>>> {
    let model_cfg = KyutaiTtsConfig::v1_6b_en_fr();
    let tokenizer = KyutaiTokenizer::load(tokenizer_path(dir))?;
    let mut m = RlxKyutaiTtsModel::open(dir, model_cfg.clone(), device, model_cfg.context)?;
    let spk = load_speaker(dir)?;
    generate_codes(&mut m, &tokenizer, prompt, cfg.clone(), Some(&spk)).map(|(f, _)| f)
}

pub fn assert_frames_match(label: &str, reference: &[Vec<u32>], actual: &[Vec<u32>]) -> Result<()> {
    ensure!(
        !reference.is_empty() && !actual.is_empty(),
        "{label}: expected non-empty frame lists"
    );
    let n = reference.len().min(actual.len());
    for (i, (r, a)) in reference.iter().zip(actual.iter()).take(n).enumerate() {
        if r != a {
            eprintln!("{label}: first mismatch at frame {i}");
            eprintln!("  ref {:?}", &r[..r.len().min(8)]);
            eprintln!("  got {:?}", &a[..a.len().min(8)]);
            bail!("{label}: code mismatch at frame {i}");
        }
    }
    ensure!(
        reference.len() == actual.len(),
        "{label}: frame count ref {} vs got {} (first {n} matched)",
        reference.len(),
        actual.len()
    );
    eprintln!("{label}: all {} frames match", reference.len());
    Ok(())
}

pub fn codes_parity_enabled() -> bool {
    std::env::var("RLX_KYUTAI_CODES_PARITY").ok().as_deref() == Some("1")
}

pub fn load_moshi_reference(path: &Path) -> Result<RefExport> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}
