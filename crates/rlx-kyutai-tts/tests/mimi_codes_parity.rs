//! Compare eager CPU Mimi code frames vs Moshi reference export.
//!
//! Export reference:
//! ```bash
//! .venv-kyutai-moshi/bin/python3 scripts/export_kyutai_mimi_codes.py
//! ```
//!
//! Run:
//! ```bash
//! RLX_KYUTAI_TTS_DIR=/path/to/weights cargo test -p rlx-kyutai-tts --test mimi_codes_parity -- --nocapture
//! ```

use anyhow::{Result, bail};
use rlx_kyutai_tts::checkpoint::KyutaiTtsCheckpoint;
use rlx_kyutai_tts::config::KyutaiTtsConfig;
use rlx_kyutai_tts::download::{
    default_kyutai_tts_dir, default_voices_dir, ensure_voice_embedding, tokenizer_path,
};
use rlx_kyutai_tts::generate::{GenerateConfig, generate_codes};
use rlx_kyutai_tts::model::{KyutaiTtsModel, load_voice_speaker_wavs};
use rlx_kyutai_tts::tokenizer::KyutaiTokenizer;
use rlx_runtime::Device;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct RefExport {
    delay: usize,
    end: Option<usize>,
    trimmed: Vec<Vec<u32>>,
}

fn model_dir() -> Option<PathBuf> {
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

fn ref_path() -> PathBuf {
    std::env::var("RLX_KYUTAI_MIMI_CODES_REF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/py_mimi_codes.json"))
}

fn gen_cfg() -> GenerateConfig {
    GenerateConfig {
        max_steps: 200,
        n_q: 32,
        cfg_alpha: 2.0,
        text_temperature: 0.0,
        audio_temperature: 0.0,
        seed: 42,
    }
}

fn rust_codes(dir: &Path) -> Result<(Vec<Vec<u32>>, Option<usize>)> {
    let cfg = KyutaiTtsConfig::v1_6b_en_fr();
    let tokenizer = KyutaiTokenizer::load(tokenizer_path(dir))?;
    let mut m = KyutaiTtsModel::open(dir, cfg.clone(), Device::Cpu)?;
    let voice = ensure_voice_embedding(
        &default_voices_dir(),
        KyutaiTtsCheckpoint::V1_6bEnFr,
        "alba-mackenna/casual.wav",
    )?;
    let spk = load_voice_speaker_wavs(&voice)?;
    generate_codes(
        &mut m,
        &tokenizer,
        "Hello world, this is a test of the Kyutai text to speech system.",
        gen_cfg(),
        Some(&spk),
    )
}

#[test]
fn eager_trimmed_codes_match_moshi_export() -> Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let path = ref_path();
    if !path.is_file() {
        eprintln!("skip: missing reference at {}", path.display());
        return Ok(());
    }
    let text = std::fs::read_to_string(&path)?;
    let reference: RefExport = serde_json::from_str(&text)?;
    let (rust, end) = rust_codes(&dir)?;

    eprintln!(
        "py: {} frames end={:?} delay={} | rust: {} frames end={end:?}",
        reference.trimmed.len(),
        reference.end,
        reference.delay,
        rust.len(),
    );

    let n = rust.len().min(reference.trimmed.len());
    for (i, (r, p)) in rust
        .iter()
        .zip(reference.trimmed.iter())
        .take(n)
        .enumerate()
    {
        if r != p {
            eprintln!("first mismatch at frame {i}");
            eprintln!("  rust {:?}", &r[..r.len().min(8)]);
            eprintln!("  py   {:?}", &p[..p.len().min(8)]);
            bail!("code mismatch at frame {i}");
        }
    }

    if rust.len() != reference.trimmed.len() {
        eprintln!(
            "frame count differs after matching first {n} frames (rust {} vs py {})",
            rust.len(),
            reference.trimmed.len()
        );
        if n > 0 {
            eprintln!("first {n} trimmed frames match");
        }
        bail!(
            "frame count: rust {} vs py {}",
            rust.len(),
            reference.trimmed.len()
        );
    }

    eprintln!("all {} trimmed frames match", rust.len());
    Ok(())
}
