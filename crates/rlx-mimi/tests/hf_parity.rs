//! Compare Rust Mimi against baked HF `transformers` fixtures.
//!
//! Generate fixture (24 kHz WAV recommended):
//!   python3 scripts/mimi_hf_parity.py \\
//!     --wav crates/rlx-qwen3-tts/examples/audio/ask_not.wav \\
//!     --out crates/rlx-mimi/tests/fixtures/hf_ask_not.json

use anyhow::{Context, Result, ensure};
use rlx_mimi::{MimiCodec, MimiCodes, default_mimi_dir};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct Fixture {
    pcm_samples: usize,
    num_quantizers: usize,
    num_frames: usize,
    input_pcm: Vec<f32>,
    codes_hf_layout: Vec<Vec<u32>>,
    recon_pcm: Vec<f32>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hf_ask_not.json")
}

fn model_dir() -> Option<PathBuf> {
    let d = default_mimi_dir();
    d.join("model.safetensors").is_file().then_some(d)
}

#[test]
fn decode_hf_codes_matches_fixture() -> Result<()> {
    let path = fixture_path();
    if !path.is_file() {
        eprintln!(
            "skip hf decode parity: missing {} (run scripts/mimi_hf_parity.py)",
            path.display()
        );
        return Ok(());
    }
    let Some(dir) = model_dir() else {
        eprintln!("skip hf decode parity: missing mimi weights");
        return Ok(());
    };
    let fixture: Fixture =
        serde_json::from_str(&std::fs::read_to_string(&path)?).context("parse fixture")?;
    let codec = MimiCodec::open(&dir)?;
    let codes = MimiCodes::from_hf_layout(fixture.codes_hf_layout.clone());
    ensure_eq_frames(&codes, &fixture)?;

    let recon = codec.decode_codes(&codes)?;
    let n = fixture.recon_pcm.len().min(recon.len());
    let mae = (0..n)
        .map(|i| (recon[i] - fixture.recon_pcm[i]).abs())
        .sum::<f32>()
        / n as f32;
    eprintln!("decode HF codes MAE vs fixture: {mae:.8}");
    assert!(mae < 1e-5, "decoder diverged from HF (MAE {mae:.8})");
    Ok(())
}

#[test]
fn encode_pcm_matches_hf_fixture() -> Result<()> {
    let path = fixture_path();
    if !path.is_file() {
        return Ok(());
    }
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let fixture: Fixture =
        serde_json::from_str(&std::fs::read_to_string(&path)?).context("parse fixture")?;
    ensure!(
        fixture.input_pcm.len() == fixture.pcm_samples,
        "fixture pcm length mismatch"
    );

    let codec = MimiCodec::open(&dir)?;
    let codes = codec.encode_pcm(&fixture.input_pcm, Some(fixture.num_quantizers))?;
    ensure_eq_frames(&codes, &fixture)?;

    let ours_hf = codes.to_hf_layout();
    let mut exact = 0usize;
    let mut total = 0usize;
    for (ours, hf) in ours_hf.iter().zip(fixture.codes_hf_layout.iter()) {
        for (&a, &b) in ours.iter().zip(hf.iter()) {
            total += 1;
            if a == b {
                exact += 1;
            }
        }
    }
    let ratio = exact as f32 / total as f32;
    eprintln!(
        "encode code index match: {exact}/{total} ({:.1}%)",
        ratio * 100.0
    );

    // Semantic codebook (index 0) should track closely; full RVQ cascade tolerates tiny FP drift.
    let sem_match = ours_hf[0]
        .iter()
        .zip(fixture.codes_hf_layout[0].iter())
        .filter(|(a, b)| a == b)
        .count();
    let sem_ratio = sem_match as f32 / fixture.num_frames as f32;
    assert!(
        sem_ratio >= 0.95,
        "semantic codebook diverged ({sem_match}/{})",
        fixture.num_frames
    );
    assert!(
        ratio >= 0.90,
        "RVQ codes diverged from HF (matched {ratio:.1}%)"
    );

    let recon = codec.decode_codes(&codes)?;
    let n = fixture.recon_pcm.len().min(recon.len());
    let mae = (0..n)
        .map(|i| (recon[i] - fixture.recon_pcm[i]).abs())
        .sum::<f32>()
        / n as f32;
    eprintln!("encode→decode MAE vs HF recon: {mae:.6}");
    assert!(mae < 0.02, "roundtrip PCM MAE {mae:.6} too high vs HF");

    Ok(())
}

fn ensure_eq_frames(codes: &MimiCodes, fixture: &Fixture) -> Result<()> {
    ensure!(
        codes.num_frames() == fixture.num_frames,
        "frame count {} != {}",
        codes.num_frames(),
        fixture.num_frames
    );
    ensure!(
        codes.num_quantizers == fixture.num_quantizers,
        "quantizers {} != {}",
        codes.num_quantizers,
        fixture.num_quantizers
    );
    Ok(())
}
