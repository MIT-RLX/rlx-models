//! Compare Rust DAC against a baked reference fixture (no Python at test time).
//!
//! Fixture: `tests/fixtures/dac_24khz_synthetic.json` (reference encode/decode outputs).

use anyhow::{Context, Result, ensure};
use rlx_dac::{DacCodec, DacCodes, resolve_model_dir};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct Fixture {
    pcm_samples: usize,
    num_frames: usize,
    n_quantizers: usize,
    input_pcm: Vec<f32>,
    codes: Vec<Vec<u32>>,
    recon_pcm: Vec<f32>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dac_24khz_synthetic.json")
}

fn model_dir() -> Option<PathBuf> {
    let d = resolve_model_dir(None);
    d.join("model.safetensors").is_file().then_some(d)
}

#[test]
fn decode_reference_codes_matches_fixture() -> Result<()> {
    let path = fixture_path();
    if !path.is_file() {
        eprintln!("skip: missing fixture {}", path.display());
        return Ok(());
    }
    let Some(dir) = model_dir() else {
        eprintln!("skip: missing DAC weights (set RLX_DAC_DIR or run --fetch)");
        return Ok(());
    };
    let fixture: Fixture =
        serde_json::from_str(&std::fs::read_to_string(&path)?).context("parse fixture")?;
    let codec = DacCodec::open(&dir)?;
    let codes = DacCodes::from_quantizer_layout(fixture.codes.clone());
    ensure_eq_frames(&codes, &fixture)?;

    let recon = codec.decode_codes(&codes)?;
    let n = fixture.recon_pcm.len().min(recon.len());
    let mae = (0..n)
        .map(|i| (recon[i] - fixture.recon_pcm[i]).abs())
        .sum::<f32>()
        / n as f32;
    eprintln!("decode reference codes MAE vs fixture: {mae:.8}");
    assert!(mae < 1e-4, "decoder diverged from reference (MAE {mae:.8})");
    Ok(())
}

#[test]
fn encode_pcm_matches_reference_fixture() -> Result<()> {
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

    let codec = DacCodec::open(&dir)?;
    let codes = codec.encode_pcm(&fixture.input_pcm, Some(fixture.n_quantizers))?;
    ensure_eq_frames(&codes, &fixture)?;

    let ours = codes.to_quantizer_layout();
    let mut exact = 0usize;
    let mut total = 0usize;
    for (ours_row, ref_row) in ours.iter().zip(fixture.codes.iter()) {
        for (&a, &b) in ours_row.iter().zip(ref_row.iter()) {
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
    assert!(
        ratio >= 0.95,
        "RVQ codes diverged from reference (matched {ratio:.1}%)"
    );

    let recon = codec.decode_codes(&codes)?;
    let n = fixture.recon_pcm.len().min(recon.len());
    let mae = (0..n)
        .map(|i| (recon[i] - fixture.recon_pcm[i]).abs())
        .sum::<f32>()
        / n as f32;
    eprintln!("encode→decode MAE vs reference recon: {mae:.6}");
    assert!(
        mae < 0.02,
        "roundtrip PCM MAE {mae:.6} too high vs reference"
    );

    Ok(())
}

fn ensure_eq_frames(codes: &DacCodes, fixture: &Fixture) -> Result<()> {
    ensure!(
        codes.num_frames() == fixture.num_frames,
        "frame count {} != {}",
        codes.num_frames(),
        fixture.num_frames
    );
    ensure!(
        codes.num_quantizers == fixture.n_quantizers,
        "quantizers {} != {}",
        codes.num_quantizers,
        fixture.n_quantizers
    );
    Ok(())
}
