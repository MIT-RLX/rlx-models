// SPDX-License-Identifier: GPL-3.0-only
//! Verify the device-routing fix: NeuTTS should load its backbone on the GPU
//! (Metal) via the non-packed F32 path instead of single-core CPU packed-K-quant.
//!
//! Run:
//!   REF_NPY=/Users/Shared/skill/src-tauri/resources/neutts-samples/jo.npy \
//!   NEUTTS_DECODER_PATH=/Users/macmini/.skill/models/neutts/neucodec_decoder.safetensors \
//!   cargo run -p rlx-neutts --example synth_metal --features llama,codec,rlx,metal
//!
//! Add `--release` for representative timing.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_neutts::NeuTTS;

/// Minimal `.npy` reader for a 1-D little-endian `<i4` array (our ref-codes).
fn read_npy_i32(path: &str) -> Result<Vec<i32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {path}"))?;
    anyhow::ensure!(&bytes[..6] == b"\x93NUMPY", "not a .npy file");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let data = &bytes[10 + header_len..];
    Ok(data
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Write 24 kHz mono f32 as a 16-bit PCM WAV (no external deps).
fn write_wav(path: &PathBuf, audio: &[f32], rate: u32) -> Result<()> {
    let mut out = Vec::with_capacity(44 + audio.len() * 2);
    let data_len = (audio.len() * 2) as u32;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in audio {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, out).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    let backbone = std::env::var("BACKBONE_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(
                "/Users/macmini/.cache/huggingface/hub/models--neuphonic--neutts-nano-q4-gguf/\
             snapshots/8ae1694877fdf9d7c4a7bee2cc9775ba7eab3923/neutts-nano-Q4_0.gguf",
            )
        });
    let ref_npy = std::env::var("REF_NPY").context("set REF_NPY to a preset .npy (e.g. jo.npy)")?;
    let ref_codes = read_npy_i32(&ref_npy)?;
    eprintln!("[synth] ref-codes: {} tokens", ref_codes.len());

    // `NeuTTS::load` now resolves the best device (Metal here) and the GPU path
    // uses F32 (non-packed) — this is the fix under test.
    let t0 = Instant::now();
    let model = NeuTTS::load(&backbone, "en").context("load NeuTTS")?;
    eprintln!(
        "[synth] backbone loaded in {:.2}s",
        t0.elapsed().as_secs_f32()
    );

    let ref_ipa = std::env::var("REF_IPA")
        .unwrap_or_else(|_| "soʊ aɪ dʒʌst traɪd njuːfɒnɪk ænd aɪm dʒɛnjuːɪnli ɪmprɛst".into());
    let input_ipa = std::env::var("INPUT_IPA")
        .unwrap_or_else(|_| "ðə kwɪk braʊn fɒks dʒʌmps oʊvɚ ðə leɪzi dɒɡ".into());

    let t1 = Instant::now();
    let audio = model
        .infer_from_ipa(&input_ipa, &ref_codes, &ref_ipa)
        .context("infer_from_ipa")?;
    let secs = audio.len() as f32 / rlx_neutts::SAMPLE_RATE as f32;
    eprintln!(
        "[synth] {} samples = {:.2}s audio, synth wall {:.2}s",
        audio.len(),
        secs,
        t1.elapsed().as_secs_f32(),
    );
    anyhow::ensure!(!audio.is_empty(), "no audio synthesized");

    let out = std::env::temp_dir().join("neutts_metal.wav");
    write_wav(&out, &audio, rlx_neutts::SAMPLE_RATE)?;
    eprintln!("[synth] wrote {}", out.display());
    Ok(())
}
