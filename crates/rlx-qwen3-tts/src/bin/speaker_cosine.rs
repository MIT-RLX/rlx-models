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

//! Compute ECAPA-TDNN x-vector cosine similarity between two WAVs.
//!
//! Used to verify voice-clone fidelity: cosine(speaker_enc(reference),
//! speaker_enc(cloned_output)) should be high (>0.7 = "same speaker"
//! per ECAPA-TDNN literature on Voxceleb).

use anyhow::{Context, Result};
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;
use rlx_qwen3_tts::speaker_encoder;
use std::path::PathBuf;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_dir = PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base");
    let mut wavs: Vec<PathBuf> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                model_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            other => {
                wavs.push(PathBuf::from(other));
                i += 1;
            }
        }
    }
    if wavs.len() < 2 {
        anyhow::bail!("usage: speaker_cosine [--model-dir DIR] <ref.wav> <clone.wav> [more.wav]");
    }
    let store = Qwen3TtsWeightStore::open(&model_dir).context("open weight store")?;
    let ref_x = speaker_encoder::encode_reference_wav(&model_dir, &store, &wavs[0])?;
    println!(
        "reference: {}  ({} dims, norm {:.3})",
        wavs[0].display(),
        ref_x.len(),
        ref_x.iter().map(|v| v * v).sum::<f32>().sqrt()
    );
    for p in wavs.iter().skip(1) {
        let x = speaker_encoder::encode_reference_wav(&model_dir, &store, p)?;
        let c = cosine(&ref_x, &x);
        let same_speaker = if c >= 0.7 {
            "✓ same speaker"
        } else if c >= 0.5 {
            "≈ similar"
        } else {
            "✗ different speaker"
        };
        println!("  vs {}  cosine = {c:.4}   ({same_speaker})", p.display());
    }
    Ok(())
}
