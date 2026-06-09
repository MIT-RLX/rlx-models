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

//! Log-mel frontend — Whisper-style features at 16 kHz via HF preprocessor.

use crate::config::VoxtralAudioConfig;
use anyhow::{Context, Result, bail, ensure};
pub use rlx_whisper::{
    N_SAMPLES, SAMPLE_RATE, SpeechSegment, load_wav_mono_f32, parse_wav_mono_f32,
    pcm_segments_by_vad,
};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Default mel frames for a 30 s chunk (`preprocessor_config.json`: `nb_max_frames = 3000`).
pub const N_FRAMES: usize = 3_000;

#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    pub n_mels: usize,
    pub n_frames: usize,
    pub data: Vec<f32>,
}

fn script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/mel_preprocess.py")
}

fn python_bin() -> String {
    std::env::var("RLX_VOXTRAL_PYTHON").unwrap_or_else(|_| "python3".into())
}

/// Build mel + transcription prompt ids using the HF Whisper frontend (requires Python + transformers).
pub fn pcm_to_mel_and_prompt(
    model_dir: &Path,
    wav: Option<&Path>,
    language: Option<&str>,
) -> Result<(MelSpectrogram, Vec<u32>)> {
    let script = script_path();
    ensure!(
        script.is_file(),
        "missing mel preprocessor script at {}",
        script.display()
    );
    let mut cmd = Command::new(python_bin());
    cmd.arg(&script)
        .arg("--model-dir")
        .arg(model_dir)
        .arg("--json");
    if let Some(lang) = language {
        cmd.arg("--language").arg(lang);
    }
    if let Some(wav) = wav {
        cmd.arg("--wav").arg(wav);
    }
    let out = cmd
        .output()
        .with_context(|| format!("run {}", script.display()))?;
    if !out.status.success() {
        bail!(
            "mel preprocess failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let payload: serde_json::Value =
        serde_json::from_slice(&out.stdout).context("parse mel preprocess json")?;
    let n_mels = payload["n_mels"].as_u64().context("n_mels")? as usize;
    let n_frames = payload["n_frames"].as_u64().context("n_frames")? as usize;
    let mel: Vec<f32> = payload["mel"]
        .as_array()
        .context("mel")?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .collect();
    let tokens: Vec<u32> = payload["tokens"]
        .as_array()
        .context("tokens")?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u32)
        .collect();
    Ok((
        MelSpectrogram {
            n_mels,
            n_frames,
            data: mel,
        },
        tokens,
    ))
}

pub fn pcm_to_mel(_cfg: &VoxtralAudioConfig, _pcm: &[f32]) -> Result<MelSpectrogram> {
    bail!("use pcm_to_mel_and_prompt(model_dir, wav, language) for HF-compatible mel features")
}

pub fn mel_from_flat(n_mels: usize, n_frames: usize, data: Vec<f32>) -> Result<MelSpectrogram> {
    ensure!(
        data.len() == n_mels * n_frames,
        "mel flat len {} != {n_mels}×{n_frames}",
        data.len()
    );
    Ok(MelSpectrogram {
        n_mels,
        n_frames,
        data,
    })
}
