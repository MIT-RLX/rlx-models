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

//! WAV dataset + manifest writer.

use anyhow::{Context, Result, bail, ensure};
use rand::thread_rng;
use rlx_voxtral_tts::codec::load_mono_wav;
use rlx_voxtral_tts::config::CodecArgs;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WavManifest {
    pub sample_rate: u32,
    pub files: Vec<WavEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WavEntry {
    pub path: String,
    pub duration_sec: f32,
    #[serde(default)]
    pub transcript: Option<String>,
}

pub struct WavBatch {
    pub pcm: Vec<f32>,
    pub n_patches: usize,
    pub transcript: Option<String>,
}

pub struct WavDataset {
    entries: Vec<PathBuf>,
    transcripts: Vec<Option<String>>,
    sample_rate: u32,
    patch_size: usize,
    max_samples: usize,
}

impl WavDataset {
    pub fn from_dir(wav_dir: &Path, cfg: &CodecArgs, max_audio_sec: f32) -> Result<Self> {
        let mut entries = Vec::new();
        for entry in fs::read_dir(wav_dir).with_context(|| format!("read {}", wav_dir.display()))? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "wav") {
                entries.push(path);
            }
        }
        entries.sort();
        if entries.is_empty() {
            bail!("no .wav files under {}", wav_dir.display());
        }
        let n = entries.len();
        Ok(Self {
            entries,
            transcripts: vec![None; n],
            sample_rate: cfg.sampling_rate as u32,
            patch_size: cfg.pretransform_patch_size,
            max_samples: (cfg.sampling_rate as f32 * max_audio_sec).ceil() as usize,
        })
    }

    pub fn from_manifest(path: &Path, cfg: &CodecArgs, max_audio_sec: f32) -> Result<Self> {
        let raw = fs::read_to_string(path)?;
        let manifest: WavManifest = serde_json::from_str(&raw)?;
        ensure!(
            manifest.sample_rate == cfg.sampling_rate as u32,
            "manifest sample_rate mismatch"
        );
        let base = path.parent().unwrap_or(Path::new("."));
        let entries: Vec<PathBuf> = manifest.files.iter().map(|e| base.join(&e.path)).collect();
        let transcripts = manifest
            .files
            .iter()
            .map(|e| e.transcript.clone())
            .collect();
        Ok(Self {
            entries,
            transcripts,
            sample_rate: manifest.sample_rate,
            patch_size: cfg.pretransform_patch_size,
            max_samples: (cfg.sampling_rate as f32 * max_audio_sec).ceil() as usize,
        })
    }

    pub fn write_manifest(&self, out: &Path) -> Result<()> {
        let mut files = Vec::new();
        for path in &self.entries {
            let pcm = load_mono_wav(path, self.sample_rate)?;
            files.push(WavEntry {
                path: path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("clip.wav")
                    .to_string(),
                duration_sec: pcm.len() as f32 / self.sample_rate as f32,
                transcript: None,
            });
        }
        let manifest = WavManifest {
            sample_rate: self.sample_rate,
            files,
        };
        fs::write(out, serde_json::to_string_pretty(&manifest)?)?;
        Ok(())
    }

    pub fn sample_batch(&self) -> Result<WavBatch> {
        let mut rng = thread_rng();
        let idx = rng.gen_range(0..self.entries.len());
        let path = &self.entries[idx];
        let mut pcm = load_mono_wav(path, self.sample_rate)?;
        if pcm.len() > self.max_samples {
            let start = rng.gen_range(0..pcm.len() - self.max_samples);
            pcm = pcm[start..start + self.max_samples].to_vec();
        }
        let rem = pcm.len() % self.patch_size;
        if rem != 0 {
            pcm.extend(std::iter::repeat_n(0f32, self.patch_size - rem));
        }
        let n_patches = pcm.len() / self.patch_size;
        Ok(WavBatch {
            pcm,
            n_patches,
            transcript: self.transcripts.get(idx).and_then(|t| t.clone()),
        })
    }

    pub fn patches_to_ncl(pcm: &[f32], patch_size: usize) -> Vec<f32> {
        let n_patches = pcm.len() / patch_size;
        let mut out = vec![0f32; patch_size * n_patches];
        for pi in 0..n_patches {
            for ic in 0..patch_size {
                out[ic * n_patches + pi] = pcm[pi * patch_size + ic];
            }
        }
        out
    }

    /// Inverse of `patches_to_ncl` — recovers time-domain PCM from NCL layout.
    pub fn ncl_to_pcm(ncl: &[f32], patch_size: usize) -> Vec<f32> {
        if patch_size == 0 {
            return Vec::new();
        }
        let n_patches = ncl.len() / patch_size;
        let mut pcm = vec![0f32; n_patches * patch_size];
        for pi in 0..n_patches {
            for ic in 0..patch_size {
                pcm[pi * patch_size + ic] = ncl[ic * n_patches + pi];
            }
        }
        pcm
    }
}

use rand::Rng;

pub fn build_manifest_from_dir(wav_dir: &Path, out: &Path, sample_rate: u32) -> Result<()> {
    let base = out.parent().unwrap_or(Path::new("."));
    let rel_wav_dir = wav_dir.strip_prefix(base).unwrap_or(wav_dir);

    let mut files = Vec::new();
    for entry in fs::read_dir(wav_dir).with_context(|| format!("read {}", wav_dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "wav") {
            continue;
        }
        let pcm = load_mono_wav(&path, sample_rate)?;
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("clip.wav");
        let rel_path = rel_wav_dir.join(name);
        files.push(WavEntry {
            path: rel_path.to_string_lossy().to_string(),
            duration_sec: pcm.len() as f32 / sample_rate as f32,
            transcript: None,
        });
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    if files.is_empty() {
        bail!("no .wav files under {}", wav_dir.display());
    }
    let manifest = WavManifest { sample_rate, files };
    fs::write(out, serde_json::to_string_pretty(&manifest)?)?;
    Ok(())
}
