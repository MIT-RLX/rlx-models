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

//! Rotating reference-audio + text prompts for LoRA distillation.

use anyhow::{Context, Result, bail};
use rlx_voxtral_tts::VoxtralTtsWeightStore;
use rlx_voxtral_tts::config::VoxtralTtsConfig;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::config::env_flag;
use crate::dataset::WavManifest;
use crate::distill_text::distill_text_for_sample;
use crate::teacher::TeacherCache;

pub struct DistillSample {
    pub inputs: Vec<f32>,
    pub targets: Vec<f32>,
    pub seq: usize,
}

pub struct DistillDataset {
    teacher: TeacherCache,
    wavs: Vec<PathBuf>,
    texts: Vec<String>,
    voice_name: String,
    max_seq: usize,
}

impl DistillDataset {
    pub fn open(
        model_dir: &Path,
        reference_dir: &Path,
        manifest: Option<&Path>,
        max_seq: usize,
        encoder_weights: Option<&Path>,
        epochs: usize,
        steps_per_epoch: usize,
    ) -> Result<Self> {
        let cfg = VoxtralTtsConfig::from_model_dir(model_dir)?;
        let store = VoxtralTtsWeightStore::open(model_dir)?;
        let voice_name = std::env::var("DISTILL_VOICE").unwrap_or_else(|_| "neutral_female".into());
        let default_text =
            std::env::var("DISTILL_TEXT").unwrap_or_else(|_| "Hello, this is my voice.".into());

        let (wavs, texts) = if let Some(m) = manifest {
            load_manifest_samples(m, &default_text)?
        } else {
            load_dir_samples(reference_dir, &default_text)?
        };
        if wavs.is_empty() {
            bail!("no reference wavs under {}", reference_dir.display());
        }

        let mut teacher = TeacherCache::open(&store, &cfg, encoder_weights)?;
        if env_flag("PRECOMPUTE_DISTILL") {
            let started = Instant::now();
            let built =
                teacher.prewarm(&wavs, &texts, &voice_name, max_seq, steps_per_epoch, epochs)?;
            eprintln!(
                "[lora] precomputed {built} distill batches in {:.1}s (cache={})",
                started.elapsed().as_secs_f64(),
                teacher.cached_batches()
            );
        }

        Ok(Self {
            teacher,
            wavs,
            texts,
            voice_name,
            max_seq,
        })
    }

    pub fn len(&self) -> usize {
        self.wavs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.wavs.is_empty()
    }

    pub fn sample(&mut self, step: usize) -> Result<DistillSample> {
        let idx = step % self.wavs.len();
        let transcript = self.texts.get(idx).map(String::as_str);
        let text = distill_text_for_sample(step, idx, transcript);
        let batch = self.teacher.build_batch(
            &text,
            &self.voice_name,
            Some(&self.wavs[idx]),
            self.max_seq,
            idx,
        )?;
        Ok(DistillSample {
            inputs: batch.inputs,
            targets: batch.targets,
            seq: batch.seq,
        })
    }
}

fn load_dir_samples(dir: &Path, default_text: &str) -> Result<(Vec<PathBuf>, Vec<String>)> {
    let mut wavs = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "wav") {
            wavs.push(path);
        }
    }
    wavs.sort();
    let texts = vec![default_text.to_string(); wavs.len()];
    Ok((wavs, texts))
}

fn load_manifest_samples(
    manifest: &Path,
    default_text: &str,
) -> Result<(Vec<PathBuf>, Vec<String>)> {
    let raw = fs::read_to_string(manifest)?;
    let m: WavManifest = serde_json::from_str(&raw)?;
    let base = manifest.parent().unwrap_or(Path::new("."));
    let mut wavs = Vec::new();
    let mut texts = Vec::new();
    for entry in &m.files {
        wavs.push(base.join(&entry.path));
        texts.push(
            entry
                .transcript
                .clone()
                .unwrap_or_else(|| default_text.to_string()),
        );
    }
    Ok((wavs, texts))
}
