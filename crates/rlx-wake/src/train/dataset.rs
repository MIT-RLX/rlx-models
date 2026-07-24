// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Labeled wake clips (16 kHz mono) for RLX training.

use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};

use crate::audio::{SAMPLE_RATE_16K, load_wav_mono_f32, resample_linear, write_wav_mono_f32};
use crate::mel::{MelConfig, MelFrontend};

#[derive(Debug, Clone)]
pub struct LabeledClip {
    pub path: Option<PathBuf>,
    pub pcm: Vec<f32>,
    /// 1.0 = wake present, 0.0 = negative.
    pub label: f32,
}

/// Load `positives/*.wav` (label 1) and `negatives/*.wav` (label 0).
pub fn load_pos_neg_dirs(pos_dir: &Path, neg_dir: &Path) -> Result<Vec<LabeledClip>> {
    let mut out = Vec::new();
    out.extend(load_dir(pos_dir, 1.0)?);
    out.extend(load_dir(neg_dir, 0.0)?);
    if out.is_empty() {
        bail!(
            "no wavs found under {} or {}",
            pos_dir.display(),
            neg_dir.display()
        );
    }
    Ok(out)
}

fn load_dir(dir: &Path, label: f32) -> Result<Vec<LabeledClip>> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut paths: Vec<_> = fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("wav"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    let mut out = Vec::new();
    for path in paths {
        let (sr, pcm) = load_wav_mono_f32(&path)?;
        let pcm = if sr != SAMPLE_RATE_16K {
            resample_linear(&pcm, sr, SAMPLE_RATE_16K)
        } else {
            pcm
        };
        out.push(LabeledClip {
            path: Some(path),
            pcm,
            label,
        });
    }
    Ok(out)
}

/// Mel frames for a clip (frame-major `[n_frames * n_mels]`).
pub fn clip_mel_frames(pcm: &[f32]) -> Vec<f32> {
    let mut mel = MelFrontend::new(MelConfig::default());
    mel.push(pcm)
}

/// Synthetic dataset for CI / quick RLX train checks (no external files).
pub fn synth_pos_neg_dataset(n_pos: usize, n_neg: usize, seconds: f32) -> Vec<LabeledClip> {
    let n = (seconds * SAMPLE_RATE_16K as f32) as usize;
    let mut out = Vec::new();
    for i in 0..n_pos {
        let freq = 220.0 + (i as f32) * 40.0;
        let pcm: Vec<f32> = (0..n)
            .map(|t| {
                let x = t as f32 / SAMPLE_RATE_16K as f32;
                (x * freq * std::f32::consts::TAU).sin() * 0.3
            })
            .collect();
        out.push(LabeledClip {
            path: None,
            pcm,
            label: 1.0,
        });
    }
    for i in 0..n_neg {
        let mut state = 0xABCDu64.wrapping_add(i as u64);
        let pcm: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = (state >> 33) as f32 / u32::MAX as f32;
                (u * 2.0 - 1.0) * 0.02
            })
            .collect();
        out.push(LabeledClip {
            path: None,
            pcm,
            label: 0.0,
        });
    }
    out
}

/// Write a tiny synthetic corpus to disk (optional for demos).
pub fn write_synth_corpus(root: &Path, n_pos: usize, n_neg: usize) -> Result<()> {
    let pos = root.join("positives");
    let neg = root.join("negatives");
    fs::create_dir_all(&pos)?;
    fs::create_dir_all(&neg)?;
    let clips = synth_pos_neg_dataset(n_pos, n_neg, 1.0);
    for (i, c) in clips.iter().enumerate() {
        let dir = if c.label > 0.5 { &pos } else { &neg };
        let path = dir.join(format!("clip_{i:03}.wav"));
        write_wav_mono_f32(&path, SAMPLE_RATE_16K as u32, &c.pcm)?;
    }
    Ok(())
}
