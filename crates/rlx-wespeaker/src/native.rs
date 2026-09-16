// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Native RLX WeSpeaker (TinyModel / rlx-onnx-import).

use anyhow::{Context, Result, bail};
use rlx_runtime::{DType, Device};
use rlx_tiny_tts::config::BundleConfig;
use rlx_tiny_tts::model::TinyModel;
use std::path::{Path, PathBuf};

use crate::fbank::log_mel_fbank;
use crate::{EMBED_DIM, l2_normalize};

/// Fixed fbank frames baked into the shipped RLX graph (~1.5 s @ 16 kHz).
pub const FIXED_FRAMES: usize = 148;

fn bundle_cfg() -> BundleConfig {
    BundleConfig {
        model: String::new(),
        sample_rate: 16_000,
        add_blank: false,
        language: "EN".into(),
        speakers: Default::default(),
        default_speaker: None,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        length_scale: 1.0,
        inter_channels: 0,
        gin_channels: 0,
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Pad / truncate fbank frames to [`FIXED_FRAMES`].
pub fn fit_frames(frames: &[[f32; 80]]) -> Vec<[f32; 80]> {
    let mut out = vec![[0f32; 80]; FIXED_FRAMES];
    let n = frames.len().min(FIXED_FRAMES);
    out[..n].copy_from_slice(&frames[..n]);
    out
}

/// Resolve a WeSpeaker model directory from common roots / env.
pub fn resolve_model_dir(search_roots: &[&Path]) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("RLX_WESPEAKER_DIR") {
        let pb = PathBuf::from(p);
        if pb.is_dir() {
            return Some(pb);
        }
    }
    if let Ok(p) = std::env::var("RLX_WESPEAKER_ONNX") {
        // Accept a file path — use its parent (or grandparent if under onnx/).
        let pb = PathBuf::from(p);
        if pb.is_file()
            && let Some(parent) = pb.parent()
        {
            if parent.file_name().and_then(|s| s.to_str()) == Some("onnx") {
                return parent.parent().map(|p| p.to_path_buf());
            }
            return Some(parent.to_path_buf());
        }
    }
    const MARKERS: &[&str] = &[
        "graphs/wespeaker.rlxp",
        "onnx/wespeaker.onnx",
        "wespeaker.onnx",
    ];
    for root in search_roots {
        for marker in MARKERS {
            let p = root.join(marker);
            if p.is_file() {
                return Some(root.to_path_buf());
            }
        }
        // root itself may be …/wespeaker-voxceleb-resnet34-LM
        let nested = root.join("wespeaker-voxceleb-resnet34-LM");
        for marker in MARKERS {
            if nested.join(marker).is_file() {
                return Some(nested);
            }
        }
    }
    None
}

/// WeSpeaker embedder on an RLX [`Device`].
pub struct WeSpeaker {
    model: TinyModel,
    device: Device,
    graph: String,
}

impl WeSpeaker {
    /// Open a WeSpeaker model dir containing `graphs/wespeaker.rlxp` and/or
    /// `onnx/wespeaker.onnx` (native import).
    pub fn open_on(dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let dir = dir.as_ref();
        anyhow::ensure!(
            dir.join("graphs/wespeaker.rlxp").is_file()
                || dir.join("onnx/wespeaker.onnx").is_file()
                || dir.join("wespeaker.onnx").is_file(),
            "WeSpeaker RLX weights not found under {} (need graphs/wespeaker.rlxp or onnx/wespeaker.onnx)",
            dir.display()
        );
        let device = rlx_tiny_tts::resolve_tts_device(device);
        Ok(Self {
            model: TinyModel::new(dir.to_path_buf(), bundle_cfg()),
            device,
            graph: "wespeaker".into(),
        })
    }

    /// Open on CPU (convenience).
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        Self::open_on(dir, Device::Cpu)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Embed 16 kHz mono PCM → L2-normalized 256-d vector.
    pub fn embed_pcm(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        let frames = log_mel_fbank(pcm);
        if frames.is_empty() {
            bail!("pcm too short for WeSpeaker fbank (need ≥25 ms @ 16 kHz)");
        }
        self.embed_fbank(&frames)
    }

    /// Embed precomputed fbank `[T, 80]` (padded/truncated to [`FIXED_FRAMES`]).
    pub fn embed_fbank(&mut self, frames: &[[f32; 80]]) -> Result<Vec<f32>> {
        let fitted = fit_frames(frames);
        let mut flat = Vec::with_capacity(FIXED_FRAMES * 80);
        for row in &fitted {
            flat.extend_from_slice(row);
        }
        let bytes = f32_bytes(&flat);
        let outs = self
            .model
            .run_named(
                &self.graph,
                self.device,
                FIXED_FRAMES,
                &[],
                &[("fbank", bytes.as_slice(), DType::F32)],
            )
            .with_context(|| {
                format!(
                    "WeSpeaker native run on {:?} (graph={})",
                    self.device, self.graph
                )
            })?;
        let Some((raw, _)) = outs.first() else {
            bail!("WeSpeaker returned no outputs");
        };
        let mut emb = as_f32(raw);
        if emb.len() < EMBED_DIM {
            bail!("embedding too short: {}", emb.len());
        }
        emb.truncate(EMBED_DIM);
        l2_normalize(&mut emb);
        Ok(emb)
    }
}
