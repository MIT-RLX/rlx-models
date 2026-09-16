// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Optional ONNX Runtime reference path (parity only — not the ship default).

use anyhow::{Context, Result, bail};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

use crate::fbank::log_mel_fbank;
use crate::{EMBED_DIM, l2_normalize};

/// ORT session over the upstream packed WeSpeaker ONNX (variable-T).
pub struct OrtWeSpeaker {
    session: Session,
}

impl OrtWeSpeaker {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        anyhow::ensure!(path.is_file(), "missing WeSpeaker ONNX: {}", path.display());
        let session = Session::builder()
            .context("ort session builder")?
            .commit_from_file(path)
            .with_context(|| format!("load WeSpeaker ONNX {}", path.display()))?;
        Ok(Self { session })
    }

    pub fn embed_pcm(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        let frames = log_mel_fbank(pcm);
        if frames.is_empty() {
            bail!("pcm too short for WeSpeaker fbank");
        }
        self.embed_fbank(&frames)
    }

    pub fn embed_fbank(&mut self, frames: &[[f32; 80]]) -> Result<Vec<f32>> {
        let t = frames.len();
        let mut fb_data = Vec::with_capacity(t * 80);
        for row in frames {
            fb_data.extend_from_slice(row);
        }
        let weights = vec![1.0f32; t];
        let fb_t = Tensor::<f32>::from_array(([1usize, t, 80], fb_data))?;
        let w_t = Tensor::<f32>::from_array(([1usize, t], weights))?;
        let outputs = self.session.run(ort::inputs![
            "fbank" => fb_t,
            "weights" => w_t,
        ])?;
        let (_shape, data) = outputs["embedding"].try_extract_tensor::<f32>()?;
        if data.len() < EMBED_DIM {
            bail!("embedding too short: {}", data.len());
        }
        let mut emb = data[..EMBED_DIM].to_vec();
        l2_normalize(&mut emb);
        Ok(emb)
    }
}
