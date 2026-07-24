// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Optional ONNX Runtime parity helpers (feature `onnx` only).

use anyhow::{Context, Result};
use std::path::Path;

pub struct OnnxWakeBundle {
    mel: ort::session::Session,
    embed: ort::session::Session,
    phrase: ort::session::Session,
}

impl OnnxWakeBundle {
    pub fn load(dir: &Path) -> Result<Self> {
        let mel = ort::session::Session::builder()?
            .commit_from_file(dir.join("melspectrogram.onnx"))
            .with_context(|| format!("load melspectrogram.onnx in {}", dir.display()))?;
        let embed = ort::session::Session::builder()?
            .commit_from_file(dir.join("embedding_model.onnx"))
            .with_context(|| format!("load embedding_model.onnx in {}", dir.display()))?;
        let phrase = ort::session::Session::builder()?
            .commit_from_file(dir.join("phrase.onnx"))
            .with_context(|| format!("load phrase.onnx in {}", dir.display()))?;
        Ok(Self { mel, embed, phrase })
    }

    /// Soft availability probe for tests.
    pub fn try_load(dir: &Path) -> Option<Self> {
        Self::load(dir).ok()
    }

    pub fn sessions_ok(&self) -> bool {
        !self.mel.inputs().is_empty()
            && !self.embed.inputs().is_empty()
            && !self.phrase.inputs().is_empty()
    }
}
