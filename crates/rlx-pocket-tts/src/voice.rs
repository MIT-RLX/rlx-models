// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Voice loading.
//!
//! The `Verylicious/pocket-tts-ungated` mirror ships voices as single-tensor
//! safetensors files: `audio_prompt: F32 [1, 125, 1024]`. The 125 frames at
//! 12.5 Hz represent 10 seconds of pre-projected voice conditioning, ready to
//! be fed straight into the FlowLM backbone.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use ndarray::Array2;

use crate::weights::WeightFile;

#[derive(Debug, Clone)]
pub struct Voice {
    /// Conditioning sequence `[T_voice, d_model]` ready for the FlowLM
    /// backbone (post-projection). Typically `T_voice = 125` (10 s @ 12.5 Hz).
    pub conditioning: Array2<f32>,
}

impl Voice {
    /// Load a voice from a single-tensor safetensors file containing
    /// `audio_prompt`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let wf = WeightFile::open(path.as_ref())
            .with_context(|| format!("open voice {}", path.as_ref().display()))?;
        let (data, shape) = wf.get_f32("audio_prompt").or_else(|_| {
            // Some mirrors use `conditioning`; try that as a fallback.
            wf.get_f32("conditioning")
        })?;
        let (t, d) = match shape.as_slice() {
            [1, t, d] => (*t, *d),
            [t, d] => (*t, *d),
            other => return Err(anyhow!("unexpected voice shape {other:?}")),
        };
        Ok(Self {
            conditioning: Array2::from_shape_vec((t, d), data)?,
        })
    }

    pub fn num_frames(&self) -> usize {
        self.conditioning.shape()[0]
    }

    pub fn embed_dim(&self) -> usize {
        self.conditioning.shape()[1]
    }
}
