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

//! Shared base types for vision-language and omni runners (PLAN.md M7).
//!
//! `rlx-qwen3-vl`, `rlx-lfm-vl`, and `rlx-nemotron-omni` all need the
//! same shape of plumbing: a per-image preprocessor (resize +
//! patchify), a vision-tower trait, an MLP projector trait, and a
//! multimodal turn interleaver that mixes image / text / (audio)
//! into a single LM token stream. This crate hosts those traits so
//! the family crates stay thin.
//!
//! **Status:** TYPE SKELETON. The traits and supporting structs are
//! in place; implementations land alongside the per-family crates as
//! M7 progresses.

use anyhow::Result;

/// Modality tag for one chunk of a multimodal prompt. Lives next to
/// the LM token stream so the runner knows when to invoke the vision
/// tower / audio encoder instead of consuming raw token ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    Text,
    Image,
    Audio,
}

/// One image as the preprocessor sees it after resize + patchify.
/// `patches.len() == grid_h * grid_w * channels * patch_h * patch_w`
/// — the exact layout depends on the family.
#[derive(Debug, Clone)]
pub struct ImagePatches {
    pub patches: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
    pub patch_h: usize,
    pub patch_w: usize,
    pub channels: usize,
}

impl ImagePatches {
    pub fn num_patches(&self) -> usize {
        self.grid_h * self.grid_w
    }
    pub fn patch_dim(&self) -> usize {
        self.channels * self.patch_h * self.patch_w
    }
}

/// Image preprocessor. Implementations resize/letterbox/normalise per
/// the family's training pipeline (Qwen3-VL uses SigLIP norms,
/// LFM2.5-VL uses its own, etc.).
pub trait ImagePreprocessor: Send {
    fn preprocess_path(&self, path: &std::path::Path) -> Result<ImagePatches>;
    fn preprocess_bytes(&self, bytes: &[u8]) -> Result<ImagePatches>;
}

/// Vision tower — embeds patches into the model's hidden dim.
/// Output shape is `[num_patches, hidden]`.
pub trait VisionTower: Send {
    fn embed(&mut self, patches: &ImagePatches) -> Result<Vec<f32>>;
    fn hidden_size(&self) -> usize;
}

/// Projector — maps vision-tower embeddings into the LM's embedding
/// space (so they slot in next to text token embeddings). Typically
/// a 2-layer MLP with GeLU.
pub trait Projector: Send {
    fn project(&mut self, vision_embed: &[f32], num_patches: usize) -> Result<Vec<f32>>;
    fn output_dim(&self) -> usize;
}

/// Audio encoder for omni models. Mel features → hidden embeddings.
/// Reuse `rlx-whisper`'s mel encoder where possible — this trait is
/// the contract a family crate adapts to.
pub trait AudioEncoder: Send {
    fn embed_audio(&mut self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>>;
    fn hidden_size(&self) -> usize;
}

/// Multimodal prompt — turn-ordered list of `(modality, payload)`
/// chunks. The runner consumes this and assembles the LM token
/// stream by interleaving text token ids with image/audio embeddings
/// after passing each non-text chunk through the relevant
/// encoder + projector.
#[derive(Debug, Clone, Default)]
pub struct MultimodalPrompt {
    pub chunks: Vec<PromptChunk>,
}

#[derive(Debug, Clone)]
pub enum PromptChunk {
    /// Raw LM token ids (caller already ran the chat template +
    /// tokenizer on the text portion).
    Text(Vec<u32>),
    /// Preprocessed image patches.
    Image(ImagePatches),
    /// PCM-f32 audio at the given sample rate.
    Audio { samples: Vec<f32>, sample_rate: u32 },
}

impl MultimodalPrompt {
    pub fn push(&mut self, chunk: PromptChunk) {
        self.chunks.push(chunk);
    }
    pub fn is_text_only(&self) -> bool {
        self.chunks
            .iter()
            .all(|c| matches!(c, PromptChunk::Text(_)))
    }
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }
}
