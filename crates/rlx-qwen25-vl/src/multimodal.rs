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

//! Multimodal prompt assembly — ChatML + vision placeholder tokens.

use crate::mrope::build_multimodal_mrope_sections;
use crate::vision::VisionEncodeOutput;
use anyhow::{Result, bail, ensure};

pub const VISION_START: &str = "<|vision_start|>";
pub const VISION_END: &str = "<|vision_end|>";
pub const IMAGE_PAD: &str = "<|image_pad|>";
/// Single-image placeholder in user prompts (replaced by vision token span).
pub const MEDIA_MARKER: &str = "<__media__>";

pub fn merge_text_and_vision_embd(
    token_embd: &[f32],
    vocab: usize,
    n_embd: usize,
    input_ids: &[u32],
    vision_embd: &[f32],
    vision_start_idx: usize,
    n_vision: usize,
) -> Vec<f32> {
    assert_eq!(token_embd.len(), input_ids.len() * n_embd);
    assert_eq!(vision_embd.len(), n_vision * n_embd);
    let _ = vocab;
    let seq = input_ids.len();
    assert!(vision_start_idx + n_vision <= seq);

    let mut out = token_embd.to_vec();
    for t in 0..n_vision {
        let dst = (vision_start_idx + t) * n_embd;
        let src = t * n_embd;
        out[dst..dst + n_embd].copy_from_slice(&vision_embd[src..src + n_embd]);
    }
    out
}

#[derive(Debug, Clone)]
pub struct MultimodalPrefill {
    pub hidden: Vec<f32>,
    pub mrope_sections: Vec<[usize; 4]>,
    pub last_token_idx: usize,
    pub seq: Vec<u32>,
    /// First `<|image_pad|>` index in `seq` (KV cache key index).
    pub vision_start_idx: usize,
    pub n_vision_tokens: usize,
}

pub struct MultimodalPrompt<'a> {
    pub prompt: &'a str,
    pub vision: &'a VisionEncodeOutput,
}

impl<'a> MultimodalPrompt<'a> {
    pub fn assemble(
        &self,
        mut tokenizer: impl FnMut(&str) -> Result<Vec<u32>>,
        token_embd_table: &[f32],
        n_embd: usize,
        text_start_pos: usize,
    ) -> Result<MultimodalPrefill> {
        let parts: Vec<&str> = self.prompt.split(MEDIA_MARKER).collect();
        if parts.len() != 2 {
            bail!("prompt must contain exactly one `{MEDIA_MARKER}` marker");
        }
        let before_ids = tokenizer(parts[0])?;
        let start_ids = tokenizer(VISION_START)?;
        let end_ids = tokenizer(VISION_END)?;
        let after_ids = tokenizer(parts[1])?;

        let n_vision = self.vision.n_tokens;
        let pad_ids = vec![0u32; n_vision];
        let mut seq = Vec::new();
        seq.extend_from_slice(&before_ids);
        seq.extend_from_slice(&start_ids);
        seq.extend_from_slice(&pad_ids);
        seq.extend_from_slice(&end_ids);
        seq.extend_from_slice(&after_ids);

        let vision_start_idx = before_ids.len() + start_ids.len();
        ensure!(
            vision_start_idx + n_vision + end_ids.len() <= seq.len(),
            "vision span does not fit assembled token sequence"
        );

        let vocab = token_embd_table.len() / n_embd;
        let mut hidden = Vec::with_capacity(seq.len() * n_embd);
        for &tok in &seq {
            let row = tok as usize;
            ensure!(row < vocab, "token id {tok} out of vocab range {vocab}");
            let off = row * n_embd;
            hidden.extend_from_slice(&token_embd_table[off..off + n_embd]);
        }
        hidden = merge_text_and_vision_embd(
            &hidden,
            vocab,
            n_embd,
            &seq,
            &self.vision.embeddings,
            vision_start_idx,
            n_vision,
        );

        let text_before = before_ids.len() + start_ids.len();
        let text_after = end_ids.len() + after_ids.len();
        let mrope_sections = build_multimodal_mrope_sections(
            text_before,
            self.vision.grid_x,
            self.vision.grid_y,
            text_after,
            text_start_pos,
        );
        ensure!(
            mrope_sections.len() == seq.len(),
            "mrope sections {} != seq {}",
            mrope_sections.len(),
            seq.len()
        );

        Ok(MultimodalPrefill {
            hidden,
            mrope_sections,
            last_token_idx: seq.len().saturating_sub(1),
            seq,
            vision_start_idx,
            n_vision_tokens: n_vision,
        })
    }
}

/// Assemble prefill from a fixed HF token sequence (parity / replay).
pub fn assemble_from_token_ids(
    input_ids: &[u32],
    vision_start_idx: usize,
    n_vision: usize,
    vision: &VisionEncodeOutput,
    token_embd_table: &[f32],
    n_embd: usize,
    text_start_pos: usize,
) -> Result<MultimodalPrefill> {
    let seq = input_ids.to_vec();
    ensure!(
        vision_start_idx + n_vision <= seq.len(),
        "vision span {vision_start_idx}+{n_vision} exceeds seq {}",
        seq.len()
    );
    ensure!(
        vision.n_tokens == n_vision,
        "vision token count {} != reference {n_vision}",
        vision.n_tokens
    );

    let vocab = token_embd_table.len() / n_embd;
    let mut hidden = Vec::with_capacity(seq.len() * n_embd);
    for &tok in &seq {
        let row = tok as usize;
        ensure!(row < vocab, "token id {tok} out of vocab range {vocab}");
        let off = row * n_embd;
        hidden.extend_from_slice(&token_embd_table[off..off + n_embd]);
    }
    hidden = merge_text_and_vision_embd(
        &hidden,
        vocab,
        n_embd,
        &seq,
        &vision.embeddings,
        vision_start_idx,
        n_vision,
    );

    let text_len_before = vision_start_idx;
    let text_len_after = seq.len() - vision_start_idx - n_vision;
    let mrope_sections = build_multimodal_mrope_sections(
        text_len_before,
        vision.grid_x,
        vision.grid_y,
        text_len_after,
        text_start_pos,
    );
    ensure!(
        mrope_sections.len() == seq.len(),
        "mrope sections {} != seq {}",
        mrope_sections.len(),
        seq.len()
    );

    Ok(MultimodalPrefill {
        hidden,
        mrope_sections,
        last_token_idx: seq.len().saturating_sub(1),
        seq,
        vision_start_idx,
        n_vision_tokens: n_vision,
    })
}
