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

//! Multimodal prompt assembly for Qwen3-VL (MEDIA_MARKER → vision embeds).

use anyhow::{Result, bail, ensure};

pub const VISION_START: &str = "<|vision_start|>";
pub const VISION_END: &str = "<|vision_end|>";
pub const IMAGE_PAD: &str = "<|image_pad|>";
/// Single-image placeholder in user prompts (replaced by vision token span).
pub const MEDIA_MARKER: &str = "<__media__>";

#[derive(Debug, Clone)]
pub struct VisionEncodeOutput {
    pub embeddings: Vec<f32>,
    pub n_tokens: usize,
    pub grid_x: usize,
    pub grid_y: usize,
}

impl VisionEncodeOutput {
    pub fn from_flat(embeddings: Vec<f32>, n_embd: usize) -> Result<Self> {
        ensure!(n_embd > 0, "n_embd must be > 0");
        ensure!(
            embeddings.len() % n_embd == 0,
            "vision embeds len {} not divisible by n_embd {n_embd}",
            embeddings.len()
        );
        let n_tokens = embeddings.len() / n_embd;
        let side = (n_tokens as f64).sqrt().round() as usize;
        let (grid_x, grid_y) = if side * side == n_tokens {
            (side, side)
        } else {
            (n_tokens, 1)
        };
        Ok(Self {
            embeddings,
            n_tokens,
            grid_x,
            grid_y,
        })
    }
}

#[derive(Debug, Clone)]
pub struct MultimodalPrefill {
    pub hidden: Vec<f32>,
    pub seq: Vec<u32>,
    pub vision_start_idx: usize,
    pub n_vision_tokens: usize,
}

pub fn merge_text_and_vision_embd(
    token_embd: &[f32],
    n_embd: usize,
    input_ids: &[u32],
    vision_embd: &[f32],
    vision_start_idx: usize,
    n_vision: usize,
) -> Vec<f32> {
    assert_eq!(token_embd.len(), input_ids.len() * n_embd);
    assert_eq!(vision_embd.len(), n_vision * n_embd);
    let mut out = token_embd.to_vec();
    for t in 0..n_vision {
        let dst = (vision_start_idx + t) * n_embd;
        let src = t * n_embd;
        out[dst..dst + n_embd].copy_from_slice(&vision_embd[src..src + n_embd]);
    }
    out
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
            n_embd,
            &seq,
            &self.vision.embeddings,
            vision_start_idx,
            n_vision,
        );

        Ok(MultimodalPrefill {
            hidden,
            seq,
            vision_start_idx,
            n_vision_tokens: n_vision,
        })
    }
}

pub fn normalize_media_prompt(prompt: &str) -> String {
    if prompt.contains(MEDIA_MARKER) {
        return prompt.to_string();
    }
    let mut p = prompt.to_string();
    if !p.is_empty() && !p.ends_with(|c: char| c.is_whitespace()) {
        p.push(' ');
    }
    p.push_str(MEDIA_MARKER);
    p
}
