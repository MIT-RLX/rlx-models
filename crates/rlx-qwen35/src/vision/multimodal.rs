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

//! Multimodal prompt assembly — MRoPE sections + embedding merge.

pub const VISION_START: &str = "<|vision_start|>";
pub const VISION_END: &str = "<|vision_end|>";
pub const MEDIA_MARKER: &str = "<__media__>";

/// Decoder MRoPE position for vision token `i`.
///
/// HF `mrope_section` is `[temporal, height, width]`. Tokens are row-major over
/// the output grid (`i / nx` = row/y, `i % nx` = col/x), matching Qwen2/3-VL
/// `get_rope_index` spatial layout (not llama.cpp's swapped sections).
pub fn image_decoder_pos(nx: usize, ny: usize, pos_0: usize, i: usize) -> [usize; 4] {
    let _ = ny;
    [pos_0, pos_0 + (i / nx), pos_0 + (i % nx), 0]
}

/// Positions consumed by one image chunk in the decoder (M-RoPE path).
pub fn image_chunk_n_pos(nx: usize, ny: usize) -> usize {
    nx.max(ny)
}

/// Build per-token MRoPE section positions for a text + vision + text layout.
pub fn build_multimodal_mrope_sections(
    text_len_before: usize,
    nx: usize,
    ny: usize,
    text_len_after: usize,
    text_start_pos: usize,
) -> Vec<[usize; 4]> {
    let mut out = Vec::with_capacity(text_len_before + nx * ny + text_len_after);
    for i in 0..text_len_before {
        let p = text_start_pos + i;
        out.push([p, p, p, 0]);
    }
    let pos_0 = text_start_pos + text_len_before;
    let n_vision = nx * ny;
    for i in 0..n_vision {
        out.push(image_decoder_pos(nx, ny, pos_0, i));
    }
    let after_start = pos_0 + image_chunk_n_pos(nx, ny);
    for i in 0..text_len_after {
        let p = after_start + i;
        out.push([p, p, p, 0]);
    }
    out
}

/// Replace the vision placeholder span in a flat token embedding buffer.
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

/// Assembled multimodal prefill payload for the Qwen3.5 decoder.
#[derive(Debug, Clone)]
pub struct MultimodalPrefill {
    /// Flat `[seq * n_embd]` hidden states (text embeds + vision rows spliced in).
    pub hidden: Vec<f32>,
    /// Per-token MRoPE section positions.
    pub mrope_sections: Vec<[usize; 4]>,
    /// Index of the last token (for last-logits-only decode).
    pub last_token_idx: usize,
    /// Token ids (vision rows may be placeholder ids).
    pub seq: Vec<u32>,
}

/// High-level prompt wrapper: tokenize + splice vision embeddings.
pub struct MultimodalPrompt<'a> {
    pub prompt: &'a str,
    pub vision: &'a super::encoder::VisionEncodeOutput,
}

impl<'a> MultimodalPrompt<'a> {
    /// Tokenize `prompt` (must contain one [`MEDIA_MARKER`]), merge vision
    /// embeddings, and build MRoPE section positions.
    pub fn assemble(
        &self,
        mut tokenizer: impl FnMut(&str) -> anyhow::Result<Vec<u32>>,
        token_embd_table: &[f32],
        n_embd: usize,
        text_start_pos: usize,
    ) -> anyhow::Result<MultimodalPrefill> {
        let parts: Vec<&str> = self.prompt.split(MEDIA_MARKER).collect();
        if parts.len() != 2 {
            return Err(anyhow::anyhow!(
                "prompt must contain exactly one `{MEDIA_MARKER}` marker"
            ));
        }
        let before_ids = tokenizer(parts[0])?;
        let start_ids = tokenizer(VISION_START)?;
        let end_ids = tokenizer(VISION_END)?;
        let after_ids = tokenizer(parts[1])?;
        let n_vision = self.vision.n_tokens;
        let mut seq = before_ids.clone();
        seq.extend(start_ids.iter().copied());
        let n_start = start_ids.len();
        let vision_start = seq.len();
        seq.extend(std::iter::repeat_n(0u32, n_vision));
        seq.extend(end_ids.iter().copied());
        let n_end = end_ids.len();
        seq.extend(after_ids.iter().copied());

        let vocab = token_embd_table.len() / n_embd;
        let mut hidden = Vec::with_capacity(seq.len() * n_embd);
        for &tid in &seq {
            let t = tid as usize;
            if t >= vocab {
                hidden.extend(std::iter::repeat_n(0.0f32, n_embd));
            } else {
                let off = t * n_embd;
                hidden.extend_from_slice(&token_embd_table[off..off + n_embd]);
            }
        }
        hidden = merge_text_and_vision_embd(
            &hidden,
            vocab,
            n_embd,
            &seq,
            &self.vision.embeddings,
            vision_start,
            n_vision,
        );

        let text_before = before_ids.len() + n_start;
        let text_after = n_end + after_ids.len();
        let mrope_sections = build_multimodal_mrope_sections(
            text_before,
            self.vision.grid_x,
            self.vision.grid_y,
            text_after,
            text_start_pos,
        );

        Ok(MultimodalPrefill {
            hidden,
            mrope_sections,
            last_token_idx: seq.len().saturating_sub(1),
            seq,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_chunk_n_pos_is_max_grid() {
        assert_eq!(image_chunk_n_pos(4, 6), 6);
    }

    #[test]
    fn mrope_sections_length_matches_tokens() {
        let sec = build_multimodal_mrope_sections(2, 2, 2, 1, 0);
        assert_eq!(sec.len(), 2 + 4 + 1);
        // Vision token i=0 at pos_0=2: [t, h=0, w=0]
        assert_eq!(sec[2], [2, 2, 2, 0]);
        // i=1 (row 0, col 1): width section advances
        assert_eq!(sec[3], [2, 2, 3, 0]);
    }
}
