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

//! Multimodal RoPE (mRoPE) helpers for Qwen2.5-VL.

use crate::config::Qwen25VlLmConfig;
use rlx_flow::rope;

/// Normalise HF / GGUF section list to four ints for `ggml_rope_multi`.
pub fn mrope_sections4(sections: &[usize]) -> [usize; 4] {
    rope::mrope_sections4(sections)
}

/// Text modality default: `[p, p, p, 0]` (llama.cpp token batches).
pub fn text_section_pos(token_pos: usize) -> [usize; 4] {
    [token_pos, token_pos, token_pos, 0]
}

/// Decoder MRoPE position for vision token `i` (llama.cpp `mtmd_image_tokens_get_decoder_pos`).
pub fn image_decoder_pos(nx: usize, ny: usize, pos_0: usize, i: usize) -> [usize; 4] {
    let _ = ny;
    [pos_0, pos_0 + (i % nx), pos_0 + (i / nx), 0]
}

/// Positions consumed by one image chunk in the decoder (M-RoPE path).
pub fn image_chunk_n_pos(nx: usize, ny: usize) -> usize {
    nx.max(ny)
}

pub fn mrope_row_for_sections(
    cfg: &Qwen25VlLmConfig,
    section_positions: [usize; 4],
    head_half: usize,
) -> (Vec<f32>, Vec<f32>) {
    rope::mrope_row_for_sections(
        cfg.lm.rope_theta,
        cfg.n_rot(),
        cfg.mrope_sections,
        section_positions,
        head_half,
    )
}

pub fn build_mrope_tables(
    cfg: &Qwen25VlLmConfig,
    max_pos: usize,
    head_half: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n_rot = cfg.n_rot();
    let half_rot = n_rot / 2;
    assert!(n_rot.is_multiple_of(2), "n_rot must be even, got {n_rot}");
    assert!(
        head_half >= half_rot,
        "head_half ({head_half}) must cover n_rot/2 ({half_rot})"
    );

    let mut cos = vec![0f32; max_pos * head_half];
    let mut sin = vec![0f32; max_pos * head_half];
    for pos in 0..max_pos {
        let (c_row, s_row) = mrope_row_for_sections(cfg, text_section_pos(pos), head_half);
        let row = pos * head_half;
        cos[row..row + head_half].copy_from_slice(&c_row);
        sin[row..row + head_half].copy_from_slice(&s_row);
    }
    (cos, sin)
}

/// Per-token cos/sin rows for prefill when section positions differ from `[p,p,p,0]`.
pub fn mrope_prefill_feeds(
    cfg: &Qwen25VlLmConfig,
    seq: usize,
    section_positions: Option<&[[usize; 4]]>,
    head_half: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0f32; seq * head_half];
    let mut sin = vec![0f32; seq * head_half];
    for t in 0..seq {
        let sec = section_positions
            .and_then(|s| s.get(t).copied())
            .unwrap_or_else(|| text_section_pos(t));
        let (c_row, s_row) = mrope_row_for_sections(cfg, sec, head_half);
        let row = t * head_half;
        cos[row..row + head_half].copy_from_slice(&c_row);
        sin[row..row + head_half].copy_from_slice(&s_row);
    }
    (cos, sin)
}

pub fn mrope_slice_at_pos(
    cfg: &Qwen25VlLmConfig,
    pos: usize,
    head_half: usize,
) -> (Vec<f32>, Vec<f32>) {
    mrope_row_for_sections(cfg, text_section_pos(pos), head_half)
}

/// Build per-token MRoPE section positions for text + vision + text layout.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth;

    #[test]
    fn multimodal_sections_length_matches_token_count() {
        let sec = build_multimodal_mrope_sections(2, 3, 2, 1, 0);
        assert_eq!(sec.len(), 2 + 6 + 1);
    }

    #[test]
    fn mrope_tables_have_expected_shape() {
        let cfg = synth::tiny_lm_cfg();
        let (cos, sin) = build_mrope_tables(&cfg, 4, cfg.head_half());
        assert_eq!(cos.len(), 4 * cfg.head_half());
        assert_eq!(sin.len(), 4 * cfg.head_half());
    }
}
