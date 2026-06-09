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

//! RoPE / MRoPE cos/sin tables for Qwen3.5 full-attention layers.
//!
//! Qwen3.5 GGUFs tag `qwen35.rope.dimension_sections` (e.g.
//! `[24, 20, 20, 0]`) and use `ggml_rope_multi` in llama.cpp. For
//! **text-only** inference the reference expands a 1-D token position
//! `p` into four section positions `[p, p, p, 0]` — see
//! `llm_graph_input_pos::set_input` in llama.cpp. Each section applies
//! section-local frequencies over its slice of the first `n_rot` dims.
//!
//! The frequency math lives upstream in `rlx_flow::rope`; this module
//! is Qwen3.5-specific glue (config-driven section resolution + a few
//! convenience helpers used by the runner).

use crate::config::Qwen35Config;

/// Normalise `rope_dim_sections` to exactly four entries (llama.cpp
/// always passes four ints to `ggml_rope_multi`).
fn sections4(cfg: &Qwen35Config) -> [usize; 4] {
    if cfg.rope_dim_sections.is_empty() {
        [cfg.rope_dim_count, 0, 0, 0]
    } else {
        rlx_flow::rope::mrope_sections4(&cfg.rope_dim_sections)
    }
}

/// Build `[max_pos, half]` cos/sin tables for MRoPE (text modality).
pub fn build_mrope_tables(
    cfg: &Qwen35Config,
    max_pos: usize,
    head_half: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n_rot = cfg.rope_dim_count;
    let half_rot = n_rot / 2;
    assert!(
        n_rot.is_multiple_of(2),
        "rope_dim_count must be even, got {n_rot}"
    );
    assert!(
        head_half >= half_rot,
        "head_half ({head_half}) must cover n_rot/2 ({half_rot})"
    );

    let sections = sections4(cfg);
    let mut cos = vec![0f32; max_pos * head_half];
    let mut sin = vec![0f32; max_pos * head_half];

    for pos in 0..max_pos {
        let (cos_row, sin_row) = rlx_flow::rope::mrope_row_for_sections(
            cfg.rope_theta,
            n_rot,
            sections,
            text_section_pos(pos),
            head_half,
        );
        let row = pos * head_half;
        cos[row..row + head_half].copy_from_slice(&cos_row);
        sin[row..row + head_half].copy_from_slice(&sin_row);
    }

    (cos, sin)
}

/// Text-modality default: `[p, p, p, 0]` per llama.cpp token batches.
pub fn text_section_pos(token_pos: usize) -> [usize; 4] {
    [token_pos, token_pos, token_pos, 0]
}

/// True when the checkpoint declares a non-zero 4th MRoPE section (vision).
pub fn supports_multimodal_mrope(cfg: &Qwen35Config) -> bool {
    sections4(cfg)[3] > 0
}

/// Build one MRoPE cos/sin row from explicit per-section positions.
pub fn mrope_row_for_sections(
    cfg: &Qwen35Config,
    section_pos: [usize; 4],
    head_half: usize,
) -> (Vec<f32>, Vec<f32>) {
    rlx_flow::rope::mrope_row_for_sections(
        cfg.rope_theta,
        cfg.rope_dim_count,
        sections4(cfg),
        section_pos,
        head_half,
    )
}

/// Flattened `[seq * head_half]` cos/sin for runtime MRoPE graph inputs.
pub fn mrope_prefill_feeds(
    cfg: &Qwen35Config,
    seq: usize,
    section_positions: Option<&[[usize; 4]]>,
    head_half: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0f32; seq * head_half];
    let mut sin = vec![0f32; seq * head_half];
    for t in 0..seq {
        let sec = section_positions
            .and_then(|rows| rows.get(t).copied())
            .unwrap_or_else(|| text_section_pos(t));
        let (c_row, s_row) = mrope_row_for_sections(cfg, sec, head_half);
        let off = t * head_half;
        cos[off..off + head_half].copy_from_slice(&c_row);
        sin[off..off + head_half].copy_from_slice(&s_row);
    }
    (cos, sin)
}

/// Slice MRoPE cos/sin at absolute text position `pos`.
pub fn mrope_slice_at_pos(
    cfg: &Qwen35Config,
    pos: usize,
    head_half: usize,
) -> (Vec<f32>, Vec<f32>) {
    mrope_row_for_sections(cfg, text_section_pos(pos), head_half)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(sections: Vec<usize>) -> Qwen35Config {
        Qwen35Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 4,
            nextn_predict_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            key_length: 8,
            value_length: 8,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            rope_dim_count: 8,
            rope_dim_sections: sections,
            full_attention_interval: 3,
            ssm_conv_kernel: 4,
            ssm_group_count: 2,
            ssm_inner_size: 8,
            ssm_state_size: 4,
            ssm_time_step_rank: 2,
            tie_word_embeddings: true,

            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        }
    }

    #[test]
    fn plain_rope_matches_std_at_pos1() {
        let c = cfg(vec![]);
        let (cos, sin) = build_mrope_tables(&c, 4, 4);
        let freq0 = 1.0 / 10_000_f64.powf(0.0);
        let (s0, c0) = (1.0_f64 * freq0).sin_cos();
        assert!((cos[4] - c0 as f32).abs() < 1e-6);
        assert!((sin[4] - s0 as f32).abs() < 1e-6);
    }

    #[test]
    fn mrope_text_sections_match_per_section_freqs() {
        let c = cfg(vec![4, 4, 0, 0]);
        let (cos, _sin) = build_mrope_tables(&c, 2, 4);
        assert!((cos[0] - 1.0).abs() < 1e-6);
        let (s, cval) = (1.0_f64).sin_cos();
        assert!((cos[4] - cval as f32).abs() < 1e-6);
        assert!((_sin[4] - s as f32).abs() < 1e-6);
    }

    #[test]
    fn multimodal_section_pos_differs_from_text() {
        let mut c = cfg(vec![4, 4, 4, 4]);
        c.rope_dim_count = 32;
        c.key_length = 32;
        let head_half = 16;
        let text = mrope_row_for_sections(&c, text_section_pos(2), head_half);
        let mm = mrope_row_for_sections(&c, [2, 2, 2, 5], head_half);
        assert_ne!(text.0, mm.0);
    }
}
