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

//! Talker MRoPE cos/sin tables (text modality — three axes share the same index).

use crate::config::TalkerConfig;

fn sections4(talker: &TalkerConfig) -> [usize; 4] {
    talker.rope_sections()
}

/// Text modality: `[p, p, p, 0]` per llama.cpp / Qwen3.5 text batches.
pub fn text_section_pos(token_pos: usize) -> [usize; 4] {
    [token_pos, token_pos, token_pos, 0]
}

pub fn mrope_row_for_sections(
    talker: &TalkerConfig,
    section_pos: [usize; 4],
    head_half: usize,
) -> (Vec<f32>, Vec<f32>) {
    rlx_flow::rope::mrope_row_for_sections(
        talker.rope_theta,
        talker.head_dim,
        sections4(talker),
        section_pos,
        head_half,
    )
}

pub fn build_mrope_tables(
    talker: &TalkerConfig,
    max_pos: usize,
    head_half: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0f32; max_pos * head_half];
    let mut sin = vec![0f32; max_pos * head_half];
    for pos in 0..max_pos {
        let (c_row, s_row) = mrope_row_for_sections(talker, text_section_pos(pos), head_half);
        let off = pos * head_half;
        cos[off..off + head_half].copy_from_slice(&c_row);
        sin[off..off + head_half].copy_from_slice(&s_row);
    }
    (cos, sin)
}

pub fn mrope_prefill_feeds(
    talker: &TalkerConfig,
    seq: usize,
    head_half: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0f32; seq * head_half];
    let mut sin = vec![0f32; seq * head_half];
    for t in 0..seq {
        let (c_row, s_row) = mrope_row_for_sections(talker, text_section_pos(t), head_half);
        let off = t * head_half;
        cos[off..off + head_half].copy_from_slice(&c_row);
        sin[off..off + head_half].copy_from_slice(&s_row);
    }
    (cos, sin)
}

pub fn mrope_slice_at_pos(
    talker: &TalkerConfig,
    pos: usize,
    head_half: usize,
) -> (Vec<f32>, Vec<f32>) {
    mrope_row_for_sections(talker, text_section_pos(pos), head_half)
}

/// HF `get_rope_index` for talker prefill (`attention_mask` 0/1, length `seq`).
pub fn talker_rope_index_prefill(attention_mask: &[u8]) -> (Vec<usize>, i64) {
    let seq = attention_mask.len();
    let mut pos = vec![0usize; seq];
    let mut running = 0i64;
    for (i, &m) in attention_mask.iter().enumerate() {
        if m != 0 {
            pos[i] = running as usize;
            running += 1;
        } else {
            pos[i] = 1;
        }
    }
    let max_pos = pos.iter().copied().max().unwrap_or(0);
    let valid = attention_mask.iter().filter(|&&m| m != 0).count() as i64;
    let rope_delta = (max_pos as i64 + 1) - valid;
    (pos, rope_delta)
}

pub fn talker_decode_position(
    talker: &TalkerConfig,
    past_seq: usize,
    rope_delta: i64,
    head_half: usize,
) -> (Vec<f32>, Vec<f32>) {
    let pos = (past_seq as i64 + rope_delta) as usize;
    talker_rope_half(talker, pos, head_half)
}

/// Per-token cos/sin half-table for talker text prefill/decode (matches HF
/// `Qwen3TTSTalkerRotaryEmbedding` + interleaved MRoPE when all three axes share
/// the same index).
pub fn talker_rope_half(
    talker: &TalkerConfig,
    pos: usize,
    head_half: usize,
) -> (Vec<f32>, Vec<f32>) {
    if talker.rope_scaling.is_some() {
        let inv = crate::talker::rope::build_inv_freq(talker.head_dim, talker.rope_theta);
        crate::talker::rope::rope_slice(&inv, pos, talker.head_dim)
    } else {
        mrope_slice_at_pos(talker, pos, head_half)
    }
}

/// Decode cos/sin into reusable buffers (talker text path uses 1D RoPE when `rope_scaling` is set).
pub fn talker_decode_rope_into(
    talker: &TalkerConfig,
    inv_freq: &[f64],
    past_seq: usize,
    rope_delta: i64,
    cos: &mut [f32],
    sin: &mut [f32],
) {
    let pos = (past_seq as i64 + rope_delta) as usize;
    if talker.rope_scaling.is_some() {
        crate::talker::rope::rope_slice_into(inv_freq, pos, talker.head_dim, cos, sin);
    } else {
        let (c, s) = talker_decode_position(talker, past_seq, rope_delta, cos.len());
        cos.copy_from_slice(&c);
        sin.copy_from_slice(&s);
    }
}

/// Flattened `[seq * head_half]` cos/sin for compiled talker prefill.
pub fn talker_prefill_rope_feeds(
    talker: &TalkerConfig,
    positions: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let half = talker.head_dim / 2;
    let mut cos = vec![0f32; positions.len() * half];
    let mut sin = vec![0f32; positions.len() * half];
    for (t, &pos) in positions.iter().enumerate() {
        let (c, s) = talker_rope_half(talker, pos, half);
        let off = t * half;
        cos[off..off + half].copy_from_slice(&c);
        sin[off..off + half].copy_from_slice(&s);
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{RopeScaling, TalkerConfig};
    use crate::talker::rope::{build_inv_freq, rope_slice};

    fn talker_with_mrope() -> TalkerConfig {
        TalkerConfig {
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 32768,
            num_code_groups: 16,
            vocab_size: 3072,
            text_hidden_size: 2048,
            text_vocab_size: 151936,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".into(),
            attention_bias: false,
            qk_norm: true,
            codec_bos_id: 0,
            codec_eos_token_id: 2150,
            codec_pad_id: 0,
            codec_think_id: 0,
            codec_think_bos_id: 0,
            codec_think_eos_id: 0,
            codec_nothink_id: 0,
            position_id_per_seconds: 13,
            rope_scaling: Some(RopeScaling {
                interleaved: true,
                mrope_section: [24, 20, 20],
            }),
            spk_id: Default::default(),
            codec_language_id: Default::default(),
            spk_is_dialect: Default::default(),
        }
    }

    #[test]
    fn talker_text_rope_matches_standard_at_pos12() {
        let talker = talker_with_mrope();
        let half = talker.head_dim / 2;
        let (tc, _) = talker_rope_half(&talker, 12, half);
        let inv = build_inv_freq(talker.head_dim, talker.rope_theta);
        let (rc, _) = rope_slice(&inv, 12, talker.head_dim);
        let mut max_d = 0f32;
        for (a, b) in tc.iter().zip(rc.iter()) {
            max_d = max_d.max((a - b).abs());
        }
        assert!(max_d < 1e-5, "talker rope vs standard max_abs={max_d}");
        let (mc, _) = mrope_slice_at_pos(&talker, 12, half);
        let mut md = 0f32;
        for (a, b) in mc.iter().zip(rc.iter()) {
            md = md.max((a - b).abs());
        }
        assert!(md < 1e-5, "mrope row vs standard max_abs={md}");
    }
}
