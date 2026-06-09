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

//! Gemma RoPE — thin family-specific glue over `rlx_flow::rope`.

use super::config::{GemmaArch, GemmaConfig};

pub use rlx_flow::rope::{
    build_tables as build_rope_tables, default_inv_freq, inv_freq_with_factors,
};

/// Apply GGUF `rope_freqs.weight` when lengths match; otherwise leave `base` unchanged.
pub fn apply_rope_freq_factors(base: &[f64], factors: Option<&[f32]>) -> Vec<f64> {
    match factors {
        Some(f) if !f.is_empty() && f.len() == base.len() => inv_freq_with_factors(base, f),
        _ => base.to_vec(),
    }
}

fn first_sliding_layer(cfg: &GemmaConfig) -> Option<usize> {
    (0..cfg.num_hidden_layers).find(|&i| !cfg.is_full_attention_layer(i))
}

fn first_full_layer(cfg: &GemmaConfig) -> Option<usize> {
    (0..cfg.num_hidden_layers).find(|&i| cfg.is_full_attention_layer(i))
}

/// `(theta, n_rot)` for the default (sliding) RoPE table.
pub fn sliding_rope_params(cfg: &GemmaConfig) -> (f64, usize) {
    if cfg.arch == GemmaArch::Gemma4 {
        if let Some(si) = first_sliding_layer(cfg) {
            return (cfg.layer_rope_theta(si), cfg.layer_n_rot(si));
        }
    }
    (cfg.rope_theta, cfg.head_dim())
}

/// `Some((theta, n_rot))` when Gemma 4 full-attention RoPE differs from sliding.
pub fn global_rope_params(cfg: &GemmaConfig) -> Option<(f64, usize)> {
    if cfg.arch != GemmaArch::Gemma4 || cfg.layer_types.is_empty() {
        return None;
    }
    let fi = first_full_layer(cfg)?;
    let si = first_sliding_layer(cfg)?;
    let theta = cfg.layer_rope_theta(fi);
    let n_rot = cfg.layer_n_rot(fi);
    let differs = (theta - cfg.layer_rope_theta(si)).abs() > 1e-3 || n_rot != cfg.layer_n_rot(si);
    differs.then_some((theta, n_rot))
}

/// Inverse frequencies for the primary (sliding) RoPE table.
pub fn resolve_inv_freq(cfg: &GemmaConfig, rope_freq_factors: Option<&[f32]>) -> Vec<f64> {
    let (theta, n_rot) = sliding_rope_params(cfg);
    let base = default_inv_freq(theta, n_rot);
    apply_rope_freq_factors(&base, rope_freq_factors)
}

/// Global (full-attention) inverse frequencies for Gemma 4 proportional RoPE.
///
/// GGUF checkpoints often ship `rope_freqs.weight` at `global_head_dim / 2`
/// (256 for Gemma 4 12B) while the sliding table uses `sliding_n_rot / 2`
/// (128). Factors apply to the global table only; sliding layers ignore them.
pub fn resolve_global_inv_freq(
    cfg: &GemmaConfig,
    rope_freq_factors: Option<&[f32]>,
) -> Option<Vec<f64>> {
    let (theta, n_rot) = global_rope_params(cfg)?;
    let base = default_inv_freq(theta, n_rot);
    if let Some(f) = rope_freq_factors.filter(|f| !f.is_empty()) {
        if f.len() == base.len() {
            return Some(apply_rope_freq_factors(&base, Some(f)));
        }
        // llama.cpp / GGUF: factors cover the full global head rotary dim.
        if let Some(gdh) = cfg.global_head_dim {
            let full = default_inv_freq(theta, gdh);
            if f.len() == full.len() {
                let scaled = inv_freq_with_factors(&full, f);
                let half = n_rot / 2;
                return Some(scaled[..half.min(scaled.len())].to_vec());
            }
        }
    }
    Some(base)
}

/// Single-row cos/sin slice for decode at absolute position `pos`.
pub fn rope_slice(inv_freq: &[f64], pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    for (i, &freq) in inv_freq.iter().enumerate() {
        let angle = pos as f64 * freq;
        let (s, c) = angle.sin_cos();
        cos[i] = c as f32;
        sin[i] = s as f32;
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        GemmaArch, GemmaConfig, GemmaLayerType, GemmaRopeKind, GemmaRopeMap, GemmaRopeParameters,
    };

    fn gemma4_12b_cfg() -> GemmaConfig {
        GemmaConfig {
            arch: GemmaArch::Gemma4,
            vocab_size: 262_144,
            hidden_size: 3840,
            intermediate_size: 15_360,
            num_hidden_layers: 48,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            tie_word_embeddings: true,
            attention_bias: false,
            head_dim: Some(256),
            attn_logit_softcapping: None,
            final_logit_softcapping: Some(30.0),
            sliding_window: Some(1024),
            query_pre_attn_scalar: None,
            effective_num_layers: None,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            expert_weights_scale: 1.0,
            layer_types: (0..48)
                .map(|i| {
                    if (i + 1) % 6 == 0 {
                        GemmaLayerType::FullAttention
                    } else {
                        GemmaLayerType::SlidingAttention
                    }
                })
                .collect(),
            rope_parameters: GemmaRopeMap {
                sliding_attention: Some(GemmaRopeParameters {
                    rope_theta: Some(10_000.0),
                    rope_type: Some(GemmaRopeKind::Default),
                    partial_rotary_factor: None,
                }),
                full_attention: Some(GemmaRopeParameters {
                    rope_theta: Some(1_000_000.0),
                    rope_type: Some(GemmaRopeKind::Proportional),
                    partial_rotary_factor: Some(0.25),
                }),
            },
            global_head_dim: Some(512),
            num_global_key_value_heads: Some(1),
            attention_k_eq_v: true,
            use_bidirectional_attention: Some("vision".into()),
        }
    }

    #[test]
    fn tables_len_matches_max_pos() {
        let inv = default_inv_freq(10_000.0, 8);
        let (c, s) = build_rope_tables(&inv, 4);
        assert_eq!(c.len(), 4 * inv.len());
        assert_eq!(s.len(), c.len());
    }

    #[test]
    fn gguf_rope_factors_sized_for_global_head_do_not_break_sliding() {
        let cfg = gemma4_12b_cfg();
        let factors = vec![1.0f32; 256];
        let sliding = resolve_inv_freq(&cfg, Some(&factors));
        assert_eq!(sliding.len(), 128);
        let global = resolve_global_inv_freq(&cfg, Some(&factors)).expect("global table");
        assert_eq!(global.len(), 64);
    }

    #[test]
    fn mismatched_factors_are_skipped_not_panicked() {
        let cfg = gemma4_12b_cfg();
        let factors = vec![1.0f32; 999];
        let sliding = resolve_inv_freq(&cfg, Some(&factors));
        assert_eq!(sliding.len(), 128);
    }
}
