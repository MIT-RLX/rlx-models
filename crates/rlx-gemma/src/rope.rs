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
    if matches!(cfg.arch, GemmaArch::Gemma3 | GemmaArch::Gemma4) {
        if let Some(si) = first_sliding_layer(cfg) {
            return (cfg.layer_rope_theta(si), cfg.layer_n_rot(si));
        }
    }
    (cfg.rope_theta, cfg.head_dim())
}

/// `Some((theta, n_rot))` when full-attention RoPE differs from sliding.
///
/// `layer_types` is empty when the config came from a GGUF (llama.cpp's GGUFs
/// don't encode the per-layer list — the strided pattern is implicit). The
/// per-layer helpers `is_full_attention_layer` / `layer_n_rot` / `layer_rope_theta`
/// already fall back to the strided pattern, so we can ask them directly.
pub fn global_rope_params(cfg: &GemmaConfig) -> Option<(f64, usize)> {
    if !matches!(cfg.arch, GemmaArch::Gemma3 | GemmaArch::Gemma4) {
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
/// **Proportional partial RoPE** (HF `_compute_proportional_rope_parameters`):
/// only the leading `n_rot` of `head_dim` dimensions rotate, but the
/// inverse-frequency exponent denominator is the **full** head_dim, NOT the
/// rotary count — `inv_freq[i] = theta^(-2i/head_dim)` for `i in 0..n_rot/2`.
/// Building the table with `default_inv_freq(theta, n_rot)` (denominator =
/// n_rot) is wrong for partial RoPE; we must use the full head_dim then keep
/// the leading `n_rot/2` pairs. For non-partial global RoPE `n_rot == head_dim`
/// so this reduces to the plain table.
///
/// GGUF checkpoints ship `rope_freqs.weight` at `global_head_dim / 2`
/// (256 for Gemma 4 12B). Factors apply to the global table only; sliding
/// layers ignore them.
pub fn resolve_global_inv_freq(
    cfg: &GemmaConfig,
    rope_freq_factors: Option<&[f32]>,
) -> Option<Vec<f64>> {
    let (theta, n_rot) = global_rope_params(cfg)?;
    let fi = first_full_layer(cfg)?;
    // Full head_dim of the full-attention layer (== global_head_dim when set).
    let full_head_dim = cfg.layer_head_dim(fi);
    // The cos/sin table MUST have `head_dim/2` columns per position: the RoPE
    // kernel indexes rows with stride `head_dim/2` (`tab_half`) and rotates the
    // leading `n_rot/2` pairs. For proportional partial RoPE the rotated freqs
    // are `theta^(-2i/head_dim)` for `i in 0..n_rot/2` (denominator = full
    // head_dim); the trailing `head_dim/2 - n_rot/2` entries are the unrotated
    // (NoPE) dims — the kernel never reads them, but they keep the row stride
    // correct. Truncating to `n_rot/2` (as before) gave a short row stride and
    // made every position > 0 read the wrong table row.
    let mut base = default_inv_freq(theta, full_head_dim); // len = head_dim/2
    // Zero the unrotated tail so the table is unambiguous (kernel ignores it).
    let rot = (n_rot / 2).min(base.len());
    for v in base.iter_mut().skip(rot) {
        *v = 0.0;
    }
    // llama.cpp / GGUF: `rope_freqs.weight` covers the full global rotary dim
    // (length head_dim/2). Apply when sized to the table.
    if let Some(f) = rope_freq_factors.filter(|f| !f.is_empty()) {
        if f.len() == base.len() {
            base = inv_freq_with_factors(&base, f);
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
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: 0,
            num_kv_shared_layers: 0,
            use_double_wide_mlp: false,
            enable_moe_block: false,
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
    fn proportional_global_rope_uses_full_head_dim_denominator() {
        // Gemma 4 full-attention layers: partial_rotary_factor=0.25, θ=1e6,
        // head_dim=512 → rotate 128 dims (64 pairs) but with denominator 512.
        let cfg = gemma4_12b_cfg();
        let global = resolve_global_inv_freq(&cfg, None).expect("global table");
        // Table row stride must be head_dim/2 = 256 (kernel's `tab_half`); the
        // first 64 entries are the rotated proportional freqs, the trailing
        // 192 are zero (unrotated / NoPE dims).
        assert_eq!(global.len(), 256, "head_dim/2 columns");
        assert!((global[0] - 1.0).abs() < 1e-12);
        // inv_freq[1] must use the FULL head_dim (512) as denominator:
        //   theta^(-2/512), NOT the buggy theta^(-2/128).
        let theta = 1_000_000.0f64;
        let expected = 1.0 / theta.powf(2.0 / 512.0);
        let buggy = 1.0 / theta.powf(2.0 / 128.0);
        assert!(
            (global[1] - expected).abs() < 1e-9,
            "global[1]={} expected {} (denom 512)",
            global[1],
            expected
        );
        assert!(
            (global[1] - buggy).abs() > 1e-6,
            "must not match the old denom-128 value"
        );
        // Rotated count = 0.25*512/2 = 64; entries past that are zeroed.
        assert!((global[64]).abs() < 1e-12, "tail (NoPE dims) must be zero");
        assert!(global[63].abs() > 0.0, "first 64 are rotated freqs");
    }

    #[test]
    fn gguf_rope_factors_sized_for_global_head_do_not_break_sliding() {
        let cfg = gemma4_12b_cfg();
        let factors = vec![1.0f32; 256];
        let sliding = resolve_inv_freq(&cfg, Some(&factors));
        assert_eq!(sliding.len(), 128);
        let global = resolve_global_inv_freq(&cfg, Some(&factors)).expect("global table");
        assert_eq!(global.len(), 256); // head_dim/2 columns
    }

    #[test]
    fn mismatched_factors_are_skipped_not_panicked() {
        let cfg = gemma4_12b_cfg();
        let factors = vec![1.0f32; 999];
        let sliding = resolve_inv_freq(&cfg, Some(&factors));
        assert_eq!(sliding.len(), 128);
    }
}
