// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Llama 3.x RoPE tables — matches HuggingFace / llama.cpp `rope_freqs.weight`.

use super::config::{Llama32Config, Llama32RopeScaling, Llama32RopeType};
use std::f64::consts::PI;

/// Per-dimension inverse frequencies before position scaling.
pub fn default_inv_freq(rope_theta: f64, head_dim: usize) -> Vec<f64> {
    (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / rope_theta.powf(i as f64 / head_dim as f64))
        .collect()
}

/// Llama 3 RoPE scaling from HF `rope_scaling` (candle / transformers formula).
pub fn llama3_inv_freq(cfg: &Llama32Config) -> Vec<f64> {
    let head_dim = cfg.head_dim();
    let base = default_inv_freq(cfg.rope_theta, head_dim);
    let Some(Llama32RopeScaling {
        factor,
        low_freq_factor,
        high_freq_factor,
        original_max_position_embeddings,
        rope_type: Llama32RopeType::Llama3,
    }) = cfg.rope_scaling.as_ref()
    else {
        return base;
    };

    let low_freq_wavelen = *original_max_position_embeddings as f64 / *low_freq_factor as f64;
    let high_freq_wavelen = *original_max_position_embeddings as f64 / *high_freq_factor as f64;

    base.into_iter()
        .map(|freq| {
            let wavelen = 2.0 * PI / freq;
            if wavelen < high_freq_wavelen {
                freq
            } else if wavelen > low_freq_wavelen {
                freq / *factor as f64
            } else {
                let smooth = (*original_max_position_embeddings as f64 / wavelen
                    - *low_freq_factor as f64)
                    / (*high_freq_factor as f64 - *low_freq_factor as f64);
                (1.0 - smooth) * freq / *factor as f64 + smooth * freq
            }
        })
        .collect()
}

/// Apply baked GGUF `rope_freqs.weight` factors (llama.cpp divides θ by ff).
pub fn inv_freq_with_factors(base: &[f64], factors: &[f32]) -> Vec<f64> {
    assert_eq!(
        base.len(),
        factors.len(),
        "rope_freqs.weight length must match head_dim/2"
    );
    base.iter()
        .zip(factors.iter())
        .map(|(f, ff)| f / *ff as f64)
        .collect()
}

/// Resolve inverse frequencies: optional GGUF tensor overrides HF config.
pub fn resolve_inv_freq(cfg: &Llama32Config, rope_freq_factors: Option<&[f32]>) -> Vec<f64> {
    let base = llama3_inv_freq(cfg);
    match rope_freq_factors {
        Some(f) if !f.is_empty() => inv_freq_with_factors(&base, f),
        _ => base,
    }
}

/// Build `[max_pos, head_dim/2]` cos/sin tables from inverse frequencies.
pub fn build_rope_tables(inv_freq: &[f64], max_pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let mut cos = vec![0f32; max_pos * half];
    let mut sin = vec![0f32; max_pos * half];
    for pos in 0..max_pos {
        for (i, &freq) in inv_freq.iter().enumerate() {
            let angle = pos as f64 * freq;
            let (s, c) = angle.sin_cos();
            cos[pos * half + i] = c as f32;
            sin[pos * half + i] = s as f32;
        }
    }
    (cos, sin)
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
    use crate::config::Llama32Config;

    #[test]
    fn llama3_scaling_reduces_low_freq_inv_freq() {
        let mut cfg = Llama32Config {
            hidden_size: 64,
            num_attention_heads: 4,
            rope_theta: 500_000.0,
            max_position_embeddings: 131_072,
            rope_scaling: Some(Llama32RopeScaling {
                factor: 32.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8192,
                rope_type: Llama32RopeType::Llama3,
            }),
            ..Llama32Config::tiny_test()
        };
        cfg.num_key_value_heads = 2;
        let plain = default_inv_freq(cfg.rope_theta, cfg.head_dim());
        let scaled = llama3_inv_freq(&cfg);
        assert!(scaled.last().unwrap() < plain.last().unwrap());
    }
}
