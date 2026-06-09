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

//! Decoder KV state — engine tensors in [`rlx_runtime::LayerKvCache`].

pub use rlx_runtime::LayerKvCache;

/// Incremental self-attention cache.
pub type WhisperKvCache = LayerKvCache;

/// Fixed cross-attention K/V from encoder hidden.
pub type WhisperCrossCache = LayerKvCache;

pub fn kv_from_prefill_outputs(
    num_layers: usize,
    batch: usize,
    past_seq: usize,
    d_model: usize,
    outputs: &[Vec<f32>],
) -> Result<WhisperKvCache, String> {
    LayerKvCache::from_layer_outputs(num_layers, batch, past_seq, d_model, outputs)
}

pub fn cross_from_outputs(
    num_layers: usize,
    batch: usize,
    enc_seq: usize,
    d_model: usize,
    outputs: &[Vec<f32>],
) -> Result<WhisperCrossCache, String> {
    LayerKvCache::from_layer_outputs(num_layers, batch, enc_seq, d_model, outputs)
}

/// Apply bucketed decode K/V (`[batch, past+1, d]` row-major per layer).
pub fn apply_bucketed_decode_step(
    cache: &mut WhisperKvCache,
    layers_k: Vec<Vec<f32>>,
    layers_v: Vec<Vec<f32>>,
    batch: usize,
    kv_dim: usize,
) -> Result<(), String> {
    let n = cache.layers_k.len();
    if layers_k.len() != n || layers_v.len() != n {
        return Err(format!(
            "apply_bucketed_decode_step: expected {n} layers, got k={} v={}",
            layers_k.len(),
            layers_v.len()
        ));
    }
    let new_len = cache.past_len + 1;
    let real_len = batch * new_len * kv_dim;
    for layer in 0..n {
        let k = &layers_k[layer];
        let v = &layers_v[layer];
        if k.len() < real_len || v.len() < real_len {
            return Err(format!(
                "layer {layer}: k.len={} v.len={} expected at least {real_len}",
                k.len(),
                v.len()
            ));
        }
        cache.layers_k[layer] = k[..real_len].to_vec();
        cache.layers_v[layer] = v[..real_len].to_vec();
    }
    cache.past_len = new_len;
    Ok(())
}
