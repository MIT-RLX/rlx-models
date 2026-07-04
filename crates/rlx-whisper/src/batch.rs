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

//! Batched KV cache stacking for multi-region / multi-beam decode.

use crate::cache::{WhisperCrossCache, WhisperKvCache};

/// Repeat each region's encoder hidden `beam` times: `[n, plane]` → `[n*beam, plane]`.
pub fn replicate_encoder_for_beams(enc: &[f32], n: usize, beam: usize, plane: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * beam * plane];
    for r in 0..n {
        let src = r * plane;
        for k in 0..beam {
            let dst = (r * beam + k) * plane;
            out[dst..dst + plane].copy_from_slice(&enc[src..src + plane]);
        }
    }
    out
}

/// Flatten prompt tokens for batch input: `[batch, seq]` row-major f32.
pub fn batched_prompt_f32(prompt: &[u32], batch: usize) -> Vec<f32> {
    let seq = prompt.len();
    let mut out = Vec::with_capacity(batch * seq);
    for _ in 0..batch {
        for &t in prompt {
            out.push(t as f32);
        }
    }
    out
}

/// One region's cross-attention slice from a batched cross cache.
pub fn slice_cross_cache(
    cross: &WhisperCrossCache,
    index: usize,
    batch: usize,
) -> Result<WhisperCrossCache, String> {
    if batch == 1 {
        return Ok(cross.clone());
    }
    let n_layers = cross.layers_k.len();
    let per = cross.layers_k[0].len() / batch;
    if per == 0 || cross.layers_k[0].len() != batch * per {
        return Err("slice_cross_cache: bad batch layout".into());
    }
    let off = index * per;
    let mut layers_k = Vec::with_capacity(n_layers);
    let mut layers_v = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        layers_k.push(cross.layers_k[layer][off..off + per].to_vec());
        layers_v.push(cross.layers_v[layer][off..off + per].to_vec());
    }
    Ok(WhisperCrossCache {
        past_len: cross.past_len,
        layers_kv_base: vec![0; n_layers],
        layers_k,
        layers_v,
    })
}

/// Stack `n` batch-1 caches with identical `past_len` into one cache (batch = n).
pub fn stack_kv_caches(caches: &[WhisperKvCache]) -> Result<WhisperKvCache, String> {
    if caches.is_empty() {
        return Err("stack_kv_caches: empty".into());
    }
    let past_len = caches[0].past_len;
    let n_layers = caches[0].layers_k.len();
    for (i, c) in caches.iter().enumerate().skip(1) {
        if c.past_len != past_len {
            return Err(format!("cache {i} past_len {} != {past_len}", c.past_len));
        }
        if c.layers_k.len() != n_layers {
            return Err(format!("cache {i} layer count mismatch"));
        }
    }
    let _n = caches.len();
    let mut layers_k = Vec::with_capacity(n_layers);
    let mut layers_v = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let mut k = Vec::new();
        let mut v = Vec::new();
        for c in caches {
            k.extend_from_slice(&c.layers_k[layer]);
            v.extend_from_slice(&c.layers_v[layer]);
        }
        layers_k.push(k);
        layers_v.push(v);
    }
    Ok(WhisperKvCache {
        past_len,
        layers_kv_base: vec![0; n_layers],
        layers_k,
        layers_v,
    })
}

/// Split a batched cache (batch = n) into n batch-1 caches.
pub fn unstack_kv_cache(cache: &WhisperKvCache, n: usize) -> Result<Vec<WhisperKvCache>, String> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let n_layers = cache.layers_k.len();
    let per_batch = cache.layers_k.first().map(|k| k.len() / n).unwrap_or(0);
    if per_batch == 0 || cache.layers_k[0].len() != n * per_batch {
        return Err("unstack_kv_cache: length not divisible by batch".into());
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n {
        let off = b * per_batch;
        let mut layers_k = Vec::with_capacity(n_layers);
        let mut layers_v = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
            layers_k.push(cache.layers_k[layer][off..off + per_batch].to_vec());
            layers_v.push(cache.layers_v[layer][off..off + per_batch].to_vec());
        }
        out.push(WhisperKvCache {
            past_len: cache.past_len,
            layers_kv_base: vec![0; n_layers],
            layers_k,
            layers_v,
        });
    }
    Ok(out)
}

/// Reorder batch rows in a stacked cache (`indices[i]` = source beam for destination i).
pub fn reorder_kv_beams(
    cache: &WhisperKvCache,
    indices: &[usize],
    batch: usize,
) -> Result<WhisperKvCache, String> {
    if indices.len() != batch {
        return Err("reorder_kv_beams: indices length mismatch".into());
    }
    let per_batch = cache.layers_k.first().map(|k| k.len() / batch).unwrap_or(0);
    let n_layers = cache.layers_k.len();
    let mut layers_k = Vec::with_capacity(n_layers);
    let mut layers_v = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let mut k = vec![0f32; batch * per_batch];
        let mut v = vec![0f32; batch * per_batch];
        for (dst, &src) in indices.iter().enumerate() {
            if src >= batch {
                return Err(format!("reorder_kv_beams: bad index {src}"));
            }
            let s0 = src * per_batch;
            let d0 = dst * per_batch;
            k[d0..d0 + per_batch].copy_from_slice(&cache.layers_k[layer][s0..s0 + per_batch]);
            v[d0..d0 + per_batch].copy_from_slice(&cache.layers_v[layer][s0..s0 + per_batch]);
        }
        layers_k.push(k);
        layers_v.push(v);
    }
    Ok(WhisperKvCache {
        past_len: cache.past_len,
        layers_kv_base: vec![0; n_layers],
        layers_k,
        layers_v,
    })
}

/// Stack cross-attention caches (same `enc_seq`, batch = n).
pub fn stack_cross_caches(
    caches: &[crate::cache::WhisperCrossCache],
) -> Result<crate::cache::WhisperCrossCache, String> {
    if caches.is_empty() {
        return Err("stack_cross_caches: empty".into());
    }
    let enc_len = caches[0].layers_k[0].len();
    let n_layers = caches[0].layers_k.len();
    for c in caches.iter().skip(1) {
        if c.layers_k.len() != n_layers {
            return Err("cross layer count mismatch".into());
        }
        if c.layers_k[0].len() != enc_len {
            return Err("cross enc length mismatch".into());
        }
    }
    let mut layers_k = Vec::with_capacity(n_layers);
    let mut layers_v = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let mut k = Vec::new();
        let mut v = Vec::new();
        for c in caches {
            k.extend_from_slice(&c.layers_k[layer]);
            v.extend_from_slice(&c.layers_v[layer]);
        }
        layers_k.push(k);
        layers_v.push(v);
    }
    Ok(crate::cache::WhisperCrossCache {
        past_len: caches[0].past_len,
        layers_kv_base: vec![0; n_layers],
        layers_k,
        layers_v,
    })
}
