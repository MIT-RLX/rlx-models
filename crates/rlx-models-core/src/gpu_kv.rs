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

//! GPU-resident KV cache via `bind_gpu_handle` (MLX device arrays; Metal/CUDA/WGPU host mirrors).

use crate::autoregressive::{KvCacheState, compact_bucketed_kv_buffer, past_kv_input_names};
use anyhow::{Context, Result, ensure};
use rlx_ir::{Graph, hir::HirModule};
use rlx_runtime::compile_cache::{BucketedCompileCache, CacheRunInput, pad_rows};
use rlx_runtime::kv_cache::LayerKvCache;
use rlx_runtime::{CompileOptions, CompiledGraph, Device};
use std::collections::HashMap;

/// Backends that support persistent K/V handles + selective logits readback.
pub fn device_supports_gpu_kv(device: Device) -> bool {
    matches!(
        device,
        Device::Mlx | Device::Metal | Device::Cuda | Device::Rocm | Device::Gpu | Device::Vulkan
    )
}

/// Tracks which bucket upper bound GPU handles were allocated for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuKvBinding {
    pub upper: u64,
}

/// Per compile-cache GPU binding state.
#[derive(Debug, Default)]
pub struct GpuKvCacheSet {
    pub causal: GpuKvBinding,
    pub decode_mtp: GpuKvBinding,
    pub mtp: GpuKvBinding,
}

impl GpuKvCacheSet {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Drop decode bindings after an MTP block advanced `past_len`.
    pub fn reset_decode_after_mtp(&mut self) {
        self.causal = GpuKvBinding::default();
        self.decode_mtp = GpuKvBinding::default();
        self.mtp = GpuKvBinding::default();
    }
}

/// True when `cross_k_0` is already bound on this graph.
pub fn cross_attn_gpu_handles_ready(compiled: &CompiledGraph) -> bool {
    compiled.has_gpu_handle("cross_k_0")
}

/// Upload fixed cross-attention K/V (`cross_k_*` / `cross_v_*`) for Whisper-style decoders.
pub fn install_cross_attn_gpu_handles(
    compiled: &mut CompiledGraph,
    cross: &LayerKvCache,
    enc_seq: usize,
    kv_dim: usize,
    num_layers: usize,
) -> Result<()> {
    let upper = enc_seq as u64;
    for i in 0..num_layers {
        let k_name = format!("cross_k_{i}");
        let v_name = format!("cross_v_{i}");
        let k_pad = pad_rows(cross.layers_k[i].as_slice(), kv_dim, upper);
        let v_pad = pad_rows(cross.layers_v[i].as_slice(), kv_dim, upper);
        ensure!(
            compiled.bind_gpu_handle(k_name.as_str(), &k_pad),
            "bind_gpu_handle failed for {k_name}"
        );
        ensure!(
            compiled.bind_gpu_handle(v_name.as_str(), &v_pad),
            "bind_gpu_handle failed for {v_name}"
        );
    }
    Ok(())
}

/// Upload `prefix_rows` of `kv` into `past_k_*` / `past_v_*` GPU handles and wire output feeds.
pub fn install_gpu_kv_handles(
    compiled: &mut CompiledGraph,
    kv: &KvCacheState,
    prefix_rows: usize,
    upper: u64,
    kv_dim: usize,
    num_layers: usize,
) -> Result<()> {
    let names = past_kv_input_names(num_layers);
    for layer in 0..num_layers {
        let k_name = names[2 * layer].as_str();
        let v_name = names[2 * layer + 1].as_str();
        let n = prefix_rows * kv_dim;
        let k_slice = &kv.layers_k[layer][..n.min(kv.layers_k[layer].len())];
        let v_slice = &kv.layers_v[layer][..n.min(kv.layers_v[layer].len())];
        let k_pad = pad_rows(k_slice, kv_dim, upper);
        let v_pad = pad_rows(v_slice, kv_dim, upper);
        ensure!(
            compiled.bind_gpu_handle(k_name, &k_pad),
            "bind_gpu_handle failed for {k_name}"
        );
        compiled.set_gpu_handle_feed(k_name, 1 + 2 * layer);
        ensure!(
            compiled.bind_gpu_handle(v_name, &v_pad),
            "bind_gpu_handle failed for {v_name}"
        );
        compiled.set_gpu_handle_feed(v_name, 2 + 2 * layer);
    }
    Ok(())
}

fn layer_host_rows(
    compiled: &CompiledGraph,
    name: &str,
    host: &[f32],
    past_len: usize,
    kv_dim: usize,
) -> Vec<f32> {
    if compiled.has_gpu_handle(name) {
        if let Some(buf) = compiled.read_gpu_handle(name) {
            return compact_bucketed_kv_buffer(&buf, past_len, kv_dim, 1);
        }
    }
    let take = (past_len * kv_dim).min(host.len());
    host[..take].to_vec()
}

/// Rebind handles after a bucket change (read back prior GPU K/V, pad to new `upper`).
pub fn reinstall_gpu_kv_handles(
    compiled: &mut CompiledGraph,
    kv: &KvCacheState,
    _old_upper: u64,
    new_upper: u64,
    kv_dim: usize,
    num_layers: usize,
) -> Result<()> {
    let names = past_kv_input_names(num_layers);
    let mut tmp = KvCacheState {
        past_len: kv.past_len,
        layers_k: Vec::with_capacity(num_layers),
        layers_v: Vec::with_capacity(num_layers),
    };
    for layer in 0..num_layers {
        tmp.layers_k.push(layer_host_rows(
            compiled,
            &names[2 * layer],
            &kv.layers_k[layer],
            kv.past_len,
            kv_dim,
        ));
        tmp.layers_v.push(layer_host_rows(
            compiled,
            &names[2 * layer + 1],
            &kv.layers_v[layer],
            kv.past_len,
            kv_dim,
        ));
    }
    install_gpu_kv_handles(compiled, &tmp, tmp.past_len, new_upper, kv_dim, num_layers)
}

/// Pull GPU K/V back to host `kv` (for MTP truncate / prefill cache).
pub fn sync_gpu_kv_to_host(
    compiled: &CompiledGraph,
    kv: &mut KvCacheState,
    kv_dim: usize,
    num_layers: usize,
) -> Result<()> {
    let names = past_kv_input_names(num_layers);
    let n = kv.past_len * kv_dim;
    for layer in 0..num_layers {
        kv.layers_k[layer] = layer_host_rows(
            compiled,
            &names[2 * layer],
            &kv.layers_k[layer],
            kv.past_len,
            kv_dim,
        );
        kv.layers_v[layer] = layer_host_rows(
            compiled,
            &names[2 * layer + 1],
            &kv.layers_v[layer],
            kv.past_len,
            kv_dim,
        );
        if kv.layers_k[layer].len() > n {
            kv.layers_k[layer].truncate(n);
        }
        if kv.layers_v[layer].len() > n {
            kv.layers_v[layer].truncate(n);
        }
    }
    Ok(())
}

fn ensure_gpu_kv_bindings(
    compiled: &mut CompiledGraph,
    kv: &KvCacheState,
    binding: &mut GpuKvBinding,
    upper: u64,
    kv_dim: usize,
    num_layers: usize,
    refresh_kv: bool,
) -> Result<()> {
    let names = past_kv_input_names(num_layers);
    let handles_live = compiled.has_gpu_handle(names[0].as_str());
    if refresh_kv || !handles_live || binding.upper != upper {
        install_gpu_kv_handles(compiled, kv, kv.past_len, upper, kv_dim, num_layers)?;
        binding.upper = upper;
    }
    Ok(())
}

/// One bucketed decode step with GPU-resident K/V (output 0 readback — logits or hidden_states).
///
/// `cache_key` indexes the compile bucket (for batch>1 use `(batch << 32) | past_seq`).
pub fn run_bucketed_kv_decode_gpu<F>(
    cache: &mut BucketedCompileCache,
    cache_key: u64,
    past_seq: usize,
    kv: &mut KvCacheState,
    binding: &mut GpuKvBinding,
    kv_dim: usize,
    num_layers: usize,
    fixed_inputs: &[CacheRunInput<'_>],
    build: F,
    options: &CompileOptions,
    refresh_kv: bool,
) -> Result<Vec<f32>>
where
    F: FnOnce(u64) -> (Graph, HashMap<String, Vec<f32>>),
{
    let (upper, compiled) = cache
        .ensure_graph_with_params(cache_key, build, options)
        .ok_or_else(|| anyhow::anyhow!("cache_key {cache_key} outside decode buckets"))?;

    ensure_gpu_kv_bindings(compiled, kv, binding, upper, kv_dim, num_layers, refresh_kv)?;

    let mut pairs: Vec<(&str, &[f32])> = Vec::with_capacity(fixed_inputs.len());
    for inp in fixed_inputs {
        pairs.push((inp.name, inp.data));
    }

    // Metal: skip active extent (CPU ignores it; Metal SDPA scaling breaks bucketed mask).
    if compiled.device() != Device::Metal {
        compiled.set_active_extent(Some((upper as usize + 1, upper as usize + 1)));
    }
    let outs = compiled.run_read_outputs(&pairs, Some(&[0]));
    compiled.set_active_extent(None);

    let logits = outs
        .into_iter()
        .next()
        .context("gpu kv decode: missing logits output")?;
    kv.past_len = past_seq + 1;
    Ok(logits)
}

/// HIR variant of [`run_bucketed_kv_decode_gpu`] (stable Metal bucketed decode).
pub fn run_bucketed_kv_decode_gpu_hir<F>(
    cache: &mut BucketedCompileCache,
    cache_key: u64,
    past_seq: usize,
    kv: &mut KvCacheState,
    binding: &mut GpuKvBinding,
    kv_dim: usize,
    num_layers: usize,
    fixed_inputs: &[CacheRunInput<'_>],
    build: F,
    options: &CompileOptions,
    refresh_kv: bool,
) -> Result<Vec<f32>>
where
    F: FnOnce(u64) -> (HirModule, HashMap<String, Vec<f32>>),
{
    let (upper, compiled) = cache
        .ensure_hir_with_params(cache_key, build, options)
        .ok_or_else(|| anyhow::anyhow!("cache_key {cache_key} outside decode buckets"))?;

    ensure_gpu_kv_bindings(compiled, kv, binding, upper, kv_dim, num_layers, refresh_kv)?;

    let mut pairs: Vec<(&str, &[f32])> = Vec::with_capacity(fixed_inputs.len());
    for inp in fixed_inputs {
        pairs.push((inp.name, inp.data));
    }

    if compiled.device() != Device::Metal {
        compiled.set_active_extent(Some((upper as usize + 1, upper as usize + 1)));
    }
    let outs = compiled.run_read_outputs(&pairs, Some(&[0]));
    compiled.set_active_extent(None);

    let logits = outs
        .into_iter()
        .next()
        .context("gpu kv decode: missing logits output")?;
    kv.past_len = past_seq + 1;
    Ok(logits)
}

/// MTP query block with GPU-resident prefix K/V (logits slab readback only).
pub fn run_bucketed_kv_mtp_gpu<F>(
    cache: &mut BucketedCompileCache,
    past_len: usize,
    q_len: usize,
    kv: &KvCacheState,
    binding: &mut GpuKvBinding,
    kv_dim: usize,
    num_layers: usize,
    fixed_inputs: &[CacheRunInput<'_>],
    build: F,
    options: &CompileOptions,
) -> Result<Vec<f32>>
where
    F: FnOnce(u64) -> (Graph, HashMap<String, Vec<f32>>),
{
    let key = past_len as u64;
    let (upper, compiled) = cache
        .ensure_graph_with_params(key, build, options)
        .ok_or_else(|| anyhow::anyhow!("past_len {past_len} outside MTP buckets"))?;

    ensure_gpu_kv_bindings(compiled, kv, binding, upper, kv_dim, num_layers, false)?;
    let actual_kv = past_len + q_len;
    let upper_kv = upper as usize + q_len;
    let mut pairs: Vec<(&str, &[f32])> = Vec::with_capacity(fixed_inputs.len());
    for inp in fixed_inputs {
        pairs.push((inp.name, inp.data));
    }
    compiled.set_active_extent(Some((actual_kv, upper_kv)));
    let outs = compiled.run_read_outputs(&pairs, Some(&[0]));
    compiled.set_active_extent(None);

    outs.into_iter()
        .next()
        .context("gpu kv mtp: missing logits output")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autoregressive::compact_bucketed_kv_buffer;
    use rlx_runtime::Device;

    #[test]
    fn gpu_kv_supported_backends() {
        assert!(device_supports_gpu_kv(Device::Mlx));
        assert!(device_supports_gpu_kv(Device::Metal));
        assert!(device_supports_gpu_kv(Device::Cuda));
        assert!(device_supports_gpu_kv(Device::Gpu));
        assert!(device_supports_gpu_kv(Device::Rocm));
        assert!(!device_supports_gpu_kv(Device::Cpu));
    }

    #[test]
    fn compact_bucketed_kv_skips_middle_padding() {
        let kv_dim = 2;
        // past_len=3: rows 0,1 real; row 2 padding; row 3 (upper) new token.
        let buf = vec![
            1.0, 1.1, //
            2.0, 2.1, //
            0.0, 0.0, // padding
            9.0, 9.1, // new K at upper
        ];
        let out = compact_bucketed_kv_buffer(&buf, 3, kv_dim, 1);
        assert_eq!(out, vec![1.0, 1.1, 2.0, 2.1, 9.0, 9.1]);
    }
}
