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

//! Layer-sharded LM execution for wgpu/Vulkan (4 GiB storage-buffer cap).

use crate::backbone::compiled::{
    PREFILL_DYNAMIC_CACHE_CAP, decode_compile_options, metal_decode_compile_guard,
    prefill_compile_options,
};
use crate::config::TextConfig;
use crate::lm_flow::{
    build_tts_backbone_decode_shard_hir_sized_ext, build_tts_backbone_prefill_shard_hir_dynamic_ext,
};
use crate::weights::CheckpointParamLoader;
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_ir::DimBinding;
use rlx_runtime::Device;
use rlx_runtime::compile_cache::{BucketedCompileCache, DynamicDimCompileCache};
use std::collections::{HashMap, HashSet};

pub(crate) fn wgpu_layer_shard_size() -> usize {
    if let Ok(raw) = std::env::var("RLX_VOXTRAL_TTS_WGPU_SHARD_LAYERS") {
        if let Ok(v) = raw.parse::<usize>() {
            if v > 0 {
                return v;
            }
        }
    }
    4
}

pub(crate) fn layer_shards_for_device(device: Device, n_layers: usize) -> Vec<(usize, usize)> {
    if !matches!(device, Device::Gpu | Device::Vulkan) {
        return Vec::new();
    }
    partition_layers(n_layers, wgpu_layer_shard_size())
}

fn partition_layers(n_layers: usize, shard_size: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < n_layers {
        let end = (start + shard_size).min(n_layers);
        out.push((start, end));
        start = end;
    }
    out
}

#[derive(Default)]
struct ShardRuntime {
    prefill_cache: Option<DynamicDimCompileCache>,
    decode_cache: Option<BucketedCompileCache>,
    decode_loaded_buckets: HashSet<usize>,
}

pub(crate) struct ShardedLmState {
    shards: Vec<(usize, usize)>,
    runtimes: Vec<ShardRuntime>,
}

impl ShardedLmState {
    pub fn new(device: Device, n_layers: usize) -> Option<Self> {
        let shards = layer_shards_for_device(device, n_layers);
        if shards.is_empty() {
            return None;
        }
        Some(Self {
            runtimes: (0..shards.len()).map(|_| ShardRuntime::default()).collect(),
            shards,
        })
    }

    pub fn reset_caches(&mut self) {
        for rt in &mut self.runtimes {
            rt.prefill_cache = None;
            rt.decode_cache = None;
            rt.decode_loaded_buckets.clear();
        }
    }

    pub fn drop_prefill_caches(&mut self) {
        for rt in &mut self.runtimes {
            rt.prefill_cache = None;
        }
    }
}

pub(crate) fn run_prefill_sharded(
    state: &mut ShardedLmState,
    cfg: &TextConfig,
    device: Device,
    max_seq: usize,
    backbone_params: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    graph_params: &mut Option<HashMap<String, Vec<f32>>>,
    seq: usize,
    embeds: &[f32],
    n_layers: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut hidden_in = embeds.to_vec();
    let mut all_kv = vec![Vec::new(); 2 * n_layers];

    for (shard_idx, &(layer_start, layer_end)) in state.shards.iter().enumerate() {
        let n_shard = layer_end - layer_start;
        let binding = DimBinding::batch_seq(1, seq);
        let opts = prefill_compile_options(device, binding.clone());
        let runtime = &mut state.runtimes[shard_idx];
        if runtime.prefill_cache.is_none() {
            runtime.prefill_cache = Some(DynamicDimCompileCache::new(
                device,
                PREFILL_DYNAMIC_CACHE_CAP,
            ));
        }
        let cache = runtime.prefill_cache.as_mut().expect("prefill cache");
        let needs_upload = !cache.contains(seq as u64);
        let needs_template = !cache.has_template();
        let template_hir = if needs_template {
            let checkpoint = backbone_params.clone();
            let mut loader = CheckpointParamLoader::new(checkpoint);
            let mut wm = WeightMap::from_weight_loader(&mut loader)?;
            let (hir, params) = build_tts_backbone_prefill_shard_hir_dynamic_ext(
                cfg,
                &mut wm,
                1,
                max_seq,
                layer_start,
                layer_end,
                true,
            )?;
            if graph_params.is_none() {
                *graph_params = Some(params);
            }
            Some(hir)
        } else {
            None
        };

        let input_name = if layer_start == 0 {
            "inputs_embeds"
        } else {
            "hidden_in"
        };
        let compiled = cache
            .get_or_specialize(
                seq as u64,
                &binding,
                || template_hir.expect("prefill shard template"),
                &opts,
            )
            .map_err(|e| {
                anyhow::anyhow!("prefill shard [{layer_start},{layer_end}) seq={seq}: {e}")
            })?;
        if needs_upload {
            let params = graph_params
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("graph params missing"))?;
            for (name, data) in params {
                compiled.set_param(name, data);
            }
        }

        let outputs = if layer_start == 0 {
            compiled.run(&[(input_name, embeds)])
        } else {
            compiled.run(&[(input_name, hidden_in.as_slice())])
        };
        ensure!(
            outputs.len() == 1 + 2 * n_shard,
            "prefill shard [{layer_start},{layer_end}) outputs {} != {}",
            outputs.len(),
            1 + 2 * n_shard
        );
        hidden_in = outputs[0].clone();
        for local in 0..n_shard {
            let global = layer_start + local;
            all_kv[2 * global] = outputs[1 + 2 * local].clone();
            all_kv[2 * global + 1] = outputs[1 + 2 * local + 1].clone();
        }
    }

    let mut outs = Vec::with_capacity(1 + 2 * n_layers);
    outs.push(hidden_in);
    outs.extend(all_kv);
    Ok(outs)
}

pub(crate) fn run_decode_sharded(
    state: &mut ShardedLmState,
    cfg: &TextConfig,
    device: Device,
    decode_max_past: usize,
    backbone_params: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    graph_params: &mut Option<HashMap<String, Vec<f32>>>,
    past_len: usize,
    embed: &[f32],
    cos: &[f32],
    sin: &[f32],
    kv_caches: &[Vec<f32>],
    kv_dim: usize,
    n_layers: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut hidden_in = embed.to_vec();
    let mut merged_kv = kv_caches.to_vec();

    for (shard_idx, &(layer_start, layer_end)) in state.shards.iter().enumerate() {
        let n_shard = layer_end - layer_start;
        let runtime = &mut state.runtimes[shard_idx];
        if runtime.decode_cache.is_none() {
            runtime.decode_cache = Some(BucketedCompileCache::power_of_two_ladder(
                device,
                1,
                decode_max_past.max(1) as u64,
            ));
        }
        let bucket_idx = runtime
            .decode_cache
            .as_ref()
            .expect("decode cache")
            .bucket_for(past_len as u64)
            .ok_or_else(|| anyhow::anyhow!("past_len {past_len} outside decode buckets"))?;
        let upper = runtime
            .decode_cache
            .as_ref()
            .expect("decode cache")
            .buckets()
            .nth(bucket_idx)
            .map(|r| r.end - 1)
            .unwrap() as usize;

        if !runtime.decode_loaded_buckets.contains(&bucket_idx) {
            let checkpoint = backbone_params.clone();
            let (hir, params) = {
                let mut loader = CheckpointParamLoader::new(checkpoint);
                let mut wm = WeightMap::from_weight_loader(&mut loader)?;
                build_tts_backbone_decode_shard_hir_sized_ext(
                    cfg,
                    &mut wm,
                    1,
                    upper,
                    layer_start,
                    layer_end,
                    true,
                )?
            };
            if graph_params.is_none() {
                *graph_params = Some(params.clone());
            }
            let opts = decode_compile_options(device);
            metal_decode_compile_guard(device, || {
                let cache = runtime.decode_cache.as_mut().expect("decode cache");
                let (_, compiled) = cache
                    .get_or_compile_hir_with_options(past_len as u64, |_| hir, &opts)
                    .expect("decode shard compile");
                for (name, data) in &params {
                    compiled.set_param(name, data);
                }
            });
            runtime.decode_loaded_buckets.insert(bucket_idx);
        }

        let mask_len = upper + 1;
        let mut mask = vec![0.0f32; mask_len];
        for v in mask.iter_mut().take(past_len + 1) {
            *v = 1.0;
        }

        let mut padded_k: Vec<Vec<f32>> = Vec::with_capacity(n_shard);
        let mut padded_v: Vec<Vec<f32>> = Vec::with_capacity(n_shard);
        for local in 0..n_shard {
            let global = layer_start + local;
            let src_k = &merged_kv[2 * global];
            let src_v = &merged_kv[2 * global + 1];
            let mut pk = vec![0f32; upper * kv_dim];
            let mut pv = vec![0f32; upper * kv_dim];
            pk[..src_k.len()].copy_from_slice(src_k);
            pv[..src_v.len()].copy_from_slice(src_v);
            padded_k.push(pk);
            padded_v.push(pv);
        }

        let opts = decode_compile_options(device);
        let (_, compiled) = runtime
            .decode_cache
            .as_mut()
            .expect("decode cache")
            .get_or_compile_hir_with_options(
                past_len as u64,
                |_| unreachable!("decode shard {shard_idx} bucket {bucket_idx} loaded"),
                &opts,
            )
            .expect("decode shard run");

        let input_name = if layer_start == 0 {
            "inputs_embeds"
        } else {
            "hidden_in"
        };
        let past_k_names: Vec<String> = (0..n_shard)
            .map(|local| format!("past_k_{local}"))
            .collect();
        let past_v_names: Vec<String> = (0..n_shard)
            .map(|local| format!("past_v_{local}"))
            .collect();
        let mut inputs: Vec<(&str, &[f32])> = Vec::with_capacity(4 + 2 * n_shard);
        inputs.push((input_name, hidden_in.as_slice()));
        inputs.push(("rope_cos", cos));
        inputs.push(("rope_sin", sin));
        inputs.push(("mask", mask.as_slice()));
        for local in 0..n_shard {
            inputs.push((past_k_names[local].as_str(), padded_k[local].as_slice()));
            inputs.push((past_v_names[local].as_str(), padded_v[local].as_slice()));
        }

        let active = past_len + 1;
        compiled.set_active_extent(Some((active, upper + 1)));
        let raw = compiled.run(&inputs);
        compiled.set_active_extent(None);

        hidden_in = raw[0].clone();
        let real_kv_len = active * kv_dim;
        for local in 0..n_shard {
            let global = layer_start + local;
            let k = &raw[1 + 2 * local];
            let v = &raw[2 + 2 * local];
            ensure!(
                k.len() >= real_kv_len && v.len() >= real_kv_len,
                "decode shard kv[{global}] short"
            );
            merged_kv[2 * global] = k[..real_kv_len].to_vec();
            merged_kv[2 * global + 1] = v[..real_kv_len].to_vec();
        }
    }

    let mut outs = Vec::with_capacity(1 + 2 * n_layers);
    outs.push(hidden_in);
    outs.extend(merged_kv);
    Ok(outs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_26_layers_by_4() {
        let shards = partition_layers(26, 4);
        assert_eq!(shards.len(), 7);
        assert_eq!(shards[0], (0, 4));
        assert_eq!(shards.last().copied(), Some((24, 26)));
    }
}
