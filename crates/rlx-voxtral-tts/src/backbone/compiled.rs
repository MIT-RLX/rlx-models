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

//! RLX-compiled Ministral backbone — prefill + cached KV decode on GPU.

use crate::backbone::compiled_sharded::{ShardedLmState, run_decode_sharded, run_prefill_sharded};
use crate::backbone::lm::MinistralLm;
use crate::config::TextConfig;
use crate::lm_flow::{
    build_tts_backbone_decode_hir_sized_ext, build_tts_backbone_prefill_hir_dynamic_ext,
};
use crate::load::{VoxtralTtsWeightStore, WeightSnapshot};
use crate::weights::{CheckpointParamLoader, snapshot_backbone_params};
use anyhow::{Result, ensure};
use ndarray::{Array1, Array2, ArrayView2};
use rlx_core::flow_bridge::compile_options_from_profile;
use rlx_core::weight_map::WeightMap;
use rlx_flow::CompileProfile;
use rlx_ir::DimBinding;
use rlx_ir::hir::HirModule;
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_llama32::rope::{resolve_inv_freq, rope_slice};
use rlx_runtime::CompileOptions;
use rlx_runtime::Device;
use rlx_runtime::compile_cache::{BucketedCompileCache, DynamicDimCompileCache};
use std::collections::{HashMap, HashSet};

const DEFAULT_DECODE_MAX_PAST: usize = 8192;
const DEFAULT_PREFILL_MAX_SEQ: usize = 8192;
pub(crate) const PREFILL_DYNAMIC_CACHE_CAP: usize = 8;

pub(crate) fn metal_decode_compile_guard<R, F>(device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    if device == Device::Metal {
        rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
        let out = f();
        rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
        out
    } else {
        f()
    }
}

pub(crate) fn portable_gpu_compile_options(
    profile: &CompileProfile,
    device: Device,
    binding: Option<DimBinding>,
) -> CompileOptions {
    let mut profile = profile.clone();
    if matches!(device, Device::Gpu | Device::Vulkan) {
        profile.fusion.skip = true;
    }
    let mut opts = compile_options_from_profile(&profile, device, KernelDispatchConfig::default());
    if let Some(b) = binding {
        opts = opts.dim_binding(b);
    }
    opts
}

pub(crate) fn decode_compile_options(device: Device) -> CompileOptions {
    portable_gpu_compile_options(&CompileProfile::llama32_decode(), device, None)
}

pub(crate) fn prefill_compile_options(device: Device, binding: DimBinding) -> CompileOptions {
    portable_gpu_compile_options(&CompileProfile::llama32_prefill(), device, Some(binding))
}

pub struct CompiledMinistralLm {
    cfg: TextConfig,
    store: VoxtralTtsWeightStore,
    device: Device,
    /// CPU fallback only — not loaded on GPU.
    eager: Option<MinistralLm>,
    backbone_params: Option<WeightSnapshot>,
    graph_params: Option<HashMap<String, Vec<f32>>>,
    inv_freq: Vec<f64>,
    hidden: usize,
    kv_dim: usize,
    n_layers: usize,
    prefill_max_seq: usize,
    decode_max_past: usize,
    past_len: usize,
    kv_caches: Vec<Vec<f32>>,
    prefill_dynamic_cache: Option<DynamicDimCompileCache>,
    decode_bucket_cache: Option<BucketedCompileCache>,
    decode_loaded_buckets: HashSet<usize>,
    past_kv_keys: Vec<String>,
    sharded: Option<ShardedLmState>,
    lora: Option<crate::lora::LoraBank>,
}

impl CompiledMinistralLm {
    pub fn open(
        store: &VoxtralTtsWeightStore,
        cfg: &TextConfig,
        device: Device,
        eager_tensors: Option<&WeightSnapshot>,
        lora: Option<&crate::lora::LoraBank>,
    ) -> Result<Self> {
        let llama = cfg.llama_config();
        let lora_owned = lora.cloned();
        let eager = if device == Device::Cpu {
            let tensors = eager_tensors.ok_or_else(|| {
                anyhow::anyhow!("CPU compiled LM requires backbone tensors for eager fallback")
            })?;
            Some(MinistralLm::from_tensors_with_lora(tensors, cfg, lora)?)
        } else {
            None
        };
        let decode_max_past = decode_max_past(cfg);
        let prefill_max_seq = prefill_max_seq(cfg);
        let n_layers = cfg.num_hidden_layers;
        let past_kv_keys: Vec<String> = (0..n_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        Ok(Self {
            hidden: cfg.hidden_size,
            kv_dim: llama.kv_proj_dim(),
            n_layers,
            cfg: cfg.clone(),
            store: store.clone(),
            device,
            eager,
            backbone_params: None,
            graph_params: None,
            inv_freq: resolve_inv_freq(&llama, None),
            prefill_max_seq,
            decode_max_past,
            past_len: 0,
            kv_caches: Vec::new(),
            prefill_dynamic_cache: None,
            decode_bucket_cache: None,
            decode_loaded_buckets: HashSet::new(),
            past_kv_keys,
            sharded: ShardedLmState::new(device, n_layers),
            lora: lora_owned,
        })
    }

    pub fn reset_cache(&mut self) {
        self.past_len = 0;
        self.kv_caches.clear();
        self.prefill_dynamic_cache = None;
        self.decode_bucket_cache = None;
        self.decode_loaded_buckets.clear();
        if let Some(sharded) = self.sharded.as_mut() {
            sharded.reset_caches();
        }
        if let Some(eager) = self.eager.as_mut() {
            eager.reset_cache();
        }
    }

    pub fn forward(&mut self, inputs_embeds: ArrayView2<f32>) -> Result<Array2<f32>> {
        let (seq, h) = inputs_embeds.dim();
        ensure!(h == self.hidden, "hidden mismatch");
        if self.device == Device::Cpu {
            return self
                .eager
                .as_mut()
                .expect("cpu eager lm")
                .forward(inputs_embeds);
        }

        if self.past_len == 0 || seq > 1 {
            let flat: Vec<f32> = inputs_embeds.iter().copied().collect();
            let outputs = self.run_prefill(seq, &flat)?;
            ensure!(
                outputs.len() == 1 + 2 * self.n_layers,
                "prefill outputs {} != 1 + 2*layers",
                outputs.len()
            );
            self.kv_caches = outputs[1..].to_vec();
            self.past_len = seq;
            if matches!(self.device, Device::Gpu | Device::Vulkan) {
                // Decode compile is peaky on portable GPU backends; drop prefill graphs first.
                self.prefill_dynamic_cache = None;
                if let Some(sharded) = self.sharded.as_mut() {
                    sharded.drop_prefill_caches();
                }
            }
            return Ok(flat_to_array2(&outputs[0], seq, self.hidden));
        }

        ensure!(
            seq == 1,
            "compiled LM decode expects one token embed per step"
        );
        ensure!(
            self.past_len <= self.decode_max_past,
            "past_len {} exceeds compiled decode cap {} — raise RLX_VOXTRAL_TTS_MAX_PAST or use --eager-lm",
            self.past_len,
            self.decode_max_past
        );
        let embed: Vec<f32> = inputs_embeds.iter().copied().collect();
        let (cos, sin) = rope_slice(&self.inv_freq, self.past_len);
        let outputs = self.run_decode(self.past_len, &embed, &cos, &sin)?;
        ensure!(
            outputs.len() == 1 + 2 * self.n_layers,
            "decode outputs {} != 1 + 2*layers",
            outputs.len()
        );
        self.kv_caches = outputs[1..].to_vec();
        self.past_len += 1;
        Ok(flat_to_array2(&outputs[0], 1, self.hidden))
    }

    pub fn last_hidden(&self, hidden: &Array2<f32>) -> Array1<f32> {
        hidden.row(hidden.dim().0 - 1).to_owned()
    }

    fn run_prefill(&mut self, seq: usize, embeds: &[f32]) -> Result<Vec<Vec<f32>>> {
        if self.sharded.is_some() {
            let params = self.ensure_backbone_params()?.clone();
            let mut sharded = self.sharded.take().unwrap();
            let max_seq = self.prefill_max_seq;
            let n_layers = self.n_layers;
            let result = run_prefill_sharded(
                &mut sharded,
                &self.cfg,
                self.device,
                max_seq,
                &params,
                &mut self.graph_params,
                seq,
                embeds,
                n_layers,
            );
            self.sharded = Some(sharded);
            return result;
        }
        let binding = DimBinding::batch_seq(1, seq);
        let opts = prefill_compile_options(self.device, binding.clone());
        let max_seq = self.prefill_max_seq;
        if self.prefill_dynamic_cache.is_none() {
            self.prefill_dynamic_cache = Some(DynamicDimCompileCache::new(
                self.device,
                PREFILL_DYNAMIC_CACHE_CAP,
            ));
        }
        let needs_upload = !self
            .prefill_dynamic_cache
            .as_ref()
            .expect("prefill cache")
            .contains(seq as u64);
        let needs_template = !self
            .prefill_dynamic_cache
            .as_ref()
            .expect("prefill cache")
            .has_template();
        let template_hir = if needs_template {
            Some(self.build_prefill_template_hir(max_seq)?)
        } else {
            None
        };

        let cache = self.prefill_dynamic_cache.as_mut().expect("prefill cache");
        let compiled = cache
            .get_or_specialize(
                seq as u64,
                &binding,
                || template_hir.expect("prefill template HIR"),
                &opts,
            )
            .map_err(|e| anyhow::anyhow!("prefill specialize seq={seq}: {e}"))?;
        if needs_upload {
            let params = self
                .graph_params
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("graph params missing after template build"))?;
            for (name, data) in params {
                compiled.set_param(name, data);
            }
        }
        Ok(compiled.run(&[("inputs_embeds", embeds)]))
    }

    fn run_decode(
        &mut self,
        past_len: usize,
        embed: &[f32],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        ensure!(
            self.kv_caches.len() == 2 * self.n_layers,
            "missing KV cache (past_len={past_len})"
        );
        if self.sharded.is_some() {
            let params = self.ensure_backbone_params()?.clone();
            let mut sharded = self.sharded.take().unwrap();
            let decode_max_past = self.decode_max_past;
            let kv_dim = self.kv_dim;
            let n_layers = self.n_layers;
            let kv = self.kv_caches.clone();
            let result = run_decode_sharded(
                &mut sharded,
                &self.cfg,
                self.device,
                decode_max_past,
                &params,
                &mut self.graph_params,
                past_len,
                embed,
                cos,
                sin,
                &kv,
                kv_dim,
                n_layers,
            );
            self.sharded = Some(sharded);
            return result;
        }
        for (i, cache) in self.kv_caches.iter().enumerate() {
            ensure!(
                cache.len() == past_len * self.kv_dim,
                "KV cache[{i}] len {} != past_len*kv_dim ({past_len}*{})",
                cache.len(),
                self.kv_dim
            );
        }

        if self.decode_bucket_cache.is_none() {
            self.decode_bucket_cache = Some(BucketedCompileCache::power_of_two_ladder(
                self.device,
                1,
                self.decode_max_past.max(1) as u64,
            ));
        }
        let bucket_idx = self
            .decode_bucket_cache
            .as_ref()
            .expect("decode buckets")
            .bucket_for(past_len as u64)
            .ok_or_else(|| anyhow::anyhow!("past_len {past_len} outside decode buckets"))?;
        let upper = self
            .decode_bucket_cache
            .as_ref()
            .expect("decode buckets")
            .buckets()
            .nth(bucket_idx)
            .map(|r| r.end - 1)
            .unwrap() as usize;

        if !self.decode_loaded_buckets.contains(&bucket_idx) {
            self.load_decode_bucket(past_len, upper, bucket_idx)?;
        }

        let mask_len = upper + 1;
        let mut mask = vec![0.0f32; mask_len];
        for v in mask.iter_mut().take(past_len + 1) {
            *v = 1.0;
        }

        let kv_dim = self.kv_dim;
        let n_layers = self.n_layers;
        let padded_k: Vec<Vec<f32>> = (0..n_layers)
            .map(|i| {
                let src = &self.kv_caches[2 * i];
                let mut out = vec![0f32; upper * kv_dim];
                out[..src.len()].copy_from_slice(src);
                out
            })
            .collect();
        let padded_v: Vec<Vec<f32>> = (0..n_layers)
            .map(|i| {
                let src = &self.kv_caches[2 * i + 1];
                let mut out = vec![0f32; upper * kv_dim];
                out[..src.len()].copy_from_slice(src);
                out
            })
            .collect();

        let opts = decode_compile_options(self.device);
        let (_, compiled) = self
            .decode_bucket_cache
            .as_mut()
            .expect("decode buckets")
            .get_or_compile_hir_with_options(
                past_len as u64,
                |_| unreachable!("decode bucket {bucket_idx} was loaded above"),
                &opts,
            )
            .expect("decode bucket run");

        let mut inputs: Vec<(&str, &[f32])> = Vec::with_capacity(4 + 2 * n_layers);
        inputs.push(("inputs_embeds", embed));
        inputs.push(("rope_cos", cos));
        inputs.push(("rope_sin", sin));
        inputs.push(("mask", mask.as_slice()));
        for i in 0..n_layers {
            inputs.push((self.past_kv_keys[2 * i].as_str(), padded_k[i].as_slice()));
            inputs.push((
                self.past_kv_keys[2 * i + 1].as_str(),
                padded_v[i].as_slice(),
            ));
        }

        let active = past_len + 1;
        let padded_seq = upper + 1;
        compiled.set_active_extent(Some((active, padded_seq)));
        let raw = compiled.run(&inputs);
        compiled.set_active_extent(None);

        let real_kv_len = active * kv_dim;
        let mut outs = Vec::with_capacity(1 + 2 * n_layers);
        outs.push(raw[0].clone());
        for layer in 0..n_layers {
            let k = raw
                .get(1 + 2 * layer)
                .ok_or_else(|| anyhow::anyhow!("decode missing past_k_{layer}"))?;
            let v = raw
                .get(2 + 2 * layer)
                .ok_or_else(|| anyhow::anyhow!("decode missing past_v_{layer}"))?;
            ensure!(
                k.len() >= real_kv_len && v.len() >= real_kv_len,
                "decode kv[{layer}] shorter than {real_kv_len}"
            );
            outs.push(k[..real_kv_len].to_vec());
            outs.push(v[..real_kv_len].to_vec());
        }
        Ok(outs)
    }

    fn build_prefill_template_hir(&mut self, max_seq: usize) -> Result<HirModule> {
        let checkpoint = self.ensure_backbone_params()?.clone();
        let mut loader = CheckpointParamLoader::new(checkpoint);
        let mut wm = WeightMap::from_weight_loader(&mut loader)?;
        let (hir, params) =
            build_tts_backbone_prefill_hir_dynamic_ext(&self.cfg, &mut wm, 1, max_seq, true)?;
        if self.graph_params.is_none() {
            self.graph_params = Some(params);
        }
        Ok(hir)
    }

    fn load_decode_bucket(
        &mut self,
        past_len: usize,
        upper: usize,
        bucket_idx: usize,
    ) -> Result<()> {
        let checkpoint = self.ensure_backbone_params()?.clone();
        let cfg = self.cfg.clone();
        let (hir, params) = {
            let mut loader = CheckpointParamLoader::new(checkpoint);
            let mut wm = WeightMap::from_weight_loader(&mut loader)?;
            build_tts_backbone_decode_hir_sized_ext(&cfg, &mut wm, 1, upper, true)?
        };
        if self.graph_params.is_none() {
            self.graph_params = Some(params.clone());
        }
        let opts = decode_compile_options(self.device);
        let device = self.device;
        metal_decode_compile_guard(device, || {
            let cache = self.decode_bucket_cache.as_mut().expect("decode buckets");
            let (_, compiled) = cache
                .get_or_compile_hir_with_options(past_len as u64, |_| hir, &opts)
                .expect("decode bucket compile");
            for (name, data) in &params {
                compiled.set_param(name, data);
            }
        });
        self.decode_loaded_buckets.insert(bucket_idx);
        Ok(())
    }

    fn ensure_backbone_params(&mut self) -> Result<&WeightSnapshot> {
        if self.backbone_params.is_none() {
            let mut params = snapshot_backbone_params(&self.store)?;
            if let Some(ref lora) = self.lora {
                crate::lora::apply_lora_to_backbone(&mut params, lora)?;
            }
            self.backbone_params = Some(params);
        }
        Ok(self.backbone_params.as_ref().unwrap())
    }
}

fn prefill_max_seq(cfg: &TextConfig) -> usize {
    if let Ok(raw) = std::env::var("RLX_VOXTRAL_TTS_MAX_SEQ") {
        if let Ok(v) = raw.parse::<usize>() {
            if v > 0 {
                return v.min(cfg.max_position_embeddings);
            }
        }
    }
    DEFAULT_PREFILL_MAX_SEQ.min(cfg.max_position_embeddings)
}

fn decode_max_past(cfg: &TextConfig) -> usize {
    if let Ok(raw) = std::env::var("RLX_VOXTRAL_TTS_MAX_PAST") {
        if let Ok(v) = raw.parse::<usize>() {
            if v > 0 {
                return v.min(cfg.max_position_embeddings);
            }
        }
    }
    DEFAULT_DECODE_MAX_PAST.min(cfg.max_position_embeddings)
}

fn flat_to_array2(flat: &[f32], seq: usize, hidden: usize) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((seq, hidden));
    for t in 0..seq {
        for h in 0..hidden {
            let idx = t * hidden + h;
            if idx < flat.len() {
                out[[t, h]] = flat[idx];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_to_array2_roundtrip() {
        let hidden = 4;
        let seq = 2;
        let flat: Vec<f32> = (0..(seq * hidden)).map(|i| i as f32).collect();
        let arr = flat_to_array2(&flat, seq, hidden);
        assert_eq!(arr[[0, 3]], 3.0);
        assert_eq!(arr[[1, 0]], 4.0);
    }

    #[test]
    fn decode_bucket_ladder_covers_short_past() {
        let cache = BucketedCompileCache::power_of_two_ladder(Device::Cpu, 1, 64);
        assert!(cache.bucket_for(4).is_some());
        assert!(cache.bucket_for(5).is_some());
    }
}
