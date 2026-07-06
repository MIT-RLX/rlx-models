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

//! Compile caches for LM prefill / MTP / decode.

use crate::compile_support::{
    lm_active_extent_enabled, lm_decode_compile_options, lm_gpu_kv_enabled, lm_host_device,
    metal_lm_compile_guard,
};
use crate::config::LocateAnythingConfig;
use crate::kv_buckets::locateanything_kv_bucket_ranges_for_device;
use crate::lm_flow::{
    build_locateanything_decode_built_ext, build_locateanything_mtp_kv_built,
    build_locateanything_prefill_built,
};
use crate::load::LocateAnythingWeightStore;
use crate::mask::{attn_bias_for_incremental_padded, mtp_decode_mask_padded};
use crate::weights::CheckpointLmWeightLoader;
use anyhow::Result;
use rlx_core::flow_util::{
    bucket_cache_ensure_built, compile_cache_ensure_built, graph_from_built,
};
use rlx_core::{
    GpuKvCacheSet, KvCacheState, prefill_cache_key, run_bucketed_kv_decode,
    run_bucketed_kv_decode_gpu, run_bucketed_kv_mtp_gpu, sync_gpu_kv_to_host,
};
use std::sync::Arc;

use rlx_runtime::Device;
use rlx_runtime::compile_cache::{BucketedCompileCache, CacheRunInput, CompileCache};

fn session_use_gpu_kv(device: Device) -> bool {
    lm_gpu_kv_enabled(device)
}

fn session_set_active_extent(
    compiled: &mut rlx_runtime::CompiledGraph,
    device: Device,
    extent: (usize, usize),
) {
    if lm_active_extent_enabled(device) {
        compiled.set_active_extent(Some(extent));
    }
}

fn session_clear_active_extent(compiled: &mut rlx_runtime::CompiledGraph, device: Device) {
    if lm_active_extent_enabled(device) {
        compiled.set_active_extent(None);
    }
}

/// Cached LM graphs + weight snapshot for one runner device.
pub struct LmSessionCaches {
    lm_store: Option<Arc<LocateAnythingWeightStore>>,
    pub projector: std::collections::HashMap<usize, rlx_runtime::CompiledGraph>,
    _device: Device,
    prefill: CompileCache,
    decode_causal: BucketedCompileCache,
    decode_mtp: BucketedCompileCache,
    mtp: BucketedCompileCache,
    #[allow(dead_code)]
    max_past: usize,
    compile_opts_decode: rlx_runtime::CompileOptions,
    device: Device,
    gpu_kv: GpuKvCacheSet,
}

impl LmSessionCaches {
    pub fn new(device: Device, max_past: usize) -> Self {
        let max_past = max_past.max(1);
        let host = lm_host_device(device);
        let bucket_ranges = locateanything_kv_bucket_ranges_for_device(host, max_past);
        Self {
            lm_store: None,
            projector: std::collections::HashMap::new(),
            _device: device,
            device,
            prefill: CompileCache::new(host, 8),
            decode_causal: BucketedCompileCache::new(host, bucket_ranges.clone()),
            decode_mtp: BucketedCompileCache::new(host, bucket_ranges.clone()),
            mtp: BucketedCompileCache::new(host, bucket_ranges),
            max_past,
            compile_opts_decode: lm_decode_compile_options(host),
            gpu_kv: GpuKvCacheSet::default(),
        }
    }

    /// Invalidate GPU K/V handle bindings (after MTP block or host KV rewrite).
    pub fn reset_gpu_kv(&mut self) {
        self.gpu_kv.reset();
    }

    /// Drop decode/MTP GPU bindings after an MTP block advanced `past_len`.
    pub fn reset_decode_after_mtp(&mut self) {
        self.gpu_kv.reset_decode_after_mtp();
    }

    /// Copy GPU-resident K/V into `kv` before a host-path MTP forward.
    pub fn sync_kv_from_gpu(
        &mut self,
        cfg: &LocateAnythingConfig,
        past_len: usize,
        kv: &mut KvCacheState,
    ) -> Result<()> {
        if !session_use_gpu_kv(self.device) {
            return Ok(());
        }
        let layers = cfg.text_config.num_hidden_layers;
        let kv_dim = cfg.text_config.num_key_value_heads * cfg.text_config.head_dim();
        let keys = if past_len > 0 {
            [past_len as u64 - 1, past_len as u64]
        } else {
            [0, 0]
        };
        for &key in &keys {
            let compiled = if self.gpu_kv.causal.upper != 0 {
                self.decode_causal.compiled_for_key_mut(key)
            } else if self.gpu_kv.mtp.upper != 0 {
                self.mtp.compiled_for_key_mut(key)
            } else {
                return Ok(());
            };
            if let Some(compiled) = compiled {
                return sync_gpu_kv_to_host(compiled, kv, kv_dim, layers);
            }
        }
        Ok(())
    }

    /// Pin mmap-backed LM weights for compile caches (no full f32 snapshot in RAM).
    pub fn ensure_lm_store(
        &mut self,
        store: Arc<LocateAnythingWeightStore>,
    ) -> Arc<LocateAnythingWeightStore> {
        if self.lm_store.is_none() {
            self.lm_store = Some(store);
        }
        Arc::clone(self.lm_store.as_ref().expect("lm store"))
    }

    fn lm_loader(store: &Arc<LocateAnythingWeightStore>) -> CheckpointLmWeightLoader {
        CheckpointLmWeightLoader::new(Arc::clone(store))
    }

    pub fn projector_graph(
        &mut self,
        n_tokens: usize,
        build: impl FnOnce() -> Result<rlx_runtime::CompiledGraph>,
    ) -> Result<&mut rlx_runtime::CompiledGraph> {
        if let std::collections::hash_map::Entry::Vacant(e) = self.projector.entry(n_tokens) {
            e.insert(build()?);
        }
        Ok(self.projector.get_mut(&n_tokens).expect("projector"))
    }

    pub fn prefill_with_kv(
        &mut self,
        cfg: &LocateAnythingConfig,
        seq: usize,
        inputs_embeds: &[f32],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let key = prefill_cache_key(1, seq);
        let cfg = cfg.clone();
        let store = Arc::clone(
            self.lm_store
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("lm store missing"))?,
        );
        let mut loader = Self::lm_loader(&store);
        let built = build_locateanything_prefill_built(&cfg, &mut loader, 1, seq, true, true)?;
        let compiled = metal_lm_compile_guard(self.device, || {
            compile_cache_ensure_built(&mut self.prefill, key, built)
        })?;
        let outs = metal_lm_compile_guard(self.device, || {
            compiled.run(&[("inputs_embeds", inputs_embeds)])
        });
        let kv_start = 1usize;
        self.reset_gpu_kv();
        Ok((outs[0].clone(), outs[kv_start..].to_vec()))
    }

    pub fn mtp_logits(
        &mut self,
        cfg: &LocateAnythingConfig,
        past_len: usize,
        q_len: usize,
        inputs_embeds: &[f32],
        full_mask_2d: &[f32],
        full_seq: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
        kv: &mut KvCacheState,
    ) -> Result<(Vec<f32>, KvCacheState)> {
        let layers = cfg.text_config.num_hidden_layers;
        let nh = cfg.text_config.num_attention_heads;
        let kv_dim = cfg.text_config.num_key_value_heads * cfg.text_config.head_dim();
        let key = past_len as u64;
        let cfg = cfg.clone();
        let store = Arc::clone(
            self.lm_store
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("lm store missing"))?,
        );

        if session_use_gpu_kv(self.device) {
            let upper = self
                .mtp
                .bucket_upper_for_key(key)
                .ok_or_else(|| anyhow::anyhow!("past_len {past_len} outside MTP buckets"))?;

            let attn_bias = attn_bias_for_incremental_padded(
                1,
                nh,
                past_len,
                upper as usize,
                q_len,
                full_mask_2d,
                full_seq,
            );
            let fixed = [
                CacheRunInput {
                    name: "inputs_embeds",
                    data: inputs_embeds,
                    row_inner: None,
                },
                CacheRunInput {
                    name: "attn_bias",
                    data: &attn_bias,
                    row_inner: None,
                },
                CacheRunInput {
                    name: "rope_cos",
                    data: rope_cos,
                    row_inner: None,
                },
                CacheRunInput {
                    name: "rope_sin",
                    data: rope_sin,
                    row_inner: None,
                },
            ];
            let logits = metal_lm_compile_guard(self.device, || {
                run_bucketed_kv_mtp_gpu(
                    &mut self.mtp,
                    past_len,
                    q_len,
                    kv,
                    &mut self.gpu_kv.mtp,
                    kv_dim,
                    layers,
                    &fixed,
                    |upper| {
                        let mut loader = Self::lm_loader(&store);
                        let built = build_locateanything_mtp_kv_built(
                            &cfg,
                            &mut loader,
                            1,
                            upper as usize,
                            q_len,
                        )
                        .expect("mtp kv graph");
                        graph_from_built(built).expect("mtp kv graph from built")
                    },
                    &self.compile_opts_decode,
                )
            })?;
            let past_after = past_len + q_len;
            if let Some(compiled) = self.mtp.compiled_for_key_mut(key) {
                kv.past_len = past_after;
                sync_gpu_kv_to_host(compiled, kv, kv_dim, layers)?;
            } else {
                anyhow::bail!("mtp gpu: compiled graph missing for past_len {past_len}");
            }
            self.gpu_kv.reset_decode_after_mtp();
            return Ok((logits, kv.clone()));
        }

        let (upper, compiled) = metal_lm_compile_guard(self.device, || {
            bucket_cache_ensure_built(
                &mut self.mtp,
                key,
                |upper| {
                    let mut loader = Self::lm_loader(&store);
                    build_locateanything_mtp_kv_built(&cfg, &mut loader, 1, upper as usize, q_len)
                },
                &self.compile_opts_decode,
            )
        })
        .ok_or_else(|| anyhow::anyhow!("past_len {past_len} outside MTP buckets"))?;

        let attn_bias = attn_bias_for_incremental_padded(
            1,
            nh,
            past_len,
            upper as usize,
            q_len,
            full_mask_2d,
            full_seq,
        );
        let (padded_k, padded_v) = kv.pad_layers_to_upper(upper, kv_dim);
        let key_past = rlx_core::past_kv_input_names(layers);
        let mut run_in: Vec<(&str, &[f32])> = vec![
            ("inputs_embeds", inputs_embeds),
            ("attn_bias", &attn_bias),
            ("rope_cos", rope_cos),
            ("rope_sin", rope_sin),
        ];
        for i in 0..layers {
            run_in.push((key_past[2 * i].as_str(), padded_k[i].as_slice()));
            run_in.push((key_past[2 * i + 1].as_str(), padded_v[i].as_slice()));
        }
        // Bucket axis: we append `q_len` KV rows (not a single decode step).
        let actual_kv = past_len + q_len;
        let upper_kv = upper as usize + q_len;
        session_set_active_extent(compiled, self.device, (actual_kv, upper_kv));
        let outs = compiled.run(&run_in);
        session_clear_active_extent(compiled, self.device);
        let past_after = past_len + q_len;
        let kv = kv_state_from_runner(past_after, &outs[1..], layers, kv_dim)?;
        self.gpu_kv.reset_decode_after_mtp();
        Ok((outs[0].clone(), kv))
    }

    /// Single-token (or MTP-mask) decode; updates `kv` in place and returns logits only.
    pub fn decode_step_in_place(
        &mut self,
        cfg: &LocateAnythingConfig,
        past_len: usize,
        token: u32,
        rope_cos: &[f32],
        rope_sin: &[f32],
        mtp_window: Option<(usize, usize)>,
        kv: &mut KvCacheState,
    ) -> Result<Vec<f32>> {
        self.decode_step(cfg, past_len, token, rope_cos, rope_sin, mtp_window, kv)
    }

    fn decode_step(
        &mut self,
        cfg: &LocateAnythingConfig,
        past_len: usize,
        token: u32,
        rope_cos: &[f32],
        rope_sin: &[f32],
        mtp_window: Option<(usize, usize)>,
        kv: &mut KvCacheState,
    ) -> Result<Vec<f32>> {
        let layers = cfg.text_config.num_hidden_layers;
        let kv_dim = cfg.text_config.num_key_value_heads * cfg.text_config.head_dim();
        let token_f = [token as f32];
        let cfg_c = cfg.clone();
        let store = Arc::clone(
            self.lm_store
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("lm store missing"))?,
        );

        let mut fixed = vec![
            CacheRunInput {
                name: "input_ids",
                data: &token_f,
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_cos",
                data: rope_cos,
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_sin",
                data: rope_sin,
                row_inner: None,
            },
        ];

        if session_use_gpu_kv(self.device) {
            let binding = if mtp_window.is_some() {
                &mut self.gpu_kv.decode_mtp
            } else {
                &mut self.gpu_kv.causal
            };
            if let Some((block_size, past)) = mtp_window {
                let key = past_len as u64;
                let upper = self
                    .decode_mtp
                    .ensure_graph_with_params(
                        key,
                        |upper| {
                            let mut loader = Self::lm_loader(&store);
                            let built = build_locateanything_decode_built_ext(
                                &cfg_c,
                                &mut loader,
                                1,
                                upper as usize,
                                true,
                                false,
                            )
                            .expect("mtp decode graph");
                            graph_from_built(built).expect("mtp decode graph from built")
                        },
                        &self.compile_opts_decode,
                    )
                    .ok_or_else(|| {
                        anyhow::anyhow!("past_len {past_len} outside MTP decode buckets")
                    })?
                    .0;
                let mask = mtp_decode_mask_padded(block_size, past, upper as usize + 1);
                fixed.push(CacheRunInput {
                    name: "mask",
                    data: &mask,
                    row_inner: None,
                });
                return metal_lm_compile_guard(self.device, || {
                    run_bucketed_kv_decode_gpu(
                        &mut self.decode_mtp,
                        key,
                        past_len,
                        kv,
                        binding,
                        kv_dim,
                        layers,
                        &fixed,
                        |upper| {
                            let mut loader = Self::lm_loader(&store);
                            let built = build_locateanything_decode_built_ext(
                                &cfg_c,
                                &mut loader,
                                1,
                                upper as usize,
                                true,
                                false,
                            )
                            .expect("mtp decode graph");
                            graph_from_built(built).expect("mtp decode graph from built")
                        },
                        &self.compile_opts_decode,
                        false,
                    )
                });
            }
            return metal_lm_compile_guard(self.device, || {
                run_bucketed_kv_decode_gpu(
                    &mut self.decode_causal,
                    past_len as u64,
                    past_len,
                    kv,
                    binding,
                    kv_dim,
                    layers,
                    &fixed,
                    |upper| {
                        let mut loader = Self::lm_loader(&store);
                        let built = build_locateanything_decode_built_ext(
                            &cfg_c,
                            &mut loader,
                            1,
                            upper as usize,
                            false,
                            false,
                        )
                        .expect("causal decode graph");
                        graph_from_built(built).expect("causal decode graph from built")
                    },
                    &self.compile_opts_decode,
                    false,
                )
            });
        }

        if let Some((block_size, past)) = mtp_window {
            let key = past_len as u64;
            let (upper, _) = self
                .decode_mtp
                .ensure_graph_with_params(
                    key,
                    |upper| {
                        let mut loader = Self::lm_loader(&store);
                        let built = build_locateanything_decode_built_ext(
                            &cfg_c,
                            &mut loader,
                            1,
                            upper as usize,
                            true,
                            false,
                        )
                        .expect("mtp decode graph");
                        graph_from_built(built).expect("mtp decode graph from built")
                    },
                    &self.compile_opts_decode,
                )
                .ok_or_else(|| anyhow::anyhow!("past_len {past_len} outside MTP decode buckets"))?;
            let mask = mtp_decode_mask_padded(block_size, past, upper as usize + 1);
            fixed.push(CacheRunInput {
                name: "mask",
                data: &mask,
                row_inner: None,
            });
            let (logits, new_k, new_v) = metal_lm_compile_guard(self.device, || {
                run_bucketed_kv_decode(
                    &mut self.decode_mtp,
                    past_len,
                    kv,
                    kv_dim,
                    layers,
                    &fixed,
                    |upper| {
                        let mut loader = Self::lm_loader(&store);
                        let built = build_locateanything_decode_built_ext(
                            &cfg_c,
                            &mut loader,
                            1,
                            upper as usize,
                            true,
                            false,
                        )
                        .expect("mtp decode graph");
                        graph_from_built(built).expect("mtp decode graph from built")
                    },
                    &self.compile_opts_decode,
                )
            })?;
            kv.past_len = past_len + 1;
            let n = kv.past_len * kv_dim;
            for i in 0..layers {
                kv.layers_k[i] = take_kv_rows(&new_k[i], n);
                kv.layers_v[i] = take_kv_rows(&new_v[i], n);
            }
            return Ok(logits);
        }

        let (logits, new_k, new_v) = metal_lm_compile_guard(self.device, || {
            run_bucketed_kv_decode(
                &mut self.decode_causal,
                past_len,
                kv,
                kv_dim,
                layers,
                &fixed,
                |upper| {
                    let mut loader = Self::lm_loader(&store);
                    let built = build_locateanything_decode_built_ext(
                        &cfg_c,
                        &mut loader,
                        1,
                        upper as usize,
                        false,
                        false,
                    )
                    .expect("causal decode graph");
                    graph_from_built(built).expect("causal decode graph from built")
                },
                &self.compile_opts_decode,
            )
        })?;
        kv.past_len = past_len + 1;
        let n = kv.past_len * kv_dim;
        for i in 0..layers {
            kv.layers_k[i] = take_kv_rows(&new_k[i], n);
            kv.layers_v[i] = take_kv_rows(&new_v[i], n);
        }
        Ok(logits)
    }
}

/// Build KV state from prefill outputs (`[k0, v0, k1, v1, …]`).
///
/// When backend outputs are bucket-padded, only the first `past_len * kv_dim` floats
/// per layer are copied (avoids retaining multi‑MiB tail padding on the host).
pub fn kv_state_from_runner(
    past_len: usize,
    kv_flat: &[Vec<f32>],
    layers: usize,
    kv_dim: usize,
) -> Result<KvCacheState> {
    anyhow::ensure!(
        kv_flat.len() == 2 * layers,
        "expected {} kv tensors, got {}",
        2 * layers,
        kv_flat.len()
    );
    let n = past_len * kv_dim;
    let mut layers_k = Vec::with_capacity(layers);
    let mut layers_v = Vec::with_capacity(layers);
    for i in 0..layers {
        layers_k.push(take_kv_rows(&kv_flat[2 * i], n));
        layers_v.push(take_kv_rows(&kv_flat[2 * i + 1], n));
    }
    Ok(KvCacheState {
        past_len,
        layers_kv_base: vec![0; layers_k.len()],
        layers_k,
        layers_v,
    })
}

fn take_kv_rows(buf: &[f32], n: usize) -> Vec<f32> {
    if buf.len() <= n {
        buf.to_vec()
    } else {
        buf[..n].to_vec()
    }
}

/// Trim MTP KV to `prefix_past + committed` tokens (MTP may compute a full block).
pub fn truncate_kv_state(
    kv: KvCacheState,
    prefix_past: usize,
    committed: usize,
    kv_dim: usize,
) -> Result<KvCacheState> {
    let want = prefix_past + committed;
    if want >= kv.past_len {
        return Ok(kv);
    }
    let n = want * kv_dim;
    let mut layers_k = Vec::with_capacity(kv.layers_k.len());
    let mut layers_v = Vec::with_capacity(kv.layers_v.len());
    for (k, v) in kv.layers_k.iter().zip(kv.layers_v.iter()) {
        anyhow::ensure!(
            k.len() >= n && v.len() >= n,
            "truncate_kv_state: layer buffer too short (k={} v={} need {n})",
            k.len(),
            v.len()
        );
        layers_k.push(k[..n].to_vec());
        layers_v.push(v[..n].to_vec());
    }
    Ok(KvCacheState {
        past_len: want,
        layers_kv_base: vec![0; layers_k.len()],
        layers_k,
        layers_v,
    })
}
