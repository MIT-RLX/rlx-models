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

//! Compiled Qwen3-shaped talker (prefill + KV decode on `inputs_embeds`).

use crate::codec_frame::build_qwen3_tts_prefill_built;
use crate::codec_frame::{talker_decode_graph_parts, talker_decode_hir_parts};
use crate::compile_opts::{metal_compile_guard, metal_mpsgraph_run_guard, talker_compile_options};
use crate::config::TalkerConfig;
use crate::kv_util::commit_kv_layers;
use crate::load::{Qwen3TtsWeightStore, remap_talker_weights};
use crate::mrope::{talker_decode_rope_into, talker_prefill_rope_feeds, talker_rope_index_prefill};
use crate::progress::Progress;
use crate::talker::eager::TalkerEagerModel;
use crate::talker::math::{bucket_decode_hidden_into, linear_logits, sample_greedy};
use crate::talker::rope::{rope_prefill_feeds, rope_tables_full};
use crate::weights::weight_map_from_cache;
use anyhow::{Context, Result, ensure};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rlx_core::autoregressive::{
    KvCacheState, kv_from_prefill_outputs, run_bucketed_kv_decode,
    run_bucketed_kv_decode_hir_uniform,
};
use rlx_core::flow_util::compile_cache_ensure_built_with_options;
use rlx_core::{
    GpuKvBinding, device_supports_gpu_kv, install_gpu_kv_handles, run_bucketed_kv_decode_gpu,
    run_bucketed_kv_decode_gpu_hir, sync_gpu_kv_to_host,
};
use rlx_flow::CompileProfile;
use rlx_qwen3::qwen3_profile_near_weights;
use rlx_runtime::compile_cache::{BucketedCompileCache, CacheRunInput, CompileCache};
use rlx_runtime::{CompileOptions, Device};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub struct TalkerEngine {
    cfg: TalkerConfig,
    qwen3: rlx_qwen3::Qwen3Config,
    device: Device,
    hidden: usize,
    kv_dim: usize,
    n_layers: usize,
    codec_head: Array2<f32>,
    codec_head_flat: Vec<f32>,
    codec_vocab: usize,
    head_half: usize,
    rope_delta: i64,
    inv_freq: Vec<f64>,
    weights: Arc<crate::load::TensorSnapshot>,
    prefill_profile: CompileProfile,
    decode_profile: CompileProfile,
    past_len: usize,
    kv: KvCacheState,
    prefill_cache: CompileCache,
    /// CPU prefill cache when [`crate::compile_opts::talker_metal_cpu_prefill`] (Metal decode only).
    prefill_cache_cpu: Option<CompileCache>,
    decode_cache: BucketedCompileCache,
    codec_eos: u32,
    eager: Option<TalkerEagerModel>,
    /// Eager prefill + compiled decode on Metal (`METAL_COMPILED=1`).
    eager_prefill_only: bool,
    decode_embed: Vec<f32>,
    mask_buf: Vec<f32>,
    last_hidden: Vec<f32>,
    use_gpu_kv: bool,
    gpu_kv_binding: GpuKvBinding,
    decode_rope_cos: Vec<f32>,
    decode_rope_sin: Vec<f32>,
    decode_opts: CompileOptions,
    prefill_flat: Vec<f32>,
}

fn bucket_decode_mask_into(past_seq: usize, upper: usize, out: &mut Vec<f32>) {
    out.resize(upper + 1, 0.0);
    for (i, slot) in out.iter_mut().enumerate().take(upper + 1) {
        *slot = if i < past_seq || i == upper { 1.0 } else { 0.0 };
    }
}

fn talker_use_gpu_kv(device: Device) -> bool {
    if !device_supports_gpu_kv(device) {
        return false;
    }
    match std::env::var("RLX_QWEN3_TTS_GPU_KV").ok().as_deref() {
        Some("0") => false,
        Some("1") => true,
        _ => crate::synth_opts::megakernel_gpu_kv_default(device),
    }
}

pub fn talker_use_eager_for_device(device: Device) -> bool {
    match std::env::var("RLX_QWEN3_TTS_TALKER_EAGER").ok().as_deref() {
        Some("1") => return true,
        Some("0") => return false,
        _ => {}
    }
    if device == Device::Metal {
        if !crate::compile_opts::talker_metal_native_compile(device) {
            return true;
        }
        // Parity hybrid: CPU eager decode (~77 ms/frame) beats compiled CPU graphs (~260 ms/frame).
        return crate::gpu_pipeline::talker_eager_decode_default(device);
    }
    if device == Device::Mlx {
        if std::env::var("RLX_QWEN3_TTS_MLX_COMPILED").ok().as_deref() == Some("1") {
            return false;
        }
        if crate::gpu_pipeline::gpu_session_enabled(device)
            && std::env::var("RLX_QWEN3_TTS_TALKER_EAGER").ok().as_deref() != Some("1")
        {
            return false;
        }
        return true;
    }
    false
}

/// Upper cap for KV bucket ladder (power-of-two compile extents).
const TALKER_DECODE_BUCKET_MAX: u64 = 1024;

fn decode_bucket_max_for_horizon(horizon: usize) -> u64 {
    let h = horizon.max(8).next_power_of_two() as u64;
    h.min(TALKER_DECODE_BUCKET_MAX)
}

fn padded_kv_for_warmup(
    base: &KvCacheState,
    sim_past: usize,
    kv_dim: usize,
    n_layers: usize,
) -> KvCacheState {
    let n = sim_past * kv_dim;
    let mut kv = KvCacheState {
        past_len: sim_past,
        layers_k: vec![vec![0f32; n]; n_layers],
        layers_v: vec![vec![0f32; n]; n_layers],
    };
    let base_n = base.past_len * kv_dim;
    for layer in 0..n_layers {
        let copy = base_n.min(n).min(base.layers_k[layer].len());
        if copy > 0 {
            kv.layers_k[layer][..copy].copy_from_slice(&base.layers_k[layer][..copy]);
            kv.layers_v[layer][..copy].copy_from_slice(&base.layers_v[layer][..copy]);
        }
    }
    kv
}

impl TalkerEngine {
    pub fn open(
        store: &Qwen3TtsWeightStore,
        talker: &TalkerConfig,
        device: Device,
    ) -> Result<Self> {
        Self::open_at(store.model_dir(), store, talker, device)
    }

    pub fn open_at(
        model_dir: &Path,
        store: &Qwen3TtsWeightStore,
        talker: &TalkerConfig,
        device: Device,
    ) -> Result<Self> {
        let debug = std::env::var("RLX_QWEN3_TTS_OPEN_TIMING").ok().as_deref() == Some("1");
        let t = std::time::Instant::now();
        let mut talker_wm = store.load_talker_backbone()?;
        if debug {
            eprintln!(
                "[open.talker]   load_backbone:  {:.3}s",
                t.elapsed().as_secs_f64()
            );
        }
        let t = std::time::Instant::now();
        let weights = remap_talker_weights(&mut talker_wm)?;
        if debug {
            eprintln!(
                "[open.talker]   remap_weights:  {:.3}s",
                t.elapsed().as_secs_f64()
            );
        }
        let t = std::time::Instant::now();
        let r = Self::open_with_weights(model_dir, store, talker, weights, device);
        if debug {
            eprintln!(
                "[open.talker]   open_w_weights: {:.3}s",
                t.elapsed().as_secs_f64()
            );
        }
        r
    }

    pub fn open_with_weights(
        model_dir: &Path,
        store: &Qwen3TtsWeightStore,
        talker: &TalkerConfig,
        weights: HashMap<String, (Vec<f32>, Vec<usize>)>,
        device: Device,
    ) -> Result<Self> {
        let mut prefill = qwen3_profile_near_weights(model_dir, false);
        let mut decode = qwen3_profile_near_weights(model_dir, true);
        crate::compile_opts::tune_qwen3_profile(
            &mut prefill,
            crate::compile_opts::talker_prefill_profile_device(device),
        );
        crate::compile_opts::tune_qwen3_profile(
            &mut decode,
            crate::compile_opts::talker_decode_compile_device(device),
        );
        Self::open_with_weights_and_profiles(
            model_dir, store, talker, weights, device, prefill, decode,
        )
    }

    pub fn open_with_weights_and_profiles(
        _model_dir: &Path,
        store: &Qwen3TtsWeightStore,
        talker: &TalkerConfig,
        weights: HashMap<String, (Vec<f32>, Vec<usize>)>,
        device: Device,
        prefill_profile: CompileProfile,
        decode_profile: CompileProfile,
    ) -> Result<Self> {
        let qwen3 = talker.to_qwen3_config();
        let hidden = talker.hidden_size;
        let kv_dim = qwen3.kv_proj_dim();
        let n_layers = talker.num_hidden_layers;
        let head_half = talker.head_dim / 2;
        let inv_freq = crate::talker::rope::build_inv_freq(talker.head_dim, talker.rope_theta);

        let (head_data, head_shape) = store.take_codec_head()?;
        ensure!(head_shape.len() == 2, "codec_head rank");
        let codec_vocab = head_shape[0];
        let codec_head = Array2::from_shape_vec((codec_vocab, head_shape[1]), head_data.clone())
            .context("codec_head")?;

        let max_past = talker
            .max_position_embeddings
            .min(8192)
            .min(TALKER_DECODE_BUCKET_MAX as usize);
        let prefill_compile_device = crate::compile_opts::talker_compile_device(device);
        let decode_compile_device = crate::compile_opts::talker_decode_compile_device(device);
        let use_eager = talker_use_eager_for_device(device);
        let eager_prefill_only = !use_eager
            && device == Device::Metal
            && crate::compile_opts::talker_metal_native_compile(device);
        let compiled_path = !use_eager;
        let eager = if use_eager || eager_prefill_only {
            // Reuse the already-loaded backbone weights (saves a 0.5s reload).
            Some(TalkerEagerModel::open_from_map(&weights, talker)?)
        } else {
            None
        };
        let decode_opts = crate::compile_opts::talker_decode_compile_options(
            &decode_profile,
            decode_compile_device,
        );
        Ok(Self {
            cfg: talker.clone(),
            qwen3,
            device,
            hidden,
            kv_dim,
            n_layers,
            codec_head,
            codec_head_flat: head_data,
            codec_vocab,
            head_half,
            rope_delta: 0,
            inv_freq,
            weights: Arc::new(weights),
            prefill_profile,
            decode_profile,
            past_len: 0,
            kv: KvCacheState {
                past_len: 0,
                layers_k: vec![Vec::new(); n_layers],
                layers_v: vec![Vec::new(); n_layers],
            },
            prefill_cache: CompileCache::new(prefill_compile_device, 16),
            prefill_cache_cpu: if crate::compile_opts::talker_metal_cpu_prefill(device) {
                Some(CompileCache::new(Device::Cpu, 16))
            } else {
                None
            },
            decode_cache: BucketedCompileCache::power_of_two_ladder(
                decode_compile_device,
                1,
                max_past as u64,
            ),
            codec_eos: talker.codec_eos_token_id,
            eager,
            eager_prefill_only,
            decode_embed: vec![0f32; hidden],
            mask_buf: Vec::new(),
            last_hidden: vec![0f32; hidden],
            use_gpu_kv: talker_use_gpu_kv(device)
                && compiled_path
                && decode_compile_device == device,
            gpu_kv_binding: GpuKvBinding::default(),
            decode_rope_cos: vec![0f32; head_half],
            decode_rope_sin: vec![0f32; head_half],
            decode_opts,
            prefill_flat: Vec::new(),
        })
    }

    /// Bucket upper bound for talker decode at `past_seq` (power-of-two ladder).
    pub fn decode_bucket_upper(&self, past_seq: usize) -> usize {
        self.decode_upper_for_key(past_seq as u64)
            .unwrap_or(past_seq)
    }

    fn decode_upper_for_key(&self, key: u64) -> Option<usize> {
        self.decode_cache.bucket_for(key).and_then(|idx| {
            self.decode_cache
                .buckets()
                .nth(idx)
                .map(|r| (r.end - 1) as usize)
        })
    }

    /// Pre-compile the single decode bucket that contains `past_seq` (cheap first-frame insurance).
    pub fn precompile_decode_bucket_for_past(&mut self, past_seq: usize) -> Result<()> {
        if self.uses_eager_decode() {
            return Ok(());
        }
        let key = past_seq as u64;
        if self.decode_cache.bucket_for(key).is_none() {
            return Ok(());
        }
        let decode_dev = self.decode_compile_device();
        let opts = &self.decode_opts;
        let weights = Arc::clone(&self.weights);
        let qwen3 = self.qwen3.clone();
        let decode_profile = self.decode_profile.clone();
        let use_hir = crate::compile_opts::talker_decode_use_hir_compile(decode_dev);
        let ensure = || {
            if use_hir {
                self.decode_cache.ensure_hir_with_params(
                    key,
                    move |upper| {
                        talker_decode_hir_parts(&qwen3, weights.as_ref(), &decode_profile, upper)
                            .expect("talker decode hir")
                    },
                    opts,
                )
            } else {
                self.decode_cache.ensure_graph_with_params(
                    key,
                    move |upper| {
                        talker_decode_graph_parts(&qwen3, weights.as_ref(), &decode_profile, upper)
                            .expect("talker decode graph")
                    },
                    opts,
                )
            }
        };
        if decode_dev == Device::Metal {
            metal_mpsgraph_run_guard(self.device, || {
                let _ = ensure();
            });
        } else {
            let _ = ensure();
        }
        Ok(())
    }

    /// Pre-bind GPU K/V handles for horizon buckets (boundary crossings by default).
    pub fn preinstall_gpu_kv_horizon(&mut self, horizon: usize) -> Result<()> {
        if !self.use_gpu_kv {
            return Ok(());
        }
        let past = self.past_len;
        let sim_pasts: Vec<usize> = if crate::synth_opts::warmup_all_talk_buckets() {
            let max_key = decode_bucket_max_for_horizon(horizon);
            self.decode_cache
                .buckets()
                .cloned()
                .collect::<Vec<_>>()
                .into_iter()
                .filter_map(|range| {
                    let upper_u = range.end.saturating_sub(1);
                    if upper_u > max_key || upper_u < past as u64 || range.start > horizon as u64 {
                        return None;
                    }
                    Some(range.start.max(past as u64).max(1) as usize)
                })
                .collect()
        } else {
            let mut sims = vec![past];
            for range in self.decode_cache.buckets() {
                let b = range.start as usize;
                if b > past && b <= horizon {
                    sims.push(b);
                }
            }
            sims.sort_unstable();
            sims.dedup();
            sims
        };
        let base_kv = self.kv.clone();
        for sim_past in sim_pasts {
            let key = sim_past as u64;
            let upper = self.decode_upper_for_key(key).unwrap_or(sim_past) as u64;
            let kv = padded_kv_for_warmup(&base_kv, sim_past, self.kv_dim, self.n_layers);
            if let Some(compiled) = self.decode_cache.compiled_for_key_mut(key) {
                install_gpu_kv_handles(compiled, &kv, sim_past, upper, self.kv_dim, self.n_layers)?;
            }
        }
        if let Some(upper) = self.decode_upper_for_key(past as u64) {
            self.gpu_kv_binding.upper = upper as u64;
        }
        Ok(())
    }

    /// Bind GPU K/V for the current prefill bucket (no dry decode).
    ///
    /// Skips re-upload when handles are already live for this bucket — rebinding from
    /// host `kv` would clobber GPU-updated prefix rows after the first decode step.
    pub fn preinstall_gpu_kv_current(&mut self) -> Result<()> {
        if !self.use_gpu_kv {
            return Ok(());
        }
        let past = self.past_len;
        let key = past as u64;
        if let Some(upper) = self.decode_upper_for_key(key) {
            let upper_u = upper as u64;
            if let Some(compiled) = self.decode_cache.compiled_for_key_mut(key) {
                if compiled.has_gpu_handle("past_k_0") && self.gpu_kv_binding.upper == upper_u {
                    return Ok(());
                }
                install_gpu_kv_handles(
                    compiled,
                    &self.kv,
                    past,
                    upper_u,
                    self.kv_dim,
                    self.n_layers,
                )?;
                self.gpu_kv_binding.upper = upper_u;
            }
        }
        Ok(())
    }

    /// Dry-run only bucket boundaries in `(from_horizon, new_horizon]` not yet warmed.
    pub fn warmup_bucket_executions_from(
        &mut self,
        from_horizon: usize,
        new_horizon: usize,
    ) -> Result<()> {
        if self.uses_eager_decode() || new_horizon <= from_horizon {
            return Ok(());
        }
        let saved_kv = self.kv.clone();
        let saved_past = self.past_len;
        let saved_hidden = self.last_hidden.clone();
        let emb = vec![0f32; self.hidden];
        let mut hidden_out = vec![0f32; self.hidden];

        let mut sim_pasts = Vec::new();
        for range in self.decode_cache.buckets() {
            let b = range.start as usize;
            if b > from_horizon && b <= new_horizon {
                sim_pasts.push(b);
            }
        }
        sim_pasts.sort_unstable();
        sim_pasts.dedup();

        for sim_past in sim_pasts {
            let upper_u = self
                .decode_upper_for_key(sim_past as u64)
                .unwrap_or(sim_past) as u64;
            self.kv = padded_kv_for_warmup(&saved_kv, sim_past, self.kv_dim, self.n_layers);
            self.past_len = sim_past;
            self.gpu_kv_binding = GpuKvBinding::default();
            if self.use_gpu_kv {
                if let Some(compiled) = self.decode_cache.compiled_for_key_mut(sim_past as u64) {
                    install_gpu_kv_handles(
                        compiled,
                        &self.kv,
                        sim_past,
                        upper_u,
                        self.kv_dim,
                        self.n_layers,
                    )?;
                    self.gpu_kv_binding.upper = upper_u;
                }
            }
            self.decode_hidden_into(ArrayView1::from(&emb), &mut hidden_out)?;
        }

        self.kv = saved_kv;
        self.past_len = saved_past;
        self.last_hidden = saved_hidden;
        self.gpu_kv_binding = GpuKvBinding::default();
        if self.use_gpu_kv {
            let key = self.past_len as u64;
            if let Some(upper) = self.decode_upper_for_key(key) {
                if let Some(compiled) = self.decode_cache.compiled_for_key_mut(key) {
                    install_gpu_kv_handles(
                        compiled,
                        &self.kv,
                        self.past_len,
                        upper as u64,
                        self.kv_dim,
                        self.n_layers,
                    )?;
                    self.gpu_kv_binding.upper = upper as u64;
                }
            }
        }
        Ok(())
    }

    /// Dry decode to warm Metal/CUDA graphs (restores KV after).
    pub fn warmup_bucket_executions(&mut self, horizon: usize) -> Result<()> {
        if self.uses_eager_decode() {
            return Ok(());
        }
        let saved_kv = self.kv.clone();
        let saved_past = self.past_len;
        let saved_hidden = self.last_hidden.clone();
        let emb = vec![0f32; self.hidden];
        let mut hidden_out = vec![0f32; self.hidden];

        let sim_pasts: Vec<usize> = if crate::synth_opts::warmup_all_talk_buckets() {
            let max_key = decode_bucket_max_for_horizon(horizon);
            self.decode_cache
                .buckets()
                .cloned()
                .collect::<Vec<_>>()
                .into_iter()
                .filter_map(|range| {
                    let upper_u = range.end.saturating_sub(1);
                    if upper_u > max_key
                        || upper_u < saved_past as u64
                        || range.start > horizon as u64
                    {
                        return None;
                    }
                    Some(range.start.max(saved_past as u64).max(1) as usize)
                })
                .collect()
        } else {
            let mut sims = vec![saved_past];
            for range in self.decode_cache.buckets() {
                let b = range.start as usize;
                if b > saved_past && b <= horizon {
                    sims.push(b);
                }
            }
            sims.sort_unstable();
            sims.dedup();
            sims
        };

        for sim_past in sim_pasts {
            let upper_u = self
                .decode_upper_for_key(sim_past as u64)
                .unwrap_or(sim_past) as u64;
            self.kv = padded_kv_for_warmup(&saved_kv, sim_past, self.kv_dim, self.n_layers);
            self.past_len = sim_past;
            self.gpu_kv_binding = GpuKvBinding::default();
            if self.use_gpu_kv {
                if let Some(compiled) = self.decode_cache.compiled_for_key_mut(sim_past as u64) {
                    install_gpu_kv_handles(
                        compiled,
                        &self.kv,
                        sim_past,
                        upper_u,
                        self.kv_dim,
                        self.n_layers,
                    )?;
                    self.gpu_kv_binding.upper = upper_u;
                }
            }
            self.decode_hidden_into(ArrayView1::from(&emb), &mut hidden_out)?;
        }

        self.kv = saved_kv;
        self.past_len = saved_past;
        self.last_hidden = saved_hidden;
        self.gpu_kv_binding = GpuKvBinding::default();
        if self.use_gpu_kv {
            let key = self.past_len as u64;
            if let Some(upper) = self.decode_upper_for_key(key) {
                if let Some(compiled) = self.decode_cache.compiled_for_key_mut(key) {
                    install_gpu_kv_handles(
                        compiled,
                        &self.kv,
                        self.past_len,
                        upper as u64,
                        self.kv_dim,
                        self.n_layers,
                    )?;
                    self.gpu_kv_binding.upper = upper as u64;
                }
            }
        }
        Ok(())
    }

    /// Pre-compile decode buckets with `upper <= horizon` (skips unused large past lengths).
    pub fn precompile_decode_buckets_up_to(
        &mut self,
        horizon: usize,
        parent: Option<&Progress>,
    ) -> Result<()> {
        if self.uses_eager_decode() {
            return Ok(());
        }
        if crate::synth_opts::skip_talk_bucket_warmup() {
            return Ok(());
        }
        let max_key = decode_bucket_max_for_horizon(horizon);
        let keys: Vec<u64> = self
            .decode_cache
            .buckets()
            .map(|r| r.end.saturating_sub(1))
            .filter(|&k| k <= max_key)
            .collect();
        let decode_dev = self.decode_compile_device();
        let opts = &self.decode_opts;
        let bucket_prog = Progress::new("talker buckets", keys.len());
        for (i, &key) in keys.iter().enumerate() {
            let detail = format!("decode bucket {}/{} (past≤{key})", i + 1, keys.len());
            if let Some(p) = parent {
                p.set(i, &detail);
            } else {
                bucket_prog.set(i, &detail);
            }
            let weights = Arc::clone(&self.weights);
            let qwen3 = self.qwen3.clone();
            let decode_profile = self.decode_profile.clone();
            let use_hir = crate::compile_opts::talker_decode_use_hir_compile(decode_dev);
            let ensure = || {
                if use_hir {
                    self.decode_cache.ensure_hir_with_params(
                        key,
                        move |upper| {
                            talker_decode_hir_parts(
                                &qwen3,
                                weights.as_ref(),
                                &decode_profile,
                                upper,
                            )
                            .expect("talker decode hir")
                        },
                        opts,
                    )
                } else {
                    self.decode_cache.ensure_graph_with_params(
                        key,
                        move |upper| {
                            talker_decode_graph_parts(
                                &qwen3,
                                weights.as_ref(),
                                &decode_profile,
                                upper,
                            )
                            .expect("talker decode graph")
                        },
                        opts,
                    )
                }
            };
            if decode_dev == Device::Metal {
                metal_mpsgraph_run_guard(self.device, || {
                    let _ = ensure();
                });
            } else {
                let _ = ensure();
            }
        }
        Ok(())
    }

    /// Warm compile caches (prefill + one decode step). Call before timed runs.
    pub fn warmup(&mut self, prefill_seq: usize) -> Result<()> {
        let hidden = self.hidden;
        let mut embeds = Array2::<f32>::zeros((prefill_seq.max(1), hidden));
        for (i, v) in embeds.iter_mut().enumerate() {
            *v = ((i % 17) as f32) * 1e-5;
        }
        let _ = self.warmup_embeds(embeds.view(), prefill_seq)?;
        Ok(())
    }

    /// Warm prefill (and optionally one decode step when buckets are fully lazy).
    pub fn warmup_embeds(
        &mut self,
        embeds: ArrayView2<f32>,
        max_frames: usize,
    ) -> Result<Array2<f32>> {
        self.reset_kv();
        let hidden = self.prefill(embeds)?;
        if self.eager.is_none() && !crate::synth_opts::auto_precompile_horizon(max_frames) {
            let emb = vec![0f32; self.hidden];
            let _ = self.decode_step(ndarray::ArrayView1::from(&emb))?;
            self.reset_kv();
            return self.prefill(embeds);
        }
        Ok(hidden)
    }

    /// Precompute eager decode RoPE bank after prefill (`rope_delta` must be set).
    pub fn warm_eager_decode_rope(&mut self) -> Result<()> {
        if let Some(e) = &mut self.eager {
            e.warm_decode_rope_bank();
        }
        Ok(())
    }

    /// Grow eager attention scratch + RoPE bank for a horizon larger than the
    /// default (256). No-op when no eager talker is loaded or the buffers
    /// already fit.
    pub fn ensure_eager_horizon(&mut self, horizon: usize) {
        if let Some(e) = &mut self.eager {
            e.ensure_attn_horizon(horizon);
        }
    }

    pub fn reset_kv(&mut self) {
        if let Some(e) = &mut self.eager {
            e.reset_kv();
        }
        self.last_hidden.fill(0.0);
        self.past_len = 0;
        self.rope_delta = 0;
        self.kv = KvCacheState {
            past_len: 0,
            layers_k: vec![Vec::new(); self.n_layers],
            layers_v: vec![Vec::new(); self.n_layers],
        };
        self.gpu_kv_binding = GpuKvBinding::default();
    }

    fn prefill_runs_on_cpu(&self) -> bool {
        self.prefill_cache_cpu.is_some()
            || crate::compile_opts::talker_compile_device(self.device) == Device::Cpu
    }

    fn prefill_compile_device(&self) -> Device {
        if self.prefill_cache_cpu.is_some() {
            Device::Cpu
        } else {
            crate::compile_opts::talker_compile_device(self.device)
        }
    }

    fn decode_compile_device(&self) -> Device {
        crate::compile_opts::talker_decode_compile_device(self.device)
    }

    fn prefill_cache_mut(&mut self) -> &mut CompileCache {
        self.prefill_cache_cpu
            .as_mut()
            .unwrap_or(&mut self.prefill_cache)
    }

    /// Compile prefill graph for `seq` (no-op when cached or eager).
    pub fn ensure_prefill_compiled(&mut self, seq: usize) -> Result<()> {
        if self.eager.is_some() || seq == 0 {
            return Ok(());
        }
        let key = ((1u64) << 32) | (seq as u64);
        if self.prefill_cache_mut().contains(key) {
            return Ok(());
        }
        let mask = vec![1u8; seq];
        let (positions, _) = talker_rope_index_prefill(&mask);
        let (rope_cos, rope_sin) = self.prefill_rope_tables(seq, &positions)?;
        let prefill_dev = self.prefill_compile_device();
        let opts = talker_compile_options(&self.prefill_profile, prefill_dev);
        let qwen3 = self.qwen3.clone();
        let weights = Arc::clone(&self.weights);
        let profile = self.prefill_profile.clone();
        let built = {
            let mut wm = weight_map_from_cache(weights.as_ref())?;
            build_qwen3_tts_prefill_built(
                &qwen3,
                &mut wm,
                seq,
                &profile,
                Some(rope_cos),
                Some(rope_sin),
            )?
        };
        if self.prefill_runs_on_cpu() {
            compile_cache_ensure_built_with_options(self.prefill_cache_mut(), key, built, &opts)?;
        } else {
            metal_compile_guard(self.device, || {
                compile_cache_ensure_built_with_options(self.prefill_cache_mut(), key, built, &opts)
            })?;
        }
        Ok(())
    }

    pub fn prefill(&mut self, embeds: ArrayView2<f32>) -> Result<Array2<f32>> {
        let (seq, h) = embeds.dim();
        ensure!(h == self.hidden, "embed hidden mismatch");
        if self.eager.is_some() {
            let out = {
                let e = self.eager.as_mut().expect("eager");
                let out = e.prefill(embeds)?;
                self.past_len = e.past_len;
                self.rope_delta = e.rope_delta();
                if !self.eager_prefill_only {
                    let rows = out.nrows();
                    self.set_last_hidden_from_flat(out.as_slice().unwrap(), rows);
                } else {
                    self.kv = e.kv_cache_state();
                    self.gpu_kv_binding = GpuKvBinding::default();
                    let rows = out.nrows();
                    self.set_last_hidden_from_flat(out.as_slice().unwrap(), rows);
                }
                out
            };
            self.eager.as_mut().expect("eager").warm_decode_rope_bank();
            return Ok(out);
        }

        let mask = vec![1u8; seq];
        let (_, rope_delta) = talker_rope_index_prefill(&mask);
        self.rope_delta = rope_delta;

        self.ensure_prefill_compiled(seq)?;

        let prefill_dev = self.prefill_compile_device();
        let opts = talker_compile_options(&self.prefill_profile, prefill_dev);
        let key = ((1u64) << 32) | (seq as u64);
        let n = seq * h;
        if self.prefill_flat.len() != n {
            self.prefill_flat.resize(n, 0.0);
        }
        for (i, v) in embeds.iter().enumerate() {
            self.prefill_flat[i] = *v;
        }
        let compiled = if let Some(cache) = &mut self.prefill_cache_cpu {
            cache.get_or_compile_with_options(
                key,
                || panic!("talker cpu prefill cache missing key {key}"),
                &opts,
            )
        } else if self.prefill_runs_on_cpu() {
            self.prefill_cache.get_or_compile_with_options(
                key,
                || panic!("talker cpu prefill cache missing key {key}"),
                &opts,
            )
        } else {
            metal_compile_guard(self.device, || {
                self.prefill_cache.get_or_compile_with_options(
                    key,
                    || panic!("talker prefill cache missing key {key}"),
                    &opts,
                )
            })
        };
        let outputs = compiled.run(&[("inputs_embeds", self.prefill_flat.as_slice())]);
        let (hidden_out, kv) =
            kv_from_prefill_outputs(outputs, 1, seq, self.kv_dim, self.n_layers)?;
        self.kv = kv;
        self.past_len = seq;
        self.gpu_kv_binding = GpuKvBinding::default();
        let rows = hidden_out.len() / self.hidden;
        self.set_last_hidden_from_flat(&hidden_out, rows);
        Ok(Array2::from_shape_vec((rows, self.hidden), hidden_out)?)
    }

    fn prefill_rope_tables(&self, seq: usize, positions: &[usize]) -> Result<(Vec<f32>, Vec<f32>)> {
        let rope_table_len = self.cfg.max_position_embeddings;
        let (rope_cos, rope_sin) = if self.uses_mrope() {
            let (seq_cos, seq_sin) = talker_prefill_rope_feeds(&self.cfg, positions);
            let (mut c, mut s) = (
                vec![0f32; rope_table_len * self.head_half],
                vec![0f32; rope_table_len * self.head_half],
            );
            for t in 0..seq {
                let off = t * self.head_half;
                c[off..off + self.head_half].copy_from_slice(&seq_cos[off..off + self.head_half]);
                s[off..off + self.head_half].copy_from_slice(&seq_sin[off..off + self.head_half]);
            }
            (c, s)
        } else {
            let (mut c, mut s) =
                rope_tables_full(&self.inv_freq, rope_table_len, self.cfg.head_dim);
            let (seq_cos, seq_sin) =
                rope_prefill_feeds(&self.inv_freq, positions, self.cfg.head_dim);
            for t in 0..seq {
                let off = t * self.head_half;
                c[off..off + self.head_half].copy_from_slice(&seq_cos[off..off + self.head_half]);
                s[off..off + self.head_half].copy_from_slice(&seq_sin[off..off + self.head_half]);
            }
            (c, s)
        };
        Ok((rope_cos, rope_sin))
    }

    pub fn past_len(&self) -> usize {
        self.past_len
    }

    pub fn rope_delta(&self) -> i64 {
        self.rope_delta
    }

    pub fn last_hidden_view(&self) -> ArrayView1<'_, f32> {
        ArrayView1::from(&self.last_hidden)
    }

    pub fn set_last_hidden(&mut self, row: ArrayView1<f32>) -> Result<()> {
        ensure!(row.len() == self.hidden, "last_hidden len mismatch");
        self.last_hidden.copy_from_slice(row.as_slice().unwrap());
        Ok(())
    }

    fn set_last_hidden_from_flat(&mut self, flat: &[f32], rows: usize) {
        let h = self.hidden;
        let off = (rows.saturating_sub(1)) * h;
        self.last_hidden.copy_from_slice(&flat[off..off + h]);
    }

    /// KV decode step; updates [`Self::last_hidden`] without sampling.
    pub fn decode_hidden_step(&mut self, embed: ArrayView1<f32>) -> Result<()> {
        ensure!(embed.len() == self.hidden, "decode embed len");
        if self.uses_eager_decode() {
            let e = self.eager.as_mut().expect("eager decode");
            e.decode_step_into(embed, &mut self.last_hidden)?;
            self.past_len = e.past_len;
            return Ok(());
        }
        self.decode_embed.copy_from_slice(embed.as_slice().unwrap());
        self.run_compiled_decode_step()?;
        Ok(())
    }

    pub fn decode_step(&mut self, embed: ArrayView1<f32>) -> Result<(Array1<f32>, u32)> {
        self.decode_hidden_step(embed)?;
        let logits = linear_logits(self.last_hidden_view(), self.codec_head.view())?;
        let token = sample_greedy(&logits);
        Ok((Array1::from_vec(self.last_hidden.clone()), token))
    }

    fn run_compiled_decode_step(&mut self) -> Result<()> {
        let past_seq = self.past_len;
        let upper = self
            .decode_upper_for_key(past_seq as u64)
            .unwrap_or(past_seq);
        talker_decode_rope_into(
            &self.cfg,
            &self.inv_freq,
            past_seq,
            self.rope_delta,
            &mut self.decode_rope_cos,
            &mut self.decode_rope_sin,
        );
        bucket_decode_mask_into(past_seq, upper, &mut self.mask_buf);
        let fixed = [
            CacheRunInput {
                name: "inputs_embeds",
                data: self.decode_embed.as_slice(),
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_cos",
                data: self.decode_rope_cos.as_slice(),
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_sin",
                data: self.decode_rope_sin.as_slice(),
                row_inner: None,
            },
            CacheRunInput {
                name: "mask",
                data: self.mask_buf.as_slice(),
                row_inner: None,
            },
        ];
        if self.use_gpu_kv {
            let weights = Arc::clone(&self.weights);
            let qwen3 = self.qwen3.clone();
            let decode_profile = self.decode_profile.clone();
            let key = past_seq as u64;
            let upper_u = upper as u64;
            let prev_upper = self.gpu_kv_binding.upper;
            let bucket_changed = prev_upper != 0 && prev_upper != upper_u;
            let handles_live = self
                .decode_cache
                .compiled_for_key_mut(key)
                .map(|c| c.has_gpu_handle("past_k_0"))
                .unwrap_or(false);
            // wgpu/Metal: re-upload prefix from host each step (GPU handle feeds do not
            // persist updated K/V across runs). CUDA/ROCm keep resident K/V until bucket change.
            let refresh_kv = matches!(self.device, Device::Gpu | Device::Metal)
                || bucket_changed
                || !handles_live;
            let use_hir =
                crate::compile_opts::talker_decode_use_hir_compile(self.decode_compile_device());
            let hidden_vec = metal_mpsgraph_run_guard(self.device, || {
                if use_hir {
                    run_bucketed_kv_decode_gpu_hir(
                        &mut self.decode_cache,
                        key,
                        past_seq,
                        &mut self.kv,
                        &mut self.gpu_kv_binding,
                        self.kv_dim,
                        self.n_layers,
                        &fixed,
                        move |upper| {
                            talker_decode_hir_parts(
                                &qwen3,
                                weights.as_ref(),
                                &decode_profile,
                                upper,
                            )
                            .expect("talker decode hir")
                        },
                        &self.decode_opts,
                        refresh_kv,
                    )
                } else {
                    run_bucketed_kv_decode_gpu(
                        &mut self.decode_cache,
                        key,
                        past_seq,
                        &mut self.kv,
                        &mut self.gpu_kv_binding,
                        self.kv_dim,
                        self.n_layers,
                        &fixed,
                        move |upper| {
                            talker_decode_graph_parts(
                                &qwen3,
                                weights.as_ref(),
                                &decode_profile,
                                upper,
                            )
                            .expect("talker decode graph")
                        },
                        &self.decode_opts,
                        refresh_kv,
                    )
                }
            })?;
            let next_key = (past_seq + 1) as u64;
            let next_upper = self.decode_upper_for_key(next_key).unwrap_or(upper);
            let leaves_bucket = next_upper != upper;
            if leaves_bucket || matches!(self.device, Device::Gpu | Device::Metal) {
                if let Some(compiled) = self.decode_cache.compiled_for_key_mut(key) {
                    sync_gpu_kv_to_host(compiled, &mut self.kv, self.kv_dim, self.n_layers)?;
                }
            }
            if leaves_bucket {
                self.gpu_kv_binding = GpuKvBinding::default();
            }
            self.past_len = past_seq + 1;
            if std::env::var("RLX_QWEN3_TTS_DECODE_DEBUG").ok().as_deref() == Some("1") {
                let rms = hidden_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                let off = upper.saturating_mul(self.hidden);
                eprintln!(
                    "talker decode(gpu_kv) hidden_vec len={} rms={rms:.4} past={past_seq} upper={upper} dev={:?} first8={:?} @upper={:?}",
                    hidden_vec.len(),
                    self.device,
                    &hidden_vec[..8.min(hidden_vec.len())],
                    if hidden_vec.len() >= off + 8 {
                        &hidden_vec[off..off + 8]
                    } else {
                        &[] as &[f32]
                    },
                );
            }
            bucket_decode_hidden_into(&hidden_vec, self.hidden, &mut self.last_hidden)?;
            return Ok(());
        }

        let weights = Arc::clone(&self.weights);
        let qwen3 = self.qwen3.clone();
        let decode_profile = self.decode_profile.clone();
        let decode_dev = self.decode_compile_device();
        let use_hir = crate::compile_opts::talker_decode_use_hir_compile(decode_dev);
        let run_decode = || {
            if use_hir {
                run_bucketed_kv_decode_hir_uniform(
                    &mut self.decode_cache,
                    past_seq,
                    &self.kv,
                    self.kv_dim,
                    self.n_layers,
                    &fixed,
                    move |upper| {
                        talker_decode_hir_parts(&qwen3, weights.as_ref(), &decode_profile, upper)
                            .expect("talker decode hir")
                    },
                    &self.decode_opts,
                )
            } else {
                run_bucketed_kv_decode(
                    &mut self.decode_cache,
                    past_seq,
                    &self.kv,
                    self.kv_dim,
                    self.n_layers,
                    &fixed,
                    move |upper| {
                        talker_decode_graph_parts(&qwen3, weights.as_ref(), &decode_profile, upper)
                            .expect("talker decode graph")
                    },
                    &self.decode_opts,
                )
            }
        };
        let (hidden_vec, new_k, new_v) = if decode_dev == Device::Metal {
            metal_mpsgraph_run_guard(self.device, run_decode)?
        } else {
            run_decode()?
        };
        commit_kv_layers(&mut self.kv.layers_k, &mut self.kv.layers_v, &new_k, &new_v);
        self.kv.past_len = past_seq + 1;
        self.past_len += 1;
        if std::env::var("RLX_QWEN3_TTS_DECODE_DEBUG").ok().as_deref() == Some("1") {
            let rms = hidden_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            let off = upper.saturating_mul(self.hidden);
            eprintln!(
                "talker decode hidden_vec len={} rms={rms:.4} past={past_seq} upper={upper} dev={:?} first8={:?} @upper={:?}",
                hidden_vec.len(),
                self.device,
                &hidden_vec[..8.min(hidden_vec.len())],
                if hidden_vec.len() >= off + 8 {
                    &hidden_vec[off..off + 8]
                } else {
                    &[] as &[f32]
                },
            );
        }
        bucket_decode_hidden_into(&hidden_vec, self.hidden, &mut self.last_hidden)?;
        Ok(())
    }

    /// Import bucketed decode outputs from a fused codec-frame graph run.
    pub fn import_fused_decode_outputs(
        &mut self,
        hidden_vec: &[f32],
        layers_k: &[Vec<f32>],
        layers_v: &[Vec<f32>],
        past_seq: usize,
    ) -> Result<()> {
        bucket_decode_hidden_into(hidden_vec, self.hidden, &mut self.last_hidden)?;
        commit_kv_layers(
            &mut self.kv.layers_k,
            &mut self.kv.layers_v,
            layers_k,
            layers_v,
        );
        self.kv.past_len = past_seq + 1;
        self.past_len = past_seq + 1;
        Ok(())
    }

    /// KV decode; writes the last hidden row into `hidden_out`.
    pub fn decode_hidden_into(
        &mut self,
        embed: ArrayView1<f32>,
        hidden_out: &mut [f32],
    ) -> Result<()> {
        ensure!(embed.len() == self.hidden, "decode embed len");
        ensure!(hidden_out.len() == self.hidden, "hidden_out len mismatch");
        if self.uses_eager_decode() {
            let e = self.eager.as_mut().expect("eager decode");
            e.decode_step_into(embed, hidden_out)?;
            self.past_len = e.past_len;
            self.last_hidden.copy_from_slice(hidden_out);
            return Ok(());
        }
        self.decode_embed.copy_from_slice(embed.as_slice().unwrap());
        self.run_compiled_decode_step()?;
        hidden_out.copy_from_slice(self.last_hidden.as_slice());
        Ok(())
    }

    fn uses_eager_decode(&self) -> bool {
        self.eager.is_some() && !self.eager_prefill_only
    }

    fn uses_mrope(&self) -> bool {
        self.cfg.rope_scaling.is_some()
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    pub fn is_eager(&self) -> bool {
        self.uses_eager_decode()
    }

    pub fn uses_gpu_kv(&self) -> bool {
        self.use_gpu_kv
    }

    pub fn codec_eos(&self) -> u32 {
        self.codec_eos
    }

    pub fn codec_head(&self) -> ArrayView2<'_, f32> {
        self.codec_head.view()
    }

    pub fn codec_head_flat(&self) -> (&[f32], usize, usize) {
        (
            self.codec_head_flat.as_slice(),
            self.codec_vocab,
            self.hidden,
        )
    }

    /// Batched decode of `M` consecutive talker tokens. Returns the final
    /// hidden rows `[M, hidden]` after `model.norm`. Appends `M` rows to the
    /// KV cache; `past_len()` advances by `M`. Requires the eager backend —
    /// for the GPU backend the caller must fall back to one-at-a-time.
    #[cfg(feature = "speculative-decode")]
    pub fn decode_batched(&mut self, embeds: ArrayView2<f32>) -> Result<ndarray::Array2<f32>> {
        let eager = self
            .eager
            .as_mut()
            .context("decode_batched: requires eager backend")?;
        eager.decode_batched(embeds)
    }

    /// Undo the last `n` rows of the KV cache (and decrement `past_len` by
    /// `n`). Used by the speculative loop to retract drafted tokens the
    /// verifier rejected. Eager backend only; no-op-and-error on GPU.
    #[cfg(feature = "speculative-decode")]
    pub fn rollback_kv(&mut self, n: usize) -> Result<()> {
        let eager = self
            .eager
            .as_mut()
            .context("rollback_kv: requires eager backend")?;
        eager.rollback_kv(n);
        Ok(())
    }

    /// Per-layer KV dimension (n_kv_heads × head_dim). Eager backend only.
    #[cfg(feature = "speculative-decode")]
    pub fn kv_dim_eager(&self) -> Result<usize> {
        let eager = self
            .eager
            .as_ref()
            .context("kv_dim_eager: requires eager backend")?;
        Ok(eager.kv_dim_for_draft())
    }

    /// Number of transformer layers. Eager backend only.
    #[cfg(feature = "speculative-decode")]
    pub fn num_layers_eager(&self) -> Result<usize> {
        let eager = self
            .eager
            .as_ref()
            .context("num_layers_eager: requires eager backend")?;
        Ok(eager.num_layers())
    }

    /// One-step early-exit decode against an external draft KV cache.
    /// Forwards to [`crate::talker::eager::TalkerEagerModel::early_exit_decode_step`].
    #[cfg(feature = "speculative-decode")]
    pub fn early_exit_decode_step(
        &mut self,
        embed: &[f32],
        kv: &mut crate::talker::eager::DraftKvCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let eager = self
            .eager
            .as_mut()
            .context("early_exit_decode_step: requires eager backend")?;
        eager.early_exit_decode_step(embed, kv, position)
    }

    /// Snapshot host K/V after prefill (parity / isolation tests).
    pub fn kv_state(&self) -> KvCacheState {
        if let Some(e) = &self.eager {
            return e.kv_cache_state();
        }
        self.kv.clone()
    }

    /// Restore host K/V, `past_len`, and MRoPE delta (parity / isolation tests).
    pub fn restore_kv_state(&mut self, kv: KvCacheState, rope_delta: i64) {
        self.past_len = kv.past_len;
        self.rope_delta = rope_delta;
        self.kv = kv;
        self.gpu_kv_binding = GpuKvBinding::default();
        if let Some(e) = self.eager.as_mut() {
            e.past_len = self.past_len;
            e.rope_delta = rope_delta;
            e.warm_decode_rope_bank();
        }
    }
}
