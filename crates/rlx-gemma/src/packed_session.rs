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

//! Packed GGUF inference with bucketed prefill + bucketed decode KV cache.

use crate::builder::{
    build_gemma_decode_graph_sized_packed_ext, build_gemma_graph_sized_packed_ext,
    drain_gemma_packed_weights, precompute_packed_decode_tied_lm_head,
};
use crate::generator::{decode_profile_for_device, metal_decode_compile_guard};
use crate::rope::{resolve_global_inv_freq, resolve_inv_freq};
use anyhow::{Context, Result, anyhow, bail};
use rlx_core::flow_bridge::{
    compile_options_for_packed_gguf_prefill_with_profile, packed_gguf_compile_guard,
    packed_gguf_execution_device,
};
use rlx_core::weight_loader::{GgufLoader, WeightLoader};
use rlx_core::{
    infer_prefill_kv_seq, kv_from_prefill_outputs_per_layer, packed_prefill_active_extent_enabled,
    run_bucketed_kv_decode_graph_layers_scratch, run_packed_prefill,
};
use rlx_flow::CompileProfile;
use rlx_ir::Graph;
use rlx_ir::quant::QuantScheme;
use rlx_qwen3::{SampleOpts, sample_token};
use rlx_runtime::compile_cache::{BucketedCompileCache, CacheRunInput, CompileCache};
use rlx_runtime::kv_cache::LayerKvCache;
use rlx_runtime::{CompileOptions, Device};
use std::collections::{HashMap, HashSet};
use std::path::Path;

type PackedWeightMap = HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>;
use std::sync::Arc;
use std::time::Instant;

use crate::config::GemmaConfig;

const TIED_LM_HEAD: &str = "gemma.packed.decode.lm_head.tied_t";

/// Stub loader for cached graph builds (weights come from session caches).
struct EmptyWeightLoader;

impl WeightLoader for EmptyWeightLoader {
    fn len(&self) -> usize {
        0
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        Err(anyhow!("packed cache miss for F32 weight {key}"))
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        Err(anyhow!("packed cache miss for F32 weight {key}"))
    }
    fn take_packed(
        &mut self,
        key: &str,
    ) -> Result<Option<rlx_core::weight_map::PackedWeightTensor>> {
        let _ = key;
        Ok(None)
    }
    fn remaining_keys(&self) -> Vec<String> {
        vec![]
    }
}

#[derive(Default)]
struct DecodeInputScratch {
    mask: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    global_cos: Vec<f32>,
    global_sin: Vec<f32>,
}

impl DecodeInputScratch {
    fn fill_mask(&mut self, past_seq: usize, upper: usize) {
        if self.mask.len() != upper + 1 {
            self.mask.resize(upper + 1, 0.0);
        }
        for (i, m) in self.mask.iter_mut().enumerate().take(upper + 1) {
            *m = if i < past_seq || i == upper { 1.0 } else { 0.0 };
        }
    }

    fn fill_rope(&mut self, inv_freq: &[f64], pos: usize) {
        let half = inv_freq.len();
        self.cos.resize(half, 0.0);
        self.sin.resize(half, 0.0);
        for (i, &freq) in inv_freq.iter().enumerate() {
            let angle = pos as f64 * freq;
            let (s, c) = angle.sin_cos();
            self.cos[i] = c as f32;
            self.sin[i] = s as f32;
        }
    }

    fn fill_global_rope(&mut self, inv_freq: &[f64], pos: usize) {
        let half = inv_freq.len();
        self.global_cos.resize(half, 0.0);
        self.global_sin.resize(half, 0.0);
        for (i, &freq) in inv_freq.iter().enumerate() {
            let angle = pos as f64 * freq;
            let (s, c) = angle.sin_cos();
            self.global_cos[i] = c as f32;
            self.global_sin[i] = s as f32;
        }
    }
}

#[derive(Default)]
struct DecodeKvScratch {
    padded_k: Vec<Vec<f32>>,
    padded_v: Vec<Vec<f32>>,
}

impl DecodeKvScratch {
    fn ensure_bucket(&mut self, upper: usize, kv_dims: &[usize]) {
        if self.padded_k.len() != kv_dims.len() {
            self.padded_k = kv_dims.iter().map(|&d| vec![0.0; upper * d]).collect();
            self.padded_v = kv_dims.iter().map(|&d| vec![0.0; upper * d]).collect();
            return;
        }
        for (i, &d) in kv_dims.iter().enumerate() {
            let need = upper * d;
            if self.padded_k[i].len() != need {
                self.padded_k[i].resize(need, 0.0);
                self.padded_v[i].resize(need, 0.0);
            }
        }
    }
}

fn packed_decode_compile_guard<R, F>(device: Device, exec_device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    metal_decode_compile_guard(device, true, || packed_gguf_compile_guard(exec_device, f))
}

fn packed_timing_enabled() -> bool {
    std::env::var("RLX_GEMMA_PACKED_TIMING")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
}

fn decode_prewarm_enabled() -> bool {
    std::env::var("RLX_GEMMA_PACKED_WARM_DECODE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
        .unwrap_or(false)
}

fn prefill_prewarm_enabled() -> bool {
    std::env::var("RLX_GEMMA_PACKED_WARM_PREFILL")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
        .unwrap_or(false)
}

fn warm_past_seqs(max_seq: usize) -> Vec<usize> {
    if let Ok(raw) = std::env::var("RLX_GEMMA_PACKED_WARM_PAST") {
        if let Ok(one) = raw.parse::<usize>() {
            return vec![one];
        }
        let parsed: Vec<usize> = raw
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        if !parsed.is_empty() {
            return parsed;
        }
    }
    let mut seqs = vec![15usize];
    seqs.retain(|&p| p <= max_seq);
    if seqs.is_empty() {
        seqs.push(max_seq.max(1));
    }
    seqs
}

/// Prefill compile bucket for prompt length `n`.
///
/// Uses an exact bucket when power-of-two rounding would waste more than
/// ~12.5% of compute; otherwise rounds up to the next power of two (fewer
/// cached graphs). Capped at `max_seq`.
pub fn prefill_bucket_len(n: usize, max_seq: usize) -> usize {
    let n = n.max(1);
    let cap = max_seq.max(1);
    let pow2 = n.next_power_of_two().min(cap);
    if pow2 > n && pow2 - n > n / 8 {
        n.min(cap)
    } else {
        pow2
    }
}

pub(crate) struct GemmaPackedSession {
    cfg: GemmaConfig,
    device: Device,
    exec_device: Device,
    max_seq: usize,
    prefill_cache: CompileCache,
    prefill_opts: CompileOptions,
    prefill_packed_loaded: HashSet<usize>,
    decode_cache: BucketedCompileCache,
    decode_opts: CompileOptions,
    f32_params: Arc<HashMap<String, Vec<f32>>>,
    packed_tensors: Arc<PackedWeightMap>,
    packed_buckets_loaded: HashSet<u64>,
    inv_freq: Vec<f64>,
    global_inv_freq: Option<Vec<f64>>,
    cache: Option<LayerKvCache>,
    tokens: Vec<u32>,
    padded_ids: Vec<u32>,
    ids_f32: Vec<f32>,
    last_idx: [f32; 1],
    decode_inputs: DecodeInputScratch,
    decode_scratch: DecodeKvScratch,
    prefill_logits: Option<Vec<f32>>,
}

impl GemmaPackedSession {
    pub fn build(
        cfg: GemmaConfig,
        weights_path: &Path,
        max_seq: usize,
        device: Device,
    ) -> Result<Self> {
        let exec_device = packed_gguf_execution_device(device);
        if exec_device != device {
            eprintln!(
                "[gemma-runner] packed GGUF on {device:?}: executes on {exec_device:?} \
                 until {device:?} packed parity is fixed upstream"
            );
        }
        let path_str = weights_path
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 weights path"))?
            .to_string();

        let mut loader = GgufLoader::from_file(&path_str)?;
        let (mut f32_params, packed) = drain_gemma_packed_weights(&cfg, &mut loader)?;
        if cfg.tie_word_embeddings {
            if let Some(embed) = f32_params.get("model.embed_tokens.weight") {
                f32_params.insert(
                    TIED_LM_HEAD.into(),
                    precompute_packed_decode_tied_lm_head(&cfg, embed)?,
                );
            }
        }

        let t_build = Instant::now();

        let inv_freq = resolve_inv_freq(&cfg, None);
        let global_inv_freq = resolve_global_inv_freq(&cfg, None).map(|v| v.to_vec());

        let prefill_opts = compile_options_for_packed_gguf_prefill_with_profile(
            &CompileProfile::gemma_prefill(),
            exec_device,
        );
        let decode_horizon = max_seq.saturating_add(16).max(32);
        let decode_cache =
            BucketedCompileCache::power_of_two_ladder(exec_device, 1, decode_horizon as u64);
        let decode_profile = decode_profile_for_device(device);
        let decode_opts =
            compile_options_for_packed_gguf_prefill_with_profile(&decode_profile, exec_device);

        let f32_arc = Arc::new(f32_params);
        let packed_arc = Arc::new(packed);

        let mut session = Self {
            cfg,
            device,
            exec_device,
            max_seq,
            prefill_cache: CompileCache::new(exec_device, 16),
            prefill_opts,
            prefill_packed_loaded: HashSet::new(),
            decode_cache,
            decode_opts,
            f32_params: f32_arc,
            packed_tensors: packed_arc,
            packed_buckets_loaded: HashSet::new(),
            inv_freq,
            global_inv_freq,
            cache: None,
            tokens: Vec::new(),
            padded_ids: Vec::new(),
            ids_f32: Vec::new(),
            last_idx: [0f32; 1],
            decode_inputs: DecodeInputScratch::default(),
            decode_scratch: DecodeKvScratch::default(),
            prefill_logits: None,
        };

        // Compile smallest prefill bucket; optional execute-warm (RLX_GEMMA_PACKED_WARM_PREFILL=1).
        let warm_seq = prefill_bucket_len(16, max_seq);
        session.ensure_prefill_bucket(warm_seq)?;
        if prefill_prewarm_enabled() {
            session.prefill_execute_warm(warm_seq)?;
        }

        if decode_prewarm_enabled() {
            session.prewarm_decode_buckets()?;
        }

        eprintln!(
            "[gemma-runner] packed session: max_seq={max_seq} init={:.0} ms prefill_bucket={warm_seq} decode_horizon={decode_horizon}",
            t_build.elapsed().as_secs_f64() * 1000.0
        );
        Ok(session)
    }

    fn build_prefill_graph(
        cfg: &GemmaConfig,
        f32_params: &HashMap<String, Vec<f32>>,
        packed_tensors: &PackedWeightMap,
        seq: usize,
    ) -> (Graph, HashMap<String, Vec<f32>>) {
        let mut loader = EmptyWeightLoader;
        let mut local_packed = HashMap::new();
        build_gemma_graph_sized_packed_ext(
            cfg,
            &mut loader,
            1,
            seq,
            true,
            true,
            true,
            &mut local_packed,
            Some(packed_tensors),
            Some(f32_params),
        )
        .expect("packed prefill graph from cache")
    }

    fn build_decode_graph(
        cfg: &GemmaConfig,
        f32_params: &HashMap<String, Vec<f32>>,
        packed_tensors: &PackedWeightMap,
        past_upper: usize,
    ) -> (Graph, HashMap<String, Vec<f32>>) {
        let mut loader = EmptyWeightLoader;
        let mut local_packed = HashMap::new();
        build_gemma_decode_graph_sized_packed_ext(
            cfg,
            &mut loader,
            1,
            past_upper,
            true,
            &mut local_packed,
            Some(packed_tensors),
            Some(f32_params),
        )
        .expect("packed decode graph from cache")
    }

    fn ensure_prefill_bucket(&mut self, seq: usize) -> Result<()> {
        let key = seq as u64;
        if self.prefill_cache.contains(key) {
            return Ok(());
        }
        let cfg = self.cfg.clone();
        let f32_params = Arc::clone(&self.f32_params);
        let packed_tensors = Arc::clone(&self.packed_tensors);
        let opts = self.prefill_opts.clone();
        let packed_loaded = &mut self.prefill_packed_loaded;
        let packed_for_upload = Arc::clone(&self.packed_tensors);
        packed_gguf_compile_guard(self.exec_device, || {
            let (graph, params) =
                Self::build_prefill_graph(&cfg, &f32_params, &packed_tensors, seq);
            let compiled = self
                .prefill_cache
                .get_or_compile_with_options(key, || graph, &opts);
            for (name, data) in &params {
                compiled.set_param(name, data);
            }
            if packed_loaded.insert(seq) {
                for (name, (bytes, _scheme, _shape)) in packed_for_upload.iter() {
                    compiled.set_param_typed(name, bytes, rlx_ir::DType::U8);
                }
            }
        });
        Ok(())
    }

    fn prefill_execute_warm(&mut self, seq: usize) -> Result<()> {
        self.padded_ids.resize(seq, 0);
        self.ids_f32.resize(seq, 1.0);
        self.last_idx[0] = 0.0;
        let key = seq as u64;
        let compiled = self.prefill_cache.get_or_compile_with_options(
            key,
            || unreachable!("warm bucket"),
            &self.prefill_opts,
        );
        let _ = compiled.run(&[
            ("input_ids", self.ids_f32.as_slice()),
            ("last_token_idx", self.last_idx.as_slice()),
        ]);
        Ok(())
    }

    fn prewarm_decode_buckets(&mut self) -> Result<()> {
        for past in warm_past_seqs(self.max_seq) {
            self.prewarm_decode_bucket(past)?;
        }
        Ok(())
    }

    fn prewarm_decode_bucket(&mut self, past_seq: usize) -> Result<()> {
        let key = past_seq as u64;
        if self.decode_cache.compiled_for_key_mut(key).is_some() {
            return Ok(());
        }
        if self.decode_cache.bucket_for(key).is_none() {
            return Ok(());
        }
        let t0 = Instant::now();
        let cfg = self.cfg.clone();
        let f32_params = Arc::clone(&self.f32_params);
        let packed_tensors = Arc::clone(&self.packed_tensors);
        let packed_for_upload = Arc::clone(&self.packed_tensors);
        let decode_opts = self.decode_opts.clone();
        let packed_buckets = &mut self.packed_buckets_loaded;
        packed_decode_compile_guard(self.device, self.exec_device, || {
            let (upper_u64, compiled) = self
                .decode_cache
                .ensure_graph_with_params(
                    key,
                    move |upper| {
                        Self::build_decode_graph(&cfg, &f32_params, &packed_tensors, upper as usize)
                    },
                    &decode_opts,
                )
                .expect("decode bucket prewarm");
            if packed_buckets.insert(upper_u64) {
                for (name, (bytes, _scheme, _shape)) in packed_for_upload.iter() {
                    compiled.set_param_typed(name, bytes, rlx_ir::DType::U8);
                }
            }
        });
        eprintln!(
            "[gemma-runner] prewarmed decode bucket past_seq={past_seq} in {:.1} s",
            t0.elapsed().as_secs_f64()
        );
        Ok(())
    }

    fn per_layer_kv_dims(&self) -> Vec<usize> {
        (0..self.cfg.num_hidden_layers)
            .map(|i| self.cfg.layer_num_kv_heads(i) * self.cfg.layer_head_dim(i))
            .collect()
    }

    fn run_prefill_with_cache(&mut self, prompt_ids: &[u32]) -> Result<(Vec<f32>, LayerKvCache)> {
        let n = prompt_ids.len().min(self.max_seq);
        let seq_bucket = prefill_bucket_len(n, self.max_seq);
        self.ensure_prefill_bucket(seq_bucket)?;

        self.padded_ids.resize(seq_bucket, 0);
        self.ids_f32.resize(seq_bucket, 0.0);
        self.padded_ids.fill(0);
        for (i, &t) in prompt_ids.iter().take(n).enumerate() {
            self.padded_ids[i] = t;
        }
        for (dst, &id) in self.ids_f32.iter_mut().zip(self.padded_ids.iter()) {
            *dst = id as f32;
        }
        self.last_idx[0] = n.saturating_sub(1) as f32;

        let t0 = Instant::now();
        let key = seq_bucket as u64;
        let compiled = self.prefill_cache.get_or_compile_with_options(
            key,
            || unreachable!("prefill bucket"),
            &self.prefill_opts,
        );
        let outputs = run_packed_prefill(
            compiled,
            self.exec_device,
            n,
            seq_bucket,
            &[
                ("input_ids", self.ids_f32.as_slice()),
                ("last_token_idx", self.last_idx.as_slice()),
            ],
        );
        if packed_timing_enabled() {
            let active = packed_prefill_active_extent_enabled(self.exec_device) && n < seq_bucket;
            eprintln!(
                "[gemma-packed] prefill n={n} bucket={seq_bucket} active={active} {:.1} ms",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }

        let kv_dims = self.per_layer_kv_dims();
        let kv_seq = infer_prefill_kv_seq(&outputs, 1, &kv_dims, n, seq_bucket);
        let (logits, mut kv) = kv_from_prefill_outputs_per_layer(
            outputs,
            1,
            kv_seq,
            &kv_dims,
            self.cfg.num_hidden_layers,
        )?;
        if kv_seq > n {
            for (i, &kd) in kv_dims.iter().enumerate() {
                let keep = n * kd;
                kv.layers_k[i].truncate(keep);
                kv.layers_v[i].truncate(keep);
            }
        }
        kv.past_len = n;
        Ok((logits, kv))
    }

    fn decode_step_bucketed(&mut self, past_seq: usize, input_tok: u32) -> Result<Vec<f32>> {
        let kv_dims = self.per_layer_kv_dims();
        let n_layers = self.cfg.num_hidden_layers;
        let upper = self
            .decode_cache
            .bucket_for(past_seq as u64)
            .and_then(|idx| {
                self.decode_cache
                    .buckets()
                    .nth(idx)
                    .map(|r| (r.end - 1) as usize)
            })
            .unwrap_or(past_seq);

        self.decode_scratch.ensure_bucket(upper, &kv_dims);
        self.decode_inputs.fill_mask(past_seq, upper);
        self.decode_inputs.fill_rope(&self.inv_freq, past_seq);
        if let Some(global) = &self.global_inv_freq {
            self.decode_inputs.fill_global_rope(global, past_seq);
        }

        let input_ids_f32 = [input_tok as f32];
        let mut fixed = vec![
            CacheRunInput {
                name: "input_ids",
                data: &input_ids_f32,
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_cos",
                data: &self.decode_inputs.cos,
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_sin",
                data: &self.decode_inputs.sin,
                row_inner: None,
            },
            CacheRunInput {
                name: "mask",
                data: &self.decode_inputs.mask,
                row_inner: None,
            },
        ];
        if self.global_inv_freq.is_some() {
            fixed.push(CacheRunInput {
                name: "rope_cos_global",
                data: &self.decode_inputs.global_cos,
                row_inner: None,
            });
            fixed.push(CacheRunInput {
                name: "rope_sin_global",
                data: &self.decode_inputs.global_sin,
                row_inner: None,
            });
        }

        let cfg = self.cfg.clone();
        let f32_params = Arc::clone(&self.f32_params);
        let packed_tensors = Arc::clone(&self.packed_tensors);
        let decode_opts = self.decode_opts.clone();
        let packed_upload = Arc::clone(&self.packed_tensors);
        let kv_cache = self.cache.as_ref().context("decode without cache")?;
        let t0 = Instant::now();
        let needs_build = self
            .decode_cache
            .compiled_for_key_mut(past_seq as u64)
            .is_none();

        let (logits, new_k, new_v) =
            packed_decode_compile_guard(self.device, self.exec_device, || {
                run_bucketed_kv_decode_graph_layers_scratch(
                    &mut self.decode_cache,
                    past_seq,
                    kv_cache,
                    &kv_dims,
                    n_layers,
                    &mut self.decode_scratch.padded_k,
                    &mut self.decode_scratch.padded_v,
                    &fixed,
                    move |upper_u64| {
                        Self::build_decode_graph(
                            &cfg,
                            &f32_params,
                            &packed_tensors,
                            upper_u64 as usize,
                        )
                    },
                    Some(packed_upload.as_ref()),
                    &mut self.packed_buckets_loaded,
                    &decode_opts,
                )
            })?;

        if packed_timing_enabled() {
            eprintln!(
                "[gemma-packed] decode past={past_seq} upper={upper} compile={needs_build} {:.1} ms",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }

        let cache_mut = self.cache.as_mut().unwrap();
        cache_mut.past_len = past_seq + 1;
        cache_mut.layers_k = new_k;
        cache_mut.layers_v = new_v;

        let vocab = self.cfg.vocab_size;
        if logits.len() < vocab {
            bail!("decode logits short: {} < {vocab}", logits.len());
        }
        Ok(logits[..vocab].to_vec())
    }

    pub fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        let (logits, kv) = self.run_prefill_with_cache(prompt_ids)?;
        self.tokens.clear();
        self.tokens.extend_from_slice(prompt_ids);
        self.cache = Some(kv);
        let vocab = self.cfg.vocab_size;
        if logits.len() < vocab {
            bail!("logits short: {} < {vocab}", logits.len());
        }
        let logits = logits[..vocab].to_vec();
        self.prefill_logits = Some(logits.clone());
        Ok(logits)
    }

    fn prompt_prefill_ready(&self, prompt_ids: &[u32]) -> bool {
        self.cache.is_some()
            && self.prefill_logits.is_some()
            && self.tokens.as_slice() == prompt_ids
    }

    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        sample: SampleOpts,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let vocab = self.cfg.vocab_size;
        let first_logits = if self.prompt_prefill_ready(prompt_ids) {
            self.prefill_logits.take().unwrap()
        } else {
            self.tokens.clear();
            self.tokens.extend_from_slice(prompt_ids);
            self.cache = None;
            self.prefill_logits = None;
            let (logits, kv) = self.run_prefill_with_cache(prompt_ids)?;
            self.cache = Some(kv);
            if logits.len() < vocab {
                bail!("logits short: {} < {vocab}", logits.len());
            }
            logits[..vocab].to_vec()
        };

        let first = sample_token(&first_logits, sample) as u32;
        on_token(first);
        self.tokens.push(first);
        let mut out = vec![first];

        for _ in 1..n_new {
            let past_seq = self.cache.as_ref().unwrap().past_len;
            let input_tok = self.tokens[past_seq];
            let logits = self.decode_step_bucketed(past_seq, input_tok)?;
            let next = sample_token(&logits, sample) as u32;
            on_token(next);
            self.tokens.push(next);
            out.push(next);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::prefill_bucket_len;

    #[test]
    fn prefill_bucket_pow2_when_waste_small() {
        assert_eq!(prefill_bucket_len(15, 128), 16);
        assert_eq!(prefill_bucket_len(8, 128), 8);
    }

    #[test]
    fn prefill_bucket_exact_when_waste_large() {
        assert_eq!(prefill_bucket_len(100, 128), 100);
        assert_eq!(prefill_bucket_len(65, 128), 65);
    }

    #[test]
    fn prefill_bucket_capped_at_max_seq() {
        assert_eq!(prefill_bucket_len(200, 128), 128);
        assert_eq!(prefill_bucket_len(0, 64), 1);
    }
}
