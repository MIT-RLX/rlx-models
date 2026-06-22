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
    precompute_packed_decode_tied_lm_head,
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
/// High bit set on prefill compile-cache keys for post-`model.norm` hidden graphs.
const PREFILL_HIDDEN_TAG: u64 = 1u64 << 62;

fn prefill_cache_key(seq: usize, hidden_only: bool) -> u64 {
    seq as u64 | if hidden_only { PREFILL_HIDDEN_TAG } else { 0 }
}

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
    prefill_packed_loaded: HashSet<u64>,
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
    /// Task #37: precomputed Q4K row size in bytes for the embed table when
    /// the builder takes the lazy-embed path. `None` ⇒ legacy in-graph gather.
    embed_row_bytes: Option<usize>,
    /// Reusable buffer for prefill embedding rows (≤ bucket_size × hidden f32).
    embed_scratch: Vec<f32>,
}

/// Dequant a single embed row from Q4K packed bytes — used by the lazy-embed
/// path (task #37). The row layout is `[blocks_per_row × block_bytes]`, where
/// each block decodes to `block_elems` f32 values. Q4K-only for now; extending
/// to Q6K is a one-line `match` on `scheme`.
fn gather_embed_row(
    packed_bytes: &[u8],
    scheme: rlx_ir::quant::QuantScheme,
    hidden: usize,
    token_id: usize,
    out: &mut [f32],
) -> Result<()> {
    use rlx_ir::quant::QuantScheme;
    debug_assert_eq!(out.len(), hidden);
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    if block_elems == 0 || !hidden.is_multiple_of(block_elems) {
        bail!(
            "gather_embed_row: scheme {scheme:?} block_elems={block_elems} doesn't divide hidden={hidden}"
        );
    }
    let blocks_per_row = hidden / block_elems;
    let row_bytes = blocks_per_row * block_bytes;
    let off = token_id * row_bytes;
    if off + row_bytes > packed_bytes.len() {
        bail!(
            "gather_embed_row: row offset {off}+{row_bytes} past packed bytes len {}",
            packed_bytes.len()
        );
    }
    let row = &packed_bytes[off..off + row_bytes];
    let dequant = match scheme {
        QuantScheme::GgufQ4K => rlx_gguf::dequant_q4_k(row, hidden)?,
        QuantScheme::GgufQ6K => rlx_gguf::dequant_q6_k(row, hidden)?,
        _ => bail!("gather_embed_row: unsupported scheme {scheme:?}"),
    };
    out.copy_from_slice(&dequant);
    Ok(())
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

        let trace_init = std::env::var("RLX_GEMMA_TRACE_INIT").is_ok();
        macro_rules! step {
            ($t:expr, $msg:expr) => {
                if trace_init {
                    eprintln!(
                        "[gemma-runner trace] {} {:.1}s",
                        $msg,
                        $t.elapsed().as_secs_f64()
                    );
                }
            };
        }
        let t_load = Instant::now();
        let mut loader = GgufLoader::from_file(&path_str)?;
        step!(t_load, "GgufLoader::from_file done at");
        let t_drain = Instant::now();
        // RoPE tables only need `max_seq` rows for prefill (decode uses a single
        // `rope_slice` row computed on the fly). For Gemma 4 12B with default
        // `max_seq=128` this caps the table at 128×256 elements instead of
        // 262144×256 — frees ~1 GB of f32 cache at LOAD with no functional
        // change. The `+16` buffer absorbs `prefill_bucket_len`'s next-pow2
        // rounding for prompts near the bucket edge.
        let rope_cap = max_seq.saturating_add(16);
        let (mut f32_params, packed) =
            crate::builder::drain_gemma_packed_weights_ext(&cfg, &mut loader, Some(rope_cap))?;
        step!(t_drain, "drain_gemma_packed_weights done at");
        if trace_init {
            let f32_bytes: usize = f32_params.values().map(|v| v.len() * 4).sum();
            let packed_bytes: usize = packed.values().map(|(b, _, _)| b.len()).sum();
            eprintln!(
                "[gemma-runner trace]   f32 params: {} entries, {:.2} GB; packed: {} entries, {:.2} GB",
                f32_params.len(),
                f32_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                packed.len(),
                packed_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        if cfg.tie_word_embeddings {
            let t_tied = Instant::now();
            if let Some(embed) = f32_params.get("model.embed_tokens.weight") {
                f32_params.insert(
                    TIED_LM_HEAD.into(),
                    precompute_packed_decode_tied_lm_head(&cfg, embed)?,
                );
            }
            step!(t_tied, "precompute_packed_decode_tied_lm_head done at");
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

        // Task #37: when the packed embed bytes are present the graph builder
        // takes the lazy path; remember the per-row byte count once so the
        // hot `gather_embed_rows` call doesn't recompute it per token.
        let embed_row_bytes = packed_arc
            .get("model.embed_tokens.weight")
            .map(|(_, scheme, _)| {
                let block_elems = scheme.gguf_block_size() as usize;
                let block_bytes = scheme.gguf_block_bytes() as usize;
                let h = cfg.hidden_size;
                (h / block_elems.max(1)) * block_bytes
            });

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
            embed_row_bytes,
            embed_scratch: Vec::new(),
        };

        // Compile smallest prefill bucket; optional execute-warm (RLX_GEMMA_PACKED_WARM_PREFILL=1).
        let warm_seq = prefill_bucket_len(16, max_seq);
        let t_compile = Instant::now();
        session.ensure_prefill_bucket(warm_seq)?;
        step!(t_compile, "ensure_prefill_bucket(warm) done at");
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
        hidden_only: bool,
    ) -> (Graph, HashMap<String, Vec<f32>>) {
        let mut loader = EmptyWeightLoader;
        let mut local_packed = HashMap::new();
        build_gemma_graph_sized_packed_ext(
            cfg,
            &mut loader,
            1,
            seq,
            !hidden_only,
            !hidden_only,
            !hidden_only,
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
        self.ensure_prefill_bucket_kind(seq, false)
    }

    fn ensure_prefill_hidden_bucket(&mut self, seq: usize) -> Result<()> {
        self.ensure_prefill_bucket_kind(seq, true)
    }

    fn ensure_prefill_bucket_kind(&mut self, seq: usize, hidden_only: bool) -> Result<()> {
        let trace = std::env::var("RLX_GEMMA_TRACE_INIT").is_ok();
        let key = prefill_cache_key(seq, hidden_only);
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
            let t_graph = Instant::now();
            let (graph, params) =
                Self::build_prefill_graph(&cfg, &f32_params, &packed_tensors, seq, hidden_only);
            if trace {
                eprintln!(
                    "[gemma-runner trace]   build_prefill_graph(seq={seq} hidden_only={hidden_only}) done at {:.1}s ({} param entries)",
                    t_graph.elapsed().as_secs_f64(),
                    params.len(),
                );
            }
            let t_compile = Instant::now();
            let compiled = self
                .prefill_cache
                .get_or_compile_with_options(key, || graph, &opts);
            if trace {
                eprintln!(
                    "[gemma-runner trace]   prefill compile done at {:.1}s",
                    t_compile.elapsed().as_secs_f64()
                );
            }
            let t_f32 = Instant::now();
            for (name, data) in &params {
                compiled.set_param(name, data);
            }
            if trace {
                eprintln!(
                    "[gemma-runner trace]   set_param f32 ({} entries) done at {:.1}s",
                    params.len(),
                    t_f32.elapsed().as_secs_f64()
                );
            }
            if packed_loaded.insert(key) {
                let t_packed = Instant::now();
                let n_packed = packed_for_upload.len();
                for (name, (bytes, _scheme, _shape)) in packed_for_upload.iter() {
                    compiled.set_param_typed(name, bytes, rlx_ir::DType::U8);
                }
                if trace {
                    eprintln!(
                        "[gemma-runner trace]   set_param_typed packed ({n_packed} entries) done at {:.1}s",
                        t_packed.elapsed().as_secs_f64()
                    );
                }
            }
        });
        Ok(())
    }

    fn prefill_execute_warm(&mut self, seq: usize) -> Result<()> {
        self.padded_ids.resize(seq, 0);
        self.ids_f32.resize(seq, 1.0);
        self.last_idx[0] = 0.0;
        let key = prefill_cache_key(seq, false);
        let compiled = self.prefill_cache.get_or_compile_with_options(
            key,
            || unreachable!("warm bucket"),
            &self.prefill_opts,
        );
        // Lazy-embed builds expect `input_embeddings` instead of `input_ids` —
        // feed a zero buffer to warm without paying the dequant cost.
        let h = self.cfg.hidden_size;
        let lazy = self.embed_row_bytes.is_some();
        if lazy {
            self.embed_scratch.resize(seq * h, 0.0);
            for v in self.embed_scratch.iter_mut() {
                *v = 0.0;
            }
            let _ = compiled.run(&[
                ("input_embeddings", self.embed_scratch.as_slice()),
                ("last_token_idx", self.last_idx.as_slice()),
            ]);
        } else {
            let _ = compiled.run(&[
                ("input_ids", self.ids_f32.as_slice()),
                ("last_token_idx", self.last_idx.as_slice()),
            ]);
        }
        Ok(())
    }

    fn prewarm_decode_buckets(&mut self) -> Result<()> {
        // Each warmed bucket uploads a full copy of the packed Q4 weights to
        // its compiled-graph param storage (~6.6 GB for 12B); warming every
        // bucket up to max_seq blows past unified memory. Default keeps the
        // single-bucket behaviour from `warm_past_seqs` and honours its env
        // override (`RLX_GEMMA_PACKED_WARM_PAST=...`). Cross-bucket weight
        // sharing is tracked separately — see [[follow-up]].
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

        // Task #37 lazy-embed path: host-gather one row per padded token from
        // the Q4K-packed embed bytes. Active rows (`< n`) get real values,
        // padding rows stay zeroed (their attention positions are masked).
        let h = self.cfg.hidden_size;
        let lazy = self.embed_row_bytes.is_some();
        if lazy {
            self.embed_scratch.resize(seq_bucket * h, 0.0);
            for v in self.embed_scratch.iter_mut() {
                *v = 0.0;
            }
            let (bytes, scheme, _shape) = self
                .packed_tensors
                .get("model.embed_tokens.weight")
                .expect("lazy embed: packed entry must be present");
            for (i, &tok) in prompt_ids.iter().take(n).enumerate() {
                let row_off = i * h;
                gather_embed_row(
                    bytes,
                    *scheme,
                    h,
                    tok as usize,
                    &mut self.embed_scratch[row_off..row_off + h],
                )?;
            }
        }

        let t0 = Instant::now();
        let key = prefill_cache_key(seq_bucket, false);
        let compiled = self.prefill_cache.get_or_compile_with_options(
            key,
            || unreachable!("prefill bucket"),
            &self.prefill_opts,
        );
        let inputs_id_pair = ("input_ids", self.ids_f32.as_slice());
        let inputs_emb_pair = if lazy {
            Some(("input_embeddings", self.embed_scratch.as_slice()))
        } else {
            None
        };
        let last_pair = ("last_token_idx", self.last_idx.as_slice());
        // Build the slice the runtime expects; one of input_ids /
        // input_embeddings is unused depending on the build path.
        let outputs = if let Some(emb_pair) = inputs_emb_pair {
            run_packed_prefill(
                compiled,
                self.exec_device,
                n,
                seq_bucket,
                &[emb_pair, last_pair],
            )
        } else {
            run_packed_prefill(
                compiled,
                self.exec_device,
                n,
                seq_bucket,
                &[inputs_id_pair, last_pair],
            )
        };
        if packed_timing_enabled() {
            let active = packed_prefill_active_extent_enabled(self.exec_device) && n < seq_bucket;
            eprintln!(
                "[gemma-packed] prefill n={n} bucket={seq_bucket} active={active} {:.1} ms",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }

        let kv_dims = self.per_layer_kv_dims();
        // RLX_TAP_L0: builder appended 11 layer-0 tap tensors as outputs after
        // [logits, k0, v0, ...]. Print stats and drop them before the KV
        // consumer (which expects exactly 1 + 2*num_layers slots).
        let mut outputs = outputs;
        if std::env::var("RLX_TAP_L0").ok().is_some() {
            let expected_kv = 2 * self.cfg.num_hidden_layers;
            let total_kv_logits = 1 + expected_kv;
            if outputs.len() > total_kv_logits {
                let tap_start = total_kv_logits;
                let labels = [
                    "1. h_id (embed*scale)",
                    "2. input_layernorm(x)",
                    "A. Q POST-PROJ (pre-norm)",
                    "B. K POST-PROJ (pre-norm)",
                    "C. V POST-PROJ (pre-norm)",
                    "D. Q reshape 4D (pre-norm)",
                    "E. Q after per-head rms_norm (4D)",
                    "3. Q post-norm (reshape back)",
                    "4. K post-norm",
                    "5. V post-norm",
                    "6. Q post-RoPE",
                    "7. K post-RoPE",
                    "F. K_rep (post repeat_kv) -> SDPA input",
                    "G. V_rep (post repeat_kv) -> SDPA input",
                    "8. attention out (pre-o_proj)",
                    "9. attn_out post post_attn_norm",
                    "10. residual h + attn_out",
                    "11. layer 0 final h",
                ];
                eprintln!(
                    "[rlx-tap-l0] device={:?} prompt_len={n} bucket={seq_bucket}",
                    self.exec_device
                );
                for (i, t) in outputs[tap_start..].iter().enumerate() {
                    let label = labels.get(i).copied().unwrap_or("?");
                    let mut n_nan = 0usize;
                    let mut n_finite = 0usize;
                    let mut min = f32::INFINITY;
                    let mut max = f32::NEG_INFINITY;
                    let mut sumsq = 0f64;
                    for &v in t {
                        if v.is_nan() {
                            n_nan += 1;
                            continue;
                        }
                        n_finite += 1;
                        if v < min {
                            min = v;
                        }
                        if v > max {
                            max = v;
                        }
                        sumsq += (v as f64) * (v as f64);
                    }
                    let rms = (sumsq / n_finite.max(1) as f64).sqrt();
                    eprintln!(
                        "[rlx-tap-l0]   tap {:<32} len={:>7} nan={n_nan:>5} finite={n_finite:>7} min={min:+.3e} max={max:+.3e} rms={rms:.3e}",
                        label,
                        t.len()
                    );
                }
                outputs.truncate(total_kv_logits);
            }
        }
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
        // Task #37 lazy embed: dequant the single decode token row host-side.
        let h = self.cfg.hidden_size;
        let lazy = self.embed_row_bytes.is_some();
        if lazy {
            self.embed_scratch.resize(h, 0.0);
            let (bytes, scheme, _shape) = self
                .packed_tensors
                .get("model.embed_tokens.weight")
                .expect("lazy embed: packed entry must be present");
            gather_embed_row(
                bytes,
                *scheme,
                h,
                input_tok as usize,
                &mut self.embed_scratch[..h],
            )?;
        }
        let mut fixed = vec![
            if lazy {
                CacheRunInput {
                    name: "input_embeddings",
                    data: self.embed_scratch.as_slice(),
                    row_inner: None,
                }
            } else {
                CacheRunInput {
                    name: "input_ids",
                    data: &input_ids_f32,
                    row_inner: None,
                }
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

    /// Post-`model.norm` hidden vector for the last prompt token (3840 dims for 12B).
    pub fn predict_last_hidden(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        let n = prompt_ids.len().min(self.max_seq);
        let seq_bucket = prefill_bucket_len(n, self.max_seq);
        self.ensure_prefill_hidden_bucket(seq_bucket)?;

        self.padded_ids.resize(seq_bucket, 0);
        self.ids_f32.resize(seq_bucket, 0.0);
        self.padded_ids.fill(0);
        for (i, &t) in prompt_ids.iter().take(n).enumerate() {
            self.padded_ids[i] = t;
        }
        for (dst, &id) in self.ids_f32.iter_mut().zip(self.padded_ids.iter()) {
            *dst = id as f32;
        }

        let h = self.cfg.hidden_size;
        let lazy = self.embed_row_bytes.is_some();
        if lazy {
            self.embed_scratch.resize(seq_bucket * h, 0.0);
            for v in self.embed_scratch.iter_mut() {
                *v = 0.0;
            }
            let (bytes, scheme, _shape) = self
                .packed_tensors
                .get("model.embed_tokens.weight")
                .expect("lazy embed: packed entry must be present");
            for (i, &tok) in prompt_ids.iter().take(n).enumerate() {
                let row_off = i * h;
                gather_embed_row(
                    bytes,
                    *scheme,
                    h,
                    tok as usize,
                    &mut self.embed_scratch[row_off..row_off + h],
                )?;
            }
        }

        let key = prefill_cache_key(seq_bucket, true);
        let compiled = self.prefill_cache.get_or_compile_with_options(
            key,
            || unreachable!("prefill hidden bucket"),
            &self.prefill_opts,
        );
        let outputs = if lazy {
            run_packed_prefill(
                compiled,
                self.exec_device,
                n,
                seq_bucket,
                &[("input_embeddings", self.embed_scratch.as_slice())],
            )
        } else {
            run_packed_prefill(
                compiled,
                self.exec_device,
                n,
                seq_bucket,
                &[("input_ids", self.ids_f32.as_slice())],
            )
        };
        let hidden = outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("prefill hidden graph returned no outputs"))?;
        let last = n.saturating_sub(1);
        let need = h;
        let start = last * need;
        let end = start + need;
        if hidden.len() < end {
            bail!(
                "hidden short for last token: {} < {end} (n={n} bucket={seq_bucket})",
                hidden.len()
            );
        }
        Ok(hidden[start..end].to_vec())
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
