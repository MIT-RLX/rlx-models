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

//! Host-side generation loop for Qwen3.
//!
//! This is the **naive** generator: each `step()` rebuilds the prefill
//! graph for the full token history and runs it from scratch
//! (O(N²) compute over N generated tokens). The API is shaped to
//! match the upcoming KV-cache version exactly so callers don't have
//! to change anything when the cached path lands — only the internal
//! implementation swaps.
//!
//! Why ship the naive version first:
//!   - Establishes the public API contract before the IR/kernel
//!     changes that the cached version needs land.
//!   - Lets you run end-to-end generation against a real checkpoint
//!     today and validate the prefill graph is numerically correct.
//!   - Provides a reference baseline for the cached version's own
//!     numerical-parity test (cached vs recompute must match).

use crate::builder::{
    build_qwen3_decode_graph_sized, build_qwen3_decode_graph_sized_ext,
    build_qwen3_decode_graph_sized_qk, build_qwen3_decode_graph_sized_ragged,
    build_qwen3_graph_sized, build_qwen3_graph_sized_last_logits,
};
use crate::capabilities::validate_device;
use crate::config::Qwen3Config;
#[cfg(feature = "mmap-kv")]
use crate::embedder::{BlockEmbedder, EmbedderKind, TokenMeanEmbedder};
use crate::profile::qwen3_profile_near_weights;
use crate::sampling::{SampleOpts, sample_token};
use anyhow::{Context, Result};
use rlx_core::autoregressive::{
    DecodeLogitsKv, KvCacheState, compile_cache_ensure_graph, kv_from_prefill_outputs,
    prefill_cache_key, run_bucketed_kv_decode, run_bucketed_kv_decode_packed,
    split_decode_logits_kv, split_decode_logits_kv_aux,
};
use rlx_core::flow_bridge::compile_options_from_profile;
use rlx_core::gpu_kv::{GpuKvBinding, device_supports_gpu_kv};
use rlx_core::weight_loader::WeightLoader;
use rlx_core::weight_map::WeightMap;
use rlx_flow::CompileProfile;
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_runtime::attn_mask::bucket_decode_mask;
use rlx_runtime::compile_cache::{BucketedCompileCache, CacheRunInput, CompileCache, pad_rows};
#[cfg(feature = "mmap-kv")]
use rlx_runtime::kv_context_store::{KvContextStore, Origin};
use rlx_runtime::kv_retention::{KvRetentionManager, KvRetentionPolicy};
use rlx_runtime::{CompileOptions, Device, Session};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Stateful Qwen3 generation handle.
///
/// Holds the (config, weight bytes, token history) and rebuilds a
/// prefill graph on each [`step`] call. Cheap to construct after
/// initial weight load; tokens stay in-memory between calls.
pub struct Qwen3Generator {
    cfg: Qwen3Config,
    /// Map of weight key → (f32 data, shape), behind an `Arc` so the
    /// per-step setup is an O(1) refcount bump, not a deep copy of every
    /// weight. `WeightMap::take` is destructive, so a fresh `WeightMap` is
    /// still materialized (`(*arc).clone()`) when a graph is *built* — but
    /// with the bucketed decode cache that happens only on a compile miss,
    /// not on every decode step (the hot path captures just the `Arc`).
    weights_cache: Arc<HashMap<String, (Vec<f32>, Vec<usize>)>>,
    /// Optional K-quant packed bytes per linear weight key (`name → (bytes,
    /// scheme)`). When set, the host decode path rewrites the F32 weight-MatMuls
    /// into packed `Op::DequantMatMul` and binds these U8 bytes instead of the
    /// dequanted f32 — ~4× fewer weight bytes moved per decode step (the
    /// weight-bytes-bound bottleneck). `None` = the default F32 decode path.
    packed_weights: Option<Arc<HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme)>>>,
    tokens: Vec<u32>,
    device: Device,
    /// Populated lazily on the first `step_cached` call (seeded from
    /// the prompt via prefill-with-cache); thereafter advanced by each
    /// decode step.
    cache: Option<KvCacheState>,
    /// Per-key LRU compile cache for prefill graphs. Keyed by `seq`.
    /// Set to `None` to disable (default for new instances; opt in via
    /// [`Qwen3Generator::with_prefill_cache`]).
    prefill_compile_cache: Option<CompileCache>,
    /// Bucketed compile cache for decode-mode graphs. Each bucket
    /// holds one compiled graph specialized at its upper-bound
    /// `past_seq`; the host pads `past_k`/`past_v` and supplies a
    /// per-step mask so a single bucket serves every `past_seq` in
    /// its range. Opt in via [`Qwen3Generator::with_decode_cache`].
    decode_compile_cache: Option<BucketedCompileCache>,
    /// Bucketed decode compile caches for **fused batched** decode, one per
    /// batch size `B` (the decode graph shape is specialized on `B`). Built
    /// lazily by [`Qwen3Generator::decode_batched_uniform`].
    batched_decode_caches: HashMap<usize, BucketedCompileCache>,
    /// Like [`Self::batched_decode_caches`] but for **ragged** batched decode
    /// (per-sequence RoPE rows), keyed by batch size `B`.
    batched_ragged_caches: HashMap<usize, BucketedCompileCache>,
    prefill_profile: CompileProfile,
    decode_profile: CompileProfile,
    /// GPU-resident decode: bind K/V to on-device handles + fold only the new
    /// token's row in place each step (row feed, no host round-trip, no handle
    /// growth). Fixes the async-MLX per-step-sync regression. Auto-on for GPU.
    use_gpu_kv: bool,
    gpu_kv_binding: GpuKvBinding,
    /// Optional selective KV retention (Stage 2). When set, after each decode
    /// step the resident K/V is reshaped per the policy's [`RetentionPlan`]
    /// (keep sinks + recent + heavy-hitters, evict/offload the rest) so effective
    /// context extends beyond a bounded resident budget. `None` = keep all.
    retention: Option<KvRetentionManager>,
    /// Optional per-step data inspection of the KV cache + selection preferences
    /// (shape/stats/histograms/dataflow), recorded during `apply_retention`.
    /// Enabled via [`enable_inspect`](Self::enable_inspect); read back with
    /// [`take_inspect_log`](Self::take_inspect_log). `None` = no recording.
    /// Optional disk-tiered million-token context store (HNSW + quantized mmap).
    /// When set, retention offloads aged-out blocks here (append-once) and each
    /// step retrieves the top-k relevant blocks from it and splices them into the
    /// resident cache — extending effective context far beyond RAM. See
    /// [`enable_kv_store`](Self::enable_kv_store).
    #[cfg(feature = "mmap-kv")]
    kv_store: Option<KvStoreState>,
    /// When true, the store's offload+splice is skipped in `apply_retention`
    /// (decode runs as plain resident-window attention) while the store stays
    /// available for read-only [`retrieve_context_spans`](Self::retrieve_context_spans).
    /// Used by the text-reinjection path (D): retrieve facts as TEXT, then
    /// generate on a clean labeled prompt with no raw-KV splice polluting it.
    #[cfg(feature = "mmap-kv")]
    kv_store_suspended: bool,
    /// Frozen copy of the token history used to recover retrieved blocks' text in
    /// [`retrieve_context_spans`](Self::retrieve_context_spans). The interleave
    /// loop re-prefills a growing reasoning transcript each hop (clobbering
    /// `self.tokens`), so retrieval must recover spans from this snapshot of the
    /// ORIGINAL stream instead. `None` = use the live `self.tokens`.
    #[cfg(feature = "mmap-kv")]
    retrieval_stream: Option<Vec<u32>>,
    /// When set, the decode step exports the model's post-RoPE query and
    /// [`apply_retention`](Self::apply_retention) scores cached blocks by the
    /// model's actual attention (Q·K) instead of the key-self-similarity proxy
    /// (K·K). Routes decode through [`decode_step_oneshot_q`](Self::decode_step_oneshot_q).
    q_scoring: bool,
    /// Optional clean retrieval query TEXT (the actual question), set by the caller
    /// per turn. When present and the KV-store embedder can embed text, retrieval
    /// uses this instead of the noisy decode-position token window.
    retrieval_query_text: Option<String>,
    /// Token ids of the blocks retrieved in the last `apply_retention` (from the
    /// original stream) — lets a harness attribute a miss to retrieval vs generation.
    last_retrieved_tokens: Vec<u32>,
    /// Newest decode token's layer-0 query, GQA-pooled to `kv_dim` — the Q·K
    /// retrieval query captured by the last Q-export decode step. `None` until
    /// the first such step. Consumed (not cleared) by `apply_retention`.
    last_q_pooled: Option<Vec<f32>>,
    inspect: Option<rlx_ir::tensor_inspect::InspectLog>,
    /// Monotonic step index for the inspect log (decode steps seen while on).
    inspect_step: usize,
}

/// Round-trip an f32 through f16 precision (round-to-nearest into a 10-bit
/// mantissa). The qwen3 K/V exponent ranges (K ±523, V ±2.5) sit inside f16's
/// range, so mantissa rounding faithfully models the f16 storage loss without a
/// `half` dependency. Used by the mixed-precision KV experiment (#4).
#[inline]
fn f16_roundtrip(x: f32) -> f32 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let bits = x.to_bits().wrapping_add(0x0000_1000) & 0xFFFF_E000;
    f32::from_bits(bits)
}

/// GQA-pool a query row (`[num_attention_heads * head_dim]`, head-major) down
/// to `kv_dim = num_kv_heads * head_dim` so it aligns with the K-cache / block
/// keys, which are stored per KV head. Each KV head `g` is shared by
/// `group_size = num_attention_heads / num_kv_heads` contiguous query heads;
/// the pooled query for head `g` is the mean of those query heads' `head_dim`
/// vectors. This mirrors how attention shares one KV head across a group, so
/// `pooled · block_key` approximates the block's aggregate attention.
fn gqa_pool_query(q: &[f32], head_dim: usize, num_kv_heads: usize, group_size: usize) -> Vec<f32> {
    let mut pooled = vec![0.0f32; num_kv_heads * head_dim];
    if group_size == 0 {
        return pooled;
    }
    let inv = 1.0 / group_size as f32;
    for g in 0..num_kv_heads {
        for h in 0..group_size {
            let qh = (g * group_size + h) * head_dim;
            if qh + head_dim > q.len() {
                break;
            }
            let base = g * head_dim;
            for d in 0..head_dim {
                pooled[base + d] += q[qh + d] * inv;
            }
        }
    }
    pooled
}

/// Group resident indices into maximal runs whose **absolute positions** are
/// consecutive (`positions[i]` differs by 1). Unlike [`consecutive_runs`] (which
/// groups by consecutive index), this splits at abs-position gaps — so a store
/// block never spans a hole left by non-resident middle context, and each block
/// is a run of adjacent tokens (stable, meaningful keys). `indices` must be in
/// ascending abs-position order (plan.evict is).
// Abs-position-aware run splitter: designed + unit-tested, not yet wired into
// the offload path (production currently uses `consecutive_runs`). Kept for it.
#[allow(dead_code)]
fn consecutive_abs_runs(indices: &[usize], positions: &[usize]) -> Vec<Vec<usize>> {
    let mut runs: Vec<Vec<usize>> = Vec::new();
    for &i in indices {
        let pos = match positions.get(i) {
            Some(&p) => p,
            None => continue,
        };
        let extend = runs
            .last()
            .and_then(|last| last.last())
            .and_then(|&li| positions.get(li))
            .is_some_and(|&lp| lp + 1 == pos);
        if extend {
            runs.last_mut().unwrap().push(i);
        } else {
            runs.push(vec![i]);
        }
    }
    runs
}

/// Split a sorted, ascending index list into maximal runs of consecutive
/// integers — so evicted resident rows can be chunked into contiguous blocks.
fn consecutive_runs(idxs: &[usize]) -> Vec<Vec<usize>> {
    let mut runs: Vec<Vec<usize>> = Vec::new();
    for &i in idxs {
        match runs.last_mut() {
            Some(last) if *last.last().unwrap() + 1 == i => last.push(i),
            _ => runs.push(vec![i]),
        }
    }
    runs
}

/// Disk-tiered context-store retention state (feature `mmap-kv`).
#[cfg(feature = "mmap-kv")]
struct KvStoreState {
    store: KvContextStore,
    /// Block start positions already offloaded (append-once dedup).
    offloaded: std::collections::HashSet<usize>,
    /// Blocks to retrieve per step.
    topk: usize,
    /// Similar-neighbor blocks to pull per hit (0 = off).
    neighbors: usize,
    /// Fuzzy relevance floor (drop matches below it); `f32::NEG_INFINITY` = off.
    min_score: f32,
    /// Apply memory-decay re-ranking on retrieval (recent/used context wins).
    decay_on: bool,
    /// Hybrid lexical blend weight ∈ [0,1] (0 = dense only). >0 tags blocks with
    /// their token ids and blends an IDF token-overlap score into retrieval.
    lexical_weight: f32,
    /// How many recent tokens form the lexical query (≈ the current question).
    query_window: usize,
    /// MaxSim late-interaction re-rank: over-fetch candidates then rank by the max
    /// over each block's rows of q·K_row (beats mean/centroid pooling for exact
    /// facts). Takes priority over the other retrieve variants when on.
    maxsim: bool,
    /// Candidate over-fetch multiplier for MaxSim re-rank (`topk * this`).
    maxsim_overfetch: usize,
    /// Exact brute-force retrieval (bypass HNSW; recovers recall the approximate
    /// index loses on clustered K keys).
    exact: bool,
    /// Pool the query over the last N resident K rows (≈ the question); 1 = newest.
    query_pool: usize,
    /// Score retrieval across all layers' K (exact); catches middle-layer facts.
    multilayer: bool,
    /// Store block size (rows/block) — the accumulation target for `pending`.
    block: usize,
    /// Aged-out rows awaiting accumulation into a coherent `block`-row store block:
    /// `(abs_pos, token_id, per-layer 1-row K, per-layer 1-row V)`, ascending abs.
    /// Offloading one row per step would fragment a multi-token fact across many
    /// tiny blocks so top-k never returns its coherent span; staging into
    /// block-sized coherent spans mirrors the in-RAM path that recalls 3/3.
    pending: Vec<(usize, u32, Vec<Vec<f32>>, Vec<Vec<f32>>)>,
    /// Optional semantic content embedder (dual-encoder path). When set, each
    /// flushed block is embedded (from its token ids) into the store's secondary
    /// HNSW, and retrieval blends embedding + lexical + optional K·K (hybrid3).
    embedder: Option<Box<dyn crate::embedder::BlockEmbedder>>,
    /// hybrid3 embedding-term / K·K-term weights (lexical uses `lexical_weight`).
    embed_weight: f32,
    dense_weight: f32,
    /// Relevance gate ∈ [0,1): keep retrieved blocks scoring ≥ gate×top (noise cut).
    relevance_gate: f32,
    /// Fuse embedding + lexical(+K·K) via reciprocal rank fusion (vs weighted-sum).
    rrf: bool,
}

#[cfg(feature = "mmap-kv")]
impl KvStoreState {
    /// Flush every complete `block`-row, abs-contiguous span from the front of
    /// `pending` into the store as one coherent block; keep the trailing partial
    /// (and anything after an abs-position gap) buffered for next time.
    fn flush_pending(&mut self, kv_dim: usize, n_layers: usize) {
        loop {
            if self.pending.is_empty() {
                break;
            }
            // Longest abs-contiguous prefix length.
            let mut run = 1usize;
            while run < self.pending.len() && self.pending[run].0 == self.pending[run - 1].0 + 1 {
                run += 1;
            }
            // A still-growing tail (contiguous to the end) shorter than a block
            // waits for more rows; a run closed by an abs-position gap is flushed
            // now (as a short block) so its context isn't stranded.
            if run == self.pending.len() && run < self.block {
                break;
            }
            let take = run.min(self.block);
            let chunk: Vec<(usize, u32, Vec<Vec<f32>>, Vec<Vec<f32>>)> =
                self.pending.drain(0..take).collect();
            let start = chunk[0].0;
            let mut k: Vec<Vec<f32>> = vec![Vec::with_capacity(take * kv_dim); n_layers];
            let mut v: Vec<Vec<f32>> = vec![Vec::with_capacity(take * kv_dim); n_layers];
            let mut toks: Vec<u32> = Vec::with_capacity(take);
            for (_, tok, kr, vr) in &chunk {
                toks.push(*tok);
                for l in 0..n_layers {
                    k[l].extend_from_slice(&kr[l]);
                    v[l].extend_from_slice(&vr[l]);
                }
            }
            // Mean-K key (HNSW nav); RowKeys mode re-derives salient rows internally.
            let mut key = vec![0.0f32; kv_dim];
            for r in 0..take {
                for j in 0..kv_dim {
                    key[j] += k[0][r * kv_dim + j];
                }
            }
            for x in key.iter_mut() {
                *x /= take as f32;
            }
            if let Ok(id) = self
                .store
                .append_block(start, Origin::Generated, 0, &k, &v, &key)
            {
                if self.lexical_weight > 0.0 {
                    self.store.attach_tokens(id, &toks);
                }
                // Semantic index: embed the block's token span (content embedding)
                // into the store's secondary HNSW for selective dual-encoder recall.
                if let Some(emb) = self.embedder.as_ref() {
                    let e = emb.embed_document(&toks);
                    self.store.append_embed(id, &e);
                }
            }
        }
    }
}

/// Builder for [`Qwen3Generator::enable_kv_store`] — the disk-tiered
/// million-token context store as the retention backend. Sensible defaults;
/// override only what you need:
///
/// ```ignore
/// gen.enable_kv_store(KvStoreConfig::new().dir("/tmp/ctx").topk(24).centroids_per_block(4).decay(0.999))?;
/// ```
#[cfg(feature = "mmap-kv")]
#[derive(Clone, Debug)]
pub struct KvStoreConfig {
    dir: Option<std::path::PathBuf>,
    capacity_tokens: usize,
    block: usize,
    sinks: usize,
    recent: usize,
    topk: usize,
    neighbors: usize,
    metric: rlx_runtime::hnsw::Metric,
    min_score: f32,
    centroids_per_block: usize,
    decay: f32,
    lexical_weight: f32,
    query_scoring: bool,
    scheme: rlx_runtime::quantized_kv::KvQuant,
    maxsim: bool,
    maxsim_overfetch: usize,
    row_keys: bool,
    exact: bool,
    query_pool: usize,
    multilayer: bool,
    embedder: crate::embedder::EmbedderKind,
    embed_weight: f32,
    dense_weight: f32,
    relevance_gate: f32,
    rrf: bool,
    query_window_override: usize,
}

#[cfg(feature = "mmap-kv")]
impl Default for KvStoreConfig {
    fn default() -> Self {
        Self {
            dir: None, // anonymous (swap-backed) mmap
            capacity_tokens: 1_000_000,
            block: 16,
            sinks: 4,
            recent: 32,
            topk: 12,
            neighbors: 2,
            metric: rlx_runtime::hnsw::Metric::L2, // true metric (sound navigation)
            min_score: f32::NEG_INFINITY,          // fuzzy floor off
            centroids_per_block: 4,                // k-means sub-keys (less averaging)
            decay: 1.0,                            // memory decay off
            lexical_weight: 0.0,                   // hybrid lexical off (dense only)
            query_scoring: false,                  // score by K·K proxy (Q·K = opt-in)
            // Q8_0: ~4× smaller than F16, and K/V round-trip within ~1% (V is
            // int8-safe; K's 24×-σ outliers are the risk — F16 isolates quant
            // as a spliced-KV corruption cause).
            scheme: rlx_runtime::quantized_kv::KvQuant::Q8_0,
            maxsim: false,       // late-interaction re-rank (opt-in)
            maxsim_overfetch: 4, // re-rank pool = topk × 4
            row_keys: false,     // late-interaction HNSW index (opt-in)
            exact: false,        // brute-force exact retrieval (bypass HNSW)
            query_pool: 1,       // query = newest token's K (1); >1 = mean last-N
            multilayer: false,   // score across all layers' K (not just layer 0)
            embedder: crate::embedder::EmbedderKind::None, // semantic index off
            embed_weight: 1.0,   // hybrid3 embedding term weight
            dense_weight: 0.0,   // hybrid3 K·K term weight (off by default)
            relevance_gate: 0.0, // adaptive-k noise gate (0 = keep all topk)
            rrf: false,          // reciprocal-rank fusion of embed+lexical(+dense)
            query_window_override: 0, // 0 = auto (recent.max(block))
        }
    }
}

#[cfg(feature = "mmap-kv")]
impl KvStoreConfig {
    pub fn new() -> Self {
        Self::default()
    }
    /// Persist to one mmap file per layer under `dir` (else anonymous).
    pub fn dir(mut self, p: impl Into<std::path::PathBuf>) -> Self {
        self.dir = Some(p.into());
        self
    }
    pub fn capacity_tokens(mut self, n: usize) -> Self {
        self.capacity_tokens = n;
        self
    }
    /// Block size (rows/block) for offload + retrieval granularity.
    pub fn block(mut self, n: usize) -> Self {
        self.block = n;
        self
    }
    /// Always-resident attention-sink + recent-window sizes.
    pub fn sinks(mut self, n: usize) -> Self {
        self.sinks = n;
        self
    }
    pub fn recent(mut self, n: usize) -> Self {
        self.recent = n;
        self
    }
    /// Blocks retrieved per step, and similar-neighbor blocks pulled per hit.
    pub fn topk(mut self, n: usize) -> Self {
        self.topk = n;
        self
    }
    pub fn neighbors(mut self, n: usize) -> Self {
        self.neighbors = n;
        self
    }
    /// Similarity metric (`L2` distance / `Dot` / `Cosine`).
    pub fn metric(mut self, m: rlx_runtime::hnsw::Metric) -> Self {
        self.metric = m;
        self
    }
    /// Fuzzy relevance floor — drop matches below it (`NEG_INFINITY` = off).
    pub fn fuzzy_floor(mut self, s: f32) -> Self {
        self.min_score = s;
        self
    }
    /// k-means centroids per block (>1 avoids averaging away detail).
    pub fn centroids_per_block(mut self, n: usize) -> Self {
        self.centroids_per_block = n;
        self
    }
    /// Per-step recency multiplier ∈ (0,1] (`1.0` = no memory decay).
    pub fn decay(mut self, d: f32) -> Self {
        self.decay = d;
        self
    }
    /// Hybrid lexical blend weight ∈ [0,1] (0 = dense only). Blends an IDF
    /// token-overlap score so exact facts (numbers, names, shared keywords) that
    /// K·K similarity misses are still retrieved.
    pub fn lexical_weight(mut self, w: f32) -> Self {
        self.lexical_weight = w;
        self
    }
    /// Score cached blocks by the model's actual attention query (Q·K) rather
    /// than the key-self-similarity proxy (K·K). Exports the decode-step query
    /// (post-RoPE, GQA-pooled to `kv_dim`) — the retrieval-relevance root fix.
    /// Forces the one-shot Q-export decode path (rebuilds the graph per step),
    /// so it's slower; intended for retrieval-quality work, not peak decode.
    pub fn query_scoring(mut self, on: bool) -> Self {
        self.query_scoring = on;
        self
    }
    /// On-disk quantization scheme for offloaded K/V (`Q8_0` default, `F16` for
    /// near-lossless, `Q4_0` for max density). Lower precision saves disk/RAM at
    /// the cost of spliced-KV fidelity — use `F16` to rule quant out as a cause
    /// of degraded generation after retrieval.
    pub fn scheme(mut self, s: rlx_runtime::quantized_kv::KvQuant) -> Self {
        self.scheme = s;
        self
    }
    /// MaxSim late-interaction re-ranking (ColBERT-style): over-fetch candidate
    /// blocks via HNSW, then rank each by the *max* over its rows of `query·K_row`
    /// instead of a mean/centroid key — so a block containing one exact-fact token
    /// scores as high as its best-matching position. The relevance fix for exact
    /// recall that mean pooling washes out.
    pub fn maxsim(mut self, on: bool) -> Self {
        self.maxsim = on;
        self
    }
    /// Candidate over-fetch multiplier for [`maxsim`](Self::maxsim) (`topk × n`
    /// blocks are re-scored). Larger widens the re-rank pool at more read cost.
    pub fn maxsim_overfetch(mut self, n: usize) -> Self {
        self.maxsim_overfetch = n.max(1);
        self
    }
    /// Index the block's most-salient actual K rows as HNSW keys (no averaging)
    /// instead of mean/k-means centroids, so navigation reaches a block via its
    /// strong-matching token — the index-side of MaxSim. Complements
    /// [`maxsim`](Self::maxsim) (index finds candidates; re-rank orders them).
    pub fn row_keys(mut self, on: bool) -> Self {
        self.row_keys = on;
        self
    }
    /// Exact brute-force retrieval (score every block, bypass the HNSW index).
    /// HNSW greedy nav has poor recall on the clustered post-RoPE-K key
    /// distribution and silently misses the true-nearest block; exact scoring
    /// recovers recall (matches the in-RAM manager). O(blocks·rows·dim) — for
    /// moderate context; HNSW is the million-block scale path.
    pub fn exact(mut self, on: bool) -> Self {
        self.exact = on;
        self
    }
    /// Pool the retrieval query over the last `w` resident K rows (the recent
    /// window ≈ the current question) instead of just the newest token's K.
    /// `1` = newest only. A cheap query-side relevance lever that works on the
    /// fast bucketed decode path (unlike Q·K, which forces the one-shot path).
    /// Ignored when [`query_scoring`](Self::query_scoring) (Q·K) is on.
    pub fn query_pool(mut self, w: usize) -> Self {
        self.query_pool = w.max(1);
        self
    }
    /// Score retrieval across ALL layers' K (sum of per-layer MaxSim), not just
    /// layer 0. Facts the model attends to in middle layers are invisible to
    /// layer-0-only scoring; the all-layer sum is a much stronger K-space signal.
    /// Implies exact brute-force retrieval (the all-layer path is exact-only).
    pub fn multilayer(mut self, on: bool) -> Self {
        self.multilayer = on;
        self
    }
    /// Semantic **content-embedding** retrieval index (dual-encoder path): each
    /// block is embedded and retrieval scores by embedding similarity (blended
    /// with lexical + optional K·K). This is the selective, 1M-scalable signal —
    /// K·K is not. `TokenMean` is self-contained (no download); `Encoder` uses a
    /// dedicated retrieval model (`dual-encoder` feature).
    pub fn embedder(mut self, kind: crate::embedder::EmbedderKind) -> Self {
        self.embedder = kind;
        self
    }
    /// hybrid3 blend weights: embedding term and K·K (dense) term. Lexical uses
    /// [`lexical_weight`](Self::lexical_weight). Only used when an embedder is set.
    pub fn embed_weight(mut self, w: f32) -> Self {
        self.embed_weight = w;
        self
    }
    pub fn dense_weight(mut self, w: f32) -> Self {
        self.dense_weight = w;
        self
    }
    /// Relevance gate ∈ [0,1) — the noise minimizer. After ranking, keep only
    /// retrieved blocks whose blended score ≥ `gate × top_score`; drop the rest.
    /// So a dominant fact block is spliced alone (no irrelevant filler flooding
    /// the model — the cause of degenerate/blank generation at large topk).
    /// `0` = keep all topk; `0.7-0.9` = aggressive noise cut.
    pub fn relevance_gate(mut self, g: f32) -> Self {
        self.relevance_gate = g;
        self
    }
    /// Fuse the embedding + lexical(+K·K) rankings with **Reciprocal Rank Fusion**
    /// instead of weighted-sum. Rank-based → immune to score-scale mismatch, so
    /// lexical (BM25 for exact tokens: numbers/names/IPs) can be combined with the
    /// semantic encoder without the noise that sank weighted-sum. Uses lexical when
    /// `lexical_weight>0`, K·K when `dense_weight>0`. Pushes recall past embed-only.
    pub fn rrf(mut self, on: bool) -> Self {
        self.rrf = on;
        self
    }
    /// Number of recent tokens that form the retrieval query (0 = auto =
    /// `recent.max(block)`). Smaller = a tighter, cleaner query (just the question
    /// tail, less chat-scaffolding noise); larger drags in stale context.
    pub fn query_window(mut self, w: usize) -> Self {
        self.query_window_override = w;
        self
    }
}

/// Parse `RLX_QWEN3_RETENTION` into a [`KvRetentionManager`] (Stage-2 opt-in):
/// `sinks:S:W` · `heavy:S:R:B` · `retrieval:BLK:RB:S:R` · `auto:MAX`. Unset or
/// unrecognized → `None` (keep-all).
fn retention_from_env(kv_dim: usize) -> Option<KvRetentionManager> {
    let spec = std::env::var("RLX_QWEN3_RETENTION").ok()?;
    let p: Vec<&str> = spec.split(':').collect();
    let n = |i: usize| p.get(i).and_then(|s| s.parse::<usize>().ok());
    let policy = match p.first().copied()? {
        "sinks" => KvRetentionPolicy::Sinks {
            sinks: n(1)?,
            window: n(2)?,
        },
        "heavy" => KvRetentionPolicy::HeavyHitter {
            sinks: n(1)?,
            recent: n(2)?,
            budget: n(3)?,
        },
        "retrieval" => KvRetentionPolicy::Retrieval {
            block: n(1)?,
            resident_blocks: n(2)?,
            sinks: n(3)?,
            recent: n(4)?,
        },
        "auto" => KvRetentionPolicy::Auto {
            max_resident: n(1)?,
        },
        _ => return None,
    };
    eprintln!("[qwen3] KV retention: {policy:?}");
    Some(KvRetentionManager::new(policy, kv_dim))
}

impl Qwen3Generator {
    /// Construct from any [`WeightLoader`] — drains it into an
    /// internal cache so the loader is free after this call.
    pub fn from_loader(
        cfg: Qwen3Config,
        loader: &mut dyn WeightLoader,
        device: Device,
    ) -> Result<Self> {
        validate_device(&cfg, device, false)?;
        // `RLX_QWEN3_F16_WEIGHTS` declares the projection/LM-head params as F16 in
        // the graph so a bandwidth-bound backend can keep them f16-resident. Only
        // Metal converts the f32 param bytes → f16 at bind; other backends read
        // the F16-declared param as raw f32 bytes → garbage weights → gibberish.
        // Warn loudly rather than silently corrupt (the flow reads the env
        // directly, so we can't cleanly no-op it per-device from here).
        if rlx_ir::env::flag("RLX_QWEN3_F16_WEIGHTS") && device != Device::Metal {
            eprintln!(
                "[qwen3] WARNING: RLX_QWEN3_F16_WEIGHTS is set but device={device:?} is not \
                 Metal — F16-resident weights are only converted on Metal; on this backend \
                 they will be misread as f32 and produce garbage output. Unset the var (or \
                 run on Metal) unless you know this backend converts F16 params."
            );
        }
        let keys = loader.remaining_keys();
        let mut weights_cache = HashMap::with_capacity(keys.len());
        for k in keys {
            let v = loader
                .take(&k)
                .with_context(|| format!("draining weight {k}"))?;
            // Normalize the cache key to the safetensors / HuggingFace
            // naming convention so subsequent builder calls that ask
            // for `model.embed_tokens.weight` (the canonical name baked
            // into the qwen3 builder) hit the cache whether the
            // loader was safetensors-native or GGUF-native.
            let canonical =
                rlx_core::weight_loader::gguf_to_hf_name(&k).unwrap_or_else(|| k.clone());
            weights_cache.insert(canonical, v);
        }
        let max_past = cfg.max_position_embeddings.clamp(1, 4096);
        let retention = retention_from_env(cfg.kv_proj_dim());
        Ok(Self {
            cfg,
            weights_cache: Arc::new(weights_cache),
            packed_weights: None,
            tokens: Vec::new(),
            device,
            cache: None,
            prefill_compile_cache: Some(CompileCache::new(device, 8)),
            decode_compile_cache: Some(BucketedCompileCache::power_of_two_ladder(
                device,
                1,
                max_past as u64,
            )),
            batched_decode_caches: HashMap::new(),
            batched_ragged_caches: HashMap::new(),
            prefill_profile: CompileProfile::qwen3_prefill(),
            decode_profile: CompileProfile::qwen3_decode(),
            // GPU-resident row-feed decode only where the backend implements
            // feed_kv_row + read_output_row AND it's validated: CUDA (a ~20% decode
            // win — no per-step PCIe K/V re-upload), MLX + Metal (correct; neutral on
            // unified memory, where the host path's upload is already cheap). ROCm /
            // Vulkan / wgpu lack the row-feed hooks → they use the host path.
            use_gpu_kv: device_supports_gpu_kv(device)
                && matches!(
                    device,
                    Device::Cuda | Device::Mlx | Device::Metal | Device::Rocm | Device::Vulkan
                ),
            gpu_kv_binding: GpuKvBinding::default(),
            retention,
            #[cfg(feature = "mmap-kv")]
            kv_store: None,
            #[cfg(feature = "mmap-kv")]
            kv_store_suspended: false,
            #[cfg(feature = "mmap-kv")]
            retrieval_stream: None,
            q_scoring: false,
            retrieval_query_text: None,
            last_retrieved_tokens: Vec::new(),
            last_q_pooled: None,
            inspect: None,
            inspect_step: 0,
        })
    }

    /// Enable selective KV retention with `policy` (Stage 2). The manager scores
    /// resident positions and, after each decode step, evicts/offloads per the
    /// policy so per-step attention stays O(budget) while effective context grows.
    pub fn with_retention(mut self, policy: KvRetentionPolicy) -> Self {
        let kv_dim = self.cfg.kv_proj_dim();
        self.retention = match policy {
            KvRetentionPolicy::Full => None,
            p => Some(KvRetentionManager::new(p, kv_dim)),
        };
        self
    }

    /// Enable the disk-tiered million-token context store as the retention
    /// backend (see [`KvStoreConfig`]): aged-out blocks are offloaded (append-
    /// once, quantized) and each decode step the top-k query-relevant blocks
    /// (+neighbors / fuzzy / memory-decayed) are HNSW-retrieved and spliced into
    /// the resident cache. Sets a matching `Retrieval` policy on the manager
    /// (evict the whole middle each step; the store owns the data + selection).
    #[cfg(feature = "mmap-kv")]
    pub fn enable_kv_store(&mut self, cfg: KvStoreConfig) -> Result<()> {
        let kv_dim = self.cfg.kv_proj_dim();
        let n_layers = self.cfg.num_hidden_layers;
        let store = KvContextStore::new(
            n_layers,
            kv_dim,
            cfg.scheme,
            cfg.capacity_tokens + cfg.block,
            cfg.dir.as_deref(),
            rlx_runtime::hnsw::HnswConfig {
                metric: cfg.metric,
                ..Default::default()
            },
            (cfg.topk * 2).max(64),
            cfg.centroids_per_block,
            cfg.decay,
        )?;
        let mut store = store;
        if cfg.row_keys {
            // Late-interaction HNSW index: navigate on salient rows, not the mean.
            store.set_key_mode(rlx_runtime::kv_context_store::KeyMode::RowKeys);
        }
        // Build the semantic content embedder (dual-encoder path) and enable the
        // store's secondary embedding HNSW. `Encoder` falls back to `TokenMean`
        // unless the `dual-encoder` feature wires a dedicated model.
        let embedder: Option<Box<dyn BlockEmbedder>> = match cfg.embedder {
            EmbedderKind::None => None,
            EmbedderKind::TokenMean | EmbedderKind::Encoder => {
                self.build_block_embedder(cfg.embedder)
            }
        };
        if let Some(e) = embedder.as_ref() {
            // Embeddings are L2-normalized → cosine == dot; a well-conditioned
            // space where HNSW navigates with near-exact recall (unlike raw K).
            store.enable_embeddings(
                e.dim(),
                rlx_runtime::hnsw::HnswConfig {
                    metric: rlx_runtime::hnsw::Metric::Cosine,
                    ..Default::default()
                },
            );
        }
        self.retention = Some(KvRetentionManager::new(
            KvRetentionPolicy::Retrieval {
                block: cfg.block,
                resident_blocks: 0,
                sinks: cfg.sinks,
                recent: cfg.recent,
            },
            kv_dim,
        ));
        self.kv_store = Some(KvStoreState {
            store,
            offloaded: std::collections::HashSet::new(),
            topk: cfg.topk,
            neighbors: cfg.neighbors,
            min_score: cfg.min_score,
            decay_on: cfg.decay < 1.0,
            lexical_weight: cfg.lexical_weight,
            query_window: if cfg.query_window_override > 0 {
                cfg.query_window_override
            } else {
                cfg.recent.max(cfg.block)
            },
            maxsim: cfg.maxsim,
            maxsim_overfetch: cfg.maxsim_overfetch,
            exact: cfg.exact,
            query_pool: cfg.query_pool,
            multilayer: cfg.multilayer,
            block: cfg.block.max(1),
            pending: Vec::new(),
            embedder,
            embed_weight: cfg.embed_weight,
            dense_weight: cfg.dense_weight,
            relevance_gate: cfg.relevance_gate,
            rrf: cfg.rrf,
        });
        self.q_scoring = cfg.query_scoring;
        Ok(())
    }

    /// Build the requested block embedder. `TokenMean` uses the model's static
    /// `model.embed_tokens.weight` (self-contained). `Encoder` uses a dedicated
    /// retrieval model when the `dual-encoder` feature is on, else falls back to
    /// `TokenMean`. Returns `None` only if the embedding table is missing.
    #[cfg(feature = "mmap-kv")]
    fn build_block_embedder(&self, _kind: EmbedderKind) -> Option<Box<dyn BlockEmbedder>> {
        let vocab = self.cfg.vocab_size;
        let hidden = self.cfg.hidden_size;
        let table = self.weights_cache.get("model.embed_tokens.weight")?;
        let t = std::sync::Arc::new(table.0.clone());
        TokenMeanEmbedder::new(t, vocab, hidden).map(|e| Box::new(e) as Box<dyn BlockEmbedder>)
    }

    /// Inject a prebuilt block embedder into the active KV store (overriding any
    /// auto-built one) and enable the store's semantic index at its dim. Used to
    /// wire the dedicated dual-encoder, which needs the generation model's
    /// tokenizer (held by the caller, not the generator). No-op if the store isn't
    /// enabled. Call right after [`enable_kv_store`](Self::enable_kv_store).
    #[cfg(feature = "mmap-kv")]
    pub fn set_kv_store_embedder(&mut self, embedder: Box<dyn BlockEmbedder>) {
        if let Some(st) = self.kv_store.as_mut() {
            st.store.enable_embeddings(
                embedder.dim(),
                rlx_runtime::hnsw::HnswConfig {
                    metric: rlx_runtime::hnsw::Metric::Cosine,
                    ..Default::default()
                },
            );
            st.embedder = Some(embedder);
        }
    }

    /// Set (or clear with `None`) the clean retrieval-query TEXT — the actual
    /// question — used by the KV-store's semantic retrieval, instead of the noisy
    /// decode-position token window. Set it just before generating an answer whose
    /// retrieval should target that question. No effect unless a text-capable
    /// embedder is active.
    pub fn set_retrieval_query(&mut self, text: Option<String>) {
        self.retrieval_query_text = text;
    }

    /// Token ids of the blocks retrieved in the last decode step — for a harness to
    /// check whether retrieval surfaced a fact (retrieval-vs-generation attribution).
    pub fn last_retrieved_tokens(&self) -> &[u32] {
        &self.last_retrieved_tokens
    }
    /// Context-store stats: `(blocks, tokens, disk_bytes)`, or `None` if disabled.
    #[cfg(feature = "mmap-kv")]
    pub fn kv_store_stats(&self) -> Option<(usize, usize, usize)> {
        self.kv_store.as_ref().map(|s| {
            (
                s.store.len_blocks(),
                s.store.total_tokens(),
                s.store.data_bytes(),
            )
        })
    }

    /// Suspend (or resume) the store's offload+splice in the decode retention
    /// step. When suspended, decode runs as plain resident-window attention and no
    /// raw KV is spliced, but the store stays queryable via
    /// [`retrieve_context_spans`](Self::retrieve_context_spans). This is the
    /// substrate for the text-reinjection path (D): retrieve facts as *text*, then
    /// generate over a clean labeled prompt with the splice off so the two signals
    /// don't confound.
    #[cfg(feature = "mmap-kv")]
    pub fn set_kv_store_suspended(&mut self, on: bool) {
        self.kv_store_suspended = on;
    }

    /// Freeze the current token history as the source for recovering retrieved
    /// block text. Call once while `self.tokens` still holds the full original
    /// stream (before any fresh `prefill`); thereafter
    /// [`retrieve_context_spans`](Self::retrieve_context_spans) recovers spans
    /// from this frozen copy, so retrieval keeps working across the interleave
    /// loop's per-hop re-prefills of the reasoning transcript.
    #[cfg(feature = "mmap-kv")]
    pub fn snapshot_retrieval_stream(&mut self) {
        self.retrieval_stream = Some(self.tokens.clone());
    }

    /// Drop the frozen retrieval stream (revert to the live token history).
    #[cfg(feature = "mmap-kv")]
    pub fn clear_retrieval_stream(&mut self) {
        self.retrieval_stream = None;
    }

    /// Retrieve the top-`k` relevant context blocks for the current query and
    /// return each block's **ordered token-id span** (recovered from the token
    /// history) with its relevance score, most-relevant first. Read-only: it does
    /// not offload, splice, or touch the resident cache — so a harness can turn
    /// retrieved KV back into *text* (detokenize the span) and re-inject it as a
    /// clean labeled prompt instead of splicing position-scrambled raw K/V (the
    /// small-LM entity-binding fix). Uses the clean retrieval-query text when set
    /// (see [`set_retrieval_query`](Self::set_retrieval_query)), else the recent
    /// token window. Empty unless a text/embedding block-embedder is active.
    ///
    /// `context_margin` widens each hit's span by that many tokens on both sides
    /// (clamped to the stream) so a fact that straddles the fixed block boundary
    /// isn't truncated in the note (e.g. a 4-digit code split across two blocks).
    ///
    /// Call this BEFORE any fresh `prefill`/`generate`, which clears the token
    /// history the spans are recovered from.
    #[cfg(feature = "mmap-kv")]
    pub fn retrieve_context_spans(
        &self,
        topk_override: Option<usize>,
        context_margin: usize,
    ) -> Vec<(Vec<u32>, f32)> {
        let st = match self.kv_store.as_ref() {
            Some(s) => s,
            None => return Vec::new(),
        };
        let embd = match st.embedder.as_ref() {
            Some(e) => e,
            None => return Vec::new(),
        };
        let topk = topk_override.unwrap_or(st.topk);
        // Query tokens (recent window ≈ the question) for lexical / hybrid blends.
        let query_tokens: Vec<u32> = {
            let n = self.tokens.len();
            self.tokens[n.saturating_sub(st.query_window.max(1))..].to_vec()
        };
        // Query embedding: prefer the clean question text over the noisy window.
        let eq = self
            .retrieval_query_text
            .as_deref()
            .and_then(|t| embd.embed_query_str(t))
            .unwrap_or_else(|| embd.embed_query(&query_tokens));
        // Dense (K·K) side, only when configured and a pooled query was exported.
        let dense: &[f32] = if st.dense_weight > 0.0 {
            self.last_q_pooled.as_deref().unwrap_or(&[])
        } else {
            &[]
        };
        let pure_embed = st.lexical_weight == 0.0 && st.dense_weight == 0.0;
        // Mirror `apply_retention`'s embedder retrieval branch so text-reinjection
        // sees exactly the blocks the splice path would have spliced.
        let mut hits = if st.rrf {
            st.store.retrieve_rrf(
                &eq,
                dense,
                &query_tokens,
                topk,
                60.0,
                st.lexical_weight > 0.0,
                st.dense_weight > 0.0,
            )
        } else if pure_embed {
            st.store.retrieve_embed_exact(&eq, topk)
        } else {
            st.store.retrieve_hybrid3(
                &eq,
                dense,
                &query_tokens,
                topk,
                st.embed_weight,
                st.lexical_weight,
                st.dense_weight,
                st.relevance_gate,
            )
        };
        // hybrid3 applies the gate internally; rrf / pure-embed gate here.
        if st.relevance_gate > 0.0 && (st.rrf || pure_embed) {
            if let Some(top) = hits.first().map(|r| r.score) {
                let floor = st.relevance_gate * top;
                hits.retain(|r| r.score >= floor);
            }
        }
        // Recover span text from the frozen original stream when set (interleave
        // re-prefills clobber the live `self.tokens`), else the live history.
        let toks: &[u32] = self.retrieval_stream.as_deref().unwrap_or(&self.tokens);
        let mut out = Vec::with_capacity(hits.len());
        let n = toks.len();
        for rb in hits {
            // Widen by `context_margin` on both sides so a boundary-straddling fact
            // survives whole in the detokenized note.
            let start = rb.start_pos.saturating_sub(context_margin);
            let end = (rb.start_pos + rb.rows + context_margin).min(n);
            if let Some(s) = toks.get(start..end) {
                out.push((s.to_vec(), rb.score));
            }
        }
        out
    }

    /// Start recording per-step cache/context telemetry (resident / evicted /
    /// retrieved / store) on the retention manager. Read back with
    /// [`take_retention_recorder`](Self::take_retention_recorder). No-op unless a
    /// retention policy is active.
    pub fn enable_retention_recording(&mut self) {
        if let Some(m) = self.retention.as_mut() {
            m.enable_recording();
        }
    }
    /// Take the recorded cache/context telemetry (leaving recording off).
    pub fn take_retention_recorder(
        &mut self,
    ) -> Option<rlx_runtime::kv_metrics::RetentionRecorder> {
        self.retention.as_mut().and_then(|m| m.take_recorder())
    }
    /// Start recording per-step **data** inspection of the KV cache + selection
    /// preferences (shape/stats/histograms/dataflow) into an
    /// [`InspectLog`](rlx_ir::tensor_inspect::InspectLog). Read back with
    /// [`take_inspect_log`](Self::take_inspect_log).
    pub fn enable_inspect(&mut self) {
        if self.inspect.is_none() {
            self.inspect = Some(rlx_ir::tensor_inspect::InspectLog::new());
        }
    }
    /// Take the recorded KV/selection inspection log (leaving recording off).
    pub fn take_inspect_log(&mut self) -> Option<rlx_ir::tensor_inspect::InspectLog> {
        self.inspect.take()
    }

    /// Like [`Self::from_loader`] but loads `qwen3.rlx.toml` from the weights directory when present.
    pub fn from_loader_at(
        cfg: Qwen3Config,
        loader: &mut dyn WeightLoader,
        device: Device,
        weights_path: &Path,
    ) -> Result<Self> {
        let mut g = Self::from_loader(cfg, loader, device)?;
        g.prefill_profile = qwen3_profile_near_weights(weights_path, false);
        g.decode_profile = qwen3_profile_near_weights(weights_path, true);
        Ok(g)
    }

    pub fn with_compile_profiles(
        mut self,
        prefill: CompileProfile,
        decode: CompileProfile,
    ) -> Self {
        self.prefill_profile = prefill;
        self.decode_profile = decode;
        self
    }

    pub fn prefill_profile(&self) -> &CompileProfile {
        &self.prefill_profile
    }

    pub fn decode_profile(&self) -> &CompileProfile {
        &self.decode_profile
    }

    fn profile_compile_options(&self, decode: bool) -> CompileOptions {
        let profile = if decode {
            &self.decode_profile
        } else {
            &self.prefill_profile
        };
        compile_options_from_profile(profile, self.device, KernelDispatchConfig::default())
    }

    /// Enable the prefill compile cache with the given LRU capacity.
    /// Useful when the same prompt length is used across multiple
    /// generation runs — the second + Nth run skip the compile +
    /// param-attach roundtrip (~30-50ms per call on CPU).
    pub fn with_prefill_cache(mut self, capacity: usize) -> Self {
        self.prefill_compile_cache = Some(CompileCache::new(self.device, capacity));
        self
    }

    /// Enable the bucketed decode compile cache spanning past-seq
    /// values in `[1, max_past]`. Buckets are power-of-two
    /// `[1..2, 2..3, 3..5, 5..9, 9..17, …]`. Each bucket compiles
    /// one graph at its upper bound; a steady-state generation loop
    /// across `N` tokens compiles `O(log N)` graphs instead of `N`.
    ///
    /// Padding compute waste is bounded at 2×: actual `past_seq` is
    /// at least half the bucket's upper bound (except possibly the
    /// smallest bucket).
    /// Override the bucketed decode compile cache after construction.
    /// Passing `None` forces the naive (O(N²)) generation path.
    pub fn set_decode_compile_cache(&mut self, cache: Option<BucketedCompileCache>) {
        self.decode_compile_cache = cache;
    }

    pub fn with_decode_cache(mut self, max_past: usize) -> Self {
        let cache = BucketedCompileCache::power_of_two_ladder(
            self.device,
            /*min*/ 1,
            max_past.max(1) as u64,
        );
        self.decode_compile_cache = Some(cache);
        self
    }

    /// Convenience: load weights from a safetensors or GGUF path
    /// (dispatch by extension; see `rlx_core::weight_loader::load_from_path`).
    pub fn from_path(cfg: Qwen3Config, path: &str, device: Device) -> Result<Self> {
        Self::from_path_at(cfg, path, device, Path::new("."))
    }

    /// Like [`Self::from_path`] with an explicit weights path for `qwen3.rlx.toml` discovery.
    pub fn from_path_at(
        cfg: Qwen3Config,
        path: &str,
        device: Device,
        weights_path: &Path,
    ) -> Result<Self> {
        let mut loader = rlx_core::weight_loader::load_from_path(path)?;
        Self::from_loader_at(cfg, loader.as_mut(), device, weights_path)
    }

    /// Same as [`from_path`] but with MTP-head visibility control.
    /// When `include_mtp=true` and the file is GGUF, MTP weights are
    /// drained into the generator's cache alongside the base
    /// weights. The base inference path still ignores them — they
    /// sit in cache for a future MTP-aware decoder. Non-GGUF formats
    /// silently ignore the flag (safetensors files publish all
    /// tensors uniformly; downstream code distinguishes by name).
    pub fn from_path_with_mtp(
        cfg: Qwen3Config,
        path: &str,
        device: Device,
        include_mtp: bool,
    ) -> Result<Self> {
        // Branch on extension so we can flip the GGUF-specific
        // visibility option. Safetensors has no equivalent — it
        // doesn't isolate MTP tensors at the loader level.
        if path.ends_with(".gguf") {
            let mut gguf = rlx_core::weight_loader::GgufLoader::from_file(path)?;
            gguf.include_mtp(include_mtp);
            Self::from_loader(cfg, &mut gguf, device)
        } else {
            Self::from_path(cfg, path, device)
        }
    }

    /// Replace the token history with `prompt_ids`. Does not run the
    /// model — the next [`step`] call processes the full sequence.
    /// Clears any KV cache from a prior generation.
    pub fn prefill(&mut self, prompt_ids: &[u32]) {
        self.tokens.clear();
        self.tokens.extend_from_slice(prompt_ids);
        self.cache = None;
        self.gpu_kv_binding = GpuKvBinding::default();
    }

    /// Run one prefill over the current token history and sample the
    /// next token. The sampled token is appended to the history and
    /// returned. Call repeatedly to generate.
    pub fn step(&mut self, opts: SampleOpts) -> Result<u32> {
        if self.tokens.is_empty() {
            anyhow::bail!("step() called with empty token history; call prefill() first");
        }
        let seq = self.tokens.len();
        let mut wm = WeightMap::from_tensors((*self.weights_cache).clone());
        let (graph, params) = build_qwen3_graph_sized_last_logits(
            &self.cfg, &mut wm, /*batch*/ 1, seq, /*with_kv_outputs*/ false,
        )?;
        let compile_opts = self.profile_compile_options(false);
        let mut compiled = Session::new(self.device).compile_with(graph, &compile_opts);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        let ids_f32: Vec<f32> = self.tokens.iter().map(|&i| i as f32).collect();
        let outputs = compiled.run(&[("input_ids", ids_f32.as_slice())]);
        let logits = outputs
            .into_iter()
            .next()
            .context("compiled.run returned no outputs")?;

        let vocab = self.cfg.vocab_size;
        let expected = vocab;
        if logits.len() < expected {
            anyhow::bail!(
                "logits length {} < expected {} (last logits, seq {seq}, vocab {vocab})",
                logits.len(),
                expected
            );
        }
        // Last-logits graph returns [B=1, 1, vocab].
        let last_row = &logits[..vocab];
        let tok = sample_token(last_row, opts) as u32;
        self.tokens.push(tok);
        Ok(tok)
    }

    /// Run `n` steps and return the newly generated token ids
    /// (excludes the prefill prompt).
    pub fn generate(&mut self, n: usize, opts: SampleOpts) -> Result<Vec<u32>> {
        if self.decode_compile_cache.is_some() {
            return self.generate_cached(n, opts);
        }
        let start = self.tokens.len();
        for _ in 0..n {
            self.step(opts)?;
        }
        Ok(self.tokens[start..].to_vec())
    }

    /// Cached step: O(L) per token instead of O(L²). First call seeds
    /// the KV cache from the prompt via prefill-with-cache; subsequent
    /// calls run the decode-mode graph on just the last token + cached
    /// past. Output is bit-identical to [`step`] modulo reduction
    /// order in the SDPA kernel.
    ///
    /// Invariant after each call: `cache.past_seq == tokens.len() - 1`
    /// (the just-sampled token is appended but not yet in the cache;
    /// it becomes the input for the next decode step).
    pub fn step_cached(&mut self, opts: SampleOpts) -> Result<u32> {
        if self.tokens.is_empty() {
            anyhow::bail!("step_cached() called with empty token history; call prefill() first");
        }
        if self.cache.is_none() {
            // The seed runs prefill, populates the cache, samples from
            // the last position, and appends the token. Return that
            // token directly — no decode step on this call.
            let tok = self.seed_cache_from_prompt(opts)?;
            if std::env::var("RLX_QWEN3_DUMP_DECODE").is_ok() {
                if let Some(c) = self.cache.as_ref() {
                    let mk = c
                        .layers_k
                        .iter()
                        .flatten()
                        .fold(0.0f32, |m, &v| m.max(v.abs()));
                    let mv = c
                        .layers_v
                        .iter()
                        .flatten()
                        .fold(0.0f32, |m, &v| m.max(v.abs()));
                    eprintln!(
                        "[seed-kv-range] max|K|={mk:.1} max|V|={mv:.1} past_len={}",
                        c.past_len
                    );
                }
            }
            // Bound the prompt's cache to the window before the first decode.
            self.rotate_cache_if_sliding();
            return Ok(tok);
        }
        let cache = self.cache.as_ref().unwrap();
        let past_seq = cache.past_len;
        // The token we feed into decode is whatever's after the cached
        // prefix in `self.tokens`. After a prior cached step this is
        // the just-sampled token; after seeding it's the same.
        if self.tokens.len() <= past_seq {
            anyhow::bail!(
                "cache invariant violated: tokens.len() {} <= past_seq {}",
                self.tokens.len(),
                past_seq
            );
        }
        // The token to feed is always the most recent one, at its *absolute*
        // position. With a rotated sliding-window cache `past_seq` (cache
        // length) is smaller than `abs_pos`; for non-windowed models they are
        // equal, so this is behavior-preserving.
        let abs_pos = self.tokens.len() - 1;
        let input_tok = self.tokens[abs_pos];

        // Branch: Q-export one-shot (retrieval Q·K scoring) > bucketed compile
        // cache > plain one-shot. The Q-export path rebuilds the graph per step
        // (slower) but captures the model's query for block relevance scoring.
        let (logits, new_k, new_v) = if self.q_scoring {
            self.decode_step_oneshot_q(past_seq, abs_pos, input_tok)?
        } else if self.decode_compile_cache.is_some()
            && self
                .decode_compile_cache
                .as_ref()
                .unwrap()
                .bucket_for(past_seq as u64)
                .is_some()
        {
            self.decode_step_bucketed(past_seq, abs_pos, input_tok)?
        } else {
            self.decode_step_oneshot(past_seq, abs_pos, input_tok)?
        };

        let (kv_dbg, kv_newrow0) = if std::env::var("RLX_QWEN3_DUMP_DECODE").is_ok() {
            let total: f64 = new_k
                .iter()
                .flat_map(|l| l.iter())
                .map(|v| v.abs() as f64)
                .sum();
            // Max |K| / |V| across all layers — an outlier > 65504 (f16 max)
            // confirms the f16-KV NaN is overflow (needs bf16 KV or scaling).
            let maxk = new_k.iter().flatten().fold(0.0f32, |m, &v| m.max(v.abs()));
            let maxv = new_v.iter().flatten().fold(0.0f32, |m, &v| m.max(v.abs()));
            eprintln!("[kv-range] max|K|={maxk:.1} max|V|={maxv:.1}  (f16 max=65504)");
            // Per-layer magnitude of ONLY the current token's new K row — pinpoints
            // which layer the decode hidden state collapses at on a broken backend.
            let kv_dim = self.cfg.kv_proj_dim();
            let per_layer: Vec<String> = new_k
                .iter()
                .map(|k| {
                    let s = k.len().saturating_sub(kv_dim);
                    format!("{:.0}", k[s..].iter().map(|v| v.abs() as f64).sum::<f64>())
                })
                .collect();
            if std::env::var("RLX_QWEN3_DUMP_LAYERS").is_ok() {
                eprintln!("[decode-newk-per-layer] {}", per_layer.join(","));
            }
            let newrow0 = per_layer.first().cloned().unwrap_or_default();
            (total, newrow0)
        } else {
            (0.0, String::new())
        };
        let cache_mut = self.cache.as_mut().unwrap();
        cache_mut.past_len = past_seq + 1;
        cache_mut.layers_k = new_k;
        cache_mut.layers_v = new_v;
        // Sliding-window: keep only the last `window` cached positions.
        self.rotate_cache_if_sliding();
        // Selective retention (Stage 2): reshape the resident K/V per policy.
        self.apply_retention();

        let vocab = self.cfg.vocab_size;
        if logits.len() != vocab {
            anyhow::bail!("decode logits length {} != vocab {}", logits.len(), vocab);
        }
        if std::env::var("RLX_QWEN3_DUMP_DECODE").is_ok() {
            let (mut mi, mut mv) = (0usize, f32::MIN);
            for (i, &v) in logits.iter().enumerate() {
                if v > mv {
                    mv = v;
                    mi = i;
                }
            }
            let s: f32 = logits.iter().map(|v| v.abs()).sum();
            eprintln!(
                "[decode-dump] past={past_seq} pos={abs_pos} in_tok={input_tok} argmax={mi} \
                 max={mv:.4} sum|l|={s:.2} kv_sum={kv_dbg:.2} kv_newrow0={kv_newrow0} l[358]={:.4} l[21122]={:.4} l[0]={:.4}",
                logits.get(358).copied().unwrap_or(0.0),
                logits.get(21122).copied().unwrap_or(0.0),
                logits.first().copied().unwrap_or(0.0),
            );
        }
        let tok = sample_token(&logits, opts) as u32;
        self.tokens.push(tok);
        Ok(tok)
    }

    /// True once a KV cache has been seeded (after the first decode/prefill
    /// step). Callers use this to decide between a fresh prefill and a
    /// [`feed_continuation`](Self::feed_continuation) into the live cache.
    pub fn has_cache(&self) -> bool {
        self.cache.is_some()
    }

    /// Whether the packed K-quant decode path is active.
    pub fn has_packed_decode(&self) -> bool {
        self.packed_weights.is_some()
    }

    /// Inject a `name → (packed_bytes, scheme)` map (HF weight keys) to enable
    /// the packed decode path directly (bypasses the GGUF re-load).
    pub fn set_packed_weights(
        &mut self,
        map: HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme)>,
    ) {
        self.packed_weights = if map.is_empty() {
            None
        } else {
            Some(Arc::new(map))
        };
    }

    /// Enable packed K-quant decode by loading the linear-projection weights'
    /// packed bytes from `gguf_path` (the loader resolves HF↔GGUF names). Only
    /// `*_proj.weight` linears are packed; embeddings / norms stay F32. Returns
    /// how many linears were packed. Call after constructing the F32 generator
    /// from the same GGUF.
    pub fn enable_packed_decode_from_gguf(&mut self, gguf_path: &str) -> Result<usize> {
        use rlx_core::weight_loader::{GgufLoader, WeightLoader};
        let loader = GgufLoader::from_file(gguf_path)?;
        let mut map: HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme)> = HashMap::new();
        let keys: Vec<String> = self.weights_cache.keys().cloned().collect();
        for key in keys {
            if !key.ends_with("_proj.weight") {
                continue;
            }
            // Only K-quant tensors have packed metadata; F16/F32 return None → skip.
            if let Some((scheme, _shape)) = loader.packed_meta(&key) {
                if let Some(bytes) = loader.tensor_bytes_borrowed(&key) {
                    map.insert(key, (bytes.to_vec(), scheme));
                }
            }
        }
        let n = map.len();
        if n > 0 {
            self.packed_weights = Some(Arc::new(map));
        }
        Ok(n)
    }

    /// Number of tokens currently in the sequence (prompt + generated + fed
    /// continuation) — i.e. the live context length. Used to decide when to
    /// evict old turns before the window fills.
    pub fn context_len(&self) -> usize {
        self.tokens.len()
    }

    /// Feed *known* tokens into the resident KV cache without a full
    /// re-prefill — the multi-turn chat primitive. Each id is folded into the
    /// cache by one cached decode step whose *sampled output is discarded* and
    /// overwritten with the known id: only the decode step's **input** token
    /// mutates cache state, so the cache advances exactly as a prefill of the
    /// extended sequence would, while reusing the already-compiled decode
    /// buckets (no O(seq) prefill-graph recompile per turn). Caller feeds only
    /// the NEW tokens since the last turn (the delta), never the whole history.
    ///
    /// Requires a live cache (a prior `prefill` + at least one decode step, i.e.
    /// [`has_cache`](Self::has_cache) is true). Leaves `new_ids.last()` as the
    /// pending input so the next `step_cached` samples the continuation.
    pub fn feed_continuation(&mut self, new_ids: &[u32]) -> Result<()> {
        if new_ids.is_empty() {
            return Ok(());
        }
        if self.cache.is_none() {
            anyhow::bail!(
                "feed_continuation requires a live KV cache; call prefill() and generate at \
                 least one token first (or use prefill for a fresh sequence)"
            );
        }
        for &id in new_ids {
            // Decode folds the current pending token into the cache and samples
            // an output we don't want; the sample never touches cache state, so
            // overwrite the just-pushed token with the known continuation `id`,
            // which becomes the next pending input. Greedy = cheapest discard.
            self.step_cached(SampleOpts::greedy())?;
            if let Some(last) = self.tokens.last_mut() {
                *last = id;
            }
        }
        Ok(())
    }

    /// Pre-compile the decode-bucket graphs covering past-seq up to `up_to`, so
    /// a conversation that grows into a new length bucket doesn't stall ~seconds
    /// on a first-use compile mid-reply. Each bucket is compiled and has its
    /// params set (but is not run); the cache's resident-bytes cap still LRU-
    /// evicts if the warmed fleet is too large. Returns the number of buckets
    /// newly compiled. No-op without a decode compile cache.
    pub fn warm_decode_buckets(&mut self, up_to: usize) -> usize {
        let decode_opts = self.profile_compile_options(true);
        let cfg = self.cfg.clone();
        let weights = self.weights_cache.clone();
        let Some(cache) = self.decode_compile_cache.as_mut() else {
            return 0;
        };
        // One representative key (the lower bound) per bucket that starts at or
        // below `up_to`. Collected first so the immutable `buckets()` borrow ends
        // before the mutable `ensure_graph_with_params` calls.
        let keys: Vec<u64> = cache
            .buckets()
            .filter(|r| r.start <= up_to as u64)
            .map(|r| r.start.max(1))
            .collect();
        let mut compiled = 0usize;
        for key in keys {
            let before = cache.compiled_count();
            cache.ensure_graph_with_params(
                key,
                |upper| {
                    let mut wm = WeightMap::from_tensors((*weights).clone());
                    build_qwen3_decode_graph_sized_ext(&cfg, &mut wm, 1, upper as usize, true)
                        .expect("qwen3 decode graph (bucket warm)")
                },
                &decode_opts,
            );
            if cache.compiled_count() > before {
                compiled += 1;
            }
        }
        compiled
    }

    /// For sliding-window models, trim the KV cache to its last `window`
    /// positions (a rotating cache). With absolute-position RoPE in the decode
    /// step, this gives windowed attention at O(window) memory instead of
    /// O(sequence). A no-op for non-windowed models.
    fn rotate_cache_if_sliding(&mut self) {
        let w = match (self.cfg.use_sliding_window, self.cfg.sliding_window) {
            (true, Some(w)) if w > 0 => w,
            _ => return,
        };
        let kv_dim = self.cfg.kv_proj_dim();
        if let Some(cache) = self.cache.as_mut() {
            if cache.past_len > w {
                let drop = (cache.past_len - w) * kv_dim;
                for k in cache.layers_k.iter_mut() {
                    k.drain(0..drop.min(k.len()));
                }
                for v in cache.layers_v.iter_mut() {
                    v.drain(0..drop.min(v.len()));
                }
                cache.past_len = w;
            }
        }
    }

    /// Reshape the resident K/V per the retention policy (Stage 2 + 2b). Each
    /// step: register the newest position, plan, offload evicted rows to the
    /// store as blocks, retrieve the top-k query-relevant blocks, and rebuild the
    /// resident K/V = kept + retrieved rows in absolute-position order. RoPE stays
    /// correct because K is stored post-rotation, so Q·K's *relative* rotation is
    /// preserved even when kept/retrieved positions are non-contiguous. Resetting
    /// the GPU binding forces a rebind from the (bounded) trimmed mirror.
    /// #4: mixed-precision KV round-trip on the resident mirror — K→f16, V→int8
    /// (per-tensor symmetric). Applies the precision loss so recall under a mixed
    /// KV cache can be measured; realized GPU traffic savings need the native
    /// int8-V attention kernel (follow-up). Bounded to O(resident) per step.
    /// Gated by `RLX_QWEN3_KV_QUANT`; pair with `RLX_QWEN3_NO_GPU_KV=1` so the
    /// quantized host mirror actually feeds attention.
    fn apply_kv_quant(&mut self) {
        let cache = match self.cache.as_mut() {
            Some(c) => c,
            None => return,
        };
        for lk in cache.layers_k.iter_mut() {
            for x in lk.iter_mut() {
                *x = f16_roundtrip(*x); // K keeps f16 (its ±523 outliers need the range)
            }
        }
        for lv in cache.layers_v.iter_mut() {
            let amax = lv.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-8);
            let scale = amax / 127.0; // per-tensor symmetric int8 (V is well-behaved)
            let inv = 1.0 / scale;
            for x in lv.iter_mut() {
                let q = (*x * inv).round().clamp(-127.0, 127.0);
                *x = q * scale;
            }
        }
    }

    fn apply_retention(&mut self) {
        if self.retention.is_none() || self.cache.is_none() {
            return;
        }
        // Text-reinjection (D) generates over a short clean labeled prompt and
        // wants plain FULL attention over it — no eviction (which would drop the
        // notes), no offload, no splice. Suspending skips retention wholesale
        // while the store stays queryable via `retrieve_context_spans`.
        #[cfg(feature = "mmap-kv")]
        if self.kv_store_suspended {
            return;
        }
        // #4: apply mixed-precision KV to the resident mirror every step (before
        // the plan's early-out) so the new row is included.
        if std::env::var_os("RLX_QWEN3_KV_QUANT").is_some() {
            self.apply_kv_quant();
        }
        let kv_dim = self.cfg.kv_proj_dim();
        let n_layers = self.cfg.num_hidden_layers;

        // Stage 3: feed an importance signal so HeavyHitter/Auto rank resident
        // positions by relevance, not just recency. importance[i] = the newest
        // token's K · resident position i's K, summed over layers — a K-similarity
        // proxy for the model's Q·K attention (the same signal retrieval uses;
        // the exact per-position softmax weights would need a kernel-level export).
        // Only computed for policies that consume it. `observe_attention` runs
        // over the *current* resident (before `append` adds the new token).
        // Kept for inspection recording after commit.
        let mut importance_snapshot: Option<Vec<f32>> = None;
        if self.retention.as_ref().unwrap().needs_attention() || self.inspect.is_some() {
            let resident_len = self.retention.as_ref().unwrap().resident_len();
            if resident_len > 0 {
                let cache = self.cache.as_ref().unwrap();
                let mut importance = vec![0.0f32; resident_len];
                for l in 0..n_layers {
                    let lk = &cache.layers_k[l];
                    let total = lk.len() / kv_dim;
                    if total == 0 {
                        continue;
                    }
                    let q = &lk[(total - 1) * kv_dim..total * kv_dim]; // newest token's K
                    let cap = resident_len.min(total.saturating_sub(1));
                    // Raw dot (cosine was reverted — the re-bench showed it lowered
                    // contrast and regressed recall; ranking is what matters).
                    for (i, imp) in importance.iter_mut().enumerate().take(cap) {
                        let base = i * kv_dim;
                        let mut d = 0.0f32;
                        for x in 0..kv_dim {
                            d += q[x] * lk[base + x];
                        }
                        *imp += d;
                    }
                }
                if self.retention.as_ref().unwrap().needs_attention() {
                    self.retention
                        .as_mut()
                        .unwrap()
                        .observe_attention(&importance);
                }
                importance_snapshot = Some(importance);
            }
        }

        // Query summary for block relevance. Prefer the model's ACTUAL attention
        // query (Q·K) captured by the Q-export decode step — layer-0 query,
        // GQA-pooled to `kv_dim`. Falls back to the newest token's layer-0 K
        // (a K·K self-similarity proxy) when Q export is off. Q·K matches how
        // the model itself weights cached keys, so relevant blocks that K·K
        // misses (e.g. an earlier fact the current question attends to) score high.
        // Query-window pool width (store path only; 1 = newest token's K).
        #[cfg(feature = "mmap-kv")]
        let qpool = self.kv_store.as_ref().map(|s| s.query_pool).unwrap_or(1);
        #[cfg(not(feature = "mmap-kv"))]
        let qpool = 1usize;
        let query_key: Option<Vec<f32>> = self.last_q_pooled.clone().or_else(|| {
            self.cache.as_ref().and_then(|c| {
                c.layers_k.first().and_then(|lk| {
                    let rows = lk.len() / kv_dim;
                    if rows == 0 {
                        return None;
                    }
                    // Mean of the last `qpool` K rows (≈ the current question),
                    // else just the newest token's K.
                    let w = qpool.min(rows).max(1);
                    let start = rows - w;
                    let mut q = vec![0.0f32; kv_dim];
                    for r in start..rows {
                        let base = r * kv_dim;
                        for j in 0..kv_dim {
                            q[j] += lk[base + j];
                        }
                    }
                    if w > 1 {
                        let inv = 1.0 / w as f32;
                        for x in q.iter_mut() {
                            *x *= inv;
                        }
                    }
                    Some(q)
                })
            })
        });

        // All-layer query (newest token's K per layer) for multilayer retrieval —
        // catches facts the model attends to in middle layers, invisible to the
        // layer-0-only `query_key`. Built before the mutable kv_store borrow.
        #[cfg(feature = "mmap-kv")]
        let multilayer = self
            .kv_store
            .as_ref()
            .map(|s| s.multilayer)
            .unwrap_or(false);
        #[cfg(not(feature = "mmap-kv"))]
        let multilayer = false;
        let _query_layers: Vec<Vec<f32>> = if multilayer {
            self.cache
                .as_ref()
                .map(|c| {
                    c.layers_k
                        .iter()
                        .map(|lk| {
                            let rows = lk.len() / kv_dim;
                            if rows == 0 {
                                vec![0.0f32; kv_dim]
                            } else {
                                lk[(rows - 1) * kv_dim..rows * kv_dim].to_vec()
                            }
                        })
                        .collect()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Plan + the abs_pos of every current resident row + the store block size.
        let (plan, positions, block_sz) = {
            let mgr = self.retention.as_mut().unwrap();
            mgr.append();
            let plan = mgr.plan(query_key.as_deref());
            (
                plan,
                mgr.resident_positions(),
                mgr.store_block_size().max(1),
            )
        };
        if plan.is_noop() {
            return;
        }

        let gather = |src: &[f32], rows: &[usize]| -> Vec<f32> {
            let mut out = Vec::with_capacity(rows.len() * kv_dim);
            for &i in rows {
                let s = i * kv_dim;
                if s + kv_dim <= src.len() {
                    out.extend_from_slice(&src[s..s + kv_dim]);
                }
            }
            out
        };

        #[cfg(feature = "mmap-kv")]
        let store_enabled = self.kv_store.is_some();
        #[cfg(not(feature = "mmap-kv"))]
        let store_enabled = false;

        // (1) Offload evicted rows to the store as blocks (retrieval policy only).
        // Extract per-layer K/V into owned blocks first (immutable cache borrow),
        // then push (mutable manager borrow) — kept as separate phases.
        //
        // For the APPEND-ONCE disk store we must dedup by ABSOLUTE POSITION, not by
        // chunk-start: each step re-evicts the retrieved blocks (already offloaded)
        // contiguously with the newly-aged-out token. Index-based chunking bundles
        // that new token into a chunk whose start is already offloaded, so a
        // chunk-start dedup drops it — freezing the store and losing all context
        // past the first eviction. So chunk only the NOT-yet-offloaded positions,
        // grouped by contiguous abs_pos. The in-RAM manager (no dedup, re-stores
        // every step) keeps its original index-based chunking.
        let mut evicted_blocks: Vec<(usize, Vec<Vec<f32>>, Vec<Vec<f32>>)> = Vec::new();
        // Store path per-row staging entries: (abs_pos, token_id, K/layer, V/layer).
        #[cfg(feature = "mmap-kv")]
        let mut new_rows: Vec<(usize, u32, Vec<Vec<f32>>, Vec<Vec<f32>>)> = Vec::new();
        if plan.store_evicted && !plan.evict.is_empty() {
            #[cfg(feature = "mmap-kv")]
            let new_evict: Vec<usize> = if store_enabled {
                let off = &self.kv_store.as_ref().unwrap().offloaded;
                plan.evict
                    .iter()
                    .copied()
                    .filter(|&i| positions.get(i).is_some_and(|p| !off.contains(p)))
                    .collect()
            } else {
                plan.evict.clone()
            };
            #[cfg(not(feature = "mmap-kv"))]
            let new_evict: Vec<usize> = plan.evict.clone();

            let cache = self.cache.as_ref().unwrap();
            if store_enabled {
                // Store path: emit ONE row per new-evicted position (with its token
                // id), in ascending abs order. The store stages these into coherent
                // block-row spans (`flush_pending`) so a multi-token fact stays one
                // retrievable block instead of fragmenting across many 1-row blocks.
                #[cfg(feature = "mmap-kv")]
                for &i in &new_evict {
                    let pos = positions[i];
                    let tok = self.tokens.get(pos).copied().unwrap_or(0);
                    let kr: Vec<Vec<f32>> = (0..n_layers)
                        .map(|l| gather(&cache.layers_k[l], &[i]))
                        .collect();
                    let vr: Vec<Vec<f32>> = (0..n_layers)
                        .map(|l| gather(&cache.layers_v[l], &[i]))
                        .collect();
                    new_rows.push((pos, tok, kr, vr));
                }
            } else {
                for run in consecutive_runs(&new_evict) {
                    for chunk in run.chunks(block_sz) {
                        let start_pos = positions[chunk[0]];
                        let k: Vec<Vec<f32>> = (0..n_layers)
                            .map(|l| gather(&cache.layers_k[l], chunk))
                            .collect();
                        let v: Vec<Vec<f32>> = (0..n_layers)
                            .map(|l| gather(&cache.layers_v[l], chunk))
                            .collect();
                        evicted_blocks.push((start_pos, k, v));
                    }
                }
            }
        }

        // (2) push evicted + take retrieved.
        let mut retrieved: Vec<(usize, Vec<Vec<f32>>, Vec<Vec<f32>>)> = Vec::new();
        if store_enabled {
            // Disk-tiered context store: offload each evicted block ONCE (append-
            // only), then retrieve the top-k relevant blocks via HNSW and splice
            // them. Blocks that came from the store are already offloaded, so
            // re-eviction is a cheap set-hit (no duplicate write).
            #[cfg(feature = "mmap-kv")]
            {
                // Query tokens = the recent-window tokens (≈ the current question),
                // needed by lexical AND the semantic embedder's query encoding.
                let need_tokens = {
                    let s = self.kv_store.as_ref().unwrap();
                    s.lexical_weight > 0.0 || s.embedder.is_some()
                };
                let query_tokens: Vec<u32> = if need_tokens {
                    let win = self.kv_store.as_ref().unwrap().query_window;
                    let n = self.tokens.len();
                    self.tokens[n.saturating_sub(win)..].to_vec()
                } else {
                    Vec::new()
                };
                // Clean query text (the actual question), if the caller set it.
                let query_text = self.retrieval_query_text.clone();

                let st = self.kv_store.as_mut().unwrap();
                // Stage the new-evicted rows (already abs-pos-deduped in phase 1),
                // marking each offloaded, then flush complete coherent block spans.
                let n_new = new_rows.len();
                for (pos, tok, kr, vr) in new_rows.into_iter() {
                    st.offloaded.insert(pos);
                    st.pending.push((pos, tok, kr, vr));
                }
                st.flush_pending(kv_dim, n_layers);
                if std::env::var_os("RLX_QWEN3_RETENTION_DEBUG").is_some() {
                    eprintln!(
                        "[stage] new_rows={} pending={} store_blocks={}",
                        n_new,
                        st.pending.len(),
                        st.store.len_blocks(),
                    );
                }
                if st.embedder.is_some() {
                    // Semantic dual-encoder path (the selective, 1M-scalable signal):
                    // embed the question tokens → hybrid3 blend of embedding +
                    // lexical (+ optional K·K). This is what makes small-topk recall
                    // land the right block where K·K cannot.
                    // Prefer the clean question text over the noisy token window.
                    let embd = st.embedder.as_ref().unwrap();
                    let eq = query_text
                        .as_deref()
                        .and_then(|t| embd.embed_query_str(t))
                        .unwrap_or_else(|| embd.embed_query(&query_tokens));
                    let pure_embed = st.lexical_weight == 0.0 && st.dense_weight == 0.0;
                    let hits = if st.rrf {
                        // Reciprocal Rank Fusion of embedding + lexical(+K·K) — the
                        // scale-robust way to combine signals; recovers exact-token
                        // needles the encoder alone misses.
                        let dense: &[f32] = if st.dense_weight > 0.0 {
                            query_key.as_deref().unwrap_or(&[])
                        } else {
                            &[]
                        };
                        let mut h = st.store.retrieve_rrf(
                            &eq,
                            dense,
                            &query_tokens,
                            st.topk,
                            60.0,
                            st.lexical_weight > 0.0,
                            st.dense_weight > 0.0,
                        );
                        if st.relevance_gate > 0.0 {
                            if let Some(top) = h.first().map(|r| r.score) {
                                let floor = st.relevance_gate * top;
                                h.retain(|r| r.score >= floor);
                            }
                        }
                        h
                    } else if pure_embed {
                        // Pure-embed: EXACT brute-force cosine (HNSW nav loses recall
                        // past a few hundred blocks; exact is correct AND cheap at
                        // ≤~1M-token block granularity). Relevance-gate applied below.
                        let mut h = st.store.retrieve_embed_exact(&eq, st.topk);
                        if st.relevance_gate > 0.0 {
                            if let Some(top) = h.first().map(|r| r.score) {
                                let floor = st.relevance_gate * top;
                                h.retain(|r| r.score >= floor);
                            }
                        }
                        h
                    } else {
                        let dense: &[f32] = if st.dense_weight > 0.0 {
                            query_key.as_deref().unwrap_or(&[])
                        } else {
                            &[]
                        };
                        st.store.retrieve_hybrid3(
                            &eq,
                            dense,
                            &query_tokens,
                            st.topk,
                            st.embed_weight,
                            st.lexical_weight,
                            st.dense_weight,
                            st.relevance_gate,
                        )
                    };
                    for rb in hits {
                        retrieved.push((rb.start_pos, rb.k, rb.v));
                    }
                } else if st.multilayer && !query_layers.is_empty() {
                    // All-layer exact MaxSim: strongest K-space signal (exact + max
                    // row + every layer's contribution → middle-layer facts count).
                    for rb in st.store.retrieve_exact_multilayer(&query_layers, st.topk) {
                        retrieved.push((rb.start_pos, rb.k, rb.v));
                    }
                } else if let Some(q) = query_key.as_deref() {
                    let hits = if st.exact {
                        // Brute-force exact MaxSim over all blocks (bypass HNSW —
                        // its greedy nav misses the true-nearest on clustered K keys).
                        st.store.retrieve_exact(q, st.topk)
                    } else if st.maxsim {
                        // Late-interaction re-rank: over-fetch candidates, score each
                        // by max over its rows of q·K_row (exact, no mean dilution).
                        st.store.retrieve_maxsim(q, st.topk, st.maxsim_overfetch)
                    } else if st.lexical_weight > 0.0 {
                        st.store
                            .retrieve_hybrid(q, &query_tokens, st.topk, st.lexical_weight)
                    } else if st.decay_on {
                        st.store.retrieve_decayed(q, st.topk)
                    } else if st.min_score.is_finite() {
                        st.store.retrieve_fuzzy(q, st.topk, st.min_score)
                    } else if st.neighbors > 0 {
                        st.store.retrieve_expanded(q, st.topk, st.neighbors)
                    } else {
                        st.store.retrieve(q, st.topk)
                    };
                    for rb in hits {
                        retrieved.push((rb.start_pos, rb.k, rb.v));
                    }
                }
            }
        } else {
            let mgr = self.retention.as_mut().unwrap();
            for (start, k, v) in evicted_blocks {
                mgr.push_evicted_block(start, k, v);
            }
            for &bid in &plan.retrieve {
                if let Some(blk) = mgr.take_block(bid) {
                    retrieved.push((blk.start_pos, blk.k, blk.v));
                }
            }
        }

        // Instrumentation: record the token ids of every retrieved block (from the
        // original token stream) so a harness can tell whether retrieval actually
        // surfaced a needle's fact — separating a RETRIEVAL miss from a GENERATION
        // miss. Overwritten each step; with a fixed clean-query it's stable.
        {
            let mut rt: Vec<u32> = Vec::new();
            for (start, k, _) in &retrieved {
                let rows = k.first().map(|l| l.len() / kv_dim).unwrap_or(0);
                if let Some(s) = self.tokens.get(*start..(*start + rows)) {
                    rt.extend_from_slice(s);
                }
            }
            self.last_retrieved_tokens = rt;
        }

        // (3) rebuild resident K/V = kept + retrieved rows, sorted by abs_pos.
        // `entries[j] = (abs_pos, source)`: a kept resident row or a retrieved
        // block row. Deduped by position (matches `commit`'s dedup).
        enum Src {
            Kept(usize),
            Retrieved(usize, usize), // (block index, row)
        }
        let mut entries: Vec<(usize, Src)> = Vec::with_capacity(plan.keep.len());
        for &i in &plan.keep {
            entries.push((positions[i], Src::Kept(i)));
        }
        for (bi, (start, k, _)) in retrieved.iter().enumerate() {
            let rows = k.first().map(|l| l.len() / kv_dim).unwrap_or(0);
            for r in 0..rows {
                entries.push((start + r, Src::Retrieved(bi, r)));
            }
        }
        entries.sort_by_key(|(p, _)| *p);
        entries.dedup_by_key(|(p, _)| *p);

        {
            let cache = self.cache.as_mut().unwrap();
            for l in 0..n_layers {
                let old_k = std::mem::take(&mut cache.layers_k[l]);
                let old_v = std::mem::take(&mut cache.layers_v[l]);
                let mut nk = Vec::with_capacity(entries.len() * kv_dim);
                let mut nv = Vec::with_capacity(entries.len() * kv_dim);
                for (_, src) in &entries {
                    match *src {
                        Src::Kept(i) => {
                            let s = i * kv_dim;
                            if s + kv_dim <= old_k.len() {
                                nk.extend_from_slice(&old_k[s..s + kv_dim]);
                                nv.extend_from_slice(&old_v[s..s + kv_dim]);
                            }
                        }
                        Src::Retrieved(bi, r) => {
                            let s = r * kv_dim;
                            let rk = &retrieved[bi].1[l];
                            let rv = &retrieved[bi].2[l];
                            if s + kv_dim <= rk.len() {
                                nk.extend_from_slice(&rk[s..s + kv_dim]);
                                nv.extend_from_slice(&rv[s..s + kv_dim]);
                            }
                        }
                    }
                }
                cache.layers_k[l] = nk;
                cache.layers_v[l] = nv;
            }
            cache.past_len = entries.len();
        }

        // (4) commit — hand the manager the retrieved positions so its resident
        // metadata matches the rebuilt K/V order (both sort+dedup by abs_pos).
        let retrieved_positions: Vec<usize> = retrieved
            .iter()
            .flat_map(|(start, k, _)| {
                let rows = k.first().map(|l| l.len() / kv_dim).unwrap_or(0);
                (0..rows).map(move |r| start + r)
            })
            .collect();
        self.retention
            .as_mut()
            .unwrap()
            .commit(&plan, &retrieved_positions);
        if std::env::var_os("RLX_QWEN3_RETENTION_DEBUG").is_some() {
            let mgr = self.retention.as_ref().unwrap();
            let (sb, st_tok) = {
                #[cfg(feature = "mmap-kv")]
                {
                    self.kv_store
                        .as_ref()
                        .map(|s| (s.store.len_blocks(), s.store.total_tokens()))
                        .unwrap_or((mgr.stored_blocks(), mgr.stored_tokens()))
                }
                #[cfg(not(feature = "mmap-kv"))]
                {
                    (mgr.stored_blocks(), mgr.stored_tokens())
                }
            };
            eprintln!(
                "[retention] resident={} evicted={} retrieved={} store_blocks={} store_tokens={}",
                mgr.resident_len(),
                plan.evict.len(),
                retrieved_positions.len(),
                sb,
                st_tok,
            );
        }
        // Record KV cache + selection-preference data for joint inspection
        // (shape / stats / histogram / dataflow), so the whole picture — what the
        // cache holds and why positions are kept — is analyzable together.
        if let Some(log) = self.inspect.as_mut() {
            let step = self.inspect_step;
            let (sel, concentration) = self.retention.as_ref().unwrap().selection_snapshot();
            let (k0, v0, rows) = {
                let c = self.cache.as_ref().unwrap();
                let k0 = c.layers_k.first().cloned().unwrap_or_default();
                let v0 = c.layers_v.first().cloned().unwrap_or_default();
                let rows = k0.len().checked_div(kv_dim).unwrap_or(0);
                (k0, v0, rows)
            };
            let attn_mass: Vec<f32> = sel.iter().map(|(_, m, _)| *m).collect();
            log.record_tensor(step, "kv.k.l0", &[rows, kv_dim], &k0, 24);
            log.record_tensor(step, "kv.v.l0", &[rows, kv_dim], &v0, 24);
            log.record_tensor(step, "selection.concentration", &[1], &[concentration], 1);
            if !attn_mass.is_empty() {
                log.record_tensor(
                    step,
                    "selection.attn_mass",
                    &[attn_mass.len()],
                    &attn_mass,
                    24,
                );
            }
            if let Some(imp) = &importance_snapshot {
                log.record_tensor(step, "selection.importance", &[imp.len()], imp, 24);
            }
            // Selection dataflow (deduped): K → importance → attn_mass → resident.
            log.edge("kv.k.l0", "selection.importance");
            log.edge("selection.importance", "selection.attn_mass");
            log.edge("selection.attn_mass", "resident");
            self.inspect_step += 1;
        }

        // Resident set changed → rebind on the next step (bounded to O(budget)).
        self.gpu_kv_binding = GpuKvBinding::default();
    }

    /// Decode path that compiles a fresh graph for the exact `past_seq`
    /// every call. Slower but always-correct fallback.
    fn decode_step_oneshot(
        &mut self,
        past_seq: usize,
        abs_pos: usize,
        input_tok: u32,
    ) -> Result<DecodeLogitsKv> {
        let cache = self.cache.as_ref().unwrap();

        let mut wm = WeightMap::from_tensors((*self.weights_cache).clone());
        let (graph, params) =
            build_qwen3_decode_graph_sized(&self.cfg, &mut wm, /*batch*/ 1, past_seq)?;
        let opts = self.profile_compile_options(true);
        let mut compiled = Session::new(self.device).compile_with(graph, &opts);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }

        let (cos, sin) = compute_rope_slice(&self.cfg, abs_pos);
        let input_ids_f32 = [input_tok as f32];
        let key_strs: Vec<String> = (0..self.cfg.num_hidden_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> =
            Vec::with_capacity(3 + 2 * self.cfg.num_hidden_layers);
        inputs.push(("input_ids", input_ids_f32.as_slice()));
        inputs.push(("rope_cos", cos.as_slice()));
        inputs.push(("rope_sin", sin.as_slice()));
        for i in 0..self.cfg.num_hidden_layers {
            inputs.push((&key_strs[2 * i], cache.layers_k[i].as_slice()));
            inputs.push((&key_strs[2 * i + 1], cache.layers_v[i].as_slice()));
        }

        let outputs = compiled.run(&inputs);
        split_decode_logits_kv(outputs, self.cfg.num_hidden_layers)
    }

    /// One-shot decode that ALSO exports the model's post-RoPE query so the
    /// KV-store can score cached blocks by the model's actual attention (Q·K)
    /// instead of key-self-similarity (K·K). Same result as
    /// [`decode_step_oneshot`](Self::decode_step_oneshot) for logits+K+V; the
    /// side effect is populating `self.last_q_pooled` with the newest token's
    /// layer-0 query, GQA-pooled to `kv_dim` (block keys live in kv-head layout).
    ///
    /// The Q-export graph appends `q_l, k_l` per layer after logits+K+V (see
    /// [`build_qwen3_decode_graph_sized_qk`]), so aux has `2 * L` tensors and
    /// `aux[0]` is layer-0's query.
    fn decode_step_oneshot_q(
        &mut self,
        past_seq: usize,
        abs_pos: usize,
        input_tok: u32,
    ) -> Result<DecodeLogitsKv> {
        let cache = self.cache.as_ref().unwrap();
        let n_layers = self.cfg.num_hidden_layers;

        let mut wm = WeightMap::from_tensors((*self.weights_cache).clone());
        let (graph, params) =
            build_qwen3_decode_graph_sized_qk(&self.cfg, &mut wm, /*batch*/ 1, past_seq)?;
        let opts = self.profile_compile_options(true);
        let mut compiled = Session::new(self.device).compile_with(graph, &opts);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }

        let (cos, sin) = compute_rope_slice(&self.cfg, abs_pos);
        let input_ids_f32 = [input_tok as f32];
        let key_strs: Vec<String> = (0..n_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> = Vec::with_capacity(3 + 2 * n_layers);
        inputs.push(("input_ids", input_ids_f32.as_slice()));
        inputs.push(("rope_cos", cos.as_slice()));
        inputs.push(("rope_sin", sin.as_slice()));
        for i in 0..n_layers {
            inputs.push((&key_strs[2 * i], cache.layers_k[i].as_slice()));
            inputs.push((&key_strs[2 * i + 1], cache.layers_v[i].as_slice()));
        }

        let outputs = compiled.run(&inputs);
        // aux = [q_0, k_0, q_1, k_1, …]; take layer-0's query and GQA-pool it.
        let (logits, layers_k, layers_v, aux) =
            split_decode_logits_kv_aux(outputs, n_layers, 2 * n_layers)?;
        if let Some(q0) = aux.first() {
            self.last_q_pooled = Some(gqa_pool_query(
                q0,
                self.cfg.head_dim,
                self.cfg.num_key_value_heads,
                self.cfg.kv_group_size(),
            ));
        }
        Ok((logits, layers_k, layers_v))
    }

    fn decode_step_bucketed(
        &mut self,
        past_seq: usize,
        abs_pos: usize,
        input_tok: u32,
    ) -> Result<DecodeLogitsKv> {
        let kv_dim = self.cfg.kv_proj_dim();
        let n_layers = self.cfg.num_hidden_layers;
        // RoPE uses the token's *absolute* position; with a rotated
        // (sliding-window) cache this differs from the cache length `past_seq`.
        let (cos, sin) = compute_rope_slice(&self.cfg, abs_pos);
        let input_ids_f32 = [input_tok as f32];
        let decode_opts = self.profile_compile_options(true);
        let upper = self
            .decode_compile_cache
            .as_ref()
            .and_then(|cache_dec| {
                cache_dec.bucket_for(past_seq as u64).map(|idx| {
                    cache_dec
                        .buckets()
                        .nth(idx)
                        .map(|r| (r.end - 1) as usize)
                        .unwrap_or(past_seq)
                })
            })
            .unwrap_or(past_seq);
        // Standard full causal decode mask: every cached key is valid. For
        // sliding-window models the cache is *rotated* to hold only the last
        // `window` keys (see `rotate_cache_if_sliding`), so the windowing is
        // already in the cache and the plain mask suffices.
        let mask = bucket_decode_mask(past_seq, upper);
        // GPU-resident row-feed path (async-MLX decode fix): keep K/V on device,
        // fold only the new row in place each step — no host round-trip, no handle
        // growth. The host path below re-uploads the whole padded cache each token.
        if self.use_gpu_kv && !rlx_ir::env::flag("RLX_QWEN3_NO_GPU_KV") {
            return self.decode_step_gpu_resident(
                past_seq,
                upper,
                kv_dim,
                n_layers,
                &input_ids_f32,
                &cos,
                &sin,
                &mask,
            );
        }
        // Host path ONLY: it re-uploads the whole padded cache as graph inputs, so
        // it needs an owned snapshot to avoid aliasing the &mut decode_compile_cache
        // borrow below. The gpu-resident path above returns first and never touches
        // `kv`, so cloning the (O(context)) cache here — instead of at the top —
        // keeps the common decode step from copying the whole KV mirror every token.
        let kv = self.cache.as_ref().unwrap().clone();
        let fixed = [
            CacheRunInput {
                name: "input_ids",
                data: &input_ids_f32,
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_cos",
                data: &cos,
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_sin",
                data: &sin,
                row_inner: None,
            },
            CacheRunInput {
                name: "mask",
                data: &mask,
                row_inner: None,
            },
        ];
        let cfg = self.cfg.clone();
        let weights = self.weights_cache.clone();
        let packed = self.packed_weights.clone();
        let cache_dec = self.decode_compile_cache.as_mut().unwrap();
        // Packed decode: build the F32 decode graph, then rewrite its weight
        // MatMuls → packed DequantMatMul and bind the U8 K-quant bytes (only the
        // compile miss pays the rewrite; steady-state captures the `Arc`s). The
        // graph structure — RoPE / QK-norm / GQA / KV-cache / SwiGLU — is
        // unchanged, so numerics match the F32 path (DequantMatMul ≡ dequant+matmul).
        if let Some(pw) = packed {
            return run_bucketed_kv_decode_packed(
                cache_dec,
                past_seq,
                &kv,
                kv_dim,
                n_layers,
                &fixed,
                |upper| {
                    let mut wm = WeightMap::from_tensors((*weights).clone());
                    let (mut graph, mut params) =
                        build_qwen3_decode_graph_sized_ext(&cfg, &mut wm, 1, upper as usize, true)
                            .expect("qwen3 packed decode graph");
                    let keys =
                        crate::packed_decode::rewrite_matmuls_to_packed(&mut graph, &|name| {
                            pw.get(name).map(|(bytes, scheme)| {
                                crate::packed_decode::PackedWeightInfo {
                                    scheme: *scheme,
                                    nbytes: bytes.len(),
                                    n: 0,
                                    n_groups: 0,
                                }
                            })
                        });
                    // Drop the rewritten f32 weights; bind their U8 packed bytes.
                    let mut packed_params = Vec::with_capacity(keys.len());
                    for k in &keys {
                        params.remove(k);
                        if let Some((bytes, _)) = pw.get(k) {
                            packed_params.push((k.clone(), bytes.clone()));
                        }
                    }
                    (graph, params, packed_params)
                },
                &decode_opts,
            );
        }
        run_bucketed_kv_decode(
            cache_dec,
            past_seq,
            &kv,
            kv_dim,
            n_layers,
            &fixed,
            |upper| {
                // Deep-copy weights only when this bucket actually compiles
                // (cache miss); steady-state decode captures just the `Arc`.
                let mut wm = WeightMap::from_tensors((*weights).clone());
                build_qwen3_decode_graph_sized_ext(&cfg, &mut wm, 1, upper as usize, true)
                    .expect("qwen3 bucketed decode graph")
            },
            &decode_opts,
        )
    }

    /// GPU-resident bucketed decode step. Binds each layer's K/V to on-device
    /// handles once per bucket and, after each run, folds ONLY the new token's row
    /// into the resident handle in place (`feed_kv_row`) — the handle never grows,
    /// only logits come back to host. This is the async-MLX fix: the host path
    /// (`run_bucketed_kv_decode`) re-uploads the whole padded cache as inputs and
    /// reads new K/V back every token, forcing a full MLX materialize + sync per
    /// step (~5× slower). Mirrors rlx-gemma's `decode_step_gpu_resident`. Returns
    /// logits + the advanced host K/V mirror (kept current for bucket-change rebind).
    #[allow(clippy::too_many_arguments)]
    fn decode_step_gpu_resident(
        &mut self,
        past_seq: usize,
        upper: usize,
        kv_dim: usize,
        n_layers: usize,
        input_ids_f32: &[f32],
        cos: &[f32],
        sin: &[f32],
        mask: &[f32],
    ) -> Result<DecodeLogitsKv> {
        let key = past_seq as u64;
        let decode_opts = self.profile_compile_options(true);
        let cfg = self.cfg.clone();
        let weights = self.weights_cache.clone();

        // Compile + set params for this bucket (only on a cache miss; hits keep the
        // weights resident on device).
        {
            let cache_dec = self.decode_compile_cache.as_mut().unwrap();
            cache_dec
                .ensure_graph_with_params(
                    key,
                    |upper| {
                        let mut wm = WeightMap::from_tensors((*weights).clone());
                        build_qwen3_decode_graph_sized_ext(&cfg, &mut wm, 1, upper as usize, true)
                            .expect("qwen3 decode graph")
                    },
                    &decode_opts,
                )
                .context("decode bucket outside compile-cache range")?;
        }

        // (Re)bind resident K/V handles on first use of this bucket or after a
        // bucket change. `register_kv_row_feed` (not `set_gpu_handle_feed`) is what
        // keeps the handle from growing — the whole-output feed is the bug that
        // grew `past_k` by 1 each step and broke the attention reshape.
        let handles_live = self
            .decode_compile_cache
            .as_mut()
            .and_then(|c| c.compiled_for_key_mut(key))
            .map(|cg| cg.has_gpu_handle("past_k_0"))
            .unwrap_or(false);
        // Rebind on: fresh generation (binding reset by prefill → upper==0),
        // bucket change, or handles not present. Reusing a stale handle (e.g. a
        // prior prompt's) is a correctness bug, so bind whenever `upper` differs.
        if self.gpu_kv_binding.upper != upper as u64 || !handles_live {
            let bufs: Vec<(String, Vec<f32>, usize)> = {
                let kv = self.cache.as_ref().context("decode cache missing")?;
                (0..n_layers)
                    .flat_map(|i| {
                        let kp = pad_rows(&kv.layers_k[i], kv_dim, upper as u64);
                        let vp = pad_rows(&kv.layers_v[i], kv_dim, upper as u64);
                        [
                            (format!("past_k_{i}"), kp, 1 + 2 * i),
                            (format!("past_v_{i}"), vp, 2 + 2 * i),
                        ]
                    })
                    .collect()
            };
            let compiled = self
                .decode_compile_cache
                .as_mut()
                .unwrap()
                .compiled_for_key_mut(key)
                .context("bucket missing after compile")?;
            for (name, buf, out_idx) in &bufs {
                compiled.bind_gpu_handle(name, buf);
                compiled.register_kv_row_feed(name, *out_idx);
            }
            self.gpu_kv_binding.upper = upper as u64;
        }

        // Run (reading back ONLY logits), then fold the new K/V row into the
        // resident handles in place on device (`feed_kv_row`) — no K/V leaves the
        // device this step.
        // Skip active-extent on the unified-memory Apple backends: it is a NO-OP for
        // this full-length (upper+1) decode, and on MLX setting it forces the
        // executable off the Compiled (lower-once + replay) path onto Lazy, which
        // re-lowers the whole ~1500-node decode graph every step (~68ms/token, the
        // dominant cost). Discrete-GPU backends (CUDA/ROCm) keep it — validated.
        let skip_active_extent = matches!(
            self.device,
            Device::Metal | Device::Mlx | Device::Vulkan | Device::Gpu
        );
        let (logits, new_rows) = {
            let compiled = self
                .decode_compile_cache
                .as_mut()
                .unwrap()
                .compiled_for_key_mut(key)
                .context("bucket missing")?;
            let run_inputs: Vec<(&str, &[f32])> = vec![
                ("input_ids", input_ids_f32),
                ("rope_cos", cos),
                ("rope_sin", sin),
                ("mask", mask),
            ];
            if !skip_active_extent {
                compiled.set_active_extent(Some((upper + 1, upper + 1)));
            }
            let mut outs = compiled.run_read_outputs(&run_inputs, Some(&[0]));
            if !skip_active_extent {
                compiled.set_active_extent(None);
            }
            compiled.feed_kv_row(upper, past_seq, kv_dim);
            // Read the new token's K/V row (output row `upper`) per layer to advance
            // the host mirror. Exact even at a bucket boundary where the device feed's
            // dst row (`past_seq == upper`) is out of range for the [upper]-row handle
            // — the next step is a bucket change that rebinds from this host mirror.
            let mut new_rows: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(n_layers);
            for i in 0..n_layers {
                let nk = compiled
                    .read_output_row(1 + 2 * i, upper, kv_dim)
                    .context("resident decode K row")?;
                let nv = compiled
                    .read_output_row(2 + 2 * i, upper, kv_dim)
                    .context("resident decode V row")?;
                new_rows.push((nk, nv));
            }
            let logits = outs
                .drain(..)
                .next()
                .context("resident decode logits missing")?;
            (logits, new_rows)
        };

        // Move the host K/V mirror OUT of the cache, append this token's new rows,
        // and hand it back for `step_cached` to store — instead of extending it in
        // place and then cloning the whole (O(context)) mirror to return. The clone
        // grew with context and ran on every token, starving the GPU between steps
        // (visible as <100% util + a multi-turn "stumble"); the move is O(1).
        let (layers_k, layers_v) = {
            let cache = self.cache.as_mut().context("decode cache missing")?;
            let mut layers_k = std::mem::take(&mut cache.layers_k);
            let mut layers_v = std::mem::take(&mut cache.layers_v);
            for (i, (nk, nv)) in new_rows.into_iter().enumerate() {
                layers_k[i].extend_from_slice(&nk);
                layers_v[i].extend_from_slice(&nv);
            }
            cache.past_len = past_seq + 1;
            (layers_k, layers_v)
        };
        let next_upper = self
            .decode_compile_cache
            .as_ref()
            .and_then(|c| {
                c.bucket_for((past_seq + 1) as u64)
                    .and_then(|idx| c.buckets().nth(idx).map(|r| (r.end - 1) as usize))
            })
            .unwrap_or(upper);
        if next_upper != upper {
            self.gpu_kv_binding = GpuKvBinding::default();
        }
        // `cache.layers_k/v` are momentarily empty (moved out above); `step_cached`
        // stores `layers_k/v` straight back, so nothing reads them in between.
        Ok((logits, layers_k, layers_v))
    }

    /// **Fused batched decode** — the throughput primitive for serving.
    ///
    /// Decodes `B = entries.len()` sequences in ONE forward, so every weight
    /// matmul (QKV/O projections, MLP, LM head) runs once over the whole batch
    /// instead of `B` times. Each `entries[i] = (token, kv)` gives sequence
    /// `i`'s next input token and its cache; **all caches must share
    /// `past_seq`** (the same `past_len`) and the same absolute position
    /// `abs_pos`, because the decode graph's RoPE slice and causal-mask bucket
    /// are per-step values shared across the batch. The continuous batcher
    /// groups decode work by cache length to satisfy this.
    ///
    /// Returns, per input sequence, `(next_token_logits[vocab], advanced_kv)`
    /// where `advanced_kv.past_len == past_seq + 1`.
    ///
    /// KV layout note: the decode graph reads/writes K/V **batch-major**
    /// `[batch, seq_pos, kv_dim]` (each sequence's positions contiguous). We
    /// pad each sequence to the compiled bucket `upper` and assemble that
    /// layout directly, then run the cached graph and de-interleave the
    /// `[batch, upper+1, kv_dim]` output (real positions `0..past_seq` plus the
    /// new token at row `upper`). The bucketed primitive's host-side
    /// pad/compact helpers are position-major and only coincide with this at
    /// batch=1, so we drive the compiled graph directly. Logits are batch-major
    /// `[batch, vocab]`.
    pub fn decode_batched_uniform(
        &mut self,
        entries: &[(u32, &KvCacheState)],
        abs_pos: usize,
        past_seq: usize,
    ) -> Result<Vec<(Vec<f32>, KvCacheState)>> {
        let b = entries.len();
        if b == 0 {
            return Ok(Vec::new());
        }
        let kv_dim = self.cfg.kv_proj_dim();
        let n_layers = self.cfg.num_hidden_layers;
        let vocab = self.cfg.vocab_size;
        let max_past = self.cfg.max_position_embeddings.clamp(1, 4096) as u64;
        let device = self.device;
        // Hoist every `self` read before the mutable cache borrow below.
        let decode_opts = self.profile_compile_options(true);
        let cfg = self.cfg.clone();
        let weights = self.weights_cache.clone();

        let input_ids_f32: Vec<f32> = entries.iter().map(|(t, _)| *t as f32).collect();
        let (cos, sin) = compute_rope_slice(&cfg, abs_pos);

        // One bucketed compile cache per batch size (graph shape depends on B).
        // Fetch the compiled graph for this past_seq bucket and its `upper`.
        let cache_b = self
            .batched_decode_caches
            .entry(b)
            .or_insert_with(|| BucketedCompileCache::power_of_two_ladder(device, 1, max_past));
        let (upper_u64, compiled) = cache_b
            .ensure_graph_with_params(
                past_seq as u64,
                |upper| {
                    let mut wm = WeightMap::from_tensors((*weights).clone());
                    build_qwen3_decode_graph_sized_ext(&cfg, &mut wm, b, upper as usize, true)
                        .expect("qwen3 batched decode graph")
                },
                &decode_opts,
            )
            .ok_or_else(|| anyhow::anyhow!("past_seq {past_seq} outside decode buckets"))?;
        let upper = upper_u64 as usize;

        // Batch-major padded KV `[batch, upper, kv_dim]`: each sequence's
        // `past_seq` real positions followed by `upper - past_seq` zero rows.
        let real = past_seq * kv_dim;
        let mut padded_k: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        let mut padded_v: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let mut pk = vec![0.0f32; b * upper * kv_dim];
            let mut pv = vec![0.0f32; b * upper * kv_dim];
            for (i, (_tok, kv)) in entries.iter().enumerate() {
                debug_assert_eq!(
                    kv.past_len, past_seq,
                    "decode_batched_uniform: ragged past_len"
                );
                let base = i * upper * kv_dim;
                pk[base..base + real].copy_from_slice(&kv.layers_k[l][..real]);
                pv[base..base + real].copy_from_slice(&kv.layers_v[l][..real]);
            }
            padded_k.push(pk);
            padded_v.push(pv);
        }

        // Per-row mask `[batch, upper + 1]`; rows are identical at uniform past_seq.
        let row_mask = bucket_decode_mask(past_seq, upper);
        let mut mask = Vec::with_capacity(b * row_mask.len());
        for _ in 0..b {
            mask.extend_from_slice(&row_mask);
        }

        let key_strs: Vec<String> = (0..n_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> = Vec::with_capacity(4 + 2 * n_layers);
        inputs.push(("input_ids", &input_ids_f32));
        inputs.push(("rope_cos", &cos));
        inputs.push(("rope_sin", &sin));
        inputs.push(("mask", &mask));
        for l in 0..n_layers {
            inputs.push((&key_strs[2 * l], &padded_k[l]));
            inputs.push((&key_strs[2 * l + 1], &padded_v[l]));
        }

        let raw = compiled.run(&inputs);
        let (logits, new_k, new_v) = split_decode_logits_kv(raw, n_layers)?;

        // De-interleave batch-major: logits `[batch, vocab]`; per layer the KV
        // output is `[batch, upper+1, kv_dim]` — take positions `0..past_seq`
        // and the new token at row `upper`.
        let new_len = past_seq + 1;
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            let lo = logits[i * vocab..(i + 1) * vocab].to_vec();
            let mut kv = KvCacheState {
                past_len: new_len,
                layers_k: Vec::with_capacity(n_layers),
                layers_v: Vec::with_capacity(n_layers),
                layers_kv_base: vec![0; n_layers],
            };
            for l in 0..n_layers {
                let stride = (upper + 1) * kv_dim;
                let base = i * stride;
                let nt = base + upper * kv_dim;
                let mut sk = vec![0.0f32; new_len * kv_dim];
                let mut sv = vec![0.0f32; new_len * kv_dim];
                sk[..real].copy_from_slice(&new_k[l][base..base + real]);
                sv[..real].copy_from_slice(&new_v[l][base..base + real]);
                sk[real..real + kv_dim].copy_from_slice(&new_k[l][nt..nt + kv_dim]);
                sv[real..real + kv_dim].copy_from_slice(&new_v[l][nt..nt + kv_dim]);
                kv.layers_k.push(sk);
                kv.layers_v.push(sv);
            }
            out.push((lo, kv));
        }
        Ok(out)
    }

    /// **Ragged fused batched decode** — decodes `B = entries.len()` sequences
    /// in ONE forward even when they sit at **different** cache lengths.
    ///
    /// Unlike [`Self::decode_batched_uniform`], each sequence carries its own
    /// past length (`kv.past_len`) and absolute position. The decode graph is
    /// built with per-sequence RoPE rows (`rope_cos`/`rope_sin` shaped
    /// `[batch, head_dim/2]`) and a per-row causal mask, and every sequence's KV
    /// is padded to the shared bucket `upper`. This is what lets a real server
    /// fuse arbitrary concurrent requests (varied prompt lengths, staggered
    /// arrivals) into a single batched forward — the full throughput win.
    ///
    /// Each `entries[i] = (token, kv)`; returns `(next_token_logits, advanced_kv)`
    /// per sequence with `advanced_kv.past_len == kv.past_len + 1`. Assumes
    /// non-sliding RoPE (absolute position == cache length).
    pub fn decode_batched_ragged(
        &mut self,
        entries: &[(u32, &KvCacheState)],
    ) -> Result<Vec<(Vec<f32>, KvCacheState)>> {
        let b = entries.len();
        if b == 0 {
            return Ok(Vec::new());
        }
        // A single sequence has nothing to fuse and no ragged RoPE need — the
        // shared-row uniform path is simpler and identical numerically.
        if b == 1 {
            let (tok, kv) = entries[0];
            let past = kv.past_len;
            return self.decode_batched_uniform(&[(tok, kv)], past, past);
        }
        let kv_dim = self.cfg.kv_proj_dim();
        let n_layers = self.cfg.num_hidden_layers;
        let vocab = self.cfg.vocab_size;
        let half = self.cfg.head_dim / 2;
        let max_past = self.cfg.max_position_embeddings.clamp(1, 4096) as u64;
        let device = self.device;
        let decode_opts = self.profile_compile_options(true);
        let cfg = self.cfg.clone();
        let weights = self.weights_cache.clone();

        // Per-sequence input token, RoPE row (at the sequence's own position),
        // and mask row (each sequence's own real past length).
        let input_ids_f32: Vec<f32> = entries.iter().map(|(t, _)| *t as f32).collect();
        let max_past_seq = entries.iter().map(|(_, kv)| kv.past_len).max().unwrap_or(0);

        // Graph + bucket `upper` for the longest sequence in the batch.
        let cache_b = self
            .batched_ragged_caches
            .entry(b)
            .or_insert_with(|| BucketedCompileCache::power_of_two_ladder(device, 1, max_past));
        let (upper_u64, compiled) = cache_b
            .ensure_graph_with_params(
                max_past_seq as u64,
                |upper| {
                    let mut wm = WeightMap::from_tensors((*weights).clone());
                    build_qwen3_decode_graph_sized_ragged(&cfg, &mut wm, b, upper as usize)
                        .expect("qwen3 ragged decode graph")
                },
                &decode_opts,
            )
            .ok_or_else(|| anyhow::anyhow!("past_seq {max_past_seq} outside decode buckets"))?;
        let upper = upper_u64 as usize;

        // Per-row RoPE `[batch, half]` and mask `[batch, upper+1]`.
        let mut cos = Vec::with_capacity(b * half);
        let mut sin = Vec::with_capacity(b * half);
        let mut mask = Vec::with_capacity(b * (upper + 1));
        for (_tok, kv) in entries {
            let (c, s) = compute_rope_slice(&cfg, kv.past_len);
            cos.extend_from_slice(&c);
            sin.extend_from_slice(&s);
            mask.extend_from_slice(&bucket_decode_mask(kv.past_len, upper));
        }

        // Batch-major padded KV `[batch, upper, kv_dim]`; each sequence's own
        // `past_len` real rows then zero padding to `upper`.
        let mut padded_k: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        let mut padded_v: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let mut pk = vec![0.0f32; b * upper * kv_dim];
            let mut pv = vec![0.0f32; b * upper * kv_dim];
            for (i, (_tok, kv)) in entries.iter().enumerate() {
                let real = kv.past_len * kv_dim;
                let base = i * upper * kv_dim;
                pk[base..base + real].copy_from_slice(&kv.layers_k[l][..real]);
                pv[base..base + real].copy_from_slice(&kv.layers_v[l][..real]);
            }
            padded_k.push(pk);
            padded_v.push(pv);
        }

        let key_strs: Vec<String> = (0..n_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> = Vec::with_capacity(4 + 2 * n_layers);
        inputs.push(("input_ids", &input_ids_f32));
        inputs.push(("rope_cos", &cos));
        inputs.push(("rope_sin", &sin));
        inputs.push(("mask", &mask));
        for l in 0..n_layers {
            inputs.push((&key_strs[2 * l], &padded_k[l]));
            inputs.push((&key_strs[2 * l + 1], &padded_v[l]));
        }

        let raw = compiled.run(&inputs);
        let (logits, new_k, new_v) = split_decode_logits_kv(raw, n_layers)?;

        // De-interleave per sequence: each has its own new length and the new
        // token's K/V at the padded row `upper`.
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            let past_i = entries[i].1.past_len;
            let real = past_i * kv_dim;
            let new_len = past_i + 1;
            let lo = logits[i * vocab..(i + 1) * vocab].to_vec();
            let mut kv = KvCacheState {
                past_len: new_len,
                layers_k: Vec::with_capacity(n_layers),
                layers_v: Vec::with_capacity(n_layers),
                layers_kv_base: vec![0; n_layers],
            };
            for l in 0..n_layers {
                let stride = (upper + 1) * kv_dim;
                let base = i * stride;
                let nt = base + upper * kv_dim;
                let mut sk = vec![0.0f32; new_len * kv_dim];
                let mut sv = vec![0.0f32; new_len * kv_dim];
                sk[..real].copy_from_slice(&new_k[l][base..base + real]);
                sv[..real].copy_from_slice(&new_v[l][base..base + real]);
                sk[real..real + kv_dim].copy_from_slice(&new_k[l][nt..nt + kv_dim]);
                sv[real..real + kv_dim].copy_from_slice(&new_v[l][nt..nt + kv_dim]);
                kv.layers_k.push(sk);
                kv.layers_v.push(sv);
            }
            out.push((lo, kv));
        }
        Ok(out)
    }

    /// Run prefill-with-cache and return the raw outputs. Uses the
    /// LRU `CompileCache` when enabled; otherwise compiles fresh each
    /// call. Keyed by `seq` because graph shape is seq-specialized.
    fn run_prefill_with_cache(
        &mut self,
        batch: usize,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        let prefill_opts = self.profile_compile_options(false);
        if let Some(cache) = &mut self.prefill_compile_cache {
            let key = prefill_cache_key(batch, seq);
            let mut wm = WeightMap::from_tensors((*self.weights_cache).clone());
            let (graph, params) = build_qwen3_graph_sized_last_logits(
                &self.cfg, &mut wm, batch, seq, /*with_kv_outputs*/ true,
            )?;
            let compiled = compile_cache_ensure_graph(cache, key, graph, params, &prefill_opts);
            Ok(compiled.run(&[("input_ids", ids_f32)]))
        } else {
            let mut wm = WeightMap::from_tensors((*self.weights_cache).clone());
            let (graph, params) = build_qwen3_graph_sized_last_logits(
                &self.cfg, &mut wm, batch, seq, /*with_kv_outputs*/ true,
            )?;
            let opts = self.profile_compile_options(false);
            let mut compiled = Session::new(self.device).compile_with(graph, &opts);
            for (name, data) in &params {
                compiled.set_param(name, data);
            }
            Ok(compiled.run(&[("input_ids", ids_f32)]))
        }
    }

    /// Run `n` cached steps and return the newly generated tokens.
    pub fn generate_cached(&mut self, n: usize, opts: SampleOpts) -> Result<Vec<u32>> {
        self.generate_cached_with(n, opts, |_| {})
    }

    /// Same as [`generate_cached`] but invokes `on_token` once per
    /// freshly sampled id, inside the decode loop. The whole `n` step
    /// loop shares the bucketed compile cache — callers wanting a
    /// streaming UI should prefer this to calling
    /// `generate_cached(1, …)` `n` times (which forces a fresh
    /// compile per token at the bucket boundaries).
    pub fn generate_cached_with(
        &mut self,
        n: usize,
        opts: SampleOpts,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        self.generate_cached_until(n, opts, |_| true, on_token)
    }

    /// Like [`generate_cached_with`] but stops early when `should_continue`
    /// returns `false` after sampling a token.
    pub fn generate_cached_until(
        &mut self,
        n: usize,
        opts: SampleOpts,
        mut should_continue: impl FnMut(u32) -> bool,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let start = self.tokens.len();
        for _ in 0..n {
            let tok = self.step_cached(opts)?;
            on_token(tok);
            if !should_continue(tok) {
                break;
            }
        }
        Ok(self.tokens[start..].to_vec())
    }

    /// Run prefill-with-cache on the current `self.tokens` (the
    /// prompt), populate `self.cache`, sample the next token from the
    /// last position's logits, and append it. Returns the sampled
    /// token. Invariant after: `cache.past_seq == tokens.len() - 1`.
    fn seed_cache_from_prompt(&mut self, opts: SampleOpts) -> Result<u32> {
        let seq = self.tokens.len();
        let batch = 1usize;
        let kv_dim = self.cfg.kv_proj_dim();

        let ids_f32: Vec<f32> = self.tokens.iter().map(|&i| i as f32).collect();
        let outputs = self.run_prefill_with_cache(batch, seq, &ids_f32)?;
        let (logits, kv) =
            kv_from_prefill_outputs(outputs, batch, seq, kv_dim, self.cfg.num_hidden_layers)?;
        self.cache = Some(kv);
        // Seed retention with the prompt's positions so the resident set tracks
        // the KV mirror row-for-row (Stage 2).
        if let Some(m) = self.retention.as_mut() {
            m.on_prefill(seq);
        }

        let vocab = self.cfg.vocab_size;
        let needed = vocab;
        if logits.len() < needed {
            anyhow::bail!("prefill logits length {} < {}", logits.len(), needed);
        }
        let last_row = &logits[..vocab];
        let tok = sample_token(last_row, opts) as u32;
        self.tokens.push(tok);
        // Prefill compiles a multi-GiB arena. Decode needs another; on discrete
        // / VirtIO wgpu keeping both OOMs (~2× act + weights). KV is already
        // on the host — drop the prefill executable so peak is one graph.
        // Weight buffers are process-shared in rlx-wgpu when layouts match.
        if matches!(
            self.device,
            Device::Gpu
                | Device::Vulkan
                | Device::WebGpu
                | Device::Cuda
                | Device::Rocm
                | Device::DirectX
        ) {
            if let Some(cache) = &mut self.prefill_compile_cache {
                cache.clear();
            }
        }
        Ok(tok)
    }

    /// Full token history (prompt + generated).
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// One teacher-forced forward over `tokens`; returns the flattened
    /// `[seq, vocab]` logits (row `i` predicts token `i+1`). A single graph
    /// run — used by the eval harness instead of O(N) single decode steps.
    pub fn sequence_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            anyhow::bail!("sequence_logits: empty token sequence");
        }
        let seq = tokens.len();
        let mut wm = WeightMap::from_tensors((*self.weights_cache).clone());
        let (graph, params) = build_qwen3_graph_sized(
            &self.cfg, &mut wm, 1, seq, /*with_lm_head*/ true, false,
        )?;
        let opts = self.profile_compile_options(false);
        let mut compiled = Session::new(self.device).compile_with(graph, &opts);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        let ids_f32: Vec<f32> = tokens.iter().map(|&i| i as f32).collect();
        let mut out = compiled.run(&[("input_ids", &ids_f32)]);
        out.drain(..)
            .next()
            .ok_or_else(|| anyhow::anyhow!("sequence_logits: graph produced no output"))
    }

    /// Teacher-forced next-token log-probabilities: `log P(tokens[i+1] |
    /// tokens[..=i])` for `i` in `0..seq-1`. One forward; host-side
    /// log-softmax. Returns `seq - 1` values (empty for a 1-token input).
    pub fn sequence_logprobs(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let seq = tokens.len();
        if seq < 2 {
            return Ok(Vec::new());
        }
        let vocab = self.cfg.vocab_size;
        let logits = self.sequence_logits(tokens)?;
        if logits.len() < seq * vocab {
            anyhow::bail!(
                "sequence_logits len {} < seq*vocab {}",
                logits.len(),
                seq * vocab
            );
        }
        let mut out = Vec::with_capacity(seq - 1);
        for i in 0..seq - 1 {
            let row = &logits[i * vocab..(i + 1) * vocab];
            out.push(log_softmax_at(row, tokens[i + 1] as usize));
        }
        Ok(out)
    }

    /// Snapshot the seeded KV cache + token history for prompt-cache reuse.
    /// `None` if no cache has been seeded yet (no `step_cached`/prefill-cache
    /// call has run).
    pub fn export_cache(&self) -> Option<(KvCacheState, Vec<u32>)> {
        self.cache
            .as_ref()
            .map(|c| (c.clone(), self.tokens.clone()))
    }

    /// Restore a previously exported KV cache + token history so generation
    /// resumes from a cached prefix instead of re-prefilling it.
    pub fn restore_cache(&mut self, cache: KvCacheState, tokens: Vec<u32>) {
        self.cache = Some(cache);
        self.tokens = tokens;
    }

    /// Prefill `prompt`, **reusing** a `restored` KV cache that already covers
    /// the first `reuse_len` tokens — only the suffix is processed (one decode
    /// step per suffix token). Returns the last-position logits and leaves
    /// `self.cache` seeded at `past_len == prompt.len()`, ready for decode.
    ///
    /// `reuse_len` is capped to `prompt.len() - 1` so the final prompt token
    /// is always re-processed: its *next-token* prediction logits are not
    /// stored in the cache, so we must recompute them. When the cache covers
    /// more than the cap, it is trimmed to the reused row count.
    pub fn prefill_with_reuse(
        &mut self,
        prompt: &[u32],
        restored: KvCacheState,
        reuse_len: usize,
    ) -> Result<Vec<f32>> {
        if prompt.is_empty() {
            anyhow::bail!("prefill_with_reuse: empty prompt");
        }
        let kv_dim = self.cfg.kv_proj_dim();
        let reuse = reuse_len.min(prompt.len() - 1).min(restored.past_len);
        let kv = if reuse == restored.past_len {
            restored
        } else {
            trim_kv(&restored, reuse, kv_dim)
        };
        self.cache = Some(kv);
        self.tokens = prompt[..reuse].to_vec();
        let mut logits = Vec::new();
        for &t in &prompt[reuse..] {
            logits = self.decode_get_logits(t)?;
        }
        Ok(logits)
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.cfg
    }

    /// The device this generator compiles + runs on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Low-level primitive: reset internal state, run prefill-with-cache
    /// over `context`, and return the *last position's* logits row
    /// (`P(next_token | context)`). Does NOT sample or append. The
    /// internal `tokens` buffer is set to `context` and the KV cache
    /// is populated to `past_seq = context.len()`.
    ///
    /// Used by [`crate::spec::Qwen3Speculator`] to compute the
    /// first row of a `Speculator::verify` / `propose` result before
    /// the decode loop runs.
    pub fn prefill_get_last_logits(&mut self, context: &[u32]) -> Result<Vec<f32>> {
        if context.is_empty() {
            anyhow::bail!("prefill_get_last_logits: empty context");
        }
        self.tokens.clear();
        self.tokens.extend_from_slice(context);
        self.cache = None;

        let seq = context.len();
        let batch = 1usize;
        let kv_dim = self.cfg.kv_proj_dim();

        let ids_f32: Vec<f32> = context.iter().map(|&i| i as f32).collect();
        let outputs = self.run_prefill_with_cache(batch, seq, &ids_f32)?;
        let (logits, kv) =
            kv_from_prefill_outputs(outputs, batch, seq, kv_dim, self.cfg.num_hidden_layers)?;
        self.cache = Some(kv);

        let vocab = self.cfg.vocab_size;
        let needed = vocab;
        if logits.len() < needed {
            anyhow::bail!("logits short: {} < {}", logits.len(), needed);
        }
        Ok(logits[..vocab].to_vec())
    }

    /// Low-level primitive: run one decode step with the caller-
    /// supplied input token (no sampling), advance the KV cache, and
    /// return the resulting logits row `P(next | history ++ input)`.
    /// Appends `input` to the `tokens` buffer so the invariant
    /// `cache.past_seq == tokens.len()` holds after this call (note:
    /// differs from `step_cached` invariant because this method does
    /// not append a sampled token).
    pub fn decode_get_logits(&mut self, input: u32) -> Result<Vec<f32>> {
        let cache = self.cache.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "decode_get_logits: cache not seeded; call prefill_get_last_logits first"
            )
        })?;
        let past_seq = cache.past_len;

        let mut wm = WeightMap::from_tensors((*self.weights_cache).clone());
        let (graph, params) =
            build_qwen3_decode_graph_sized(&self.cfg, &mut wm, /*batch*/ 1, past_seq)?;
        let opts = self.profile_compile_options(true);
        let mut compiled = Session::new(self.device).compile_with(graph, &opts);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }

        let (cos, sin) = compute_rope_slice(&self.cfg, past_seq);
        let input_ids_f32 = [input as f32];
        let key_strs: Vec<String> = (0..self.cfg.num_hidden_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> =
            Vec::with_capacity(3 + 2 * self.cfg.num_hidden_layers);
        inputs.push(("input_ids", input_ids_f32.as_slice()));
        inputs.push(("rope_cos", cos.as_slice()));
        inputs.push(("rope_sin", sin.as_slice()));
        for i in 0..self.cfg.num_hidden_layers {
            let pk = &cache.layers_k[i];
            let pv = &cache.layers_v[i];
            inputs.push((&key_strs[2 * i], pk.as_slice()));
            inputs.push((&key_strs[2 * i + 1], pv.as_slice()));
        }

        let outputs = compiled.run(&inputs);
        let (logits, new_k, new_v) = split_decode_logits_kv(outputs, self.cfg.num_hidden_layers)?;

        let cache_mut = self.cache.as_mut().unwrap();
        cache_mut.past_len = past_seq + 1;
        cache_mut.layers_k = new_k;
        cache_mut.layers_v = new_v;
        self.tokens.push(input);

        Ok(logits)
    }
}

/// `log softmax(row)[idx]` computed stably (`row[idx] - logsumexp(row)`).
fn log_softmax_at(row: &[f32], idx: usize) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sumexp: f32 = row.iter().map(|&x| (x - max).exp()).sum();
    let lse = max + sumexp.ln();
    row.get(idx).copied().unwrap_or(f32::NEG_INFINITY) - lse
}

/// Trim a KV cache to its first `rows` sequence positions (uniform `kv_dim`).
/// Each layer's `[past_len * kv_dim]` row-major buffer keeps its first
/// `rows * kv_dim` elements.
fn trim_kv(kv: &KvCacheState, rows: usize, kv_dim: usize) -> KvCacheState {
    let take = rows * kv_dim;
    let cut = |layers: &[Vec<f32>]| -> Vec<Vec<f32>> {
        layers
            .iter()
            .map(|buf| buf[..take.min(buf.len())].to_vec())
            .collect()
    };
    KvCacheState {
        past_len: rows,
        layers_k: cut(&kv.layers_k),
        layers_v: cut(&kv.layers_v),
        layers_kv_base: kv.layers_kv_base.clone(),
    }
}

/// Compute the single-row (cos, sin) RoPE slice for absolute position
/// `pos`. Matches the formula in the prefill builder so cached decode
/// and recompute prefill produce the same RoPE rotation.
fn compute_rope_slice(cfg: &Qwen3Config, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let dh = cfg.head_dim;
    let half = dh / 2;
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    for i in 0..half {
        let freq = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / dh as f64);
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
    use crate::config::Qwen3Config;

    #[test]
    fn gqa_pool_query_averages_within_group() {
        // 2 kv heads, group size 2 (=> 4 query heads), head_dim 2.
        // q head-major: h0=[1,1] h1=[3,3] | h2=[10,10] h3=[20,20]
        let q = vec![1.0, 1.0, 3.0, 3.0, 10.0, 10.0, 20.0, 20.0];
        let pooled = gqa_pool_query(
            &q, /*head_dim*/ 2, /*num_kv_heads*/ 2, /*group*/ 2,
        );
        // kv head 0 = mean(h0,h1) = [2,2]; kv head 1 = mean(h2,h3) = [15,15]
        assert_eq!(pooled, vec![2.0, 2.0, 15.0, 15.0]);
    }

    #[test]
    fn consecutive_abs_runs_splits_at_position_gaps() {
        // resident indices 0..5 map to abs positions with a gap between 31 and N-16:
        // this is exactly the retrieval case — retrieved block (4..7) spliced next
        // to a newly aged-out token (100). The run must SPLIT at the gap so the new
        // token forms its own block (not bundled with the already-offloaded run).
        let positions = vec![4, 5, 6, 7, 100, 101];
        let indices = vec![0, 1, 2, 3, 4, 5];
        let runs = consecutive_abs_runs(&indices, &positions);
        assert_eq!(runs, vec![vec![0, 1, 2, 3], vec![4, 5]]);
    }

    #[test]
    fn consecutive_abs_runs_handles_filtered_indices() {
        // After abs-pos dedup only the new tail survives (indices 4,5 → abs 100,101).
        let positions = vec![4, 5, 6, 7, 100, 101];
        let runs = consecutive_abs_runs(&[4, 5], &positions);
        assert_eq!(runs, vec![vec![4, 5]]);
    }

    #[test]
    fn gqa_pool_query_mha_is_identity_layout() {
        // group size 1 (MHA): pooled == q, dim unchanged.
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let pooled = gqa_pool_query(
            &q, /*head_dim*/ 2, /*num_kv_heads*/ 2, /*group*/ 1,
        );
        assert_eq!(pooled, q);
    }

    fn tiny_cfg() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 16,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: false,
            qk_norm: true,
            sliding_window: None,
            max_window_layers: usize::MAX,
            use_sliding_window: false,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        }
    }

    fn synthetic_weights(cfg: &Qwen3Config) -> WeightMap {
        let h = cfg.hidden_size;
        let q_dim = cfg.q_proj_dim();
        let kv_dim = cfg.kv_proj_dim();
        let int_dim = cfg.intermediate_size;
        let dh = cfg.head_dim;
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        // Use a deterministic non-zero pattern so logits aren't all 0
        // (sampling on an all-zero row is undefined order).
        let pat = |n: usize, salt: u32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(salt)) >> 8;
                    (x as f32 / (1u32 << 24) as f32) - 0.5
                })
                .collect()
        };
        t.insert(
            "model.embed_tokens.weight".into(),
            (pat(cfg.vocab_size * h, 1), vec![cfg.vocab_size, h]),
        );
        for i in 0..cfg.num_hidden_layers {
            let lp = format!("model.layers.{i}");
            t.insert(
                format!("{lp}.input_layernorm.weight"),
                (pat(h, 100 + i as u32), vec![h]),
            );
            t.insert(
                format!("{lp}.post_attention_layernorm.weight"),
                (pat(h, 200 + i as u32), vec![h]),
            );
            t.insert(
                format!("{lp}.self_attn.q_proj.weight"),
                (pat(q_dim * h, 300 + i as u32), vec![q_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.k_proj.weight"),
                (pat(kv_dim * h, 400 + i as u32), vec![kv_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.v_proj.weight"),
                (pat(kv_dim * h, 500 + i as u32), vec![kv_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.o_proj.weight"),
                (pat(h * q_dim, 600 + i as u32), vec![h, q_dim]),
            );
            t.insert(
                format!("{lp}.self_attn.q_norm.weight"),
                (pat(dh, 700 + i as u32), vec![dh]),
            );
            t.insert(
                format!("{lp}.self_attn.k_norm.weight"),
                (pat(dh, 800 + i as u32), vec![dh]),
            );
            t.insert(
                format!("{lp}.mlp.gate_proj.weight"),
                (pat(int_dim * h, 900 + i as u32), vec![int_dim, h]),
            );
            t.insert(
                format!("{lp}.mlp.up_proj.weight"),
                (pat(int_dim * h, 1000 + i as u32), vec![int_dim, h]),
            );
            t.insert(
                format!("{lp}.mlp.down_proj.weight"),
                (pat(h * int_dim, 1100 + i as u32), vec![h, int_dim]),
            );
        }
        t.insert("model.norm.weight".into(), (pat(h, 2000), vec![h]));
        t.insert(
            "lm_head.weight".into(),
            (pat(cfg.vocab_size * h, 3000), vec![cfg.vocab_size, h]),
        );
        WeightMap::from_tensors(t)
    }

    #[test]
    fn generator_drains_loader_and_runs_one_step() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut gn = Qwen3Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();
        assert_eq!(wm.len(), 0, "loader should be drained");
        gn.prefill(&[1, 2, 3]);
        let t = gn.step(SampleOpts::greedy()).unwrap();
        assert!((t as usize) < cfg.vocab_size);
        assert_eq!(gn.tokens().len(), 4);
    }

    /// Fused batched decode must equal independent single-sequence decode,
    /// token-for-token in logits AND in the advanced KV cache. This is the
    /// throughput primitive's correctness oracle: it pins the position-major
    /// `[seq, batch, kv_dim]` KV layout, the per-row mask, and the
    /// bucket-padded custom-mask graph at batch > 1, all on CPU.
    #[test]
    fn batched_decode_matches_single_sequence() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut g = Qwen3Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();

        let close = |a: &[f32], b: &[f32], who: &str| {
            assert_eq!(a.len(), b.len(), "{who}: length");
            for (j, (x, y)) in a.iter().zip(b).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-3 + 1e-3 * y.abs(),
                    "{who} elem[{j}]: batched {x} vs single {y}"
                );
            }
        };

        // Cover several prompt lengths: len 4 → bucket 3..5 (upper 4, no pad);
        // len 3 → bucket 3..5 (upper 4, ONE padded row, exercises the mask);
        // len 2 → bucket 2..3 (upper 2, no pad).
        for len in [2usize, 3, 4] {
            let prompt_a: Vec<u32> = (1..=len as u32).collect();
            let prompt_b: Vec<u32> = (1..=len as u32).map(|t| t + 5).collect();
            let tok_a = 5u32;
            let tok_b = 11u32;
            let past = len;

            // Reference A: single-seq decode + its advanced cache.
            g.prefill_get_last_logits(&prompt_a).unwrap();
            let (kv_a, _) = g.export_cache().unwrap();
            let exp_a = g.decode_get_logits(tok_a).unwrap();
            let (kv_a_next, _) = g.export_cache().unwrap();

            // Reference B (fresh prefill resets the cache).
            g.prefill_get_last_logits(&prompt_b).unwrap();
            let (kv_b, _) = g.export_cache().unwrap();
            let exp_b = g.decode_get_logits(tok_b).unwrap();
            let (kv_b_next, _) = g.export_cache().unwrap();

            // B=1 through the batched path must match single-seq decode exactly
            // (isolates mask/rope/graph from the multi-sequence interleaving).
            let solo = g
                .decode_batched_uniform(&[(tok_a, &kv_a)], past, past)
                .unwrap();
            close(&solo[0].0, &exp_a, &format!("len{len} B=1 logits"));

            // Fused: both sequences in ONE batched forward.
            let out = g
                .decode_batched_uniform(&[(tok_a, &kv_a), (tok_b, &kv_b)], past, past)
                .unwrap();
            assert_eq!(out.len(), 2);
            close(&out[0].0, &exp_a, &format!("len{len} seq A logits"));
            close(&out[1].0, &exp_b, &format!("len{len} seq B logits"));

            // The de-interleaved KV must equal each single-seq advanced cache.
            assert_eq!(out[0].1.past_len, past + 1);
            assert_eq!(out[1].1.past_len, past + 1);
            for l in 0..cfg.num_hidden_layers {
                close(
                    &out[0].1.layers_k[l],
                    &kv_a_next.layers_k[l],
                    &format!("len{len} A K"),
                );
                close(
                    &out[0].1.layers_v[l],
                    &kv_a_next.layers_v[l],
                    &format!("len{len} A V"),
                );
                close(
                    &out[1].1.layers_k[l],
                    &kv_b_next.layers_k[l],
                    &format!("len{len} B K"),
                );
                close(
                    &out[1].1.layers_v[l],
                    &kv_b_next.layers_v[l],
                    &format!("len{len} B V"),
                );
            }
        }
    }

    /// Ragged fused decode (sequences at DIFFERENT cache lengths) must equal
    /// independent single-sequence decode — logits and advanced KV. This pins
    /// the per-sequence RoPE table + per-row mask path that lets arbitrary
    /// concurrent requests fuse into one forward.
    #[test]
    fn ragged_batched_decode_matches_single_sequence() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut g = Qwen3Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();

        let close = |a: &[f32], b: &[f32], who: &str| {
            assert_eq!(a.len(), b.len(), "{who} len");
            for (j, (x, y)) in a.iter().zip(b).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-3 + 1e-3 * y.abs(),
                    "{who}[{j}]: {x} vs {y}"
                );
            }
        };

        // Seq A at past_len 2, seq B at past_len 5 — different RoPE positions,
        // different mask rows, padded to a shared bucket upper.
        let prompt_a = vec![1u32, 2];
        let prompt_b = vec![3u32, 4, 5, 6, 7];
        let tok_a = 8u32;
        let tok_b = 9u32;

        g.prefill_get_last_logits(&prompt_a).unwrap();
        let (kv_a, _) = g.export_cache().unwrap();
        let exp_a = g.decode_get_logits(tok_a).unwrap();
        let (kv_a_next, _) = g.export_cache().unwrap();

        g.prefill_get_last_logits(&prompt_b).unwrap();
        let (kv_b, _) = g.export_cache().unwrap();
        let exp_b = g.decode_get_logits(tok_b).unwrap();
        let (kv_b_next, _) = g.export_cache().unwrap();

        let out = g
            .decode_batched_ragged(&[(tok_a, &kv_a), (tok_b, &kv_b)])
            .unwrap();
        assert_eq!(out.len(), 2);
        close(&out[0].0, &exp_a, "ragged A logits");
        close(&out[1].0, &exp_b, "ragged B logits");

        // Order-independence: swapping the batch must swap the outputs, not
        // corrupt them (guards the per-sequence RoPE/mask indexing).
        let sw = g
            .decode_batched_ragged(&[(tok_b, &kv_b), (tok_a, &kv_a)])
            .unwrap();
        close(&sw[0].0, &exp_b, "swapped B logits");
        close(&sw[1].0, &exp_a, "swapped A logits");

        assert_eq!(out[0].1.past_len, prompt_a.len() + 1);
        assert_eq!(out[1].1.past_len, prompt_b.len() + 1);
        for l in 0..cfg.num_hidden_layers {
            close(&out[0].1.layers_k[l], &kv_a_next.layers_k[l], "ragged A K");
            close(&out[0].1.layers_v[l], &kv_a_next.layers_v[l], "ragged A V");
            close(&out[1].1.layers_k[l], &kv_b_next.layers_k[l], "ragged B K");
            close(&out[1].1.layers_v[l], &kv_b_next.layers_v[l], "ragged B V");
        }
    }

    #[test]
    fn generate_n_appends_n_tokens() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut gn = Qwen3Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();
        gn.prefill(&[5, 6]);
        let new_tokens = gn.generate(3, SampleOpts::greedy()).unwrap();
        assert_eq!(new_tokens.len(), 3);
        assert_eq!(gn.tokens().len(), 5);
        for t in &new_tokens {
            assert!((*t as usize) < cfg.vocab_size);
        }
    }

    #[test]
    fn step_without_prefill_errors() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut gn = Qwen3Generator::from_loader(cfg, &mut wm, Device::Cpu).unwrap();
        let r = gn.step(SampleOpts::greedy());
        assert!(r.is_err());
    }

    #[test]
    fn cached_matches_naive_on_greedy() {
        // The cached and naive paths must produce the same token
        // sequence given the same prompt + opts. This is the
        // load-bearing test for the KV-cache implementation: if the
        // decode-mode graph, the kernel's Lq!=Lk fix, the cache
        // wiring, or the RoPE position-slice is wrong, the sequences
        // diverge here.
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];
        let steps = 4;

        let mut wm_n = synthetic_weights(&cfg);
        let mut gn_naive =
            Qwen3Generator::from_loader(cfg.clone(), &mut wm_n, Device::Cpu).unwrap();
        gn_naive.prefill_compile_cache = None;
        gn_naive.decode_compile_cache = None;
        gn_naive.prefill(&prompt);
        let naive_tokens = gn_naive.generate(steps, SampleOpts::greedy()).unwrap();

        let mut wm_c = synthetic_weights(&cfg);
        let mut gn_cached =
            Qwen3Generator::from_loader(cfg.clone(), &mut wm_c, Device::Cpu).unwrap();
        gn_cached.prefill(&prompt);
        let cached_tokens = gn_cached
            .generate_cached(steps, SampleOpts::greedy())
            .unwrap();

        assert_eq!(
            cached_tokens, naive_tokens,
            "cached vs naive token mismatch — KV cache or kernel-Lq!=Lk bug"
        );
    }

    #[test]
    fn sliding_window_cached_decode_matches_naive() {
        // With a sliding window, the KV-cached decode path (windowed custom
        // mask) must reproduce naive prefill-recompute (SlidingWindow op),
        // generating past the window so the masking actually bites.
        let mut cfg = tiny_cfg();
        cfg.use_sliding_window = true;
        cfg.sliding_window = Some(3);
        let prompt: Vec<u32> = vec![1, 2, 3, 5, 4];
        let steps = 5;

        let mut wm_n = synthetic_weights(&cfg);
        let mut gn_naive =
            Qwen3Generator::from_loader(cfg.clone(), &mut wm_n, Device::Cpu).unwrap();
        gn_naive.prefill_compile_cache = None;
        gn_naive.decode_compile_cache = None;
        gn_naive.prefill(&prompt);
        let naive = gn_naive.generate(steps, SampleOpts::greedy()).unwrap();

        let mut wm_c = synthetic_weights(&cfg);
        let mut gn_cached =
            Qwen3Generator::from_loader(cfg.clone(), &mut wm_c, Device::Cpu).unwrap();
        gn_cached.prefill(&prompt);
        let cached = gn_cached
            .generate_cached(steps, SampleOpts::greedy())
            .unwrap();

        assert_eq!(
            cached, naive,
            "windowed cached decode must match naive windowed prefill"
        );
    }

    #[test]
    fn sliding_window_rotates_cache_to_bound_memory() {
        // The rotating cache stays at `window` rows no matter how long the
        // prompt + generation get — O(window) memory instead of O(sequence).
        let mut cfg = tiny_cfg();
        cfg.use_sliding_window = true;
        cfg.sliding_window = Some(3);
        let mut wm = synthetic_weights(&cfg);
        let mut gn = Qwen3Generator::from_loader(cfg, &mut wm, Device::Cpu).unwrap();
        gn.prefill(&[1, 2, 3, 5, 4, 6, 7]); // prompt len 7 > window 3
        let _ = gn.generate_cached(5, SampleOpts::greedy()).unwrap();
        let cache = gn.cache.as_ref().unwrap();
        assert_eq!(cache.past_len, 3, "cache must be bounded to the window");
        // And each layer's K/V buffer holds exactly `window` rows.
        let kv_dim = gn.cfg.kv_proj_dim();
        assert_eq!(cache.layers_k[0].len(), 3 * kv_dim);
    }

    #[test]
    fn sliding_window_reduces_to_causal_when_wide() {
        // A sliding window ≥ seq must reproduce full causal attention; a
        // narrow window must change the result (the mask actually restricts).
        let prompt: Vec<u32> = vec![1, 2, 3, 5, 4];
        let base = tiny_cfg();

        let logits = |mut cfg: Qwen3Config, use_sw: bool, win: Option<usize>| {
            cfg.use_sliding_window = use_sw;
            cfg.sliding_window = win;
            let mut wm = synthetic_weights(&cfg);
            let mut g = Qwen3Generator::from_loader(cfg, &mut wm, Device::Cpu).unwrap();
            g.sequence_logits(&prompt).unwrap()
        };

        let causal = logits(base.clone(), false, None);
        let wide = logits(base.clone(), true, Some(100)); // ≥ seq
        let narrow = logits(base.clone(), true, Some(2)); // < seq

        assert_eq!(causal.len(), wide.len());
        for (a, b) in causal.iter().zip(&wide) {
            assert!(
                (a - b).abs() < 1e-4,
                "wide window must equal causal: {a} vs {b}"
            );
        }
        let diff: f32 = causal.iter().zip(&narrow).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-3,
            "narrow window should differ from causal (sum |Δ|={diff})"
        );
    }

    #[test]
    fn sequence_logprobs_shape_and_consistency() {
        let cfg = tiny_cfg();
        let vocab = cfg.vocab_size;
        let tokens: Vec<u32> = vec![1, 2, 3, 5, 4];
        let mut wm = synthetic_weights(&cfg);
        let mut g = Qwen3Generator::from_loader(cfg, &mut wm, Device::Cpu).unwrap();

        let logits = g.sequence_logits(&tokens).unwrap();
        assert_eq!(logits.len(), tokens.len() * vocab);

        let lps = g.sequence_logprobs(&tokens).unwrap();
        assert_eq!(lps.len(), tokens.len() - 1);
        // All log-probs are <= 0 and finite.
        assert!(lps.iter().all(|&x| x <= 1e-4 && x.is_finite()));
        // Consistent with a hand log-softmax of the corresponding row.
        let row0 = &logits[0..vocab];
        let hand = log_softmax_at(row0, tokens[1] as usize);
        assert!((lps[0] - hand).abs() < 1e-5);
    }

    #[test]
    fn prefill_with_reuse_matches_full_prefill() {
        // Reusing a cached prefix + decoding the suffix must predict the
        // same next token as a full prefill of the whole prompt.
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5, 4, 6];

        let mut wm_ref = synthetic_weights(&cfg);
        let mut g_ref = Qwen3Generator::from_loader(cfg.clone(), &mut wm_ref, Device::Cpu)
            .unwrap()
            .with_decode_cache(64);
        let ref_logits = g_ref.prefill_get_last_logits(&prompt).unwrap();
        let ref_tok = sample_token(&ref_logits, SampleOpts::greedy());

        // Build a prefix cache covering the first 3 tokens.
        let mut wm_p = synthetic_weights(&cfg);
        let mut g_pref = Qwen3Generator::from_loader(cfg.clone(), &mut wm_p, Device::Cpu)
            .unwrap()
            .with_decode_cache(64);
        let _ = g_pref.prefill_get_last_logits(&prompt[..3]).unwrap();
        let (prefix_kv, prefix_tokens) = g_pref.export_cache().unwrap();
        assert_eq!(prefix_kv.past_len, 3);
        assert_eq!(prefix_tokens, prompt[..3].to_vec());

        // Reuse the prefix, prefill only the suffix.
        let mut wm_r = synthetic_weights(&cfg);
        let mut g_reuse = Qwen3Generator::from_loader(cfg.clone(), &mut wm_r, Device::Cpu)
            .unwrap()
            .with_decode_cache(64);
        let reuse_logits = g_reuse.prefill_with_reuse(&prompt, prefix_kv, 3).unwrap();
        let reuse_tok = sample_token(&reuse_logits, SampleOpts::greedy());

        assert_eq!(
            reuse_tok, ref_tok,
            "suffix-prefill must predict the same next token as full prefill"
        );
        // Cache is left ready for decode at the full prompt length.
        assert_eq!(g_reuse.cache.as_ref().unwrap().past_len, prompt.len());
    }

    #[test]
    fn cached_step_advances_cache_invariant() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut gn = Qwen3Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();
        gn.prefill(&[1, 2, 3]);
        let _ = gn.step_cached(SampleOpts::greedy()).unwrap();
        // After seed: tokens.len() == 4, cache.past_seq == 3 (cache holds prompt).
        assert_eq!(gn.tokens().len(), 4);
        assert_eq!(gn.cache.as_ref().unwrap().past_len, 3);
        let _ = gn.step_cached(SampleOpts::greedy()).unwrap();
        // After one decode: tokens.len() == 5, cache.past_seq == 4.
        assert_eq!(gn.tokens().len(), 5);
        assert_eq!(gn.cache.as_ref().unwrap().past_len, 4);
    }

    #[test]
    fn bucketed_decode_matches_oneshot() {
        // The bucketed compile-cache path (padded K/V + custom mask)
        // must produce the same token sequence as the one-shot
        // path. Load-bearing for the bucketed cache feature: if the
        // mask, padding, or output slicing is wrong, sequences
        // diverge here.
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];
        let steps = 6;

        let mut wm_one = synthetic_weights(&cfg);
        let mut gn_one =
            Qwen3Generator::from_loader(cfg.clone(), &mut wm_one, Device::Cpu).unwrap();
        gn_one.prefill(&prompt);
        let oneshot_tokens = gn_one.generate_cached(steps, SampleOpts::greedy()).unwrap();

        let mut wm_buc = synthetic_weights(&cfg);
        let mut gn_buc = Qwen3Generator::from_loader(cfg.clone(), &mut wm_buc, Device::Cpu)
            .unwrap()
            .with_decode_cache(/*max_past*/ 32);
        gn_buc.prefill(&prompt);
        let bucketed_tokens = gn_buc.generate_cached(steps, SampleOpts::greedy()).unwrap();

        assert_eq!(
            bucketed_tokens, oneshot_tokens,
            "bucketed-cache decode diverged from one-shot decode — \
             mask, padding, or output-slice bug"
        );
    }

    #[test]
    fn bucketed_decode_q_proj_seq_is_one() {
        use rlx_ir::Op;

        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let (graph, _) = build_qwen3_decode_graph_sized_ext(&cfg, &mut wm, 1, 4, true).unwrap();
        for node in graph.nodes() {
            if let Op::MatMul = &node.op {
                let sh = graph.shape(node.id);
                if sh.rank() == 3 && sh.dim(2).unwrap_static() == cfg.q_proj_dim() {
                    assert_eq!(
                        sh.dim(1).unwrap_static(),
                        1,
                        "decode q_proj matmul seq dim must be 1, got {sh} on node {}",
                        node.id
                    );
                }
            }
        }

        let fused = rlx_opt::CompilePipeline::new(rlx_opt::FusionTarget::Metal)
            .with_assert_fusion_clean(false)
            .compile_graph(graph)
            .lir
            .into_graph();
        for node in fused.nodes() {
            if let Op::Narrow { len, .. } = &node.op {
                let sh = fused.shape(node.id);
                if sh.rank() == 3 && *len == cfg.q_proj_dim() {
                    assert_eq!(
                        sh.dim(1).unwrap_static(),
                        1,
                        "fused decode q narrow seq dim must be 1, got {sh} on node {}",
                        node.id
                    );
                }
            }
        }
    }

    #[test]
    fn prefill_compile_cache_does_not_change_output() {
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];
        let mut wm_a = synthetic_weights(&cfg);
        let mut gn_a = Qwen3Generator::from_loader(cfg.clone(), &mut wm_a, Device::Cpu).unwrap();
        gn_a.prefill(&prompt);
        let a = gn_a.generate_cached(4, SampleOpts::greedy()).unwrap();

        let mut wm_b = synthetic_weights(&cfg);
        let mut gn_b = Qwen3Generator::from_loader(cfg.clone(), &mut wm_b, Device::Cpu)
            .unwrap()
            .with_prefill_cache(/*capacity*/ 4);
        gn_b.prefill(&prompt);
        let b = gn_b.generate_cached(4, SampleOpts::greedy()).unwrap();

        assert_eq!(a, b, "enabling prefill_cache must not change output");
    }

    #[test]
    fn greedy_is_deterministic_across_runs() {
        let cfg = tiny_cfg();
        let weights = synthetic_weights(&cfg);
        let mk = || {
            let mut wm = WeightMap::from_tensors(weights_as_hashmap(&weights));
            Qwen3Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap()
        };
        let mut a = mk();
        let mut b = mk();
        a.prefill(&[1, 2, 3]);
        b.prefill(&[1, 2, 3]);
        let ta = a.generate(4, SampleOpts::greedy()).unwrap();
        let tb = b.generate(4, SampleOpts::greedy()).unwrap();
        assert_eq!(ta, tb);
    }

    fn weights_as_hashmap(wm: &WeightMap) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
        // Reconstruct the underlying map by re-running synthetic_weights
        // — WeightMap doesn't expose its inner map. Sufficient for the
        // determinism test since synthetic_weights is itself
        // deterministic.
        let _ = wm; // silence unused
        let cfg = tiny_cfg();
        let mut new = synthetic_weights(&cfg);
        let keys: Vec<String> = new.keys().map(|s| s.to_string()).collect();
        let mut out = HashMap::new();
        for k in keys {
            out.insert(k.clone(), new.take(&k).unwrap());
        }
        out
    }
}
