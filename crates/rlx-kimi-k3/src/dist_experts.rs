// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Kimi-K3's [`ExpertProvider`] — the model-specific WORKER half of the
//! `rlx-distributed` MoE expert-parallel offload.
//!
//! Given the latent FFN input `h_lat [rows, L]` and the fired routing
//! (`ids/probs [rows, top_k]`), it computes the **pre-norm routed-expert partial in
//! latent space** — `Σ_{token,slot : owns(id)} prob · expert_id(h_lat)` → `[rows, L]`
//! — for the experts THIS worker holds (paged from its local checkpoint). The
//! orchestrator sums the per-worker latent partials, then applies `routed_norm` +
//! `routed_expert_up_proj` + the shared experts (the norm is nonlinear, so it
//! CANNOT distribute over the worker sum — hence the cut is in latent, pre-norm).
//!
//! Reuses the exact routed-expert math from [`crate::moe`] (grouped gate_up → situ
//! → grouped down → prob-weighted sum over k), restricted to the owned expert set.

use crate::common::{reg, situ};
use crate::loader::CheckpointLoader;
use crate::moe::{MoeDims, build_moe_route, build_moe_tail};
use anyhow::Result;
use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_distributed::{ExpertProvider, ExpertShards, Transport, dispatch_experts};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::Op;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::Device;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Cumulative worker-side timing (expert PAGING from disk vs graph compile+run on
/// the engine) — the two costs a worker actually pays per request. Printed at
/// worker shutdown so we can see where a run's wall time goes.
#[derive(Default, Clone, Copy)]
pub struct WorkerTiming {
    pub page_ms: f64,    // load_expert (MXFP4→f32 from local shard) on cache MISS
    pub graph_ms: f64,   // compile_built + set_param + run (the GPU/CPU expert GEMMs)
    pub compile_ms: f64, // compile_built ONLY (graph lowering/kernel build) — the recompile cost
    pub run_ms: f64,     // set_param + run ONLY (weight upload + the GEMM)
    pub calls: u64,
    pub experts_paged: u64, // cache misses (actually read+dequant'd from disk)
    pub cache_hits: u64,    // experts served from the resident RAM cache
    pub graph_hits: u64,    // compiled-graph cache hits (recompile avoided)
}

/// Cumulative orchestrator-side MoE timing (per phase, summed over offloaded layers).
#[derive(Default, Clone, Copy)]
pub struct OrchTiming {
    pub phase1_ms: f64,   // load_moe_dense + router + down_latent (Mac)
    pub dispatch_ms: f64, // send hidden → workers → gather (network + REMOTE engine compute)
    pub local_ms: f64,    // Mac-local overflow shard compute
    pub tail_ms: f64,     // routed_norm + up_proj + shared experts (Mac)
    pub layers: u64,
}

/// One dequantized expert kept resident: (gate_up [L,2mi], down [mi,L]).
type ResidentExpert = Arc<(Vec<f32>, Vec<f32>)>;

/// A worker's shard: the routed experts it holds locally, paged on demand.
pub struct KimiExpertProvider {
    ck: CheckpointLoader,
    d: MoeDims,
    owned: HashSet<usize>,
    device: Device,
    timing: WorkerTiming,
    /// Resident (layer,expert)→dequantized-weights cache. Gated by
    /// `RLX_KIMI_EXPERT_CACHE=<MB>`: on a cache hit we skip BOTH the disk read AND
    /// the CPU MXFP4 dequant+transpose (the dominant "paging" cost). Amortizes across
    /// decode steps that re-route to the same expert. `None` when the env is unset.
    cache: Option<HashMap<(u32, usize), ResidentExpert>>,
    cache_budget: usize, // bytes
    cache_bytes: usize,  // bytes resident
    /// Use the MXFP4-PACKED expert path (`DequantGroupedMatMulMlx`, GPU dequant, no CPU
    /// dequant + ~7.5× less data moved) instead of the f32 `GroupedMatMul` path. Gated
    /// by `RLX_KIMI_PACKED_EXPERTS`; requires the device to support that op (CPU / CUDA /
    /// Metal / MLX / Vulkan; ROCm via its host-delegate arm).
    packed: bool,
    /// Use the W4A8 GPU-NATIVE path (`Op::ScaledGroupedMatMul`): MXFP4 weights +
    /// MXFP8-quantized activations, decoded+GEMM'd ON the GPU (no host-delegate, no
    /// device↔host copies — the real speedup). Gated by `RLX_KIMI_SCALED_EXPERTS`.
    /// Weights are nibble-unpacked (op wants 1 code/byte); a touch lossier (FP8 acts).
    scaled: bool,
    /// Compiled packed-expert graphs, keyed by `(n_bucket, rows)` — the only shape dims
    /// baked into the graph. The worker previously `compile_built` a fresh graph EVERY
    /// MoE call (per-call recompile co-dominated GPU compute). We bucket the fired-expert
    /// count to a power of two so a handful of shapes recur across layers/decode steps;
    /// on a hit we only re-upload the (padded) codes/scales and `run` — no recompile.
    /// Gated by `RLX_KIMI_NO_GRAPH_CACHE`.
    graph_cache: Option<HashMap<(usize, usize), rlx_runtime::CompiledGraph>>,
}

impl KimiExpertProvider {
    /// Open a provider over `model_dir` owning `owned` expert ids, computing on
    /// `device` (Cpu locally / Cuda-Rocm on the real workers).
    pub fn open(
        model_dir: &str,
        d: MoeDims,
        owned: HashSet<usize>,
        device: Device,
    ) -> Result<Self> {
        // RLX_KIMI_EXPERT_CACHE = resident cache budget in MB (unset/0 = no cache).
        let cache_budget = std::env::var("RLX_KIMI_EXPERT_CACHE")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(0);
        Ok(Self {
            ck: CheckpointLoader::open(model_dir)?,
            d,
            owned,
            device,
            timing: WorkerTiming::default(),
            cache: (cache_budget > 0).then(HashMap::new),
            cache_budget,
            cache_bytes: 0,
            packed: std::env::var("RLX_KIMI_PACKED_EXPERTS")
                .map(|v| v != "0")
                .unwrap_or(false),
            scaled: std::env::var("RLX_KIMI_SCALED_EXPERTS")
                .map(|v| v != "0")
                .unwrap_or(false),
            graph_cache: std::env::var("RLX_KIMI_NO_GRAPH_CACHE")
                .is_err()
                .then(HashMap::new),
        })
    }

    /// Cumulative paging/compute timing since open (for post-run reporting).
    pub fn timing(&self) -> WorkerTiming {
        self.timing
    }

    /// Force the packed / f32 expert path (overrides the env default). Used by the
    /// `expert-selfcheck` packed-vs-f32 parity comparison.
    pub fn set_packed(&mut self, packed: bool) {
        self.packed = packed;
    }

    /// Force the W4A8 scaled path on/off (overrides env). For selfcheck parity.
    pub fn set_scaled(&mut self, scaled: bool) {
        self.scaled = scaled;
    }

    /// Packed MXFP4 path: page OWNED fired experts as raw codes+scales (no CPU dequant),
    /// feed them to the shared [`build_packed_routed_latent`] graph (GPU dequant+matmul),
    /// and return the pre-norm latent partial `[rows, L]` — same result as the f32
    /// [`ExpertProvider::compute`], but only the ~17.5MB packed bytes move per expert.
    #[allow(clippy::too_many_arguments)]
    fn compute_packed(
        &mut self,
        _layer: u32,
        h_lat: &[f32],
        rows: usize,
        l: usize,
        k: usize,
        uniq: &[usize],
        remap: &HashMap<usize, usize>,
        ids: &[u32],
        probs: &[f32],
        lp: &str,
    ) -> Result<Vec<f32>> {
        use half::bf16;
        let n = uniq.len();
        // per-slot compact expert index [rows,k] (0 dummy for non-owned) + prob [rows,k]
        // (0 there — so non-owned slots contribute 0 to the sum).
        let mut eidx = vec![0f32; rows * k];
        let mut pmask = vec![0f32; rows * k];
        for i in 0..rows * k {
            if let Some(&ci) = remap.get(&(ids[i] as usize)) {
                eidx[i] = ci as f32;
                pmask[i] = probs[i];
            }
        }
        // page owned experts' RAW packed bytes (codes + e8m0 scales), stack compactly.
        let e8m0_bf16 = |s: &[u8]| -> Vec<u8> {
            s.iter()
                .flat_map(|&b| bf16::from_bits((b as u16) << 7).to_le_bytes())
                .collect()
        };
        // Page owned fired experts CONCURRENTLY. Was: serial `load_expert_packed`
        // (whole-shard mmap, one expert at a time) + serial `extend` assembly — the
        // measured 135 MB/s (msi) / 77 MB/s (amd) cold-paging floor, overhead-bound
        // not disk-bound (it left every worker core but one idle). Now mirrors the Mac
        // runner's `moe:paging(packed)`:
        //   1. resolve on-disk ranges on this thread (needs shard headers),
        //   2. `pread` each expert on a rayon worker (concurrent fault-in, no dequant),
        //   3. assemble the contiguous packed GPU buffers in parallel (par_chunks_mut).
        // Re-fired experts are served from the shared byte-budgeted io_opt RAM cache
        // (RLX_KIMI_EXPERT_CACHE=<MB>) — skips BOTH the disk read and the byte copy.
        use rayon::prelude::*;
        use std::sync::Arc;
        type Ep = crate::loader::ExpertPacked;
        let t_page = Instant::now();
        let loaded: Vec<Arc<Ep>> = if crate::io_opt::io_opt_active() {
            let cached: Vec<Option<Arc<Ep>>> = uniq
                .iter()
                .map(|&e| crate::io_opt::cache_get(lp, e))
                .collect();
            let miss: Vec<usize> = (0..uniq.len()).filter(|&i| cached[i].is_none()).collect();
            let miss_ranges: Vec<_> = miss
                .iter()
                .map(|&i| self.ck.expert_ranges(lp, uniq[i]))
                .collect::<Result<_>>()?;
            let paged: Vec<Ep> = miss_ranges
                .par_iter()
                .map(crate::loader::load_expert_ranges_packed)
                .collect::<Result<_>>()?;
            let mut items = cached;
            for (j, p) in paged.into_iter().enumerate() {
                let i = miss[j];
                let a = Arc::new(p);
                crate::io_opt::cache_put(lp, uniq[i], a.clone());
                items[i] = Some(a);
            }
            items
                .into_iter()
                .map(|it| it.expect("expert missing after paging"))
                .collect()
        } else {
            let ranges: Vec<_> = uniq
                .iter()
                .map(|&e| self.ck.expert_ranges(lp, e))
                .collect::<Result<_>>()?;
            ranges
                .par_iter()
                .map(|r| crate::loader::load_expert_ranges_packed(r).map(Arc::new))
                .collect::<Result<_>>()?
        };
        // Bucket the fired-expert count to a power of two so the compiled graph shape
        // recurs across layers/decode steps (the cache key is only `(nb, rows)`). The
        // padded expert slots `[n, nb)` are never referenced by `eidx` (routing indexes
        // 0..n), so they contribute 0 — we just leave their codes/scales zero. Without
        // the cache (`RLX_KIMI_NO_GRAPH_CACHE`) we keep the exact count (`nb == n`).
        let nb = if self.graph_cache.is_some() {
            n.max(1).next_power_of_two()
        } else {
            n
        };
        let (mut gc, mut gsc, mut uc, mut usc, mut dc, mut dsc) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        if n > 0 {
            let e0 = &loaded[0];
            let (g1, u1, d1) = (e0.w1_q.len(), e0.w3_q.len(), e0.w2_q.len());
            let (gs, us, ds) = (e0.w1_s.len() * 2, e0.w3_s.len() * 2, e0.w2_s.len() * 2);
            // Buffers sized to nb; the zip fills only the n real experts (padded tail 0).
            gc = vec![0u8; nb * g1];
            uc = vec![0u8; nb * u1];
            dc = vec![0u8; nb * d1];
            gsc = vec![0u8; nb * gs];
            usc = vec![0u8; nb * us];
            dsc = vec![0u8; nb * ds];
            gc.par_chunks_mut(g1)
                .zip(loaded.par_iter())
                .for_each(|(s, p)| s.copy_from_slice(&p.w1_q));
            uc.par_chunks_mut(u1)
                .zip(loaded.par_iter())
                .for_each(|(s, p)| s.copy_from_slice(&p.w3_q));
            dc.par_chunks_mut(d1)
                .zip(loaded.par_iter())
                .for_each(|(s, p)| s.copy_from_slice(&p.w2_q));
            gsc.par_chunks_mut(gs)
                .zip(loaded.par_iter())
                .for_each(|(s, p)| s.copy_from_slice(&e8m0_bf16(&p.w1_s)));
            usc.par_chunks_mut(us)
                .zip(loaded.par_iter())
                .for_each(|(s, p)| s.copy_from_slice(&e8m0_bf16(&p.w3_s)));
            dsc.par_chunks_mut(ds)
                .zip(loaded.par_iter())
                .for_each(|(s, p)| s.copy_from_slice(&e8m0_bf16(&p.w2_s)));
        }
        self.timing.page_ms += t_page.elapsed().as_secs_f64() * 1e3;
        self.timing.experts_paged += n as u64;
        let (gb, ub, db) = (
            vec![0u8; gsc.len()],
            vec![0u8; usc.len()],
            vec![0u8; dsc.len()],
        );

        // Compile the `(nb, rows)` graph on a cache MISS; reuse it on a HIT (only the
        // codes/scales are re-uploaded per call). This kills the per-call recompile that
        // co-dominated worker compute.
        let key = (nb, rows);
        let need_compile = match &self.graph_cache {
            Some(gcache) => !gcache.contains_key(&key),
            None => true,
        };
        let mut fresh = None;
        if need_compile {
            let f = DType::F32;
            let mut hir = HirModule::new("routed_partial_packed");
            let mut g = HirMut::new(&mut hir);
            let hlat_n = g.input("hlat", Shape::new(&[rows, l], f));
            let idx_n = g.input("idx", Shape::new(&[rows, k], f));
            let prob_n = g.input("prob", Shape::new(&[rows, k], f));
            let out = crate::moe::build_packed_routed_latent(
                &mut g, hlat_n, idx_n, prob_n, nb, rows, self.d,
            );
            g.set_outputs(vec![out]);
            let t_c = Instant::now();
            let compiled = compile_built(built_from_hir(hir, HashMap::new())?, self.device)?;
            self.timing.compile_ms += t_c.elapsed().as_secs_f64() * 1e3;
            match self.graph_cache.as_mut() {
                Some(gcache) => {
                    gcache.insert(key, compiled);
                }
                None => fresh = Some(compiled),
            }
        } else {
            self.timing.graph_hits += 1;
        }
        let c: &mut rlx_runtime::CompiledGraph = match self.graph_cache.as_mut() {
            Some(gcache) => gcache
                .get_mut(&key)
                .expect("compiled graph present after insert"),
            None => fresh
                .as_mut()
                .expect("fresh compiled graph on the no-cache path"),
        };
        let t_r = Instant::now();
        for (nm, data) in [
            ("moe.gate_codes", &gc),
            ("moe.up_codes", &uc),
            ("moe.down_codes", &dc),
        ] {
            c.set_param_typed(nm, data, DType::U8);
        }
        for (nm, data) in [
            ("moe.gate_scales", &gsc),
            ("moe.gate_biases", &gb),
            ("moe.up_scales", &usc),
            ("moe.up_biases", &ub),
            ("moe.down_scales", &dsc),
            ("moe.down_biases", &db),
        ] {
            c.set_param_typed(nm, data, DType::BF16);
        }
        let out_v = c
            .run(&[("hlat", h_lat), ("idx", &eidx), ("prob", pmask.as_slice())])
            .remove(0);
        self.timing.run_ms += t_r.elapsed().as_secs_f64() * 1e3;
        self.timing.graph_ms = self.timing.compile_ms + self.timing.run_ms;
        self.timing.calls += 1;
        Ok(out_v)
    }
}

impl ExpertProvider for KimiExpertProvider {
    fn owns(&self, e: u32) -> bool {
        self.owned.contains(&(e as usize))
    }

    fn compute(
        &mut self,
        layer: u32,
        h_lat: &[f32],
        rows: usize,
        latent: usize,
        ids: &[u32],
        probs: &[f32],
    ) -> Result<Vec<f32>> {
        let f = DType::F32;
        let (l, mi) = (self.d.latent, self.d.moe_inter);
        debug_assert_eq!(latent, l, "provider latent {l} != orchestrator {latent}");
        let k = ids.len() / rows;
        let lp = format!("language_model.model.layers.{layer}");

        // compact set of OWNED fired experts (this worker's contribution only).
        let mut uniq: Vec<usize> = ids
            .iter()
            .map(|&e| e as usize)
            .filter(|e| self.owned.contains(e))
            .collect();
        uniq.sort_unstable();
        uniq.dedup();
        if uniq.is_empty() {
            return Ok(vec![0f32; rows * l]); // owns nothing fired → zero partial
        }
        let remap: HashMap<usize, usize> = uniq.iter().enumerate().map(|(i, &e)| (e, i)).collect();
        let n = uniq.len();

        // W4A8 GPU-native path (ScaledGroupedMatMul): decode+GEMM on the GPU.
        if self.scaled && !matches!(self.device, Device::Ane) {
            return self.compute_scaled(layer, h_lat, rows, l, k, &uniq, &remap, ids, probs, &lp);
        }
        // `DequantGroupedMatMulMlx` runs on CPU/CUDA/Metal/MLX/Vulkan/wgpu + ROCm (host-
        // delegate arm); only CoreML/ANE lacks it → that worker falls back to f32.
        if self.packed && !matches!(self.device, Device::Ane) {
            return self.compute_packed(layer, h_lat, rows, l, k, &uniq, &remap, ids, probs, &lp);
        }

        // Page owned experts from local storage (f32 gate_up [L,2mi] + down [mi,L]),
        // consulting the resident cache first (hit = skip disk read + CPU dequant).
        // Was: serial `load_expert` (whole-shard mmap + f32 dequant+transpose) one expert
        // at a time — the same overhead-bound floor the packed path had, and heavier (this
        // path also dequants to f32 on the main thread). Now: resolve on-disk ranges here,
        // read+dequant+transpose each MISS on a rayon worker (concurrent fault-in, uses the
        // worker's idle cores), then assemble the two contiguous buffers in parallel.
        use rayon::prelude::*;
        use std::sync::Arc;
        let t_page = Instant::now();
        let cached: Vec<Option<Arc<(Vec<f32>, Vec<f32>)>>> = uniq
            .iter()
            .map(|&e| {
                self.cache
                    .as_ref()
                    .and_then(|c| c.get(&(layer, e)).cloned())
            })
            .collect();
        let miss: Vec<usize> = (0..n).filter(|&i| cached[i].is_none()).collect();
        let miss_ranges: Vec<_> = miss
            .iter()
            .map(|&i| self.ck.expert_ranges(&lp, uniq[i]))
            .collect::<Result<_>>()?;
        let paged: Vec<(Vec<f32>, Vec<f32>)> = miss_ranges
            .par_iter()
            .map(|r| crate::loader::load_expert_ranges(r, l, mi))
            .collect::<Result<_>>()?;
        // Fold misses back into slot order (paged is ordered by the ascending `miss`
        // indices, so a single pass consuming one paged item per cache-miss preserves
        // `uniq` order) + insert into the resident cache.
        let mut items: Vec<Arc<(Vec<f32>, Vec<f32>)>> = Vec::with_capacity(n);
        let mut paged_it = paged.into_iter();
        for (i, c) in cached.into_iter().enumerate() {
            match c {
                Some(a) => {
                    self.timing.cache_hits += 1;
                    items.push(a);
                }
                None => {
                    let a = Arc::new(paged_it.next().expect("paged/miss order"));
                    self.timing.experts_paged += 1;
                    if let Some(cc) = self.cache.as_mut() {
                        let bytes = (a.0.len() + a.1.len()) * 4;
                        if self.cache_bytes + bytes <= self.cache_budget {
                            cc.insert((layer, uniq[i]), a.clone());
                            self.cache_bytes += bytes;
                        }
                    }
                    items.push(a);
                }
            }
        }
        let (gu1, dn1) = (l * 2 * mi, mi * l);
        let mut gate_up = vec![0f32; n * gu1];
        let mut down = vec![0f32; n * dn1];
        gate_up
            .par_chunks_mut(gu1)
            .zip(items.par_iter())
            .for_each(|(s, a)| s.copy_from_slice(&a.0));
        down.par_chunks_mut(dn1)
            .zip(items.par_iter())
            .for_each(|(s, a)| s.copy_from_slice(&a.1));
        self.timing.page_ms += t_page.elapsed().as_secs_f64() * 1e3;
        // per-slot compact expert index (0 dummy for non-owned) + prob mask (0 there).
        let mut eidx = vec![0f32; rows * k];
        let mut pmask = vec![0f32; rows * k];
        for i in 0..rows * k {
            if let Some(&ci) = remap.get(&(ids[i] as usize)) {
                eidx[i] = ci as f32;
                pmask[i] = probs[i];
            }
        }

        // graph: broadcast h_lat → [rows*k, L]; grouped gate_up → situ → grouped
        // down; prob-weight; Σ over k → [rows, L]. Non-owned slots contribute 0.
        let mut hir = HirModule::new("routed_partial");
        let mut g = HirMut::new(&mut hir);
        let hlat_n = g.input("hlat", Shape::new(&[rows, l], f));
        let eidx_n = g.input("eidx", Shape::new(&[rows * k], f));
        let prob_n = g.input("prob", Shape::new(&[rows * k, 1], f));
        let mut p = HashMap::new();
        let gu_w = reg(&mut g, &mut p, "gate_up", gate_up, &[n, l, 2 * mi]);
        let dn_w = reg(&mut g, &mut p, "down", down, &[n, mi, l]);
        let zeros = reg(&mut g, &mut p, "z", vec![0f32; k], &[1, k, 1]);
        let hlat3 = g.reshape_(hlat_n, vec![rows as i64, 1, l as i64]);
        let hexp = g.add(hlat3, zeros); // [rows,k,L] broadcast
        let hexp = g.reshape_(hexp, vec![(rows * k) as i64, l as i64]);
        let gate_up_o = g.add_node(
            Op::GroupedMatMul,
            vec![hexp, gu_w, eidx_n],
            Shape::new(&[rows * k, 2 * mi], f),
        );
        let hx = situ(
            &mut g,
            gate_up_o,
            rows * k,
            mi,
            self.d.situ_beta,
            self.d.situ_linear_beta,
        );
        let down_o = g.add_node(
            Op::GroupedMatMul,
            vec![hx, dn_w, eidx_n],
            Shape::new(&[rows * k, l], f),
        );
        let weighted = g.mul(down_o, prob_n);
        let w3 = g.reshape_(weighted, vec![rows as i64, k as i64, l as i64]);
        // Reduce over k. Move k to the trailing axis first so the sum is over a
        // contiguous trailing block: rlx-rocm's Reduce only supports a trailing-axis
        // suffix (a middle-axis reduce panics). Transpose is pure data movement and k
        // still accumulates in 0..K order, so this is bit-identical on CPU/CUDA/Metal.
        let w3t = g.transpose_(w3, vec![0, 2, 1]); // [rows, L, k]
        let out = g.sum(w3t, vec![2], false); // [rows, L]
        g.set_outputs(vec![out]);
        let t_graph = Instant::now();
        let mut c = compile_built(built_from_hir(hir, p)?, self.device)?;
        let out_v = c
            .run(&[("hlat", h_lat), ("eidx", &eidx), ("prob", pmask.as_slice())])
            .remove(0);
        self.timing.graph_ms += t_graph.elapsed().as_secs_f64() * 1e3;
        self.timing.calls += 1;
        Ok(out_v)
    }
}

impl KimiExpertProvider {
    /// W4A8 GPU-native path via `Op::ScaledGroupedMatMul`: MXFP4 weights (nibble-unpacked
    /// to the op's 1-code/byte layout, raw e8m0 scales) + MXFP8-quantized activations,
    /// decoded + GEMM'd ON the GPU. Same pre-norm latent partial `Σ_k prob·expert(h_lat)`
    /// as the f32/packed paths, but the dequant runs in a GPU kernel (no host-delegate).
    #[allow(clippy::too_many_arguments)]
    fn compute_scaled(
        &mut self,
        _layer: u32,
        h_lat: &[f32],
        rows: usize,
        l: usize,
        k: usize,
        uniq: &[usize],
        remap: &HashMap<usize, usize>,
        ids: &[u32],
        probs: &[f32],
        lp: &str,
    ) -> Result<Vec<f32>> {
        use rlx_ir::quant::{ScaleLayout, ScaledFormat};
        let f = DType::F32;
        let mi = self.d.moe_inter;
        let n = uniq.len();
        let gs = 32usize;
        let (act, wgt, lay) = (
            ScaledFormat::F8E4M3,
            ScaledFormat::F4E2M1,
            ScaleLayout::mx(),
        );
        // per-slot compact eidx [rows,k] + prob [rows,k].
        let mut eidx = vec![0f32; rows * k];
        let mut pmask = vec![0f32; rows * k];
        for i in 0..rows * k {
            if let Some(&ci) = remap.get(&(ids[i] as usize)) {
                eidx[i] = ci as f32;
                pmask[i] = probs[i];
            }
        }
        // page owned experts; UNPACK weight codes (2 nibbles/byte → 1 code/byte, low nibble
        // first — matches the MXFP4 decode order), keep raw e8m0 scales.
        let unpack = |q: &[u8]| -> Vec<u8> {
            let mut v = Vec::with_capacity(q.len() * 2);
            for &b in q {
                v.push(b & 0x0f);
                v.push(b >> 4);
            }
            v
        };
        let (mut gc, mut gsc, mut uc, mut usc, mut dc, mut dsc) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        let t_page = Instant::now();
        for &e in uniq {
            let p = self.ck.load_expert_packed(lp, e)?;
            gc.extend(unpack(&p.w1_q));
            gsc.extend_from_slice(&p.w1_s);
            uc.extend(unpack(&p.w3_q));
            usc.extend_from_slice(&p.w3_s);
            dc.extend(unpack(&p.w2_q));
            dsc.extend_from_slice(&p.w2_s);
        }
        self.timing.page_ms += t_page.elapsed().as_secs_f64() * 1e3;
        self.timing.experts_paged += n as u64;

        let mut hir = HirModule::new("routed_partial_scaled");
        let mut g = HirMut::new(&mut hir);
        let hlat_n = g.input("hlat", Shape::new(&[rows, l], f));
        let idx_n = g.input("idx", Shape::new(&[rows, k], f));
        let prob_n = g.input("prob", Shape::new(&[rows, k], f));
        // unpacked codes [n,N,K] U8 + e8m0 scales [n,N,K/gs] U8 (fed post-compile).
        let gc_p = g.param("s.gate_codes", Shape::new(&[n, mi, l], DType::U8));
        let gs_p = g.param("s.gate_scales", Shape::new(&[n, mi, l / gs], DType::U8));
        let uc_p = g.param("s.up_codes", Shape::new(&[n, mi, l], DType::U8));
        let us_p = g.param("s.up_scales", Shape::new(&[n, mi, l / gs], DType::U8));
        let dc_p = g.param("s.down_codes", Shape::new(&[n, l, mi], DType::U8));
        let ds_p = g.param("s.down_scales", Shape::new(&[n, l, mi / gs], DType::U8));
        // quantize activations to MXFP8 once (h_lat is shared across all k slots).
        let (iq, is) = g.scaled_quantize(hlat_n, act, lay);
        let mut acc: Option<HirNodeId> = None;
        for ki in 0..k {
            let col = g.narrow_(idx_n, 1, ki, 1);
            let eidx_ki = g.reshape_(col, vec![rows as i64]);
            let pcol = g.narrow_(prob_n, 1, ki, 1);
            let prob_ki = g.reshape_(pcol, vec![rows as i64, 1]);
            let sgmm = |g: &mut HirMut, x, xs, codes, scales, out_n: usize| {
                g.add_node(
                    Op::ScaledGroupedMatMul {
                        lhs_format: act,
                        rhs_format: wgt,
                        scale_layout: lay,
                        has_bias: false,
                    },
                    vec![x, codes, xs, scales, eidx_ki],
                    Shape::new(&[rows, out_n], f),
                )
            };
            let gate = sgmm(&mut g, iq, is, gc_p, gs_p, mi);
            let up = sgmm(&mut g, iq, is, uc_p, us_p, mi);
            let gate_up = g.concat_(vec![gate, up], 1);
            let hx = situ(
                &mut g,
                gate_up,
                rows,
                mi,
                self.d.situ_beta,
                self.d.situ_linear_beta,
            );
            let (hxq, hxs) = g.scaled_quantize(hx, act, lay);
            let down = sgmm(&mut g, hxq, hxs, dc_p, ds_p, l);
            let weighted = g.mul(down, prob_ki);
            acc = Some(match acc {
                Some(a) => g.add(a, weighted),
                None => weighted,
            });
        }
        let out = acc.expect("top_k >= 1");
        g.set_outputs(vec![out]);
        let t_graph = Instant::now();
        let mut c = compile_built(built_from_hir(hir, HashMap::new())?, self.device)?;
        for (nm, data) in [
            ("s.gate_codes", &gc),
            ("s.gate_scales", &gsc),
            ("s.up_codes", &uc),
            ("s.up_scales", &usc),
            ("s.down_codes", &dc),
            ("s.down_scales", &dsc),
        ] {
            c.set_param_typed(nm, data, DType::U8);
        }
        let out_v = c
            .run(&[("hlat", h_lat), ("idx", &eidx), ("prob", pmask.as_slice())])
            .remove(0);
        self.timing.graph_ms += t_graph.elapsed().as_secs_f64() * 1e3;
        self.timing.calls += 1;
        Ok(out_v)
    }
}

/// **Orchestrator**: run one MoE layer's FFN via expert-parallel offload, returning
/// `[rows*hidden]` (add to the residual). Phase 1 (router + latent down-proj) and
/// the tail (norm + up-proj + shared) run HERE; the routed experts are dispatched to
/// the workers that own them (experts stay on their local storage). Byte-equivalent
/// to the local [`crate::runner::run_moe_paged`] modulo f32 accumulation order.
///
/// `local` is the OPTIONAL orchestrator-owned expert shard (config-driven): when the
/// experts don't all fit on the workers, the Mac holds a small overflow shard served
/// from `/Volumes/FOUR`. Its partial is computed HERE and added to the gathered
/// worker partials before the tail. `None` when every expert lives on a worker.
/// `shards` must map only the WORKER-owned experts (so `dispatch_experts` skips the
/// local ones); `local` owns exactly the complement.
#[allow(clippy::too_many_arguments)]
pub fn run_moe_offload(
    ck: &mut CheckpointLoader,
    transport: &dyn Transport,
    shards: &ExpertShards,
    layer: u32,
    h_in: &[f32],
    d: MoeDims,
    device: Device,
    local: Option<&mut dyn ExpertProvider>,
) -> Result<Vec<f32>> {
    let f = DType::F32;
    let (rows, hidden, l) = (d.batch * d.seq, d.hidden, d.latent);
    let lp = format!("language_model.model.layers.{layer}");
    let prefix = format!("{lp}.block_sparse_moe");
    let t_p1 = Instant::now();
    let w = ck.load_moe_dense(&lp, d)?;

    // ── phase 1: router + latent down-proj → (h_lat [rows,L], ids, probs) ──
    let mut hir = HirModule::new("moe_route");
    let mut g = HirMut::new(&mut hir);
    let h_node = g.input("h", Shape::new(&[d.batch, d.seq, hidden], f));
    let mut params = HashMap::new();
    let (h_lat, top_idx, top_probs) = build_moe_route(
        &mut g,
        &mut params,
        &prefix,
        h_node,
        &w.router,
        &w.e_score_bias,
        &w.down_latent,
        d,
    )?;
    g.set_outputs(vec![h_lat, top_idx, top_probs]);
    let a = {
        let mut c = compile_built(built_from_hir(hir, params)?, device)?;
        c.run(&[("h", h_in)])
    };
    let h_lat_v = a[0].clone();
    let ids: Vec<u32> = a[1].iter().map(|&x| x.round().max(0.0) as u32).collect();
    let probs = a[2].clone();
    let phase1_ms = t_p1.elapsed().as_secs_f64() * 1e3;

    // ── phase 2a: dispatch worker experts → gather+sum, then add the orchestrator's
    //    own (config-driven) local overflow shard partial. Both are pre-norm latent
    //    partials [rows,L], so they simply add (the nonlinear norm is in the tail).
    let t_disp = Instant::now();
    let mut routed = dispatch_experts(transport, shards, layer, &h_lat_v, rows, l, &ids, &probs)?;
    let dispatch_ms = t_disp.elapsed().as_secs_f64() * 1e3;
    let mut local_ms = 0.0;
    if let Some(lp) = local {
        let t_loc = Instant::now();
        let part = lp.compute(layer, &h_lat_v, rows, l, &ids, &probs)?;
        local_ms = t_loc.elapsed().as_secs_f64() * 1e3;
        debug_assert_eq!(part.len(), routed.len());
        for (a, b) in routed.iter_mut().zip(&part) {
            *a += *b;
        }
    }

    // ── phase 2b: tail (routed_norm + up-proj + shared) on the orchestrator ──
    let t_tail = Instant::now();
    let mut hir = HirModule::new("moe_tail");
    let mut g = HirMut::new(&mut hir);
    let h_node = g.input("h", Shape::new(&[d.batch, d.seq, hidden], f));
    let routed_node = g.input("routed", Shape::new(&[rows, l], f));
    let h2d = g.reshape_(h_node, vec![rows as i64, hidden as i64]);
    let mut params = HashMap::new();
    let out = build_moe_tail(&mut g, &mut params, &prefix, routed_node, h2d, &w, d)?;
    g.set_outputs(vec![out]);
    let mut c = compile_built(built_from_hir(hir, params)?, device)?;
    let out_v = c
        .run(&[("h", h_in), ("routed", routed.as_slice())])
        .remove(0);
    let tail_ms = t_tail.elapsed().as_secs_f64() * 1e3;
    ORCH_TIMING.with(|t| {
        let mut b = t.borrow_mut();
        b.phase1_ms += phase1_ms;
        b.dispatch_ms += dispatch_ms;
        b.local_ms += local_ms;
        b.tail_ms += tail_ms;
        b.layers += 1;
    });
    Ok(out_v)
}

// ── decode-loop integration: install a cluster context, dispatch MoE layers ──

use std::cell::RefCell;
use std::sync::Arc;

/// The config-driven cluster MoE context: how to offload routed experts. Installed
/// once before a forward (from the hostfile/shard config); the decode loop consults
/// it per MoE layer via [`try_offload`].
pub struct ClusterMoe {
    pub transport: Arc<dyn Transport>,
    pub shards: ExpertShards,
    /// The orchestrator's own (overflow) shard, served locally from `/Volumes/FOUR`.
    pub local: Option<KimiExpertProvider>,
}

thread_local! {
    static CLUSTER_MOE: RefCell<Option<ClusterMoe>> = const { RefCell::new(None) };
    static ORCH_TIMING: RefCell<OrchTiming> = const {
        RefCell::new(OrchTiming { phase1_ms: 0.0, dispatch_ms: 0.0, local_ms: 0.0, tail_ms: 0.0, layers: 0 })
    };
}

/// Drain the accumulated orchestrator MoE timing (call after generate).
pub fn orch_timing_take() -> OrchTiming {
    ORCH_TIMING.with(|t| {
        let v = *t.borrow();
        *t.borrow_mut() = OrchTiming::default();
        v
    })
}

/// Install the cluster MoE context for this thread's forward (call before generate;
/// drain with [`take_cluster_moe`] after). While set, [`try_offload`] dispatches
/// every MoE layer to the workers instead of running it locally.
pub fn install_cluster_moe(cm: ClusterMoe) {
    CLUSTER_MOE.with(|c| *c.borrow_mut() = Some(cm));
}

/// Remove and return the installed context.
pub fn take_cluster_moe() -> Option<ClusterMoe> {
    CLUSTER_MOE.with(|c| c.borrow_mut().take())
}

/// If a [`ClusterMoe`] is installed, offload this MoE layer's FFN to the workers and
/// return `Some(out)`; otherwise `None` (caller runs the local path). `layer` is the
/// GLOBAL layer index.
pub fn try_offload(
    ck: &mut CheckpointLoader,
    layer: u32,
    mn: &[f32],
    d: MoeDims,
    device: Device,
) -> Result<Option<Vec<f32>>> {
    CLUSTER_MOE.with(|c| {
        let mut b = c.borrow_mut();
        match b.as_mut() {
            Some(cm) => {
                let local = cm.local.as_mut().map(|p| p as &mut dyn ExpertProvider);
                let out =
                    run_moe_offload(ck, &*cm.transport, &cm.shards, layer, mn, d, device, local)?;
                Ok(Some(out))
            }
            None => Ok(None),
        }
    })
}
