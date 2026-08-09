// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Host-in-the-loop runners that make Kimi-K3 runnable without materializing the
//! 1.45 TB of experts: [`run_moe_paged`] runs the resident router, reads which
//! experts fired, pages ONLY those from disk (MXFP4-dequantized), and finishes
//! the routed FFN over the compact set. This is the per-layer primitive a
//! single-node streaming decode / the distributed expert-parallel runner build on.

use crate::config::KimiLinearConfig;
use crate::flow::{
    AttnDecodeIn, AttnDecodeOut, AttnWeights, FfnWeights, FlowConfig, LayerWeights,
    build_layer_decode_step, build_layer_pre_ffn,
};
use crate::kda::{KdaDims, KdaWeights, build_kda_decode_step};
use crate::loader::CheckpointLoader;
use crate::moe::{
    MoeDims, MoeWeights, build_dense_mlp, build_moe_experts_paged, build_moe_experts_paged_packed,
    build_moe_route,
};
use anyhow::Result;
use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;
use std::time::Instant;

// ── per-operation benchmark (RLX_KIMI_BENCH=1) ──────────────────────────────
// Accumulates wall-time per named step (embed / attn body compile+run / MoE
// router / expert paging / expert compute / head) across all layers, then
// [`bench_report`] prints a breakdown. For per-KERNEL timing inside a graph, set
// the backend profiler too (e.g. RLX_METAL_THUNK_PROFILE=1 / RLX_PROFILE_THUNKS=1).
thread_local! {
    static BENCH: std::cell::RefCell<Vec<(&'static str, u64, f64)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}
fn bench_on() -> bool {
    std::env::var("RLX_KIMI_BENCH").is_ok()
}
/// Add `secs` to step `cat`'s accumulator (once-per-call count).
fn bench_add(cat: &'static str, secs: f64) {
    BENCH.with(|b| {
        let mut v = b.borrow_mut();
        if let Some(e) = v.iter_mut().find(|(c, _, _)| *c == cat) {
            e.1 += 1;
            e.2 += secs;
        } else {
            v.push((cat, 1, secs));
        }
    });
}
/// Time `f`, accumulate under `cat` (if benching), return its value.
fn timed<T>(cat: &'static str, f: impl FnOnce() -> T) -> T {
    if !bench_on() {
        return f();
    }
    let t = Instant::now();
    let out = f();
    bench_add(cat, t.elapsed().as_secs_f64());
    out
}
/// Print the accumulated per-step breakdown (called once at the end).
fn bench_report() {
    if !bench_on() {
        return;
    }
    BENCH.with(|b| {
        let mut v = b.borrow().clone();
        let total: f64 = v.iter().map(|(_, _, s)| s).sum();
        v.sort_by(|a, c| c.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("\n  ── per-operation benchmark ─────────────────────────────────");
        eprintln!(
            "  {:<22} {:>7} {:>10} {:>10} {:>7}",
            "step", "calls", "total(s)", "avg(ms)", "%"
        );
        for (cat, n, s) in &v {
            eprintln!(
                "  {:<22} {:>7} {:>10.2} {:>10.2} {:>6.1}%",
                cat,
                n,
                s,
                (s / *n as f64) * 1e3,
                s / total * 100.0
            );
        }
        eprintln!("  {:<22} {:>7} {:>10.2}", "TOTAL", "", total);
    });
}

/// Run one MoE layer, dispatching to the expert-parallel cluster workers when a
/// [`crate::dist_experts::ClusterMoe`] is installed (config-driven), else locally
/// via [`run_moe_paged`]. `layer` is the global layer index.
#[cfg_attr(not(feature = "cluster"), allow(unused_variables))]
fn run_moe_layer(
    ck: &mut CheckpointLoader,
    layer_prefix: &str,
    layer: usize,
    mn: &[f32],
    d: MoeDims,
    device: Device,
) -> Result<Vec<f32>> {
    #[cfg(feature = "cluster")]
    if let Some(out) = crate::dist_experts::try_offload(ck, layer as u32, mn, d, device)? {
        return Ok(out);
    }
    run_moe_paged(ck, layer_prefix, mn, d, device)
}

/// Run one LatentMoE layer with **paged** experts on `h_in` `[rows*hidden]`.
/// `layer_prefix` e.g. `language_model.model.layers.1`. Returns `[rows*hidden]`.
/// Only the ≤ `rows*top_k` distinct routed experts are dequantized from disk.
pub fn run_moe_paged(
    ck: &mut CheckpointLoader,
    layer_prefix: &str,
    h_in: &[f32],
    d: MoeDims,
    device: Device,
) -> Result<Vec<f32>> {
    let prefix = format!("{layer_prefix}.block_sparse_moe");
    let (rows, hidden, l, mi, k) = (d.batch * d.seq, d.hidden, d.latent, d.moe_inter, d.top_k);
    let w = ck.load_moe_dense(layer_prefix, d)?; // router/bias/latent/norm/shared (dense)

    // ── phase 1: router → (h_lat, top_idx, top_probs) ──
    let mut hir = HirModule::new("moe_route");
    let mut g = HirMut::new(&mut hir);
    let h_node = g.input("h", Shape::new(&[d.batch, d.seq, hidden], DType::F32));
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
    let a = timed("moe:router", || -> Result<Vec<Vec<f32>>> {
        let built = built_from_hir(hir, params)?;
        let mut ca = compile_built(built, dev_override("RLX_KIMI_ROUTER_DEVICE", device))?;
        Ok(ca.run(&[("h", h_in)]))
    })?;
    let h_lat_v = a[0].clone();
    // In-graph f32 router by default (backend-consistent once the hybrid MLA/KDA
    // miscompile is fixed). `RLX_KIMI_HOST_ROUTER=1` forces the deterministic
    // bf16/f64 host router for extra robustness against genuine near-tie flips.
    let (top_idx_v, top_probs_v) = if std::env::var("RLX_KIMI_HOST_ROUTER").is_ok() {
        timed("moe:host_router", || {
            route_experts_host(h_in, &w.router, &w.e_score_bias, d)
        })
    } else {
        (a[1].clone(), a[2].clone())
    };

    // ── page the fired experts (dequant only the distinct ones) ──
    let ids: Vec<usize> = top_idx_v
        .iter()
        .map(|&x| x.round().max(0.0) as usize)
        .collect();
    let mut uniq = ids.clone();
    uniq.sort_unstable();
    uniq.dedup();
    if std::env::var("RLX_KIMI_LOG_EXPERTS").is_ok() {
        eprintln!("{layer_prefix}: experts {uniq:?}");
    }
    let remap: HashMap<usize, usize> = uniq.iter().enumerate().map(|(i, &e)| (e, i)).collect();
    let n_uniq = uniq.len();
    // batched-paging amortization: this call routes `rows` tokens to `n_uniq` experts.
    crate::io_opt::note_routing(d.batch * d.seq, n_uniq);
    // ROUTE-ONLY (RLX_KIMI_ROUTE_ONLY=1): skip the expensive expert paging+compute and
    // return a zero MoE contribution — for cheaply measuring routing/amortization only
    // (final logits are meaningless; use with a single MoE layer).
    if std::env::var("RLX_KIMI_ROUTE_ONLY").is_ok() {
        return Ok(vec![0.0; rows * hidden]);
    }
    let remapped: Vec<f32> = ids.iter().map(|&e| remap[&e] as f32).collect();

    // OPT #2: run the routed experts as a fused DequantGroupedMatMul{MlxMxfp4}
    // straight off the RAW MXFP4 bytes — no CPU dequant, no 4× f32 materialize, no
    // transpose, and the experts stay 4× smaller in RAM. Bit-exact vs the f32 path.
    // DEVICE-AWARE: this fused op is a GPU kernel — on **CPU it is ~30× slower** than
    // dequant→BLAS (measured), so packed is the default ONLY when experts run on a
    // GPU (the cluster), where it also saves the 4× f32 bandwidth. `RLX_KIMI_PACKED_MOE=1`
    // forces it on (e.g. CPU testing); `RLX_KIMI_NO_PACKED_MOE=1` forces it off.
    let experts_dev = dev_override("RLX_KIMI_EXPERTS_DEVICE", device);
    let packed = !std::env::var("RLX_KIMI_NO_PACKED_MOE").is_ok()
        && (std::env::var("RLX_KIMI_PACKED_MOE").is_ok() || !matches!(experts_dev, Device::Cpu));
    if packed {
        return run_moe_experts_packed(
            ck,
            &prefix,
            layer_prefix,
            h_in,
            &h_lat_v,
            &remapped,
            &top_probs_v,
            &uniq,
            &w,
            d,
            device,
        );
    }

    // Page the fired experts CONCURRENTLY: resolve on-disk ranges (main thread),
    // then read+dequant+transpose across a rayon pool so the distinct experts'
    // reads hit different SSD offsets in parallel (queue depth) — MoE paging is the
    // dominant, disk-bound cost. `RLX_KIMI_SEQ_PAGING` forces the old serial loop.
    let (gate_up, down) = timed(
        "moe:paging(disk+dequant)",
        || -> Result<(Vec<f32>, Vec<f32>)> {
            use rayon::prelude::*;
            let mut gate_up = Vec::with_capacity(n_uniq * l * 2 * mi);
            let mut down = Vec::with_capacity(n_uniq * mi * l);
            if std::env::var("RLX_KIMI_SEQ_PAGING").is_ok() {
                for &eid in &uniq {
                    let (gu, dn) = ck.load_expert(layer_prefix, eid, l, mi)?;
                    gate_up.extend(gu);
                    down.extend(dn);
                }
            } else {
                let ranges: Vec<_> = uniq
                    .iter()
                    .map(|&e| ck.expert_ranges(layer_prefix, e))
                    .collect::<Result<_>>()?;
                let loaded: Vec<(Vec<f32>, Vec<f32>)> = ranges
                    .par_iter()
                    .map(|r| crate::loader::load_expert_ranges(r, l, mi))
                    .collect::<Result<_>>()?;
                for (gu, dn) in loaded {
                    gate_up.extend(gu);
                    down.extend(dn);
                }
            }
            Ok((gate_up, down))
        },
    )?;

    // ── phase 2: routed FFN over the compact set + shared experts ──
    let mut hir = HirModule::new("moe_experts");
    let mut g = HirMut::new(&mut hir);
    let h_node = g.input("h", Shape::new(&[d.batch, d.seq, hidden], DType::F32));
    let hlat_node = g.input("hlat", Shape::new(&[rows, l], DType::F32));
    let idx_node = g.input("idx", Shape::new(&[rows, k], DType::F32));
    let prob_node = g.input("prob", Shape::new(&[rows, k], DType::F32));
    let mut params = HashMap::new();
    // bf16-resident backbone also covers the MoE DENSE matmuls (latent up-proj +
    // shared experts) — the routed experts stay MXFP4-packed. Collect them into a
    // fresh scope (the layer-body collector was already drained upstream).
    let bf16_bb = std::env::var("RLX_KIMI_BF16_BACKBONE").is_ok();
    if bf16_bb {
        crate::common::bf16_backbone_begin();
    }
    let int8_bb = crate::common::int8_backbone_requested();
    if int8_bb {
        crate::common::int8_backbone_begin();
    }
    let out = build_moe_experts_paged(
        &mut g,
        &mut params,
        &prefix,
        h_node,
        hlat_node,
        idx_node,
        prob_node,
        &gate_up,
        &down,
        n_uniq,
        &w,
        d,
    )?;
    g.set_outputs(vec![out]);
    timed("moe:experts(compute)", || -> Result<Vec<f32>> {
        let built = built_from_hir(hir, params)?;
        let mut cb = compile_built(built, dev_override("RLX_KIMI_EXPERTS_DEVICE", device))?;
        if bf16_bb {
            for (name, bytes) in crate::common::bf16_backbone_take() {
                cb.set_param_typed(&name, &bytes, DType::BF16);
            }
        }
        if int8_bb {
            for (name, bytes) in crate::common::int8_backbone_take() {
                cb.set_param_typed(&name, &bytes, DType::I8);
            }
        }
        Ok(cb
            .run(&[
                ("h", h_in),
                ("hlat", &h_lat_v),
                ("idx", &remapped),
                ("prob", &top_probs_v),
            ])
            .remove(0))
    })
}

/// OPT #2 expert FFN: stack the fired experts' RAW MXFP4 bytes (codes as-is,
/// E8M0 scales → `bf16(e<<7)`, zero biases — Kimi's format IS `MlxMxfp4`), build
/// the [`build_moe_experts_paged_packed`] graph, and feed the packed weights via
/// `set_param_typed` so the GPU does the fused dequant+matmul. No CPU dequant.
#[allow(clippy::too_many_arguments)]
fn run_moe_experts_packed(
    ck: &mut CheckpointLoader,
    prefix: &str,
    layer_prefix: &str,
    h_in: &[f32],
    h_lat_v: &[f32],
    remapped: &[f32],
    top_probs_v: &[f32],
    uniq: &[usize],
    w: &MoeWeights,
    d: MoeDims,
    device: Device,
) -> Result<Vec<f32>> {
    let (rows, hidden, l, k) = (d.batch * d.seq, d.hidden, d.latent, d.top_k);
    let e8m0_bf16 = |s: &[u8]| -> Vec<u8> {
        s.iter()
            .flat_map(|&b| half::bf16::from_bits((b as u16) << 7).to_le_bytes())
            .collect()
    };
    // stack the compact experts' packed codes (raw) + bf16 scales
    let (mut gc, mut gsc, mut uc, mut usc, mut dc, mut dsc) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    // Page the fired experts' RAW packed bytes CONCURRENTLY (resolve ranges on the
    // main thread, then read across a rayon pool — same queue-depth win as the f32
    // path, but ~4× less data). `RLX_KIMI_SEQ_PAGING` forces the serial loop.
    timed("moe:paging(packed)", || -> Result<()> {
        use rayon::prelude::*;
        use std::sync::Arc;
        type Ep = crate::loader::ExpertPacked;
        // 1. Page the fired experts → `loaded` (Arc<ExpertPacked>, in `uniq` order).
        let t = std::time::Instant::now();
        let loaded: Vec<Arc<Ep>> = if crate::io_opt::io_opt_active() {
            // OPT #1: persistent expert cache — re-fired experts served from RAM
            // (RLX_KIMI_EXPERT_CACHE=<MB>); misses paged concurrently, then inserted.
            let cached: Vec<Option<Arc<Ep>>> = uniq
                .iter()
                .map(|&e| crate::io_opt::cache_get(layer_prefix, e))
                .collect();
            let miss: Vec<usize> = (0..uniq.len()).filter(|&i| cached[i].is_none()).collect();
            let miss_ranges: Vec<_> = miss
                .iter()
                .map(|&i| ck.expert_ranges(layer_prefix, uniq[i]))
                .collect::<Result<_>>()?;
            let paged: Vec<Ep> = miss_ranges
                .par_iter()
                .map(crate::loader::load_expert_ranges_packed)
                .collect::<Result<_>>()?;
            let mut items = cached;
            for (j, p) in paged.into_iter().enumerate() {
                let i = miss[j];
                let a = Arc::new(p);
                crate::io_opt::cache_put(layer_prefix, uniq[i], a.clone());
                items[i] = Some(a);
            }
            items
                .into_iter()
                .map(|it| it.expect("expert missing after paging"))
                .collect()
        } else if std::env::var("RLX_KIMI_SEQ_PAGING").is_ok() {
            uniq.iter()
                .map(|&e| ck.load_expert_packed(layer_prefix, e).map(Arc::new))
                .collect::<Result<_>>()?
        } else {
            let ranges: Vec<_> = uniq
                .iter()
                .map(|&e| ck.expert_ranges(layer_prefix, e))
                .collect::<Result<_>>()?;
            ranges
                .par_iter()
                .map(|r| crate::loader::load_expert_ranges_packed(r).map(Arc::new))
                .collect::<Result<_>>()?
        };
        let t_read = t.elapsed().as_secs_f64();
        // 2. Assemble the contiguous GPU buffers IN PARALLEL. The old serial `extend`
        // loop was the measured 85–90% of paging (assembling ~1 GB of codes single-
        // threaded). All experts share code/scale sizes → pre-size once and fill each
        // expert's slot concurrently (copy codes, convert e8m0→bf16 scales).
        let t = std::time::Instant::now();
        let n = loaded.len();
        if n > 0 {
            let e0 = &loaded[0];
            let (g1, u1, d1) = (e0.w1_q.len(), e0.w3_q.len(), e0.w2_q.len());
            let (gs, us, ds) = (e0.w1_s.len() * 2, e0.w3_s.len() * 2, e0.w2_s.len() * 2);
            gc = vec![0u8; n * g1];
            uc = vec![0u8; n * u1];
            dc = vec![0u8; n * d1];
            gsc = vec![0u8; n * gs];
            usc = vec![0u8; n * us];
            dsc = vec![0u8; n * ds];
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
        let t_push = t.elapsed().as_secs_f64();
        if std::env::var("RLX_KIMI_PAGING_DIAG").is_ok() {
            eprintln!(
                "[paging] {n} experts: READ {:.0}ms + ASSEMBLY(par) {:.0}ms",
                t_read * 1e3,
                t_push * 1e3
            );
        }
        Ok(())
    })?;
    let (gb, ub, db) = (
        vec![0u8; gsc.len()],
        vec![0u8; usc.len()],
        vec![0u8; dsc.len()],
    );
    let n_uniq = uniq.len();

    let mut hir = HirModule::new("moe_experts_packed");
    let mut g = HirMut::new(&mut hir);
    let h_node = g.input("h", Shape::new(&[d.batch, d.seq, hidden], DType::F32));
    let hlat_node = g.input("hlat", Shape::new(&[rows, l], DType::F32));
    let idx_node = g.input("idx", Shape::new(&[rows, k], DType::F32));
    let prob_node = g.input("prob", Shape::new(&[rows, k], DType::F32));
    let mut params = HashMap::new();
    let out = build_moe_experts_paged_packed(
        &mut g,
        &mut params,
        prefix,
        h_node,
        hlat_node,
        idx_node,
        prob_node,
        n_uniq,
        w,
        d,
    )?;
    g.set_outputs(vec![out]);
    timed("moe:experts(packed-gpu)", || -> Result<Vec<f32>> {
        let diag = std::env::var("RLX_KIMI_PAGING_DIAG").is_ok();
        let t = std::time::Instant::now();
        let built = built_from_hir(hir, params)?;
        let mut cb = compile_built(built, dev_override("RLX_KIMI_EXPERTS_DEVICE", device))?;
        let t_compile = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        for (n, data) in [
            ("moe.gate_codes", &gc),
            ("moe.up_codes", &uc),
            ("moe.down_codes", &dc),
        ] {
            cb.set_param_typed(n, data, DType::U8);
        }
        let t_codes = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        for (n, data) in [
            ("moe.gate_scales", &gsc),
            ("moe.gate_biases", &gb),
            ("moe.up_scales", &usc),
            ("moe.up_biases", &ub),
            ("moe.down_scales", &dsc),
            ("moe.down_biases", &db),
        ] {
            cb.set_param_typed(n, data, DType::BF16);
        }
        let t_scales = t.elapsed().as_secs_f64();
        // DIAGNOSTIC (RLX_KIMI_PAGING_DIAG): does the payoff of a persistent buffer
        // exist? (1) does set_param_range return true → codes are ARENA-resident
        // (zero-copy, shared) vs false → private weight-buffer (blit); (2) is an
        // IN-PLACE range write (same data, existing buffer, no alloc/setup) much
        // faster than the full set_param above → the cost is alloc/setup (buffer
        // reuse = path A wins) vs ~equal → the write itself dominates (path B needed).
        let (arena_resident, t_range) = if diag {
            let t = std::time::Instant::now();
            let ok = cb.set_param_range("moe.gate_codes", 0, &gc);
            (ok, t.elapsed().as_secs_f64())
        } else {
            (false, 0.0)
        };
        let t = std::time::Instant::now();
        let out = cb
            .run(&[
                ("h", h_in),
                ("hlat", h_lat_v),
                ("idx", remapped),
                ("prob", top_probs_v),
            ])
            .remove(0);
        if diag {
            eprintln!(
                "[experts] compile {:.0}ms | codes(U8 {:.0}MB) {:.0}ms + scales(bf16) {:.0}ms | run {:.0}ms",
                t_compile * 1e3,
                gc.len() as f64 / 1e6,
                t_codes * 1e3,
                t_scales * 1e3,
                t.elapsed().as_secs_f64() * 1e3
            );
            eprintln!(
                "[experts-diag] gate_codes {:.0}MB: arena_resident(zero-copy)={} | in-place range-write {:.0}ms  (full gate_codes ≈ {:.0}ms)",
                gc.len() as f64 / 1e6,
                arena_resident,
                t_range * 1e3,
                t_codes * 1e3 / 3.0
            );
        }
        Ok(out)
    })
}

/// Add two host vectors elementwise.
fn add_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

// ── KDA decode (O(1) per token) ─────────────────────────────────────────────

/// The O(1) recurrent state one KDA layer carries between decode steps: the
/// short causal-conv left context (q/k/v, each `[(kk-1)·proj]`) plus the
/// gated-delta-net scan state `[h·hd·hd]`. This replaces re-streaming the whole
/// prefix each token for Kimi-K3's 69 KDA layers (`O(seq)` → `O(1)`).
#[derive(Clone)]
pub struct KdaState {
    pub conv_q: Vec<f32>,
    pub conv_k: Vec<f32>,
    pub conv_v: Vec<f32>,
    pub scan: Vec<f32>,
}

impl KdaState {
    /// Zeroed state — the start of a sequence (no left context, empty scan).
    pub fn zeros(d: KdaDims) -> Self {
        let cs = (d.conv_kernel - 1) * d.proj();
        Self {
            conv_q: vec![0.0; cs],
            conv_k: vec![0.0; cs],
            conv_v: vec![0.0; cs],
            scan: vec![0.0; d.num_heads * d.head_dim * d.head_dim],
        }
    }
}

/// Run ONE KDA-attention **decode step** for `seq` new tokens (`seq == 1` = pure
/// per-token decode), resuming from `state` and returning `(attn_out [seq*hidden],
/// next_state)`. Wraps [`build_kda_decode_step`] and threads the conv + scan state
/// host-side — the O(1) decode primitive the 69 KDA layers use during generation
/// (verified bit-exact vs the `build_kda_layer` prefill in `kda_decode_step`).
pub fn run_kda_decode_step(
    kw: &KdaWeights,
    h_in: &[f32],
    seq: usize,
    state: &KdaState,
    d: KdaDims,
    device: Device,
) -> Result<(Vec<f32>, KdaState)> {
    let (b, hidden, h, hd) = (d.batch, d.hidden, d.num_heads, d.head_dim);
    let (proj, kk) = (d.proj(), d.conv_kernel);
    let dd = KdaDims { seq, ..d };
    let mut hir = HirModule::new("kda_decode");
    let mut g = HirMut::new(&mut hir);
    let h_node = g.input("h", Shape::new(&[b, seq, hidden], DType::F32));
    let csq = g.input("csq", Shape::new(&[b, kk - 1, proj], DType::F32));
    let csk = g.input("csk", Shape::new(&[b, kk - 1, proj], DType::F32));
    let csv = g.input("csv", Shape::new(&[b, kk - 1, proj], DType::F32));
    let st = g.input("state", Shape::new(&[b, h, hd, hd], DType::F32));
    let mut params = HashMap::new();
    let (out, ncq, nck, ncv) = build_kda_decode_step(
        &mut g,
        &mut params,
        "kda",
        h_node,
        csq,
        csk,
        csv,
        st,
        kw,
        dd,
    )?;
    // `state` is written back in place by the carry op — carry it as an output.
    g.set_outputs(vec![out, ncq, nck, ncv, st]);
    let built = built_from_hir(hir, params)?;
    let mut c = compile_built(built, device)?;
    let mut r = c.run(&[
        ("h", h_in),
        ("csq", &state.conv_q),
        ("csk", &state.conv_k),
        ("csv", &state.conv_v),
        ("state", &state.scan),
    ]);
    let out_v = r.remove(0);
    let next = KdaState {
        conv_q: r.remove(0),
        conv_k: r.remove(0),
        conv_v: r.remove(0),
        scan: r.remove(0),
    };
    Ok((out_v, next))
}

/// **Deterministic, backend-independent MoE router.** The expert top-k is a
/// DISCRETE decision, so the ~5e-6 f32 reduction-order difference between CPU and
/// GPU tips near-tied experts differently → the two backends select different
/// experts from ~layer 3 on and generate different tokens ("f32 drift"). The model
/// is bf16-trained, so routing is meant to be a bf16-precision decision: rounding
/// the hidden state to bf16 (relative step ~4e-3) collapses the 5e-6 cross-backend
/// noise, and evaluating the router in ordered f64 on the host makes the decision
/// bit-identical on every backend AND faithful to the reference's routing
/// precision. Replicates `llada2_gate` semantics for `n_group == 1` (Kimi):
/// top-k by `route = sigmoid(logits)+bias` (stable tie-break → lower index), with
/// `prob = sigmoid[picked] / Σsigmoid[picked] · routed_scaling`.
/// Returns `(top_idx [rows*k] as f32, top_probs [rows*k])`.
fn route_experts_host(
    h_in: &[f32],
    router_w: &[f32],
    e_bias: &[f32],
    d: MoeDims,
) -> (Vec<f32>, Vec<f32>) {
    use rayon::prelude::*;
    let (rows, hidden, e, k) = (d.batch * d.seq, d.hidden, d.num_experts, d.top_k);
    let (mut top_idx, mut top_probs) = (Vec::with_capacity(rows * k), Vec::with_capacity(rows * k));
    for t in 0..rows {
        let h = &h_in[t * hidden..(t + 1) * hidden];
        // bf16-round the hidden state → same value on every backend (the 5e-6
        // f32 divergence is far below bf16's step), so routing can't diverge.
        let hb: Vec<f32> = h
            .iter()
            .map(|&x| half::bf16::from_f32(x).to_f32())
            .collect();
        // logit[j] = Σ_i hb[i]·W[i,j] in ordered f64; sig=σ(logit); route=sig+bias
        let scored: Vec<(f64, f64)> = (0..e)
            .into_par_iter()
            .map(|j| {
                let mut acc = 0f64;
                for i in 0..hidden {
                    acc += hb[i] as f64 * router_w[i * e + j] as f64;
                }
                let s = 1.0 / (1.0 + (-acc).exp());
                (s, s + e_bias[j] as f64)
            })
            .collect();
        // top-k by route, stable (ties → lower index), matching group_limited_topk
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_by(|&a, &b| {
            scored[b]
                .1
                .partial_cmp(&scored[a].1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let picked = &order[..k];
        let sum: f64 = picked.iter().map(|&ei| scored[ei].0).sum::<f64>() + 1e-20;
        let norm = if k > 1 { 1.0 / sum } else { 1.0 };
        for &ei in picked {
            top_idx.push(ei as f32);
            top_probs.push((scored[ei].0 * norm * d.routed_scaling as f64) as f32);
        }
    }
    (top_idx, top_probs)
}

/// Load one layer's **backbone** weights (attention + the 6 norm/proj vectors +
/// dense FFN, or a MoE placeholder). This is the disk-read + dequant/transpose
/// work per layer — run AHEAD on a producer thread by [`run_layer_range_streaming`]
/// so it overlaps the previous layer's compute. Kept a free fn so the producer
/// thread can call it with its OWN [`CheckpointLoader`].
pub fn load_layer_backbone(
    ck: &mut CheckpointLoader,
    tc: &KimiLinearConfig,
    cfg: &FlowConfig,
    i: usize,
) -> Result<LayerWeights> {
    let lp = format!("language_model.model.layers.{i}");
    let attn = if tc.is_kda_layer(i) {
        AttnWeights::Kda(Box::new(ck.load_kda(&lp, cfg.kda)?))
    } else {
        AttnWeights::Mla(Box::new(ck.load_mla(&lp, cfg.mla)?))
    };
    let ffn = if tc.is_moe_layer(i) {
        FfnWeights::Dense(Box::default()) // paged in run_moe_paged
    } else {
        FfnWeights::Dense(Box::new(ck.load_dense_mlp(
            &lp,
            cfg.hidden,
            cfg.dense_inter,
        )?))
    };
    Ok(LayerWeights {
        input_ln: ck.tensor_f32(&format!("{lp}.input_layernorm.weight"))?,
        post_ln: ck.tensor_f32(&format!("{lp}.post_attention_layernorm.weight"))?,
        sa_res_norm: ck.tensor_f32(&format!("{lp}.self_attention_res_norm.weight"))?,
        sa_res_proj: ck.tensor_f32(&format!("{lp}.self_attention_res_proj.weight"))?,
        mlp_res_norm: ck.tensor_f32(&format!("{lp}.mlp_res_norm.weight"))?,
        mlp_res_proj: ck.tensor_f32(&format!("{lp}.mlp_res_proj.weight"))?,
        attn,
        ffn,
    })
}

/// In-RAM cache of per-layer backbone [`LayerWeights`], keyed by GLOBAL layer
/// index: load a layer's backbone from disk ONCE and reuse it across every decode
/// token. This is the "resident weights" half of O(1) decode — the [`DecodeState`]
/// (KDA conv/scan + MLA KV) is the other half. Sized for a node whose layer-range
/// fits RAM (the cluster premise). `cap` bounds how many layers are held (env
/// `RLX_KIMI_WEIGHT_CACHE_LAYERS`, default unbounded); layers past the cap always
/// stream fresh, so the cached set is the STABLE first-`cap` layers of the range.
/// `RLX_KIMI_NO_WEIGHT_CACHE` disables it (streaming every token, the old path).
/// MoE expert weights are NOT cached here — they are paged per token in
/// [`run_moe_paged`] (1.45 TB, never resident); only the compact backbone is.
pub struct LayerCache {
    map: HashMap<usize, LayerWeights>,
    /// Compiled seq==1 decode-step graphs for SHAPE-INVARIANT (KDA) layers, keyed
    /// by global layer index → `(graph, n_snaps_out)`. The layer weights are baked
    /// into the graph, so those layers are kept HERE and NOT in `map` (caching both
    /// would double their RAM). Env `RLX_KIMI_NO_GRAPH_CACHE` disables this.
    graphs: HashMap<usize, (CompiledGraph, usize)>,
    graphs_enabled: bool,
    cap: usize,
    enabled: bool,
}

impl LayerCache {
    /// Cache configured from the environment (the worker / generation default).
    pub fn from_env() -> Self {
        let enabled = std::env::var("RLX_KIMI_NO_WEIGHT_CACHE").is_err();
        let cap = std::env::var("RLX_KIMI_WEIGHT_CACHE_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(usize::MAX);
        Self {
            map: HashMap::new(),
            graphs: HashMap::new(),
            graphs_enabled: std::env::var("RLX_KIMI_NO_GRAPH_CACHE").is_err(),
            cap,
            enabled,
        }
    }
    /// A no-op cache for single-forward callers that never reuse across tokens.
    pub fn disabled() -> Self {
        Self {
            map: HashMap::new(),
            graphs: HashMap::new(),
            graphs_enabled: false,
            cap: 0,
            enabled: false,
        }
    }
    /// Is the compiled-graph cache active AND does layer `i` fit the weight cap
    /// (graphs bake in weights, so they consume the same layer budget)?
    fn graphs_on(&self, i: usize) -> bool {
        self.graphs_enabled
            && self.enabled
            && (self.graphs.contains_key(&i) || self.map.len() + self.graphs.len() < self.cap)
    }
    /// Number of layers currently resident.
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    /// Should layer `i` be admitted (enabled, and either already in or under cap)?
    fn admits(&self, i: usize) -> bool {
        self.enabled && (self.map.contains_key(&i) || self.map.len() < self.cap)
    }
}

/// Per-graph device override from an env var (`cpu`/`metal`/`mlx`/`gpu`), else
/// `default`. Used to bisect backend bugs (e.g. run the MoE router on CPU while
/// the rest runs on Metal) without touching the call sites.
fn dev_override(key: &str, default: Device) -> Device {
    match std::env::var(key).ok().as_deref() {
        Some("cpu") => Device::Cpu,
        Some("metal") | Some("mtl") => Device::Metal,
        Some("mlx") => Device::Mlx,
        Some("gpu") | Some("wgpu") => Device::Gpu,
        _ => default,
    }
}

/// **Single-node STREAMING forward** through the first `n_layers` REAL layers:
/// embed → for each layer, load its backbone from disk, run attention + AttnRes,
/// then the FFN (dense in-graph, or MoE **paged** via [`run_moe_paged`]), freeing
/// each layer's weights before the next. Threads the hidden state + AttnRes
/// snapshots host-side. Peak RAM ≈ one layer, so it runs a model far larger than
/// RAM (disk-bound). Returns `(hidden [seq*hidden], attn_res_snapshots)` after
/// `n_layers` — the snapshots feed the head ([`run_prefix_logits`]).
pub fn run_prefix_streaming(
    ck: &mut CheckpointLoader,
    tc: &KimiLinearConfig,
    cfg: &FlowConfig,
    tokens: &[u32],
    n_layers: usize,
    device: Device,
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let h = timed("embed(gather)", || {
        ck.gather_embed(
            "language_model.model.embed_tokens.weight",
            tokens,
            cfg.hidden,
        )
    })?;
    run_layer_range_streaming(ck, tc, cfg, h, Vec::new(), 0, n_layers, device)
}

/// Stream a **layer RANGE** `[start, end)` from `(hidden_in, snapshots_in)` →
/// `(hidden_out, snapshots_out)`. This is the per-NODE slice a distributed
/// pipeline runs (each node owns a contiguous range + its shards on local disk);
/// the boundary state — hidden + all AttnRes snapshots — is what crosses between
/// nodes. `[0, n_layers)` from the embedding is [`run_prefix_streaming`].
///
/// **Pipelined I/O:** a producer thread (its own [`CheckpointLoader`]) loads and
/// dequantizes each layer's backbone AHEAD into a bounded channel, so the full
/// disk-read + dequant/transpose cost overlaps the previous layer's compute — not
/// just the page fault. Set `RLX_KIMI_NO_PIPELINE` to disable.
#[allow(clippy::too_many_arguments)]
pub fn run_layer_range_streaming(
    ck: &mut CheckpointLoader,
    tc: &KimiLinearConfig,
    cfg: &FlowConfig,
    mut h: Vec<f32>,
    mut snaps: Vec<Vec<f32>>,
    start: usize,
    end: usize,
    device: Device,
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let (hidden, seq) = (cfg.hidden, cfg.seq);
    let profile = std::env::var("RLX_KIMI_PROFILE").is_ok();
    // The producer overlaps CPU dequant with compute — a WIN only when compute is
    // off-CPU (Metal/MLX/GPU); on the CPU device the producer's rayon dequant
    // steals cores from the main compute and it's a net loss. Gate to GPU devices
    // (override with RLX_KIMI_PIPELINE=1/0).
    let pipeline_on = match std::env::var("RLX_KIMI_PIPELINE").ok().as_deref() {
        Some("1") => true,
        Some("0") => false,
        _ => !matches!(device, Device::Cpu) && std::env::var("RLX_KIMI_NO_PIPELINE").is_err(),
    };
    let (mut t_load, mut t_compute) = (0f64, 0f64);

    // Producer pipeline: a background thread with its OWN CheckpointLoader loads
    // (disk + dequant + transpose) each layer's backbone AHEAD into a bounded
    // channel, so the load fully overlaps the previous layer's compute. The main
    // thread keeps `ck` only for the routing-dependent MoE expert paging.
    //
    // Prefetch DEPTH is bounded by the shared [`ResourceBudget`] RAM ceiling
    // (`RLX_MAX_RAM_BYTES`, else physical-RAM soft budget): each in-flight backbone
    // is ~`per_layer_bytes`, and we reserve ~2 layers for the compute + producer
    // working set, so a small machine keeps 1 in flight and a big one prefetches
    // deeper — never blowing RAM.
    let depth = {
        let proj = cfg.kda.num_heads * cfg.kda.head_dim; // KDA q/k/v/g/o dominate
        let per_layer_bytes = 5 * cfg.hidden * proj * 4;
        match rlx_runtime::resource_budget::ResourceBudget::from_env().effective_ram_bytes() {
            Some(ram) => (ram / per_layer_bytes.max(1)).saturating_sub(2).clamp(1, 4),
            None => 2,
        }
    };
    if profile {
        eprintln!("  pipeline prefetch depth = {depth} (ResourceBudget RAM ceiling)");
    }
    #[allow(clippy::type_complexity)]
    let producer: Option<std::sync::mpsc::Receiver<Result<LayerWeights>>> = if pipeline_on {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<LayerWeights>>(depth);
        let dir = ck.dir().to_path_buf();
        let (tc, cfg) = (tc.clone(), cfg.clone());
        std::thread::spawn(move || match CheckpointLoader::open(&dir) {
            Ok(mut ck2) => {
                for i in start..end {
                    let lw = load_layer_backbone(&mut ck2, &tc, &cfg, i);
                    let err = lw.is_err();
                    if tx.send(lw).is_err() || err {
                        break;
                    }
                }
            }
            Err(e) => {
                let _ = tx.send(Err(e));
            }
        });
        Some(rx)
    } else {
        None
    };

    for i in start..end {
        let is_moe = tc.is_moe_layer(i);
        let lp = format!("language_model.model.layers.{i}");
        // per-layer quant context (drives RLX_KIMI_QUANT=adaptive's depth schedule).
        crate::common::set_quant_layer(Some((i, tc.num_hidden_layers)));
        let t_layer = Instant::now();

        // backbone: from the producer (already loaded + dequantized) or inline.
        let lw = match &producer {
            Some(rx) => rx
                .recv()
                .map_err(|_| anyhow::anyhow!("weight producer died at layer {i}"))??,
            None => load_layer_backbone(ck, tc, cfg, i)?,
        };
        let load_s = t_layer.elapsed().as_secs_f64();
        t_load += load_s;
        if bench_on() {
            bench_add("backbone:load(wait)", load_s);
        }
        let t_c = Instant::now();
        // Per-graph device (overridable to bisect backend bugs). The KDA-bearing
        // layer BODY currently miscompiles on Metal (finite but wrong token) for
        // layers with incoming AttnRes snapshots — set `RLX_KIMI_BODY_DEVICE=cpu`
        // to run the body on CPU while the MoE (correct on Metal) stays on GPU.
        let body_dev = dev_override("RLX_KIMI_BODY_DEVICE", device);

        // per-layer graph: pre-FFN body (+ dense FFN in-graph). MoE outputs the
        // FFN input + stream so the host can page + add.
        let int8_bb = crate::common::int8_backbone_requested();
        if int8_bb {
            crate::common::int8_backbone_begin();
        }
        let mut hir = HirModule::new("layer");
        let mut g = HirMut::new(&mut hir);
        let h_node = g.input("h", Shape::new(&[1, seq, hidden], DType::F32));
        let snap_nodes: Vec<_> = (0..snaps.len())
            .map(|j| {
                g.input(
                    format!("snap_{j}"),
                    Shape::new(&[1, seq, hidden], DType::F32),
                )
            })
            .collect();
        let mut params = HashMap::new();
        let (mn, stream, snaps_out) =
            build_layer_pre_ffn(&mut g, &mut params, i, h_node, snap_nodes, &lw, cfg)?;
        let mut outputs = Vec::new();
        if is_moe {
            outputs.push(mn);
            outputs.push(stream);
        } else {
            let FfnWeights::Dense(dw) = &lw.ffn else {
                unreachable!()
            };
            let ffn = build_dense_mlp(
                &mut g,
                &mut params,
                &format!("l{i}.mlp"),
                mn,
                dw,
                hidden,
                cfg.dense_inter,
                1,
                seq,
                cfg.situ_beta,
                cfg.situ_linear_beta,
            )?;
            let h_out = g.add(stream, ffn);
            outputs.push(h_out);
        }
        outputs.extend(snaps_out);
        g.set_outputs(outputs);
        let mut compiled = timed("layer:body(compile)", || -> Result<_> {
            let built = built_from_hir(hir, params)?;
            compile_built(built, body_dev)
        })?;
        if int8_bb {
            for (name, bytes) in crate::common::int8_backbone_take() {
                compiled.set_param_typed(&name, &bytes, DType::I8);
            }
        }

        // bind h + carried snapshots
        let snap_names: Vec<String> = (0..snaps.len()).map(|j| format!("snap_{j}")).collect();
        let mut inputs: Vec<(&str, &[f32])> = vec![("h", h.as_slice())];
        for (j, sn) in snaps.iter().enumerate() {
            inputs.push((snap_names[j].as_str(), sn.as_slice()));
        }
        let mut res = timed("layer:body(run)", || compiled.run(&inputs));

        if is_moe {
            let mn_v = res.remove(0);
            let stream_v = res.remove(0);
            snaps = res; // remaining outputs are the snapshots
            let moe = run_moe_layer(ck, &lp, i, &mn_v, cfg.moe, device)?;
            h = add_vec(&stream_v, &moe);
        } else {
            h = res.remove(0);
            snaps = res;
        }
        let comp_s = t_c.elapsed().as_secs_f64();
        t_compute += comp_s;
        if profile {
            eprintln!(
                "  layer {i:2} {}{}: load {load_s:5.1}s + compute {comp_s:5.1}s",
                if tc.is_kda_layer(i) { "KDA" } else { "MLA" },
                if is_moe { "+MoE" } else { "+dense" }
            );
        }
    }
    crate::common::set_quant_layer(None);
    if profile {
        eprintln!(
            "  streaming totals: load {t_load:.1}s (backbone wait; pipeline {}) + compute {t_compute:.1}s over layers {start}..{end}",
            if pipeline_on { "on" } else { "off" }
        );
    }
    Ok((h, snaps))
}

/// Full streaming forward → **logits** `[seq*vocab]`: run [`run_prefix_streaming`]
/// over `n_layers`, then load + apply the head (out-residual + final norm + untied
/// `lm_head`). With `n_layers == num_hidden_layers` this is a real next-token
/// forward of the whole model. `argmax` of the last row is the greedy next token.
pub fn run_prefix_logits(
    ck: &mut CheckpointLoader,
    tc: &KimiLinearConfig,
    cfg: &FlowConfig,
    tokens: &[u32],
    n_layers: usize,
    device: Device,
) -> Result<Vec<f32>> {
    let (h, snaps) = run_prefix_streaming(ck, tc, cfg, tokens, n_layers, device)?;
    apply_head(ck, cfg, &h, &snaps, device)
}

/// Resident compiled **head** graph (out-residual + final norm + untied lm_head,
/// with the ~4.7 GB `lm_head.weight` baked in). `apply_head` re-loads that 4.7 GB
/// from disk AND re-lowers the graph every call — the dominant per-token cost once
/// the backbone is resident. For seq==1 decode the head graph is shape-invariant,
/// so we compile it ONCE and reuse it every token (keyed by the snapshot count, so
/// a differing depth — e.g. a speculative draft — rebuilds). `RLX_KIMI_NO_HEAD_CACHE`
/// disables it; `disabled()` is the one-shot default for callers that don't loop.
#[derive(Default)]
pub struct HeadCache {
    graph: Option<(usize, CompiledGraph)>, // (n_snaps, compiled seq==1 head)
    enabled: bool,
}

impl HeadCache {
    pub fn from_env() -> Self {
        Self {
            graph: None,
            enabled: std::env::var("RLX_KIMI_NO_HEAD_CACHE").is_err(),
        }
    }
    pub fn disabled() -> Self {
        Self {
            graph: None,
            enabled: false,
        }
    }
}

/// Build + compile the head graph, baking in the freshly-loaded head weights
/// (final norm, out-res norm/proj, untied lm_head). `n_snaps` fixes the incoming
/// AttnRes snapshot count; the graph is otherwise seq-shaped by `cfg`.
fn build_head_graph(
    ck: &mut CheckpointLoader,
    cfg: &FlowConfig,
    n_snaps: usize,
    device: Device,
) -> Result<CompiledGraph> {
    let (hidden, seq) = (cfg.hidden, cfg.seq);
    // bf16-resident head (opt-in): keep lm_head BF16 ([hidden,vocab], 2.35 GB) and
    // drive the CPU dequant-on-the-fly SgemmBf16 — ~2× the head GEMV. Precision-
    // approximate (bf16 weights) → verify tokens before relying on it.
    let bf16_head = std::env::var("RLX_KIMI_BF16_HEAD").is_ok();
    // int8-resident head (the 2.35 GB lm_head is the single biggest per-token load):
    // route it through `emit_int8_resident` like the layers so its codes are
    // recorded / mmapped by the prequant path. `ck.lin` returns empty in LOAD mode
    // (the codes are mmapped by name) → the 8.8 s bf16 head:load disappears.
    let int8_head = crate::common::int8_backbone_requested() && !bf16_head;
    let (final_norm, out_res_norm, out_res_proj, lm_head, lm_bf16) =
        timed("head:load(lm_head+norms)", || -> Result<_> {
            let lm_bf16 = if bf16_head {
                Some(ck.lm_head_bf16_kn("language_model.lm_head.weight", cfg.vocab, hidden)?)
            } else {
                None
            };
            let lm_f32 = if bf16_head {
                Vec::new()
            } else {
                ck.lin("language_model.lm_head.weight", cfg.vocab, hidden)?
            };
            Ok((
                ck.tensor_f32("language_model.model.norm.weight")?,
                ck.tensor_f32("language_model.model.output_attn_res_norm.weight")?,
                ck.tensor_f32("language_model.model.output_attn_res_proj.weight")?,
                lm_f32,
                lm_bf16,
            ))
        })?;
    if int8_head {
        crate::common::int8_backbone_begin();
    }
    let mut hir = HirModule::new("head");
    let mut g = HirMut::new(&mut hir);
    let h_node = g.input("h", Shape::new(&[1, seq, hidden], DType::F32));
    let snap_nodes: Vec<_> = (0..n_snaps)
        .map(|j| {
            g.input(
                format!("snap_{j}"),
                Shape::new(&[1, seq, hidden], DType::F32),
            )
        })
        .collect();
    let mut params = HashMap::new();
    let logits = crate::flow::build_head(
        &mut g,
        &mut params,
        h_node,
        &snap_nodes,
        &final_norm,
        &out_res_norm,
        &out_res_proj,
        &lm_head,
        bf16_head,
        cfg,
    );
    g.set_outputs(vec![logits]);
    timed("head:compile", || -> Result<_> {
        let built = built_from_hir(hir, params)?;
        // The head is a big dense MatMul (`[seq,hidden]@[hidden,vocab]`) — offloadable to
        // ANE/CoreML (`RLX_KIMI_HEAD_DEVICE=ane`) to free the CPU for the sequential KDA
        // scan, or to Metal/MLX. Defaults to the body device.
        let mut compiled = compile_built(built, dev_override("RLX_KIMI_HEAD_DEVICE", device))?;
        // feed the BF16 lm_head bytes into the compiled (cached) graph, once.
        if let Some(bytes) = &lm_bf16 {
            compiled.set_param_typed("lm_head.weight", bytes, DType::BF16);
        }
        // int8-resident head: feed the recorded/quantized lm_head codes (I8).
        if int8_head {
            for (name, bytes) in crate::common::int8_backbone_take() {
                compiled.set_param_typed(&name, &bytes, DType::I8);
            }
        }
        Ok(compiled)
    })
}

/// Run a (possibly cached) compiled head graph on `(h, snaps)` → logits.
fn run_head_graph(compiled: &mut CompiledGraph, h: &[f32], snaps: &[Vec<f32>]) -> Vec<f32> {
    let snap_names: Vec<String> = (0..snaps.len()).map(|j| format!("snap_{j}")).collect();
    let mut inputs: Vec<(&str, &[f32])> = vec![("h", h)];
    for (j, sn) in snaps.iter().enumerate() {
        inputs.push((snap_names[j].as_str(), sn.as_slice()));
    }
    timed("head:run(lm_head matmul)", || {
        compiled.run(&inputs).remove(0)
    })
}

/// Load + apply the head to a final `(hidden, snapshots)` → logits `[seq*vocab]`.
/// One-shot (rebuilds every call); see [`apply_head_cached`] for the looped path.
pub fn apply_head(
    ck: &mut CheckpointLoader,
    cfg: &FlowConfig,
    h: &[f32],
    snaps: &[Vec<f32>],
    device: Device,
) -> Result<Vec<f32>> {
    let mut cache = HeadCache::disabled();
    apply_head_cached(ck, cfg, h, snaps, &mut cache, device)
}

/// Apply the head, reusing a resident compiled seq==1 head graph across tokens
/// (the ~4.7 GB `lm_head` is loaded/lowered ONCE, not per token). Falls back to a
/// fresh build for prefill (seq>1), a disabled cache, or a changed snapshot count.
pub fn apply_head_cached(
    ck: &mut CheckpointLoader,
    cfg: &FlowConfig,
    h: &[f32],
    snaps: &[Vec<f32>],
    cache: &mut HeadCache,
    device: Device,
) -> Result<Vec<f32>> {
    let n_snaps = snaps.len();
    let cacheable = cache.enabled && cfg.seq == 1;
    // drop a stale cached graph built for a different snapshot count
    if cache.graph.as_ref().is_some_and(|(ns, _)| *ns != n_snaps) {
        cache.graph = None;
    }
    let logits = if cacheable {
        if cache.graph.is_none() {
            cache.graph = Some((n_snaps, build_head_graph(ck, cfg, n_snaps, device)?));
        }
        let (_, compiled) = cache.graph.as_mut().unwrap();
        run_head_graph(compiled, h, snaps)
    } else {
        let mut compiled = build_head_graph(ck, cfg, n_snaps, device)?;
        run_head_graph(&mut compiled, h, snaps)
    };
    bench_report();
    Ok(logits)
}

// ── full O(1) decode / generation ──────────────────────────────────────────

/// One layer's cross-token decode state: KDA carries conv+scan, MLA carries the
/// growing key/value cache (`[s_past·h·qk]` each).
#[derive(Clone)]
pub enum AttnState {
    Kda(KdaState),
    Mla { k: Vec<f32>, v: Vec<f32> },
}

/// The whole model's decode state: one [`AttnState`] per layer + the shared KV
/// cache length `s_past` (all MLA layers grow in lockstep). Snapshots are NOT here
/// — AttnRes is per-position, so they're internal to each token's forward.
/// `Clone` enables speculative decode's snapshot / re-sync.
#[derive(Clone)]
pub struct DecodeState {
    pub layers: Vec<AttnState>,
    pub s_past: usize,
}

impl DecodeState {
    /// Zeroed initial state for a fresh sequence.
    pub fn zeros(tc: &KimiLinearConfig, cfg: &FlowConfig) -> Self {
        let layers = (0..tc.num_hidden_layers)
            .map(|i| {
                if tc.is_kda_layer(i) {
                    AttnState::Kda(KdaState::zeros(cfg.kda))
                } else {
                    AttnState::Mla {
                        k: Vec::new(),
                        v: Vec::new(),
                    }
                }
            })
            .collect();
        Self { layers, s_past: 0 }
    }
}

pub(crate) fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold(
            (0usize, f32::MIN),
            |m, (i, &x)| if x > m.1 { (i, x) } else { m },
        )
        .0 as u32
}

/// Build + compile one layer's decode-step graph (weights from `lw` baked in),
/// returning `(compiled, n_snaps_out)`. `n_snaps_in` = incoming AttnRes snapshot
/// count; `s_past` sizes the MLA KV inputs (unused for KDA — its state is fixed-
/// shape, which is why KDA graphs are cacheable across tokens). No `ck`: the MoE
/// experts are paged separately, so this graph is self-contained + reusable.
#[allow(clippy::too_many_arguments)]
fn build_decode_layer_compiled(
    tc: &KimiLinearConfig,
    cfg: &FlowConfig,
    i: usize,
    lw: &LayerWeights,
    n_snaps_in: usize,
    s_past: usize,
    device: Device,
) -> Result<(CompiledGraph, usize)> {
    let (hidden, seq) = (cfg.hidden, cfg.seq);
    let is_moe = tc.is_moe_layer(i);
    let is_kda = tc.is_kda_layer(i);
    // bf16-resident backbone (opt-in): emit this layer's `linear` matmul weights as
    // BF16 params (fed post-compile) instead of baked f32 — half the weight bytes,
    // consumed by the backends' bf16 matmul kernels. On a graph-cached KDA layer the
    // bf16 bakes into the cached graph (f32 source transient) → genuinely resident.
    let bf16_bb = std::env::var("RLX_KIMI_BF16_BACKBONE").is_ok();
    if bf16_bb {
        crate::common::bf16_backbone_begin();
    }
    // int8-resident backbone (RLX_KIMI_INT8_BACKBONE): bakes per-channel int8 into
    // the cached decode graph → the whole backbone fits resident at ~57 GB. Takes
    // precedence over bf16 inside `common::linear`.
    let int8_bb = crate::common::int8_backbone_requested();
    if int8_bb {
        crate::common::int8_backbone_begin();
    }
    // per-layer quant context (drives RLX_KIMI_QUANT=adaptive's depth schedule).
    crate::common::set_quant_layer(Some((i, tc.num_hidden_layers)));
    let mut hir = HirModule::new("decode_layer");
    let mut g = HirMut::new(&mut hir);
    let h_node = g.input("h", Shape::new(&[1, seq, hidden], DType::F32));
    let snap_nodes: Vec<_> = (0..n_snaps_in)
        .map(|j| {
            g.input(
                format!("snap_{j}"),
                Shape::new(&[1, seq, hidden], DType::F32),
            )
        })
        .collect();
    let attn_in = if is_kda {
        let (proj, kk) = (cfg.kda.proj(), cfg.kda.conv_kernel);
        let sh = |g: &mut HirMut, n: &str| g.input(n, Shape::new(&[1, kk - 1, proj], DType::F32));
        AttnDecodeIn::Kda {
            csq: sh(&mut g, "csq"),
            csk: sh(&mut g, "csk"),
            csv: sh(&mut g, "csv"),
            scan: g.input(
                "scan",
                Shape::new(
                    &[1, cfg.kda.num_heads, cfg.kda.head_dim, cfg.kda.head_dim],
                    DType::F32,
                ),
            ),
        }
    } else {
        let hq = cfg.mla.num_heads * cfg.mla.qk();
        // The V cache is v_head_dim-wide in the (default) vdim path — a SMALLER cache
        // than the qk-wide K cache. Declaring `cv` at `hq` (qk-wide) is a latent shape
        // lie CPU/Metal tolerate (runtime len) but MLX rejects at the KV-cache concat
        // (`cached_v [.,.,h·qk]` ⊕ `vf [.,.,h·vd]`). Declare it at the real width.
        let hv = if crate::mla::mla_vdim() {
            cfg.mla.num_heads * cfg.mla.v_head_dim
        } else {
            hq
        };
        AttnDecodeIn::Mla {
            ck: g.input("ck", Shape::new(&[1, s_past, hq], DType::F32)),
            cv: g.input("cv", Shape::new(&[1, s_past, hv], DType::F32)),
        }
    };
    let mut params = HashMap::new();
    let (mn, stream, snaps_out, attn_out) =
        build_layer_decode_step(&mut g, &mut params, i, h_node, snap_nodes, attn_in, lw, cfg)?;
    let mut outputs = Vec::new();
    if is_moe {
        outputs.push(mn);
        outputs.push(stream);
    } else {
        let FfnWeights::Dense(dw) = &lw.ffn else {
            unreachable!()
        };
        let ffn = build_dense_mlp(
            &mut g,
            &mut params,
            &format!("l{i}.mlp"),
            mn,
            dw,
            hidden,
            cfg.dense_inter,
            1,
            seq,
            cfg.situ_beta,
            cfg.situ_linear_beta,
        )?;
        outputs.push(g.add(stream, ffn));
    }
    let n_snaps_out = snaps_out.len();
    outputs.extend(snaps_out);
    match &attn_out {
        AttnDecodeOut::Kda {
            csq,
            csk,
            csv,
            scan,
        } => outputs.extend([*csq, *csk, *csv, *scan]),
        AttnDecodeOut::Mla { k, v } => outputs.extend([*k, *v]),
    }
    g.set_outputs(outputs);
    let mut compiled = timed("layer:body(compile)", || -> Result<_> {
        let built = built_from_hir(hir, params)?;
        compile_built(built, dev_override("RLX_KIMI_BODY_DEVICE", device))
    })?;
    if bf16_bb {
        // feed the collected BF16 backbone weights into the compiled (cached) graph.
        for (name, bytes) in crate::common::bf16_backbone_take() {
            compiled.set_param_typed(&name, &bytes, DType::BF16);
        }
    }
    if int8_bb {
        // feed the collected int8 backbone weights into the compiled (cached) graph.
        for (name, bytes) in crate::common::int8_backbone_take() {
            compiled.set_param_typed(&name, &bytes, DType::I8);
        }
    }
    crate::common::set_quant_layer(None);
    Ok((compiled, n_snaps_out))
}

/// Run `cfg.seq` new tokens through ALL layers via the O(1) decode-step builders,
/// threading `state` (per-layer KDA conv/scan or growing MLA KV cache) and
/// streaming each layer's backbone from disk. This is the shared primitive for
/// PREFILL (`seq = prompt_len`, zeroed state) and DECODE (`seq = 1`, carried
/// state). Returns `(hidden [seq*hidden], attn_res_snapshots)` for the head.
#[allow(clippy::too_many_arguments)]
pub fn decode_forward(
    ck: &mut CheckpointLoader,
    tc: &KimiLinearConfig,
    cfg: &FlowConfig,
    h: Vec<f32>,
    state: &mut DecodeState,
    n_layers: usize,
    device: Device,
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let mut cache = LayerCache::disabled();
    decode_forward_range(
        ck,
        tc,
        cfg,
        h,
        Vec::new(),
        state,
        0,
        n_layers,
        &mut cache,
        device,
    )
}

/// One decode/prefill pass over layers `start..end` only (the distributed
/// per-node slice), continuing the upstream `snaps_in` accumulator. `state`
/// carries this node's cross-token KDA conv/scan + MLA KV for those layers
/// (`state.layers[i]` indexed by GLOBAL `i`); it is mutated in place and
/// `state.s_past` is bumped once. AttnRes snapshots are per-token — they flow in
/// (`snaps_in`, empty at the first node) and out to the next node. Returns the
/// boundary `(hidden, snaps)`. [`decode_forward`] = the whole-model `0..n_layers`
/// case with an empty incoming accumulator.
#[allow(clippy::too_many_arguments)]
pub fn decode_forward_range(
    ck: &mut CheckpointLoader,
    tc: &KimiLinearConfig,
    cfg: &FlowConfig,
    mut h: Vec<f32>,
    mut snaps: Vec<Vec<f32>>,
    state: &mut DecodeState,
    start: usize,
    end: usize,
    cache: &mut LayerCache,
    device: Device,
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let seq = cfg.seq;
    for i in start..end {
        let lp = format!("language_model.model.layers.{i}");
        let is_moe = tc.is_moe_layer(i);
        let is_kda = tc.is_kda_layer(i);
        // KDA decode-step graphs are shape-invariant (fixed-size recurrent state),
        // so for seq==1 we compile them ONCE and reuse every token. MLA graphs grow
        // with the KV cache → never cached. A graph bakes its weights in, so a
        // graph-cached layer is NOT also weight-cached (that would double its RAM).
        let use_graph_cache = seq == 1 && is_kda && cache.graphs_on(i);

        // Ensure a compiled graph is available; `local` holds a freshly-built one
        // for the non-cached path (MLA, seq>1, or graph cache off/full).
        let mut local: Option<(CompiledGraph, usize)> = None;
        if use_graph_cache {
            if let std::collections::hash_map::Entry::Vacant(e) = cache.graphs.entry(i) {
                let lw = timed("backbone:load(cache-miss)", || {
                    load_layer_backbone(ck, tc, cfg, i)
                })?;
                let built = build_decode_layer_compiled(
                    tc,
                    cfg,
                    i,
                    &lw,
                    snaps.len(),
                    state.s_past,
                    device,
                )?;
                e.insert(built);
            }
        } else {
            // Resident weights: load this layer's backbone ONCE, reuse across tokens
            // (`scratch` = an uncached layer — past cap / cache off — streams fresh).
            if cache.admits(i) && !cache.map.contains_key(&i) {
                let lw = timed("backbone:load(cache-miss)", || {
                    load_layer_backbone(ck, tc, cfg, i)
                })?;
                cache.map.insert(i, lw);
            }
            let scratch;
            let lw: &LayerWeights = if let Some(lw) = cache.map.get(&i) {
                lw
            } else {
                scratch = timed("backbone:load(uncached)", || {
                    load_layer_backbone(ck, tc, cfg, i)
                })?;
                &scratch
            };
            local = Some(build_decode_layer_compiled(
                tc,
                cfg,
                i,
                lw,
                snaps.len(),
                state.s_past,
                device,
            )?);
        }
        let (compiled, n_snaps): (&mut CompiledGraph, usize) = if use_graph_cache {
            let (c, n) = cache.graphs.get_mut(&i).unwrap();
            (c, *n)
        } else {
            let (c, n) = local.as_mut().unwrap();
            (c, *n)
        };

        // bind (scoped so the immutable borrow of `state` ends before we mutate it)
        let snap_names: Vec<String> = (0..snaps.len()).map(|j| format!("snap_{j}")).collect();
        let mut res = timed("layer:body(run)", || {
            let mut inputs: Vec<(&str, &[f32])> = vec![("h", h.as_slice())];
            for (j, sn) in snaps.iter().enumerate() {
                inputs.push((snap_names[j].as_str(), sn.as_slice()));
            }
            match &state.layers[i] {
                AttnState::Kda(ks) => inputs.extend([
                    ("csq", ks.conv_q.as_slice()),
                    ("csk", ks.conv_k.as_slice()),
                    ("csv", ks.conv_v.as_slice()),
                    ("scan", ks.scan.as_slice()),
                ]),
                AttnState::Mla { k, v } => {
                    inputs.extend([("ck", k.as_slice()), ("cv", v.as_slice())])
                }
            }
            compiled.run(&inputs)
        });

        // extract: [h_or_(mn,stream), snaps.., state..]
        let new_h = if is_moe {
            let mn_v = res.remove(0);
            let stream_v = res.remove(0);
            let new_snaps: Vec<Vec<f32>> = res.drain(0..n_snaps).collect();
            snaps = new_snaps;
            let moe = run_moe_layer(ck, &lp, i, &mn_v, cfg.moe, device)?;
            add_vec(&stream_v, &moe)
        } else {
            let h_out = res.remove(0);
            snaps = res.drain(0..n_snaps).collect();
            h_out
        };
        h = new_h;
        match &mut state.layers[i] {
            AttnState::Kda(ks) => {
                ks.conv_q = res.remove(0);
                ks.conv_k = res.remove(0);
                ks.conv_v = res.remove(0);
                ks.scan = res.remove(0);
            }
            AttnState::Mla { k, v } => {
                *k = res.remove(0);
                *v = res.remove(0);
            }
        }
    }
    state.s_past += seq;
    Ok((h, snaps))
}

/// **Greedy generation.** Prefill `prompt` (one `decode_forward` over the whole
/// prompt from a zeroed [`DecodeState`]), take the head on the last position for
/// the first token, then generate `n_gen-1` more tokens O(1) each. `make_cfg(seq)`
/// builds a seq-shaped [`FlowConfig`]. Returns the generated token ids.
#[allow(clippy::too_many_arguments)]
pub fn run_generate(
    ck: &mut CheckpointLoader,
    tc: &KimiLinearConfig,
    make_cfg: impl Fn(usize) -> FlowConfig,
    prompt: &[u32],
    n_gen: usize,
    n_layers: usize,
    device: Device,
) -> Result<Vec<u32>> {
    let cfg1 = make_cfg(1);
    let hidden = cfg1.hidden;
    let mut state = DecodeState::zeros(tc, &cfg1);
    // resident backbone weights: load each layer once, reuse across all tokens
    // (only helps when the range fits RAM — bounded by RLX_KIMI_WEIGHT_CACHE_LAYERS).
    let mut cache = LayerCache::from_env();
    // resident head: compile the seq==1 head graph (4.7 GB lm_head baked in) once.
    let mut head = HeadCache::from_env();
    let emb = "language_model.model.embed_tokens.weight";

    // ── prefill the prompt ──
    let cfg_p = make_cfg(prompt.len());
    let h0 = ck.gather_embed(emb, prompt, hidden)?;
    let (h, snaps) = decode_forward_range(
        ck,
        tc,
        &cfg_p,
        h0,
        Vec::new(),
        &mut state,
        0,
        n_layers,
        &mut cache,
        device,
    )?;
    // head on the LAST prompt position → first generated token
    let last = prompt.len() - 1;
    let h_last = h[last * hidden..(last + 1) * hidden].to_vec();
    let snaps_last: Vec<Vec<f32>> = snaps
        .iter()
        .map(|s| s[last * hidden..(last + 1) * hidden].to_vec())
        .collect();
    let mut logits = apply_head_cached(ck, &cfg1, &h_last, &snaps_last, &mut head, device)?;
    let mut tok = argmax(&logits);
    let mut out = vec![tok];

    // ── decode the rest, one token at a time (backbone now resident in `cache`) ──
    for _ in 1..n_gen {
        let hin = ck.gather_embed(emb, &[tok], hidden)?;
        let (h, snaps) = decode_forward_range(
            ck,
            tc,
            &cfg1,
            hin,
            Vec::new(),
            &mut state,
            0,
            n_layers,
            &mut cache,
            device,
        )?;
        logits = apply_head_cached(ck, &cfg1, &h, &snaps, &mut head, device)?;
        tok = argmax(&logits);
        out.push(tok);
    }
    Ok(out)
}

// ── speculative decode ──────────────────────────────────────────────────────

const EMB: &str = "language_model.model.embed_tokens.weight";

/// Feed `tokens` (seq) through the first `n_layers` from `state` (mutating it),
/// apply the head per position, and return the greedy token AT each position
/// (position `p` = the model's prediction after `tokens[..=p]`).
fn decode_and_head(
    ck: &mut CheckpointLoader,
    tc: &KimiLinearConfig,
    make_cfg: &impl Fn(usize) -> FlowConfig,
    tokens: &[u32],
    state: &mut DecodeState,
    n_layers: usize,
    device: Device,
) -> Result<Vec<u32>> {
    let cfg = make_cfg(tokens.len());
    let hidden = cfg.hidden;
    let h0 = ck.gather_embed(EMB, tokens, hidden)?;
    let (h, snaps) = decode_forward(ck, tc, &cfg, h0, state, n_layers, device)?;
    let cfg1 = make_cfg(1);
    (0..tokens.len())
        .map(|p| {
            let hp = &h[p * hidden..(p + 1) * hidden];
            let sp: Vec<Vec<f32>> = snaps
                .iter()
                .map(|s| s[p * hidden..(p + 1) * hidden].to_vec())
                .collect();
            Ok(argmax(&apply_head(ck, &cfg1, hp, &sp, device)?))
        })
        .collect()
}

/// **Speculative decode** (self-speculative, greedy): an early-exit DRAFT (first
/// `n_draft` layers + head) proposes `k` tokens, the full TARGET (`n_layers`)
/// verifies them in ONE batched forward, and the longest matching prefix + one
/// bonus token are accepted. States are snapshot/`clone`d and the accepted prefix
/// is re-run to re-sync (the KDA scan is recurrent and can't be truncated). Output
/// is IDENTICAL to greedy [`run_generate`]; the draft only affects speed. Returns
/// `(tokens, accepted_drafts)` — the second is telemetry (higher = better draft).
#[allow(clippy::too_many_arguments)]
pub fn run_speculative_generate(
    ck: &mut CheckpointLoader,
    tc: &KimiLinearConfig,
    make_cfg: impl Fn(usize) -> FlowConfig,
    prompt: &[u32],
    n_gen: usize,
    n_layers: usize,
    n_draft: usize,
    k: usize,
    device: Device,
) -> Result<(Vec<u32>, usize)> {
    let cfg1 = make_cfg(1);
    let hidden = cfg1.hidden;
    let mut tgt = DecodeState::zeros(tc, &cfg1);
    let mut drf = DecodeState::zeros(tc, &cfg1);

    // prefill both models over the prompt; the target head gives the first token.
    let cfg_p = make_cfg(prompt.len());
    let h0 = ck.gather_embed(EMB, prompt, hidden)?;
    let (ht, snt) = decode_forward(ck, tc, &cfg_p, h0, &mut tgt, n_layers, device)?;
    let hd = ck.gather_embed(EMB, prompt, hidden)?;
    let _ = decode_forward(ck, tc, &cfg_p, hd, &mut drf, n_draft, device)?;
    let last = prompt.len() - 1;
    let sp: Vec<Vec<f32>> = snt
        .iter()
        .map(|s| s[last * hidden..(last + 1) * hidden].to_vec())
        .collect();
    let mut cur = argmax(&apply_head(
        ck,
        &cfg1,
        &ht[last * hidden..(last + 1) * hidden],
        &sp,
        device,
    )?);
    let mut out = vec![cur];
    let mut accepted_drafts = 0usize;

    while out.len() < n_gen {
        let (tgt_snap, drf_snap) = (tgt.clone(), drf.clone());

        // DRAFT: from `cur`, roll `k` tokens forward through the draft model.
        let mut drafts = Vec::with_capacity(k);
        let mut x = cur;
        for _ in 0..k {
            x = decode_and_head(ck, tc, &make_cfg, &[x], &mut drf, n_draft, device)?[0];
            drafts.push(x);
        }

        // VERIFY: target over [cur, d0..d_{k-1}] in ONE forward → t0..tk.
        let mut verify_toks = Vec::with_capacity(k + 1);
        verify_toks.push(cur);
        verify_toks.extend(&drafts);
        let t = decode_and_head(ck, tc, &make_cfg, &verify_toks, &mut tgt, n_layers, device)?;
        // accept d_i while t_i == d_i; bonus = t_J.
        let mut j = 0;
        while j < k && t[j] == drafts[j] {
            j += 1;
        }
        accepted_drafts += j;
        for &d in &drafts[..j] {
            out.push(d);
        }
        let bonus = t[j];
        out.push(bonus);

        // RE-SYNC both states to the confirmed prefix [cur, d0..d_{j-1}] (the tokens
        // fed BEFORE the new `cur`); re-running from the snapshot rebuilds the
        // recurrent KDA scan exactly. (j==k ⇒ nothing over-fed on target, but the
        // draft over-shot by one; re-run keeps them aligned.)
        let mut confirmed = Vec::with_capacity(j + 1);
        confirmed.push(cur);
        confirmed.extend(&drafts[..j]);
        tgt = tgt_snap;
        drf = drf_snap;
        let cfg_c = make_cfg(confirmed.len());
        let ht = ck.gather_embed(EMB, &confirmed, hidden)?;
        let _ = decode_forward(ck, tc, &cfg_c, ht, &mut tgt, n_layers, device)?;
        let hd = ck.gather_embed(EMB, &confirmed, hidden)?;
        let _ = decode_forward(ck, tc, &cfg_c, hd, &mut drf, n_draft, device)?;
        cur = bonus;
    }
    out.truncate(n_gen);
    Ok((out, accepted_drafts))
}
