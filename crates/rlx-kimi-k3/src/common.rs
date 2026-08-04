// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Shared graph-builder helpers for Kimi-K3 — the custom **situ** activation and
//! scalar constants.

use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::{Activation, Op};
use rlx_ir::quant::{ScaleLayout, ScaledFormat};
use rlx_ir::{DType, HirGraphExt, QuantScheme, Shape};
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// When `Some`, backbone [`linear`] matmul weights are emitted as **BF16**
    /// `g.param` nodes (their bytes collected here for a post-compile
    /// `set_param_typed`) instead of baked f32 `reg`s — the bf16-resident backbone
    /// (`RLX_KIMI_BF16_BACKBONE`). The layer-compile site installs it around ONE
    /// layer's HIR build, then drains it after compile; norms / non-matmul `reg`s
    /// stay f32 (only `linear` weights, the big GEMMs, are packed).
    static BF16_BACKBONE: RefCell<Option<Vec<(String, Vec<u8>)>>> =
        const { RefCell::new(None) };

    /// Like [`BF16_BACKBONE`] but for the **int8-resident** backbone
    /// (`RLX_KIMI_INT8_BACKBONE`): `linear` weights are per-output-channel int8
    /// (¼ the f32 / ½ the bf16 bytes, 57 GB for the whole backbone) baked into the
    /// cached decode graph as an `Op::DequantMatMul{Int8Block}` — the pack that
    /// makes the full backbone RESIDENT on a 64 GB Mac. The int8 bytes are collected
    /// here for a post-compile `set_param_typed(name, bytes, DType::I8)`; the
    /// per-column f32 scale is a baked `reg`.
    static INT8_BACKBONE: RefCell<Option<Vec<(String, Vec<u8>)>>> =
        const { RefCell::new(None) };

    /// Current `(layer_index, num_layers)` for the layer being built — set by the
    /// runner around each layer's HIR build so the `adaptive` weight-quant policy
    /// (`RLX_KIMI_QUANT=adaptive`) can pick a per-layer scheme by depth. `None`
    /// outside a layer build (adaptive then falls back to safe per-channel int8).
    static QUANT_LAYER: std::cell::Cell<Option<(usize, usize)>> =
        const { std::cell::Cell::new(None) };
}

/// Record the layer index / total-layer count for the layer about to be built, so
/// the `adaptive` quant policy can choose its scheme by depth. Clear with `None`.
pub fn set_quant_layer(ctx: Option<(usize, usize)>) {
    QUANT_LAYER.with(|c| c.set(ctx));
}

/// Start collecting backbone `linear` weights as BF16 for this layer's build.
pub fn bf16_backbone_begin() {
    BF16_BACKBONE.with(|c| *c.borrow_mut() = Some(Vec::new()));
}

/// Take the collected `(param_name, bf16_bytes)` and stop collecting. Feed each
/// via `CompiledGraph::set_param_typed(name, bytes, DType::BF16)` after compile.
pub fn bf16_backbone_take() -> Vec<(String, Vec<u8>)> {
    BF16_BACKBONE.with(|c| c.borrow_mut().take().unwrap_or_default())
}

pub fn bf16_backbone_active() -> bool {
    BF16_BACKBONE.with(|c| c.borrow().is_some())
}

/// Start collecting backbone `linear` weights as int8 for this layer's build.
pub fn int8_backbone_begin() {
    INT8_BACKBONE.with(|c| *c.borrow_mut() = Some(Vec::new()));
}

/// Take the collected `(param_name, i8_bytes)` and stop collecting. Feed each via
/// `CompiledGraph::set_param_typed(name, bytes, DType::I8)` after compile.
pub fn int8_backbone_take() -> Vec<(String, Vec<u8>)> {
    INT8_BACKBONE.with(|c| c.borrow_mut().take().unwrap_or_default())
}

pub fn int8_backbone_active() -> bool {
    INT8_BACKBONE.with(|c| c.borrow().is_some())
}

/// Emit a **int8-resident** `x @ w` under `full`: bake per-output-channel int8
/// weight (bytes collected for a post-compile `set_param_typed(I8)`) + f32
/// per-column scale into an `Op::DequantMatMul{Int8Block, block_size=K}`. Shared by
/// [`linear`] and `kda::fused_input_proj` so BOTH the separate projections and the
/// fused KDA input matmul are packed (`w` layout `[in,out]=[K,N]`).
pub fn emit_int8_resident(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    full: &str,
    x: HirNodeId,
    w: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> HirNodeId {
    if std::env::var("RLX_KIMI_INT8_DIAG").is_ok() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let c = N.fetch_add(1, Ordering::Relaxed);
        if c < 12 {
            eprintln!("[int8-resident] emit #{c}: {full} [{in_dim},{out_dim}]");
        }
    }
    // Prequant: LOAD mmaps the pre-recorded int8 codes+scale (no bf16 read, no
    // upcast, no transpose — `w` is ignored/empty); RECORD dumps them once during a
    // normal bf16 forward; otherwise quantize inline. Keyed by the graph-param name
    // `full`, so FUSED weights (in_proj_fused / qkv_a_fused) are captured post-fusion.
    let (q_bytes, scale) = if let Some(dir) = prequant_load_dir() {
        load_prequant(&dir, full, in_dim, out_dim).unwrap_or_else(|| {
            // Missing .q8 with no bf16 fallback (the loader skipped it) is a
            // misconfiguration — fail loudly instead of indexing empty `w`.
            assert!(
                !w.is_empty(),
                "prequant LOAD: {dir}/{full}.q8 missing and bf16 was skipped — re-record with RLX_KIMI_PREQUANT_RECORD"
            );
            quantize_int8_col(w, in_dim, out_dim)
        })
    } else {
        let (q, s) = quantize_int8_col(w, in_dim, out_dim);
        if let Some(dir) = prequant_record_dir() {
            write_prequant(&dir, full, &q, &s);
        }
        (q, s)
    };
    let wid = g.param(full, Shape::new(&[in_dim, out_dim], DType::I8));
    let sid = reg(g, params, &format!("{full}.iscale"), scale, &[1, out_dim]);
    let zid = reg(
        g,
        params,
        &format!("{full}.izp"),
        vec![0.0; out_dim],
        &[1, out_dim],
    );
    INT8_BACKBONE.with(|c| {
        if let Some(v) = c.borrow_mut().as_mut() {
            v.push((full.to_string(), q_bytes));
        }
    });
    // out_shape = x's shape with the last dim replaced by out_dim.
    let mut osh: Vec<usize> = g
        .shape(x)
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let last = osh.len() - 1;
    osh[last] = out_dim;
    g.0.dequant_matmul(
        x,
        wid,
        Some(sid),
        Some(zid),
        QuantScheme::Int8Block {
            block_size: in_dim as u32,
        },
        Shape::new(&osh, DType::F32),
    )
}

/// Reader for the **scaled** (tensor-core minifloat) backbone linear path:
///   `RLX_KIMI_SCALED_BACKBONE=fp8|w8a8`  → MXFP8 activations · MXFP8 weights,
///   `RLX_KIMI_SCALED_BACKBONE=mxfp4|w4a8` → MXFP8 activations · MXFP4 weights.
/// Both use OCP block-32 e8m0 microscaling (`ScaleLayout::mx`). Returns
/// `(act_fmt, weight_fmt, layout)` or `None` when unset. This is the *real*
/// activation-quantized arithmetic (`Op::ScaledMatMul`) — W×A8, not weight-only
/// fake-quant — and doubles as the low-precision compute path.
pub fn scaled_backbone() -> Option<(ScaledFormat, ScaledFormat, ScaleLayout)> {
    match std::env::var("RLX_KIMI_SCALED_BACKBONE").ok().as_deref() {
        Some("fp8") | Some("w8a8") => Some((
            ScaledFormat::F8E4M3,
            ScaledFormat::F8E4M3,
            ScaleLayout::mx(),
        )),
        Some("mxfp4") | Some("w4a8") => Some((
            ScaledFormat::F8E4M3,
            ScaledFormat::F4E2M1,
            ScaleLayout::mx(),
        )),
        _ => None,
    }
}

/// Emit `x @ w` as a native low-precision `Op::ScaledMatMul`: the activation is
/// dynamically quantized to `afmt` and the weight to `wfmt` (both `lay`), decoded
/// with f32 accumulation. `w` is `[in,out]=[K,N]` (loader layout); `ScaledMatMul`
/// wants `rhs` K-last `[N,K]`, so the weight is transposed once at build (baked).
#[allow(clippy::too_many_arguments)]
pub fn emit_scaled_linear(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    full: &str,
    x: HirNodeId,
    w: &[f32],
    in_dim: usize,
    out_dim: usize,
    afmt: ScaledFormat,
    wfmt: ScaledFormat,
    lay: ScaleLayout,
) -> HirNodeId {
    // x logical [.., in] → flatten leading dims to [rows, in].
    let xs: Vec<usize> = g
        .shape(x)
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let rows: usize = xs.iter().product::<usize>() / in_dim.max(1);
    let x2d = if xs.len() == 2 {
        x
    } else {
        g.reshape_(x, vec![rows as i64, in_dim as i64])
    };
    // weight [K,N] → [N,K] (row-major transpose), baked as an f32 param.
    let mut w_nk = vec![0f32; out_dim * in_dim];
    for r in 0..in_dim {
        for c in 0..out_dim {
            w_nk[c * in_dim + r] = w[r * out_dim + c];
        }
    }
    let wid = reg(g, params, &format!("{full}.wnk"), w_nk, &[out_dim, in_dim]);
    let (xq, xsq) = g.scaled_quantize(x2d, afmt, lay);
    let (wq, wsq) = g.scaled_quantize(wid, wfmt, lay);
    let out = g.add_node(
        Op::ScaledMatMul {
            lhs_format: afmt,
            rhs_format: wfmt,
            scale_layout: lay,
            has_bias: false,
        },
        vec![xq, wq, xsq, wsq],
        Shape::new(&[rows, out_dim], DType::F32),
    );
    if xs.len() == 2 {
        out
    } else {
        let mut osh = xs.clone();
        let last = osh.len() - 1;
        osh[last] = out_dim;
        g.reshape_(out, osh.iter().map(|&d| d as i64).collect())
    }
}

// ── prequantized (pre-transposed, pre-fused) backbone on disk ───────────────
// The int8-resident graph is byte-identical whether the codes come from an inline
// `quantize_int8_col(bf16)` or from a mmapped `.q8` file — so recording once and
// loading thereafter turns the CPU-bound bf16→f32 upcast+transpose load (measured
// ~0.32 GB/s) into a bare int8 mmap+memcpy (~5.7× faster, ¼ the bytes).

/// Directory to DUMP int8 codes into during a normal forward (`RLX_KIMI_PREQUANT_RECORD`).
pub fn prequant_record_dir() -> Option<String> {
    std::env::var("RLX_KIMI_PREQUANT_RECORD")
        .ok()
        .filter(|s| !s.is_empty())
}
/// Directory to MMAP pre-recorded int8 codes from (`RLX_KIMI_PREQUANT_LOAD`).
pub fn prequant_load_dir() -> Option<String> {
    std::env::var("RLX_KIMI_PREQUANT_LOAD")
        .ok()
        .filter(|s| !s.is_empty())
}
/// True when the loader should SKIP bf16 `linear_t` (the weight is mmapped in
/// [`emit_int8_resident`] by name instead).
pub fn prequant_load_active() -> bool {
    prequant_load_dir().is_some()
}

/// int8-resident backbone is requested by `RLX_KIMI_INT8_BACKBONE` OR by either
/// prequant mode (record needs the codes; load feeds them) — both route through
/// [`emit_int8_resident`].
pub fn int8_backbone_requested() -> bool {
    std::env::var("RLX_KIMI_INT8_BACKBONE").is_ok()
        || prequant_load_dir().is_some()
        || prequant_record_dir().is_some()
}

/// `{dir}/{full}.q8` = `[codes: in·out u8][scale: out f32-le]`.
fn write_prequant(dir: &str, full: &str, codes: &[u8], scale: &[f32]) {
    use std::io::Write;
    let _ = std::fs::create_dir_all(dir);
    let path = format!("{dir}/{full}.q8");
    if let Ok(mut f) = std::fs::File::create(&path) {
        let _ = f.write_all(codes);
        let sb: Vec<u8> = scale.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = f.write_all(&sb);
    }
}

/// mmap `{dir}/{full}.q8` → `(codes [in·out], scale [out])`, or `None` if absent /
/// wrong size (caller then falls back to inline quant).
fn load_prequant(
    dir: &str,
    full: &str,
    in_dim: usize,
    out_dim: usize,
) -> Option<(Vec<u8>, Vec<f32>)> {
    let path = format!("{dir}/{full}.q8");
    let f = std::fs::File::open(&path).ok()?;
    let m = unsafe { memmap2::Mmap::map(&f).ok()? };
    let ncodes = in_dim * out_dim;
    let nscale = out_dim * 4;
    if m.len() != ncodes + nscale {
        eprintln!(
            "[prequant] {full}: size {} != {}+{} — falling back",
            m.len(),
            ncodes,
            nscale
        );
        return None;
    }
    let codes = m[..ncodes].to_vec();
    let scale: Vec<f32> = m[ncodes..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some((codes, scale))
}

/// Per-output-channel symmetric int8: `q[k,n]=round(w[k,n]/s[n])`, `s[n]=amax_n/127`.
/// Returns the i8 codes (row-major `[k,n]`, as bytes) and the per-column scale `[n]`.
pub fn quantize_int8_col(w: &[f32], k: usize, n: usize) -> (Vec<u8>, Vec<f32>) {
    let mut scale = vec![0f32; n];
    let mut q = vec![0u8; k * n];
    for col in 0..n {
        let mut amax = 0f32;
        for row in 0..k {
            amax = amax.max(w[row * n + col].abs());
        }
        let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        scale[col] = s;
        for row in 0..k {
            let v = (w[row * n + col] / s).round().clamp(-127.0, 127.0) as i8;
            q[row * n + col] = v as u8;
        }
    }
    (q, scale)
}

/// Backbone weight-quantization mode for [`linear`] projections — a FAKE-QUANT
/// (quantize→dequantize in f32) so precision can be MEASURED against the bf16
/// baseline before building real int-kernels. Set via `RLX_KIMI_QUANT`:
///   `int8`     = per-output-channel int8 (the recording-justified default)
///   `int8t`    = per-tensor int8 (coarser; for the mild mid-depth layers)
///   `int4`     = int4, group-64 along the input dim (aggressive; symmetric ±7 —
///                the per-channel outliers crush it, ~24% at every depth)
///   `mxfp4`    = FP4 e2m1 + e8m0 block-32 scale (the model's OWN expert format):
///                non-uniform levels {0,±.5,±1,±1.5,±2,±3,±4,±6} give the dynamic
///                range symmetric int4 lacks → outliers no longer crush the bulk.
///   `int4mix`  = outlier-channel mix: the top-`RLX_KIMI_INT4MIX_FRAC` (default
///                1/8) highest-amax OUTPUT channels stay per-channel int8, the rest
///                go int4-g64 — mixed precision at the RIGHT (per-channel, not
///                per-layer) granularity the recording identified. ~4.5 avg bits.
///   `nf4`      = NormalFloat-4 (QLoRA): 16 codebook levels at the quantiles of a
///                normal, dense near zero — the 4-bit grid matched to the (near-
///                Gaussian) weight distribution, block-64 absmax scale.
///   `adaptive` = per-layer MIX: per-channel int8 on the fp32-sensitive hotspots
///                (AttnRes-snapshot boundaries every 12 layers + the last few
///                layers near the head), int4-g64 on the mild mid-depth layers.
///                REFUTED by measurement — kept only to reproduce the negative.
///   unset / `off` / `bf16` = no quantization (baseline).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeightQuant {
    None,
    Int8Ch,
    Int8Tensor,
    Int7G64,
    Int7Ch,
    Int6Ch,
    Int5Ch,
    Int6G64,
    Mxfp6,
    Int4G64,
    Mxfp4,
    Int4Mix,
    Nf4,
}

/// The requested top-level policy — a fixed scheme, or `Adaptive` (resolved
/// per-layer via [`QUANT_LAYER`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuantPolicy {
    Fixed(WeightQuant),
    Adaptive,
    /// Per-PROJECTION mix: int8 on the sensitivity-ranked projections whose error
    /// the recurrence amplifies (`RLX_KIMI_MIXED_HI`, default `o_proj`), int4-g64
    /// on the rest. The measured 4-bit fix — ~4.8 effective bits, 2.7× lower error
    /// than uniform int4 (the amplification is concentrated in `o_proj`).
    Mixed,
    /// Like [`Mixed`] but int6-g64 (not int4) on the robust projections: int8 on the
    /// sensitive ones (`RLX_KIMI_MIXED_HI`, default `o_proj`), int6-g64 elsewhere.
    /// ~6.1 effective bits — pushes uniform-int6 accuracy toward int8 by protecting
    /// only the recurrence-amplified projection, for ~⅜ the bf16 bytes.
    Mixed6,
    /// Like [`Mixed6`] but int7-g64 on the robust projections: int8 on the sensitive
    /// ones, int7-g64 elsewhere. ~7.1 effective bits — closes nearly all of the
    /// remaining int6→int8 gap for ~7/16 the bf16 bytes.
    Mixed7,
}

pub fn quant_policy() -> QuantPolicy {
    match std::env::var("RLX_KIMI_QUANT").ok().as_deref() {
        Some("int8") => QuantPolicy::Fixed(WeightQuant::Int8Ch),
        Some("int8t") => QuantPolicy::Fixed(WeightQuant::Int8Tensor),
        Some("int7") => QuantPolicy::Fixed(WeightQuant::Int7G64),
        Some("int7ch") => QuantPolicy::Fixed(WeightQuant::Int7Ch),
        Some("int6") => QuantPolicy::Fixed(WeightQuant::Int6Ch),
        Some("int5") => QuantPolicy::Fixed(WeightQuant::Int5Ch),
        Some("int6g") => QuantPolicy::Fixed(WeightQuant::Int6G64),
        Some("mxfp6") => QuantPolicy::Fixed(WeightQuant::Mxfp6),
        Some("int4") => QuantPolicy::Fixed(WeightQuant::Int4G64),
        Some("mxfp4") => QuantPolicy::Fixed(WeightQuant::Mxfp4),
        Some("int4mix") => QuantPolicy::Fixed(WeightQuant::Int4Mix),
        Some("nf4") => QuantPolicy::Fixed(WeightQuant::Nf4),
        Some("adaptive") => QuantPolicy::Adaptive,
        Some("mixed") => QuantPolicy::Mixed,
        Some("mixed6") => QuantPolicy::Mixed6,
        Some("mixed7") => QuantPolicy::Mixed7,
        _ => QuantPolicy::Fixed(WeightQuant::None),
    }
}

/// Does this projection `name` belong to the int8-kept "high-sensitivity" set for
/// the `mixed` policy? Comma-separated `RLX_KIMI_MIXED_HI` (default `o_proj`).
fn mixed_hi(name: &str) -> bool {
    let hi = std::env::var("RLX_KIMI_MIXED_HI").unwrap_or_else(|_| "o_proj".into());
    hi.split(',')
        .any(|p| !p.trim().is_empty() && name.contains(p.trim()))
}

/// Is a layer index a fp32-sensitive **hotspot** (keep at per-channel int8 under
/// `adaptive`)? AttnRes snapshots land every 12 layers (`0,12,24,…`) — those
/// residual-stream checkpoints carry the largest per-channel outliers in the
/// recording — and the last few layers feed the LM head directly.
pub fn is_quant_hotspot(idx: usize, n_layers: usize) -> bool {
    idx.is_multiple_of(12) || idx + 4 >= n_layers
}

/// Resolve the concrete scheme for a [`linear`] projection named `name` under the
/// requested policy: `adaptive` consults [`QUANT_LAYER`] (int8 on depth hotspots),
/// `mixed` consults [`mixed_hi`] (int8 on the sensitive projections, e.g. `o_proj`,
/// int4 elsewhere). `Fixed` ignores `name`.
pub fn resolve_quant(name: &str) -> WeightQuant {
    match quant_policy() {
        QuantPolicy::Fixed(q) => q,
        QuantPolicy::Adaptive => match QUANT_LAYER.with(|c| c.get()) {
            Some((idx, n)) if !is_quant_hotspot(idx, n) => WeightQuant::Int4G64,
            _ => WeightQuant::Int8Ch,
        },
        QuantPolicy::Mixed => {
            if mixed_hi(name) {
                WeightQuant::Int8Ch
            } else {
                WeightQuant::Int4G64
            }
        }
        QuantPolicy::Mixed6 => {
            if mixed_hi(name) {
                WeightQuant::Int8Ch
            } else {
                WeightQuant::Int6G64
            }
        }
        QuantPolicy::Mixed7 => {
            if mixed_hi(name) {
                WeightQuant::Int8Ch
            } else {
                WeightQuant::Int7G64
            }
        }
    }
}

/// Name-agnostic resolution (equivalent to `resolve_quant("")`).
pub fn weight_quant() -> WeightQuant {
    resolve_quant("")
}

/// Round a magnitude to the nearest **e2m1** (FP4) level and reapply the sign —
/// the non-uniform grid `{0, ±.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` MXFP4 uses.
fn e2m1_quant(v: f32) -> f32 {
    const LVL: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let a = v.abs();
    let mut best = LVL[0];
    let mut bd = (a - LVL[0]).abs();
    for &l in &LVL[1..] {
        let d = (a - l).abs();
        if d < bd {
            bd = d;
            best = l;
        }
    }
    v.signum() * best
}

/// Fake-quantize one strided block (elements `w[base + r*stride]`, `r<len`) as
/// MXFP4: a shared power-of-two (e8m0) scale chosen so the block max lands in the
/// e2m1 range, then each value rounded to e2m1.
fn mxfp4_block(out: &mut [f32], base: usize, stride: usize, len: usize) {
    let mut amax = 0f32;
    for r in 0..len {
        amax = amax.max(out[base + r * stride].abs());
    }
    if amax == 0.0 {
        return;
    }
    // scale = 2^ceil(log2(amax/6)) so max|v/scale| <= 6 (top e2m1 level).
    let e = (amax / 6.0).log2().ceil();
    let scale = 2f32.powf(e);
    for r in 0..len {
        let i = base + r * stride;
        out[i] = e2m1_quant(out[i] / scale) * scale;
    }
}

/// The 16 NormalFloat-4 codebook levels (QLoRA), quantiles of a unit normal in
/// `[-1, 1]` — dense near zero where a Gaussian's mass sits.
const NF4_LVL: [f32; 16] = [
    -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.0910500, 0.0, 0.0795803,
    0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.562_617, 0.7229568, 1.0,
];

fn nf4_quant(v: f32) -> f32 {
    let mut best = NF4_LVL[0];
    let mut bd = (v - NF4_LVL[0]).abs();
    for &l in &NF4_LVL[1..] {
        let d = (v - l).abs();
        if d < bd {
            bd = d;
            best = l;
        }
    }
    best
}

/// Per-output-channel symmetric N-level int fake-quant (`maxlvl` = 2^(bits-1)-1:
/// 127→int8, 31→int6, 15→int5). Parallel over columns (disjoint strided writes).
fn fake_quant_intch(w: &[f32], k: usize, n: usize, maxlvl: f32) -> Vec<f32> {
    use rayon::prelude::*;
    let mut out = w.to_vec();
    let p = out.as_mut_ptr() as usize;
    (0..n).into_par_iter().for_each(|col| {
        let out = unsafe { std::slice::from_raw_parts_mut(p as *mut f32, k * n) };
        let mut amax = 0f32;
        for row in 0..k {
            amax = amax.max(w[row * n + col].abs());
        }
        let s = if amax > 0.0 { amax / maxlvl } else { 1.0 };
        for row in 0..k {
            out[row * n + col] = (w[row * n + col] / s).round().clamp(-maxlvl, maxlvl) * s;
        }
    });
    out
}

/// Group-`g`-along-K symmetric N-level int fake-quant, per column (finer scales
/// than per-channel → better at low bit-widths). Parallel over columns.
fn fake_quant_intg(w: &[f32], k: usize, n: usize, maxlvl: f32, g: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let mut out = w.to_vec();
    let p = out.as_mut_ptr() as usize;
    (0..n).into_par_iter().for_each(|col| {
        let out = unsafe { std::slice::from_raw_parts_mut(p as *mut f32, k * n) };
        let mut row = 0;
        while row < k {
            let end = (row + g).min(k);
            let mut amax = 0f32;
            for r in row..end {
                amax = amax.max(w[r * n + col].abs());
            }
            let s = if amax > 0.0 { amax / maxlvl } else { 1.0 };
            for r in row..end {
                out[r * n + col] = (w[r * n + col] / s).round().clamp(-maxlvl, maxlvl) * s;
            }
            row = end;
        }
    });
    out
}

/// One MX block (e8m0 scale + `fmt` minifloat codes) round-tripped in place — the
/// generic form of [`mxfp4_block`] for any [`ScaledFormat`] (e.g. F6E2M3 for FP6).
fn mx_block_fmt(out: &mut [f32], base: usize, stride: usize, len: usize, fmt: ScaledFormat) {
    use rlx_ir::lowp_codec::{decode, e8m0_to_f32, encode, f32_to_e8m0, max_finite};
    let mut amax = 0f32;
    for r in 0..len {
        amax = amax.max(out[base + r * stride].abs());
    }
    if amax == 0.0 {
        return;
    }
    let e8 = f32_to_e8m0(amax / max_finite(fmt));
    let s = e8m0_to_f32(e8).max(f32::MIN_POSITIVE);
    for r in 0..len {
        let i = base + r * stride;
        out[i] = decode(fmt, encode(fmt, out[i] / s)) * s;
    }
}

/// Fake-quantize a `[K, N]` (row-major, N = output channels) weight per the mode
/// and dequantize back to f32 — the exact values a real int-kernel would use, so
/// running the f32 matmul measures the quantization's precision impact directly.
pub fn fake_quant_weight(w: &[f32], k: usize, n: usize, q: WeightQuant) -> Vec<f32> {
    match q {
        WeightQuant::None => w.to_vec(),
        WeightQuant::Int8Tensor => {
            let amax = w.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            w.iter()
                .map(|&v| (v / s).round().clamp(-127.0, 127.0) * s)
                .collect()
        }
        WeightQuant::Int8Ch => {
            // per-output-channel (column n): scale = max_k|w[k,n]| / 127. Columns
            // touch disjoint strided indices → parallelize over columns (each rayon
            // task writes only its own `col`, no aliasing).
            use rayon::prelude::*;
            let mut out = w.to_vec();
            let p = out.as_mut_ptr() as usize;
            (0..n).into_par_iter().for_each(|col| {
                let out = unsafe { std::slice::from_raw_parts_mut(p as *mut f32, k * n) };
                let mut amax = 0f32;
                for row in 0..k {
                    amax = amax.max(w[row * n + col].abs());
                }
                let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                for row in 0..k {
                    out[row * n + col] = (w[row * n + col] / s).round().clamp(-127.0, 127.0) * s;
                }
            });
            out
        }
        // Between int8 and int4: 6-bit (±31) and 5-bit (±15). Per-channel, plus a
        // group-64 6-bit variant (finer scales). MX FP6 (F6E2M3) is the minifloat
        // analogue — all ~¾ the int8 bytes (⅜ of bf16).
        WeightQuant::Int7G64 => fake_quant_intg(w, k, n, 63.0, 64),
        WeightQuant::Int7Ch => fake_quant_intch(w, k, n, 63.0),
        WeightQuant::Int6Ch => fake_quant_intch(w, k, n, 31.0),
        WeightQuant::Int5Ch => fake_quant_intch(w, k, n, 15.0),
        WeightQuant::Int6G64 => fake_quant_intg(w, k, n, 31.0, 64),
        WeightQuant::Mxfp6 => {
            use rayon::prelude::*;
            const B: usize = 32;
            let mut out = w.to_vec();
            let p = out.as_mut_ptr() as usize;
            (0..n).into_par_iter().for_each(|col| {
                let out = unsafe { std::slice::from_raw_parts_mut(p as *mut f32, k * n) };
                let mut row = 0;
                while row < k {
                    let len = (row + B).min(k) - row;
                    mx_block_fmt(out, row * n + col, n, len, ScaledFormat::F6E2M3);
                    row += len;
                }
            });
            out
        }
        WeightQuant::Int4G64 => {
            // int4 symmetric (±7), group-64 along the input dim K, per column.
            const G: usize = 64;
            let mut out = w.to_vec();
            for col in 0..n {
                let mut row = 0;
                while row < k {
                    let end = (row + G).min(k);
                    let mut amax = 0f32;
                    for r in row..end {
                        amax = amax.max(w[r * n + col].abs());
                    }
                    let s = if amax > 0.0 { amax / 7.0 } else { 1.0 };
                    for r in row..end {
                        out[r * n + col] = (w[r * n + col] / s).round().clamp(-7.0, 7.0) * s;
                    }
                    row = end;
                }
            }
            out
        }
        WeightQuant::Nf4 => {
            // NormalFloat-4, block-64 absmax along the input dim K, per column.
            const G: usize = 64;
            let mut out = w.to_vec();
            for col in 0..n {
                let mut row = 0;
                while row < k {
                    let end = (row + G).min(k);
                    let mut amax = 0f32;
                    for r in row..end {
                        amax = amax.max(w[r * n + col].abs());
                    }
                    let s = if amax > 0.0 { amax } else { 1.0 };
                    for r in row..end {
                        out[r * n + col] = nf4_quant(w[r * n + col] / s) * s;
                    }
                    row = end;
                }
            }
            out
        }
        WeightQuant::Mxfp4 => {
            // FP4 e2m1, e8m0 block-32 along the input dim K, per output column.
            // Parallel over columns (disjoint strided writes, as Int8Ch).
            use rayon::prelude::*;
            const B: usize = 32;
            let mut out = w.to_vec();
            let p = out.as_mut_ptr() as usize;
            (0..n).into_par_iter().for_each(|col| {
                let out = unsafe { std::slice::from_raw_parts_mut(p as *mut f32, k * n) };
                let mut row = 0;
                while row < k {
                    let len = (row + B).min(k) - row;
                    mxfp4_block(out, row * n + col, n, len);
                    row += len;
                }
            });
            out
        }
        WeightQuant::Int4Mix => {
            // outlier-channel mixed precision: the top-`frac` highest-amax output
            // columns stay per-channel int8, the rest go int4-g64. Uses exactly the
            // per-channel outlier structure the recording found.
            let frac: f32 = std::env::var("RLX_KIMI_INT4MIX_FRAC")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.125);
            let mut amax_col = vec![0f32; n];
            for col in 0..n {
                let mut a = 0f32;
                for row in 0..k {
                    a = a.max(w[row * n + col].abs());
                }
                amax_col[col] = a;
            }
            // cutoff = the amax above which a column is treated as int8.
            let mut sorted = amax_col.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = ((1.0 - frac) * n as f32) as usize;
            let cutoff = sorted[idx.min(n.saturating_sub(1))];

            let mut out = w.to_vec();
            for col in 0..n {
                if amax_col[col] >= cutoff {
                    // per-channel int8.
                    let s = if amax_col[col] > 0.0 {
                        amax_col[col] / 127.0
                    } else {
                        1.0
                    };
                    for row in 0..k {
                        out[row * n + col] =
                            (w[row * n + col] / s).round().clamp(-127.0, 127.0) * s;
                    }
                } else {
                    // int4-g64.
                    const G: usize = 64;
                    let mut row = 0;
                    while row < k {
                        let end = (row + G).min(k);
                        let mut a = 0f32;
                        for r in row..end {
                            a = a.max(w[r * n + col].abs());
                        }
                        let s = if a > 0.0 { a / 7.0 } else { 1.0 };
                        for r in row..end {
                            out[r * n + col] = (w[r * n + col] / s).round().clamp(-7.0, 7.0) * s;
                        }
                        row = end;
                    }
                }
            }
            out
        }
    }
}

/// Register a param `data` of `shape` under `name` and return its node.
pub fn reg(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: Vec<f32>,
    shape: &[usize],
) -> HirNodeId {
    debug_assert_eq!(
        data.len(),
        shape.iter().product::<usize>(),
        "{name} shape mismatch"
    );
    params.insert(name.to_string(), data);
    g.param(name, Shape::new(shape, DType::F32))
}

/// `x[.,in] @ w[in,out]`, registering `w` (row-major `[in, out]`) under
/// `{prefix}.{name}`. HF `nn.Linear` weights are `[out, in]` and must be
/// transposed by the loader before being passed here.
#[allow(clippy::too_many_arguments)]
pub fn linear(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    name: &str,
    x: HirNodeId,
    w: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> HirNodeId {
    let full = format!("{prefix}.{name}");
    if let Some((afmt, wfmt, lay)) = scaled_backbone() {
        return emit_scaled_linear(g, params, &full, x, w, in_dim, out_dim, afmt, wfmt, lay);
    }
    if int8_backbone_active() {
        return emit_int8_resident(g, params, &full, x, w, in_dim, out_dim);
    }
    if bf16_backbone_active() {
        // bf16-resident: emit a BF16 weight `param` (bytes fed post-compile) and
        // LOSSLESSLY downconvert the (bf16-sourced) f32 weight. Layout `[in,out]` =
        // `[K,N]` — what every backend's bf16 matmul kernel expects for
        // `g.mm(x[M,K], w[K,N])`. On a graph-cached layer this bakes bf16 into the
        // cached graph (the f32 source is transient) → genuinely bf16-resident.
        let wid = g.param(&full, Shape::new(&[in_dim, out_dim], DType::BF16));
        let bytes: Vec<u8> = w
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        BF16_BACKBONE.with(|c| {
            if let Some(v) = c.borrow_mut().as_mut() {
                v.push((full, bytes));
            }
        });
        g.mm(x, wid)
    } else {
        // Optional fake-quant of the projection weight (RLX_KIMI_QUANT) so we can
        // measure the precision of int8/int4 schemes against the bf16 baseline.
        // Name-aware so the `mixed` policy can keep the sensitive projections
        // (e.g. `o_proj`, which the recurrence amplifies) at int8.
        let wq = resolve_quant(name);
        let wd = if wq == WeightQuant::None {
            w.to_vec()
        } else {
            fake_quant_weight(w, in_dim, out_dim, wq)
        };
        let wid = reg(g, params, &full, wd, &[in_dim, out_dim]);
        // Optional activation fake-quant (RLX_KIMI_ACT_QUANT): round-trip x through
        // scaled_quantize → ScaledDequantize to inject the A8 (MXFP8) activation
        // error, then keep the FAST f32 matmul. This measures true W×A8 accuracy
        // without the slow ScaledMatMul decode-and-accumulate oracle.
        let x = if let Some((afmt, lay)) = act_quant() {
            act_fakequant(g, x, afmt, lay)
        } else {
            x
        };
        g.mm(x, wid)
    }
}

/// Reader for the activation fake-quant (`RLX_KIMI_ACT_QUANT`): `fp8` → MXFP8
/// (F8E4M3, block-32 e8m0), `fp8t` → per-tensor FP8. `None` when unset. Composes
/// with the weight fake-quant (`RLX_KIMI_QUANT`) to form W×A8 configs.
pub fn act_quant() -> Option<(ScaledFormat, ScaleLayout)> {
    match std::env::var("RLX_KIMI_ACT_QUANT").ok().as_deref() {
        Some("fp8") | Some("mxfp8") | Some("a8") => Some((ScaledFormat::F8E4M3, ScaleLayout::mx())),
        Some("fp8t") => Some((ScaledFormat::F8E4M3, ScaleLayout::PerTensor)),
        _ => None,
    }
}

/// Round-trip `x` through quantize → dequantize at `fmt`/`lay` — an f32→f32
/// fake-quant that injects the activation quantization error (elementwise, fast)
/// while leaving the downstream matmul in f32.
pub fn act_fakequant(
    g: &mut HirMut,
    x: HirNodeId,
    fmt: ScaledFormat,
    lay: ScaleLayout,
) -> HirNodeId {
    let xs = g.shape(x).clone();
    let (codes, scale) = g.scaled_quantize(x, fmt, lay);
    g.add_node(
        Op::ScaledDequantize {
            format: fmt,
            scale_layout: lay,
        },
        vec![codes, scale],
        xs.with_dtype(DType::F32),
    )
}

/// A broadcastable f32 scalar constant node (shape `[1]`).
pub fn scalar_const(g: &mut HirMut, value: f32) -> HirNodeId {
    g.add_node(
        Op::Constant {
            data: value.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F32),
    )
}

/// An elementwise activation op preserving the input `shape`.
pub fn act(g: &mut HirMut, kind: Activation, x: HirNodeId, shape: Shape) -> HirNodeId {
    g.add_node(Op::Activation(kind), vec![x], shape)
}

/// Sigmoid via the activation op (there is no `HirGraphExt::sigmoid`), preserving
/// the input shape.
pub fn sigmoid(g: &mut HirMut, x: HirNodeId, shape: Shape) -> HirNodeId {
    act(g, Activation::Sigmoid, x, shape)
}

/// The Kimi **situ** GLU activation applied to a concatenated `gate_up` tensor
/// `[rows, 2*d]` (last-axis split into `gate = [..:d]`, `up = [..d:]`):
///
/// ```text
///   situ_a = beta * tanh(gate / beta) * sigmoid(gate)
///   up'    = linear_beta * tanh(up / linear_beta)   (only if linear_beta set)
///   out    = situ_a * up'                            -> [rows, d]
/// ```
///
/// `tanh(g)·sigmoid(g)` — distinct from silu (`g·sigmoid(g)`).
pub fn situ(
    g: &mut HirMut,
    gate_up: HirNodeId,
    rows: usize,
    d: usize,
    beta: f32,
    linear_beta: Option<f32>,
) -> HirNodeId {
    let f = DType::F32;
    let half = Shape::new(&[rows, d], f);
    let gate = g.narrow_(gate_up, 1, 0, d);
    let up = g.narrow_(gate_up, 1, d, d);

    // situ_a = beta * tanh(gate / beta) * sigmoid(gate)
    let beta_c = scalar_const(g, beta);
    let gate_scaled = g.div(gate, beta_c);
    let gate_tanh = g.tanh(gate_scaled);
    let beta_c2 = scalar_const(g, beta);
    let situ_a = g.mul(beta_c2, gate_tanh);
    let gate_sig = sigmoid(g, gate, half.clone());
    let situ_a = g.mul(situ_a, gate_sig);

    // up' = linear_beta * tanh(up / linear_beta), else up.
    let up = match linear_beta {
        Some(lb) => {
            let lb_c = scalar_const(g, lb);
            let up_scaled = g.div(up, lb_c);
            let up_tanh = g.tanh(up_scaled);
            let lb_c2 = scalar_const(g, lb);
            g.mul(lb_c2, up_tanh)
        }
        None => up,
    };

    g.mul(situ_a, up)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::flow_util::{built_from_hir, compile_built};
    use rlx_ir::hir::HirModule;
    use rlx_runtime::Device;
    use std::collections::HashMap;

    // Reference situ on the host.
    fn situ_ref(gate: f32, up: f32, beta: f32, linear_beta: Option<f32>) -> f32 {
        let situ_a = beta * (gate / beta).tanh() * (1.0 / (1.0 + (-gate).exp()));
        let up = match linear_beta {
            Some(lb) => lb * (up / lb).tanh(),
            None => up,
        };
        situ_a * up
    }

    #[test]
    fn situ_matches_reference() {
        let (rows, d) = (2usize, 3usize);
        let beta = 4.0f32;
        let linear_beta = Some(25.0f32);

        let mut hir = HirModule::new("situ_test");
        let mut g = HirMut::new(&mut hir);
        let x = g.input("x", Shape::new(&[rows, 2 * d], DType::F32));
        let out = situ(&mut g, x, rows, d, beta, linear_beta);
        g.set_outputs(vec![out]);
        let built = built_from_hir(hir, HashMap::new()).expect("build situ graph");
        let mut compiled = compile_built(built, Device::Cpu).expect("compile situ");

        // gate = first d, up = last d, per row.
        let gate = [0.5f32, -1.2, 2.0, -0.3, 0.8, 1.5];
        let up = [1.0f32, 0.4, -0.7, 2.2, -1.1, 0.6];
        let mut xin = Vec::new();
        for r in 0..rows {
            xin.extend_from_slice(&gate[r * d..r * d + d]);
            xin.extend_from_slice(&up[r * d..r * d + d]);
        }
        let y = compiled
            .run(&[("x", xin.as_slice())])
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(y.len(), rows * d);
        for i in 0..rows * d {
            let want = situ_ref(gate[i], up[i], beta, linear_beta);
            assert!((y[i] - want).abs() < 1e-5, "situ[{i}] = {} != {want}", y[i]);
        }
    }
}
