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

//! CPU evaluator for the Espresso graphs the NMT ships in.
//!
//! Eleven layer types cover every graph in the model (verified by
//! `tests/real_graphs.rs`): `quantized_gather`, `elementwise`,
//! `dynamic_quantize`, `inner_product`, `dynamic_dequantize`, `reshape`,
//! `transpose`, `batch_matmul`, `softmax`, `instancenorm_1d`, `copy`.
//!
//! # Quantization
//!
//! Weights are int8 with a per-matrix divisor (`w_quantization_scale` on the
//! following `dynamic_dequantize`); activations are quantized per tensor at run
//! time by `dynamic_quantize`, which emits both `<x>.q` and `<x>.q_scale`.
//! Dequantization is therefore
//!
//! ```text
//!   y = acc / (act_scale · w_scale) + bias        (then ReLU if has_relu)
//! ```
//!
//! with both scales stored as *divisors* (`127 / max|·|`).
//!
//! Embedding tables use `quantized_gather`, which is **companded**, not affine:
//! `Q_meta` holds four increasing f32 per column, and a byte is mapped through
//! the resulting 3-segment piecewise-linear curve. See [`dequant_gather_row`].
//! That layout is inferred from the shipped tables rather than documented, so
//! it is isolated in one function and flagged here.

use crate::net::{Graph, Layer};
use crate::tensor::Tensor;
use anyhow::{Result, anyhow, bail, ensure};
use rayon::prelude::*;
use std::collections::HashMap;

/// A value flowing between layers.
#[derive(Debug, Clone)]
pub enum Value {
    /// Ordinary activation.
    F32(Tensor),
    /// Output of `dynamic_quantize`.
    Q8 { dims: Vec<usize>, data: Vec<i8> },
    /// Integer accumulator out of `inner_product`.
    Acc { dims: Vec<usize>, data: Vec<i32> },
    /// f32 accumulator from the quantization-bypass path.
    AccF32 { dims: Vec<usize>, data: Vec<f32> },
    /// The scale companion emitted beside a `Q8`.
    Scale(f32),
}

impl Value {
    /// Borrows as an activation, erroring if it is not one.
    pub fn f32(&self) -> Result<&Tensor> {
        match self {
            Self::F32(t) => Ok(t),
            other => bail!("expected an f32 tensor, found {}", other.kind()),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::F32(_) => "f32",
            Self::Q8 { .. } => "int8",
            Self::Acc { .. } => "int32",
            Self::AccF32 { .. } => "f32-acc",
            Self::Scale(_) => "scale",
        }
    }
}

/// Named blobs during a graph run.
pub type Env = HashMap<String, Value>;

/// Whether q·kᵀ products are divided by `sqrt(head_dim)`.
///
/// **Off by default, because it was measured and made things worse.** Espresso
/// emits no scale layer. Without scaling the shipped input net gives softmax
/// entropies 0.00 / 2.50 / 3.97 / 3.35 (layer 0 saturated, the rest sane); with
/// scaling they become 0.17 / 4.14 / 4.15 / 4.14 — layers 1-3 collapse to
/// uniform attention (max entropy is 4.159) and the encoder's token
/// discrimination degrades from cos 0.991 to 0.9975. So the factor is not
/// uniformly missing; layer 0's saturation has some other cause.
///
/// Set `RLX_TRANSLATE_ATTN_SCALE=1` to re-enable for experiments.
/// Bypass int8 activation quantization, using the f32 fold
/// `y = x·(W/w_scale) + b` instead (validated at cosine 0.999993 against the
/// int8 path on a real encoder linear). Set `RLX_TRANSLATE_F32=1`.
///
/// Exists to separate "our int8 activation path is wrong" from "something else
/// is wrong": if token discrimination improves markedly with this on, the fault
/// is in the quantization, not the graph wiring.
/// Diagnostic multiplier applied to every `dynamic_dequantize` output.
///
/// Used to test whether the dequantization scale is systematically off: sweep
/// it and see whether any value restores token discrimination through the
/// encoder. 1.0 (no change) unless `RLX_TRANSLATE_DEQUANT_GAIN` is set.
pub fn dequant_gain() -> f32 {
    std::env::var("RLX_TRANSLATE_DEQUANT_GAIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0)
}

/// Treat `inner_product` weights as `[nB, nC]` (input-major) instead of the
/// default `[nC, nB]` (output-major).
///
/// Both orientations hold nB*nC values and yield an nC-wide output, so no shape
/// check can distinguish them — a transposed weight matrix produces perfectly
/// well-formed garbage. Set `RLX_TRANSLATE_WT=1` to compare.
pub fn weights_are_input_major() -> bool {
    matches!(std::env::var("RLX_TRANSLATE_WT").as_deref(), Ok("1"))
}

pub fn bypass_activation_quantization() -> bool {
    matches!(std::env::var("RLX_TRANSLATE_F32").as_deref(), Ok("1"))
}

pub fn scale_attention_scores() -> bool {
    matches!(
        std::env::var("RLX_TRANSLATE_ATTN_SCALE").as_deref(),
        Ok("1")
    )
}

/// Runs `graph` with `inputs` bound, returning the full environment so callers
/// can read intermediate blobs (the decoder's `.next` state tensors, the
/// handover K/V, an alignment layer) and not just the final output.
pub fn run(graph: &Graph, inputs: Env) -> Result<Env> {
    let mut env = inputs;
    for layer in &graph.layers {
        timed(graph, layer, &mut env)?;
    }
    Ok(env)
}

/// Runs `graph` checking every produced blob's **full geometry** against
/// `.espresso.shape`.
///
/// Stricter than [`run_checked`], which only compares the innermost width so it
/// can tolerate the length-polymorphic embedding graph. Attention tensors are
/// all `[8, 64, 64]` in element count and width, so only a full-dims comparison
/// can catch a wrong permutation or contraction axis there — which is exactly
/// where a layout bug would hide.
pub fn run_strict(graph: &Graph, inputs: Env) -> Result<Env> {
    let mut env = inputs;
    for layer in &graph.layers {
        timed(graph, layer, &mut env)?;
        for top in &layer.tops {
            let Some(Value::F32(t)) = env.get(top) else {
                continue;
            };
            let Some(want) = graph.declared_shape(top) else {
                continue;
            };
            let expect = want.dims();
            // Ignore leading 1s on either side: a rank-3 [1,64,512] and a
            // rank-2 [64,512] describe the same buffer.
            let trim = |d: &[usize]| -> Vec<usize> {
                let mut v = d.to_vec();
                while v.len() > 1 && v[0] == 1 {
                    v.remove(0);
                }
                v
            };
            ensure!(
                trim(t.dims()) == trim(&expect),
                "layer {:?} ({}) produced {:?} but {top:?} is declared {:?}",
                layer.name,
                layer.kind,
                t.dims(),
                expect
            );
        }
    }
    Ok(env)
}

/// Runs `graph` and additionally checks every produced blob against the
/// geometry declared in `.espresso.shape`.
///
/// The shape file covers every intermediate, so this turns a wiring or reshape
/// mistake into an error naming the offending layer instead of wrong numbers
/// many layers later.
pub fn run_checked(graph: &Graph, inputs: Env) -> Result<Env> {
    let mut env = inputs;
    for layer in &graph.layers {
        timed(graph, layer, &mut env)?;
        for top in &layer.tops {
            let Some(Value::F32(t)) = env.get(top) else {
                continue;
            };
            let Some(want) = graph.declared_shape(top) else {
                continue;
            };
            // Attention scores are `[heads, queries, keys]`, so their innermost
            // dimension is the key sequence length, not a weight-fixed width.
            // A shorter source legitimately narrows them, and `softmax`
            // inherits the same shape. Every other layer's width is pinned by
            // its weights and is still checked below.
            if matches!(layer.kind.as_str(), "batch_matmul" | "softmax") {
                continue;
            }
            // The shared embedding graph is length-polymorphic: `.espresso.shape`
            // records the 64-token source case, but the decoder reuses it for a
            // single token (its own declared input is `[1, 512]`). So the check
            // is on the innermost width, which is fixed by the weights, plus a
            // whole-number-of-rows requirement — enough to catch a wrong reshape
            // or transpose without rejecting a legitimate shorter run.
            ensure!(
                t.width() == want.w,
                "layer {:?} produced width {} but {top:?} is declared width {}",
                layer.name,
                t.width(),
                want.w
            );
            ensure!(
                want.w == 0 || t.len().is_multiple_of(want.w),
                "layer {:?} produced {} elements, not a whole number of {}-wide rows for {top:?}",
                layer.name,
                t.len(),
                want.w
            );
        }
    }
    Ok(env)
}

/// [`eval`] plus the per-op tally, which every entry point shares.
///
/// The decoder runs graphs through `run_checked`, not `run`, so instrumenting
/// only the latter recorded nothing at all — the first version of this did
/// exactly that and produced an empty table.
fn timed(graph: &Graph, layer: &Layer, env: &mut Env) -> Result<()> {
    let started = crate::profile::enabled().then(std::time::Instant::now);
    let r = eval(graph, layer, env)
        .map_err(|e| anyhow!("layer {:?} ({}): {e:#}", layer.name, layer.kind));
    if let Some(t) = started {
        crate::profile::record(&layer.kind, &graph.name, t.elapsed());
    }
    r
}

fn get<'a>(env: &'a Env, name: &str) -> Result<&'a Value> {
    env.get(name)
        .ok_or_else(|| anyhow!("blob {name:?} is not bound"))
}

fn eval(graph: &Graph, l: &Layer, env: &mut Env) -> Result<()> {
    match l.kind.as_str() {
        "copy" => {
            let v = get(env, &l.bottoms[0])?.clone();
            env.insert(l.tops[0].clone(), v);
        }
        "reshape" => {
            let t = get(env, &l.bottoms[0])?.f32()?;
            // Espresso writes the destination innermost-last as (k, h, w),
            // trimmed to `dst_nd_rank`; -1 means "solve me".
            let rank = l.int("dst_nd_rank").unwrap_or(3).clamp(1, 3) as usize;
            let raw = [l.int("dst_k"), l.int("dst_h"), l.int("dst_w")];
            let dims: Vec<Option<usize>> = raw[3 - rank..]
                .iter()
                .map(|d| match d {
                    Some(v) if *v >= 0 => Some(*v as usize),
                    _ => None,
                })
                .collect();
            env.insert(l.tops[0].clone(), Value::F32(t.reshape(&dims)?));
        }
        "transpose" => {
            let t = get(env, &l.bottoms[0])?.f32()?;
            env.insert(l.tops[0].clone(), Value::F32(transpose(t, l)?));
        }
        "elementwise" => {
            env.insert(l.tops[0].clone(), Value::F32(elementwise(l, env)?));
        }
        "softmax" => {
            let t = get(env, &l.bottoms[0])?.f32()?;
            env.insert(l.tops[0].clone(), Value::F32(softmax_last(t)?));
        }
        "instancenorm_1d" => {
            let t = get(env, &l.bottoms[0])?.f32()?;
            // 1344 calls a translation, same two vectors every time.
            let gamma = graph.weights.f32s_shared(l.req_blob("wGamma")?)?;
            let beta = graph.weights.f32s_shared(l.req_blob("wBeta")?)?;
            let eps = l.float("eps").unwrap_or(1e-6) as f32;
            env.insert(
                l.tops[0].clone(),
                Value::F32(layernorm(t, &gamma, &beta, eps)?),
            );
        }
        "batch_matmul" => {
            let a = get(env, &l.bottoms[0])?.f32()?;
            let b = get(env, &l.bottoms[1])?.f32()?;
            let transpose_y = l.flag("transpose_y");
            let mut out = batch_matmul(a, b, transpose_y)?;
            // Attention score scaling. Espresso emits no explicit scale layer,
            // and the query weights do NOT absorb it: measured on the shipped
            // encoder, the first softmax saw a logit RMS of 58 and saturated to
            // max-prob 1.0000 / entropy 0.000, which averages every position
            // together and destroys the token signal. Dividing the q·kᵀ product
            // by sqrt(head_dim) restores a sane range.
            //
            // Only the `transpose_y` product is scaled: that is the q·kᵀ form.
            // The second attention matmul (probs·v) has no `transpose_y` and
            // must not be touched.
            if transpose_y && scale_attention_scores() {
                let head_dim = a.width().max(1) as f32;
                let inv = 1.0 / head_dim.sqrt();
                for v in out.data_mut() {
                    *v *= inv;
                }
            }
            env.insert(l.tops[0].clone(), Value::F32(out));
        }
        "quantized_gather" => {
            let idx = get(env, &l.bottoms[0])?.f32()?;
            let out = quantized_gather(graph, l, idx)?;
            env.insert(l.tops[0].clone(), Value::F32(out));
        }
        "dynamic_quantize" if bypass_activation_quantization() => {
            // Pass the f32 activation straight through; `inner_product` and
            // `dynamic_dequantize` pick it up below.
            let t = get(env, &l.bottoms[0])?.f32()?.clone();
            ensure!(
                l.tops.len() >= 2,
                "dynamic_quantize must emit a value and a scale"
            );
            env.insert(l.tops[0].clone(), Value::F32(t));
            env.insert(l.tops[1].clone(), Value::Scale(1.0));
        }
        "dynamic_quantize" => {
            let t = get(env, &l.bottoms[0])?.f32()?;
            let (q, scale) = dynamic_quantize(t);
            ensure!(
                l.tops.len() >= 2,
                "dynamic_quantize must emit a value and a scale"
            );
            env.insert(l.tops[0].clone(), q);
            env.insert(l.tops[1].clone(), Value::Scale(scale));
        }
        "inner_product" => {
            let x = get(env, &l.bottoms[0])?;
            let out = inner_product(graph, l, x)?;
            env.insert(l.tops[0].clone(), out);
        }
        "dynamic_dequantize" => {
            let acc = get(env, &l.bottoms[0])?;
            let act = match get(env, &l.bottoms[1])? {
                Value::Scale(s) => *s,
                other => bail!("dequantize expected a scale, found {}", other.kind()),
            };
            let out = dynamic_dequantize(graph, l, acc, act)?;
            env.insert(l.tops[0].clone(), Value::F32(out));
        }
        other => bail!("unsupported Espresso layer type {other:?}"),
    }
    Ok(())
}

/// Espresso stores a permutation as `axis_w/k/h/n/seq`, naming for each *source*
/// axis where it lands. Only the axes within the tensor's rank participate.
fn transpose(t: &Tensor, l: &Layer) -> Result<Tensor> {
    let rank = t.rank();
    // Espresso's axis ids are innermost-first (w=0, h=1, k=2 …) whereas our
    // dims are outermost-first, so ids are mirrored into our index space.
    let ids = ["axis_w", "axis_h", "axis_k", "axis_n", "axis_seq"];
    let mut dest = Vec::with_capacity(rank);
    for i in 0..rank {
        // Source axis `i` (outermost-first) is Espresso axis `rank-1-i`.
        let key = ids[rank - 1 - i];
        let a = l.int(key).unwrap_or((rank - 1 - i) as i64);
        ensure!(a >= 0, "{key} is negative");
        dest.push(a as usize);
    }
    // `dest[i]` = Espresso destination id of our axis i; convert back and invert.
    let mut perm = vec![0usize; rank];
    for (src, &d) in dest.iter().enumerate() {
        ensure!(d < rank, "transpose axis {d} out of range for rank {rank}");
        perm[rank - 1 - d] = src;
    }
    t.permute(&perm)
}

fn elementwise(l: &Layer, env: &Env) -> Result<Tensor> {
    let op = l.int("operation").unwrap_or(0);
    let alpha = l.float("alpha").unwrap_or(1.0) as f32;
    let beta = l.float("beta").unwrap_or(0.0) as f32;
    let a = get(env, &l.bottoms[0])?.f32()?;
    match op {
        // add
        0 => {
            let b = get(
                env,
                l.bottoms
                    .get(1)
                    .ok_or_else(|| anyhow!("add needs two inputs"))?,
            )?
            .f32()?;
            broadcast_zip(a, b, |x, y| x + y)
        }
        // multiply
        1 => {
            let b = get(
                env,
                l.bottoms
                    .get(1)
                    .ok_or_else(|| anyhow!("mul needs two inputs"))?,
            )?
            .f32()?;
            broadcast_zip(a, b, |x, y| x * y)
        }
        // scale by alpha (plus beta)
        3 => Tensor::new(
            a.dims().to_vec(),
            a.data().iter().map(|v| v * alpha + beta).collect(),
        ),
        // reciprocal — the decoder's 1/position for its average-attention mean
        10 => Tensor::new(
            a.dims().to_vec(),
            a.data()
                .iter()
                .map(|v| if *v == 0.0 { 0.0 } else { alpha / v })
                .collect(),
        ),
        other => bail!("unsupported elementwise operation {other}"),
    }
}

/// Elementwise combine, broadcasting `b` when it is a single row or scalar.
fn broadcast_zip(a: &Tensor, b: &Tensor, f: impl Fn(f32, f32) -> f32) -> Result<Tensor> {
    if a.dims() == b.dims() {
        return Tensor::new(
            a.dims().to_vec(),
            a.data()
                .iter()
                .zip(b.data())
                .map(|(x, y)| f(*x, *y))
                .collect(),
        );
    }
    if b.len() == 1 {
        let y = b.data()[0];
        return Tensor::new(
            a.dims().to_vec(),
            a.data().iter().map(|x| f(*x, y)).collect(),
        );
    }
    if b.len() == a.width() {
        let w = a.width();
        let mut out = Vec::with_capacity(a.len());
        for (i, x) in a.data().iter().enumerate() {
            out.push(f(*x, b.data()[i % w]));
        }
        return Tensor::new(a.dims().to_vec(), out);
    }
    // A single trailing row broadcast across rows (the AAN's 1/position).
    if a.len().is_multiple_of(b.len()) {
        let n = b.len();
        let mut out = Vec::with_capacity(a.len());
        let per = a.len() / n;
        for (i, x) in a.data().iter().enumerate() {
            out.push(f(*x, b.data()[i / per]));
        }
        return Tensor::new(a.dims().to_vec(), out);
    }
    bail!("cannot broadcast {:?} against {:?}", a.dims(), b.dims())
}

/// Softmax over the innermost axis.
fn softmax_last(t: &Tensor) -> Result<Tensor> {
    let w = t.width();
    ensure!(w > 0, "softmax on an empty axis");
    let mut out = t.data().to_vec();
    for r in out.chunks_exact_mut(w) {
        let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !m.is_finite() {
            continue;
        }
        let mut sum = 0.0f32;
        for v in r.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in r.iter_mut() {
                *v /= sum;
            }
        }
    }
    Tensor::new(t.dims().to_vec(), out)
}

/// LayerNorm over the innermost axis (`tf_layernorm`).
fn layernorm(t: &Tensor, gamma: &[f32], beta: &[f32], eps: f32) -> Result<Tensor> {
    let w = t.width();
    ensure!(
        gamma.len() == w && beta.len() == w,
        "layernorm gamma/beta are {}/{} for width {w}",
        gamma.len(),
        beta.len()
    );
    let mut out = t.data().to_vec();
    for r in out.chunks_exact_mut(w) {
        let n = w as f32;
        let mean = r.iter().sum::<f32>() / n;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + eps).sqrt();
        for (i, v) in r.iter_mut().enumerate() {
            *v = (*v - mean) * inv * gamma[i] + beta[i];
        }
    }
    Tensor::new(t.dims().to_vec(), out)
}

/// Batched matmul over the trailing two axes, optionally transposing `b`.
fn batch_matmul(a: &Tensor, b: &Tensor, transpose_y: bool) -> Result<Tensor> {
    ensure!(
        a.rank() >= 2 && b.rank() >= 2,
        "batch_matmul needs rank >= 2"
    );
    let (m, k) = (a.dims()[a.rank() - 2], a.dims()[a.rank() - 1]);
    let (bk, bn) = (b.dims()[b.rank() - 2], b.dims()[b.rank() - 1]);
    let (kb, n) = if transpose_y { (bn, bk) } else { (bk, bn) };
    ensure!(k == kb, "inner dimensions {k} and {kb} disagree");

    let a_batch: usize = a.dims()[..a.rank() - 2].iter().product();
    let b_batch: usize = b.dims()[..b.rank() - 2].iter().product();
    let batch = a_batch.max(b_batch);
    ensure!(
        a_batch == batch || a_batch == 1,
        "batch {a_batch} cannot broadcast to {batch}"
    );
    ensure!(
        b_batch == batch || b_batch == 1,
        "batch {b_batch} cannot broadcast to {batch}"
    );

    let mut out = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        let ao = if a_batch == 1 { 0 } else { bi * m * k };
        let bo = if b_batch == 1 { 0 } else { bi * bk * bn };
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    let av = a.data()[ao + i * k + kk];
                    let bv = if transpose_y {
                        b.data()[bo + j * bn + kk]
                    } else {
                        b.data()[bo + kk * bn + j]
                    };
                    acc += av * bv;
                }
                out[bi * m * n + i * n + j] = acc;
            }
        }
    }
    let mut dims: Vec<usize> = if a_batch == batch {
        a.dims()[..a.rank() - 2].to_vec()
    } else {
        b.dims()[..b.rank() - 2].to_vec()
    };
    dims.push(m);
    dims.push(n);
    Tensor::new(dims, out)
}

/// Per-tensor activation quantization. The emitted scale is a **divisor**:
/// `q = round(x · s)` with `s = 127 / max|x|`, so dequantization divides by it.
fn dynamic_quantize(t: &Tensor) -> (Value, f32) {
    let peak = t.data().iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if peak > 0.0 { 127.0 / peak } else { 1.0 };
    let data = t
        .data()
        .iter()
        .map(|v| (v * scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (
        Value::Q8 {
            dims: t.dims().to_vec(),
            data,
        },
        scale,
    )
}

/// int8 × int8 → int32. Weights are `[out, in]` row-major (`nC` × `nB`).
/// `sum(xs[i] * ws[i])` over int8, in i32.
///
/// The inner loop of `inner_product`, which is 78-81% of executor time — so it
/// looks like the obvious thing to hand-vectorise. **It is already fast.**
/// `examples/dot_bench.rs` measures this loop at ~75 GMAC/s standalone and
/// ~50 GMAC/s in the GEMV shape the decoder runs, which is near what the core
/// can do; LLVM vectorises the zip perfectly well.
///
/// A NEON version was written and measured against it: `vmull_s8` +
/// `vpadalq_s16` (the widening multiply and pairwise accumulate, which are
/// stable, unlike `vdotq_s32`) ran at 34 GMAC/s at n=512 and 5 GMAC/s at
/// n=2048 — 2x to 14x *slower*. It was deleted. If this is revisited, measure
/// against `dot_bench` first.
///
/// The 118-218 us per call the profiler reports against 21 us of measured
/// arithmetic is not a code problem: the development machine sits at load 78
/// on 14 cores, and the benchmark's best-of-5 filters out the preemption the
/// profiler's every-call mean includes.
#[inline]
pub fn dot_i8(xs: &[i8], ws: &[i8]) -> i32 {
    xs.iter()
        .zip(ws)
        .map(|(a, b)| i32::from(*a) * i32::from(*b))
        .sum()
}

/// Work below which threading costs more than it saves, in multiply-adds.
///
/// Measured on this machine: the cross-over is a few tens of thousands of MACs.
/// Below it the fan-out dominates, and there are thousands of small ops per
/// translation to pay it on.
const PARALLEL_MIN_MACS: usize = 1 << 16;

/// Threads to split a GEMM across; `RLX_TRANSLATE_GEMM_LANES` overrides.
///
/// **Defaults to 1 — single-threaded — because that is what measured fastest.**
/// `inner_product` is 81% of executor time and its output rows are independent,
/// so fanning it out looks like the obvious win. Interleaved A/B on the
/// development machine, which is shared and sat at load 78 on 14 cores:
///
/// | lanes | beam search |
/// | --- | --- |
/// | 1 | 926 / 945 ms |
/// | 4 | 1509 / 1252 ms |
/// | 14 | 1550 / 1147 ms |
///
/// Threads lose when the cores are already oversubscribed, and this machine is
/// never idle. On a quiet machine more lanes should win — that is the honest
/// reason this is a switch and not a deletion, and it is *not* something these
/// measurements establish.
pub fn gemm_lanes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("RLX_TRANSLATE_GEMM_LANES")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(1)
            .max(1)
    })
}

/// Fills `out[r * n_out + o]` from `f(r, o)`, across threads when it pays.
///
/// The output rows of a GEMM are independent, so this is embarrassingly
/// parallel; the only care needed is that a decoder step has `rows == 1`, so
/// splitting by row alone would leave a 512x2048 matmul on one core. Chunking
/// the *flat* output covers both shapes with one code path.
fn for_each_output<T, F>(out: &mut [T], rows: usize, n_out: usize, n_in: usize, f: F)
where
    T: Send,
    F: Fn(usize, usize) -> T + Sync,
{
    let fill = |slice: &mut [T], base: usize| {
        for (j, slot) in slice.iter_mut().enumerate() {
            let idx = base + j;
            *slot = f(idx / n_out, idx % n_out);
        }
    };
    let total = rows * n_out;
    let lanes = gemm_lanes();
    if lanes < 2 || total.saturating_mul(n_in) < PARALLEL_MIN_MACS {
        fill(out, 0);
        return;
    }
    let chunk = total.div_ceil(lanes).max(1);
    out.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(c, slice)| fill(slice, c * chunk));
}

fn inner_product(graph: &Graph, l: &Layer, x: &Value) -> Result<Value> {
    if let Value::F32(t) = x {
        // Bypass path: accumulate in f32 against the raw int8 weights. The
        // following dequantize divides by `w_scale` and adds the bias, so this
        // is exactly the export's folded linear.
        let n_in = l
            .int("nB")
            .ok_or_else(|| anyhow!("inner_product has no nB"))? as usize;
        let n_out = l
            .int("nC")
            .ok_or_else(|| anyhow!("inner_product has no nC"))? as usize;
        let w = graph.weights.i8s(l.req_blob("W_int8")?)?;
        ensure!(t.width() == n_in, "input width {} != nB {n_in}", t.width());
        let rows = t.len() / n_in;
        let mut out = vec![0.0f32; rows * n_out];
        let xall = t.data();
        for_each_output(&mut out, rows, n_out, n_in, |r, o| {
            let xs = &xall[r * n_in..(r + 1) * n_in];
            let ws = &w[o * n_in..(o + 1) * n_in];
            xs.iter().zip(ws).map(|(a, b)| a * f32::from(*b)).sum()
        });
        let mut d = t.dims()[..t.rank() - 1].to_vec();
        d.push(n_out);
        return Ok(Value::AccF32 { dims: d, data: out });
    }
    let Value::Q8 { dims, data } = x else {
        bail!(
            "inner_product expects a quantized input, found {}",
            x.kind()
        );
    };
    let n_in = l
        .int("nB")
        .ok_or_else(|| anyhow!("inner_product has no nB"))? as usize;
    let n_out = l
        .int("nC")
        .ok_or_else(|| anyhow!("inner_product has no nC"))? as usize;
    let w = graph.weights.i8s(l.req_blob("W_int8")?)?;
    ensure!(
        w.len() == n_in * n_out,
        "weight blob is {} bytes for {n_out}x{n_in}",
        w.len()
    );
    let width = *dims.last().unwrap_or(&n_in);
    ensure!(
        width == n_in,
        "input width {width} does not match nB {n_in}"
    );
    let rows = data.len() / n_in;

    let input_major = weights_are_input_major();
    let mut out = vec![0i32; rows * n_out];
    for_each_output(&mut out, rows, n_out, n_in, |r, o| {
        let xs = &data[r * n_in..(r + 1) * n_in];
        if input_major {
            // Strided gather; nothing to vectorise, and no shipped graph uses
            // this layout — it exists to test the reading.
            let mut acc = 0i32;
            for i in 0..n_in {
                acc += i32::from(xs[i]) * i32::from(w[i * n_out + o]);
            }
            return acc;
        }
        dot_i8(xs, &w[o * n_in..(o + 1) * n_in])
    });
    let mut d = dims[..dims.len() - 1].to_vec();
    d.push(n_out);
    Ok(Value::Acc { dims: d, data: out })
}

/// `y = acc / (act_scale · w_scale) + bias`, then ReLU when `has_relu`.
fn dynamic_dequantize(graph: &Graph, l: &Layer, acc: &Value, act_scale: f32) -> Result<Tensor> {
    let (dims, data): (Vec<usize>, Vec<f32>) = match acc {
        Value::Acc { dims, data } => (dims.clone(), data.iter().map(|v| *v as f32).collect()),
        Value::AccF32 { dims, data } => (dims.clone(), data.clone()),
        other => bail!("dequantize expects an accumulator, found {}", other.kind()),
    };
    let (dims, data) = (&dims, &data);
    let w_scale = l.float("w_quantization_scale").unwrap_or(1.0) as f32;
    let denom = act_scale * w_scale;
    ensure!(denom != 0.0, "dequantize scale is zero");
    let bias = match l.blob("biases") {
        Some(b) => graph.weights.f32s_shared(b)?,
        None => std::sync::Arc::from(&[][..]),
    };
    let width = *dims.last().unwrap_or(&1);
    ensure!(
        bias.is_empty() || bias.len() == width,
        "bias is {} wide for a {width}-wide output",
        bias.len()
    );
    let relu = l.flag("has_relu");
    let gain = dequant_gain();
    let out: Vec<f32> = data
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let mut v = *a / denom * gain;
            if !bias.is_empty() {
                v += bias[i % width];
            }
            if relu && v < 0.0 {
                v = 0.0;
            }
            v
        })
        .collect();
    Tensor::new(dims.clone(), out)
}

/// Gathers rows of a companded 8-bit table.
fn quantized_gather(graph: &Graph, l: &Layer, idx: &Tensor) -> Result<Tensor> {
    let cols = l.int("nCol").ok_or_else(|| anyhow!("gather has no nCol"))? as usize;
    let rows = l.int("nRow").ok_or_else(|| anyhow!("gather has no nRow"))? as usize;
    let bits = l.int("n_bits").unwrap_or(8);
    ensure!(
        bits == 8,
        "only 8-bit gather tables are supported, got {bits}"
    );
    let table = graph.weights.raw(l.req_blob("weights_u8")?)?;
    let meta = graph.weights.f32s_shared(l.req_blob("Q_meta")?)?;
    ensure!(
        table.len() == rows * cols,
        "gather table is {} bytes for {rows}x{cols}",
        table.len()
    );
    ensure!(
        meta.len() == cols * 4,
        "Q_meta is {} floats for {cols} columns (expected 4 per column)",
        meta.len()
    );

    let mut out = Vec::with_capacity(idx.len() * cols);
    for v in idx.data() {
        let r = *v as isize;
        ensure!(
            r >= 0 && (r as usize) < rows,
            "gather index {r} out of range for {rows} rows"
        );
        let src = &table[r as usize * cols..(r as usize + 1) * cols];
        for (c, byte) in src.iter().enumerate() {
            out.push(dequant_gather_row(*byte, &meta[c * 4..c * 4 + 4]));
        }
    }
    // Output is one row per index. Do NOT drop a trailing 1 — a single-token
    // gather must stay [1, cols], otherwise a one-step decode produces rank-1
    // [cols] and every downstream shape check is bypassed.
    let mut dims: Vec<usize> = idx.dims().to_vec();
    dims.push(cols);
    Tensor::new(dims, out)
}

/// Maps one byte through a column's 3-segment companding curve.
///
/// `q` is four increasing f32 acting as the values at index 0, 85, 170 and 255.
/// The middle two sit near the column mean (measured ratios 0.43 / 0.57 of the
/// range, not the 0.33 / 0.67 of uniform levels), which puts the fine
/// resolution around zero where embedding values concentrate.
///
/// **Inferred, not documented.** The evidence is that all 512 columns are
/// strictly increasing and that mid-index maps to ≈0. If translations come out
/// fluent but subtly wrong, re-examine this first.
pub fn dequant_gather_row(byte: u8, q: &[f32]) -> f32 {
    debug_assert_eq!(q.len(), 4);
    dequant_gather_row_scaled(byte, q) * gather_gain()
}

/// Diagnostic gain on gathered embeddings, to separate the curve's *shape* from
/// its *magnitude*. The `inner` curve improves decoded text purely by narrowing
/// the range ~7x, so magnitude is the variable worth isolating.
pub fn gather_gain() -> f32 {
    // Read once: this sits in the innermost loop of the readout table, which is
    // 168 000 x 512 elements, and an env lookup per element dominated the run.
    static GAIN: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *GAIN.get_or_init(|| {
        std::env::var("RLX_TRANSLATE_GATHER_GAIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0)
    })
}

/// Inner knots of the [`GatherCurve::Quartile`] curve, at bytes 64 and 192.
///
/// The four `Q_meta` values are the 0/25/75/100th percentiles of a uniformly
/// coded byte, which puts the knots here and is what makes a column Gaussian
/// (kurtosis 3.02 against 1.96 for the equal-thirds reading).
///
/// A cautionary note, because it cost real time: searching these knots against
/// the OS's output *while the source was terminated with the wrong piece* picked
/// 72/184 instead, consistently across four held-out slices. It was compensating
/// for an unrelated bug. With the source terminated correctly, 64/192 scores a
/// perfect 21/21 top-1 and 72/184 does not. Tuning a parameter on end-to-end
/// quality will happily absorb a defect somewhere else.
///
/// Overridable via `RLX_TRANSLATE_KNOTS="k1,k2"`.
pub fn quartile_knots() -> (f32, f32) {
    static KNOTS: std::sync::OnceLock<(f32, f32)> = std::sync::OnceLock::new();
    *KNOTS.get_or_init(|| {
        std::env::var("RLX_TRANSLATE_KNOTS")
            .ok()
            .and_then(|v| {
                let (a, b) = v.split_once(',')?;
                Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
            })
            .filter(|(a, b)| *a > 0.0 && b > a && *b < 255.0)
            .unwrap_or((64.0, 192.0))
    })
}

fn dequant_gather_row_scaled(byte: u8, q: &[f32]) -> f32 {
    let x = byte as f32;
    match gather_curve() {
        // Three linear segments through all four control points.
        GatherCurve::Companded => {
            let seg = 255.0 / 3.0;
            if x <= seg {
                q[0] + (q[1] - q[0]) * (x / seg)
            } else if x <= 2.0 * seg {
                q[1] + (q[2] - q[1]) * ((x - seg) / seg)
            } else {
                q[2] + (q[3] - q[2]) * (((x - 2.0 * seg) / seg).min(1.0))
            }
        }
        // `Q_meta` stores four per-column percentiles at 0/25/75/100%, and the
        // byte is a uniform quantile index, so the knots sit at bytes 0, 64,
        // 192 and 255. Measured over 30 000 rows of the shipped table this is
        // the only natural placement whose value distribution is Gaussian
        // (kurtosis 3.02); the three-equal-segment reading gives 1.96, far too
        // flat, and inflates the embedding norm by 1.6x.
        GatherCurve::Quartile => {
            let (k1, k2) = quartile_knots();
            if x <= k1 {
                q[0] + (q[1] - q[0]) * (x / k1)
            } else if x <= k2 {
                q[1] + (q[2] - q[1]) * ((x - k1) / (k2 - k1))
            } else {
                q[2] + (q[3] - q[2]) * ((x - k2) / (255.0 - k2))
            }
        }
        // The same knots, but eased within each segment instead of straight.
        // If the shipped curve is smooth, a 3-segment linear fit to it will
        // place its breakpoints away from the true quantiles, which is what the
        // knot search found (72/184 rather than 64/192).
        GatherCurve::Smooth => {
            let (k1, k2) = quartile_knots();
            let (lo, hi, a, b) = if x <= k1 {
                (0.0, k1, q[0], q[1])
            } else if x <= k2 {
                (k1, k2, q[1], q[2])
            } else {
                (k2, 255.0, q[2], q[3])
            };
            let f = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
            a + (b - a) * (f * f * (3.0 - 2.0 * f))
        }
        // Plain ramp between the outer control points.
        GatherCurve::AffineOuter => q[0] + (q[3] - q[0]) * (x / 255.0),
        // Plain ramp between the inner pair.
        GatherCurve::AffineInner => q[1] + (q[2] - q[1]) * (x / 255.0),
        // Four-entry palette selected by the top two bits.
        GatherCurve::Palette => q[(byte >> 6) as usize],
        // 256 = 4 x 64: top two bits select a control point, the low six
        // interpolate towards the next one (the last segment holds).
        GatherCurve::SegInterp => {
            let seg = (byte >> 6) as usize;
            let frac = f32::from(byte & 0x3f) / 64.0;
            let next = q[(seg + 1).min(3)];
            q[seg] + (next - q[seg]) * frac
        }
        // Same split, but the top segment extrapolates along its own slope
        // instead of holding flat.
        GatherCurve::SegExtrap => {
            let seg = (byte >> 6) as usize;
            let frac = f32::from(byte & 0x3f) / 64.0;
            if seg < 3 {
                q[seg] + (q[seg + 1] - q[seg]) * frac
            } else {
                q[3] + (q[3] - q[2]) * frac
            }
        }
    }
}

/// How a `quantized_gather` byte maps onto its column's four control points.
///
/// The shipped tables give no direct evidence for the curve *shape* — the
/// cosine-geometry test proved the per-column **layout** but was insensitive to
/// any monotonic reparameterisation. Selectable so the choice can be arbitrated
/// against decoded text, which is the only metric that has tracked correctness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatherCurve {
    Companded,
    Quartile,
    Smooth,
    AffineOuter,
    AffineInner,
    Palette,
    SegInterp,
    SegExtrap,
}

/// Reads `RLX_TRANSLATE_GATHER`; `quartile` is the default.
///
/// The alternatives are kept because each was a live hypothesis about how the
/// four `Q_meta` control points map onto a byte, and they are cheap to re-run.
pub fn gather_curve() -> GatherCurve {
    // Read once; see `gather_gain` for why.
    static CURVE: std::sync::OnceLock<GatherCurve> = std::sync::OnceLock::new();
    *CURVE.get_or_init(|| match std::env::var("RLX_TRANSLATE_GATHER").as_deref() {
        Ok("outer") => GatherCurve::AffineOuter,
        Ok("inner") => GatherCurve::AffineInner,
        Ok("palette") => GatherCurve::Palette,
        Ok("companded") => GatherCurve::Companded,
        Ok("smooth") => GatherCurve::Smooth,
        Ok("seg") => GatherCurve::SegInterp,
        Ok("segx") => GatherCurve::SegExtrap,
        _ => GatherCurve::Quartile,
    })
}

#[cfg(test)]
mod tests {

    #[test]
    fn the_dot_product_handles_the_extremes_and_every_tail_length() {
        // Deterministic pseudo-random i8s, including the extremes: -128 is the
        // one value whose negation does not fit, and a widening bug shows there
        // first.
        let mk = |seed: u64, n: usize| -> Vec<i8> {
            let mut s = seed;
            (0..n)
                .map(|i| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    match i % 37 {
                        0 => -128,
                        1 => 127,
                        _ => (s >> 33) as i8,
                    }
                })
                .collect()
        };
        // Lengths either side of the 16-wide and 32-wide steps, plus the real
        // contraction widths.
        for n in [0, 1, 15, 16, 17, 31, 32, 33, 63, 512, 2048] {
            let (a, b) = (mk(1, n), mk(2, n));
            let want: i32 = a
                .iter()
                .zip(&b)
                .map(|(x, w)| i32::from(*x) * i32::from(*w))
                .sum();
            assert_eq!(super::dot_i8(&a, &b), want, "length {n}");
        }
    }
    use super::*;

    #[test]
    fn softmax_normalises_each_row() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]).expect("build");
        let s = softmax_last(&t).expect("softmax");
        for r in 0..2 {
            let sum: f32 = s.row(r).iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "row {r} sums to {sum}");
        }
        assert!(s.row(0)[2] > s.row(0)[0], "larger logit wins");
        assert!((s.row(1)[0] - 1.0 / 3.0).abs() < 1e-6, "uniform row");
    }

    #[test]
    fn softmax_is_shift_invariant_and_stable() {
        let a = Tensor::new(vec![1, 3], vec![1000.0, 1001.0, 1002.0]).expect("build");
        let s = softmax_last(&a).expect("softmax");
        assert!(s.data().iter().all(|v| v.is_finite()), "must not overflow");
        let b = Tensor::new(vec![1, 3], vec![0.0, 1.0, 2.0]).expect("build");
        let s2 = softmax_last(&b).expect("softmax");
        for (x, y) in s.data().iter().zip(s2.data()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn layernorm_zero_means_and_unit_variance_before_affine() {
        let t = Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("build");
        let g = vec![1.0; 4];
        let b = vec![0.0; 4];
        let n = layernorm(&t, &g, &b, 1e-6).expect("layernorm");
        let mean: f32 = n.data().iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean {mean}");
        let var: f32 = n.data().iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-3, "var {var}");
    }

    #[test]
    fn layernorm_applies_gamma_and_beta() {
        let t = Tensor::new(vec![1, 2], vec![1.0, -1.0]).expect("build");
        let n = layernorm(&t, &[2.0, 2.0], &[5.0, 5.0], 1e-6).expect("layernorm");
        // Normalised is (+1,-1); scaled by 2 and shifted by 5.
        assert!((n.data()[0] - 7.0).abs() < 1e-3, "{:?}", n.data());
        assert!((n.data()[1] - 3.0).abs() < 1e-3, "{:?}", n.data());
    }

    #[test]
    fn layernorm_rejects_mismatched_parameters() {
        let t = Tensor::new(vec![1, 4], vec![0.0; 4]).expect("build");
        assert!(layernorm(&t, &[1.0; 3], &[0.0; 4], 1e-6).is_err());
    }

    #[test]
    fn batch_matmul_matches_a_hand_computed_product() {
        // [1,2,3] @ [1,3,2] -> [1,2,2]
        let a = Tensor::new(vec![1, 2, 3], vec![1., 2., 3., 4., 5., 6.]).expect("build");
        let b = Tensor::new(vec![1, 3, 2], vec![7., 8., 9., 10., 11., 12.]).expect("build");
        let c = batch_matmul(&a, &b, false).expect("matmul");
        assert_eq!(c.dims(), &[1, 2, 2]);
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn batch_matmul_transpose_y_matches_the_untransposed_form() {
        let a = Tensor::new(vec![1, 2, 3], vec![1., 2., 3., 4., 5., 6.]).expect("build");
        // bT is [1,2,3]; b would be [1,3,2].
        let bt = Tensor::new(vec![1, 2, 3], vec![7., 9., 11., 8., 10., 12.]).expect("build");
        let c = batch_matmul(&a, &bt, true).expect("matmul");
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn batch_matmul_rejects_a_dimension_mismatch() {
        let a = Tensor::new(vec![1, 2, 3], vec![0.0; 6]).expect("build");
        let b = Tensor::new(vec![1, 4, 2], vec![0.0; 8]).expect("build");
        assert!(batch_matmul(&a, &b, false).is_err());
    }

    #[test]
    fn quantize_then_dequantize_round_trips_within_a_step() {
        let t = Tensor::new(vec![1, 4], vec![0.5, -1.0, 0.25, 0.0]).expect("build");
        let (q, scale) = dynamic_quantize(&t);
        let Value::Q8 { data, .. } = &q else {
            panic!("expected int8")
        };
        // Peak is 1.0 so the scale is 127 and -1.0 saturates the range.
        assert!((scale - 127.0).abs() < 1e-3, "scale {scale}");
        assert_eq!(data[1], -127);
        for (i, orig) in t.data().iter().enumerate() {
            let back = data[i] as f32 / scale;
            assert!((back - orig).abs() < 1.0 / 127.0, "element {i}");
        }
    }

    #[test]
    fn quantize_handles_an_all_zero_tensor() {
        let t = Tensor::zeros(vec![1, 4]);
        let (q, scale) = dynamic_quantize(&t);
        assert_eq!(scale, 1.0, "must not divide by zero");
        let Value::Q8 { data, .. } = &q else {
            panic!("expected int8")
        };
        assert!(data.iter().all(|v| *v == 0));
    }

    #[test]
    fn companding_curve_is_monotonic_and_hits_its_endpoints() {
        let q = [-0.563281f32, -0.0435761, 0.116473, 0.646564];
        assert!((dequant_gather_row(0, &q) - q[0]).abs() < 1e-6);
        assert!((dequant_gather_row(255, &q) - q[3]).abs() < 1e-5);
        let mut prev = f32::NEG_INFINITY;
        for b in 0..=255u8 {
            let v = dequant_gather_row(b, &q);
            assert!(v >= prev - 1e-6, "not monotonic at {b}");
            prev = v;
        }
        // Mid-index should land near zero — that is the point of companding.
        let mid = dequant_gather_row(128, &q);
        assert!(mid.abs() < 0.15, "mid-index dequantised to {mid}");
    }

    #[test]
    fn broadcast_zip_handles_scalar_row_and_per_row_forms() {
        let a = Tensor::new(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).expect("build");
        let scalar = Tensor::new(vec![1], vec![10.0]).expect("build");
        let s = broadcast_zip(&a, &scalar, |x, y| x + y).expect("scalar");
        assert_eq!(s.data(), &[11., 12., 13., 14., 15., 16.]);

        let row = Tensor::new(vec![3], vec![10., 20., 30.]).expect("build");
        let r = broadcast_zip(&a, &row, |x, y| x + y).expect("row");
        assert_eq!(r.data(), &[11., 22., 33., 14., 25., 36.]);

        let per_row = Tensor::new(vec![2], vec![100., 200.]).expect("build");
        let p = broadcast_zip(&a, &per_row, |x, y| x + y).expect("per-row");
        assert_eq!(p.data(), &[101., 102., 103., 204., 205., 206.]);
    }

    #[test]
    fn reciprocal_leaves_zero_alone() {
        let t = Tensor::new(vec![1, 3], vec![2.0, 0.0, 4.0]).expect("build");
        let out: Vec<f32> = t
            .data()
            .iter()
            .map(|v| if *v == 0.0 { 0.0 } else { 1.0 / v })
            .collect();
        assert_eq!(out, vec![0.5, 0.0, 0.25]);
    }
}
