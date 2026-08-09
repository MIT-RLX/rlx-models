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

//! Shared helpers for autoregressive decode loops (KV cache + bucketed compile cache).

use anyhow::{Context, Result};
use rlx_ir::{Graph, hir::HirModule};
use rlx_runtime::compile_cache::{
    BucketedCompileCache, CacheRunInput, CompileCache, pad_rows, pad_rows_into, slice_rows,
};
use rlx_runtime::kv_cache::LayerKvCache;
use rlx_runtime::{CompileOptions, CompiledGraph, Device};
use std::collections::HashMap;

/// Decode step outputs: logits plus per-layer K/V (row-major `[seq * kv_dim]`).
pub type DecodeLogitsKv = (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>);

pub use rlx_runtime::kv_cache::LayerKvCache as KvCacheState;

/// LRU prefill cache key: `(batch << 32) | seq`.
pub fn prefill_cache_key(batch: usize, seq: usize) -> u64 {
    ((batch as u64) << 32) | (seq as u64)
}

/// Whether packed prefill should hint [`CompiledGraph::set_active_extent`].
///
/// Skips padded rows when `actual_seq < upper_seq` on CPU, MLX, and wgpu/Vulkan.
///
/// **Metal is intentionally excluded.** Setting active-extent flips Metal prefill
/// off the validated MPSGraph-hybrid path onto the per-op MSL thunk path, which
/// trips a Gemma 3 Q4 `Op::Attention` → `o_proj` dequant arena-aliasing defect
/// (task #50) → all-NaN logits on any padded bucket (prompts past the first
/// bucket, e.g. multi-turn history). Leaving Metal on the full-bucket MPSGraph
/// path both fixes that and lets pow2 bucketing reuse compiled prefill graphs
/// across prompt lengths.
///
/// **CUDA is excluded for Qwen3.5/Bonsai** — enabling it scaled elementwise /
/// GDN correctly but Q1 packed prefill still diverged (zero tokens). Prefer a
/// tight `prefill_seq` compile for short CUDA prompts instead.
///
/// wgpu (Gpu) / Vulkan are back on: the active-extent divergence on MLX-packed
/// Qwen3 prefill was a `Step::GatherAxis` bug (it scaled `total` by
/// `actual/upper`, dropping trailing lanes for the seq-collapsing last-token
/// gather where `outer=1`). Fixed in rlx-wgpu to scale only the leading `outer`
/// dim; wgpu now matches CPU with active-extent on.
/// Set `RLX_DISABLE_ACTIVE_EXTENT=1` to force full-bucket compute everywhere.
pub fn packed_prefill_active_extent_enabled(device: Device) -> bool {
    if rlx_ir::env::var("RLX_DISABLE_ACTIVE_EXTENT").as_deref() == Some("1") {
        return false;
    }
    matches!(
        device,
        Device::Cpu | Device::Mlx | Device::Gpu | Device::Vulkan
    )
}

/// Run a packed prefill graph, trimming compute to `actual_seq` rows inside bucket `upper_seq`.
pub fn run_packed_prefill(
    compiled: &mut CompiledGraph,
    device: Device,
    actual_seq: usize,
    upper_seq: usize,
    inputs: &[(&str, &[f32])],
) -> Vec<Vec<f32>> {
    let use_active =
        packed_prefill_active_extent_enabled(device) && actual_seq > 0 && actual_seq < upper_seq;
    if use_active {
        compiled.set_active_extent(Some((actual_seq, upper_seq)));
    }
    let out = compiled.run(inputs);
    if use_active {
        compiled.set_active_extent(None);
    }
    out
}

/// Infer how many KV rows a prefill returned (active-extent vs full bucket).
pub fn infer_prefill_kv_seq(
    outputs: &[Vec<f32>],
    batch: usize,
    kv_dims: &[usize],
    actual: usize,
    upper: usize,
) -> usize {
    let Some(k) = outputs.get(1) else {
        return actual;
    };
    if kv_dims.is_empty() {
        return actual;
    }
    let kd = kv_dims[0];
    let actual_len = batch * actual * kd;
    let upper_len = batch * upper * kd;
    if k.len() == actual_len {
        actual
    } else if k.len() == upper_len {
        upper
    } else {
        actual
    }
}

fn output_inners_for_kv(kv_dims: &[usize]) -> Vec<usize> {
    let mut output_inners = vec![0usize];
    for &kd in kv_dims {
        output_inners.push(kd);
        output_inners.push(kd);
    }
    output_inners
}

/// Metal fast path: run bucketed decode, read logits only, fetch new K/V rows at `upper`.
fn run_bucketed_decode_on_compiled_metal_rows(
    compiled: &mut CompiledGraph,
    upper: usize,
    past_seq: usize,
    specs: &[CacheRunInput<'_>],
    kv: &LayerKvCache,
    kv_dims: &[usize],
    batch: usize,
) -> Result<DecodeLogitsKv> {
    let batch = batch.max(1);
    let num_layers = kv_dims.len();
    let pairs: Vec<(&str, &[f32])> = specs.iter().map(|inp| (inp.name, inp.data)).collect();

    let outs = compiled.run_read_outputs(&pairs, Some(&[0]));
    let logits = outs
        .into_iter()
        .next()
        .context("bucketed decode logits missing")?;

    let mut new_k = Vec::with_capacity(num_layers);
    let mut new_v = Vec::with_capacity(num_layers);
    let slots = upper + 1;
    for layer in 0..num_layers {
        let kd = kv_dims[layer];
        let row_size = batch * kd;
        let need = (past_seq + 1) * row_size;
        let mut k_out = Vec::with_capacity(need);
        let mut v_out = Vec::with_capacity(need);
        k_out.extend_from_slice(&kv.layers_k[layer]);
        v_out.extend_from_slice(&kv.layers_v[layer]);
        for b in 0..batch {
            let row_ix = b * slots + upper;
            let row_k = compiled
                .read_output_row(1 + 2 * layer, row_ix, kd)
                .with_context(|| format!("Metal K row read layer {layer} batch {b}"))?;
            let row_v = compiled
                .read_output_row(2 + 2 * layer, row_ix, kd)
                .with_context(|| format!("Metal V row read layer {layer} batch {b}"))?;
            k_out.extend_from_slice(&row_k);
            v_out.extend_from_slice(&row_v);
        }
        new_k.push(k_out);
        new_v.push(v_out);
    }
    Ok((logits, new_k, new_v))
}

fn metal_full_kv_readback() -> bool {
    matches!(
        rlx_ir::env::var("RLX_GEMMA_METAL_FULL_KV_READBACK").as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

fn finish_bucketed_decode(
    compiled: &mut CompiledGraph,
    upper: usize,
    past_seq: usize,
    specs: &[CacheRunInput<'_>],
    output_inners: &[usize],
    kv: &LayerKvCache,
    kv_dims: &[usize],
    batch: usize,
) -> Result<DecodeLogitsKv> {
    if compiled.device() == Device::Metal && !metal_full_kv_readback() {
        return run_bucketed_decode_on_compiled_metal_rows(
            compiled, upper, past_seq, specs, kv, kv_dims, batch,
        );
    }
    let raw =
        run_bucketed_decode_on_compiled(compiled, upper, past_seq, batch, specs, output_inners);
    split_bucketed_decode_kv_per_layer(raw, past_seq, kv_dims, kv_dims.len(), batch)
}

/// Run a bucketed decode graph with correct active extent for pad-to-upper KV.
///
/// Decode graphs concat `past_k` padded to bucket `upper` with `k_new` at row `upper`,
/// so attention must run at full length `upper + 1` (see `rlx_runtime::bucket_decode_mask`).
fn run_bucketed_decode_on_compiled(
    compiled: &mut CompiledGraph,
    upper: usize,
    _past_seq: usize,
    batch: usize,
    specs: &[CacheRunInput<'_>],
    output_inners: &[usize],
) -> Vec<Vec<f32>> {
    let batch = batch.max(1);
    let kv_cap = upper + 1;
    let kv_rows = kv_cap * batch;
    let mut owned: Vec<(String, Vec<f32>)> = Vec::new();
    let mut use_owned = vec![false; specs.len()];
    for (i, inp) in specs.iter().enumerate() {
        if let Some(inner) = inp.row_inner {
            let min_len = upper * inner;
            if inp.data.len() >= min_len {
                // Already padded (e.g. batched KV `[batch, upper, dim]` → `batch * upper` rows).
            } else {
                owned.push((
                    inp.name.to_string(),
                    pad_rows(inp.data, inner, upper as u64),
                ));
                use_owned[i] = true;
            }
        }
    }
    let mut owned_idx = 0usize;
    let mut pairs: Vec<(&str, &[f32])> = Vec::with_capacity(specs.len());
    for (i, inp) in specs.iter().enumerate() {
        if use_owned[i] {
            pairs.push((owned[owned_idx].0.as_str(), owned[owned_idx].1.as_slice()));
            owned_idx += 1;
        } else {
            pairs.push((inp.name, inp.data));
        }
    }
    // MPSGraph decode on Metal ignores active-extent; thunks-only path is slower.
    // CPU/MLX use active-extent to skip padded compute where supported.
    let use_active = compiled.device() != Device::Metal
        && rlx_ir::env::var("RLX_DISABLE_ACTIVE_EXTENT").as_deref() != Some("1");
    if use_active {
        compiled.set_active_extent(Some((kv_cap, kv_cap)));
    }
    let raw_outputs = compiled.run(&pairs);
    if use_active {
        compiled.set_active_extent(None);
    }
    raw_outputs
        .into_iter()
        .enumerate()
        .map(|(i, out)| match output_inners.get(i).copied() {
            Some(0) | None => out,
            Some(inner) => slice_rows(&out, inner, kv_rows),
        })
        .collect()
}

/// `past_k_{i}` / `past_v_{i}` input names for `num_layers` decoder layers.
pub fn past_kv_input_names(num_layers: usize) -> Vec<String> {
    (0..num_layers)
        .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
        .collect()
}

/// Split decode graph outputs into logits + per-layer K/V (one-shot decode, no slicing).
pub fn split_decode_logits_kv(outputs: Vec<Vec<f32>>, num_layers: usize) -> Result<DecodeLogitsKv> {
    if outputs.len() != 1 + 2 * num_layers {
        anyhow::bail!(
            "decode graph produced {} outputs, expected {}",
            outputs.len(),
            1 + 2 * num_layers
        );
    }
    let mut iter = outputs.into_iter();
    let logits = iter.next().context("decode logits missing")?;
    let mut layers_k = Vec::with_capacity(num_layers);
    let mut layers_v = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        layers_k.push(iter.next().context("decode k missing")?);
        layers_v.push(iter.next().context("decode v missing")?);
    }
    Ok((logits, layers_k, layers_v))
}

/// Decoded outputs: `(logits, per-layer K, per-layer V, auxiliary hidden
/// states)` — the return shape of [`split_decode_logits_kv_aux`].
pub type DecodeLogitsKvAux = (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>);

/// Split decode graph outputs into logits + per-layer K/V + EAGLE3
/// auxiliary hidden states. Output order is fixed by the build code:
/// `logits, K_0, V_0, K_1, V_1, …, aux_0, aux_1, … aux_{N-1}` where
/// `N == num_aux`.
///
/// Returns `(logits, [K_l], [V_l], [aux_l])`.
pub fn split_decode_logits_kv_aux(
    outputs: Vec<Vec<f32>>,
    num_layers: usize,
    num_aux: usize,
) -> Result<DecodeLogitsKvAux> {
    let expected = 1 + 2 * num_layers + num_aux;
    if outputs.len() != expected {
        anyhow::bail!(
            "decode (+aux) graph produced {} outputs, expected {} \
             ({} logits + {} KV + {} aux)",
            outputs.len(),
            expected,
            1,
            2 * num_layers,
            num_aux,
        );
    }
    let mut iter = outputs.into_iter();
    let logits = iter.next().context("decode logits missing")?;
    let mut layers_k = Vec::with_capacity(num_layers);
    let mut layers_v = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        layers_k.push(iter.next().context("decode k missing")?);
        layers_v.push(iter.next().context("decode v missing")?);
    }
    let mut aux = Vec::with_capacity(num_aux);
    for _ in 0..num_aux {
        aux.push(iter.next().context("aux hidden state missing")?);
    }
    Ok((logits, layers_k, layers_v, aux))
}

/// Build KV state from prefill-with-cache outputs (`logits` + `2 * num_layers` tensors).
pub fn kv_from_prefill_outputs(
    outputs: Vec<Vec<f32>>,
    batch: usize,
    seq: usize,
    kv_dim: usize,
    num_layers: usize,
) -> Result<(Vec<f32>, LayerKvCache)> {
    let dims: Vec<usize> = vec![kv_dim; num_layers];
    kv_from_prefill_outputs_per_layer(outputs, batch, seq, &dims, num_layers)
}

/// Like [`kv_from_prefill_outputs`] but takes a per-layer `kv_dims`
/// slice (one entry per layer). Required for Gemma 4 12B and any
/// other architecture whose full-attention vs sliding-attention
/// layers carry different `num_kv_heads * head_dim`.
pub fn kv_from_prefill_outputs_per_layer(
    outputs: Vec<Vec<f32>>,
    batch: usize,
    seq: usize,
    kv_dims: &[usize],
    num_layers: usize,
) -> Result<(Vec<f32>, LayerKvCache)> {
    if outputs.len() != 1 + 2 * num_layers {
        anyhow::bail!(
            "prefill produced {} outputs, expected {}",
            outputs.len(),
            1 + 2 * num_layers
        );
    }
    if kv_dims.len() != num_layers {
        anyhow::bail!(
            "kv_dims has {} entries, expected {num_layers}",
            kv_dims.len()
        );
    }
    let mut iter = outputs.into_iter();
    let logits = iter.next().context("prefill logits missing")?;
    let mut layers_k = Vec::with_capacity(num_layers);
    let mut layers_v = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let k = iter.next().context("prefill k missing")?;
        let v = iter.next().context("prefill v missing")?;
        let expected_kv_len = batch * seq * kv_dims[layer];
        if k.len() != expected_kv_len || v.len() != expected_kv_len {
            anyhow::bail!(
                "layer {layer}: k.len={} v.len={} expected {expected_kv_len} (kv_dim={})",
                k.len(),
                v.len(),
                kv_dims[layer],
            );
        }
        layers_k.push(k);
        layers_v.push(v);
    }
    Ok((
        logits,
        LayerKvCache {
            past_len: seq,
            layers_kv_base: vec![0; layers_k.len()],
            layers_k,
            layers_v,
        },
    ))
}

/// One bucketed decode step: compile bucket if needed, pad K/V, run, slice updated K/V.
///
/// `fixed_inputs` must include `mask` (use `bucket_decode_mask`) when the graph uses
/// `MaskKind::Custom`. Rope / token inputs should use `row_inner: None`.
pub fn run_bucketed_kv_decode<F>(
    cache: &mut BucketedCompileCache,
    past_seq: usize,
    kv: &LayerKvCache,
    kv_dim: usize,
    num_layers: usize,
    fixed_inputs: &[CacheRunInput<'_>],
    build: F,
    options: &CompileOptions,
) -> Result<DecodeLogitsKv>
where
    F: FnOnce(u64) -> (Graph, HashMap<String, Vec<f32>>),
{
    run_bucketed_kv_decode_keyed_batched(
        cache,
        past_seq as u64,
        past_seq,
        1,
        kv,
        kv_dim,
        num_layers,
        fixed_inputs,
        build,
        options,
    )
}

/// Like [`run_bucketed_kv_decode_keyed`] but indexes the compile bucket with `cache_key`
/// (e.g. Whisper `(batch << 32) | past_seq`).
pub fn run_bucketed_kv_decode_keyed<F>(
    cache: &mut BucketedCompileCache,
    cache_key: u64,
    past_seq: usize,
    kv: &LayerKvCache,
    kv_dim: usize,
    num_layers: usize,
    fixed_inputs: &[CacheRunInput<'_>],
    build: F,
    options: &CompileOptions,
) -> Result<DecodeLogitsKv>
where
    F: FnOnce(u64) -> (Graph, HashMap<String, Vec<f32>>),
{
    run_bucketed_kv_decode_keyed_batched(
        cache,
        cache_key,
        past_seq,
        1,
        kv,
        kv_dim,
        num_layers,
        fixed_inputs,
        build,
        options,
    )
}

/// Batched decode: K/V tensors are `[batch, past_seq, kv_dim]` row-major (`batch * past_seq` rows).
pub fn run_bucketed_kv_decode_keyed_batched<F>(
    cache: &mut BucketedCompileCache,
    cache_key: u64,
    past_seq: usize,
    batch: usize,
    kv: &LayerKvCache,
    kv_dim: usize,
    num_layers: usize,
    fixed_inputs: &[CacheRunInput<'_>],
    build: F,
    options: &CompileOptions,
) -> Result<DecodeLogitsKv>
where
    F: FnOnce(u64) -> (Graph, HashMap<String, Vec<f32>>),
{
    let batch = batch.max(1);
    let (upper_u64, compiled) = cache
        .ensure_graph_with_params(cache_key, build, options)
        .ok_or_else(|| anyhow::anyhow!("cache_key {cache_key} outside decode buckets"))?;
    let upper = upper_u64 as usize;

    let row_upper = upper_u64.saturating_mul(batch as u64);
    let padded_k: Vec<Vec<f32>> = kv
        .layers_k
        .iter()
        .map(|k| pad_rows(k, kv_dim, row_upper))
        .collect();
    let padded_v: Vec<Vec<f32>> = kv
        .layers_v
        .iter()
        .map(|v| pad_rows(v, kv_dim, row_upper))
        .collect();
    let key_names = past_kv_input_names(num_layers);

    let mut specs: Vec<CacheRunInput<'_>> = Vec::with_capacity(fixed_inputs.len() + 2 * num_layers);
    for inp in fixed_inputs {
        specs.push(CacheRunInput {
            name: inp.name,
            data: inp.data,
            row_inner: inp.row_inner,
        });
    }
    for i in 0..num_layers {
        specs.push(CacheRunInput {
            name: key_names[2 * i].as_str(),
            data: padded_k[i].as_slice(),
            row_inner: Some(kv_dim),
        });
        specs.push(CacheRunInput {
            name: key_names[2 * i + 1].as_str(),
            data: padded_v[i].as_slice(),
            row_inner: Some(kv_dim),
        });
    }

    let kv_dims = vec![kv_dim; num_layers];
    let output_inners = output_inners_for_kv(&kv_dims);
    finish_bucketed_decode(
        compiled,
        upper,
        past_seq,
        &specs,
        &output_inners,
        kv,
        &kv_dims,
        batch,
    )
}

/// Packed-weights variant of [`run_bucketed_kv_decode`]: `build` returns
/// `(graph, f32_params, packed_params)` and the U8 packed params are bound via
/// `set_param_typed` on first compile — K-quant linears stay packed in the arena
/// (`Op::DequantMatMul`) instead of dequant-to-f32. Batch = 1. Everything after
/// the compile+bind (K/V padding, run) is identical to the f32 path.
pub fn run_bucketed_kv_decode_packed<F>(
    cache: &mut BucketedCompileCache,
    past_seq: usize,
    kv: &LayerKvCache,
    kv_dim: usize,
    num_layers: usize,
    fixed_inputs: &[CacheRunInput<'_>],
    build: F,
    options: &CompileOptions,
) -> Result<DecodeLogitsKv>
where
    F: FnOnce(u64) -> (Graph, HashMap<String, Vec<f32>>, Vec<(String, Vec<u8>)>),
{
    let batch = 1usize;
    let (upper_u64, compiled) = cache
        .ensure_graph_with_packed(past_seq as u64, build, options)
        .ok_or_else(|| anyhow::anyhow!("past_seq {past_seq} outside decode buckets"))?;
    let upper = upper_u64 as usize;

    let row_upper = upper_u64.saturating_mul(batch as u64);
    let padded_k: Vec<Vec<f32>> = kv
        .layers_k
        .iter()
        .map(|k| pad_rows(k, kv_dim, row_upper))
        .collect();
    let padded_v: Vec<Vec<f32>> = kv
        .layers_v
        .iter()
        .map(|v| pad_rows(v, kv_dim, row_upper))
        .collect();
    let key_names = past_kv_input_names(num_layers);

    let mut specs: Vec<CacheRunInput<'_>> = Vec::with_capacity(fixed_inputs.len() + 2 * num_layers);
    for inp in fixed_inputs {
        specs.push(CacheRunInput {
            name: inp.name,
            data: inp.data,
            row_inner: inp.row_inner,
        });
    }
    for i in 0..num_layers {
        specs.push(CacheRunInput {
            name: key_names[2 * i].as_str(),
            data: padded_k[i].as_slice(),
            row_inner: Some(kv_dim),
        });
        specs.push(CacheRunInput {
            name: key_names[2 * i + 1].as_str(),
            data: padded_v[i].as_slice(),
            row_inner: Some(kv_dim),
        });
    }

    let kv_dims = vec![kv_dim; num_layers];
    let output_inners = output_inners_for_kv(&kv_dims);
    finish_bucketed_decode(
        compiled,
        upper,
        past_seq,
        &specs,
        &output_inners,
        kv,
        &kv_dims,
        batch,
    )
}

/// Like [`run_bucketed_kv_decode`] but compiles decode graphs from HIR (Gemma / Llama / Qwen3.5).
pub fn run_bucketed_kv_decode_hir<F>(
    cache: &mut BucketedCompileCache,
    past_seq: usize,
    kv: &LayerKvCache,
    kv_dim: usize,
    num_layers: usize,
    fixed_inputs: &[CacheRunInput<'_>],
    build: F,
    options: &CompileOptions,
) -> Result<DecodeLogitsKv>
where
    F: FnOnce(u64) -> (HirModule, HashMap<String, Vec<f32>>),
{
    let key = past_seq as u64;
    let (upper_u64, compiled) = cache
        .ensure_hir_with_params(key, build, options)
        .ok_or_else(|| anyhow::anyhow!("past_seq {past_seq} outside decode buckets"))?;
    let upper = upper_u64 as usize;

    let (padded_k, padded_v) = kv.pad_layers_to_upper(upper_u64, kv_dim);
    let key_names = past_kv_input_names(num_layers);

    let mut specs: Vec<CacheRunInput<'_>> = Vec::with_capacity(fixed_inputs.len() + 2 * num_layers);
    for inp in fixed_inputs {
        specs.push(CacheRunInput {
            name: inp.name,
            data: inp.data,
            row_inner: inp.row_inner,
        });
    }
    for i in 0..num_layers {
        specs.push(CacheRunInput {
            name: key_names[2 * i].as_str(),
            data: padded_k[i].as_slice(),
            row_inner: Some(kv_dim),
        });
        specs.push(CacheRunInput {
            name: key_names[2 * i + 1].as_str(),
            data: padded_v[i].as_slice(),
            row_inner: Some(kv_dim),
        });
    }

    let kv_dims = vec![kv_dim; num_layers];
    let output_inners = output_inners_for_kv(&kv_dims);
    finish_bucketed_decode(
        compiled,
        upper,
        past_seq,
        &specs,
        &output_inners,
        kv,
        &kv_dims,
        1,
    )
}
pub fn run_bucketed_kv_decode_hir_layers<F>(
    cache: &mut BucketedCompileCache,
    past_seq: usize,
    kv: &LayerKvCache,
    kv_dims: &[usize],
    num_layers: usize,
    fixed_inputs: &[CacheRunInput<'_>],
    build: F,
    options: &CompileOptions,
) -> Result<DecodeLogitsKv>
where
    F: FnOnce(u64) -> (HirModule, HashMap<String, Vec<f32>>),
{
    let key = past_seq as u64;
    let (upper_u64, compiled) = cache
        .ensure_hir_with_params(key, build, options)
        .ok_or_else(|| anyhow::anyhow!("past_seq {past_seq} outside decode buckets"))?;
    let upper = upper_u64 as usize;

    if kv_dims.len() != num_layers {
        anyhow::bail!(
            "run_bucketed_kv_decode_hir: kv_dims len {} != num_layers {num_layers}",
            kv_dims.len()
        );
    }

    let (padded_k, padded_v) = kv.pad_layers_to_upper_per_layer(upper_u64, kv_dims);
    let key_names = past_kv_input_names(num_layers);

    let mut specs: Vec<CacheRunInput<'_>> = Vec::with_capacity(fixed_inputs.len() + 2 * num_layers);
    for inp in fixed_inputs {
        specs.push(CacheRunInput {
            name: inp.name,
            data: inp.data,
            row_inner: inp.row_inner,
        });
    }
    for i in 0..num_layers {
        specs.push(CacheRunInput {
            name: key_names[2 * i].as_str(),
            data: padded_k[i].as_slice(),
            row_inner: Some(kv_dims[i]),
        });
        specs.push(CacheRunInput {
            name: key_names[2 * i + 1].as_str(),
            data: padded_v[i].as_slice(),
            row_inner: Some(kv_dims[i]),
        });
    }

    let output_inners = output_inners_for_kv(kv_dims);
    finish_bucketed_decode(
        compiled,
        upper,
        past_seq,
        &specs,
        &output_inners,
        kv,
        kv_dims,
        1,
    )
}

/// Like [`run_bucketed_kv_decode_hir`] but reuses caller-provided padded K/V scratch (no alloc).
pub fn run_bucketed_kv_decode_hir_scratch<F>(
    cache: &mut BucketedCompileCache,
    past_seq: usize,
    kv: &LayerKvCache,
    kv_dims: &[usize],
    num_layers: usize,
    padded_k: &mut [Vec<f32>],
    padded_v: &mut [Vec<f32>],
    fixed_inputs: &[CacheRunInput<'_>],
    build: F,
    options: &CompileOptions,
) -> Result<DecodeLogitsKv>
where
    F: FnOnce(u64) -> (HirModule, HashMap<String, Vec<f32>>),
{
    let key = past_seq as u64;
    let (upper_u64, compiled) = cache
        .ensure_hir_with_params(key, build, options)
        .ok_or_else(|| anyhow::anyhow!("past_seq {past_seq} outside decode buckets"))?;
    let upper = upper_u64 as usize;

    if kv_dims.len() != num_layers || padded_k.len() != num_layers || padded_v.len() != num_layers {
        anyhow::bail!("run_bucketed_kv_decode_hir_scratch: layer count mismatch");
    }
    for i in 0..num_layers {
        let kd = kv_dims[i];
        let need = upper * kd;
        if padded_k[i].len() != need {
            padded_k[i].resize(need, 0.0);
            padded_v[i].resize(need, 0.0);
        }
        pad_rows_into(&mut padded_k[i], kv.layers_k[i].as_slice(), kd);
        pad_rows_into(&mut padded_v[i], kv.layers_v[i].as_slice(), kd);
    }

    let key_names = past_kv_input_names(num_layers);
    let mut specs: Vec<CacheRunInput<'_>> = Vec::with_capacity(fixed_inputs.len() + 2 * num_layers);
    for inp in fixed_inputs {
        specs.push(CacheRunInput {
            name: inp.name,
            data: inp.data,
            row_inner: inp.row_inner,
        });
    }
    for i in 0..num_layers {
        specs.push(CacheRunInput {
            name: key_names[2 * i].as_str(),
            data: padded_k[i].as_slice(),
            row_inner: Some(kv_dims[i]),
        });
        specs.push(CacheRunInput {
            name: key_names[2 * i + 1].as_str(),
            data: padded_v[i].as_slice(),
            row_inner: Some(kv_dims[i]),
        });
    }

    let output_inners = output_inners_for_kv(kv_dims);
    finish_bucketed_decode(
        compiled,
        upper,
        past_seq,
        &specs,
        &output_inners,
        kv,
        kv_dims,
        1,
    )
}

/// Graph variant of [`run_bucketed_kv_decode_hir_scratch`] for packed tier-0 builders.
pub fn run_bucketed_kv_decode_graph_layers_scratch<F, U>(
    cache: &mut BucketedCompileCache,
    past_seq: usize,
    kv: &LayerKvCache,
    kv_dims: &[usize],
    num_layers: usize,
    padded_k: &mut [Vec<f32>],
    padded_v: &mut [Vec<f32>],
    fixed_inputs: &[CacheRunInput<'_>],
    build: F,
    // Uploads this bucket's packed weights into the freshly-built graph — called
    // once per bucket on first build. Decouples the helper from how weights are
    // stored (owned bytes vs. borrowed-from-mmap recipes).
    upload_packed: Option<U>,
    packed_loaded: &mut std::collections::HashSet<u64>,
    options: &CompileOptions,
) -> Result<DecodeLogitsKv>
where
    F: FnOnce(u64) -> (Graph, HashMap<String, Vec<f32>>),
    U: FnOnce(&mut CompiledGraph),
{
    let key = past_seq as u64;
    let needs_build = cache.compiled_for_key_mut(key).is_none();
    let (upper_u64, compiled) = cache
        .ensure_graph_with_params(key, build, options)
        .ok_or_else(|| anyhow::anyhow!("past_seq {past_seq} outside decode buckets"))?;
    if needs_build {
        if let Some(upload) = upload_packed {
            if packed_loaded.insert(upper_u64) {
                upload(compiled);
            }
        }
    }
    let upper = upper_u64 as usize;

    if kv_dims.len() != num_layers || padded_k.len() != num_layers || padded_v.len() != num_layers {
        anyhow::bail!("run_bucketed_kv_decode_graph_layers_scratch: layer count mismatch");
    }
    for i in 0..num_layers {
        let kd = kv_dims[i];
        let need = upper * kd;
        if padded_k[i].len() != need {
            padded_k[i].resize(need, 0.0);
            padded_v[i].resize(need, 0.0);
        }
        pad_rows_into(&mut padded_k[i], kv.layers_k[i].as_slice(), kd);
        pad_rows_into(&mut padded_v[i], kv.layers_v[i].as_slice(), kd);
    }

    let key_names = past_kv_input_names(num_layers);
    let mut specs: Vec<CacheRunInput<'_>> = Vec::with_capacity(fixed_inputs.len() + 2 * num_layers);
    for inp in fixed_inputs {
        specs.push(CacheRunInput {
            name: inp.name,
            data: inp.data,
            row_inner: inp.row_inner,
        });
    }
    for i in 0..num_layers {
        specs.push(CacheRunInput {
            name: key_names[2 * i].as_str(),
            data: padded_k[i].as_slice(),
            row_inner: Some(kv_dims[i]),
        });
        specs.push(CacheRunInput {
            name: key_names[2 * i + 1].as_str(),
            data: padded_v[i].as_slice(),
            row_inner: Some(kv_dims[i]),
        });
    }

    let output_inners = output_inners_for_kv(kv_dims);
    finish_bucketed_decode(
        compiled,
        upper,
        past_seq,
        &specs,
        &output_inners,
        kv,
        kv_dims,
        1,
    )
}

/// Per-layer variant of [`split_bucketed_decode_kv`].
pub fn split_bucketed_decode_kv_per_layer(
    outputs: Vec<Vec<f32>>,
    past_seq: usize,
    kv_dims: &[usize],
    num_layers: usize,
    batch: usize,
) -> Result<DecodeLogitsKv> {
    if outputs.len() != 1 + 2 * num_layers {
        anyhow::bail!(
            "bucketed decode produced {} outputs, expected {}",
            outputs.len(),
            1 + 2 * num_layers
        );
    }
    let mut iter = outputs.into_iter();
    let logits = iter.next().context("bucketed decode logits missing")?;
    let mut new_k = Vec::with_capacity(num_layers);
    let mut new_v = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let kv_dim = kv_dims[layer];
        let row_size = batch * kv_dim;
        let k = iter.next().context("bucketed k missing")?;
        let v = iter.next().context("bucketed v missing")?;
        if k.len() < (past_seq + 1) * row_size {
            anyhow::bail!(
                "bucketed K layer {layer}: got {} f32, need at least {} (past_seq={past_seq}, kv_dim={kv_dim})",
                k.len(),
                (past_seq + 1) * row_size
            );
        }
        new_k.push(compact_bucketed_kv_buffer(&k, past_seq + 1, kv_dim, batch));
        new_v.push(compact_bucketed_kv_buffer(&v, past_seq + 1, kv_dim, batch));
    }
    Ok((logits, new_k, new_v))
}

/// Legacy uniform-`kv_dim` HIR bucketed decode (Whisper-shaped graphs).
pub fn run_bucketed_kv_decode_hir_uniform<F>(
    cache: &mut BucketedCompileCache,
    past_seq: usize,
    kv: &LayerKvCache,
    kv_dim: usize,
    num_layers: usize,
    fixed_inputs: &[CacheRunInput<'_>],
    build: F,
    options: &CompileOptions,
) -> Result<DecodeLogitsKv>
where
    F: FnOnce(u64) -> (HirModule, HashMap<String, Vec<f32>>),
{
    run_bucketed_kv_decode_hir(
        cache,
        past_seq,
        kv,
        kv_dim,
        num_layers,
        fixed_inputs,
        build,
        options,
    )
}

/// Slice bucketed decode K/V outputs back to `past_seq + 1` rows.
pub fn split_bucketed_decode_kv(
    outputs: Vec<Vec<f32>>,
    past_seq: usize,
    kv_dim: usize,
    num_layers: usize,
    batch: usize,
) -> Result<DecodeLogitsKv> {
    if outputs.len() != 1 + 2 * num_layers {
        anyhow::bail!(
            "bucketed decode produced {} outputs, expected {}",
            outputs.len(),
            1 + 2 * num_layers
        );
    }
    let mut iter = outputs.into_iter();
    let logits = iter.next().context("bucketed decode logits missing")?;
    let row_size = batch * kv_dim;
    let mut new_k = Vec::with_capacity(num_layers);
    let mut new_v = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let k = iter.next().context("bucketed k missing")?;
        let v = iter.next().context("bucketed v missing")?;
        // Bucketed decode K/V layout: `concat(past_k_padded[upper], new_k[1], dim=1)`,
        // so the buffer is `(upper + 1) * row_size` long with the real new token at
        // the LAST row, not at row `past_seq`. Earlier versions returned
        // `k[..(past_seq+1)*row_size]` which captured the zero padding at row
        // `past_seq` and silently dropped the new K — every cached step absorbed
        // padding as the "current" K, then attention couldn't see the model's own
        // prior tokens and decode degenerated into repetitive fragments.
        if k.len() < (past_seq + 1) * row_size {
            anyhow::bail!(
                "bucketed K layer {layer}: got {} f32, need at least {} (past_seq={past_seq}, kv_dim={kv_dim})",
                k.len(),
                (past_seq + 1) * row_size
            );
        }
        new_k.push(compact_bucketed_kv_buffer(&k, past_seq + 1, kv_dim, batch));
        new_v.push(compact_bucketed_kv_buffer(&v, past_seq + 1, kv_dim, batch));
    }
    Ok((logits, new_k, new_v))
}

/// Compact pad-to-upper K/V (new token at row `upper`) to contiguous `past_len` rows.
///
/// GPU-resident decode feeds the full `(upper + 1) * row_size` buffer back into handles;
/// host cache expects dense `[0..past_len)` rows without zero padding in the middle.
pub fn compact_bucketed_kv_buffer(
    buf: &[f32],
    past_len: usize,
    kv_dim: usize,
    batch: usize,
) -> Vec<f32> {
    if past_len == 0 {
        return Vec::new();
    }
    let row_size = batch * kv_dim;
    if buf.len() < row_size {
        return buf.to_vec();
    }
    let total_rows = buf.len() / row_size;
    if total_rows <= past_len {
        let n = past_len * row_size;
        return buf[..n.min(buf.len())].to_vec();
    }
    let past_seq = past_len - 1;
    let past_bytes = past_seq * row_size;
    let new_row = total_rows - 1;
    let mut out = Vec::with_capacity(past_len * row_size);
    out.extend_from_slice(&buf[..past_bytes]);
    out.extend_from_slice(&buf[new_row * row_size..(new_row + 1) * row_size]);
    out
}

/// Insert a pre-sized graph into an LRU [`CompileCache`].
pub fn compile_cache_ensure_graph<'a>(
    cache: &'a mut CompileCache,
    key: u64,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
    options: &CompileOptions,
) -> &'a mut CompiledGraph {
    if !cache.contains(key) {
        let compiled = cache.get_or_compile_with_options(key, || graph, options);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
    }
    cache.get_or_compile_with_options(
        key,
        || panic!("compile_cache_ensure_graph: missing {key}"),
        options,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_aux_returns_logits_kv_and_aux_in_order() {
        // 2 layers, 3 aux → expect 1+4+3 = 8 outputs.
        let outs = vec![
            vec![0.0],  // logits
            vec![1.0],  // K0
            vec![2.0],  // V0
            vec![3.0],  // K1
            vec![4.0],  // V1
            vec![10.0], // aux 0
            vec![20.0], // aux 1
            vec![30.0], // aux 2
        ];
        let (logits, ks, vs, aux) = split_decode_logits_kv_aux(outs, 2, 3).unwrap();
        assert_eq!(logits, vec![0.0]);
        assert_eq!(ks, vec![vec![1.0], vec![3.0]]);
        assert_eq!(vs, vec![vec![2.0], vec![4.0]]);
        assert_eq!(aux, vec![vec![10.0], vec![20.0], vec![30.0]]);
    }

    #[test]
    fn split_aux_rejects_wrong_count() {
        let outs = vec![vec![0.0]; 5];
        let err = split_decode_logits_kv_aux(outs, 2, 3).unwrap_err();
        assert!(
            format!("{err}").contains("expected 8"),
            "expected count error, got: {err}"
        );
    }

    #[test]
    fn split_aux_zero_aux_matches_split_decode_logits_kv() {
        let outs = vec![vec![0.0], vec![1.0], vec![2.0]];
        let outs2 = outs.clone();
        let (l1, k1, v1, a1) = split_decode_logits_kv_aux(outs, 1, 0).unwrap();
        let (l2, k2, v2) = split_decode_logits_kv(outs2, 1).unwrap();
        assert!(a1.is_empty());
        assert_eq!(l1, l2);
        assert_eq!(k1, k2);
        assert_eq!(v1, v2);
    }
}
