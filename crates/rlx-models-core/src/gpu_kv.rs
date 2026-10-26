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

//! GPU-resident KV cache via `bind_gpu_handle` (MLX device arrays; Metal/CUDA/WGPU host mirrors).

use crate::autoregressive::{
    KvCacheState, compact_bucketed_kv_buffer, pad_kv_sequences, past_kv_input_names,
};
use anyhow::{Context, Result, ensure};
use rlx_ir::{Graph, hir::HirModule};
use rlx_runtime::compile_cache::{BucketedCompileCache, CacheRunInput, pad_rows};
use rlx_runtime::kv_cache::LayerKvCache;
use rlx_runtime::{CompileOptions, CompiledGraph, Device};
use std::collections::HashMap;

/// Backends that support persistent K/V handles + selective logits readback.
///
/// `RLX_DISABLE_GPU_KV=1` forces the host K/V path (read decode outputs back to
/// host, re-feed as inputs) — a diagnostic/fallback for backends whose on-GPU KV
/// handle propagation is broken (e.g. wgpu native GGUF decode stuck-token bug).
/// Optionally scope it to one backend: `RLX_DISABLE_GPU_KV=wgpu|cuda|vulkan|...`.
pub fn device_supports_gpu_kv(device: Device) -> bool {
    if let Some(v) = rlx_ir::env::var("RLX_DISABLE_GPU_KV") {
        let v = v.trim().to_ascii_lowercase();
        let name = match device {
            Device::Gpu => "wgpu",
            Device::Cuda => "cuda",
            Device::Vulkan => "vulkan",
            Device::Metal => "metal",
            Device::Mlx => "mlx",
            Device::Rocm => "rocm",
            _ => "",
        };
        if v == "1" || v == "true" || v == name {
            return false;
        }
    }
    matches!(
        device,
        Device::Mlx | Device::Metal | Device::Cuda | Device::Rocm | Device::Gpu | Device::Vulkan
    )
}

/// [`device_supports_gpu_kv`] plus the CPU, for a decode loop that uses only
/// the single-row resident contract: `bind_gpu_handle` +
/// `register_kv_row_feed` + `run_read_outputs` + `feed_kv_row` +
/// `read_output_row`. The CPU backend implements exactly those (its "resident"
/// cache is a persistent handle in the executable) and none of the batch-major
/// or prefix-seed hooks, so a family opts in here only once its loop is
/// checked against them — qwen3 is, token-identical to the host path. It is
/// kept apart from `device_supports_gpu_kv` so families that have not been
/// checked keep the host path on CPU. `RLX_DISABLE_GPU_KV=1|cpu` turns it off.
pub fn device_supports_resident_kv_row_feed(device: Device) -> bool {
    if device != Device::Cpu {
        return device_supports_gpu_kv(device);
    }
    !matches!(
        rlx_ir::env::var("RLX_DISABLE_GPU_KV")
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "cpu")
    )
}

/// How a model's decode step returns each self-attention K / V cache. A
/// model says which its graph does ([`GpuKvBinding::new`]), or leaves it to
/// [`KvLayout::Detect`]; the resident handles are kept by that rule.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KvLayout {
    /// Read off the step when a bucket compiles ([`KvLayout::of_graph`]):
    /// a cache output one row longer than its input is
    /// [`Self::AppendLast`], the same length [`Self::InPlace`]. Until then,
    /// and where the shapes say neither, `InPlace`. A family whose layout
    /// depends on the batch or the backend (`rlx-flow` appends in place only
    /// where the cache can alias) gets the right one either way.
    #[default]
    Detect,
    /// The step writes its new row into the cache it returns, at the
    /// position (`Op::KvAppend`, an output aliasing its input): the output's
    /// leading rows are the next cache, copied into the handle after each
    /// run.
    InPlace,
    /// The step returns each cache as its past rows, padded to the bucket,
    /// then the new row: `concat(past, new)`, `[batch, upper + 1, kv]`.
    /// Only the new row moves, to `past` in each sequence's handle rows
    /// (`register_kv_row_feed` + `feed_kv_row`); a copy of the leading rows
    /// would drop it (and, at batch > 1, shift every sequence after the
    /// first by a row each).
    AppendLast,
}

impl KvLayout {
    /// The layout `past_k_0` and its output (index 1) say, `[…, rows, kv]`.
    fn of_rows(input: &rlx_ir::Shape, output: &rlx_ir::Shape) -> Option<Self> {
        let rows = |s: &rlx_ir::Shape| {
            let d = s.dims();
            d.len().checked_sub(2).and_then(|i| match d[i] {
                rlx_ir::Dim::Static(n) => Some(n),
                rlx_ir::Dim::Dynamic(_) => None,
            })
        };
        match (rows(input)?, rows(output)?) {
            (i, o) if o == i + 1 => Some(Self::AppendLast),
            (i, o) if o == i => Some(Self::InPlace),
            _ => None,
        }
    }

    /// The layout of a decode step graph (`past_k_0` → output 1), if its
    /// shapes say one.
    pub fn of_graph(graph: &Graph) -> Option<Self> {
        let input = graph
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, rlx_ir::Op::Input { name } if name == "past_k_0"))?;
        let output = graph.node(*graph.outputs.get(1)?);
        Self::of_rows(&input.shape, &output.shape)
    }

    /// [`Self::of_graph`] for a step built as HIR.
    pub fn of_hir(module: &HirModule) -> Option<Self> {
        let input = module
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, rlx_ir::hir::HirOp::Input { name } if name == "past_k_0"))?;
        let output = module.node(*module.outputs.get(1)?);
        Self::of_rows(&input.shape, &output.shape)
    }
}

/// Which bucket the GPU handles were allocated for, and how the model's
/// step returns its caches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuKvBinding {
    pub upper: u64,
    pub layout: KvLayout,
    /// Sequences a step decodes (0 counts as 1).
    pub batch: usize,
}

impl GpuKvBinding {
    /// No handles yet, for a step with `layout` decoding `batch` sequences.
    pub fn new(layout: KvLayout, batch: usize) -> Self {
        Self {
            upper: 0,
            layout,
            batch,
        }
    }

    fn sequences(&self) -> usize {
        self.batch.max(1)
    }
}

/// Per compile-cache GPU binding state.
#[derive(Debug, Default)]
pub struct GpuKvCacheSet {
    pub causal: GpuKvBinding,
    pub decode_mtp: GpuKvBinding,
    pub mtp: GpuKvBinding,
}

impl GpuKvCacheSet {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Drop decode bindings after an MTP block advanced `past_len`.
    pub fn reset_decode_after_mtp(&mut self) {
        self.causal = GpuKvBinding::default();
        self.decode_mtp = GpuKvBinding::default();
        self.mtp = GpuKvBinding::default();
    }
}

/// True when `cross_k_0` is already bound on this graph.
pub fn cross_attn_gpu_handles_ready(compiled: &CompiledGraph) -> bool {
    compiled.has_gpu_handle("cross_k_0")
}

/// Upload fixed cross-attention K/V (`cross_k_*` / `cross_v_*`) for Whisper-style decoders.
pub fn install_cross_attn_gpu_handles(
    compiled: &mut CompiledGraph,
    cross: &LayerKvCache,
    enc_seq: usize,
    kv_dim: usize,
    num_layers: usize,
) -> Result<()> {
    // `[batch, enc_seq, kv]`: whole sequences, the last padded.
    let rows = cross
        .layers_k
        .first()
        .map_or(0, |k| k.len() / kv_dim.max(1));
    let enc = enc_seq.max(1);
    let upper = (rows.div_ceil(enc).max(1) * enc) as u64;
    for i in 0..num_layers {
        let k_name = format!("cross_k_{i}");
        let v_name = format!("cross_v_{i}");
        let k_pad = pad_rows(cross.layers_k[i].as_slice(), kv_dim, upper);
        let v_pad = pad_rows(cross.layers_v[i].as_slice(), kv_dim, upper);
        ensure!(
            compiled.bind_gpu_handle(k_name.as_str(), &k_pad),
            "bind_gpu_handle failed for {k_name}"
        );
        ensure!(
            compiled.bind_gpu_handle(v_name.as_str(), &v_pad),
            "bind_gpu_handle failed for {v_name}"
        );
    }
    Ok(())
}

/// Upload `prefix_rows` of `kv` into `past_k_*` / `past_v_*` GPU handles and wire output feeds.
pub fn install_gpu_kv_handles(
    compiled: &mut CompiledGraph,
    kv: &KvCacheState,
    prefix_rows: usize,
    upper: u64,
    kv_dim: usize,
    num_layers: usize,
) -> Result<()> {
    install_gpu_kv_handles_as(
        compiled,
        kv,
        prefix_rows,
        upper,
        kv_dim,
        num_layers,
        GpuKvBinding::default(),
    )
}

/// [`install_gpu_kv_handles`] by `binding`'s layout and batch.
pub fn install_gpu_kv_handles_as(
    compiled: &mut CompiledGraph,
    kv: &KvCacheState,
    prefix_rows: usize,
    upper: u64,
    kv_dim: usize,
    num_layers: usize,
    binding: GpuKvBinding,
) -> Result<()> {
    let names = past_kv_input_names(num_layers);
    let batch = binding.sequences();
    for layer in 0..num_layers {
        for (name, host, out) in [
            (
                names[2 * layer].as_str(),
                &kv.layers_k[layer],
                1 + 2 * layer,
            ),
            (
                names[2 * layer + 1].as_str(),
                &kv.layers_v[layer],
                2 + 2 * layer,
            ),
        ] {
            let padded = pad_kv_sequences(host, batch, kv.past_len, kv_dim, prefix_rows, upper);
            ensure!(
                compiled.bind_gpu_handle(name, &padded),
                "bind_gpu_handle failed for {name}"
            );
            match binding.layout {
                KvLayout::InPlace | KvLayout::Detect => {
                    compiled.set_gpu_handle_feed(name, out);
                }
                KvLayout::AppendLast => {
                    compiled.register_kv_row_feed(name, out);
                }
            }
        }
    }
    Ok(())
}

fn layer_host_rows(
    compiled: &CompiledGraph,
    name: &str,
    host: &[f32],
    past_len: usize,
    kv_dim: usize,
) -> Vec<f32> {
    if compiled.has_gpu_handle(name)
        && let Some(buf) = compiled.read_gpu_handle(name)
    {
        return compact_bucketed_kv_buffer(&buf, past_len, kv_dim, 1);
    }
    let take = (past_len * kv_dim).min(host.len());
    host[..take].to_vec()
}

/// Rebind handles after a bucket change (read back prior GPU K/V, pad to new `upper`).
pub fn reinstall_gpu_kv_handles(
    compiled: &mut CompiledGraph,
    kv: &KvCacheState,
    _old_upper: u64,
    new_upper: u64,
    kv_dim: usize,
    num_layers: usize,
) -> Result<()> {
    let names = past_kv_input_names(num_layers);
    let mut tmp = KvCacheState {
        past_len: kv.past_len,
        layers_k: Vec::with_capacity(num_layers),
        layers_v: Vec::with_capacity(num_layers),
        layers_kv_base: kv.layers_kv_base.clone(),
    };
    for layer in 0..num_layers {
        tmp.layers_k.push(layer_host_rows(
            compiled,
            &names[2 * layer],
            &kv.layers_k[layer],
            kv.past_len,
            kv_dim,
        ));
        tmp.layers_v.push(layer_host_rows(
            compiled,
            &names[2 * layer + 1],
            &kv.layers_v[layer],
            kv.past_len,
            kv_dim,
        ));
    }
    install_gpu_kv_handles(compiled, &tmp, tmp.past_len, new_upper, kv_dim, num_layers)
}

/// Pull GPU K/V back to host `kv` (for MTP truncate / prefill cache).
pub fn sync_gpu_kv_to_host(
    compiled: &CompiledGraph,
    kv: &mut KvCacheState,
    kv_dim: usize,
    num_layers: usize,
) -> Result<()> {
    let names = past_kv_input_names(num_layers);
    let n = kv.past_len * kv_dim;
    for layer in 0..num_layers {
        kv.layers_k[layer] = layer_host_rows(
            compiled,
            &names[2 * layer],
            &kv.layers_k[layer],
            kv.past_len,
            kv_dim,
        );
        kv.layers_v[layer] = layer_host_rows(
            compiled,
            &names[2 * layer + 1],
            &kv.layers_v[layer],
            kv.past_len,
            kv_dim,
        );
        if kv.layers_k[layer].len() > n {
            kv.layers_k[layer].truncate(n);
        }
        if kv.layers_v[layer].len() > n {
            kv.layers_v[layer].truncate(n);
        }
    }
    Ok(())
}

/// [`sync_gpu_kv_to_host`] by `binding`'s layout: for
/// [`KvLayout::AppendLast`], each sequence's first `past_len - 1` rows from
/// its handle and its newest from the step's output (row `upper` of the
/// sequence), which a bucket's last step has no handle row for. Read right
/// after the step, before another run.
pub fn sync_gpu_kv_to_host_as(
    compiled: &CompiledGraph,
    kv: &mut KvCacheState,
    kv_dim: usize,
    num_layers: usize,
    binding: &GpuKvBinding,
) -> Result<()> {
    if binding.layout != KvLayout::AppendLast {
        return sync_gpu_kv_to_host(compiled, kv, kv_dim, num_layers);
    }
    let names = past_kv_input_names(num_layers);
    let batch = binding.sequences();
    let upper = binding.upper as usize;
    let past = kv.past_len;
    if past == 0 {
        return Ok(());
    }
    for layer in 0..num_layers {
        for (i, out) in [(2 * layer, 1 + 2 * layer), (2 * layer + 1, 2 + 2 * layer)] {
            let handle = compiled
                .read_gpu_handle(&names[i])
                .with_context(|| format!("reading cache {}", names[i]))?;
            let mut rows = Vec::with_capacity(batch * past * kv_dim);
            for b in 0..batch {
                let at = b * upper * kv_dim;
                let old = handle
                    .get(at..at + (past - 1) * kv_dim)
                    .with_context(|| format!("cache {} holds no {past} rows", names[i]))?;
                rows.extend_from_slice(old);
                let new = compiled
                    .read_output_row(out, b * (upper + 1) + upper, kv_dim)
                    .with_context(|| format!("reading the new row of output {out}"))?;
                rows.extend_from_slice(&new);
            }
            if i % 2 == 0 {
                kv.layers_k[layer] = rows;
            } else {
                kv.layers_v[layer] = rows;
            }
        }
    }
    Ok(())
}

/// After a step with [`KvLayout::AppendLast`]: each sequence's new row (row
/// `upper` of its output) into its handle at `past_seq`. A bucket's last
/// step (`past_seq == upper`) has no handle row for it, and a backend
/// without row feeds keeps none: the caches then go to `kv` on the host and
/// the next step installs them again.
fn append_new_rows(
    compiled: &mut CompiledGraph,
    kv: &mut KvCacheState,
    binding: &mut GpuKvBinding,
    past_seq: usize,
    kv_dim: usize,
    num_layers: usize,
) -> Result<()> {
    if binding.layout != KvLayout::AppendLast {
        return Ok(());
    }
    let upper = binding.upper as usize;
    if past_seq >= upper {
        // The next step is in a longer bucket, installed from `kv`.
        return sync_gpu_kv_to_host_as(compiled, kv, kv_dim, num_layers, binding);
    }
    let fed = (0..binding.sequences())
        .all(|b| compiled.feed_kv_row(b * (upper + 1) + upper, b * upper + past_seq, kv_dim));
    if !fed {
        sync_gpu_kv_to_host_as(compiled, kv, kv_dim, num_layers, binding)?;
        binding.upper = 0;
    }
    Ok(())
}

fn ensure_gpu_kv_bindings(
    compiled: &mut CompiledGraph,
    kv: &KvCacheState,
    binding: &mut GpuKvBinding,
    upper: u64,
    kv_dim: usize,
    num_layers: usize,
    refresh_kv: bool,
) -> Result<()> {
    let names = past_kv_input_names(num_layers);
    let handles_live = compiled.has_gpu_handle(names[0].as_str());
    if refresh_kv || !handles_live || binding.upper != upper {
        binding.upper = upper;
        install_gpu_kv_handles_as(
            compiled,
            kv,
            kv.past_len,
            upper,
            kv_dim,
            num_layers,
            *binding,
        )?;
    }
    Ok(())
}

/// One bucketed decode step with GPU-resident K/V (output 0 readback — logits or hidden_states).
///
/// `cache_key` indexes the compile bucket (for batch>1 use `(batch << 32) | past_seq`).
pub fn run_bucketed_kv_decode_gpu<F>(
    cache: &mut BucketedCompileCache,
    cache_key: u64,
    past_seq: usize,
    kv: &mut KvCacheState,
    binding: &mut GpuKvBinding,
    kv_dim: usize,
    num_layers: usize,
    fixed_inputs: &[CacheRunInput<'_>],
    build: F,
    options: &CompileOptions,
    refresh_kv: bool,
) -> Result<Vec<f32>>
where
    F: FnOnce(u64) -> (Graph, HashMap<String, Vec<f32>>),
{
    let detect = binding.layout == KvLayout::Detect;
    let mut detected = None;
    let build = |upper| {
        let (graph, params) = build(upper);
        if detect {
            detected = KvLayout::of_graph(&graph);
        }
        (graph, params)
    };
    let (upper, compiled) = cache
        .ensure_graph_with_params(cache_key, build, options)
        .ok_or_else(|| anyhow::anyhow!("cache_key {cache_key} outside decode buckets"))?;
    if let Some(layout) = detected {
        binding.layout = layout;
    }

    ensure_gpu_kv_bindings(compiled, kv, binding, upper, kv_dim, num_layers, refresh_kv)?;

    let mut pairs: Vec<(&str, &[f32])> = Vec::with_capacity(fixed_inputs.len());
    for inp in fixed_inputs {
        pairs.push((inp.name, inp.data));
    }

    // Metal: skip active extent (CPU ignores it; Metal SDPA scaling breaks bucketed mask).
    if compiled.device() != Device::Metal {
        compiled.set_active_extent(Some((upper as usize + 1, upper as usize + 1)));
    }
    let outs = compiled.run_read_outputs(&pairs, Some(&[0]));
    compiled.set_active_extent(None);

    let logits = outs
        .into_iter()
        .next()
        .context("gpu kv decode: missing logits output")?;
    kv.past_len = past_seq + 1;
    append_new_rows(compiled, kv, binding, past_seq, kv_dim, num_layers)?;
    Ok(logits)
}

/// HIR variant of [`run_bucketed_kv_decode_gpu`] (stable Metal bucketed decode).
pub fn run_bucketed_kv_decode_gpu_hir<F>(
    cache: &mut BucketedCompileCache,
    cache_key: u64,
    past_seq: usize,
    kv: &mut KvCacheState,
    binding: &mut GpuKvBinding,
    kv_dim: usize,
    num_layers: usize,
    fixed_inputs: &[CacheRunInput<'_>],
    build: F,
    options: &CompileOptions,
    refresh_kv: bool,
) -> Result<Vec<f32>>
where
    F: FnOnce(u64) -> (HirModule, HashMap<String, Vec<f32>>),
{
    let detect = binding.layout == KvLayout::Detect;
    let mut detected = None;
    let build = |upper| {
        let (module, params) = build(upper);
        if detect {
            detected = KvLayout::of_hir(&module);
        }
        (module, params)
    };
    let (upper, compiled) = cache
        .ensure_hir_with_params(cache_key, build, options)
        .ok_or_else(|| anyhow::anyhow!("cache_key {cache_key} outside decode buckets"))?;
    if let Some(layout) = detected {
        binding.layout = layout;
    }

    ensure_gpu_kv_bindings(compiled, kv, binding, upper, kv_dim, num_layers, refresh_kv)?;

    let mut pairs: Vec<(&str, &[f32])> = Vec::with_capacity(fixed_inputs.len());
    for inp in fixed_inputs {
        pairs.push((inp.name, inp.data));
    }

    if compiled.device() != Device::Metal {
        compiled.set_active_extent(Some((upper as usize + 1, upper as usize + 1)));
    }
    let outs = compiled.run_read_outputs(&pairs, Some(&[0]));
    compiled.set_active_extent(None);

    let logits = outs
        .into_iter()
        .next()
        .context("gpu kv decode: missing logits output")?;
    kv.past_len = past_seq + 1;
    append_new_rows(compiled, kv, binding, past_seq, kv_dim, num_layers)?;
    Ok(logits)
}

/// MTP query block with GPU-resident prefix K/V (logits slab readback only).
pub fn run_bucketed_kv_mtp_gpu<F>(
    cache: &mut BucketedCompileCache,
    past_len: usize,
    q_len: usize,
    kv: &KvCacheState,
    binding: &mut GpuKvBinding,
    kv_dim: usize,
    num_layers: usize,
    fixed_inputs: &[CacheRunInput<'_>],
    build: F,
    options: &CompileOptions,
) -> Result<Vec<f32>>
where
    F: FnOnce(u64) -> (Graph, HashMap<String, Vec<f32>>),
{
    let key = past_len as u64;
    let (upper, compiled) = cache
        .ensure_graph_with_params(key, build, options)
        .ok_or_else(|| anyhow::anyhow!("past_len {past_len} outside MTP buckets"))?;

    ensure_gpu_kv_bindings(compiled, kv, binding, upper, kv_dim, num_layers, false)?;
    let actual_kv = past_len + q_len;
    let upper_kv = upper as usize + q_len;
    let mut pairs: Vec<(&str, &[f32])> = Vec::with_capacity(fixed_inputs.len());
    for inp in fixed_inputs {
        pairs.push((inp.name, inp.data));
    }
    compiled.set_active_extent(Some((actual_kv, upper_kv)));
    let outs = compiled.run_read_outputs(&pairs, Some(&[0]));
    compiled.set_active_extent(None);

    outs.into_iter()
        .next()
        .context("gpu kv mtp: missing logits output")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autoregressive::compact_bucketed_kv_buffer;
    use rlx_runtime::Device;

    /// A step's layout from its shapes: `concat(past, new)` returns one row
    /// more than it takes, an in-place append the same.
    #[test]
    fn a_step_says_its_kv_layout() {
        use rlx_ir::{DType, GraphExt, Shape};
        let step = |out_rows: usize| {
            let f = |d: &[usize]| Shape::new(d, DType::F32);
            let mut g = Graph::new("step");
            let x = g.input("x", f(&[1, 4]));
            let past = g.input("past_k_0", f(&[2, 8, 4]));
            let out = g.input("other", f(&[2, out_rows, 4]));
            let y = g.add(x, x);
            let _ = past;
            g.set_outputs(vec![y, out]);
            g
        };
        assert_eq!(KvLayout::of_graph(&step(9)), Some(KvLayout::AppendLast));
        assert_eq!(KvLayout::of_graph(&step(8)), Some(KvLayout::InPlace));
        assert_eq!(KvLayout::of_graph(&step(3)), None);
        assert_eq!(KvLayout::default(), KvLayout::Detect);
    }

    /// Each sequence of a `[batch, past, kv]` cache padded to its own
    /// `upper` rows, not the whole cache padded at its end.
    #[test]
    fn sequences_pad_each_to_the_bucket() {
        // 2 sequences, 2 rows of width 1 each: [a0 a1 | b0 b1] → upper 3.
        let host = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            pad_kv_sequences(&host, 2, 2, 1, 2, 3),
            vec![1.0, 2.0, 0.0, 3.0, 4.0, 0.0]
        );
        assert_eq!(
            pad_kv_sequences(&host[..2], 1, 2, 1, 2, 3),
            vec![1.0, 2.0, 0.0]
        );
    }

    #[test]
    fn gpu_kv_supported_backends() {
        assert!(device_supports_gpu_kv(Device::Mlx));
        assert!(device_supports_gpu_kv(Device::Metal));
        assert!(device_supports_gpu_kv(Device::Cuda));
        assert!(device_supports_gpu_kv(Device::Gpu));
        assert!(device_supports_gpu_kv(Device::Rocm));
        assert!(!device_supports_gpu_kv(Device::Cpu));
    }

    #[test]
    fn compact_bucketed_kv_skips_middle_padding() {
        let kv_dim = 2;
        // past_len=3: rows 0,1 real; row 2 padding; row 3 (upper) new token.
        let buf = vec![
            1.0, 1.1, //
            2.0, 2.1, //
            0.0, 0.0, // padding
            9.0, 9.1, // new K at upper
        ];
        let out = compact_bucketed_kv_buffer(&buf, 3, kv_dim, 1);
        assert_eq!(out, vec![1.0, 1.1, 2.0, 2.1, 9.0, 9.1]);
    }
}
