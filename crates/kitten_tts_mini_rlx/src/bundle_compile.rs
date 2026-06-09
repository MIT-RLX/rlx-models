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

//! Build HIR directly from an RLX ONNX bundle (no generated `graph.rs` stubs).

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once};

use anyhow::{Context, Result};
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut};
use rlx_ir::{HirNodeId, Op};
use rlx_onnx_import::tensor_data::TypedParams;
use rlx_onnx_import::{
    BundleNode, ImportOptions, ImportReport, build_hir_from_bundle, build_hir_from_parts,
    load_bundle,
};
use rlx_runtime::{AotCache, CompileOptions, CompiledGraph, DType, Device, Session};

use crate::kernels::register_native_kernels;
use crate::opts::GraphOptions;

static KERNELS: Once = Once::new();
fn import_cache() -> &'static Mutex<HashMap<(PathBuf, usize, usize, String), BundleImport>> {
    static CACHE: std::sync::OnceLock<
        Mutex<HashMap<(PathBuf, usize, usize, String), BundleImport>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn probe_graph_cache() -> &'static Mutex<HashMap<String, CompiledGraph>> {
    static CACHE: std::sync::OnceLock<Mutex<HashMap<String, CompiledGraph>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn bundle_dir_near_weights(weights: &Path) -> Option<PathBuf> {
    for key in [crate::opts::ONNX_BUNDLE_ENV, crate::opts::LEGACY_BUNDLE_ENV] {
        if let Ok(p) = std::env::var(key) {
            let p = PathBuf::from(p);
            if p.join("graph.json").is_file() {
                return Some(p);
            }
        }
    }
    let in_weights = weights.join("rlx_bundle");
    if in_weights.join("graph.json").is_file() {
        return Some(in_weights);
    }
    weights
        .parent()
        .map(|p| p.join("rlx_bundle"))
        .filter(|p| p.join("graph.json").is_file())
}

fn aot_cache_root() -> PathBuf {
    if let Ok(p) = std::env::var("KITTEN_RLX_AOT_CACHE") {
        return PathBuf::from(p);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("rlx/kitten_tts_aot")
}

fn ensure_compile_arena_policy() {
    // Kitten's graph packs many small params next to `input_ids` (I64). With
    // liveness reuse, occasional kernel tail writes have stomped the index
    // buffer (word-embedding Gather read row ~118 instead of 0). Disabling
    // slot reuse keeps compile memory higher but matches ORT numerics.
    static ONCE: Once = Once::new();
    ONCE.call_once(|| crate::set_env_var("RLX_ARENA_NO_REUSE", "1"));
}

fn skip_fusion_from_env() -> bool {
    std::env::var("KITTEN_RLX_SKIP_FUSION")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
}

/// Production compile options. Fusion stays off by default: dual waveform+duration
/// outputs alias under fusion today. Set `KITTEN_RLX_ENABLE_FUSION=1` to opt in.
pub fn compile_options_for(_device: Device) -> CompileOptions {
    let mut opts = CompileOptions::default();
    let enable_fusion = std::env::var("KITTEN_RLX_ENABLE_FUSION")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"));
    opts.fusion_opts.skip_fusion = !enable_fusion || skip_fusion_from_env();
    opts
}

/// Debug graphs with extra intermediate outputs: fusion off to avoid buffer aliasing.
pub fn compile_options_probe() -> CompileOptions {
    let mut opts = CompileOptions::default();
    opts.fusion_opts.skip_fusion = true;
    opts
}

pub fn ensure_kernels_registered() {
    KERNELS.call_once(register_native_kernels);
}

/// Lowered HIR + params from a bundle (cheap to clone vs re-import).
#[derive(Clone)]
pub struct BundleImport {
    pub hir: HirModule,
    pub params: HashMap<String, Vec<f32>>,
    pub typed: HashMap<String, (Vec<u8>, DType)>,
}

fn import_cache_key(bundle_dir: &Path, opts: &GraphOptions) -> (PathBuf, usize, usize, String) {
    (
        bundle_dir
            .canonicalize()
            .unwrap_or_else(|_| bundle_dir.to_path_buf()),
        opts.sequence_length,
        opts.max_waveform_samples,
        IMPORT_CACHE_TAG.to_string(),
    )
}

/// Import ONNX bundle → HIR once per `(bundle, seq, max_waveform)`; reused by debug tools.
pub fn import_from_bundle_cached(bundle_dir: &Path, opts: &GraphOptions) -> Result<BundleImport> {
    ensure_kernels_registered();
    let key = import_cache_key(bundle_dir, opts);
    if let Some(hit) = import_cache()
        .lock()
        .expect("import cache")
        .get(&key)
        .cloned()
    {
        return Ok(hit);
    }
    crate::opts::set_compile_sequence_length(opts.sequence_length);
    let mut bundle = load_bundle(bundle_dir).context("load RLX bundle")?;
    crate::bundle_patches::patch_bundle_nodes(&mut bundle.nodes, opts.sequence_length);
    let (hir, params, typed, _report) = build_hir_from_bundle(&bundle, import_opts(opts))?;
    let import = BundleImport { hir, params, typed };
    import_cache()
        .lock()
        .expect("import cache")
        .insert(key, import.clone());
    Ok(import)
}

/// Copy probe tensors into a dedicated buffer so graph outputs are not aliased views.
fn materialize_probe_output(hir: &mut HirModule, node_id: HirNodeId) -> HirNodeId {
    let shape = hir.node(node_id).shape.clone();
    let mut m = HirMut::new(hir);
    if shape.dtype() == DType::F32 {
        return m.add_node(
            Op::Custom {
                name: crate::kernels::ACT_COPY.to_string(),
                num_inputs: 1,
                attrs: vec![],
            },
            vec![node_id],
            shape,
        );
    }
    let cast = m.cast(node_id, DType::F32);
    let f32_shape = m.shape(cast).clone();
    m.add_node(
        Op::Custom {
            name: crate::kernels::ACT_COPY.to_string(),
            num_inputs: 1,
            attrs: vec![],
        },
        vec![cast],
        f32_shape,
    )
}

/// First probe tensor index when the compiled graph still exposes waveform + duration
/// (production graph). Probe-only graphs use index `0`.
pub const PROBE_OUTPUT_INDEX: usize = 2;

/// Read f32 probe output from a typed graph run (waveform, duration, then probe).
pub fn probe_output_f32(outs: &[(Vec<u8>, DType)]) -> Option<&[u8]> {
    outs.get(PROBE_OUTPUT_INDEX)
        .or_else(|| outs.first())
        .map(|(b, _)| b.as_slice())
}

/// Decode one f32 probe slot. For probe-only graphs, `probe_index` is 0-based in outputs.
pub fn probe_output_f32_at(outs: &[(Vec<u8>, DType)], probe_index: usize) -> Option<Vec<f32>> {
    let (bytes, dt) = outs.get(probe_index)?;
    if *dt != DType::F32 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

/// Run a compiled graph with parity-style inputs (`input_ids`, `style`, `speed`).
pub fn run_parity_inputs(
    graph: &mut CompiledGraph,
    seq: usize,
    ids: &[i64],
    style: &[f32],
) -> Vec<(Vec<u8>, DType)> {
    set_runtime_input_ids_shape(graph, seq).expect("input_ids shape");
    let speed = 1.0f32.to_le_bytes();
    graph.run_typed(&[
        (
            "input_ids",
            &ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            DType::I64,
        ),
        (
            "style",
            &style
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            DType::F32,
        ),
        ("speed", &speed, DType::F32),
    ])
}

fn sanitize_cache_token(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn probe_cache_key(device: Device, opts: &GraphOptions, probe: &str) -> String {
    format!(
        "{}_probe_po_{}",
        cache_key(device, opts),
        sanitize_cache_token(probe),
    )
}

fn multi_probe_cache_key(device: Device, opts: &GraphOptions, labels: &[&str]) -> String {
    let mut sorted: Vec<&str> = labels.to_vec();
    sorted.sort_unstable();
    let mut hasher = DefaultHasher::new();
    sorted.hash(&mut hasher);
    format!(
        "{}_multiprobe_po_{:016x}",
        cache_key(device, opts),
        hasher.finish()
    )
}

/// Compile graph with one extra HIR output (parity dumps). Uses disk AOT cache per probe name.
pub fn compile_probe_graph(
    device: Device,
    bundle_dir: &Path,
    opts: &GraphOptions,
    import: &BundleImport,
    extra_output: HirNodeId,
    probe_name: &str,
) -> Result<CompiledGraph> {
    ensure_compile_arena_policy();
    let _ = bundle_dir;
    ensure_kernels_registered();
    let key = probe_cache_key(device, opts, probe_name);
    if let Some(hit) = probe_graph_cache()
        .lock()
        .expect("probe cache")
        .get(&key)
        .cloned()
    {
        return Ok(hit);
    }
    // Probes target runtime token width; headroom compile length breaks static ActCopy shapes.
    crate::opts::set_compile_sequence_length(opts.sequence_length);
    let mut hir = import.hir.clone();
    let probe = materialize_probe_output(&mut hir, extra_output);
    hir.set_outputs(vec![probe]);
    let cache = AotCache::new(aot_cache_root());
    let mut compiled = cache
        .compile_hir_cached(&key, device, hir, &compile_options_probe())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    attach_params(&mut compiled, &import.params, &import.typed);
    probe_graph_cache()
        .lock()
        .expect("probe cache")
        .insert(key, compiled.clone());
    Ok(compiled)
}

/// Compile one graph with many ActCopy probe outputs (single compile + single run for parity sweeps).
pub fn compile_multi_probe_graph(
    device: Device,
    bundle_dir: &Path,
    opts: &GraphOptions,
    import: &BundleImport,
    probes: &[(HirNodeId, &str)],
) -> Result<CompiledGraph> {
    ensure_compile_arena_policy();
    let _ = bundle_dir;
    ensure_kernels_registered();
    if probes.is_empty() {
        anyhow::bail!("compile_multi_probe_graph: empty probe list");
    }
    let labels: Vec<&str> = probes.iter().map(|(_, l)| *l).collect();
    let key = multi_probe_cache_key(device, opts, &labels);
    if let Some(hit) = probe_graph_cache()
        .lock()
        .expect("probe cache")
        .get(&key)
        .cloned()
    {
        return Ok(hit);
    }
    crate::opts::set_compile_sequence_length(opts.sequence_length);
    let mut hir = import.hir.clone();
    let mut outputs = Vec::with_capacity(probes.len());
    for (node_id, _) in probes {
        outputs.push(materialize_probe_output(&mut hir, *node_id));
    }
    hir.set_outputs(outputs);
    let cache = AotCache::new(aot_cache_root());
    let mut compiled = cache
        .compile_hir_cached(&key, device, hir, &compile_options_probe())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    attach_params(&mut compiled, &import.params, &import.typed);
    probe_graph_cache()
        .lock()
        .expect("probe cache")
        .insert(key, compiled.clone());
    Ok(compiled)
}

pub fn set_runtime_input_ids_shape(graph: &mut CompiledGraph, seq: usize) -> Result<()> {
    let shape_bytes: Vec<u8> = [1i64, seq as i64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    graph.set_param_typed(
        crate::opts::RUNTIME_INPUT_IDS_SHAPE,
        &shape_bytes,
        DType::I64,
    );
    Ok(())
}

/// Bump when `rlx-onnx-import` lowering changes affect compiled graphs.
const IMPORT_CACHE_TAG: &str = "kitten_cf_v55";

/// Graph output index for per-token duration (after waveform).
pub const DURATION_OUTPUT_INDEX: usize = 1;

/// Max fixed-point iterations for the duration feedback loop (`Expand_1` / `Where_1`).
pub const DURATION_FIXED_POINT_ITERS: usize = 24;

/// Run graph with duration carry updated from the prior `duration` output until stable.
pub fn run_with_duration_fixed_point(
    graph: &mut CompiledGraph,
    inputs: &[(&str, &[u8], DType)],
) -> Vec<(Vec<u8>, DType)> {
    let mut prev_carry: Option<Vec<u8>> = None;
    let mut outs = graph.run_typed(inputs);
    for _ in 1..DURATION_FIXED_POINT_ITERS {
        let Some((dur_bytes, DType::I64)) = outs.get(DURATION_OUTPUT_INDEX) else {
            break;
        };
        if prev_carry.as_deref() == Some(dur_bytes.as_slice()) {
            break;
        }
        graph.set_param_typed(crate::opts::DURATION_CARRY, dur_bytes, DType::I64);
        prev_carry = Some(dur_bytes.clone());
        outs = graph.run_typed(inputs);
    }
    outs
}

/// Extra sequence slots when compiling so duration epilogue buffers do not alias
/// real token positions (broken through seq+5 when compile length equals token count).
pub const DURATION_COMPILE_HEADROOM: usize = 6;

/// Graph compile length for a given active token count.
pub fn compile_sequence_length(token_len: usize) -> usize {
    token_len.saturating_add(DURATION_COMPILE_HEADROOM)
}

fn cache_key(device: Device, opts: &GraphOptions) -> String {
    format!(
        "kitten_{:?}_{IMPORT_CACHE_TAG}_nf_s{}_w{}",
        device, opts.sequence_length, opts.max_waveform_samples
    )
}

fn import_opts(opts: &GraphOptions) -> ImportOptions {
    crate::bundle_patches::set_import_sequence_length(opts.sequence_length);
    ImportOptions {
        sequence_length: opts.sequence_length,
        max_waveform_samples: opts.max_waveform_samples,
        output_shape_fix: Some(crate::bundle_patches::import_output_shape_fix),
        ..ImportOptions::quant_bundle()
    }
}

/// Lower bundle → HIR, applying model-specific node rewrites when the exported
/// graph still wires the `duration` input (idempotent when carry is pre-baked).
pub fn build_hir_from_bundle_with_rewrites(
    bundle: &rlx_onnx_import::RlxBundle,
    opts: ImportOptions,
) -> Result<(
    HirModule,
    HashMap<String, Vec<f32>>,
    TypedParams,
    ImportReport,
)> {
    crate::bundle_patches::set_import_sequence_length(opts.sequence_length);
    let mut nodes = bundle.nodes.clone();
    crate::bundle_patches::patch_bundle_nodes(&mut nodes, opts.sequence_length);
    let import_opts = import_opts(&GraphOptions {
        sequence_length: opts.sequence_length,
        max_waveform_samples: opts.max_waveform_samples,
    });
    let mut params = HashMap::new();
    let mut init_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let st = bundle.weights()?;
    for name in st.names() {
        let key = name.to_string();
        let view = st.tensor(&key)?;
        init_shapes.insert(key.clone(), view.shape().to_vec());
        match view.dtype() {
            safetensors::tensor::Dtype::I64 | safetensors::tensor::Dtype::BOOL => {}
            safetensors::tensor::Dtype::U8 | safetensors::tensor::Dtype::I8
                if opts.use_quantized_kernels && key.ends_with("_quantized") => {}
            _ => {
                load_bundle_f32_param(bundle, &mut params, &key)?;
            }
        }
    }
    let i64_params =
        rlx_onnx_import::tensor_data::load_i64_params(&bundle.weight_bytes).unwrap_or_default();
    let (typed_params, quant_shapes) = if opts.use_quantized_kernels {
        rlx_onnx_import::tensor_data::load_typed_quant_params(&bundle.weight_bytes)?
    } else {
        (TypedParams::new(), HashMap::new())
    };
    init_shapes.extend(quant_shapes);
    rlx_onnx_import::tensor_data::materialize_quantized_f32(
        &bundle.weight_bytes,
        &mut params,
        &mut init_shapes,
    )?;
    rlx_onnx_import::shape_propagate::propagate_shapes(
        &mut nodes,
        &bundle.manifest,
        &init_shapes,
        &opts,
    );
    build_hir_from_parts(
        &bundle.manifest,
        nodes,
        params,
        typed_params,
        i64_params,
        &init_shapes,
        import_opts,
    )
}

fn load_bundle_f32_param(
    bundle: &rlx_onnx_import::RlxBundle,
    params: &mut HashMap<String, Vec<f32>>,
    key: &str,
) -> Result<Vec<f32>> {
    if let Some(v) = params.get(key) {
        return Ok(v.clone());
    }
    let st = bundle.weights()?;
    let view = st
        .tensor(key)
        .with_context(|| format!("missing weight {key}"))?;
    let out: Vec<f32> = match view.dtype() {
        safetensors::tensor::Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        safetensors::tensor::Dtype::F16 => view
            .data()
            .chunks_exact(2)
            .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
            .collect(),
        safetensors::tensor::Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect(),
        safetensors::tensor::Dtype::I32 => view
            .data()
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32)
            .collect(),
        safetensors::tensor::Dtype::U8 | safetensors::tensor::Dtype::I8 => {
            view.data().iter().map(|&b| b as f32).collect()
        }
        other => anyhow::bail!("load_bundle_f32_param: unsupported dtype {other:?} for {key}"),
    };
    params.insert(key.to_string(), out.clone());
    Ok(out)
}

/// Break the duration feedback cycle for single-pass import (ORT uses stale `duration`).
pub fn rewrite_duration_carry(nodes: &mut [BundleNode]) {
    for node in nodes.iter_mut() {
        if node.name == "/Expand_1" || node.name == "/Where_1" {
            for inp in node.inputs.iter_mut() {
                if inp == "duration" {
                    *inp = crate::opts::DURATION_CARRY.to_string();
                }
            }
        }
    }
}

fn attach_params(
    compiled: &mut CompiledGraph,
    params: &HashMap<String, Vec<f32>>,
    typed: &HashMap<String, (Vec<u8>, DType)>,
) {
    for (name, data) in params {
        compiled.set_param(name.as_str(), data);
    }
    for (name, (bytes, dtype)) in typed {
        compiled.set_param_typed(name.as_str(), bytes, *dtype);
    }
}

/// LRU-ish cache of graphs compiled at exact sequence lengths (ORT-style `[1, seq]`).
pub struct SeqCompileCache {
    device: Device,
    bundle_dir: PathBuf,
    max_waveform_samples: usize,
    max_sequence_length: usize,
    entries: Mutex<HashMap<usize, CompiledGraph>>,
    order: Mutex<Vec<usize>>,
    capacity: usize,
}

impl SeqCompileCache {
    pub fn new(
        device: Device,
        bundle_dir: PathBuf,
        max_sequence_length: usize,
        max_waveform_samples: usize,
        capacity: usize,
    ) -> Self {
        Self {
            device,
            bundle_dir,
            max_waveform_samples,
            max_sequence_length,
            entries: Mutex::new(HashMap::new()),
            order: Mutex::new(Vec::new()),
            capacity: capacity.max(1),
        }
    }

    pub fn max_sequence_length(&self) -> usize {
        self.max_sequence_length
    }

    pub fn graph_for_seq(&self, seq: usize) -> Result<CompiledGraph> {
        if seq > self.max_sequence_length {
            anyhow::bail!(
                "sequence {seq} exceeds compiled max {}",
                self.max_sequence_length
            );
        }
        {
            let entries = self.entries.lock().expect("seq cache entries");
            if let Some(g) = entries.get(&seq) {
                return Ok(g.clone());
            }
        }
        let compile_seq = seq;
        let max_waveform = self
            .max_waveform_samples
            .max(compile_seq.saturating_mul(600).saturating_add(12_000));
        let mut graph = compile_from_bundle(
            self.device,
            &self.bundle_dir,
            &GraphOptions {
                sequence_length: compile_seq,
                max_waveform_samples: max_waveform,
            },
        )?;
        set_runtime_input_ids_shape(&mut graph, compile_seq)?;
        let mut entries = self.entries.lock().expect("seq cache entries");
        let mut order = self.order.lock().expect("seq cache order");
        if entries.len() >= self.capacity {
            if let Some(evict) = order.first().copied() {
                entries.remove(&evict);
                order.retain(|&k| k != evict);
            }
        }
        entries.insert(seq, graph.clone());
        order.retain(|&k| k != seq);
        order.push(seq);
        Ok(graph)
    }
}

pub fn compile_from_bundle(
    device: Device,
    bundle_dir: &Path,
    opts: &GraphOptions,
) -> Result<CompiledGraph> {
    ensure_compile_arena_policy();
    let import = import_from_bundle_cached(bundle_dir, opts)?;
    crate::opts::set_compile_sequence_length(opts.sequence_length);
    let compile_opts = compile_options_for(device);
    let key = cache_key(device, opts);
    let cache = AotCache::new(aot_cache_root());
    let mut compiled = cache
        .compile_hir_cached(&key, device, import.hir, &compile_opts)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    attach_params(&mut compiled, &import.params, &import.typed);
    Ok(compiled)
}

/// Compile without disk cache (tests / benchmarking cold compile).
pub fn compile_from_bundle_fresh(
    device: Device,
    bundle_dir: &Path,
    opts: &GraphOptions,
) -> Result<CompiledGraph> {
    ensure_compile_arena_policy();
    let import = import_from_bundle_cached(bundle_dir, opts)?;
    crate::opts::set_compile_sequence_length(opts.sequence_length);
    let compile_opts = compile_options_for(device);
    let mut compiled = Session::new(device)
        .compile_hir_with(import.hir, &compile_opts)
        .map_err(|e| anyhow::anyhow!("{0}", e))?;
    attach_params(&mut compiled, &import.params, &import.typed);
    Ok(compiled)
}
