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
use std::sync::{Mutex, Once, OnceLock};
use std::thread::ScopedJoinHandle;

use anyhow::{Context, Result};
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut};
use rlx_ir::{HirNodeId, Op};
use rlx_onnx_import::tensor_data::TypedParams;
use rlx_onnx_import::{
    BundleNode, ImportOptions, ImportReport, build_hir_from_bundle, build_hir_from_parts,
    load_bundle,
};
use rlx_runtime::{AotCache, CompileOptions, CompiledGraph, DType, Device, RngOptions, Session};

use crate::compile_profile::{self, CompileProfile, optimized_split_graphs_enabled};
use crate::kernels::register_native_kernels;
use crate::opts::GraphOptions;
use crate::seq_cache::{CachedSeqGraphs, SeqGraphCache};

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

fn probe_graph_cache_enabled() -> bool {
    crate::compile_profile::infer_mode() == crate::compile_profile::InferMode::Parity
        || env_flag("KITTEN_RLX_PROBE_GRAPH_CACHE")
}

/// Join a compile worker, turning a thread panic into an `anyhow` error so callers
/// (and the multi-backend bench) see the real message instead of `Any { .. }`.
pub(crate) fn join_compile<'scope, T>(
    name: &str,
    handle: ScopedJoinHandle<'scope, Result<T>>,
) -> Result<T> {
    match handle.join() {
        Ok(r) => r,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panicked".to_string());
            anyhow::bail!("{name}: {msg}")
        }
    }
}

fn bundle_dir_local(weights: &Path) -> Option<PathBuf> {
    let in_weights = weights.join("rlx_bundle");
    if in_weights.join("graph.json").is_file() {
        return Some(in_weights);
    }
    weights
        .parent()
        .map(|p| p.join("rlx_bundle"))
        .filter(|p| p.join("graph.json").is_file())
}

pub fn bundle_dir_near_weights(weights: &Path) -> Option<PathBuf> {
    let weights = weights
        .canonicalize()
        .unwrap_or_else(|_| weights.to_path_buf());
    if let Some(local) = bundle_dir_local(&weights) {
        return Some(local);
    }
    for key in [crate::opts::ONNX_BUNDLE_ENV, crate::opts::LEGACY_BUNDLE_ENV] {
        if let Ok(p) = std::env::var(key) {
            let p = PathBuf::from(p);
            if p.join("graph.json").is_file() {
                return p.canonicalize().ok().or(Some(p));
            }
        }
    }
    None
}

fn aot_cache_root() -> PathBuf {
    if let Ok(p) = std::env::var("KITTEN_RLX_AOT_CACHE") {
        return PathBuf::from(p);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("rlx/kitten_tts_aot")
}

pub fn ensure_compile_arena_policy() {
    if crate::compile_profile::arena_reuse_allowed_for_kitten() {
        return;
    }
    crate::set_env_var("RLX_ARENA_NO_REUSE", "1");
}

pub fn skip_fusion_from_env() -> bool {
    std::env::var("KITTEN_RLX_SKIP_FUSION")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
}

/// RNG policy for vocoder noise and other in-graph random ops.
pub fn rng_options_from_env() -> RngOptions {
    if env_flag("RLX_RNG_ZERO") {
        return RngOptions::zero();
    }
    let seed = std::env::var("KITTEN_RLX_RNG_SEED")
        .or_else(|_| std::env::var("RLX_RNG_SEED"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let backend = std::env::var("KITTEN_RLX_RNG_BACKEND")
        .or_else(|_| std::env::var("RLX_RNG_BACKEND"))
        .ok()
        .as_deref()
        .map(str::to_ascii_lowercase);
    // Vocoder noise: match ONNX Runtime (Philox is quieter / mismatched vs ORT).
    if backend.is_none() {
        return RngOptions::ort(seed);
    }
    match backend.as_deref() {
        Some("ort") | Some("onnx") | Some("onnxruntime") => RngOptions::ort(seed),
        Some("zero") => RngOptions::zero(),
        Some("philox") => RngOptions::philox(seed),
        _ => RngOptions::philox(seed),
    }
}

/// Production compile options. Fusion on by default for single-output profiles
/// ([`CompileProfile::DurationRefinement`] / [`CompileProfile::WaveformOnly`]).
pub fn compile_options_base(device: Device) -> CompileOptions {
    use rlx_opt::{FusionOptions, FusionTarget};
    use rlx_runtime::fusion_target_for;

    let target = fusion_target_for(device);
    let fusion_opts = match target {
        FusionTarget::Metal => FusionOptions::for_metal(),
        FusionTarget::Cpu => FusionOptions::for_cpu(),
        FusionTarget::Wgpu => FusionOptions::for_wgpu(),
        _ => FusionOptions::default(),
    };
    let mut opts = CompileOptions::default()
        .fusion_target(target)
        .fusion_opts(fusion_opts);
    opts.fusion_opts.skip_fusion = skip_fusion_from_env();
    opts.rng = rng_options_from_env();
    opts
}

/// Dual-output full graph (legacy / parity).
pub fn compile_options_for(device: Device) -> CompileOptions {
    let mut opts = compile_options_base(device);
    opts.fusion_opts.skip_fusion = true;
    opts
}

/// Device the duration-refine graph should compile/run on.
///
/// See [`crate::device_policy::duration_device`].
#[inline]
pub fn parity_duration_device(target: Device) -> Device {
    crate::device_policy::duration_device(target)
}

/// Device the waveform graph should compile/run on.
///
/// See [`crate::device_policy::wave_device`]. Discrete Gpu/Vulkan stay on-device
/// with wave caps from [`crate::device_policy::prepare`].
#[inline]
pub fn parity_wave_device(target: Device) -> Device {
    crate::device_policy::wave_device(target)
}

/// Debug graphs with extra intermediate outputs: fusion off to avoid buffer aliasing.
pub fn compile_options_probe() -> CompileOptions {
    let mut opts = CompileOptions::default();
    opts.fusion_opts.skip_fusion = true;
    opts
}

pub fn ensure_kernels_registered() {
    // Route rank-3 InstanceNorms (F0/N AdaIN) to the active-mel-frame host kernel so they
    // don't normalize over the padded mel slot. `rlx-onnx-import` reads this at lower time,
    // so it must be set before any bundle import. Callers may pre-set it; keep idempotent.
    if std::env::var("RLX_KITTEN_INORM_ACTIVE").is_err() {
        crate::set_env_var("RLX_KITTEN_INORM_ACTIVE", "1");
    }
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
    // Always rebind mel/wave caps — import/AOT cache hits used to skip this, leaving
    // `KittenInstanceNormActive` with stale or zero caps on a fresh process.
    crate::opts::set_compile_sequence_length(opts.sequence_length);
    crate::bundle_patches::set_import_max_waveform_samples(opts.max_waveform_samples);
    let key = import_cache_key(bundle_dir, opts);
    if let Some(hit) = import_cache()
        .lock()
        .expect("import cache")
        .get(&key)
        .cloned()
    {
        return Ok(hit);
    }
    let mut bundle = load_bundle(bundle_dir).context("load RLX bundle")?;
    crate::bundle_patches::patch_bundle_nodes(
        &mut bundle.nodes,
        opts.sequence_length,
        opts.max_waveform_samples,
    );
    let (hir, params, typed, _report) = build_hir_from_bundle(&bundle, import_opts(opts))?;
    let import = BundleImport { hir, params, typed };
    import_cache_insert(key, import.clone());
    Ok(import)
}

fn import_cache_insert(key: (PathBuf, usize, usize, String), import: BundleImport) {
    let cap = crate::compile_profile::seq_compile_cache_capacity();
    let mut cache = import_cache().lock().expect("import cache");
    if cache.len() >= cap && !cache.contains_key(&key) {
        cache.clear();
    }
    cache.insert(key, import);
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
    // ORT duration for framed həˈloʊ (`…` token 10); pad/truncate to `seq`.
    let mut hello_dur = vec![3i64, 2, 2, 3, 4, 4, 13, 2, 1];
    if hello_dur.len() < seq {
        hello_dur.resize(seq, 0);
    } else {
        hello_dur.truncate(seq);
    }
    let active = ids.len().min(seq);
    run_parity_inputs_with_duration(graph, seq, active, ids, style, Some(&hello_dur))
}

/// Like [`run_parity_inputs`], but seeds duration carry from `duration` when provided.
pub fn run_parity_inputs_with_duration(
    graph: &mut CompiledGraph,
    compile_seq: usize,
    active_tokens: usize,
    ids: &[i64],
    style: &[f32],
    duration: Option<&[i64]>,
) -> Vec<(Vec<u8>, DType)> {
    set_runtime_input_ids_shape(graph, compile_seq).expect("input_ids shape");
    let active = active_tokens.min(compile_seq);
    set_runtime_active_sequence(graph, active, compile_seq);
    let mut dur: Vec<i64> = duration
        .map(|d| {
            let mut v = d.to_vec();
            if v.len() < compile_seq {
                v.resize(compile_seq, 0);
            } else if v.len() > compile_seq {
                v.truncate(compile_seq);
            }
            v
        })
        .unwrap_or_else(|| vec![3, 2, 2, 3, 4, 4, 13, 2, 1]);
    if dur.len() < compile_seq {
        dur.resize(compile_seq, 0);
    } else {
        dur.truncate(compile_seq);
    }
    let dur_bytes: Vec<u8> = dur.iter().flat_map(|v| v.to_le_bytes()).collect();
    reset_duration_carry(graph, compile_seq);
    set_duration_carry(graph, &dur_bytes);
    apply_alignment_hint(graph, &dur_bytes, active);
    let mut ids_padded = ids.to_vec();
    if ids_padded.len() < compile_seq {
        ids_padded.resize(compile_seq, 0);
    } else if ids_padded.len() > compile_seq {
        ids_padded.truncate(compile_seq);
    }
    let ids_bytes: Vec<u8> = ids_padded.iter().flat_map(|v| v.to_le_bytes()).collect();
    let style_bytes: Vec<u8> = style.iter().flat_map(|v| v.to_le_bytes()).collect();
    let speed = 1.0f32.to_le_bytes();
    let inputs: [(&str, &[u8], DType); 3] = [
        ("input_ids", &ids_bytes, DType::I64),
        ("style", &style_bytes, DType::F32),
        ("speed", &speed, DType::F32),
    ];
    // ORT duration oracle: one forward with trusted carry + alignment (no native re-seed).
    if duration.is_some() {
        return graph.run_typed(&inputs);
    }
    let mut outs = run_with_duration_fixed_point_on_graph(graph, &inputs);
    apply_alignment_hint(graph, &dur_bytes, active);
    if active < 32 {
        if let Some(carry) = first_duration_i64_bytes(&outs) {
            set_duration_carry(graph, &carry);
        }
        outs = graph.run_typed(&inputs);
    }
    outs
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
    if probe_graph_cache_enabled() {
        if let Some(hit) = probe_graph_cache()
            .lock()
            .expect("probe cache")
            .get(&key)
            .cloned()
        {
            return Ok(hit);
        }
    }
    // Probes target runtime token width; headroom compile length breaks static ActCopy shapes.
    let (mut hir, params) = prepare_hir_for_compile(
        import.hir.clone(),
        &import.params,
        &import.typed,
        opts.sequence_length,
        opts.max_waveform_samples,
    );
    let probe = materialize_probe_output(&mut hir, extra_output);
    hir.set_outputs(vec![probe]);
    let cache = AotCache::new(aot_cache_root());
    let mut compiled = cache
        .compile_hir_cached(&key, device, hir, &compile_options_probe())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    attach_params(&mut compiled, &params, &import.typed);
    if probe_graph_cache_enabled() {
        probe_graph_cache()
            .lock()
            .expect("probe cache")
            .insert(key, compiled.clone());
    }
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
    if probe_graph_cache_enabled() {
        if let Some(hit) = probe_graph_cache()
            .lock()
            .expect("probe cache")
            .get(&key)
            .cloned()
        {
            return Ok(hit);
        }
    }
    let (mut hir, params) = prepare_hir_for_compile(
        import.hir.clone(),
        &import.params,
        &import.typed,
        opts.sequence_length,
        opts.max_waveform_samples,
    );
    let mut outputs = Vec::with_capacity(probes.len());
    for (node_id, _) in probes {
        outputs.push(materialize_probe_output(&mut hir, *node_id));
    }
    hir.set_outputs(outputs);
    let cache = AotCache::new(aot_cache_root());
    let mut compiled = cache
        .compile_hir_cached(&key, device, hir, &compile_options_probe())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    attach_params(&mut compiled, &params, &import.typed);
    if probe_graph_cache_enabled() {
        probe_graph_cache()
            .lock()
            .expect("probe cache")
            .insert(key, compiled.clone());
    }
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

/// Hint backends (MLX) of active token rows vs compile slot width.
pub fn set_runtime_active_sequence(
    graph: &mut CompiledGraph,
    active_tokens: usize,
    compile_slots: usize,
) {
    crate::opts::set_runtime_active_tokens(active_tokens);
    let _ = compile_slots;
    // Kitten duration/vocoder epilogues use compile-slot tensors; active-extent
    // dispatch scales row counts and corrupts wide-sequence duration (rlx #L1).
    graph.set_active_extent(None);
    // After clearing backend extent: `set_active_extent(None)` also clears onnx_active.
    rlx_runtime::onnx_active::set_active_token_count(Some(active_tokens));
}

pub fn shape_all_graphs_for_infer(
    graphs: &CachedSeqGraphs,
    active_tokens: usize,
    compile_slots: usize,
) -> Result<()> {
    {
        let mut g = graphs.full.lock().expect("full graph");
        set_runtime_input_ids_shape(&mut g, compile_slots)?;
        set_runtime_active_sequence(&mut g, active_tokens, compile_slots);
    }
    if let Some(d) = &graphs.duration_refine {
        let mut g = d.lock().expect("dur graph");
        set_runtime_input_ids_shape(&mut g, compile_slots)?;
        set_runtime_active_sequence(&mut g, active_tokens, compile_slots);
    }
    if let Some(w) = &graphs.waveform_only {
        let mut g = w.lock().expect("wave graph");
        set_runtime_input_ids_shape(&mut g, compile_slots)?;
        set_runtime_active_sequence(&mut g, active_tokens, compile_slots);
    }
    Ok(())
}

/// Bump when `rlx-onnx-import` lowering changes affect compiled graphs.
const IMPORT_CACHE_TAG: &str = "kitten_rlx_align_v127_wgpu_mel16";

/// Static alignment buffer multiplier (`seq * N`); must match import + HIR inject.
///
/// Default 64 is very conservative (long IPA ≈ 3 frames/token). On NVIDIA wgpu the
/// mel axis dominates the act arena — use [`max_frames_per_token`] (Gpu defaults to 16).
pub const MAX_FRAMES_PER_TOKEN: usize = 64;

/// Effective `max_frames_per_token` for import / mel caps.
#[inline]
pub fn max_frames_per_token() -> usize {
    crate::device_policy::max_frames_per_token()
}

/// Graph output index for per-token duration (after waveform).
pub const DURATION_OUTPUT_INDEX: usize = 1;

/// Max fixed-point iterations for the duration feedback loop (`Expand_1` / `Where_1`).
pub const DURATION_FIXED_POINT_ITERS: usize = 24;

/// Process-wide cache of refined duration carries so Metal/MLX backends in the same
/// process reuse the exact CPU-computed i64 durations (avoids multi-stable fixed points
/// from leftover mel-frame globals across sequential backend loads).
fn duration_parity_cache() -> &'static Mutex<HashMap<u64, Vec<u8>>> {
    static CACHE: OnceLock<Mutex<HashMap<u64, Vec<u8>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn duration_parity_cache_enabled() -> bool {
    !compile_profile::env_flag("KITTEN_RLX_NO_DURATION_CACHE")
}

fn inputs_fingerprint(inputs: &[(&str, &[u8], DType)]) -> u64 {
    let mut h = DefaultHasher::new();
    for (name, bytes, dt) in inputs {
        name.hash(&mut h);
        bytes.hash(&mut h);
        format!("{dt:?}").hash(&mut h);
    }
    crate::opts::runtime_active_tokens()
        .unwrap_or(0)
        .hash(&mut h);
    h.finish()
}

/// Run graph with duration carry updated from the prior `duration` output until stable.
pub fn run_with_duration_fixed_point(
    graph: &mut CompiledGraph,
    inputs: &[(&str, &[u8], DType)],
) -> Vec<(Vec<u8>, DType)> {
    run_with_duration_fixed_point_on_graph(graph, inputs)
}

fn run_with_duration_fixed_point_on_graph(
    graph: &mut CompiledGraph,
    inputs: &[(&str, &[u8], DType)],
) -> Vec<(Vec<u8>, DType)> {
    let compile_seq = carry_len_from_inputs(inputs);
    let active_tokens = crate::opts::runtime_active_tokens().unwrap_or(compile_seq);
    let max_iters = duration_refine_iters(active_tokens);
    let mut outs = graph.run_typed(inputs);
    if max_iters == 1 {
        return outs;
    }
    let mut prev_carry: Option<Vec<u8>> = None;
    for _ in 1..max_iters {
        let Some((raw, dt)) = outs.get(DURATION_OUTPUT_INDEX) else {
            break;
        };
        let Some(dur_bytes) = duration_output_as_i64_bytes(raw, *dt) else {
            break;
        };
        if prev_carry.as_deref() == Some(dur_bytes.as_slice()) {
            break;
        }
        set_duration_carry(graph, &dur_bytes);
        prev_carry = Some(dur_bytes);
        outs = graph.run_typed(inputs);
    }
    outs
}

fn carry_len_from_inputs(inputs: &[(&str, &[u8], DType)]) -> usize {
    if let Some(active) = crate::opts::runtime_active_tokens() {
        return compile_profile::compile_slot_length(active);
    }
    let token_len = inputs
        .iter()
        .find(|(name, _, _)| *name == "input_ids")
        .map(|(_, bytes, _)| bytes.len() / 8)
        .unwrap_or(1);
    compile_profile::compile_slot_length(token_len.max(1))
}

/// Zero-initialize duration carry on a single graph (each infer starts fresh).
pub fn reset_duration_carry(graph: &mut CompiledGraph, compile_seq: usize) {
    let seed = compile_profile::duration_carry_seed_bytes(compile_seq);
    graph.set_param_typed(crate::opts::DURATION_CARRY, &seed, DType::I64);
}

fn set_duration_carry(graph: &mut CompiledGraph, dur_bytes: &[u8]) {
    graph.set_param_typed(crate::opts::DURATION_CARRY, dur_bytes, DType::I64);
}

/// Metal/MLX f32-uniform arenas often type the duration output as F32. Normalize to
/// little-endian i64 bytes so carry / alignment / trim all share one layout.
fn duration_output_as_i64_bytes(bytes: &[u8], dtype: DType) -> Option<Vec<u8>> {
    match dtype {
        DType::I64 => Some(bytes.to_vec()),
        DType::F32 => {
            let vals: Vec<i64> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).round() as i64)
                .collect();
            Some(vals.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        DType::I32 => {
            let vals: Vec<i64> = bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                .collect();
            Some(vals.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        _ => None,
    }
}

fn first_duration_i64_bytes(outs: &[(Vec<u8>, DType)]) -> Option<Vec<u8>> {
    // Prefer a real I64 duration tensor (CPU / typed arenas).
    if let Some((b, dt)) = outs.iter().find(|(_, dt)| *dt == DType::I64) {
        return duration_output_as_i64_bytes(b, *dt);
    }
    // Metal/MLX: duration is often F32. Full graphs keep waveform at slot 0 and
    // duration at slot 1; duration-refine graphs expose duration alone at slot 0.
    let idx = if outs.len() >= 2 {
        DURATION_OUTPUT_INDEX
    } else {
        0
    };
    outs.get(idx)
        .and_then(|(b, dt)| duration_output_as_i64_bytes(b, *dt))
}

fn duration_refine_iters(active_tokens: usize) -> usize {
    if compile_profile::duration_external_fixed_point_enabled() {
        return DURATION_FIXED_POINT_ITERS;
    }
    // Short utterances need a few carry passes (true single-pass diverges from ORT).
    // Cap well below the parity fixed-point budget — early-exit still applies.
    if active_tokens <= 32 {
        if let Ok(raw) = std::env::var("KITTEN_RLX_DURATION_ITERS") {
            if let Ok(n) = raw.parse::<usize>() {
                return n.clamp(1, DURATION_FIXED_POINT_ITERS);
            }
        }
        // Production: 4 is enough for hello-class IPA on CPU/Cuda/Vulkan (peak-held).
        // Parity keeps the full 24-iter budget.
        return if compile_profile::infer_mode() == compile_profile::InferMode::Production {
            4
        } else {
            DURATION_FIXED_POINT_ITERS
        };
    }
    1
}

fn alignment_frames_from_duration_bytes(dur_bytes: &[u8], active_tokens: usize) -> usize {
    let vals: Vec<i64> = dur_bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("i64")))
        .collect();
    crate::alignment::alignment_frame_count(&vals[..active_tokens.min(vals.len())])
}

fn set_alignment_frame_param(graph: &mut CompiledGraph, frames: usize) {
    let v = frames.max(1) as i64;
    graph.set_param_typed(
        crate::opts::ALIGNMENT_FRAME_COUNT,
        &v.to_le_bytes(),
        DType::I64,
    );
}

fn apply_alignment_hint(graph: &mut CompiledGraph, dur_bytes: &[u8], active_tokens: usize) {
    let frames = alignment_frames_from_duration_bytes(dur_bytes, active_tokens);
    if compile_profile::env_flag("KITTEN_RLX_DEBUG_DURATION") {
        eprintln!("[kitten] alignment_frame_count={frames} active={active_tokens}");
    }
    set_alignment_frame_param(graph, frames);
    crate::opts::set_runtime_mel_frames(frames);
}

fn apply_alignment_hint_preferred(
    graph: &mut CompiledGraph,
    active_tokens: usize,
    alignment_duration: Option<&[u8]>,
    fallback: Option<&[u8]>,
) {
    if let Some(hint) = alignment_duration {
        apply_alignment_hint(graph, hint, active_tokens);
    } else if let Some(fb) = fallback {
        apply_alignment_hint(graph, fb, active_tokens);
    }
}

pub(crate) fn attach_alignment_frame_param(compiled: &mut CompiledGraph) {
    set_alignment_frame_param(compiled, 1);
    crate::opts::set_runtime_mel_frames(1);
}

/// Optimized infer: waveform-only in production, or duration refine + waveform pass.
///
/// When `carry_seed` is set, it is written to `DURATION_CARRY` on the waveform graph
/// (and skips duration refinement) so native vocoder length matches ORT exactly.
pub fn run_kitten_inference(
    graphs: &CachedSeqGraphs,
    inputs: &[(&str, &[u8], DType)],
    carry_seed: Option<&[u8]>,
    alignment_duration: Option<&[u8]>,
) -> Vec<(Vec<u8>, DType)> {
    let compile_seq = carry_len_from_inputs(inputs);
    let active_tokens = crate::opts::runtime_active_tokens().unwrap_or(compile_seq);
    let _ = shape_all_graphs_for_infer(graphs, active_tokens, compile_seq);
    // Refresh process-wide hint immediately before execute (bucket cache may clear it).
    rlx_runtime::onnx_active::set_active_token_count(Some(active_tokens));

    if let Some(seed) = carry_seed {
        let mut g = graphs.lock_infer_graph();
        g.set_param_typed(crate::opts::DURATION_CARRY, seed, DType::I64);
        apply_alignment_hint_preferred(&mut g, active_tokens, alignment_duration, Some(seed));
        if compile_profile::env_flag("KITTEN_RLX_DEBUG_DURATION") {
            eprintln!(
                "[kitten] carry infer active_tokens={active_tokens} compile_seq={compile_seq} onnx_active={:?}",
                rlx_runtime::onnx_active::active_token_count()
            );
        }
        // Waveform-only production graphs have no duration output; one ORT-seeded pass only.
        if compile_profile::production_waveform_only_infer() {
            return g.run_typed(inputs);
        }
        let mut outs = run_with_duration_fixed_point_on_graph(&mut g, inputs);
        apply_alignment_hint_preferred(&mut g, active_tokens, alignment_duration, Some(seed));
        if active_tokens < 32 {
            if let Some(dur_bytes) = first_duration_i64_bytes(&outs) {
                set_duration_carry(&mut g, &dur_bytes);
            }
            outs = g.run_typed(inputs);
        }
        if outs.len() >= 2 {
            return outs;
        }
        if let Some(wave_arc) = &graphs.waveform_only {
            let mut wave = wave_arc.lock().expect("waveform graph");
            wave.set_param_typed(crate::opts::DURATION_CARRY, seed, DType::I64);
            apply_alignment_hint_preferred(
                &mut wave,
                active_tokens,
                alignment_duration,
                Some(seed),
            );
            return wave.run_typed(inputs);
        }
        return outs;
    }

    if let Some(wave_arc) = &graphs.waveform_only {
        if compile_profile::production_waveform_only_infer() || graphs.duration_refine.is_none() {
            let mut wave = wave_arc.lock().expect("waveform graph");
            // Waveform-only graphs strip the duration loop; seed carry from ORT when available.
            if let Some(hint) = alignment_duration {
                set_duration_carry(&mut wave, hint);
                apply_alignment_hint(&mut wave, hint, active_tokens);
            } else {
                reset_duration_carry(&mut wave, compile_seq);
            }
            return wave.run_typed(inputs);
        }
    }

    if let (Some(dur_arc), Some(wave_arc)) = (&graphs.duration_refine, &graphs.waveform_only) {
        let fp = inputs_fingerprint(inputs);
        // Cache refined duration for any backend (CPU was the original parity
        // case). Cuda/Vulkan benefit on repeated phrases / warm benches.
        let cached = if duration_parity_cache_enabled() {
            duration_parity_cache()
                .lock()
                .expect("duration parity cache")
                .get(&fp)
                .cloned()
        } else {
            None
        };
        let timing = compile_profile::env_flag("KITTEN_RLX_TIMING");
        let dur_t0 = timing.then(std::time::Instant::now);
        let mut dur_iters_used = 0usize;
        let last_dur = if let Some(dur_bytes) = cached {
            if compile_profile::env_flag("KITTEN_RLX_DEBUG_DURATION") {
                eprintln!(
                    "[kitten] duration cache hit fp={fp:#x} bytes={}",
                    dur_bytes.len()
                );
            }
            let frames = alignment_frames_from_duration_bytes(&dur_bytes, active_tokens);
            crate::opts::set_runtime_mel_frames(frames.max(1));
            Some(dur_bytes)
        } else {
            if compile_profile::env_flag("KITTEN_RLX_DEBUG_DURATION") {
                eprintln!("[kitten] duration cache miss fp={fp:#x}");
            }
            let mut dur = dur_arc.lock().expect("duration refine graph");
            reset_duration_carry(&mut dur, compile_seq);
            // Prosody AdaIN reads process-global `runtime_mel_frames` (= alignment-frame sum, NOT
            // 2×). Leftover values from another backend's prewarm/infer make the (CPU-compiled)
            // duration graph yield device-varying durations. Reset to a deterministic seed, then
            // self-update from each iteration's own alignment sum so the fixed point is shared
            // across CPU / Metal / MLX.
            crate::opts::set_runtime_mel_frames(1);
            let mut prev_carry: Option<Vec<u8>> = None;
            let mut last_dur: Option<Vec<u8>> = None;
            let max_iters = duration_refine_iters(active_tokens);
            for _ in 0..max_iters {
                dur_iters_used += 1;
                let outs = dur.run_typed(inputs);
                let Some(dur_bytes) = first_duration_i64_bytes(&outs) else {
                    break;
                };
                let frames = alignment_frames_from_duration_bytes(&dur_bytes, active_tokens);
                crate::opts::set_runtime_mel_frames(frames.max(1));
                if prev_carry.as_deref() == Some(dur_bytes.as_slice()) {
                    last_dur = Some(dur_bytes);
                    break;
                }
                last_dur = Some(dur_bytes.clone());
                if max_iters == 1 {
                    break;
                }
                set_duration_carry(&mut dur, &dur_bytes);
                prev_carry = Some(dur_bytes);
            }
            if duration_parity_cache_enabled() {
                if let Some(ref d) = last_dur {
                    duration_parity_cache()
                        .lock()
                        .expect("duration parity cache")
                        .insert(fp, d.clone());
                }
            }
            last_dur
        };
        let dur_secs = dur_t0.map(|t| t.elapsed().as_secs_f64());
        let wave_t0 = timing.then(std::time::Instant::now);
        let wave_outs = {
            let mut wave = wave_arc.lock().expect("waveform graph");
            if let Some(dur_bytes) = &last_dur {
                set_duration_carry(&mut wave, dur_bytes);
            } else {
                reset_duration_carry(&mut wave, compile_seq);
            }
            apply_alignment_hint_preferred(
                &mut wave,
                active_tokens,
                alignment_duration,
                last_dur.as_deref(),
            );
            wave.run_typed(inputs)
        };
        if let (Some(d), Some(t0)) = (dur_secs, wave_t0) {
            eprintln!(
                "[kitten] stage duration={d:.3}s (iters={dur_iters_used}) wave={:.3}s",
                t0.elapsed().as_secs_f64()
            );
        }
        if let Some(dur) = last_dur {
            let mut out = wave_outs;
            out.push((dur, DType::I64));
            return out;
        }
        return wave_outs;
    }

    let mut g = graphs.lock_infer_graph();
    reset_duration_carry(&mut g, compile_seq);
    if let Some(hint) = alignment_duration {
        apply_alignment_hint(&mut g, hint, active_tokens);
    }
    if compile_profile::env_flag("KITTEN_RLX_DEBUG_DURATION") {
        eprintln!(
            "[kitten] infer active_tokens={active_tokens} compile_seq={compile_seq} onnx_active={:?}",
            rlx_runtime::onnx_active::active_token_count()
        );
    }
    let mut outs = run_with_duration_fixed_point_on_graph(&mut g, inputs);
    if let Some(hint) = alignment_duration {
        if active_tokens < 32 {
            if let Some(dur_bytes) = first_duration_i64_bytes(&outs) {
                set_duration_carry(&mut g, &dur_bytes);
            }
            apply_alignment_hint(&mut g, hint, active_tokens);
            outs = g.run_typed(inputs);
        }
    } else if active_tokens >= 32 && carry_seed.is_none() && alignment_duration.is_none() {
        if let Some(dur_bytes) = first_duration_i64_bytes(&outs) {
            if compile_profile::env_flag("KITTEN_RLX_DEBUG_DURATION") {
                eprintln!(
                    "[kitten] wide-seq second pass (duration bytes={})",
                    dur_bytes.len()
                );
            }
            apply_alignment_hint(&mut g, &dur_bytes, active_tokens);
            set_duration_carry(&mut g, &dur_bytes);
            outs = run_with_duration_fixed_point_on_graph(&mut g, inputs);
        }
    }
    outs
}

fn compile_hir_profile(
    device: Device,
    key_prefix: &str,
    profile: CompileProfile,
    hir: rlx_ir::hir::HirModule,
    params: &HashMap<String, Vec<f32>>,
    typed: &HashMap<String, (Vec<u8>, DType)>,
    carry_bytes: Option<&[u8]>,
    sequence_length: usize,
    max_waveform_samples: usize,
) -> Result<CompiledGraph> {
    let key = format!(
        "{key_prefix}_{}",
        compile_profile::aot_cache_suffix(profile)
    );
    if mem_graph_cache_enabled() {
        if let Some(hit) = mem_graph_cache().lock().expect("mem graph cache").get(&key) {
            let mut compiled = hit.clone();
            if let Some(carry) = carry_bytes {
                compiled.set_param_typed(crate::opts::DURATION_CARRY, carry, DType::I64);
            }
            return Ok(compiled);
        }
    }
    let (mut hir, params) =
        prepare_hir_for_compile(hir, params, typed, sequence_length, max_waveform_samples);
    if crate::device_policy::native_qmatmul(device) {
        let n = crate::hir_qdq_fuse::rewrite_qmatmul_to_native_f32(&mut hir);
        if n > 0 && compile_profile::env_flag("KITTEN_RLX_TIMING") {
            eprintln!(
                "[kitten] native QMatMul: {n} quantized matmul(s) → on-device GEMM ({device:?})"
            );
        }
    }
    let compile_opts = compile_profile::compile_options_for_profile(device, profile);
    let mut compiled = compile_prepared_hir_cached(&key, device, hir, &params, &compile_opts)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    attach_params(&mut compiled, &params, typed);
    if let Some(carry) = carry_bytes {
        compiled.set_param_typed(crate::opts::DURATION_CARRY, carry, DType::I64);
    }
    attach_alignment_frame_param(&mut compiled);
    if mem_graph_cache_enabled() {
        mem_graph_cache_insert(key, compiled.clone());
    }
    Ok(compiled)
}

static MEM_GRAPH_CACHE: OnceLock<Mutex<HashMap<String, CompiledGraph>>> = OnceLock::new();

fn mem_graph_cache() -> &'static Mutex<HashMap<String, CompiledGraph>> {
    MEM_GRAPH_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn mem_graph_cache_enabled() -> bool {
    env_flag("KITTEN_RLX_GRAPH_CACHE")
}

fn mem_graph_cache_insert(key: String, compiled: CompiledGraph) {
    let cap = crate::compile_profile::seq_compile_cache_capacity();
    let mut cache = mem_graph_cache().lock().expect("mem graph cache");
    if cache.len() >= cap && !cache.contains_key(&key) {
        cache.clear();
    }
    cache.insert(key, compiled);
}

pub fn build_cached_graphs_from_import(
    device: Device,
    key_prefix: &str,
    import: &BundleImport,
    carry_bytes: Option<&[u8]>,
    sequence_length: usize,
    max_waveform_samples: usize,
) -> Result<CachedSeqGraphs> {
    // Bind before split-graph worker threads run `prepare_hir_for_compile`.
    crate::opts::set_compile_sequence_length(sequence_length);
    crate::bundle_patches::set_import_sequence_length(sequence_length);
    crate::bundle_patches::set_import_max_waveform_samples(max_waveform_samples);
    if optimized_split_graphs_enabled() {
        let compile_dur = compile_profile::compile_duration_refine_graph();
        // Duration/alignment is an integer subgraph. On GPU (Metal/MLX) it runs through the
        // f32-uniform arena and diverges a few frames from CPU (e.g. sum 39→45), which shifts
        // the whole prosody/alignment envelope and flips ASR on short clips. Compile the
        // duration-refine graph on CPU so both backends share the exact same i64 durations;
        // the GPU vocoder is then seeded from that carry (see `run_kitten_inference`).
        let dur_device = parity_duration_device(device);
        let wave_device = parity_wave_device(device);
        let dur_key = if dur_device == device {
            key_prefix.to_string()
        } else {
            format!("{key_prefix}_dur{dur_device:?}")
        };
        let wave_key = if wave_device == device {
            key_prefix.to_string()
        } else {
            format!("{key_prefix}_wave{wave_device:?}")
        };
        let params = import.params.clone();
        let typed = import.typed.clone();
        let carry_owned = carry_bytes.map(|b| b.to_vec());
        // Parallel dur/wave compile races on process-global mel/wave caps when devices differ.
        let serial = dur_device != device
            || wave_device != device
            || compile_profile::env_flag("KITTEN_RLX_SERIAL_COMPILE");

        let (dur, wave) = if compile_dur {
            if serial {
                let mut hir_dur = import.hir.clone();
                compile_profile::apply_profile(&mut hir_dur, CompileProfile::DurationRefinement);
                let dur = compile_hir_profile(
                    dur_device,
                    &dur_key,
                    CompileProfile::DurationRefinement,
                    hir_dur,
                    &params,
                    &typed,
                    carry_bytes,
                    sequence_length,
                    max_waveform_samples,
                )?;
                let mut hir_wave = import.hir.clone();
                compile_profile::apply_profile(&mut hir_wave, CompileProfile::WaveformOnly);
                let wave = compile_hir_profile(
                    wave_device,
                    &wave_key,
                    CompileProfile::WaveformOnly,
                    hir_wave,
                    &params,
                    &typed,
                    carry_bytes,
                    sequence_length,
                    max_waveform_samples,
                )?;
                (Some(dur), wave)
            } else {
                let key_d = dur_key.clone();
                let key_w = wave_key.clone();
                let import_d = import.clone();
                let import_w = import.clone();
                let params_d = params.clone();
                let params_w = params.clone();
                let typed_d = typed.clone();
                let typed_w = typed.clone();
                let carry_d = carry_owned.clone();
                let carry_w = carry_owned;
                std::thread::scope(|scope| -> Result<(Option<CompiledGraph>, CompiledGraph)> {
                    let dur_h = scope.spawn(move || {
                        let mut hir_dur = import_d.hir;
                        compile_profile::apply_profile(
                            &mut hir_dur,
                            CompileProfile::DurationRefinement,
                        );
                        compile_hir_profile(
                            dur_device,
                            &key_d,
                            CompileProfile::DurationRefinement,
                            hir_dur,
                            &params_d,
                            &typed_d,
                            carry_d.as_deref(),
                            sequence_length,
                            max_waveform_samples,
                        )
                    });
                    let wave_h = scope.spawn(move || {
                        let mut hir_wave = import_w.hir;
                        compile_profile::apply_profile(&mut hir_wave, CompileProfile::WaveformOnly);
                        compile_hir_profile(
                            wave_device,
                            &key_w,
                            CompileProfile::WaveformOnly,
                            hir_wave,
                            &params_w,
                            &typed_w,
                            carry_w.as_deref(),
                            sequence_length,
                            max_waveform_samples,
                        )
                    });
                    let dur = join_compile("duration compile thread", dur_h)?;
                    let wave = join_compile("waveform compile thread", wave_h)?;
                    Ok((Some(dur), wave))
                })?
            }
        } else {
            let mut hir_wave = import.hir.clone();
            compile_profile::apply_profile(&mut hir_wave, CompileProfile::WaveformOnly);
            let wave = compile_hir_profile(
                wave_device,
                &wave_key,
                CompileProfile::WaveformOnly,
                hir_wave,
                &params,
                &typed,
                carry_bytes,
                sequence_length,
                max_waveform_samples,
            )?;
            (None, wave)
        };

        let wave_arc = std::sync::Arc::new(std::sync::Mutex::new(wave));
        let dur_arc = dur.map(|g| std::sync::Arc::new(std::sync::Mutex::new(g)));
        let full_arc = if compile_profile::compile_split_full_fallback() {
            let mut hir_full = import.hir.clone();
            compile_profile::apply_profile(&mut hir_full, CompileProfile::Full);
            let full = compile_hir_profile(
                device,
                key_prefix,
                CompileProfile::Full,
                hir_full,
                &import.params,
                &import.typed,
                carry_bytes,
                sequence_length,
                max_waveform_samples,
            )?;
            std::sync::Arc::new(std::sync::Mutex::new(full))
        } else {
            std::sync::Arc::clone(&wave_arc)
        };
        return Ok(CachedSeqGraphs {
            full: full_arc,
            duration_refine: dur_arc,
            waveform_only: Some(wave_arc),
            duration_on_cpu: dur_device == Device::Cpu,
        });
    }

    let mut hir = import.hir.clone();
    compile_profile::apply_profile(&mut hir, CompileProfile::Full);
    let full_device = parity_wave_device(device);
    let full_key = if full_device == device {
        key_prefix.to_string()
    } else {
        format!("{key_prefix}_wave{full_device:?}")
    };
    let full = compile_hir_profile(
        full_device,
        &full_key,
        CompileProfile::Full,
        hir,
        &import.params,
        &import.typed,
        carry_bytes,
        sequence_length,
        max_waveform_samples,
    )?;
    Ok(CachedSeqGraphs::full(full))
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

pub(crate) fn import_opts(opts: &GraphOptions) -> ImportOptions {
    crate::bundle_patches::set_import_sequence_length(opts.sequence_length);
    let duration_loop_lowering = if compile_profile::duration_external_fixed_point_enabled() {
        rlx_onnx_import::DurationLoopLowering::RuntimeCarry
    } else {
        rlx_onnx_import::DurationLoopLowering::FusedKernel
    };
    let mel_full = crate::mel_align::import_mel_propagate_enabled(opts.sequence_length);
    let mel_post = crate::mel_align::import_f0_feed_meta_enabled(opts.sequence_length);
    ImportOptions {
        sequence_length: opts.sequence_length,
        max_waveform_samples: opts.max_waveform_samples,
        max_frames_per_token: max_frames_per_token(),
        duration_loop_lowering,
        pre_shape_propagate: mel_full.then_some(crate::mel_align::pre_shape_propagate),
        post_shape_propagate: if mel_full {
            Some(crate::mel_align::post_shape_propagate)
        } else if mel_post {
            Some(crate::mel_align::post_shape_propagate_minimal)
        } else {
            None
        },
        output_shape_fix: Some(crate::bundle_patches::import_output_shape_fix),
        ..ImportOptions::quant_bundle()
    }
}

/// Lower bundle → HIR, applying model-specific node rewrites when the exported
/// graph still wires the `duration` input (idempotent when carry is pre-baked).
pub fn build_hir_from_bundle_with_rewrites(
    bundle: &rlx_onnx_import::RlxBundle,
    mut opts: ImportOptions,
) -> Result<(
    HirModule,
    HashMap<String, Vec<f32>>,
    TypedParams,
    ImportReport,
)> {
    crate::bundle_patches::set_import_sequence_length(opts.sequence_length);
    crate::bundle_patches::set_import_max_waveform_samples(opts.max_waveform_samples);
    let mut nodes = bundle.nodes.clone();
    crate::bundle_patches::patch_bundle_nodes(
        &mut nodes,
        opts.sequence_length,
        opts.max_waveform_samples,
    );
    let defaults = import_opts(&GraphOptions {
        sequence_length: opts.sequence_length,
        max_waveform_samples: opts.max_waveform_samples,
    });
    if opts.pre_shape_propagate.is_none() {
        opts.pre_shape_propagate = defaults.pre_shape_propagate;
    }
    if opts.post_shape_propagate.is_none() {
        opts.post_shape_propagate = defaults.post_shape_propagate;
    }
    if opts.output_shape_fix.is_none() {
        opts.output_shape_fix = defaults.output_shape_fix;
    }
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
    build_hir_from_parts(
        &bundle.manifest,
        nodes,
        params,
        typed_params,
        i64_params,
        &init_shapes,
        opts,
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

/// Break the duration feedback cycle: `/Expand_1` always reads stale carry; `/Where_1`
/// keeps live `duration` in ORT single-pass mode (see `patch_duration_where_input`).
pub fn rewrite_duration_carry(nodes: &mut [BundleNode]) {
    for node in nodes.iter_mut() {
        if node.name != "/Expand_1" {
            continue;
        }
        for inp in node.inputs.iter_mut() {
            if inp == "duration" {
                *inp = crate::opts::DURATION_CARRY.to_string();
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
    compiled.finalize_params();
}

fn compile_prepared_hir_cached(
    key: &str,
    device: Device,
    hir: rlx_ir::hir::HirModule,
    params: &HashMap<String, Vec<f32>>,
    compile_opts: &CompileOptions,
) -> Result<CompiledGraph, rlx_runtime::AotCacheError> {
    let cache = AotCache::new(aot_cache_root());
    if compile_profile::mir_qdq_fusion_enabled() {
        let baked: HashMap<String, Vec<f32>> = params
            .iter()
            .filter(|(k, _)| k.ends_with(crate::qmatmul_bake::BAKED_SUFFIX))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let mut graph = hir
            .lower_to_mir()
            .map_err(rlx_runtime::AotCacheError::from)?
            .into_graph();
        let fused = crate::mir_qdq_fuse::fuse_graph_qmatmul_baked(&mut graph, &baked);
        if compile_profile::env_flag("KITTEN_RLX_TIMING") && fused > 0 {
            eprintln!("[kitten] MIR QDQ fuse: {fused} QMatMul → baked f32 weights");
        }
        return cache.compile_graph_cached(key, device, graph, compile_opts);
    }
    cache.compile_hir_cached(key, device, hir, compile_opts)
}

pub fn prepare_hir_for_compile(
    mut hir: rlx_ir::hir::HirModule,
    params: &HashMap<String, Vec<f32>>,
    typed: &HashMap<String, (Vec<u8>, DType)>,
    sequence_length: usize,
    max_waveform_samples: usize,
) -> (rlx_ir::hir::HirModule, HashMap<String, Vec<f32>>) {
    // Rebind on the calling thread (split-graph workers included) before inject.
    crate::opts::set_compile_sequence_length(sequence_length);
    crate::bundle_patches::set_import_sequence_length(sequence_length);
    crate::bundle_patches::set_import_max_waveform_samples(max_waveform_samples);
    let mut params = params.clone();
    #[cfg(feature = "native")]
    crate::native::flow::finish_bundle_hir_for_compile(
        &mut hir,
        &mut params,
        sequence_length,
        max_waveform_samples,
    );
    #[cfg(not(feature = "native"))]
    {
        let _ = crate::bundle_patches::inject_vocoder_dynamic_alignment(
            &mut hir,
            sequence_length,
            max_waveform_samples,
        );
    }
    if compile_profile::env_flag("KITTEN_RLX_DEBUG_DURATION") {
        eprintln!(
            "[kitten] hir inject wide_seq={} max_wave={} nodes={}",
            sequence_length >= 32,
            max_waveform_samples,
            hir.len()
        );
    }
    let baked = crate::qmatmul_bake::bake_qmatmul_weights(typed, &params);
    if !baked.is_empty() {
        let fused = crate::hir_qdq_fuse::fuse_qmatmul_baked_weights(&mut hir, &baked);
        if compile_profile::env_flag("KITTEN_RLX_TIMING") && fused > 0 {
            eprintln!("[kitten] QDQ fuse: {fused} QMatMul → baked f32 weights");
        }
        for (name, data) in baked {
            params.insert(name, data);
        }
        // Drop raw quant tensors from the compile param map once baked f32 weights exist.
        params.retain(|k, _| !k.ends_with("_quantized"));
    }
    (hir, params)
}

/// LRU-ish cache of graphs compiled at exact sequence lengths (ORT-style `[1, seq]`).
pub struct SeqCompileCache {
    device: Device,
    bundle_dir: PathBuf,
    max_waveform_samples: usize,
    max_sequence_length: usize,
    graphs: SeqGraphCache,
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
            graphs: SeqGraphCache::new(capacity),
        }
    }

    pub fn max_sequence_length(&self) -> usize {
        self.max_sequence_length
    }

    pub fn prewarm(&self, buckets: &[usize]) -> Result<()> {
        self.graphs
            .prewarm(buckets, |seq| self.build_seq_graphs(seq))
    }

    pub fn cached_graphs_for_seq(&self, seq: usize) -> Result<CachedSeqGraphs> {
        if seq > self.max_sequence_length {
            anyhow::bail!(
                "sequence {seq} exceeds compiled max {}",
                self.max_sequence_length
            );
        }
        // Rebind InstanceNorm caps even on in-memory graph hits (AOT cold start after
        // process restart otherwise leaves wave_cap=0 → generator AdaIN full-axis mush).
        let compile_seq = compile_profile::compile_slot_length(seq);
        let max_waveform = compile_profile::compile_waveform_cap(seq, self.max_waveform_samples);
        crate::opts::set_compile_sequence_length(compile_seq);
        crate::bundle_patches::set_import_max_waveform_samples(max_waveform);
        if let Some(hit) = self.graphs.get(seq) {
            return Ok(hit);
        }
        let graphs = self.build_seq_graphs(seq)?;
        self.graphs.insert(seq, graphs.clone());
        Ok(graphs)
    }

    /// Legacy: returns a clone of the full graph (prefer [`Self::cached_graphs_for_seq`]).
    pub fn graph_for_seq(&self, seq: usize) -> Result<CompiledGraph> {
        let graphs = self.cached_graphs_for_seq(seq)?;
        let g = graphs.full.lock().expect("full graph").clone();
        Ok(g)
    }

    fn build_seq_graphs(&self, seq: usize) -> Result<CachedSeqGraphs> {
        // Compile at runtime token width; headroom is optional via env for arena experiments.
        let compile_seq = compile_profile::compile_slot_length(seq);
        // Scale vocoder arena to this bucket's token width; cap at engine max from load.
        let max_waveform = compile_profile::compile_waveform_cap(seq, self.max_waveform_samples);
        let opts = GraphOptions {
            sequence_length: compile_seq,
            max_waveform_samples: max_waveform,
        };
        ensure_compile_arena_policy();
        ensure_kernels_registered();
        let import = import_from_bundle_cached(&self.bundle_dir, &opts)?;
        crate::opts::set_compile_sequence_length(compile_seq);
        let key = format!("{}_seq{seq}", cache_key(self.device, &opts));
        let carry = compile_profile::duration_carry_seed_bytes(compile_seq);
        let graphs = build_cached_graphs_from_import(
            self.device,
            &key,
            &import,
            Some(&carry),
            compile_seq,
            max_waveform,
        )?;
        {
            let mut g = graphs.full.lock().expect("full graph");
            set_runtime_input_ids_shape(&mut g, compile_seq)?;
            set_runtime_active_sequence(&mut g, seq, compile_seq);
        }
        if let Some(d) = &graphs.duration_refine {
            let mut g = d.lock().expect("dur graph");
            set_runtime_input_ids_shape(&mut g, compile_seq)?;
            set_runtime_active_sequence(&mut g, seq, compile_seq);
        }
        if let Some(w) = &graphs.waveform_only {
            let mut g = w.lock().expect("wave graph");
            set_runtime_input_ids_shape(&mut g, compile_seq)?;
            set_runtime_active_sequence(&mut g, seq, compile_seq);
        }
        Ok(graphs)
    }
}

pub fn compile_from_bundle(
    device: Device,
    bundle_dir: &Path,
    opts: &GraphOptions,
) -> Result<CompiledGraph> {
    ensure_compile_arena_policy();
    let import = import_from_bundle_cached(bundle_dir, opts)?;
    let (hir, params) = prepare_hir_for_compile(
        import.hir.clone(),
        &import.params,
        &import.typed,
        opts.sequence_length,
        opts.max_waveform_samples,
    );
    let compile_opts = compile_options_for(device);
    let key = cache_key(device, opts);
    let mut compiled = compile_prepared_hir_cached(&key, device, hir, &params, &compile_opts)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    attach_params(&mut compiled, &params, &import.typed);
    Ok(compiled)
}

/// Compile full waveform graph with F0/N projection replaced by ORT curves.
///
/// Diagnostic partition: if this yields intelligible audio while free-run native does not,
/// the vocoder is fine and the bug is in the native F0/N (prosody) predictor.
pub fn compile_from_bundle_with_ort_f0n(
    device: Device,
    bundle_dir: &Path,
    opts: &GraphOptions,
    f0: &[f32],
    n: &[f32],
) -> Result<CompiledGraph> {
    ensure_compile_arena_policy();
    ensure_kernels_registered();
    let import = import_from_bundle_cached(bundle_dir, opts)?;
    let (mut hir, mut params) = prepare_hir_for_compile(
        import.hir.clone(),
        &import.params,
        &import.typed,
        opts.sequence_length,
        opts.max_waveform_samples,
    );
    let Some((f0_name, f0_len, n_name, n_len)) =
        crate::bundle_patches::replace_f0n_proj_with_params(&mut hir, &mut params, f0, n)
    else {
        anyhow::bail!("replace_f0n_proj_with_params: F0/N proj nodes not found in HIR");
    };
    eprintln!(
        "[ort_f0n] injected {f0_name}[{f0_len}] (live={}) {n_name}[{n_len}] (live={})",
        f0.len(),
        n.len()
    );
    let compile_opts = compile_options_for(device);
    // Unique key so AOT cache never mixes free-run and injected graphs.
    let key = format!("{}_ort_f0n_v1", cache_key(device, opts));
    let mut compiled = compile_prepared_hir_fresh(device, hir, &params, &compile_opts)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let _ = key; // fresh compile; key kept for future disk-cache option
    attach_params(&mut compiled, &params, &import.typed);
    attach_alignment_frame_param(&mut compiled);
    Ok(compiled)
}

/// Compile with ORT `/decoder/Concat` + F0/N injected (upstream bypass for encode *and* NSF).
pub fn compile_from_bundle_with_ort_concat(
    device: Device,
    bundle_dir: &Path,
    opts: &GraphOptions,
    concat: &[f32],
    ort_channels: usize,
    ort_frames: usize,
    f0: &[f32],
    n: &[f32],
) -> Result<CompiledGraph> {
    ensure_compile_arena_policy();
    ensure_kernels_registered();
    let import = import_from_bundle_cached(bundle_dir, opts)?;
    let (mut hir, mut params) = prepare_hir_for_compile(
        import.hir.clone(),
        &import.params,
        &import.typed,
        opts.sequence_length,
        opts.max_waveform_samples,
    );
    let Some((name, len)) = crate::bundle_patches::replace_decoder_concat_with_param(
        &mut hir,
        &mut params,
        concat,
        ort_channels,
        ort_frames,
    ) else {
        anyhow::bail!("replace_decoder_concat_with_param: /decoder/Concat not found");
    };
    // NSF sine still reads F0/N via F0IfSelect (not via Concat) — inject those too.
    let _ = crate::bundle_patches::replace_f0n_proj_with_params(&mut hir, &mut params, f0, n);
    eprintln!(
        "[ort_concat] injected {name}[{len}] (live={} C={ort_channels} T={ort_frames}) + F0/N",
        concat.len()
    );
    let compile_opts = compile_options_for(device);
    let mut compiled = compile_prepared_hir_fresh(device, hir, &params, &compile_opts)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    attach_params(&mut compiled, &params, &import.typed);
    attach_alignment_frame_param(&mut compiled);
    Ok(compiled)
}

/// Compile with ORT `/MatMul_1` injected (ASR-only bypass; native F0/N→conv).
pub fn compile_from_bundle_with_ort_matmul1(
    device: Device,
    bundle_dir: &Path,
    opts: &GraphOptions,
    matmul1: &[f32],
    ort_channels: usize,
    ort_frames: usize,
) -> Result<CompiledGraph> {
    compile_from_bundle_with_ort_matmul1_opts(
        device,
        bundle_dir,
        opts,
        matmul1,
        ort_channels,
        ort_frames,
        None,
        None,
    )
}

/// Like [`compile_from_bundle_with_ort_matmul1`], optionally also injecting ORT F0/N.
pub fn compile_from_bundle_with_ort_matmul1_opts(
    device: Device,
    bundle_dir: &Path,
    opts: &GraphOptions,
    matmul1: &[f32],
    ort_channels: usize,
    ort_frames: usize,
    f0: Option<&[f32]>,
    n: Option<&[f32]>,
) -> Result<CompiledGraph> {
    ensure_compile_arena_policy();
    ensure_kernels_registered();
    let import = import_from_bundle_cached(bundle_dir, opts)?;
    let (mut hir, mut params) = prepare_hir_for_compile(
        import.hir.clone(),
        &import.params,
        &import.typed,
        opts.sequence_length,
        opts.max_waveform_samples,
    );
    let Some((name, len)) = crate::bundle_patches::replace_matmul1_with_param(
        &mut hir,
        &mut params,
        matmul1,
        ort_channels,
        ort_frames,
    ) else {
        anyhow::bail!("replace_matmul1_with_param: /MatMul_1 not found");
    };
    if let (Some(f0), Some(n)) = (f0, n) {
        let _ = crate::bundle_patches::replace_f0n_proj_with_params(&mut hir, &mut params, f0, n);
        eprintln!(
            "[ort_matmul1] injected {name}[{len}] (live={} C={ort_channels} T={ort_frames}) + F0/N",
            matmul1.len()
        );
    } else {
        eprintln!(
            "[ort_matmul1] injected {name}[{len}] (live={} C={ort_channels} T={ort_frames})",
            matmul1.len()
        );
    }
    let compile_opts = compile_options_for(device);
    let mut compiled = compile_prepared_hir_fresh(device, hir, &params, &compile_opts)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    attach_params(&mut compiled, &params, &import.typed);
    attach_alignment_frame_param(&mut compiled);
    Ok(compiled)
}

fn compile_prepared_hir_fresh(
    device: Device,
    hir: rlx_ir::hir::HirModule,
    params: &HashMap<String, Vec<f32>>,
    compile_opts: &CompileOptions,
) -> Result<CompiledGraph, rlx_ir::hir::LowerError> {
    if compile_profile::mir_qdq_fusion_enabled() {
        let baked: HashMap<String, Vec<f32>> = params
            .iter()
            .filter(|(k, _)| k.ends_with(crate::qmatmul_bake::BAKED_SUFFIX))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let mut graph = hir.lower_to_mir()?.into_graph();
        let fused = crate::mir_qdq_fuse::fuse_graph_qmatmul_baked(&mut graph, &baked);
        if compile_profile::env_flag("KITTEN_RLX_TIMING") && fused > 0 {
            eprintln!("[kitten] MIR QDQ fuse: {fused} QMatMul → baked f32 weights");
        }
        return Ok(Session::new(device).compile_with(graph, compile_opts));
    }
    Session::new(device).compile_hir_with(hir, compile_opts)
}

/// Compile without disk cache (tests / benchmarking cold compile).
pub fn compile_from_bundle_fresh(
    device: Device,
    bundle_dir: &Path,
    opts: &GraphOptions,
) -> Result<CompiledGraph> {
    ensure_compile_arena_policy();
    let import = import_from_bundle_cached(bundle_dir, opts)?;
    let (hir, params) = prepare_hir_for_compile(
        import.hir.clone(),
        &import.params,
        &import.typed,
        opts.sequence_length,
        opts.max_waveform_samples,
    );
    let compile_opts = compile_options_for(device);
    let mut compiled = compile_prepared_hir_fresh(device, hir, &params, &compile_opts)
        .map_err(|e| anyhow::anyhow!("{0}", e))?;
    attach_params(&mut compiled, &params, &import.typed);
    Ok(compiled)
}

/// Waveform parity metrics for ONNX vs native (parity diagnostics).
#[derive(Debug, Clone, Copy)]
pub struct ParityOnnxMetrics {
    pub native_samples: usize,
    pub ort_samples: usize,
    pub align_lag: usize,
    pub max_abs_aligned: f32,
}

pub fn parity_onnx_metrics(
    reference: &[f32],
    candidate: &[f32],
    max_lag: usize,
) -> ParityOnnxMetrics {
    let (align_lag, max_abs_aligned) = max_abs_best_lag(reference, candidate, max_lag);
    ParityOnnxMetrics {
        native_samples: candidate.len(),
        ort_samples: reference.len(),
        align_lag,
        max_abs_aligned,
    }
}

fn max_abs_best_lag(reference: &[f32], candidate: &[f32], max_lag: usize) -> (usize, f32) {
    let n = reference.len().min(candidate.len());
    if n == 0 {
        return (0, 0.0);
    }
    let max_lag = max_lag.min(n.saturating_sub(1));
    let mut best_lag = 0usize;
    let mut best = f32::MAX;
    for lag in 0..=max_lag {
        let m = n - lag;
        let mut peak = 0.0f32;
        for i in 0..m {
            peak = peak.max((reference[i] - candidate[i + lag]).abs());
        }
        if peak < best {
            best = peak;
            best_lag = lag;
        }
    }
    (best_lag, best)
}

pub fn log_parity_onnx_metrics(label: &str, m: &ParityOnnxMetrics) {
    eprintln!(
        "[kitten] parity ONNX {label}: ort={} native={} lag={} max_abs_aligned={:.6}",
        m.ort_samples, m.native_samples, m.align_lag, m.max_abs_aligned
    );
}

/// One-shot full-graph infer with `RLX_METAL_THUNK_PROFILE=1` (parity diagnostics).
pub fn run_parity_thunk_profile(
    graphs: &CachedSeqGraphs,
    token_len: usize,
    compile_seq: usize,
) -> Result<()> {
    if !compile_profile::parity_thunk_profile_enabled() {
        return Ok(());
    }
    crate::set_env_var("RLX_METAL_THUNK_PROFILE", "1");
    let ids = vec![0i64; compile_seq];
    let ids_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let style = vec![0u8; 256 * 4];
    let speed = 1.0f32.to_le_bytes().to_vec();
    shape_all_graphs_for_infer(graphs, token_len, compile_seq)?;
    let inputs: [(&str, &[u8], DType); 3] = [
        ("input_ids", ids_bytes.as_slice(), DType::I64),
        ("style", style.as_slice(), DType::F32),
        ("speed", speed.as_slice(), DType::F32),
    ];
    let outs = run_kitten_inference(graphs, &inputs, None, None);
    if let Some((wave, dt)) = outs.first() {
        if *dt == DType::F32 {
            let n = wave.len() / 4;
            let peak = wave
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).abs())
                .fold(0.0f32, f32::max);
            eprintln!("[kitten] parity profile waveform: {n} samples peak={peak:.4}");
        }
    }
    Ok(())
}
