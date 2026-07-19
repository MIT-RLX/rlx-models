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

//! Compile native Kitten graphs from `model.safetensors` / `model.gguf` (no bundle).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Once;

use anyhow::{Context, Result};
use rlx_ir::hir::HirModule;
use rlx_runtime::{AotCache, CompiledGraph, DType, Device, Session};

use crate::bundle_compile::{
    compile_options_for, ensure_compile_arena_policy, ensure_kernels_registered,
    set_runtime_active_sequence, set_runtime_input_ids_shape,
};
use crate::compile_profile::{self, CompileProfile, optimized_split_graphs_enabled};
use crate::kernels::register_native_kernels;
use crate::native::flow::build_native_hir;
use crate::opts::{DURATION_CARRY, GraphOptions};
use crate::seq_cache::{CachedSeqGraphs, SeqGraphCache};
use crate::weights::{load_weights, native_weights_available};

static KERNELS: Once = Once::new();

fn ensure_kernels() {
    KERNELS.call_once(register_native_kernels);
}

fn aot_cache_root() -> PathBuf {
    if let Ok(p) = std::env::var("KITTEN_RLX_AOT_CACHE") {
        return PathBuf::from(p);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("rlx/kitten_tts_native_aot")
}

fn cache_key(device: Device, weights_dir: &Path, opts: &GraphOptions, seq: usize) -> String {
    let canonical = weights_dir
        .canonicalize()
        .unwrap_or_else(|_| weights_dir.to_path_buf());
    format!(
        "kitten_native_{:?}_rt{seq}_s{}_w{}_{}",
        device,
        opts.sequence_length,
        opts.max_waveform_samples,
        canonical.display()
    )
}

fn attach_params(compiled: &mut CompiledGraph, params: &HashMap<String, Vec<f32>>) {
    for (name, data) in params {
        compiled.set_param(name.as_str(), data);
    }
    compiled.finalize_params();
}

fn attach_duration_carry(compiled: &mut CompiledGraph, carry_bytes: &[u8]) {
    compiled.set_param_typed(DURATION_CARRY, carry_bytes, DType::I64);
}

fn compile_hir_profile_native(
    device: Device,
    key: &str,
    profile: CompileProfile,
    mut hir: HirModule,
    params: &HashMap<String, Vec<f32>>,
    carry_bytes: &[u8],
) -> Result<CompiledGraph> {
    compile_profile::apply_profile(&mut hir, profile);
    let compile_opts = compile_profile::compile_options_for_profile(device, profile);
    let cache = AotCache::new(aot_cache_root());
    let full_key = format!("{key}_{}", compile_profile::aot_cache_suffix(profile));
    let mut compiled = cache
        .compile_hir_cached(&full_key, device, hir, &compile_opts)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    attach_params(&mut compiled, params);
    attach_duration_carry(&mut compiled, carry_bytes);
    crate::bundle_compile::attach_alignment_frame_param(&mut compiled);
    Ok(compiled)
}

fn build_native_cached_graphs(
    device: Device,
    weights_dir: &Path,
    opts: &GraphOptions,
    runtime_seq: usize,
) -> Result<CachedSeqGraphs> {
    ensure_compile_arena_policy();
    ensure_kernels();
    ensure_kernels_registered();
    crate::opts::set_compile_sequence_length(opts.sequence_length);
    let weights = load_weights(weights_dir).context("load native weights")?;
    let (base_hir, params, carry_bytes) = build_native_hir(&weights, opts)?;
    let key = cache_key(device, weights_dir, opts, runtime_seq);

    if optimized_split_graphs_enabled() {
        let compile_dur = compile_profile::compile_duration_refine_graph();
        let (dur, wave) = if compile_dur {
            if compile_profile::env_flag("KITTEN_RLX_SERIAL_COMPILE") {
                let dur = compile_hir_profile_native(
                    device,
                    &key,
                    CompileProfile::DurationRefinement,
                    base_hir.clone(),
                    &params,
                    &carry_bytes,
                )?;
                let wave = compile_hir_profile_native(
                    device,
                    &key,
                    CompileProfile::WaveformOnly,
                    base_hir.clone(),
                    &params,
                    &carry_bytes,
                )?;
                (Some(dur), wave)
            } else {
                let key_d = key.clone();
                let key_w = key.clone();
                let hir_d = base_hir.clone();
                let hir_w = base_hir.clone();
                let params_d = params.clone();
                let params_w = params.clone();
                let carry_d = carry_bytes.clone();
                let carry_w = carry_bytes.clone();
                std::thread::scope(|scope| -> Result<(Option<CompiledGraph>, CompiledGraph)> {
                    let dur_h = scope.spawn(move || {
                        compile_hir_profile_native(
                            device,
                            &key_d,
                            CompileProfile::DurationRefinement,
                            hir_d,
                            &params_d,
                            &carry_d,
                        )
                    });
                    let wave_h = scope.spawn(move || {
                        compile_hir_profile_native(
                            device,
                            &key_w,
                            CompileProfile::WaveformOnly,
                            hir_w,
                            &params_w,
                            &carry_w,
                        )
                    });
                    let dur = dur_h.join().expect("duration compile thread")?;
                    let wave = wave_h.join().expect("waveform compile thread")?;
                    Ok((Some(dur), wave))
                })?
            }
        } else {
            let wave = compile_hir_profile_native(
                device,
                &key,
                CompileProfile::WaveformOnly,
                base_hir.clone(),
                &params,
                &carry_bytes,
            )?;
            (None, wave)
        };
        let wave_arc = std::sync::Arc::new(std::sync::Mutex::new(wave));
        let dur_arc = dur.map(|g| std::sync::Arc::new(std::sync::Mutex::new(g)));
        let full_arc = if compile_profile::compile_split_full_fallback() {
            let full = compile_hir_profile_native(
                device,
                &key,
                CompileProfile::Full,
                base_hir,
                &params,
                &carry_bytes,
            )?;
            std::sync::Arc::new(std::sync::Mutex::new(full))
        } else {
            std::sync::Arc::clone(&wave_arc)
        };
        return Ok(CachedSeqGraphs {
            full: full_arc,
            duration_refine: dur_arc,
            waveform_only: Some(wave_arc),
        });
    }

    let full = compile_hir_profile_native(
        device,
        &key,
        CompileProfile::Full,
        base_hir,
        &params,
        &carry_bytes,
    )?;
    Ok(CachedSeqGraphs::full(full))
}

/// Compile the native Rust graph from weights on disk (full dual-output graph).
pub fn compile_native(
    device: Device,
    weights_dir: &Path,
    opts: &GraphOptions,
) -> Result<CompiledGraph> {
    let graphs = build_native_cached_graphs(device, weights_dir, opts, opts.sequence_length)?;
    Ok(graphs.full.lock().expect("full graph").clone())
}

/// Compile without disk cache (tests / cold-compile benchmarks).
pub fn compile_native_fresh(
    device: Device,
    weights_dir: &Path,
    opts: &GraphOptions,
) -> Result<CompiledGraph> {
    ensure_compile_arena_policy();
    ensure_kernels();
    ensure_kernels_registered();
    crate::opts::set_compile_sequence_length(opts.sequence_length);
    let weights = load_weights(weights_dir).context("load native weights")?;
    let (hir, params, carry_bytes) = build_native_hir(&weights, opts)?;
    let compile_opts = compile_options_for(device);
    let mut compiled = Session::new(device)
        .compile_hir_with(hir, &compile_opts)
        .map_err(|e| anyhow::anyhow!("{0}", e))?;
    attach_params(&mut compiled, &params);
    attach_duration_carry(&mut compiled, &carry_bytes);
    Ok(compiled)
}

/// LRU-ish cache of native graphs compiled at exact token sequence lengths.
pub struct NativeSeqCompileCache {
    device: Device,
    weights_dir: PathBuf,
    max_waveform_samples: usize,
    max_sequence_length: usize,
    graphs: SeqGraphCache,
}

impl NativeSeqCompileCache {
    pub fn new(
        device: Device,
        weights_dir: PathBuf,
        max_sequence_length: usize,
        max_waveform_samples: usize,
        capacity: usize,
    ) -> Result<Self> {
        if !native_weights_available(&weights_dir) {
            anyhow::bail!(
                "native weights not found under {} (expected model.safetensors or model.gguf)",
                weights_dir.display()
            );
        }
        Ok(Self {
            device,
            weights_dir,
            max_waveform_samples,
            max_sequence_length,
            graphs: SeqGraphCache::new(capacity),
        })
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
        if let Some(hit) = self.graphs.get(seq) {
            return Ok(hit);
        }
        let graphs = self.build_seq_graphs(seq)?;
        self.graphs.insert(seq, graphs.clone());
        Ok(graphs)
    }

    /// Legacy: clone of the full graph (prefer [`Self::cached_graphs_for_seq`]).
    pub fn graph_for_seq(&self, seq: usize) -> Result<CompiledGraph> {
        let graphs = self.cached_graphs_for_seq(seq)?;
        Ok(graphs.full.lock().expect("full graph").clone())
    }

    fn build_seq_graphs(&self, seq: usize) -> Result<CachedSeqGraphs> {
        let compile_seq = compile_profile::compile_slot_length(seq);
        let max_waveform = compile_profile::compile_waveform_cap(seq, self.max_waveform_samples);
        let opts = GraphOptions {
            sequence_length: compile_seq,
            max_waveform_samples: max_waveform,
        };
        let graphs = build_native_cached_graphs(self.device, &self.weights_dir, &opts, seq)?;
        let mut g = graphs.full.lock().expect("full graph");
        set_runtime_input_ids_shape(&mut g, compile_seq)?;
        set_runtime_active_sequence(&mut g, seq, compile_seq);
        drop(g);
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

pub use crate::bundle_compile::{run_kitten_inference, run_with_duration_fixed_point};
