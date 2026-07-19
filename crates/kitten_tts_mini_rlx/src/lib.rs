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

//! Native RLX model for Kitten TTS mini (Rust graph + external weights).

#![allow(clippy::type_complexity, clippy::too_many_arguments)]

pub mod alignment;
pub mod bundle_compile;
pub mod bundle_patches;
pub mod compile_profile;
pub mod gpu_kernels;
pub mod hir_qdq_fuse;
pub mod kernels;
pub mod lstm;
pub mod mel_align;
pub mod mir_qdq_fuse;
pub mod opts;
pub mod probe_watch;
pub mod qmatmul;
pub mod qmatmul_bake;
pub mod qmatmul_gpu;
pub mod random;
pub mod scatter;
pub mod seq_cache;
pub mod weights;

// The native path is data-driven: `bundle_compile` imports `rlx_bundle/graph.json`
// via rlx-onnx-import (the path `compile()` prefers). The old ~100k-line transpiled
// `graph.rs` HIR builder was removed — it was legacy, unreliable for some sequence
// lengths, and made native builds enormous/slow. (Recoverable from git history.)
#[cfg(feature = "native")]
pub mod native;

pub use bundle_compile::{ParityOnnxMetrics, log_parity_onnx_metrics, parity_onnx_metrics};
pub use bundle_compile::{build_cached_graphs_from_import, run_kitten_inference};
pub use compile_profile::{
    CompileProfile, InferMode, apply_device_runtime_defaults, compile_split_full_fallback,
    compile_waveform_cap, duration_external_fixed_point_enabled, infer_mode,
    mir_qdq_fusion_enabled, optimized_split_graphs_enabled, parity_onnx_profile_enabled,
    parity_thunk_profile_enabled, prewarm_buckets, prewarm_enabled, qdq_fusion_enabled,
    seq_compile_cache_capacity,
};
pub use mel_align::{
    ALIGNMENT_MASK_NODES, F0_FEED_PROBE_NODES, compile_mel_cap, import_mel_propagate_enabled,
};
pub use opts::GraphOptions;
pub use seq_cache::{CachedSeqGraphs, SeqGraphCache};
pub use weights::{load_weights, native_weights_available, resolve_weights_file};

/// Set a process environment variable (Rust 2024 `set_var` wrapper).
pub fn set_env_var<K: AsRef<std::ffi::OsStr>, V: AsRef<std::ffi::OsStr>>(key: K, value: V) {
    // SAFETY: compile paths and tests set env before graph build / worker threads.
    unsafe { std::env::set_var(key, value) }
}

#[cfg(feature = "native")]
pub use native::{NativeSeqCompileCache, build_native_hir, compile_native, compile_native_fresh};

fn force_bundle_from_env() -> bool {
    std::env::var("KITTEN_RLX_FORCE_BUNDLE")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
}

/// Prefer `rlx_bundle` when both bundle and `model.safetensors` exist (graph.rs
/// compile is not yet reliable for all sequence lengths).
fn prefer_bundle_import(weights_path: &std::path::Path) -> bool {
    if force_bundle_from_env() {
        return true;
    }
    if std::env::var("KITTEN_RLX_FORCE_WEIGHTS")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
    {
        return false;
    }
    let has_weights = native_weights_available(weights_path)
        || weights_path.parent().is_some_and(native_weights_available);
    let has_bundle = bundle_compile::bundle_dir_near_weights(weights_path).is_some();
    has_weights && has_bundle
}

/// Compile the Kitten graph on `device`.
///
/// Prefers `rlx_bundle/graph.json` when both bundle and native weights exist.
/// Use `KITTEN_RLX_FORCE_WEIGHTS=1` for the Rust `graph.rs` path, or
/// `KITTEN_RLX_FORCE_BUNDLE=1` to require the bundle.
pub fn compile(
    device: rlx_runtime::Device,
    weights_path: &std::path::Path,
    opts: &GraphOptions,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {
    if prefer_bundle_import(weights_path) {
        if let Some(bundle) = bundle_compile::bundle_dir_near_weights(weights_path) {
            return bundle_compile::compile_from_bundle(device, &bundle, opts);
        }
    }
    #[cfg(feature = "native")]
    if !force_bundle_from_env() {
        if let Some(dir) = native::native_weights_dir_near(weights_path) {
            return native::compile_native(device, &dir, opts);
        }
    }
    if let Some(bundle) = bundle_compile::bundle_dir_near_weights(weights_path) {
        return bundle_compile::compile_from_bundle(device, &bundle, opts);
    }
    #[cfg(feature = "native")]
    {
        anyhow::bail!(
            "no native weights or rlx_bundle found under {}",
            weights_path.display()
        );
    }
    #[cfg(not(feature = "native"))]
    anyhow::bail!(
        "no rlx_bundle found under {} (enable `native` feature for safetensors path)",
        weights_path.display()
    )
}

pub const WEIGHTS_FORMAT: &str = "safetensors";
