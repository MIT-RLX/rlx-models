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

//! Compile profiles for fusion-safe single-output graphs and duration refinement.

use rlx_ir::hir::{HirModule, HirMut};
use rlx_runtime::CompileOptions;

/// Production vs parity infer (see `KITTEN_RLX_INFER` and `native-fast` feature).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferMode {
    /// Split fused graphs, in-place infer, minimal prewarm (latency).
    Production,
    /// Full-graph fallback, dual outputs, prewarm buckets (tests / parity).
    Parity,
}

/// Resolve infer mode: `KITTEN_RLX_INFER=production|parity`, or `native-fast` default.
pub fn infer_mode() -> InferMode {
    match std::env::var("KITTEN_RLX_INFER")
        .ok()
        .as_deref()
        .map(str::trim)
        .map(|s| s.to_ascii_lowercase())
    {
        Some(s) if matches!(s.as_str(), "production" | "prod" | "fast") => InferMode::Production,
        Some(s) if matches!(s.as_str(), "parity" | "debug" | "test") => InferMode::Parity,
        _ => {
            #[cfg(feature = "native-fast")]
            {
                InferMode::Production
            }
            #[cfg(not(feature = "native-fast"))]
            {
                InferMode::Parity
            }
        }
    }
}

/// Which graph outputs to retain after lowering (enables fusion when a single output).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompileProfile {
    /// Waveform + duration (parity / legacy).
    Full,
    /// Duration tensor only — vocoder stripped by DCE (fixed-point refinement passes).
    DurationRefinement,
    /// Waveform only — final pass after carry converges.
    WaveformOnly,
}

pub fn optimized_split_graphs_enabled() -> bool {
    if env_flag("KITTEN_RLX_SPLIT_GRAPHS") {
        return !env_flag("KITTEN_RLX_FULL_GRAPH");
    }
    if env_flag("KITTEN_RLX_FULL_GRAPH") {
        return false;
    }
    // Production: compile duration-refine + waveform-only slices instead of one dual-output
    // Full monolith (parity-sized arena, tens of GB at large max_wave).
    infer_mode() == InferMode::Production
}

/// When split graphs are built, also compile the dual-output full graph (parity fallback).
pub fn compile_split_full_fallback() -> bool {
    infer_mode() == InferMode::Parity || env_flag("KITTEN_RLX_COMPILE_FULL_FALLBACK")
}

pub fn compile_duration_refine_graph() -> bool {
    optimized_split_graphs_enabled() && !production_waveform_only_infer()
}

/// Skip the duration-refine graph and run waveform-only (opt-in; default uses native two-pass).
pub fn production_waveform_only_infer() -> bool {
    if env_flag("KITTEN_RLX_DURATION_REFINE") {
        return false;
    }
    if env_flag("KITTEN_RLX_WAVEFORM_ONLY_INFER") {
        return true;
    }
    if env_flag("KITTEN_RLX_NATIVE_DURATION") {
        return false;
    }
    // WaveformOnly compile DCE removes the in-graph duration epilogue the vocoder needs;
    // production defaults to duration-refine + waveform (low RAM vs full monolith).
    false
}

/// On Apple Silicon, prefer Metal over CPU for production unless opted out.
pub fn prefer_metal_device(device: rlx_runtime::Device) -> rlx_runtime::Device {
    if device != rlx_runtime::Device::Cpu {
        return device;
    }
    if infer_mode() != InferMode::Production {
        return device;
    }
    if std::env::var("KITTEN_RLX_PREFER_METAL")
        .is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("no"))
    {
        return device;
    }
    if rlx_runtime::is_available(rlx_runtime::Device::Metal) {
        return rlx_runtime::Device::Metal;
    }
    device
}

/// ORT single-pass duration loop (Expand stale, Where live) vs external fixed-point.
pub fn duration_loop_single_pass() -> bool {
    !duration_external_fixed_point_enabled()
}

/// External duration carry iteration (parity / debug). Production defaults to single-pass ORT semantics.
pub fn duration_external_fixed_point_enabled() -> bool {
    if env_flag("KITTEN_RLX_DURATION_FIXED_POINT") {
        return true;
    }
    if env_flag("KITTEN_RLX_SINGLE_PASS") {
        return false;
    }
    infer_mode() == InferMode::Parity
}

pub fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
}

/// Fold baked QMatMul weights at compile (production + parity default).
pub fn qdq_fusion_enabled() -> bool {
    if env_flag("KITTEN_RLX_NO_QDQ_FUSION") {
        return false;
    }
    if env_flag("KITTEN_RLX_QDQ_FUSION") {
        return true;
    }
    matches!(infer_mode(), InferMode::Production | InferMode::Parity)
}

/// Run one Metal thunk profile on the full graph after load (parity).
pub fn parity_thunk_profile_enabled() -> bool {
    env_flag("KITTEN_RLX_PARITY_PROFILE") && infer_mode() == InferMode::Parity
}

/// Log native vs ONNX waveform parity after load (parity mode).
pub fn parity_onnx_profile_enabled() -> bool {
    env_flag("KITTEN_RLX_PARITY_ONNX") && infer_mode() == InferMode::Parity
}

/// Lowered-graph QDQ fusion (safety net after HIR fuse; default on with [`qdq_fusion_enabled`]).
pub fn mir_qdq_fusion_enabled() -> bool {
    if env_flag("KITTEN_RLX_NO_MIR_QDQ_FUSION") {
        return false;
    }
    if env_flag("KITTEN_RLX_MIR_QDQ_FUSION") {
        return true;
    }
    // Production: HIR fuse only. MIR re-lowers the full graph (duplicate IR + peak RSS in GB).
    if infer_mode() == InferMode::Production {
        return false;
    }
    qdq_fusion_enabled()
}

pub fn prewarm_enabled() -> bool {
    if env_flag("KITTEN_RLX_SKIP_PREWARM") {
        return false;
    }
    if env_flag("KITTEN_RLX_PREWARM") {
        return true;
    }
    // Production: lazy compile on first infer (avoids multi-GB arenas at load).
    infer_mode() == InferMode::Parity
}

/// Vocoder samples per duration unit (matches ONNX Kitten mini 0.8).
pub const COMPILE_SAMPLES_PER_DURATION_UNIT: usize = 600;

const TYPICAL_MAX_DURATION_UNITS_PER_TOKEN: usize = 8;
const COMPILE_WAVEFORM_HEADROOM: usize = 12_000;
const COMPILE_WAVEFORM_FLOOR: usize = 24_000;

/// Re-export discrete wave cap (see [`crate::device_policy`]).
pub use crate::device_policy::{
    WGPU_WAVEFORM_CAP as WGPU_ACT_SAFE_WAVEFORM_CAP, clamp_waveform_for_device,
};

/// Per-bucket waveform compile cap from runtime token width, bounded by [`engine_cap`].
pub fn compile_waveform_cap(runtime_tokens: usize, engine_cap: usize) -> usize {
    let est = runtime_tokens
        .saturating_mul(COMPILE_SAMPLES_PER_DURATION_UNIT)
        .saturating_mul(TYPICAL_MAX_DURATION_UNITS_PER_TOKEN)
        .saturating_add(COMPILE_WAVEFORM_HEADROOM)
        .max(COMPILE_WAVEFORM_FLOOR);
    est.min(engine_cap.max(COMPILE_WAVEFORM_FLOOR))
}

/// Max compiled seq graphs kept resident (LRU); production keeps one bucket.
pub fn seq_compile_cache_capacity() -> usize {
    if let Ok(raw) = std::env::var("KITTEN_RLX_SEQ_CACHE_CAPACITY") {
        if let Ok(n) = raw.parse::<usize>() {
            return n.max(1);
        }
    }
    match infer_mode() {
        InferMode::Production => 1,
        InferMode::Parity => 4,
    }
}

/// Effective alignment frames/token — see [`crate::device_policy::max_frames_per_token`].
#[inline]
pub fn max_frames_per_token() -> usize {
    crate::device_policy::max_frames_per_token()
}

/// Device-specific defaults before compile cache construction.
pub fn apply_device_runtime_defaults(device: rlx_runtime::Device) {
    // Metal's dual-output parity/full graph still zeros the wave after the
    // f32-uniform duration-carry fix; split production graphs are the
    // validated on-device path. Prefer production when the user did not
    // pin `KITTEN_RLX_INFER` explicitly.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if device == rlx_runtime::Device::Metal && std::env::var("KITTEN_RLX_INFER").is_err() {
        crate::set_env_var("KITTEN_RLX_INFER", "production");
    }
    if infer_mode() == InferMode::Production && std::env::var("KITTEN_RLX_SINGLE_PASS").is_err() {
        crate::set_env_var("KITTEN_RLX_SINGLE_PASS", "1");
    }
    crate::device_policy::apply_defaults(device);
    // Do not set KITTEN_RLX_FULL_GRAPH here — that forces the largest dual-output compile.
    if infer_mode() == InferMode::Production || infer_mode() == InferMode::Parity {
        if std::env::var("KITTEN_RLX_QMATMUL_PARALLEL").is_err() {
            crate::set_env_var("KITTEN_RLX_QMATMUL_PARALLEL", "1");
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if device == rlx_runtime::Device::Metal {
            if std::env::var("KITTEN_RLX_QMATMUL_INGRAPH").is_err()
                && std::env::var("RLX_METAL_ONNX_QMATMUL_GPU").is_err()
            {
                crate::set_env_var("KITTEN_RLX_QMATMUL_INGRAPH", "1");
            }
            if std::env::var("RLX_METAL_PIPELINE_CACHE").is_err() {
                if let Ok(aot) = std::env::var("KITTEN_RLX_AOT_CACHE") {
                    crate::set_env_var(
                        "RLX_METAL_PIPELINE_CACHE",
                        format!("{aot}/metal_pipelines"),
                    );
                }
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                let _ = rlx_metal::kernels::prewarm();
            }
        }
        return;
    }
    if device != rlx_runtime::Device::Mlx {
        return;
    }
    if std::env::var("KITTEN_RLX_FULL_GRAPH").is_err() {
        crate::set_env_var("KITTEN_RLX_FULL_GRAPH", "1");
    }
    if std::env::var("KITTEN_RLX_SKIP_PREWARM").is_err() {
        crate::set_env_var("KITTEN_RLX_SKIP_PREWARM", "1");
    }
}

#[deprecated(note = "use apply_device_runtime_defaults")]
pub fn apply_mlx_runtime_defaults(device: rlx_runtime::Device) {
    apply_device_runtime_defaults(device);
}

pub fn prewarm_buckets(max_seq: usize) -> Vec<usize> {
    if let Ok(raw) = std::env::var("KITTEN_RLX_PREWARM_BUCKETS") {
        let parsed: Vec<usize> = raw
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .filter(|&n| n > 0 && n <= max_seq)
            .collect();
        if !parsed.is_empty() {
            return parsed;
        }
    }
    if infer_mode() == InferMode::Production {
        return vec![max_seq.max(1)];
    }
    let mut buckets: Vec<usize> = [8, 16, 32, 64, 128]
        .into_iter()
        .filter(|&n| n <= max_seq)
        .collect();
    if buckets.is_empty() {
        buckets.push(max_seq.max(1));
    }
    buckets
}

pub fn arena_no_reuse_for_kitten() -> bool {
    !arena_reuse_allowed_for_kitten()
}

pub fn arena_reuse_allowed_for_kitten() -> bool {
    if env_flag("KITTEN_RLX_ARENA_NO_REUSE") {
        return false;
    }
    if env_flag("KITTEN_RLX_ARENA_ALLOW_REUSE") {
        return true;
    }
    infer_mode() == InferMode::Production
}

/// Graph compile slot count.
///
/// Default is the **exact** runtime token width. Adding duration headroom without a pad
/// attention mask lets BERT attend to zero pads and inflates duration token-0 (~19 vs ORT ~3).
/// Opt into headroom with `KITTEN_RLX_COMPILE_HEADROOM=1` for arena experiments.
pub fn compile_slot_length(token_len: usize) -> usize {
    let n = token_len.max(1);
    if env_flag("KITTEN_RLX_COMPILE_HEADROOM") && !env_flag("KITTEN_RLX_COMPILE_EXACT") {
        return crate::bundle_compile::compile_sequence_length(n);
    }
    n
}

/// Trim HIR outputs for a compile profile (in-place).
pub fn apply_profile(hir: &mut HirModule, profile: CompileProfile) {
    match profile {
        CompileProfile::Full => {}
        CompileProfile::DurationRefinement => {
            if hir.outputs.len() >= 2 {
                let dur = hir.outputs[1];
                let mut m = HirMut::new(hir);
                m.set_outputs(vec![dur]);
            }
        }
        CompileProfile::WaveformOnly => {
            if !hir.outputs.is_empty() {
                let wave = hir.outputs[0];
                let mut m = HirMut::new(hir);
                m.set_outputs(vec![wave]);
            }
        }
    }
}

pub fn compile_options_for_profile(
    device: rlx_runtime::Device,
    profile: CompileProfile,
) -> CompileOptions {
    let mut opts = crate::bundle_compile::compile_options_base(device);
    match profile {
        CompileProfile::Full => {
            opts.fusion_opts.skip_fusion = true;
        }
        CompileProfile::DurationRefinement | CompileProfile::WaveformOnly => {
            opts.fusion_opts.skip_fusion = crate::bundle_compile::skip_fusion_from_env();
        }
    }
    opts
}

pub fn duration_carry_seed_bytes(sequence_length: usize) -> Vec<u8> {
    vec![0i64; sequence_length]
        .into_iter()
        .flat_map(|d| d.to_le_bytes())
        .collect()
}

pub fn aot_cache_suffix(profile: CompileProfile) -> &'static str {
    match profile {
        CompileProfile::Full => "full",
        CompileProfile::DurationRefinement => "dur",
        CompileProfile::WaveformOnly => "wave",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_waveform_cap_scales_with_tokens() {
        let hello = compile_waveform_cap(8, usize::MAX);
        assert_eq!(
            hello,
            8 * COMPILE_SAMPLES_PER_DURATION_UNIT * 8 + COMPILE_WAVEFORM_HEADROOM
        );
        let chunk = compile_waveform_cap(25, usize::MAX);
        let long = compile_waveform_cap(74, usize::MAX);
        assert!(chunk < long);
        assert!(compile_waveform_cap(25, hello) <= hello);
    }
}
