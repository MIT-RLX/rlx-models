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

//! Runtime toggles for synthesis latency vs compile-ahead tradeoffs.

use rlx_runtime::Device;

/// Upper bound on codec frames for one utterance. Synthesis **stops at talker EOS** before this.
///
/// - `user_max_frames == 0` (default): auto budget from prompt length + `generation_config` ceiling.
/// - `user_max_frames > 0`: explicit hard cap (`--max-frames`, benches).
pub fn codec_frame_budget(text: &str, gen_cfg_max: usize, user_max_frames: usize) -> usize {
    const ABS_MAX: usize = 2048;
    let words = text.split_whitespace().count().max(1);
    let chars = text.chars().count().max(1);
    // 12 Hz codec ≈ 12 frames/s; English ~2–3 words/s → ~4–6 frames/word with headroom.
    let auto = words.saturating_mul(6).max(chars / 2).max(48);
    let ceiling = gen_cfg_max.max(auto).min(ABS_MAX);
    if user_max_frames > 0 {
        user_max_frames.min(ceiling)
    } else {
        ceiling
    }
}

/// Defer talker decode-bucket compile until synthesis frames (fast one-shot CLI).
/// Set `RLX_QWEN3_TTS_PRECOMPILE_BUCKETS=1` to precompile all buckets through the utterance horizon at warmup.
pub fn lazy_talk_buckets() -> bool {
    if std::env::var("RLX_QWEN3_TTS_PRECOMPILE_BUCKETS")
        .ok()
        .as_deref()
        == Some("1")
    {
        return false;
    }
    std::env::var("RLX_QWEN3_TTS_LAZY_BUCKETS").ok().as_deref() != Some("0")
        || std::env::var("RLX_QWEN3_TTS_SKIP_BUCKET_WARMUP")
            .ok()
            .as_deref()
            == Some("1")
}

pub fn skip_talk_bucket_warmup() -> bool {
    lazy_talk_buckets()
}

/// Precompile talker decode buckets through `prefill + max_frames` for short utterances.
pub fn auto_precompile_horizon(max_frames: usize) -> bool {
    if lazy_talk_buckets() {
        return max_frames <= 64;
    }
    // `PRECOMPILE_BUCKETS=1`: compile-ahead for short clips only unless `WARMUP_ALL_BUCKETS=1`.
    max_frames <= 64 || warmup_all_talk_buckets()
}

/// Dry-run talker decode at bucket boundaries during warmup (slow on Metal). Default: horizon ≤ 64.
pub fn talk_bucket_execution_warmup(horizon: usize) -> bool {
    if !megakernel_fast_path() {
        return false;
    }
    warmup_all_talk_buckets() || horizon <= 64
}

pub fn synth_timing_enabled() -> bool {
    std::env::var("RLX_QWEN3_TTS_TIMING").ok().as_deref() == Some("1")
}

/// Per-frame talker/CP/decode ms (`RLX_QWEN3_TTS_STEP_TIMING=1`).
pub fn step_timing_enabled() -> bool {
    std::env::var("RLX_QWEN3_TTS_STEP_TIMING").ok().as_deref() == Some("1")
}

/// Megakernel synthesis path (default on). Set `RLX_QWEN3_TTS_MEGAKERNEL=0` to disable GPU KV seeding / pipeline warmup.
pub fn megakernel_fast_path() -> bool {
    std::env::var("RLX_QWEN3_TTS_MEGAKERNEL").ok().as_deref() != Some("0")
}

/// Prefer RLX fused graphs end-to-end when parity allows (`RLX_QWEN3_TTS_FUSED_E2E=0` to opt out).
pub fn fused_e2e_target() -> bool {
    std::env::var("RLX_QWEN3_TTS_FUSED_E2E").ok().as_deref() != Some("0")
}

/// Host CP greedy + compiled talker decode per frame. Default on GPU sessions.
pub fn codec_frame_fused_enabled(device: Device) -> bool {
    match std::env::var("RLX_QWEN3_TTS_CODEC_FRAME_FUSED")
        .ok()
        .as_deref()
    {
        Some("0") => false,
        Some("1") => true,
        _ => crate::gpu_pipeline::gpu_session_enabled(device),
    }
}

/// Warm/run full CP+talker megagraph for parity or bench (`RLX_QWEN3_TTS_CODEC_FRAME_MEGAGRAPH=1`).
pub fn codec_frame_megagraph_enabled() -> bool {
    std::env::var("RLX_QWEN3_TTS_CODEC_FRAME_MEGAGRAPH")
        .ok()
        .as_deref()
        == Some("1")
}

/// Dry-run every decode bucket on the synthesis path (slow warmup). Default: boundary crossings only.
pub fn warmup_all_talk_buckets() -> bool {
    std::env::var("RLX_QWEN3_TTS_WARMUP_ALL_BUCKETS")
        .ok()
        .as_deref()
        == Some("1")
}

/// GPU-resident talker K/V during megakernel synthesis (Metal/CUDA/ROCm). Opt out: `RLX_QWEN3_TTS_GPU_KV=0`.
pub fn megakernel_gpu_kv_default(device: Device) -> bool {
    if !megakernel_fast_path() {
        return false;
    }
    matches!(device, Device::Metal | Device::Cuda | Device::Rocm)
}

/// Scale quiet PCM to ~0.95 peak before writing WAV. Opt in: `RLX_QWEN3_TTS_WAV_NORMALIZE=1`.
pub fn wav_peak_normalize_enabled() -> bool {
    std::env::var("RLX_QWEN3_TTS_WAV_NORMALIZE").ok().as_deref() == Some("1")
}

/// Best available accelerator for Qwen3-TTS (Metal → MLX → CUDA → ROCm → CPU).
pub fn fastest_device() -> Device {
    if rlx_runtime::is_available(Device::Metal) {
        Device::Metal
    } else if rlx_runtime::is_available(Device::Mlx) {
        Device::Mlx
    } else if rlx_runtime::is_available(Device::Cuda) {
        Device::Cuda
    } else if rlx_runtime::is_available(Device::Rocm) {
        Device::Rocm
    } else {
        Device::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::codec_frame_budget;

    #[test]
    fn auto_budget_scales_with_prompt() {
        assert!(codec_frame_budget("Hi.", 128, 0) >= 48);
        assert_eq!(codec_frame_budget("Hi.", 128, 96), 96);
        let long = "Hello from RLX. The quick brown fox jumps over the lazy dog. We are testing whether speech synthesis produces clear intelligible audio.";
        assert!(codec_frame_budget(long, 128, 0) > codec_frame_budget("Hi.", 128, 0));
    }
}
