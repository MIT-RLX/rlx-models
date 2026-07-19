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

//! Default accelerator routing for Qwen3-TTS synthesis (Metal / MLX / CUDA / ROCm).

use rlx_runtime::Device;

/// True when the session should prefer GPU backends for talker, CP, and speech decode.
/// Opt out entirely: `RLX_QWEN3_TTS_CPU_PIPELINE=1`.
pub fn gpu_session_enabled(device: Device) -> bool {
    if std::env::var("RLX_QWEN3_TTS_CPU_PIPELINE").ok().as_deref() == Some("1") {
        return false;
    }
    matches!(
        device,
        Device::Metal | Device::Mlx | Device::Cuda | Device::Rocm
    )
}

pub fn metal_compiled_default() -> bool {
    std::env::var("RLX_QWEN3_TTS_METAL_COMPILED")
        .ok()
        .as_deref()
        != Some("0")
}

/// Metal bucketed talker decode on native Metal graphs. Default **off** — native decode
/// diverges from HF greedy today; GPU sessions use CPU decode graphs (hybrid) until parity.
/// Opt in: `RLX_QWEN3_TTS_METAL_DECODE_NATIVE=1`. Force CPU graphs: `RLX_QWEN3_TTS_METAL_DECODE_CPU=1`.
pub fn metal_decode_native_default() -> bool {
    if std::env::var("RLX_QWEN3_TTS_METAL_DECODE_CPU")
        .ok()
        .as_deref()
        == Some("1")
    {
        return false;
    }
    std::env::var("RLX_QWEN3_TTS_METAL_DECODE_NATIVE")
        .ok()
        .as_deref()
        == Some("1")
}

pub fn cp_compiled_default() -> bool {
    std::env::var("RLX_QWEN3_TTS_CP_EAGER").ok().as_deref() != Some("1")
}

/// Metal: CPU eager CP micro-kernel wins today (~58 ms/frame vs ~1.7 s compiled).
/// Opt into Metal CP graphs: `RLX_QWEN3_TTS_CP_METAL=1`.
pub fn cp_use_gpu_on_device(device: Device) -> bool {
    if !cp_compiled_default() {
        return false;
    }
    match device {
        Device::Metal => {
            std::env::var("RLX_QWEN3_TTS_CP_METAL").ok().as_deref() == Some("1")
                || std::env::var("RLX_QWEN3_TTS_CP_COMPILED").ok().as_deref() == Some("1")
        }
        Device::Mlx => std::env::var("RLX_QWEN3_TTS_MLX_COMPILED").ok().as_deref() == Some("1"),
        Device::Cuda | Device::Rocm => true,
        _ => false,
    }
}

pub fn speech_compiled_default() -> bool {
    std::env::var("RLX_QWEN3_TTS_SPEECH_EAGER").ok().as_deref() != Some("1")
}

/// Speech conv/vocoder tail (causal + transposed conv matmuls on GPU when available).
pub fn speech_conv_gpu_default() -> bool {
    std::env::var("RLX_QWEN3_TTS_SPEECH_CONV_CPU")
        .ok()
        .as_deref()
        != Some("1")
        && speech_compiled_default()
}

/// Device for progressive partial speech decode.
///
/// Metal/MLX compiled speech historically disagreed across prefix lengths; the
/// speech `PreTransformerGpu` now pads every forward to the warmup length so
/// progressive and one-shot share one compiled graph. Default remains CPU on
/// Apple until `streaming_pcm_parity` is green with
/// `RLX_QWEN3_TTS_PROGRESSIVE_SPEECH_GPU=1`. CUDA/ROCm always stay on the
/// session device.
pub fn progressive_speech_decode_device(device: Device) -> Device {
    match device {
        Device::Metal | Device::Mlx => {
            if std::env::var("RLX_QWEN3_TTS_PROGRESSIVE_SPEECH_GPU")
                .ok()
                .as_deref()
                == Some("1")
            {
                device
            } else {
                Device::Cpu
            }
        }
        other => other,
    }
}

/// Metal / MLX: CPU conv default (depthwise flat im2col + parity); opt in GPU conv via
/// `RLX_QWEN3_TTS_SPEECH_CONV_GPU=1`.
pub fn speech_conv_use_gpu(device: Device) -> bool {
    if !speech_conv_gpu_default()
        || !gpu_session_enabled(device)
        || device == Device::Cpu
        || !rlx_runtime::is_available(device)
    {
        return false;
    }
    if matches!(device, Device::Metal | Device::Mlx) {
        return std::env::var("RLX_QWEN3_TTS_SPEECH_CONV_GPU")
            .ok()
            .as_deref()
            == Some("1");
    }
    true
}

/// Metal GPU sessions: CPU eager talker decode (HF parity) instead of slow compiled CPU bucket graphs.
/// Native Metal decode remains opt-in via `RLX_QWEN3_TTS_METAL_DECODE_NATIVE=1`.
pub fn talker_eager_decode_default(device: Device) -> bool {
    device == Device::Metal
        && gpu_session_enabled(device)
        && metal_compiled_default()
        && !metal_decode_native_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_decode_native_off_by_default() {
        assert!(
            !metal_decode_native_default(),
            "native Metal talker decode must be opt-in until HF parity"
        );
    }

    #[test]
    fn metal_eager_decode_default_when_gpu_session() {
        unsafe {
            std::env::remove_var("RLX_QWEN3_TTS_CPU_PIPELINE");
            std::env::remove_var("RLX_QWEN3_TTS_METAL_DECODE_NATIVE");
        }
        assert!(talker_eager_decode_default(Device::Metal));
    }

    #[test]
    fn progressive_speech_metal_defaults_to_cpu() {
        unsafe {
            std::env::remove_var("RLX_QWEN3_TTS_PROGRESSIVE_SPEECH_GPU");
        }
        assert_eq!(progressive_speech_decode_device(Device::Metal), Device::Cpu);
        assert_eq!(progressive_speech_decode_device(Device::Cuda), Device::Cuda);
    }

    #[test]
    fn progressive_speech_metal_opt_in_gpu() {
        unsafe {
            std::env::set_var("RLX_QWEN3_TTS_PROGRESSIVE_SPEECH_GPU", "1");
        }
        assert_eq!(
            progressive_speech_decode_device(Device::Metal),
            Device::Metal
        );
        unsafe {
            std::env::remove_var("RLX_QWEN3_TTS_PROGRESSIVE_SPEECH_GPU");
        }
    }
}
