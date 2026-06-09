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

//! Compile-time feature probes and backend labels.

#[inline]
pub fn codec_feature_enabled() -> bool {
    cfg!(feature = "codec")
}

/// RLX `rlx-llama32` backbone (production).
#[inline]
pub fn llama_feature_enabled() -> bool {
    cfg!(feature = "llama")
}

/// Alias for [`llama_feature_enabled`].
#[inline]
pub fn llama_cpp_feature_enabled() -> bool {
    llama_feature_enabled()
}

#[inline]
pub fn backbone_feature_enabled() -> bool {
    llama_feature_enabled()
}

#[inline]
pub fn parity_llama_cpp_feature_enabled() -> bool {
    cfg!(feature = "parity-llama-cpp")
}

#[inline]
pub fn burn_feature_enabled() -> bool {
    cfg!(feature = "burn")
}

#[inline]
pub fn wgpu_feature_enabled() -> bool {
    burn_feature_enabled()
}

#[inline]
pub fn rlx_feature_enabled() -> bool {
    cfg!(feature = "rlx")
}

#[inline]
pub fn metal_feature_enabled() -> bool {
    cfg!(feature = "metal")
}

#[inline]
pub fn mlx_feature_enabled() -> bool {
    cfg!(feature = "mlx")
}

#[inline]
pub fn cuda_feature_enabled() -> bool {
    cfg!(feature = "cuda")
}

#[inline]
pub fn rocm_feature_enabled() -> bool {
    cfg!(feature = "rocm")
}

#[inline]
pub fn vulkan_feature_enabled() -> bool {
    cfg!(feature = "vulkan")
}

#[inline]
pub fn gpu_feature_enabled() -> bool {
    cfg!(feature = "gpu")
}

pub fn enabled_backend_labels() -> Vec<&'static str> {
    let mut v = Vec::new();
    if codec_feature_enabled() {
        v.push("codec/eager-ndarray");
    }
    if burn_feature_enabled() {
        v.push("burn/wgpu");
    }
    if rlx_feature_enabled() {
        v.push("rlx/codec-parity");
    }
    if llama_feature_enabled() {
        v.push("rlx-llama32");
        if metal_feature_enabled() {
            v.push("backbone/metal");
        }
        if mlx_feature_enabled() {
            v.push("backbone/mlx");
        }
        if cuda_feature_enabled() {
            v.push("backbone/cuda");
        }
        if rocm_feature_enabled() {
            v.push("backbone/rocm");
        }
        if vulkan_feature_enabled() {
            v.push("backbone/vulkan");
        }
        if gpu_feature_enabled() {
            v.push("backbone/gpu");
        }
    }
    if parity_llama_cpp_feature_enabled() {
        v.push("llama-cpp-ref");
    }
    v
}
