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
pub fn rlx_feature_enabled() -> bool {
    cfg!(feature = "rlx")
}

#[inline]
pub fn native_feature_enabled() -> bool {
    cfg!(feature = "native")
}

#[inline]
pub fn espeak_feature_enabled() -> bool {
    cfg!(feature = "espeak")
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
pub fn gpu_feature_enabled() -> bool {
    cfg!(feature = "gpu")
}

pub fn enabled_backend_labels() -> Vec<&'static str> {
    let mut v = Vec::new();
    if native_feature_enabled() {
        v.push("rlx/native");
    }
    if espeak_feature_enabled() {
        v.push("espeak/phonemize");
    }
    if rlx_feature_enabled() {
        v.push("rlx/runtime");
        if metal_feature_enabled() {
            v.push("rlx/metal");
        }
        if mlx_feature_enabled() {
            v.push("rlx/mlx");
        }
        if cuda_feature_enabled() {
            v.push("rlx/cuda");
        }
        if rocm_feature_enabled() {
            v.push("rlx/rocm");
        }
        if gpu_feature_enabled() {
            v.push("rlx/gpu");
        }
    }
    v
}
