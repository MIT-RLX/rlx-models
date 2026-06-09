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

//! Backend implementations for [`crate::backend::MambaBackend`].
//!
//! `cpu` is always built. The accelerator backends are feature-gated
//! and additionally `#[cfg(target_os = ...)]`-gated where appropriate
//! (Metal/MLX only build on Apple; CUDA pulls native NVIDIA libs;
//! wgpu/ROCm have their own platform constraints).

pub mod cpu;
mod ssm_dispatch;

#[cfg(feature = "metal")]
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod metal;

#[cfg(feature = "mlx")]
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod mlx;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "wgpu")]
pub mod wgpu;

#[cfg(feature = "rocm")]
pub mod rocm;
