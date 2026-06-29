#![allow(dead_code)]

//! Shared backend list for Orpheus integration tests.

use rlx_runtime::Device;

pub const BACKENDS: &[(Device, &str)] = &[
    (Device::Cpu, "CPU"),
    (Device::Metal, "Metal"),
    (Device::Mlx, "MLX"),
    (Device::Cuda, "CUDA"),
    (Device::Rocm, "ROCm"),
    (Device::Gpu, "wgpu"),
    (Device::Vulkan, "Vulkan"),
];
