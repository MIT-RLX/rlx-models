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

//! Largest FFT batch a device can run for a given `(n_fft, dtype, limbs)`.
//!
//! Two independent ceilings bound the batch:
//!   * **memory**   — `batch · row_bytes` must fit the device working-set budget;
//!   * **dispatch** — GPU grid/workgroup-count caps (wgpu/Vulkan reject a compute
//!     dispatch whose per-dimension workgroup count exceeds **65535** — the
//!     concrete crash you hit at `batch = 65536`).
//!
//! [`max_fft_batch`] is a **pure** function (fully unit-testable, no hardware);
//! [`detect_device_caps`] reads the live machine — unified-memory `sysctl`
//! ([`rlx_runtime::soft_memory_budget_bytes`], honoring `RLX_SOFT_MEMORY_*`)
//! plus per-backend dispatch limits — and feeds the formula. [`auto_max_fft_batch`]
//! is the one-call convenience.
//!
//! On Apple Silicon every backend (CPU / Metal / MLX / wgpu / ANE) draws from the
//! same unified pool, so one `hw.memsize` budget is correct across all of them.
//! On a discrete GPU the memory bound is VRAM, which we can't introspect here —
//! override with `RLX_FFT_MEM_BUDGET_MB` in that case.

use rlx_runtime::{Device, soft_memory_budget_bytes};

/// wgpu / Vulkan hard limit: `max_compute_workgroups_per_dimension`.
/// A dispatch with more than this many workgroups along any dim is rejected.
pub const WGPU_MAX_WORKGROUPS_PER_DIM: u64 = 65535;

/// Fallback memory budget when the machine can't be queried and no override is
/// set (non-macOS without `RLX_SOFT_MEMORY_BUDGET_BYTES`): a conservative 2 GiB.
pub const DEFAULT_MEM_BUDGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// FFT problem shape that sets the per-batch-row memory footprint.
#[derive(Clone, Copy, Debug)]
pub struct FftProblem {
    /// Transform length.
    pub n_fft: usize,
    /// Bytes per real scalar of the base float (f16=2, f32=4, f64=8).
    pub elem_bytes: usize,
    /// Compensated multi-limb factor (1 = plain, 2 = double-word, 4 = quad).
    pub limbs: usize,
    /// In + out + twiddle/scratch safety factor (≈3 for an out-of-place FFT).
    pub working_copies: f64,
}

impl FftProblem {
    /// Plain single-limb f32, 3 working copies — the native butterfly default.
    pub fn f32(n_fft: usize) -> Self {
        Self {
            n_fft,
            elem_bytes: 4,
            limbs: 1,
            working_copies: 3.0,
        }
    }
    /// Build from a base-float byte width and limb count.
    pub fn new(n_fft: usize, elem_bytes: usize, limbs: usize) -> Self {
        Self {
            n_fft,
            elem_bytes,
            limbs,
            working_copies: 3.0,
        }
    }
    /// Bytes one batch row occupies: complex (`×2`) · limbs · copies.
    pub fn row_footprint_bytes(&self) -> u64 {
        let base = (self.n_fft as u64) * 2 * (self.elem_bytes as u64) * (self.limbs.max(1) as u64);
        ((base as f64) * self.working_copies.max(1.0)).ceil() as u64
    }
}

/// Which ceiling bounds the batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitedBy {
    Memory,
    Dispatch,
}

/// Hardware ceilings for one device.
#[derive(Clone, Copy, Debug)]
pub struct DeviceCaps {
    /// Working-set memory budget in bytes.
    pub mem_budget_bytes: u64,
    /// Max batch from grid/workgroup limits; `None` = grid-unbounded (memory only).
    pub dispatch_cap: Option<u64>,
    /// Human label for where `mem_budget_bytes` came from (for reporting).
    pub mem_source: &'static str,
}

/// Result of [`max_fft_batch`].
#[derive(Clone, Copy, Debug)]
pub struct BatchCapacity {
    /// Largest safe batch.
    pub max_batch: usize,
    /// Which ceiling won.
    pub limited_by: LimitedBy,
    /// Batch the memory budget alone permits.
    pub mem_cap: usize,
    /// Batch the dispatch cap alone permits (`None` = unbounded).
    pub dispatch_cap: Option<usize>,
    /// Bytes one batch row costs.
    pub row_footprint_bytes: u64,
    /// Memory budget used.
    pub mem_budget_bytes: u64,
}

/// Pure formula: smallest of the memory and dispatch ceilings.
pub fn max_fft_batch(problem: FftProblem, caps: DeviceCaps) -> BatchCapacity {
    let row = problem.row_footprint_bytes().max(1);
    let mem_cap = (caps.mem_budget_bytes / row) as usize;
    let disp = caps.dispatch_cap.map(|c| c as usize);
    let (max_batch, limited_by) = match disp {
        Some(d) if d < mem_cap => (d, LimitedBy::Dispatch),
        _ => (mem_cap, LimitedBy::Memory),
    };
    BatchCapacity {
        max_batch,
        limited_by,
        mem_cap,
        dispatch_cap: disp,
        row_footprint_bytes: row,
        mem_budget_bytes: caps.mem_budget_bytes,
    }
}

/// Dispatch ceiling for a device's batch dimension. wgpu/Vulkan map one
/// workgroup per row along a single dim and reject counts above 65535; Metal /
/// CUDA-x / CPU map the batch onto a grid dim large enough to be memory-bound
/// in practice, so they return `None`.
///
/// Caveat: this is the small-/mid-`n_fft` (batch-along-a-dim) cap and is
/// **necessary but not sufficient** on wgpu. At large `n_fft` the FFT kernel
/// tiles work as `batch · n_fft/32` along one dim, so the true ceiling tightens
/// with `n_fft` — modeling that exactly needs the kernel's tiling, so we keep
/// the safe-for-typical batch cap here and let `bench-sweep` skip + the latency
/// heatmap chart the empirical envelope (its `cap` cells).
pub fn dispatch_cap_for(device: Device) -> Option<u64> {
    match device {
        Device::Gpu | Device::Vulkan => Some(WGPU_MAX_WORKGROUPS_PER_DIM),
        _ => None,
    }
}

/// Read the live machine's ceilings for `device`. Memory comes from the unified
/// budget (`RLX_SOFT_MEMORY_*` aware); an explicit `RLX_FFT_MEM_BUDGET_MB`
/// overrides it (needed for discrete-GPU VRAM, which isn't introspected here).
pub fn detect_device_caps(device: Device) -> DeviceCaps {
    let (mem_budget_bytes, mem_source) = match mem_budget_override() {
        Some(b) => (b, "RLX_FFT_MEM_BUDGET_MB override"),
        None => match soft_memory_budget_bytes() {
            Some(b) => (b as u64, "unified memory (hw.memsize × soft fraction)"),
            None => (
                DEFAULT_MEM_BUDGET_BYTES,
                "default 2 GiB (set RLX_FFT_MEM_BUDGET_MB)",
            ),
        },
    };
    DeviceCaps {
        mem_budget_bytes,
        dispatch_cap: dispatch_cap_for(device),
        mem_source,
    }
}

fn mem_budget_override() -> Option<u64> {
    std::env::var("RLX_FFT_MEM_BUDGET_MB")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|mb| mb * 1024 * 1024)
}

/// One call: detect `device` ceilings and compute the max batch for `problem`.
pub fn auto_max_fft_batch(device: Device, problem: FftProblem) -> BatchCapacity {
    max_fft_batch(problem, detect_device_caps(device))
}

/// Clamp a requested batch to the device's capacity. Returns the safe batch and
/// `Some(reason)` when it had to be lowered (so callers can warn rather than
/// crash — e.g. the wgpu 65536 dispatch panic).
pub fn clamp_batch(
    device: Device,
    problem: FftProblem,
    requested: usize,
) -> (usize, Option<LimitedBy>) {
    let cap = auto_max_fft_batch(device, problem);
    if requested > cap.max_batch {
        (cap.max_batch, Some(cap.limited_by))
    } else {
        (requested, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(mem_gib: u64, dispatch: Option<u64>) -> DeviceCaps {
        DeviceCaps {
            mem_budget_bytes: mem_gib * 1024 * 1024 * 1024,
            dispatch_cap: dispatch,
            mem_source: "test",
        }
    }

    #[test]
    fn memory_bound_when_no_dispatch_cap() {
        // n=256 f32: row = 256·2·4·3 = 6144 B. 6 GiB / 6144 ≈ 1.05M rows.
        let cap = max_fft_batch(FftProblem::f32(256), caps(6, None));
        assert_eq!(cap.limited_by, LimitedBy::Memory);
        assert_eq!(cap.row_footprint_bytes, 256 * 2 * 4 * 3);
        assert_eq!(cap.max_batch, (6u64 * 1024 * 1024 * 1024 / 6144) as usize);
        assert_eq!(cap.dispatch_cap, None);
    }

    #[test]
    fn dispatch_bound_kicks_in_for_wgpu() {
        // Plenty of memory, but the 65535 workgroup cap wins → batch 65536 unsafe.
        let cap = max_fft_batch(
            FftProblem::f32(256),
            caps(64, Some(WGPU_MAX_WORKGROUPS_PER_DIM)),
        );
        assert_eq!(cap.limited_by, LimitedBy::Dispatch);
        assert_eq!(cap.max_batch, 65535);
        assert!(
            65536 > cap.max_batch,
            "the crash batch is correctly excluded"
        );
    }

    #[test]
    fn memory_can_be_tighter_than_dispatch() {
        // Tiny budget: even wgpu is memory-bound below its 65535 cap.
        let cap = max_fft_batch(FftProblem::new(4096, 4, 1), caps(1, Some(65535)));
        // row = 4096·2·4·3 = 98304 B → 1 GiB / 98304 ≈ 10922 < 65535.
        assert_eq!(cap.limited_by, LimitedBy::Memory);
        assert!(cap.max_batch < 65535);
    }

    #[test]
    fn precision_costs_batch() {
        // Same budget (chosen to divide evenly): f64 halves the batch vs f32,
        // and quad-limb f32 quarters it again — wider words cost proportional VRAM.
        let mem = DeviceCaps {
            mem_budget_bytes: 24576 * 100_000, // = f32x4 row · 100k
            dispatch_cap: None,
            mem_source: "test",
        };
        let f32x1 = max_fft_batch(FftProblem::new(256, 4, 1), mem).max_batch;
        let f64x1 = max_fft_batch(FftProblem::new(256, 8, 1), mem).max_batch;
        let f32x4 = max_fft_batch(FftProblem::new(256, 4, 4), mem).max_batch;
        assert_eq!(f32x1, 2 * f64x1);
        assert_eq!(f32x1, 4 * f32x4);
    }

    #[test]
    fn detect_caps_sets_wgpu_dispatch() {
        assert_eq!(dispatch_cap_for(Device::Gpu), Some(65535));
        assert_eq!(dispatch_cap_for(Device::Cpu), None);
        assert_eq!(dispatch_cap_for(Device::Metal), None);
        // Live detection yields a positive budget on this (macOS) machine.
        let c = detect_device_caps(Device::Gpu);
        assert!(c.mem_budget_bytes > 0);
        assert_eq!(c.dispatch_cap, Some(65535));
    }

    #[test]
    fn clamp_lowers_over_cap_batch() {
        // Force the wgpu cap via a huge memory override so dispatch is the bound.
        // SAFETY: single-threaded test; env set+read locally.
        unsafe { std::env::set_var("RLX_FFT_MEM_BUDGET_MB", "1048576") }; // 1 TiB
        let (b, why) = clamp_batch(Device::Gpu, FftProblem::f32(256), 200_000);
        assert_eq!(b, 65535);
        assert_eq!(why, Some(LimitedBy::Dispatch));
        let (b2, why2) = clamp_batch(Device::Gpu, FftProblem::f32(256), 1024);
        assert_eq!(b2, 1024);
        assert_eq!(why2, None);
        unsafe { std::env::remove_var("RLX_FFT_MEM_BUDGET_MB") };
    }
}
