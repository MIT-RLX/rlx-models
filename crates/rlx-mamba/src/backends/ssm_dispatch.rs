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

//! Shared [`rlx_ssm`] flow dispatch for accelerator backends.

use anyhow::{Result, ensure};
use rlx_runtime::Device;

/// Device used to compile/run the Mamba1 SSM flow for this backend name.
#[allow(dead_code)]
pub fn execution_device(backend_name: &str) -> Device {
    match backend_name {
        #[cfg(feature = "cuda")]
        "cuda" => Device::Cuda,
        #[cfg(feature = "wgpu")]
        "wgpu" => Device::Gpu,
        #[cfg(feature = "rocm")]
        "rocm" => Device::Rocm,
        #[cfg(feature = "metal")]
        "metal" => Device::Metal,
        #[cfg(feature = "mlx")]
        "mlx" => Device::Mlx,
        _ => Device::Cpu,
    }
}

/// Prefill selective scan via [`crate::scan::selective_scan_on_device`].
#[allow(clippy::too_many_arguments)]
pub fn run_prefill_scan(
    device: Device,
    out: &mut [f32],
    u: &[f32],
    dt_raw: &[f32],
    b_mat: &[f32],
    c_mat: &[f32],
    a_log: &[f32],
    d_skip: &[f32],
    batch: usize,
    seq: usize,
    hidden: usize,
    state: usize,
) -> Result<()> {
    let y = crate::scan::selective_scan_on_device(
        device, batch, seq, hidden, state, u, dt_raw, b_mat, c_mat, a_log, d_skip,
    )?;
    ensure!(out.len() == y.len(), "prefill scan output length");
    out.copy_from_slice(&y);
    Ok(())
}
