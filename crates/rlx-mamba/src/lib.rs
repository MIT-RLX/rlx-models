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

//! Native Mamba1 (selective state-space model) for the rlx workspace.
//!
//! Algorithmically equivalent to `burn-mamba::mamba1` (same paper §3.2
//! / Algorithm 2: in_proj → causal conv1d → SiLU → SSM → SiLU gate →
//! out_proj). Numerical parity with burn-mamba is verified element-wise
//! on shared weights — see `crates/rlx-mamba-bench/tests/parity.rs`.
//!
//! Two host-side modes (CPU only) live in [`block::Mamba1Block`]:
//! * `forward` — parallel prefill (SSM core via [`scan::selective_scan_flow`]
//!   → `rlx_ssm::MambaScanStage` / `Op::SelectiveScan`)
//! * `step` — single-token decode with rolling cache (`scan::selective_scan_step_flow`)
//!
//! For multi-backend execution, use the [`backend::MambaBackend`] trait
//! and the [`driver::mamba1_forward`] driver. All backends route the SSM
//! through the same `rlx-ssm` flows as [`block::Mamba1Block`]; linears and
//! conv stay on per-backend matmul paths. Enable `metal`, `mlx`, `cuda`,
//! `wgpu`, or `rocm` features for accelerators (see `src/backends/`).

pub mod backend;
pub mod backends;
pub mod block;
pub mod cache;
pub mod config;
pub mod driver;
pub mod network;
pub mod scan;

pub use backend::{MambaBackend, MambaTensor};
pub use backends::cpu::{CpuBackend, CpuTensor};
pub use block::Mamba1Block;
pub use cache::{Mamba1Cache, Mamba1Caches};
pub use config::{Mamba1Config, Mamba1NetworkConfig};
pub use driver::{Mamba1ResidentBlock, mamba1_forward};
pub use network::{Mamba1Layer, Mamba1Network};
pub use scan::{
    effective_scan_device, ensure_ssm_ops_registered, selective_scan_flow,
    selective_scan_on_device, selective_scan_step_flow, selective_scan_step_on_device,
};
