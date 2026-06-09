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

//! State-space model flow stages and custom IR ops for hybrid / Mamba runners.
//!
//! * **Prefill:** [`MambaScanStage`] → `Op::SelectiveScan` (softplus on `dt_raw`, `A = -exp(A_log)`).
//! * **Decode:** [`Mamba1StepStage`], [`Mamba2StepStage`], [`LfmSsmStepStage`],
//!   [`LightningAttentionStepStage`] → packed outputs + CPU reference kernels.
//!
//! Call [`register_ir_ops`] once before building or compiling flows that use these stages.

mod kernels;
mod registry;
mod stages;

pub use registry::ssm_kernels;
pub use stages::{
    LfmSsmStepStage, LightningAttentionStage, LightningAttentionStepStage, Mamba1StepStage,
    Mamba2StepStage, MambaScanStage, MambaScanWeightKeys,
};

pub use kernels::{
    execute_lfm_ssm_step_f32, execute_lightning_attention_step_f32, execute_mamba1_step_f32,
    execute_mamba2_step_f32,
};

/// Register SSM custom-step IR ops + CPU kernels.
pub fn register_ir_ops() {
    registry::register_ir_ops();
}

/// Compatibility shim for older callers.
pub fn register_ssm_kernels() {
    register_ir_ops();
}
