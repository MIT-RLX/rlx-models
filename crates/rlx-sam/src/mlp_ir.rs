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

//! Compile SAM1 mask-decoder ReLU MLP heads to IR.

use super::mask_decoder::HypernetMlp;
use anyhow::Result;
use rlx_flow::CompileProfile;
use rlx_runtime::Device;
use rlx_sam_ir::mlp_relu_ir::{MlpLayerSpec, MlpReluCompiled};

pub fn compile_hyper_mlp(mlp: &HypernetMlp, device: Device) -> Result<MlpReluCompiled> {
    compile_hyper_mlp_with_profile(mlp, device, &CompileProfile::sam_encoder())
}

pub fn compile_hyper_mlp_with_profile(
    mlp: &HypernetMlp,
    device: Device,
    profile: &CompileProfile,
) -> Result<MlpReluCompiled> {
    let layers = mlp
        .layers
        .iter()
        .map(|l| MlpLayerSpec {
            w: l.w.clone(),
            b: l.b.clone(),
            in_d: l.in_d,
            out_d: l.out_d,
        })
        .collect::<Vec<_>>();
    MlpReluCompiled::compile_with_profile(&layers, false, 1, device, profile)
}

pub fn compile_hyper_mlps(mlps: &[HypernetMlp], device: Device) -> Result<Vec<MlpReluCompiled>> {
    compile_hyper_mlps_with_profile(mlps, device, &CompileProfile::sam_encoder())
}

pub fn compile_hyper_mlps_with_profile(
    mlps: &[HypernetMlp],
    device: Device,
    profile: &CompileProfile,
) -> Result<Vec<MlpReluCompiled>> {
    mlps.iter()
        .map(|m| compile_hyper_mlp_with_profile(m, device, profile))
        .collect()
}

pub fn compile_iou_head(mlp: &HypernetMlp, device: Device) -> Result<MlpReluCompiled> {
    compile_iou_head_with_profile(mlp, device, &CompileProfile::sam_encoder())
}

pub fn compile_iou_head_with_profile(
    mlp: &HypernetMlp,
    device: Device,
    profile: &CompileProfile,
) -> Result<MlpReluCompiled> {
    compile_hyper_mlp_with_profile(mlp, device, profile)
}
