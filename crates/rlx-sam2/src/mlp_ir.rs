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

//! Compile SAM2 mask-decoder ReLU MLP heads to IR.

use super::mask_decoder::Sam2HypernetMlp;
use anyhow::Result;
use rlx_flow::CompileProfile;
use rlx_runtime::Device;
use rlx_sam_ir::mlp_relu_ir::{MlpLayerSpec, MlpReluCompiled};

fn layers_from_mlp(mlp: &Sam2HypernetMlp) -> Vec<MlpLayerSpec> {
    mlp.layers
        .iter()
        .map(|l| MlpLayerSpec {
            w: l.w.clone(),
            b: l.b.clone(),
            in_d: l.in_d,
            out_d: l.out_d,
        })
        .collect()
}

pub fn compile_hyper_mlp(mlp: &Sam2HypernetMlp, device: Device) -> Result<MlpReluCompiled> {
    compile_hyper_mlp_with_profile(mlp, device, &CompileProfile::sam_encoder())
}

pub fn compile_hyper_mlp_with_profile(
    mlp: &Sam2HypernetMlp,
    device: Device,
    profile: &CompileProfile,
) -> Result<MlpReluCompiled> {
    MlpReluCompiled::compile_with_profile(
        &layers_from_mlp(mlp),
        mlp.sigmoid_output,
        1,
        device,
        profile,
    )
}

pub fn compile_hyper_mlps(
    mlps: &[Sam2HypernetMlp],
    device: Device,
) -> Result<Vec<MlpReluCompiled>> {
    compile_hyper_mlps_with_profile(mlps, device, &CompileProfile::sam_encoder())
}

pub fn compile_hyper_mlps_with_profile(
    mlps: &[Sam2HypernetMlp],
    device: Device,
    profile: &CompileProfile,
) -> Result<Vec<MlpReluCompiled>> {
    mlps.iter()
        .map(|m| compile_hyper_mlp_with_profile(m, device, profile))
        .collect()
}

pub fn compile_optional_hyper_mlp(
    mlp: &Option<Sam2HypernetMlp>,
    rows: usize,
    device: Device,
) -> Result<Option<MlpReluCompiled>> {
    compile_optional_hyper_mlp_with_profile(mlp, rows, device, &CompileProfile::sam_encoder())
}

pub fn compile_optional_hyper_mlp_with_profile(
    mlp: &Option<Sam2HypernetMlp>,
    rows: usize,
    device: Device,
    profile: &CompileProfile,
) -> Result<Option<MlpReluCompiled>> {
    mlp.as_ref()
        .map(|m| {
            MlpReluCompiled::compile_with_profile(
                &layers_from_mlp(m),
                m.sigmoid_output,
                rows,
                device,
                profile,
            )
        })
        .transpose()
}

/// Rows for `obj_ptr_proj`: 1, or `num_mask_tokens - 1` when multimask tokens feed the pointer.
pub fn obj_ptr_proj_rows(num_mask_tokens: usize, use_multimask_token_for_obj_ptr: bool) -> usize {
    if use_multimask_token_for_obj_ptr {
        num_mask_tokens.saturating_sub(1).max(1)
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask_decoder::{Sam2HypernetMlp, Sam2MlpLayer, mlp_forward};

    #[test]
    fn mlp_relu_ir_matches_host_on_synth() {
        let layers = vec![
            Sam2MlpLayer {
                w: vec![0.01; 256 * 256],
                b: vec![0.0; 256],
                in_d: 256,
                out_d: 256,
            },
            Sam2MlpLayer {
                w: vec![0.02; 256 * 32],
                b: vec![0.0; 32],
                in_d: 256,
                out_d: 32,
            },
        ];
        let mlp = Sam2HypernetMlp {
            layers,
            sigmoid_output: false,
        };
        let x: Vec<f32> = (0..256).map(|i| (i as f32) * 0.001).collect();
        let host = mlp_forward(&mlp, &x, 1);
        let mut ir = compile_hyper_mlp(&mlp, Device::Cpu).unwrap();
        let got = ir.run(&x, 1).unwrap();
        let fd = host
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(fd < 1e-3, "mlp IR vs host max |Δ| = {fd:.3e}");
    }
}
