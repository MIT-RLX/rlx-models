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

//! Classifier-free guidance: `neg + scale * (pos - neg)` (matches Modular MAX).

use anyhow::Result;
use rlx_ir::hir::{FusionPolicy, HirModule};
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Op, Shape};
use rlx_runtime::Device;

/// Native CFG blend in float32.
pub fn cfg_combine(pos: &[f32], neg: &[f32], scale: f32) -> Vec<f32> {
    pos.iter()
        .zip(neg)
        .map(|(&p, &n)| n + scale * (p - n))
        .collect()
}

pub struct Flux2CfgCombineGraph {
    pub hir: HirModule,
}

/// Emit `neg + guidance_scale * (pos - neg)` into an existing HIR module (inputs must exist).
pub fn emit_flux2_cfg_combine(
    hir: &mut HirModule,
    pos: rlx_ir::HirNodeId,
    neg: rlx_ir::HirNodeId,
    scale: rlx_ir::HirNodeId,
    shape: Shape,
) -> rlx_ir::HirNodeId {
    let diff = hir.mir(Op::Binary(BinaryOp::Sub), vec![pos, neg], shape.clone());
    let scaled = hir.mir(Op::Binary(BinaryOp::Mul), vec![diff, scale], shape.clone());
    hir.mir(Op::Binary(BinaryOp::Add), vec![neg, scaled], shape)
}

/// HIR graph: `neg + guidance_scale * (pos - neg)` in f32.
///
/// Inputs: `pos`, `neg` `[batch, seq, channels]`; `guidance_scale` scalar f32.
pub fn build_flux2_cfg_combine_hir(
    batch: usize,
    seq: usize,
    channels: usize,
) -> Flux2CfgCombineGraph {
    let mut hir = HirModule::new("flux2_cfg_combine").with_fusion_policy(FusionPolicy::Direct);
    let f = DType::F32;
    let shape = Shape::new(&[batch, seq, channels], f);
    let pos = hir.input("pos", shape.clone());
    let neg = hir.input("neg", shape.clone());
    let scale = hir.input("guidance_scale", Shape::scalar(f));
    let out = emit_flux2_cfg_combine(&mut hir, pos, neg, scale, shape);
    hir.outputs = vec![out];
    Flux2CfgCombineGraph { hir }
}

pub fn compile_flux2_cfg_combine(
    batch: usize,
    seq: usize,
    channels: usize,
    device: Device,
    aot: Option<&rlx_runtime::AotCache>,
) -> Result<rlx_runtime::CompiledGraph> {
    use crate::compile_util::{compile_hir_cached, flux2_cfg_aot_key};

    crate::device::assert_flux2_device_available(device)?;
    let g = build_flux2_cfg_combine_hir(batch, seq, channels);
    let key = flux2_cfg_aot_key(device, batch, seq, channels);
    compile_hir_cached(
        device,
        aot,
        &key,
        g.hir,
        &super::compile_util::flux2_compile_profile(),
    )
}

/// Tier-0 CFG combine via [`ModelFlow`](rlx_flow::ModelFlow).
pub fn build_flux2_cfg_combine_built(
    batch: usize,
    seq: usize,
    channels: usize,
) -> Result<rlx_flow::BuiltModel> {
    use rlx_flow::{MapWeights, ModelFlow};

    let f = DType::F32;
    let shape = Shape::new(&[batch, seq, channels], f);
    let out_shape = shape.clone();
    ModelFlow::new("flux2_cfg_combine")
        .input("pos", shape.clone())
        .input("neg", shape.clone())
        .input("guidance_scale", Shape::scalar(f))
        .plugin_named("flux2.cfg.combine", move |emit, _| {
            let pos = emit.flow_input("pos")?.hir_id();
            let neg = emit.flow_input("neg")?.hir_id();
            let scale = emit.flow_input("guidance_scale")?.hir_id();
            let (hir, _) = emit.hir_and_params();
            let out = emit_flux2_cfg_combine(hir, pos, neg, scale, out_shape.clone());
            Ok(Some(emit.wrap(out, out_shape.clone())))
        })
        .output("output")
        .build(&mut MapWeights::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_runtime::Device;

    #[test]
    fn cfg_native_matches_hir_cpu() {
        let pos = vec![1.0f32, 2.0, 3.0, 4.0];
        let neg = vec![0.0f32, 0.5, 1.0, 1.5];
        let scale = 2.5f32;
        let native = cfg_combine(&pos, &neg, scale);

        let mut compiled = compile_flux2_cfg_combine(1, 2, 2, Device::Cpu, None).unwrap();
        let out = compiled
            .run(&[
                ("pos", pos.as_slice()),
                ("neg", neg.as_slice()),
                ("guidance_scale", &[scale]),
            ])
            .remove(0);

        assert_eq!(out.len(), native.len());
        let max = out
            .iter()
            .zip(&native)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-5, "max_diff={max}");
    }
}
