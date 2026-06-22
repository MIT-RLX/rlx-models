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

//! On-device multi-scale deformable attention: emits the fused
//! [`crate::deform_op`] custom op into a tiny HIR graph and dispatches it through
//! a [`Device`]. On `Cpu` it runs the registered CPU kernel; on GPU backends a
//! host-delegate kernel runs the same fused math (Metal auto-registers it; MLX
//! registers consumer-side; WGPU/CUDA via the engine's Step table). Memory-bounded.

use crate::deform_attn::{DeformWeights, LevelShape, RefPoints};
use crate::deform_op::{NUM_INPUTS, OP_NAME, encode_attrs, ensure_registered};
use crate::ir::Params;
use anyhow::Result;
use rlx_ir::{DType, HirGraphExt, HirModule, HirMut, HirNodeId, Op, Shape};
use rlx_runtime::Device;

/// Build the fused multi-scale deformable-attention `Op::Custom` node into a
/// shared HIR graph, returning the output node `[out_rows, d]`.
///
/// This is the single source of truth for the op's 11-input ABI and the eight
/// projection-weight param shapes — the enhancer, the decoder, and the
/// standalone [`MsDeformAttnIr`] runner all go through it, so the input order
/// and shapes can't drift out of sync with [`crate::deform_op`]. `out_rows` is
/// the query-row count (nq); `ref_dim` is 2 (centers) or 4 (boxes); `prefix`
/// namespaces the weight params within the caller's graph.
#[allow(clippy::too_many_arguments)]
pub fn build_deform_node(
    g: &mut HirMut<'_>,
    params: &mut Params,
    prefix: &str,
    query_n: HirNodeId,
    value_n: HirNodeId,
    ref_n: HirNodeId,
    w: &DeformWeights<'_>,
    d: usize,
    nh: usize,
    np: usize,
    ref_dim: usize,
    out_rows: usize,
    shapes: &[LevelShape],
) -> HirNodeId {
    let nl = shapes.len();
    let mut wparam = |suffix: &str, data: &[f32], shape: Vec<usize>| {
        let name = format!("{prefix}{suffix}");
        params.insert(name.clone(), data.to_vec());
        g.param(&name, Shape::new(&shape, DType::F32))
    };
    let vpw = wparam("dval.w", w.value_proj_w, vec![d, d]);
    let vpb = wparam("dval.b", w.value_proj_b, vec![d]);
    let sow = wparam("dsamp.w", w.sampling_offsets_w, vec![nh * nl * np * 2, d]);
    let sob = wparam("dsamp.b", w.sampling_offsets_b, vec![nh * nl * np * 2]);
    let aww = wparam("dattw.w", w.attention_weights_w, vec![nh * nl * np, d]);
    let awb = wparam("dattw.b", w.attention_weights_b, vec![nh * nl * np]);
    let opw = wparam("dout.w", w.output_proj_w, vec![d, d]);
    let opb = wparam("dout.b", w.output_proj_b, vec![d]);
    let attrs = encode_attrs(d, nh, np, ref_dim, shapes);
    g.add_node(
        Op::Custom {
            name: OP_NAME.to_string(),
            num_inputs: NUM_INPUTS as u32,
            attrs,
        },
        vec![
            query_n, value_n, ref_n, vpw, vpb, sow, sob, aww, awb, opw, opb,
        ],
        Shape::new(&[out_rows, d], DType::F32),
    )
}

/// Weights for one deformable-attention module (PyTorch `[out, in]` layout).
#[derive(Clone)]
pub struct DeformParams {
    pub value_proj_w: Vec<f32>,
    pub value_proj_b: Vec<f32>,
    pub sampling_offsets_w: Vec<f32>,
    pub sampling_offsets_b: Vec<f32>,
    pub attention_weights_w: Vec<f32>,
    pub attention_weights_b: Vec<f32>,
    pub output_proj_w: Vec<f32>,
    pub output_proj_b: Vec<f32>,
}

/// IR deformable-attention runner.
pub struct MsDeformAttnIr {
    w: DeformParams,
    d: usize,
    nh: usize,
    np: usize,
    device: Device,
}

impl MsDeformAttnIr {
    pub fn new(w: DeformParams, d: usize, nh: usize, np: usize, device: Device) -> Self {
        ensure_registered();
        Self {
            w,
            d,
            nh,
            np,
            device,
        }
    }

    /// `query` is `[nq, d]`, `value_src` is `[seq, d]`. Returns `[nq, d]`.
    pub fn forward(
        &self,
        query: &[f32],
        value_src: &[f32],
        ref_points: &RefPoints<'_>,
        shapes: &[LevelShape],
    ) -> Result<Vec<f32>> {
        let d = self.d;
        let nh = self.nh;
        let nq = query.len() / d;
        let seq = value_src.len() / d;
        let nl = shapes.len();
        let (ref_slice, ref_dim): (&[f32], usize) = match ref_points {
            RefPoints::Two(rp) => (rp, 2),
            RefPoints::Four(rp) => (rp, 4),
        };

        let mut hir = HirModule::new("ms_deform_attn");
        let mut params = Params::new();
        let mut g = HirMut::new(&mut hir);

        let query_n = g.input("query", Shape::new(&[nq, d], DType::F32));
        let value_n = g.input("value", Shape::new(&[seq, d], DType::F32));
        let ref_n = g.input("ref", Shape::new(&[nq, nl, ref_dim], DType::F32));

        let dw = DeformWeights {
            value_proj_w: &self.w.value_proj_w,
            value_proj_b: &self.w.value_proj_b,
            sampling_offsets_w: &self.w.sampling_offsets_w,
            sampling_offsets_b: &self.w.sampling_offsets_b,
            attention_weights_w: &self.w.attention_weights_w,
            attention_weights_b: &self.w.attention_weights_b,
            output_proj_w: &self.w.output_proj_w,
            output_proj_b: &self.w.output_proj_b,
        };
        let node = build_deform_node(
            &mut g,
            &mut params,
            "",
            query_n,
            value_n,
            ref_n,
            &dw,
            d,
            nh,
            self.np,
            ref_dim,
            nq,
            shapes,
        );
        g.set_outputs(vec![node]);

        let outs = crate::ir::compile_and_run(
            hir,
            params,
            self.device,
            &[("query", query), ("value", value_src), ("ref", ref_slice)],
        )?;
        Ok(outs.into_iter().next().unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deform_attn::{MsDeformAttn, level_start_index};

    fn det(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 13 + seed * 7) % 17) as f32 - 8.0) * 0.02)
            .collect()
    }

    /// Build a small synthetic module and return (native, ir-on-`device`).
    fn run_parity(device: Device) -> (Vec<f32>, Vec<f32>) {
        let (d, nh, nl, np) = (8usize, 2usize, 2usize, 2usize);
        let w = DeformParams {
            value_proj_w: det(d * d, 1),
            value_proj_b: vec![0.0; d],
            sampling_offsets_w: det(nh * nl * np * 2 * d, 2),
            sampling_offsets_b: det(nh * nl * np * 2, 3),
            attention_weights_w: det(nh * nl * np * d, 4),
            attention_weights_b: det(nh * nl * np, 5),
            output_proj_w: det(d * d, 6),
            output_proj_b: vec![0.0; d],
        };
        let shapes = [LevelShape { h: 4, w: 5 }, LevelShape { h: 2, w: 3 }];
        let seq: usize = shapes.iter().map(|s| s.h * s.w).sum();
        let starts = level_start_index(&shapes);
        let nq = 6usize;
        let query = det(nq * d, 7);
        let value = det(seq * d, 8);
        // 2-dim reference centers in (0,1).
        let rp: Vec<f32> = (0..nq * nl * 2)
            .map(|i| 0.3 + 0.01 * (i % 7) as f32)
            .collect();

        // Native reference (shares the same deform_forward core).
        let native_mod = MsDeformAttn::from_parts(
            d,
            nh,
            nl,
            np,
            w.value_proj_w.clone(),
            w.value_proj_b.clone(),
            w.sampling_offsets_w.clone(),
            w.sampling_offsets_b.clone(),
            w.attention_weights_w.clone(),
            w.attention_weights_b.clone(),
            w.output_proj_w.clone(),
            w.output_proj_b.clone(),
        );
        let native =
            native_mod.forward(&query, &value, &RefPoints::Two(&rp), &shapes, &starts, None);

        let ir = MsDeformAttnIr::new(w, d, nh, np, device);
        let got = ir
            .forward(&query, &value, &RefPoints::Two(&rp), &shapes)
            .unwrap();
        (native, got)
    }

    fn max_err(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn ir_deform_matches_native() {
        let (native, got) = run_parity(Device::Cpu);
        let e = max_err(&native, &got);
        assert!(e < 1e-5, "native vs IR(cpu) deform max_err={e}");
    }

    /// Runs the fused op on the WGPU backend (engine Step host-delegate).
    #[cfg(feature = "gpu")]
    #[test]
    fn ir_deform_matches_native_wgpu() {
        let (native, got) = run_parity(Device::Gpu);
        let e = max_err(&native, &got);
        assert!(e < 1e-4, "native vs IR(wgpu) deform max_err={e}");
    }
}
