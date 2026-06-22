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

//! `Op::Custom` for fused multi-scale deformable attention.
//!
//! The whole module forward runs in one kernel (value/offset/weight projections,
//! bilinear sampling, output projection), so it stays memory-bounded — no
//! `[nq·heads·levels·points·4, head_dim]` intermediate. The CPU kernel is
//! self-contained here (keeps the default build crates.io-compatible). The
//! Metal/MLX backends auto-register their own host-delegate kernels for this op
//! on first lookup (see `../rlx` `rlx-metal`/`rlx-mlx` `op_registry`), so running
//! grounding-dino on those devices needs no extra registration or cargo feature.

use crate::deform_attn::{DeformWeights, LevelShape, deform_forward, level_start_index};
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};
use rlx_ir::{Node, NodeId, OpExtension, Shape, VjpContext, register_op};
use std::sync::{Arc, Mutex, OnceLock};

pub const OP_NAME: &str = "gdino.ms_deform_attn";

/// Input order for the custom op node.
pub const NUM_INPUTS: usize = 11;

/// Encode the op attributes: `[d, nh, np, ref_dim, nl, (h,w)*nl]` as LE u32.
pub fn encode_attrs(
    d: usize,
    nh: usize,
    np: usize,
    ref_dim: usize,
    shapes: &[LevelShape],
) -> Vec<u8> {
    let mut words: Vec<u32> = vec![
        d as u32,
        nh as u32,
        np as u32,
        ref_dim as u32,
        shapes.len() as u32,
    ];
    for s in shapes {
        words.push(s.h as u32);
        words.push(s.w as u32);
    }
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

struct Attrs {
    d: usize,
    nh: usize,
    np: usize,
    ref_dim: usize,
    shapes: Vec<LevelShape>,
}

fn decode_attrs(bytes: &[u8]) -> Result<Attrs, String> {
    let rd = |i: usize| -> u32 {
        u32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ])
    };
    if bytes.len() < 5 * 4 {
        return Err("ms_deform_attn: attrs too short".into());
    }
    let d = rd(0) as usize;
    let nh = rd(1) as usize;
    let np = rd(2) as usize;
    let ref_dim = rd(3) as usize;
    let nl = rd(4) as usize;
    if bytes.len() < (5 + nl * 2) * 4 {
        return Err("ms_deform_attn: attrs truncated shapes".into());
    }
    let shapes = (0..nl)
        .map(|l| LevelShape {
            h: rd(5 + l * 2) as usize,
            w: rd(5 + l * 2 + 1) as usize,
        })
        .collect();
    Ok(Attrs {
        d,
        nh,
        np,
        ref_dim,
        shapes,
    })
}

struct DeformIr;
impl OpExtension for DeformIr {
    fn name(&self) -> &str {
        OP_NAME
    }
    fn num_inputs(&self) -> usize {
        NUM_INPUTS
    }
    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        // Output matches the query shape [nq, d].
        inputs[0].clone()
    }
    fn vjp(&self, _node: &Node, _ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        vec![]
    }
}

/// Shared host compute (also used by the engine's GPU host-delegates).
pub fn run_host(inputs: &[&[f32]], attrs: &[u8]) -> Result<Vec<f32>, String> {
    let a = decode_attrs(attrs)?;
    let starts = level_start_index(&a.shapes);
    let w = DeformWeights {
        value_proj_w: inputs[3],
        value_proj_b: inputs[4],
        sampling_offsets_w: inputs[5],
        sampling_offsets_b: inputs[6],
        attention_weights_w: inputs[7],
        attention_weights_b: inputs[8],
        output_proj_w: inputs[9],
        output_proj_b: inputs[10],
    };
    Ok(deform_forward(
        inputs[0], // query
        inputs[1], // value_src
        inputs[2], // reference points
        a.ref_dim, &a.shapes, &starts, a.d, a.nh, a.np, &w, None,
    ))
}

struct DeformCpu;
impl CpuKernel for DeformCpu {
    fn name(&self) -> &str {
        OP_NAME
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let mut ins: Vec<&[f32]> = Vec::with_capacity(inputs.len());
        for (i, t) in inputs.iter().enumerate() {
            ins.push(t.expect_f32(&format!("ms_deform_attn input {i}"))?);
        }
        let out = output.expect_f32_mut("ms_deform_attn output")?;
        let res = run_host(&ins, attrs)?;
        if res.len() != out.len() {
            return Err(format!(
                "ms_deform_attn output len {} != expected {}",
                res.len(),
                out.len()
            ));
        }
        out.copy_from_slice(&res);
        Ok(())
    }
}

/// Register the IR shape rule + CPU kernel (idempotent). Metal auto-registers
/// its host-delegate kernel on first lookup (rlx-metal `op_registry`); MLX
/// registers here under the `rlx-mlx` feature; CUDA/WGPU dispatch via the
/// engine's Step table.
pub fn ensure_registered() {
    static ONCE: OnceLock<Mutex<bool>> = OnceLock::new();
    let m = ONCE.get_or_init(|| Mutex::new(false));
    let mut done = m.lock().unwrap();
    if !*done {
        register_op(Arc::new(DeformIr));
        register_cpu_kernel(Arc::new(DeformCpu));
        #[cfg(all(feature = "rlx-mlx", target_os = "macos"))]
        rlx_mlx::ms_deform_attn::register();
        *done = true;
    }
}
