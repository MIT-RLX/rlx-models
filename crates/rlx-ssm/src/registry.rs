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

use std::sync::Arc;

use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};
use rlx_ir::op_registry::{OpExtension, register_op};
use rlx_ir::{DType, Dim, Shape};

use crate::kernels;

fn dim(d: Dim) -> usize {
    d.unwrap_static()
}

struct LightningStepOp;
struct LfmStepOp;
struct Mamba1StepOp;
struct Mamba2StepOp;

impl OpExtension for LightningStepOp {
    fn name(&self) -> &str {
        "lightning_attention_step"
    }

    fn num_inputs(&self) -> usize {
        6
    }

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        let b = dim(inputs[0].dims()[0]);
        let one = dim(inputs[0].dims()[1]);
        let h = dim(inputs[0].dims()[2]);
        let n = dim(inputs[0].dims()[3]);
        Shape::new(&[b, one, h, n + n * n], DType::F32)
    }
}

impl OpExtension for LfmStepOp {
    fn name(&self) -> &str {
        "lfm_ssm_step"
    }

    fn num_inputs(&self) -> usize {
        6
    }

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        let b = dim(inputs[0].dims()[0]);
        let one = dim(inputs[0].dims()[1]);
        let c = dim(inputs[0].dims()[2]);
        let n = dim(inputs[5].dims()[2]);
        Shape::new(&[b, one, c + c * n], DType::F32)
    }
}

impl OpExtension for Mamba1StepOp {
    fn name(&self) -> &str {
        "mamba1_step"
    }

    fn num_inputs(&self) -> usize {
        7
    }

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        let b = dim(inputs[0].dims()[0]);
        let h = dim(inputs[0].dims()[1]);
        let n = dim(inputs[6].dims()[2]);
        Shape::new(&[b, h + h * n], DType::F32)
    }
}

impl OpExtension for Mamba2StepOp {
    fn name(&self) -> &str {
        "mamba2_step"
    }

    fn num_inputs(&self) -> usize {
        7
    }

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        let b = dim(inputs[0].dims()[0]);
        let h = dim(inputs[0].dims()[2]);
        let n = dim(inputs[6].dims()[2]);
        Shape::new(&[b, h + h * n], DType::F32)
    }
}

struct LightningCpu;
struct LfmCpu;
struct Mamba1Cpu;
struct Mamba2Cpu;

impl CpuKernel for LightningCpu {
    fn name(&self) -> &str {
        "lightning_attention_step"
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let q = inputs[0].expect_f32("q")?;
        let k = inputs[1].expect_f32("k")?;
        let v = inputs[2].expect_f32("v")?;
        let gate = inputs[3].expect_f32("gate")?;
        let beta = inputs[4].expect_f32("beta")?;
        let state = inputs[5].expect_f32("state")?;
        let out = output.expect_f32_mut("out")?;
        let shape = inputs[0].shape();
        let b = dim(shape.dims()[0]);
        let h = dim(shape.dims()[2]);
        let n = dim(shape.dims()[3]);
        let y_len = b * h * n;
        let mut y = vec![0f32; y_len];
        let mut state_out = vec![0f32; b * h * n * n];
        kernels::execute_lightning_attention_step_f32(
            q,
            k,
            v,
            gate,
            beta,
            state,
            &mut y,
            &mut state_out,
            b,
            h,
            n,
        )
        .map_err(|e| e.to_string())?;
        for bi in 0..b {
            for hi in 0..h {
                let base = (bi * h + hi) * (n + n * n);
                out[base..base + n]
                    .copy_from_slice(&y[bi * h * n + hi * n..bi * h * n + (hi + 1) * n]);
                out[base + n..base + n + n * n]
                    .copy_from_slice(&state_out[bi * h * n * n + hi * n * n..][..n * n]);
            }
        }
        Ok(())
    }
}

impl CpuKernel for LfmCpu {
    fn name(&self) -> &str {
        "lfm_ssm_step"
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f32("x")?;
        let a = inputs[1].expect_f32("a")?;
        let b_in = inputs[2].expect_f32("b")?;
        let c = inputs[3].expect_f32("c")?;
        let gate = inputs[4].expect_f32("gate")?;
        let state = inputs[5].expect_f32("state")?;
        let out = output.expect_f32_mut("out")?;
        let shape = inputs[0].shape();
        let batch = dim(shape.dims()[0]);
        let channels = dim(shape.dims()[2]);
        let n = dim(inputs[5].shape().dims()[2]);
        kernels::execute_lfm_ssm_step_f32(x, a, b_in, c, gate, state, out, batch, channels, n)
            .map_err(|e| e.to_string())
    }
}

impl CpuKernel for Mamba1Cpu {
    fn name(&self) -> &str {
        "mamba1_step"
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f32("x")?;
        let dt = inputs[1].expect_f32("dt")?;
        let a_log = inputs[2].expect_f32("a_log")?;
        let b_in = inputs[3].expect_f32("b")?;
        let c = inputs[4].expect_f32("c")?;
        let d = inputs[5].expect_f32("d")?;
        let state = inputs[6].expect_f32("state")?;
        let out = output.expect_f32_mut("out")?;
        let shape = inputs[0].shape();
        let batch = dim(shape.dims()[0]);
        let heads = dim(shape.dims()[1]);
        let n = dim(inputs[6].shape().dims()[2]);
        kernels::execute_mamba1_step_f32(x, dt, a_log, b_in, c, d, state, out, batch, heads, n)
            .map_err(|e| e.to_string())
    }
}

impl CpuKernel for Mamba2Cpu {
    fn name(&self) -> &str {
        "mamba2_step"
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f32("x")?;
        let dt = inputs[1].expect_f32("dt")?;
        let a_log = inputs[2].expect_f32("a_log")?;
        let b_in = inputs[3].expect_f32("b")?;
        let c = inputs[4].expect_f32("c")?;
        let d = inputs[5].expect_f32("d")?;
        let state = inputs[6].expect_f32("state")?;
        let out = output.expect_f32_mut("out")?;
        let shape = inputs[0].shape();
        let batch = dim(shape.dims()[0]);
        let heads = dim(shape.dims()[2]);
        let n = dim(inputs[6].shape().dims()[2]);
        kernels::execute_mamba2_step_f32(x, dt, a_log, b_in, c, d, state, out, batch, heads, n)
            .map_err(|e| e.to_string())
    }
}

/// Register IR shape rules + CPU kernels for SSM custom ops.
pub fn register_ir_ops() {
    register_op(Arc::new(LightningStepOp));
    register_op(Arc::new(LfmStepOp));
    register_op(Arc::new(Mamba1StepOp));
    register_op(Arc::new(Mamba2StepOp));
    register_cpu_kernel(Arc::new(LightningCpu));
    register_cpu_kernel(Arc::new(LfmCpu));
    register_cpu_kernel(Arc::new(Mamba1Cpu));
    register_cpu_kernel(Arc::new(Mamba2Cpu));
}

/// Compatibility shim for tests that used `rlx_cpu::ssm_kernels::register_ssm_kernels`.
pub mod ssm_kernels {
    pub use crate::kernels::{
        execute_lfm_ssm_step_f32, execute_lightning_attention_step_f32, execute_mamba2_step_f32,
    };

    pub fn register_ssm_kernels() {
        super::register_ir_ops();
    }
}
