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

// RLX — `Op::Custom` for LLaDA2 group-limited MoE gate (TIDE `LLaDA2MoeGate`).

pub use rlx_cpu::llada2_gate::group_limited_topk;
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};
use rlx_ir::{DType, Node, NodeId, OpExtension, Shape, VjpContext, register_op};
use std::sync::{Arc, Mutex, OnceLock};

pub const OP_NAME: &str = "llada2.group_limited_gate";

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GateAttrs {
    n_group: u32,
    topk_group: u32,
    top_k: u32,
    routed_scaling: f32,
    num_experts: u32,
}

impl GateAttrs {
    fn to_bytes(self) -> Vec<u8> {
        bytemuck::bytes_of(&self).to_vec()
    }

    fn from_bytes(attrs: &[u8]) -> Self {
        if attrs.len() >= std::mem::size_of::<Self>() {
            *bytemuck::from_bytes(&attrs[..std::mem::size_of::<Self>()])
        } else {
            GateAttrs {
                n_group: 8,
                topk_group: 4,
                top_k: 8,
                routed_scaling: 2.5,
                num_experts: 256,
            }
        }
    }
}

struct GroupLimitedGateIr;
impl OpExtension for GroupLimitedGateIr {
    fn name(&self) -> &str {
        OP_NAME
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn infer_shape(&self, inputs: &[&Shape], attrs: &[u8]) -> Shape {
        let rows = inputs[0].dim(0).unwrap_static();
        let a = GateAttrs::from_bytes(attrs);
        let k = a.top_k as usize;
        Shape::new(&[rows, k * 2], DType::F32)
    }
    fn vjp(&self, _node: &Node, _ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        vec![]
    }
}

struct GroupLimitedGateCpu;
impl CpuKernel for GroupLimitedGateCpu {
    fn name(&self) -> &str {
        OP_NAME
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let scores_sigmoid = inputs[0].expect_f32("sigmoid scores")?;
        let scores_route = inputs[1].expect_f32("routing scores")?;
        let out = output.expect_f32_mut("gate out")?;
        let a = GateAttrs::from_bytes(attrs);
        let rows = scores_sigmoid.len() / a.num_experts as usize;
        let _e = a.num_experts as usize;
        let k = a.top_k as usize;
        if scores_route.len() != scores_sigmoid.len() {
            return Err("gate: sigmoid and routing score lengths differ".into());
        }
        if out.len() != rows * k * 2 {
            return Err(format!("output len {} != rows*k*2", out.len()));
        }

        rlx_cpu::llada2_gate::execute_gate_f32(scores_sigmoid, scores_route, out, attrs)
    }
}

pub fn gate_attrs_bytes(
    n_group: usize,
    topk_group: usize,
    top_k: usize,
    routed_scaling: f32,
    num_experts: usize,
) -> Vec<u8> {
    GateAttrs {
        n_group: n_group as u32,
        topk_group: topk_group as u32,
        top_k: top_k as u32,
        routed_scaling,
        num_experts: num_experts as u32,
    }
    .to_bytes()
}

pub fn ensure_group_limited_gate_registered() {
    static ONCE: OnceLock<Mutex<bool>> = OnceLock::new();
    let m = ONCE.get_or_init(|| Mutex::new(false));
    let mut done = m.lock().unwrap();
    if !*done {
        register_op(Arc::new(GroupLimitedGateIr));
        register_cpu_kernel(Arc::new(GroupLimitedGateCpu));
        #[cfg(all(feature = "metal", target_os = "macos"))]
        rlx_metal::llada2_gate::register();
        #[cfg(all(feature = "mlx", target_os = "macos"))]
        rlx_mlx::llada2_gate::register();
        *done = true;
    }
}
