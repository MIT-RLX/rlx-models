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

//! Cached `Op::DequantMatMul` / `Op::DequantGroupedMatMul` sessions for Laguna
//! packed mats on Metal / MLX / …
//!
//! Used by [`crate::packed_forward`] when `--device` is set. Graphs are keyed by
//! shape; identical weight pointers skip re-upload so a full MoE expert stack
//! stays resident across decode steps.

use anyhow::{Result, bail};
use rlx_core::flow_bridge::{
    compile_options_for_packed_gguf_prefill, packed_gguf_compile_guard,
    packed_gguf_execution_device,
};
use rlx_ir::{DType, Graph, GraphExt, Op, Shape, op::BinaryOp, quant::QuantScheme};
use rlx_runtime::{CompiledGraph, Device, Session, is_available};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum GraphKind {
    Plain,
    Grouped,
    /// gate + up + silu* + down in one compiled graph (one GPU sync).
    GroupedSwiglu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct ShapeKey {
    kind: GraphKind,
    m: u32,
    k: u32,
    n: u32,
    /// Intermediate width for SwiGLU fused MoE (0 otherwise).
    inter: u32,
    scheme: u8,
    scheme_up: u8,
    scheme_down: u8,
    /// Packed weight byte length (Metal/MLX compile-time param shape).
    w_bytes: u32,
    w_up_bytes: u32,
    w_down_bytes: u32,
    /// 0 = plain DequantMatMul; otherwise num_experts for grouped MoE.
    num_experts: u32,
    /// Weight identity — one compiled graph + resident buffer per MoE layer
    /// so decode does not re-upload the full expert stack every step.
    w_ptr: usize,
    w_up_ptr: usize,
    w_down_ptr: usize,
}

fn scheme_tag(scheme: QuantScheme) -> u8 {
    match scheme {
        QuantScheme::GgufQ4K => 1,
        QuantScheme::GgufQ5K => 2,
        QuantScheme::GgufQ6K => 3,
        QuantScheme::GgufQ8K => 4,
        QuantScheme::GgufQ2K => 5,
        QuantScheme::GgufQ3K => 6,
        QuantScheme::GgufQ4_0 => 7,
        QuantScheme::GgufQ8_0 => 8,
        _ => 255,
    }
}

#[derive(Clone, Copy, Debug)]
enum UploadedW {
    One(usize),
    Three([usize; 3]),
}

/// Persistent DequantMatMul / DequantGroupedMatMul runner for one RLX device.
pub struct DeviceMatmul {
    device: Device,
    exec: Device,
    cache: HashMap<ShapeKey, CompiledGraph>,
    /// Last uploaded weight pointer(s) per shape (skip redundant `set_param`).
    uploaded_ptr: HashMap<ShapeKey, UploadedW>,
}

impl DeviceMatmul {
    pub fn try_new(device: Device) -> Result<Self> {
        if !is_available(device) {
            bail!("device {device:?} is not available on this host");
        }
        let exec = packed_gguf_execution_device(device);
        // Packaged GGUF DequantMatMul path; set once (not per-run) so decode
        // does not thrash env lookups around every expert GEMM.
        if exec == Device::Metal {
            rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
        }
        Ok(Self {
            device,
            exec,
            cache: HashMap::new(),
            uploaded_ptr: HashMap::new(),
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn exec(&self) -> Device {
        self.exec
    }

    /// `y[m,n] = x[m,k] @ W^T` with packed GGUF `W` stored `[n,k]`.
    pub fn matmul(
        &mut self,
        x: &[f32],
        w_bytes: &[u8],
        m: usize,
        k: usize,
        n: usize,
        scheme: QuantScheme,
    ) -> Result<Vec<f32>> {
        if x.len() != m * k {
            bail!("DeviceMatmul x len {} != m*k={}", x.len(), m * k);
        }
        let tag = scheme_tag(scheme);
        if tag == 255 {
            bail!("DeviceMatmul: unsupported scheme {scheme:?}");
        }
        if w_bytes.len() > u32::MAX as usize {
            bail!("DeviceMatmul: weight bytes too large");
        }
        let key = ShapeKey {
            kind: GraphKind::Plain,
            m: m as u32,
            k: k as u32,
            n: n as u32,
            inter: 0,
            scheme: tag,
            scheme_up: 0,
            scheme_down: 0,
            w_bytes: w_bytes.len() as u32,
            w_up_bytes: 0,
            w_down_bytes: 0,
            num_experts: 0,
            w_ptr: w_bytes.as_ptr() as usize,
            w_up_ptr: 0,
            w_down_ptr: 0,
        };
        self.ensure_compiled(key, || build_matmul_graph(m, k, n, w_bytes.len(), scheme))?;
        self.run_with_w(key, &[("w", w_bytes)], &[("x", x)], m * n)
    }

    /// Batched MoE: `out[i] = x[i] @ expert[expert_idx[i]]` with the full packed
    /// expert stack resident on device (`Op::DequantGroupedMatMul`).
    ///
    /// `w_bytes` holds `num_experts` contiguous slabs (same layout as
    /// [`rlx_cpu::gguf_matmul::gguf_grouped_matmul_bt_fused`]).
    pub fn grouped_matmul(
        &mut self,
        x: &[f32],
        w_bytes: &[u8],
        expert_idx: &[f32],
        m: usize,
        k: usize,
        n: usize,
        num_experts: usize,
        scheme: QuantScheme,
    ) -> Result<Vec<f32>> {
        if x.len() != m * k {
            bail!("DeviceMatmul grouped x len {} != m*k={}", x.len(), m * k);
        }
        if expert_idx.len() != m {
            bail!(
                "DeviceMatmul grouped expert_idx len {} != m={m}",
                expert_idx.len()
            );
        }
        if num_experts == 0 {
            bail!("DeviceMatmul grouped: num_experts == 0");
        }
        let tag = scheme_tag(scheme);
        if tag == 255 {
            bail!("DeviceMatmul: unsupported scheme {scheme:?}");
        }
        if w_bytes.len() > u32::MAX as usize {
            bail!("DeviceMatmul: weight bytes too large");
        }
        let key = ShapeKey {
            kind: GraphKind::Grouped,
            m: m as u32,
            k: k as u32,
            n: n as u32,
            inter: 0,
            scheme: tag,
            scheme_up: 0,
            scheme_down: 0,
            w_bytes: w_bytes.len() as u32,
            w_up_bytes: 0,
            w_down_bytes: 0,
            num_experts: num_experts as u32,
            w_ptr: w_bytes.as_ptr() as usize,
            w_up_ptr: 0,
            w_down_ptr: 0,
        };
        self.ensure_compiled(key, || build_grouped_graph(m, k, n, w_bytes.len(), scheme))?;
        self.run_with_w(
            key,
            &[("w", w_bytes)],
            &[("x", x), ("idx", expert_idx)],
            m * n,
        )
    }

    /// One-shot MoE expert SwiGLU: `down(silu(gate(x)) * up(x))` with resident
    /// expert stacks. One compiled graph → one GPU sync instead of three.
    #[allow(clippy::too_many_arguments)]
    pub fn grouped_swiglu(
        &mut self,
        x: &[f32],
        expert_idx: &[f32],
        w_gate: &[u8],
        gate_scheme: QuantScheme,
        w_up: &[u8],
        up_scheme: QuantScheme,
        w_down: &[u8],
        down_scheme: QuantScheme,
        m: usize,
        h: usize,
        inter: usize,
        num_experts: usize,
    ) -> Result<Vec<f32>> {
        if x.len() != m * h {
            bail!("DeviceMatmul swiglu x len {} != m*h={}", x.len(), m * h);
        }
        if expert_idx.len() != m {
            bail!(
                "DeviceMatmul swiglu expert_idx len {} != m={m}",
                expert_idx.len()
            );
        }
        if num_experts == 0 {
            bail!("DeviceMatmul swiglu: num_experts == 0");
        }
        let tg = scheme_tag(gate_scheme);
        let tu = scheme_tag(up_scheme);
        let td = scheme_tag(down_scheme);
        if tg == 255 || tu == 255 || td == 255 {
            bail!("DeviceMatmul swiglu: unsupported scheme");
        }
        for (name, bytes) in [("gate", w_gate), ("up", w_up), ("down", w_down)] {
            if bytes.len() > u32::MAX as usize {
                bail!("DeviceMatmul swiglu: {name} weight bytes too large");
            }
        }
        let key = ShapeKey {
            kind: GraphKind::GroupedSwiglu,
            m: m as u32,
            k: h as u32,
            n: h as u32,
            inter: inter as u32,
            scheme: tg,
            scheme_up: tu,
            scheme_down: td,
            w_bytes: w_gate.len() as u32,
            w_up_bytes: w_up.len() as u32,
            w_down_bytes: w_down.len() as u32,
            num_experts: num_experts as u32,
            w_ptr: w_gate.as_ptr() as usize,
            w_up_ptr: w_up.as_ptr() as usize,
            w_down_ptr: w_down.as_ptr() as usize,
        };
        self.ensure_compiled(key, || {
            build_swiglu_graph(
                m,
                h,
                inter,
                w_gate.len(),
                w_up.len(),
                w_down.len(),
                gate_scheme,
                up_scheme,
                down_scheme,
            )
        })?;
        self.run_with_w(
            key,
            &[("w_gate", w_gate), ("w_up", w_up), ("w_down", w_down)],
            &[("x", x), ("idx", expert_idx)],
            m * h,
        )
    }

    fn ensure_compiled(&mut self, key: ShapeKey, build: impl FnOnce() -> Graph) -> Result<()> {
        if !self.cache.contains_key(&key) {
            let g = build();
            let opts = compile_options_for_packed_gguf_prefill(self.exec);
            let compiled = packed_gguf_compile_guard(self.exec, || {
                Session::new(self.exec).compile_with(g, &opts)
            });
            self.cache.insert(key, compiled);
            self.uploaded_ptr.remove(&key);
        }
        Ok(())
    }

    fn run_with_w(
        &mut self,
        key: ShapeKey,
        weights: &[(&str, &[u8])],
        feeds: &[(&str, &[f32])],
        expect_len: usize,
    ) -> Result<Vec<f32>> {
        let ptrs: Vec<usize> = weights.iter().map(|(_, b)| b.as_ptr() as usize).collect();
        let need_upload = match self.uploaded_ptr.get(&key) {
            Some(UploadedW::One(p)) if weights.len() == 1 => *p != ptrs[0],
            Some(UploadedW::Three(p)) if weights.len() == 3 => p[..] != ptrs[..],
            Some(_) => true,
            None => true,
        };
        let compiled = self.cache.get_mut(&key).expect("DeviceMatmul cache insert");
        if need_upload {
            for &(name, bytes) in weights {
                compiled.set_param_typed(name, bytes, DType::U8);
            }
        }
        let out = packed_gguf_compile_guard(self.exec, || compiled.run(feeds));
        if need_upload {
            let uploaded = if weights.len() == 3 {
                UploadedW::Three([ptrs[0], ptrs[1], ptrs[2]])
            } else {
                UploadedW::One(ptrs[0])
            };
            self.uploaded_ptr.insert(key, uploaded);
        }
        let y = out.into_iter().next().unwrap_or_default();
        if y.len() != expect_len {
            bail!(
                "DeviceMatmul output len {} != {expect_len} (device={:?})",
                y.len(),
                self.device
            );
        }
        Ok(y)
    }
}

fn build_matmul_graph(m: usize, k: usize, n: usize, w_len: usize, scheme: QuantScheme) -> Graph {
    let mut g = Graph::new("laguna_dequant_matmul");
    // Match packed-GGUF prefill layout used by backend_bench / Qwen35.
    let x_id = g.input("x", Shape::new(&[1, m, k], DType::F32));
    let w_id = g.param("w", Shape::new(&[w_len], DType::U8));
    let y_id = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_id, w_id],
        Shape::new(&[1, m, n], DType::F32),
    );
    g.set_outputs(vec![y_id]);
    g
}

fn build_grouped_graph(m: usize, k: usize, n: usize, w_len: usize, scheme: QuantScheme) -> Graph {
    let mut g = Graph::new("laguna_dequant_grouped_matmul");
    let x_id = g.input("x", Shape::new(&[m, k], DType::F32));
    let w_id = g.param("w", Shape::new(&[w_len], DType::U8));
    let idx_id = g.input("idx", Shape::new(&[m], DType::F32));
    let y_id = g.add_node(
        Op::DequantGroupedMatMul { scheme },
        vec![x_id, w_id, idx_id],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y_id]);
    g
}

fn build_swiglu_graph(
    m: usize,
    h: usize,
    inter: usize,
    wg_len: usize,
    wu_len: usize,
    wd_len: usize,
    gate_scheme: QuantScheme,
    up_scheme: QuantScheme,
    down_scheme: QuantScheme,
) -> Graph {
    let mut g = Graph::new("laguna_moe_grouped_swiglu");
    let x_id = g.input("x", Shape::new(&[m, h], DType::F32));
    let idx_id = g.input("idx", Shape::new(&[m], DType::F32));
    let wg = g.param("w_gate", Shape::new(&[wg_len], DType::U8));
    let wu = g.param("w_up", Shape::new(&[wu_len], DType::U8));
    let wd = g.param("w_down", Shape::new(&[wd_len], DType::U8));
    let mid_shape = Shape::new(&[m, inter], DType::F32);
    let gate = g.add_node(
        Op::DequantGroupedMatMul {
            scheme: gate_scheme,
        },
        vec![x_id, wg, idx_id],
        mid_shape.clone(),
    );
    let up = g.add_node(
        Op::DequantGroupedMatMul { scheme: up_scheme },
        vec![x_id, wu, idx_id],
        mid_shape.clone(),
    );
    let gate_s = g.silu(gate);
    let mid = g.binary(BinaryOp::Mul, gate_s, up, mid_shape);
    let down = g.add_node(
        Op::DequantGroupedMatMul {
            scheme: down_scheme,
        },
        vec![mid, wd, idx_id],
        Shape::new(&[m, h], DType::F32),
    );
    g.set_outputs(vec![down]);
    g
}

/// Parse `cpu|host|metal|mlx|gpu|wgpu|coreml|ane|auto` for Laguna CLI.
///
/// `auto` prefers the host fused GGUF path — Metal/MLX DequantMatMul is slower
/// for MoE decode / short chat prefills on current Apple Silicon measurements.
pub fn parse_device(s: &str) -> Result<Option<Device>> {
    match s.trim().to_ascii_lowercase().as_str() {
        "cpu" | "host" => Ok(None),
        "metal" => Ok(Some(Device::Metal)),
        "mlx" => Ok(Some(Device::Mlx)),
        "gpu" | "wgpu" => Ok(Some(Device::Gpu)),
        "coreml" | "ane" | "neural-engine" => Ok(Some(Device::Ane)),
        "auto" => Ok(None),
        other => bail!("unknown --device {other} (cpu|metal|mlx|gpu|wgpu|coreml|auto)"),
    }
}
