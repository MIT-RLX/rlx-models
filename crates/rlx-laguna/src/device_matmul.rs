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

//! Cached `Op::DequantMatMul` sessions for Laguna packed mats on Metal / MLX / …
//!
//! Used by [`crate::packed_forward`] when `--device` is set. Graphs are keyed by
//! `(m, k, n, scheme, w_bytes)`; identical weight pointers skip re-upload.

use anyhow::{Result, bail};
use rlx_core::flow_bridge::{
    compile_options_for_packed_gguf_prefill, packed_gguf_compile_guard,
    packed_gguf_execution_device,
};
use rlx_ir::{DType, Graph, Op, Shape, quant::QuantScheme};
use rlx_runtime::{CompiledGraph, Device, Session, is_available};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct ShapeKey {
    m: u32,
    k: u32,
    n: u32,
    scheme: u8,
    /// Packed weight byte length (Metal/MLX compile-time param shape).
    w_bytes: u32,
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

/// Persistent DequantMatMul runner for one RLX device.
pub struct DeviceMatmul {
    device: Device,
    exec: Device,
    cache: HashMap<ShapeKey, CompiledGraph>,
    /// Last uploaded weight pointer per shape (skip redundant `set_param`).
    uploaded_ptr: HashMap<ShapeKey, usize>,
}

impl DeviceMatmul {
    pub fn try_new(device: Device) -> Result<Self> {
        if !is_available(device) {
            bail!("device {device:?} is not available on this host");
        }
        let exec = packed_gguf_execution_device(device);
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
            m: m as u32,
            k: k as u32,
            n: n as u32,
            scheme: tag,
            w_bytes: w_bytes.len() as u32,
        };
        if !self.cache.contains_key(&key) {
            let g = build_graph(m, k, n, w_bytes.len(), scheme);
            let opts = compile_options_for_packed_gguf_prefill(self.exec);
            let compiled = packed_gguf_compile_guard(self.exec, || {
                Session::new(self.exec).compile_with(g, &opts)
            });
            self.cache.insert(key, compiled);
            self.uploaded_ptr.remove(&key);
        }
        let ptr = w_bytes.as_ptr() as usize;
        let need_upload = self.uploaded_ptr.get(&key).copied() != Some(ptr);
        // Keep MPSGraph off for every run (guard only wraps compile).
        if self.exec == Device::Metal {
            rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
        }
        {
            let compiled = self
                .cache
                .get_mut(&key)
                .expect("DeviceMatmul cache insert");
            if need_upload {
                compiled.set_param_typed("w", w_bytes, DType::U8);
            }
            let out = packed_gguf_compile_guard(self.exec, || compiled.run(&[("x", x)]));
            if self.exec == Device::Metal {
                rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
            }
            if need_upload {
                self.uploaded_ptr.insert(key, ptr);
            }
            let y = out.into_iter().next().unwrap_or_default();
            if y.len() != m * n {
                bail!(
                    "DeviceMatmul output len {} != m*n={} (device={:?})",
                    y.len(),
                    m * n,
                    self.device
                );
            }
            Ok(y)
        }
    }
}

fn build_graph(m: usize, k: usize, n: usize, w_len: usize, scheme: QuantScheme) -> Graph {
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

/// Parse `cpu|host|metal|mlx|gpu|auto` for Laguna CLI.
pub fn parse_device(s: &str) -> Result<Option<Device>> {
    match s.trim().to_ascii_lowercase().as_str() {
        "cpu" | "host" => Ok(None),
        "metal" => Ok(Some(Device::Metal)),
        "mlx" => Ok(Some(Device::Mlx)),
        "gpu" | "wgpu" => Ok(Some(Device::Gpu)),
        "auto" => {
            if is_available(Device::Metal) {
                Ok(Some(Device::Metal))
            } else if is_available(Device::Mlx) {
                Ok(Some(Device::Mlx))
            } else {
                Ok(None)
            }
        }
        other => bail!("unknown --device {other} (cpu|metal|mlx|gpu|auto)"),
    }
}
