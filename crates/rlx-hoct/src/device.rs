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

//! Multi-device HOCT score-head execution via compiled [`rlx_flow::ModelFlow`].
//!
//! The gated 3D-RoPE transformer body stays on the eager CPU path (parity
//! reference). The LayerNorm→Linear score head is compiled with
//! [`CompileProfile::encoder`](rlx_flow::CompileProfile::encoder) and runs on
//! any RLX backend (CPU / Metal / MLX / CUDA / ROCm / wgpu / Vulkan).

use crate::config::HoctConfig;
use crate::flow::HoctFlow;
use crate::model::{HoctModel, HoctOutput};
use crate::weights::HoctWeights;
use anyhow::{Context, Result, bail};
use ndarray::{Array2, Array3, ArrayView3};
use rlx_core::flow_bridge::compile_options_for_profile;
use rlx_core::flow_util::{attach_built_params, graph_from_built};
use rlx_core::validate_standard_device;
use rlx_flow::CompileProfile;
use rlx_runtime::{CompiledGraph, Device, Session};
use std::path::Path;

/// Eager transformer body + compiled score head on a chosen [`Device`].
///
/// Build with [`HoctDeviceRunner::from_weights`] or [`HoctDeviceRunner::from_parts`],
/// then call [`HoctDeviceRunner::forward`].
pub struct HoctDeviceRunner {
    /// Eager model used for the transformer body.
    pub model: HoctModel,
    /// Backend that owns the compiled score head.
    pub device: Device,
    head: CompiledGraph,
    max_edges: usize,
    batch: usize,
}

impl HoctDeviceRunner {
    /// Load weights and compile the score head for `device`.
    pub fn from_weights(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        validate_standard_device("hoct", device)?;
        if !rlx_runtime::is_available(device) && device != Device::Cpu {
            bail!("device {device:?} is not available in this build");
        }
        let weights = crate::weights::load_hoct_weights(path)?;
        Self::from_parts(weights, HoctConfig::default(), device, 1024, 1)
    }

    /// Compile the score head from in-memory weights (pad to `max_edges`).
    pub fn from_parts(
        weights: HoctWeights,
        cfg: HoctConfig,
        device: Device,
        max_edges: usize,
        batch: usize,
    ) -> Result<Self> {
        validate_standard_device("hoct", device)?;
        let flow = HoctFlow::new(cfg.clone())
            .with_pad(256, max_edges)
            .with_batch(batch);
        let mut wm = HoctFlow::head_weight_map(&weights);
        let built = flow.build_head_flow(&mut wm)?;
        let typed = built.typed_params.clone();
        let (graph, params) = graph_from_built(built)?;
        let opts = compile_options_for_profile(&CompileProfile::encoder(), device);
        let mut compiled = Session::new(device).compile_with(graph, &opts);
        attach_built_params(&mut compiled, params, &typed);
        Ok(Self {
            model: HoctModel::new(cfg, weights),
            device,
            head: compiled,
            max_edges,
            batch,
        })
    }

    /// Backend this runner was compiled for.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Full forward: eager transformer body, compiled score head on `self.device`.
    pub fn forward(
        &mut self,
        node_features: &ArrayView3<f32>,
        node_pos: &ArrayView3<f32>,
        edge_pos: &ArrayView3<f32>,
        edge_indices: &Array3<i64>,
        node_mask: &Array2<bool>,
        edge_mask: &Array2<bool>,
    ) -> Result<HoctOutput> {
        let body = self.model.forward(
            node_features,
            node_pos,
            edge_pos,
            edge_indices,
            node_mask,
            edge_mask,
        );
        let e = body.edge_hidden.len_of(ndarray::Axis(1));
        if e > self.max_edges {
            bail!("edge count {e} exceeds compiled pad {}", self.max_edges);
        }
        let logits = self.run_head(&body.edge_hidden)?;
        Ok(HoctOutput {
            edge_logits: logits,
            node_hidden: body.node_hidden,
            edge_hidden: body.edge_hidden,
            orphan_logits: body.orphan_logits,
        })
    }

    /// Run only the compiled score head on padded `edge_h` `[B, E, C]`.
    pub fn run_head(&mut self, edge_h: &Array3<f32>) -> Result<Array3<f32>> {
        let b = edge_h.len_of(ndarray::Axis(0));
        let e = edge_h.len_of(ndarray::Axis(1));
        let c = edge_h.len_of(ndarray::Axis(2));
        if b != self.batch {
            bail!("batch {b} != compiled batch {}", self.batch);
        }
        let mut padded = vec![0.0f32; self.batch * self.max_edges * c];
        for bi in 0..b {
            for ei in 0..e {
                for k in 0..c {
                    padded[(bi * self.max_edges + ei) * c + k] = edge_h[[bi, ei, k]];
                }
            }
        }
        let outs = self.head.run(&[("edge_h", padded.as_slice())]);
        let flat = outs
            .into_iter()
            .next()
            .context("hoct head produced no output")?;
        // Output is `[B, E_max, 1]` — crop to live edges.
        let mut logits = Array3::<f32>::zeros((b, e, 1));
        for bi in 0..b {
            for ei in 0..e {
                logits[[bi, ei, 0]] = flat[(bi * self.max_edges + ei) * 1];
            }
        }
        Ok(logits)
    }
}

/// Devices to exercise in backend matrix tests (feature- + availability-gated).
#[allow(unused_mut)] // pushes are behind backend feature cfgs
pub fn parity_backends() -> Vec<Device> {
    let mut out = vec![Device::Cpu];
    #[cfg(feature = "metal")]
    if rlx_runtime::is_available(Device::Metal) {
        out.push(Device::Metal);
    }
    #[cfg(feature = "mlx")]
    if rlx_runtime::is_available(Device::Mlx) {
        out.push(Device::Mlx);
    }
    #[cfg(feature = "gpu")]
    if rlx_runtime::is_available(Device::Gpu) {
        out.push(Device::Gpu);
    }
    #[cfg(feature = "cuda")]
    if rlx_runtime::is_available(Device::Cuda) {
        out.push(Device::Cuda);
    }
    #[cfg(feature = "rocm")]
    if rlx_runtime::is_available(Device::Rocm) {
        out.push(Device::Rocm);
    }
    #[cfg(feature = "vulkan")]
    if rlx_runtime::is_available(Device::Vulkan) {
        out.push(Device::Vulkan);
    }
    out
}

/// Short label for logging (`"cuda"`, `"metal"`, …).
pub fn device_label(d: Device) -> &'static str {
    match d {
        Device::Cpu => "cpu",
        Device::Metal => "metal",
        Device::Mlx => "mlx",
        Device::Gpu => "wgpu",
        Device::Cuda => "cuda",
        Device::Rocm => "rocm",
        Device::Vulkan => "vulkan",
        _ => "other",
    }
}
