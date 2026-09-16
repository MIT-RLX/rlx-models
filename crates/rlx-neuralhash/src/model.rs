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

//! The compiled NeuralHash descriptor network: `[1, 3, 360, 360]` → 128 floats.
//!
//! Entirely native. The architecture comes from Apple's Espresso container via
//! [`crate::spec`], the graph from [`crate::flow`], and execution from
//! `rlx-runtime` — there is no ONNX runtime on this path, and the `onnx-parity`
//! feature that can import `model.onnx` exists only to diff against this.

use anyhow::{Context, Result, anyhow, bail, ensure};
use rlx_core::flow_bridge::compile_options_for_profile;
use rlx_core::flow_util::attach_built_params;
use rlx_core::weight_map::WeightMap;
use rlx_flow::CompileProfile;
use rlx_runtime::{CompiledGraph, Device, Session};
use std::path::Path;

use crate::espresso::EspressoNet;
use crate::preprocess::{INPUT_ELEMS, INPUT_SIZE};
use crate::seed::EMBED_DIM;
use crate::spec::NeuralHashSpec;

/// A compiled NeuralHash descriptor network.
pub struct NeuralHashModel {
    graph: CompiledGraph,
    spec: NeuralHashSpec,
    device: Device,
    graph_nodes: usize,
}

impl NeuralHashModel {
    /// Build from `NeuralHashv3b_fp16-current.espresso.net` (its `.shape` and
    /// `.weights` siblings are located automatically).
    pub fn open_espresso(net: impl AsRef<Path>, device: Device) -> Result<Self> {
        Self::open_espresso_fused(net, device, true)
    }

    /// [`Self::open_espresso`] with the fusion pass toggled (see
    /// [`Self::from_espresso_fused`]).
    pub fn open_espresso_fused(net: impl AsRef<Path>, device: Device, fuse: bool) -> Result<Self> {
        let net = net.as_ref();
        ensure!(
            net.is_file(),
            "neuralhash: espresso net not found at {}. On macOS the shipping model is \
             /System/Library/Frameworks/Vision.framework/Versions/A/Resources/\
             NeuralHashv3b_fp16-current.espresso.net (with its .shape and .weights \
             siblings); earlier OS versions name it NeuralHashv3b-current. See the README.",
            net.display()
        );
        let net = EspressoNet::open(net)?;
        Self::from_espresso_fused(&net, device, fuse)
    }

    /// Build from an already-parsed Espresso container.
    pub fn from_espresso(net: &EspressoNet, device: Device) -> Result<Self> {
        Self::from_espresso_fused(net, device, true)
    }

    /// Build with the hard-swish / hard-sigmoid fusion pass toggled.
    ///
    /// `fuse = false` emits Espresso's literal four-op chains. Both forms must
    /// hash identically — that is what makes fusion safe to enable by default —
    /// so this exists for A/B timing and for the parity test that pins them.
    pub fn from_espresso_fused(net: &EspressoNet, device: Device, fuse: bool) -> Result<Self> {
        let spec = if fuse {
            NeuralHashSpec::from_espresso(net)
        } else {
            NeuralHashSpec::from_espresso_unfused(net)
        }
        .context("deriving the native architecture from the Espresso layer list")?;
        let weights = crate::weights::from_espresso(net).context("extracting Espresso weights")?;
        Self::from_spec(spec, weights, device)
    }

    /// Build from a previously exported spec (`--export-spec`) plus weights.
    pub fn from_spec(spec: NeuralHashSpec, mut weights: WeightMap, device: Device) -> Result<Self> {
        rlx_core::validate_standard_device("neuralhash", device)?;
        spec.validate_neuralhash_io()?;
        let (graph, params) = crate::flow::build_graph_with(
            &spec,
            &mut weights,
            crate::flow::LoweringOpts::for_device(device),
        )
        .context("building the native rlx graph")?;
        let graph_nodes = graph.nodes().len();
        let opts = compile_options_for_profile(&CompileProfile::encoder(), device);
        let mut graph = Session::new(device).compile_with(graph, &opts);
        attach_built_params(&mut graph, params, &[]);
        Ok(Self {
            graph,
            spec,
            device,
            graph_nodes,
        })
    }

    /// The backend this model was compiled for.
    pub fn device(&self) -> Device {
        self.device
    }

    /// The architecture this model was built from.
    pub fn spec(&self) -> &NeuralHashSpec {
        &self.spec
    }

    /// Node count of the compiled graph.
    ///
    /// This is the number the runtime actually executes, and it is *not* the
    /// spec op count: composites (instance norm, broadcasts, bias adds) expand
    /// into several nodes each, so this is the honest measure of how much the
    /// emitter is asking the backend to do.
    pub fn graph_nodes(&self) -> usize {
        self.graph_nodes
    }

    /// Run one `[3, 360, 360]` NCHW tensor through the network.
    ///
    /// Returns the 128-float descriptor — the reference's `outs[0].flatten()`.
    pub fn embed(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            input.len() == INPUT_ELEMS,
            "neuralhash: expected a [3, {INPUT_SIZE}, {INPUT_SIZE}] tensor ({INPUT_ELEMS} floats), got {}",
            input.len()
        );
        let outs = self.graph.run(&[(self.spec.input.as_str(), input)]);
        let out = outs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("neuralhash: graph produced no output"))?;
        ensure!(
            out.len() == EMBED_DIM,
            "neuralhash: expected a {EMBED_DIM}-float descriptor, got {}",
            out.len()
        );
        if let Some(bad) = out.iter().position(|v| !v.is_finite()) {
            bail!(
                "neuralhash: descriptor element {bad} is {} on {:?} — backend numerics failure",
                out[bad],
                self.device
            );
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `NeuralHashModel` holds a `CompiledGraph`, which is not `Debug`, so
    /// `unwrap_err` is unavailable — take the error out of the `Result` instead.
    fn err<T>(r: Result<T>) -> String {
        format!("{:#}", r.err().expect("expected an error"))
    }

    #[test]
    fn missing_espresso_net_is_a_clear_error() {
        let s = err(NeuralHashModel::open_espresso(
            "/nonexistent/NeuralHashv3b.espresso.net",
            Device::Cpu,
        ));
        assert!(s.contains("espresso net not found"), "{s}");
        assert!(s.contains("NeuralHashv3b"), "{s}");
    }

    #[test]
    fn wrong_io_shape_is_rejected_before_compiling() {
        // A spec that is structurally fine but not NeuralHash-shaped.
        let spec = NeuralHashSpec {
            name: "toy".into(),
            input: "i".into(),
            input_shape: [1, 3, 64, 64],
            output: "o".into(),
            ops: vec![],
        };
        let wm = WeightMap::from_tensors(Default::default());
        let e = err(NeuralHashModel::from_spec(spec, wm, Device::Cpu));
        assert!(e.contains("spec input is [1, 3, 64, 64]"), "{e}");
    }
}
