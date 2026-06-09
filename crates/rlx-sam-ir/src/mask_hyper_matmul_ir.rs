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

//! Mask logits from hypernetwork coeffs × upscaled embedding planes (`hyper @ up`).

use anyhow::Result;
use rlx_flow::CompileProfile;
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Graph, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device};

/// `masks = hyper_in @ up2` with `hyper_in` `[num_masks, q8]`, `up2` `[q8, spat]` NCHW-flat.
pub struct MaskHyperMatmulCompiled {
    graph: CompiledGraph,
    pub num_masks: usize,
    pub q8: usize,
    pub spat: usize,
}

impl MaskHyperMatmulCompiled {
    pub fn compile(num_masks: usize, q8: usize, grid: usize, device: Device) -> Result<Self> {
        Self::compile_with_profile(num_masks, q8, grid, device, &CompileProfile::sam_encoder())
    }

    pub fn compile_with_profile(
        num_masks: usize,
        q8: usize,
        grid: usize,
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let spat = (4 * grid) * (4 * grid);
        let f = DType::F32;
        let mut hir = HirModule::new("mask_hyper_matmul");
        let mut g = HirMut::new(&mut hir);

        let hyper = g.input("hyper", Shape::new(&[num_masks, q8], f));
        let up = g.input("up", Shape::new(&[q8, spat], f));
        let masks = g.mm(hyper, up);

        hir.set_outputs(vec![masks]);
        let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("{e}"))?;
        let compiled = rlx_core::flow_bridge::compile_graph_with_profile(device, graph, profile)?;
        Ok(Self {
            graph: compiled,
            num_masks,
            q8,
            spat,
        })
    }

    pub fn run(&mut self, hyper_in: &[f32], up2: &[f32], masks_out: &mut [f32]) -> Result<()> {
        let hn = self.num_masks * self.q8;
        let un = self.q8 * self.spat;
        let mn = self.num_masks * self.spat;
        anyhow::ensure!(hyper_in.len() == hn, "hyper len {} ≠ {hn}", hyper_in.len());
        anyhow::ensure!(up2.len() == un, "up2 len {} ≠ {un}", up2.len());
        anyhow::ensure!(
            masks_out.len() == mn,
            "masks_out len {} ≠ {mn}",
            masks_out.len()
        );
        let outs = self.graph.run(&[("hyper", hyper_in), ("up", up2)]);
        let out = outs.into_iter().next().expect("mask_hyper_matmul output");
        masks_out.copy_from_slice(&out);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_runtime::Device;

    #[test]
    fn hyper_matmul_ir_matches_blas() {
        let nm = 4usize;
        let q8 = 32usize;
        let grid = 64usize;
        let spat = (4 * grid) * (4 * grid);
        let hyper: Vec<f32> = (0..nm * q8).map(|i| (i as f32) * 0.01).collect();
        let up: Vec<f32> = (0..q8 * spat).map(|i| ((i % 17) as f32) * 0.02).collect();
        let mut blas_out = vec![0f32; nm * spat];
        rlx_cpu::blas::sgemm_auto(&hyper, &up, &mut blas_out, nm, q8, spat);
        let mut ir_out = vec![0f32; nm * spat];
        let mut compiled = MaskHyperMatmulCompiled::compile(nm, q8, grid, Device::Cpu).unwrap();
        compiled.run(&hyper, &up, &mut ir_out).unwrap();
        let fd = blas_out
            .iter()
            .zip(&ir_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(fd < 1e-3, "IR vs BLAS max |Δ| = {fd:.3e}");
    }
}
