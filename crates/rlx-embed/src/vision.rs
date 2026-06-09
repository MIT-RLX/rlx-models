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

//! RLX-compiled NomicVision encoder for image embeddings.

use std::path::Path;

use anyhow::Result;
use rlx_flow::CompileProfile;
use rlx_runtime::CompiledGraph;
use rlx_runtime::Device;

use rlx_core::config::NomicVisionConfig;
use rlx_core::weight_map::WeightMap;
use rlx_vision::vision::{VisionPreprocessWeights, build_vision_graph_sized};

/// Assemble encoder input `[batch, seq, hidden]` from NCHW pixels + preprocess weights.
pub fn assemble_vision_hidden(
    pixel_values: &[f32],
    batch: usize,
    img: usize,
    ps: usize,
    h: usize,
    preprocess: &VisionPreprocessWeights,
) -> Vec<f32> {
    let np = (img / ps) * (img / ps);
    let seq = np + 1;
    let patch_dim = 3 * ps * ps;
    let patches_per_row = img / ps;
    let pw = preprocess;

    let mut patches = vec![0f32; batch * np * patch_dim];
    for bi in 0..batch {
        for py in 0..patches_per_row {
            for px in 0..patches_per_row {
                let pi = bi * np + py * patches_per_row + px;
                let dst = &mut patches[pi * patch_dim..(pi + 1) * patch_dim];
                let mut di = 0;
                for c in 0..3usize {
                    for dy in 0..ps {
                        for dx in 0..ps {
                            let y = py * ps + dy;
                            let x = px * ps + dx;
                            dst[di] =
                                pixel_values[bi * 3 * img * img + c * img * img + y * img + x];
                            di += 1;
                        }
                    }
                }
            }
        }
    }

    let m = batch * np;
    let k = patch_dim;
    let n = h;
    let mut projected = vec![0f32; m * n];
    rlx_cpu::blas::sgemm_bias(&patches, &pw.proj_w, &pw.proj_b, &mut projected, m, k, n);

    let mut hidden = vec![0f32; batch * seq * h];
    let cls = &pw.cls_token[..h.min(pw.cls_token.len())];
    let pos = &pw.pos_embed;
    for bi in 0..batch {
        let base = bi * seq * h;
        hidden[base..base + h].copy_from_slice(cls);
        let proj_start = bi * np * h;
        hidden[base + h..base + (np + 1) * h]
            .copy_from_slice(&projected[proj_start..proj_start + np * h]);
        let pos_len = (seq * h).min(pos.len());
        for i in 0..pos_len {
            hidden[base + i] += pos[i];
        }
    }
    hidden
}

/// RLX-compiled NomicVision encoder (patch preprocess host-side, trunk on RLX).
pub struct RlxVisionModel {
    compiled: CompiledGraph,
    config: NomicVisionConfig,
    preprocess: VisionPreprocessWeights,
    #[allow(dead_code)]
    compiled_batch: usize,
}

impl RlxVisionModel {
    pub fn load_sized(config_path: &Path, weights_path: &str, batch: usize) -> Result<Self> {
        Self::load_sized_on(config_path, weights_path, batch, Device::Cpu)
    }

    pub fn load_sized_on(
        config_path: &Path,
        weights_path: &str,
        batch: usize,
        device: Device,
    ) -> Result<Self> {
        let config = NomicVisionConfig::from_file(config_path)?;
        let mut wm = WeightMap::from_file(weights_path)?;
        let (graph, params, preprocess) = build_vision_graph_sized(&config, &mut wm, batch)?;
        let mut compiled = rlx_core::flow_bridge::compile_graph_with_profile(
            device,
            graph,
            &CompileProfile::encoder(),
        )?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        Ok(Self {
            compiled,
            config,
            preprocess,
            compiled_batch: batch,
        })
    }

    /// Forward: `pixel_values` `[batch, 3, img, img]` row-major NCHW → CLS `[batch, hidden]`.
    pub fn forward(&mut self, pixel_values: &[f32], batch: usize) -> Vec<f32> {
        let hidden = assemble_vision_hidden(
            pixel_values,
            batch,
            self.config.img_size,
            self.config.patch_size,
            self.config.hidden_size,
            &self.preprocess,
        );
        self.compiled
            .run(&[("hidden", &hidden)])
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    pub fn forward_all(&mut self, pixel_values: &[f32], batch: usize) -> Vec<Vec<f32>> {
        let hidden = assemble_vision_hidden(
            pixel_values,
            batch,
            self.config.img_size,
            self.config.patch_size,
            self.config.hidden_size,
            &self.preprocess,
        );
        self.compiled.run(&[("hidden", &hidden)])
    }

    pub fn forward_slots(&mut self, hidden: &[f32]) -> (*const f32, usize) {
        let slots = self.compiled.run_slots(&[hidden]);
        if slots.is_empty() {
            return (std::ptr::null(), 0);
        }
        let (off, len) = slots[0];
        unsafe {
            let ptr = self.compiled.arena_ptr().add(off) as *const f32;
            (ptr, len)
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    pub fn img_size(&self) -> usize {
        self.config.img_size
    }

    pub fn patch_size(&self) -> usize {
        self.config.patch_size
    }

    pub fn num_patches(&self) -> usize {
        (self.config.img_size / self.config.patch_size).pow(2)
    }

    pub fn preprocess_weights(&self) -> &VisionPreprocessWeights {
        &self.preprocess
    }
}
