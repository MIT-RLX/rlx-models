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

//! Qwen3.5 VLM vision encoder — CPU encode + multimodal prompt helpers.

use super::config::MmProjConfig;
use super::flow::build_qwen35_vision_built;
use super::preprocess::{build_vision_positions, preprocess_rgb};
use super::weights::MmProjWeights;
use anyhow::{Context, Result};
use rlx_core::flow_util::compile_built;
use rlx_core::weight_loader::GgufLoader;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

/// Vision tower forward output.
#[derive(Debug, Clone)]
pub struct VisionEncodeOutput {
    pub embeddings: Vec<f32>,
    pub grid_x: usize,
    pub grid_y: usize,
    pub n_tokens: usize,
}

/// CPU vision encoder wrapping a compiled mmproj graph.
pub struct Qwen35VisionEncoder {
    cfg: MmProjConfig,
    weights: MmProjWeights,
    params: std::collections::HashMap<String, Vec<f32>>,
    graph_key: (usize, usize),
    compiled: rlx_runtime::CompiledGraph,
}

impl Qwen35VisionEncoder {
    /// Load mmproj GGUF from disk and compile for the given image size.
    pub fn from_mmproj(path: impl AsRef<Path>, img_w: usize, img_h: usize) -> Result<Self> {
        let path = path.as_ref();
        let path_str = path.to_str().context("mmproj path utf8")?;
        let mut loader = GgufLoader::from_file(path_str)?;
        let cfg = MmProjConfig::from_gguf(loader.file())?;
        let weights = MmProjWeights::from_loader(&cfg, &mut loader)?;
        Self::from_parts(cfg, weights, img_w, img_h)
    }

    /// Build from already-loaded config + weights (tests).
    pub fn from_parts(
        cfg: MmProjConfig,
        weights: MmProjWeights,
        img_w: usize,
        img_h: usize,
    ) -> Result<Self> {
        let built = build_qwen35_vision_built(&cfg, &weights, img_w, img_h)?;
        let params = built.params().clone();
        let compiled = compile_built(built, Device::Cpu)?;
        Ok(Self {
            graph_key: (img_w, img_h),
            cfg,
            weights,
            params,
            compiled,
        })
    }

    pub fn config(&self) -> &MmProjConfig {
        &self.cfg
    }

    /// Encode an RGB u8 buffer. Recompiles when smart-resize changes dimensions.
    pub fn encode_rgb(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<VisionEncodeOutput> {
        let (nchw, tw, th) = preprocess_rgb(rgb, w, h, &self.cfg);
        self.ensure_compiled(tw, th)?;
        let (gx, gy) = self.cfg.output_grid(tw, th);
        let n_tokens = gx * gy;
        let proj = self.cfg.llm_hidden_size;

        let _positions = build_vision_positions(tw, th, &self.cfg);

        let outs = self.compiled.run(&[("image", &nchw)]);
        let emb = outs
            .into_iter()
            .next()
            .context("vision graph produced no outputs")?;

        anyhow::ensure!(
            emb.len() == n_tokens * proj,
            "vision output len {} != n_tokens*proj {}*{}",
            emb.len(),
            n_tokens,
            proj
        );

        Ok(VisionEncodeOutput {
            embeddings: emb,
            grid_x: gx,
            grid_y: gy,
            n_tokens,
        })
    }

    fn ensure_compiled(&mut self, img_w: usize, img_h: usize) -> Result<()> {
        if self.graph_key == (img_w, img_h) {
            return Ok(());
        }
        let built = build_qwen35_vision_built(&self.cfg, &self.weights, img_w, img_h)?;
        self.params = built.params().clone();
        self.compiled = compile_built(built, Device::Cpu)?;
        self.graph_key = (img_w, img_h);
        Ok(())
    }
}

/// Convenience: load encoder from path string.
pub fn load_vision_encoder(
    mmproj_path: &str,
    img_w: usize,
    img_h: usize,
) -> Result<Qwen35VisionEncoder> {
    Qwen35VisionEncoder::from_mmproj(PathBuf::from(mmproj_path), img_w, img_h)
}

#[cfg(feature = "qwen35-vlm")]
pub fn encode_image_file(
    encoder: &mut Qwen35VisionEncoder,
    path: &str,
) -> Result<VisionEncodeOutput> {
    let (rgb, w, h) = super::preprocess::load_rgb_image(path)?;
    encoder.encode_rgb(&rgb, w, h)
}
