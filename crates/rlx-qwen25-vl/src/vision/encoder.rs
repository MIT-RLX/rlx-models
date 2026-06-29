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

//! Qwen2.5-VL vision encoder — compile + run mmproj graph.

use super::builder::build_qwen25_vl_vision_built;
use super::config::MmProjConfig;
use super::preprocess::{
    DEFAULT_ATTN_WINDOW_SIZE, build_vision_position_hw, build_window_attn_inputs,
    expand_window_attn_bias, preprocess_rgb, preprocess_rgb_to_size, reorder_seq_by_window_inv,
    vision_rope_feeds,
};
use super::weights::MmProjWeights;
use anyhow::{Context, Result};
use rlx_core::flow_util::compile_built;
use rlx_core::weight_loader::GgufLoader;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct VisionEncodeOutput {
    pub embeddings: Vec<f32>,
    pub grid_x: usize,
    pub grid_y: usize,
    pub n_tokens: usize,
}

pub struct Qwen25VlVisionEncoder {
    cfg: MmProjConfig,
    weights: MmProjWeights,
    params: std::collections::HashMap<String, Vec<f32>>,
    graph_key: (usize, usize),
    compiled: rlx_runtime::CompiledGraph,
    device: Device,
}

impl Qwen25VlVisionEncoder {
    pub fn from_mmproj(path: impl AsRef<Path>, img_w: usize, img_h: usize) -> Result<Self> {
        Self::from_mmproj_device(path, img_w, img_h, Device::Cpu)
    }

    pub fn from_mmproj_device(
        path: impl AsRef<Path>,
        img_w: usize,
        img_h: usize,
        device: Device,
    ) -> Result<Self> {
        let path = path.as_ref();
        let path_str = path.to_str().context("mmproj path utf8")?;
        let mut loader = GgufLoader::from_file(path_str)?;
        let cfg = MmProjConfig::from_gguf(loader.file())?;
        let weights = MmProjWeights::from_loader(&cfg, &mut loader)?;
        Self::from_parts_device(cfg, weights, img_w, img_h, device)
    }

    pub fn from_parts(
        cfg: MmProjConfig,
        weights: MmProjWeights,
        img_w: usize,
        img_h: usize,
    ) -> Result<Self> {
        Self::from_parts_device(cfg, weights, img_w, img_h, Device::Cpu)
    }

    pub fn from_parts_device(
        cfg: MmProjConfig,
        weights: MmProjWeights,
        img_w: usize,
        img_h: usize,
        device: Device,
    ) -> Result<Self> {
        let built = build_qwen25_vl_vision_built(&cfg, &weights, img_w, img_h)?;
        let params = built.params().clone();
        let compiled = compile_built(built, device)?;
        Ok(Self {
            graph_key: (img_w, img_h),
            cfg,
            weights,
            params,
            compiled,
            device,
        })
    }

    pub fn config(&self) -> &MmProjConfig {
        &self.cfg
    }

    pub fn encode_rgb(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<VisionEncodeOutput> {
        self.encode_rgb_resized(rgb, w, h, None, None)
    }

    /// Encode with optional explicit resize (matches HF `image_grid_thw` replay).
    pub fn encode_rgb_resized(
        &mut self,
        rgb: &[u8],
        w: usize,
        h: usize,
        target_w: Option<usize>,
        target_h: Option<usize>,
    ) -> Result<VisionEncodeOutput> {
        let (nchw, tw, th) = match (target_w, target_h) {
            (Some(tw), Some(th)) => preprocess_rgb_to_size(rgb, w, h, tw, th, &self.cfg),
            _ => preprocess_rgb(rgb, w, h, &self.cfg),
        };
        self.ensure_compiled(tw, th)?;
        self.run_vision_graph(&nchw, tw, th)
    }

    fn run_vision_graph(
        &mut self,
        nchw: &[f32],
        tw: usize,
        th: usize,
    ) -> Result<VisionEncodeOutput> {
        let (gx, gy) = self.cfg.output_grid(tw, th);
        let n_tokens = gx * gy;
        let proj = self.cfg.llm_hidden_size;

        let ps = self.cfg.patch_size;
        let n_pos = (th / ps) * (tw / ps);
        let position_hw = build_vision_position_hw(tw, th, &self.cfg);
        let head_dim = self.cfg.n_embd / self.cfg.n_head;
        let (mut rope_cos, mut rope_sin) = vision_rope_feeds(&position_hw, head_dim);

        let mut feeds: Vec<(&str, &[f32])> = vec![("image", nchw)];
        let window_owned;
        let window_bias_owned;
        if self.cfg.n_wa_pattern > 0 {
            window_owned = build_window_attn_inputs(tw, th, &self.cfg, DEFAULT_ATTN_WINDOW_SIZE);
            let merge_sq = self.cfg.n_merge * self.cfg.n_merge;
            reorder_seq_by_window_inv(
                &mut rope_cos,
                &window_owned.inv_window_idx,
                head_dim,
                merge_sq,
            );
            reorder_seq_by_window_inv(
                &mut rope_sin,
                &window_owned.inv_window_idx,
                head_dim,
                merge_sq,
            );
            window_bias_owned =
                expand_window_attn_bias(&window_owned.window_mask, 1, self.cfg.n_head, n_pos);
            feeds.push(("inv_window_idx", &window_owned.inv_window_idx));
            feeds.push(("window_idx", &window_owned.window_idx));
            feeds.push(("window_mask", &window_bias_owned));
        }
        feeds.push(("vision_rope_cos", &rope_cos));
        feeds.push(("vision_rope_sin", &rope_sin));

        let outs = self.compiled.run(&feeds);
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
        let built = build_qwen25_vl_vision_built(&self.cfg, &self.weights, img_w, img_h)?;
        self.params = built.params().clone();
        self.compiled = compile_built(built, self.device)?;
        self.graph_key = (img_w, img_h);
        Ok(())
    }
}

pub fn load_vision_encoder(
    mmproj_path: &str,
    img_w: usize,
    img_h: usize,
) -> Result<Qwen25VlVisionEncoder> {
    Qwen25VlVisionEncoder::from_mmproj(PathBuf::from(mmproj_path), img_w, img_h)
}

#[cfg(feature = "qwen25-vl-vision")]
pub fn encode_image_file(
    encoder: &mut Qwen25VlVisionEncoder,
    path: &str,
) -> Result<VisionEncodeOutput> {
    let (rgb, w, h) = super::preprocess::load_rgb_image(path)?;
    encoder.encode_rgb(&rgb, w, h)
}
