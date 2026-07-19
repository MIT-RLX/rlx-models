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

use crate::{DinoV3Config, DinoV3PreprocessWeights, assemble_hidden, rgb_u8_to_imagenet_nchw};
use anyhow::{Result, anyhow, ensure};
use rlx_core::validate_standard_device;
use rlx_flow::CompileProfile;
use rlx_runtime::Device;
use std::path::PathBuf;

/// Forward output: the pooled `[CLS]` DINOv3 embedding plus the full
/// post-final-norm token sequence.
#[derive(Debug, Clone)]
pub struct DinoV3Output {
    /// Pooled `[CLS]` feature per batch (HF `pooler_output`), `[hidden]`.
    pub embeddings: Vec<Vec<f32>>,
    /// Full post-final-norm token sequence per batch, `[seq · hidden]`
    /// flat row-major (`[CLS, reg…, patches]`) — HF `last_hidden_state`.
    pub tokens: Vec<Vec<f32>>,
    /// Sequence length (`1 + reg + num_patches`).
    pub seq: usize,
    /// Encoder width.
    pub hidden: usize,
}

/// Builder for [`DinoV3Runner`].
#[derive(Debug, Clone, Default)]
pub struct DinoV3RunnerBuilder {
    weights: Option<PathBuf>,
    device: Option<Device>,
    img_size: Option<usize>,
    batch: Option<usize>,
    config: Option<DinoV3Config>,
}

impl DinoV3RunnerBuilder {
    /// Path to the checkpoint (`.safetensors` or `.gguf`) with HF weight keys.
    pub fn weights<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.weights = Some(p.into());
        self
    }
    /// Backend to compile + run on (default [`Device::Cpu`]).
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    /// Image side length (square). Must be a multiple of the patch size
    /// (16). Default 224. When set without an explicit config, it also
    /// drives the preset's `image_size` (and hence the RoPE grid).
    pub fn img_size(mut self, n: usize) -> Self {
        self.img_size = Some(n);
        self
    }
    /// Batch size (images per forward). Default 1.
    pub fn batch(mut self, n: usize) -> Self {
        self.batch = Some(n);
        self
    }
    /// Provide an explicit [`DinoV3Config`] (e.g. from a checkpoint's
    /// `config.json`) instead of the ViT-B/16 preset.
    pub fn config(mut self, cfg: DinoV3Config) -> Self {
        self.config = Some(cfg);
        self
    }

    /// Load the weights, build + compile the encoder graph for the chosen
    /// device, and return a ready [`DinoV3Runner`].
    pub fn build(self) -> Result<DinoV3Runner> {
        use rlx_runtime::Session;

        let weights_path = self
            .weights
            .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?;
        let device = self.device.unwrap_or(Device::Cpu);
        validate_standard_device("dinov3", device)?;
        let img_size = self.img_size.unwrap_or(224);
        let batch = self.batch.unwrap_or(1);
        let had_config = self.config.is_some();
        let mut cfg = self
            .config
            .unwrap_or_else(|| DinoV3Config::vit_b16(img_size));
        // If the caller passed a config *and* an explicit img_size, honour
        // the img_size for the RoPE grid / sequence layout.
        if had_config && self.img_size.is_some() {
            cfg.image_size = img_size;
        }
        ensure!(
            cfg.image_size.is_multiple_of(cfg.patch_size),
            "image_size {} must be a multiple of patch_size {}",
            cfg.image_size,
            cfg.patch_size
        );

        let mut wm = rlx_core::load_weight_map(&weights_path, &[])?;
        let built = super::flow::build_dinov3_built(&cfg, &mut wm, batch)?;
        let typed = built.model.typed_params.clone();
        let pre = built.preprocess;
        let (graph, params) = rlx_core::flow_util::graph_from_built(built.model)?;
        let opts =
            rlx_core::flow_bridge::compile_options_for_profile(&CompileProfile::encoder(), device);
        let mut compiled = Session::new(device).compile_with(graph, &opts);
        rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);
        Ok(DinoV3Runner {
            compiled,
            cfg,
            preprocess: pre,
            device,
            batch,
        })
    }
}

/// Resolved DINOv3 runner.
pub struct DinoV3Runner {
    compiled: rlx_runtime::CompiledGraph,
    cfg: DinoV3Config,
    preprocess: DinoV3PreprocessWeights,
    device: Device,
    batch: usize,
}

impl DinoV3Runner {
    /// Start building a runner (see [`DinoV3RunnerBuilder`]).
    pub fn builder() -> DinoV3RunnerBuilder {
        DinoV3RunnerBuilder::default()
    }
    /// The resolved configuration (preset or from `config.json`).
    pub fn config(&self) -> &DinoV3Config {
        &self.cfg
    }
    /// The backend this runner compiled for.
    pub fn device(&self) -> Device {
        self.device
    }

    /// End-to-end forward on a single image. `rgb` is HWC u8 of any
    /// resolution; it is resized + ImageNet-normalized to the configured
    /// `image_size`. Returns the pooled `[CLS]` embedding and token grid.
    pub fn predict_image(&mut self, rgb: &[u8], h_in: usize, w_in: usize) -> Result<DinoV3Output> {
        let img_size = self.cfg.image_size;
        let mut nchw = rgb_u8_to_imagenet_nchw(rgb, h_in, w_in, img_size);
        if self.batch > 1 {
            let per = nchw.len();
            let mut batched = Vec::with_capacity(per * self.batch);
            for _ in 0..self.batch {
                batched.extend_from_slice(&nchw);
            }
            nchw = batched;
        }
        self.forward_nchw(&nchw)
    }

    /// Forward on an already-normalized NCHW f32 tensor
    /// (`batch · C · image_size · image_size`). This is the rigorous
    /// parity entry point: feed the exact `pixel_values` the reference saw.
    pub fn forward_nchw(&mut self, nchw: &[f32]) -> Result<DinoV3Output> {
        let img_size = self.cfg.image_size;
        let hidden = assemble_hidden(
            &self.preprocess,
            nchw,
            self.batch,
            self.cfg.patch_size,
            img_size,
        )?;

        let outputs = self.compiled.run(&[("hidden", hidden.as_slice())]);
        let flat = outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("dinov3 forward returned no output"))?;

        let seq = self.cfg.seq_len();
        let hidden_dim = self.cfg.hidden_size;
        let per = seq * hidden_dim;
        let mut embeddings = Vec::with_capacity(self.batch);
        let mut tokens = Vec::with_capacity(self.batch);
        for b in 0..self.batch {
            let slice = &flat[b * per..(b + 1) * per];
            embeddings.push(slice[..hidden_dim].to_vec());
            tokens.push(slice.to_vec());
        }
        Ok(DinoV3Output {
            embeddings,
            tokens,
            seq,
            hidden: hidden_dim,
        })
    }

    /// Convenience for `batch == 1`: just the pooled `[CLS]` embedding.
    pub fn embed_image(&mut self, rgb: &[u8], h_in: usize, w_in: usize) -> Result<Vec<f32>> {
        let out = self.predict_image(rgb, h_in, w_in)?;
        out.embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("dinov3 produced no embedding"))
    }
}
