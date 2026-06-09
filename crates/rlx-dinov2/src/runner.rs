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

use crate::{DinoV2Config, DinoV2PreprocessWeights, assemble_hidden, rgb_u8_to_imagenet_nchw};
use anyhow::{Result, anyhow};
use rlx_core::validate_standard_device;
use rlx_flow::CompileProfile;
use rlx_runtime::Device;
use std::path::PathBuf;

/// Which DINOv2 backbone size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DinoV2Variant {
    Small,
    Base,
    Large,
}

/// Forward output: classifier logits or token features.
#[derive(Debug, Clone)]
pub enum DinoV2Output {
    Logits {
        per_batch: Vec<Vec<f32>>,
        num_classes: usize,
    },
    Tokens {
        per_batch: Vec<Vec<f32>>,
        seq: usize,
        hidden: usize,
    },
}

/// Builder for [`DinoV2Runner`]. Mirrors the qwen3 / sam shape.
#[derive(Debug, Clone, Default)]
pub struct DinoV2RunnerBuilder {
    weights: Option<PathBuf>,
    device: Option<Device>,
    variant: Option<DinoV2Variant>,
    img_size: Option<usize>,
    batch: Option<usize>,
    config: Option<DinoV2Config>,
}

impl DinoV2RunnerBuilder {
    pub fn weights<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.weights = Some(p.into());
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    /// One of the published HF presets. Default `Base` (vit-b/14).
    pub fn variant(mut self, v: DinoV2Variant) -> Self {
        self.variant = Some(v);
        self
    }
    /// Image side length (square). Must be a multiple of the patch
    /// size (14 for the standard DINOv2 checkpoints). Default 518.
    pub fn img_size(mut self, n: usize) -> Self {
        self.img_size = Some(n);
        self
    }
    pub fn batch(mut self, n: usize) -> Self {
        self.batch = Some(n);
        self
    }
    /// Skip preset selection and use an explicit
    /// [`DinoV2Config`].
    pub fn config(mut self, cfg: DinoV2Config) -> Self {
        self.config = Some(cfg);
        self
    }

    pub fn build(self) -> Result<DinoV2Runner> {
        use rlx_runtime::Session;

        let weights_path = self
            .weights
            .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?;
        let device = self.device.unwrap_or(Device::Cpu);
        validate_standard_device("dinov2", device)?;
        let img_size = self.img_size.unwrap_or(518);
        let batch = self.batch.unwrap_or(1);
        let cfg = match (self.config, self.variant) {
            (Some(c), _) => c,
            (None, Some(DinoV2Variant::Small)) => DinoV2Config::vit_small(img_size),
            (None, Some(DinoV2Variant::Large)) => DinoV2Config::vit_large(img_size),
            // Default: vit_base.
            (None, _) => DinoV2Config::vit_base(img_size),
        };

        let is_gguf = weights_path.extension().is_some_and(|e| e == "gguf");
        if is_gguf {
            rlx_core::gguf_validate_arch(&weights_path, rlx_core::DINOV2_GGUF_ARCHES)?;
        }
        let (mut wm, gguf_packed) =
            if is_gguf && crate::packed_gguf::gguf_has_packed_linears(&weights_path)? {
                eprintln!(
                    "[dinov2] loading GGUF with packed DequantMatMul {:?}",
                    weights_path
                );
                let (wm, packed) = crate::packed_gguf::load_dinov2_from_gguf(&weights_path)?;
                (wm, Some(packed))
            } else {
                (
                    rlx_core::load_weight_map(&weights_path, rlx_core::DINOV2_GGUF_ARCHES)?,
                    None,
                )
            };
        let built = super::flow::build_dinov2_built_with_packed(
            &cfg,
            &mut wm,
            batch,
            gguf_packed.as_ref(),
        )?;
        let typed = built.model.typed_params.clone();
        let pre = built.preprocess;
        let (graph, params) = rlx_core::flow_util::graph_from_built(built.model)?;
        let opts =
            rlx_core::flow_bridge::compile_options_for_profile(&CompileProfile::encoder(), device);
        let mut compiled = Session::new(device).compile_with(graph, &opts);
        rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);
        Ok(DinoV2Runner {
            compiled,
            cfg,
            preprocess: pre,
            device,
            batch,
        })
    }
}

/// Resolved DINOv2 runner.
pub struct DinoV2Runner {
    compiled: rlx_runtime::CompiledGraph,
    cfg: DinoV2Config,
    preprocess: DinoV2PreprocessWeights,
    device: Device,
    batch: usize,
}

impl DinoV2Runner {
    pub fn builder() -> DinoV2RunnerBuilder {
        DinoV2RunnerBuilder::default()
    }
    pub fn config(&self) -> &DinoV2Config {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }

    /// End-to-end forward on a single image. `rgb` is HWC u8 of any
    /// resolution; will be resized + normalized to the configured
    /// `img_size`. Returns logits when the loaded checkpoint
    /// includes a classifier head, otherwise the post-LN feature
    /// tokens.
    pub fn predict_image(&mut self, rgb: &[u8], h_in: usize, w_in: usize) -> Result<DinoV2Output> {
        // 1. resize + normalize
        let img_size = self.cfg.img_size;
        let mut nchw = rgb_u8_to_imagenet_nchw(rgb, h_in, w_in, img_size);
        // Replicate across batch dim if batch > 1.
        if self.batch > 1 {
            let per = nchw.len();
            let mut batched = Vec::with_capacity(per * self.batch);
            for _ in 0..self.batch {
                batched.extend_from_slice(&nchw);
            }
            nchw = batched;
        }

        // 2. host-side patchify + token assembly
        let hidden = assemble_hidden(
            &self.preprocess,
            &nchw,
            self.batch,
            self.cfg.patch_size,
            img_size,
        )?;

        // 3. forward through the compiled graph
        let outputs = self.compiled.run(&[("hidden", hidden.as_slice())]);
        let flat = outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("dinov2 forward returned no output"))?;

        // 4. split the flat output back into per-batch slices.
        if self.cfg.num_classes > 0 {
            let nc = self.cfg.num_classes;
            let mut per_batch = Vec::with_capacity(self.batch);
            for b in 0..self.batch {
                per_batch.push(flat[b * nc..(b + 1) * nc].to_vec());
            }
            Ok(DinoV2Output::Logits {
                per_batch,
                num_classes: nc,
            })
        } else {
            let seq = self.cfg.seq_len();
            let hidden_dim = self.cfg.hidden_size;
            let per = seq * hidden_dim;
            let mut per_batch = Vec::with_capacity(self.batch);
            for b in 0..self.batch {
                per_batch.push(flat[b * per..(b + 1) * per].to_vec());
            }
            Ok(DinoV2Output::Tokens {
                per_batch,
                seq,
                hidden: hidden_dim,
            })
        }
    }
}
