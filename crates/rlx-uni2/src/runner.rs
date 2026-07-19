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

use crate::{Uni2Config, Uni2PreprocessWeights, assemble_hidden, rgb_u8_to_imagenet_nchw};
use anyhow::{Result, anyhow, ensure};
use rlx_core::validate_standard_device;
use rlx_flow::CompileProfile;
use rlx_runtime::Device;
use std::path::PathBuf;

/// Forward output: the pooled `[CLS]` UNI2 embedding plus the full
/// post-norm token sequence.
#[derive(Debug, Clone)]
pub struct Uni2Output {
    /// Pooled `[CLS]` feature per batch — the `[embed_dim]` (1536) UNI2
    /// embedding used for downstream tasks (timm `global_pool='token'`).
    pub embeddings: Vec<Vec<f32>>,
    /// Full post-final-norm token sequence per batch, `[seq · hidden]`
    /// flat row-major (`[CLS, reg…, patches]`).
    pub tokens: Vec<Vec<f32>>,
    pub seq: usize,
    pub hidden: usize,
}

/// Builder for [`Uni2Runner`]. Mirrors the dinov2 / sam shape.
#[derive(Debug, Clone, Default)]
pub struct Uni2RunnerBuilder {
    weights: Option<PathBuf>,
    device: Option<Device>,
    img_size: Option<usize>,
    batch: Option<usize>,
    config: Option<Uni2Config>,
}

impl Uni2RunnerBuilder {
    pub fn weights<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.weights = Some(p.into());
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    /// Image side length (square). Must be a multiple of the patch size
    /// (14). Default 224 (the checkpoint's native resolution).
    pub fn img_size(mut self, n: usize) -> Self {
        self.img_size = Some(n);
        self
    }
    pub fn batch(mut self, n: usize) -> Self {
        self.batch = Some(n);
        self
    }
    /// Skip the `uni2_h` preset and use an explicit [`Uni2Config`].
    pub fn config(mut self, cfg: Uni2Config) -> Self {
        self.config = Some(cfg);
        self
    }

    pub fn build(self) -> Result<Uni2Runner> {
        use rlx_runtime::Session;

        let weights_path = self
            .weights
            .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?;
        let device = self.device.unwrap_or(Device::Cpu);
        validate_standard_device("uni2", device)?;
        let img_size = self.img_size.unwrap_or(224);
        let batch = self.batch.unwrap_or(1);
        let cfg = self.config.unwrap_or_else(|| Uni2Config::uni2_h(img_size));
        ensure!(
            cfg.img_size.is_multiple_of(cfg.patch_size),
            "img_size {} must be a multiple of patch_size {}",
            cfg.img_size,
            cfg.patch_size
        );

        // wgpu correctness workaround: the wgpu executor mis-synchronizes
        // reused arena buffers for this graph shape (UNI2's unfused encoder
        // SwiGLU diamond + FusedResidualLN → the output is silently corrupted
        // under slot reuse, while CPU/Metal/MLX are bit-exact). Disabling arena
        // reuse restores bit-exact output (verified cos 1.0 vs the CPU/PyTorch
        // reference). This is an in-process override (safe, no `std::env`), and
        // costs extra GPU arena memory — see the crate README for the tracked
        // rlx-wgpu executor bug.
        if matches!(device, Device::Gpu | Device::WebGpu)
            && !rlx_ir::env::flag("RLX_ARENA_NO_REUSE")
        {
            eprintln!(
                "[uni2] wgpu: forcing RLX_ARENA_NO_REUSE=1 (arena-reuse hazard workaround; uses more GPU memory)"
            );
            rlx_ir::env::set("RLX_ARENA_NO_REUSE", "1");
        }

        let mut wm = rlx_core::load_weight_map(&weights_path, &[])?;
        let built = super::flow::build_uni2_built(&cfg, &mut wm, batch)?;
        let typed = built.model.typed_params.clone();
        let pre = built.preprocess;
        let (graph, params) = rlx_core::flow_util::graph_from_built(built.model)?;
        let opts =
            rlx_core::flow_bridge::compile_options_for_profile(&CompileProfile::encoder(), device);
        let mut compiled = Session::new(device).compile_with(graph, &opts);
        rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);
        Ok(Uni2Runner {
            compiled,
            cfg,
            preprocess: pre,
            device,
            batch,
        })
    }
}

/// Resolved UNI2-h runner.
pub struct Uni2Runner {
    compiled: rlx_runtime::CompiledGraph,
    cfg: Uni2Config,
    preprocess: Uni2PreprocessWeights,
    device: Device,
    batch: usize,
}

impl Uni2Runner {
    /// Start configuring a runner. See [`Uni2RunnerBuilder`].
    pub fn builder() -> Uni2RunnerBuilder {
        Uni2RunnerBuilder::default()
    }
    /// The resolved model configuration (dimensions, patch/image size, …).
    pub fn config(&self) -> &Uni2Config {
        &self.cfg
    }
    /// The device this runner compiled for.
    pub fn device(&self) -> Device {
        self.device
    }

    /// End-to-end forward on a single image. `rgb` is HWC u8 of any
    /// resolution; it is resized + ImageNet-normalized to the configured
    /// `img_size` (224). Returns the pooled `[CLS]` embedding and the
    /// post-norm token sequence.
    pub fn predict_image(&mut self, rgb: &[u8], h_in: usize, w_in: usize) -> Result<Uni2Output> {
        let img_size = self.cfg.img_size;
        let mut nchw = rgb_u8_to_imagenet_nchw(rgb, h_in, w_in, img_size);
        if self.batch > 1 {
            let per = nchw.len();
            let mut batched = Vec::with_capacity(per * self.batch);
            for _ in 0..self.batch {
                batched.extend_from_slice(&nchw);
            }
            nchw = batched;
        }

        let hidden = assemble_hidden(
            &self.preprocess,
            &nchw,
            self.batch,
            self.cfg.patch_size,
            img_size,
        )?;

        let outputs = self.compiled.run(&[("hidden", hidden.as_slice())]);
        let flat = outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("uni2 forward returned no output"))?;

        let seq = self.cfg.seq_len();
        let hidden_dim = self.cfg.hidden_size;
        let per = seq * hidden_dim;
        let mut embeddings = Vec::with_capacity(self.batch);
        let mut tokens = Vec::with_capacity(self.batch);
        for b in 0..self.batch {
            let slice = &flat[b * per..(b + 1) * per];
            // CLS token (row 0) → the pooled UNI2 feature.
            embeddings.push(slice[..hidden_dim].to_vec());
            tokens.push(slice.to_vec());
        }
        Ok(Uni2Output {
            embeddings,
            tokens,
            seq,
            hidden: hidden_dim,
        })
    }

    /// Convenience for `batch == 1`: return just the pooled `[CLS]`
    /// embedding `[embed_dim]`.
    pub fn embed_image(&mut self, rgb: &[u8], h_in: usize, w_in: usize) -> Result<Vec<f32>> {
        let out = self.predict_image(rgb, h_in, w_in)?;
        out.embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("uni2 produced no embedding"))
    }
}
