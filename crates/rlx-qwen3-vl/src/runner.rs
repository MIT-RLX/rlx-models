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

//! Qwen3-VL vision-tower runner.
//!
//! Compiles the SigLIP encoder + multimodal projector graph from
//! [`super::flow`] and exposes a host API that takes an image and
//! returns `[num_patches, projector_output_dim]` LM-aligned embeddings
//! ready to slot into the Qwen3 text token stream.
//!
//! The text side (Qwen3 LM) is wrapped separately by the consumer —
//! this crate's responsibility ends at the projected vision embeds.

use anyhow::{Context, Result, anyhow};
use rlx_flow::CompileProfile;
use rlx_runtime::{Device, Session};
use rlx_vlm_base::{ImagePatches, ImagePreprocessor, Projector, VisionTower};
use std::path::{Path, PathBuf};

use super::config::Qwen3VlVisionConfig;
use super::flow::{Qwen3VlVisionBuilt, build_qwen3_vl_vision};
use super::preprocess::{Qwen3VlImagePreprocessor, Qwen3VlPreprocessWeights, assemble_hidden};

#[derive(Debug, Clone, Default)]
pub struct Qwen3VlVisionRunnerBuilder {
    mmproj: Option<PathBuf>,
    hf_config: Option<PathBuf>,
    config: Option<Qwen3VlVisionConfig>,
    device: Option<Device>,
}

impl Qwen3VlVisionRunnerBuilder {
    pub fn mmproj(mut self, p: impl Into<PathBuf>) -> Self {
        self.mmproj = Some(p.into());
        self
    }
    /// Path to the HuggingFace `config.json` for hyperparameter
    /// discovery. Optional if `.config(...)` is supplied directly.
    pub fn hf_config(mut self, p: impl Into<PathBuf>) -> Self {
        self.hf_config = Some(p.into());
        self
    }
    pub fn config(mut self, cfg: Qwen3VlVisionConfig) -> Self {
        self.config = Some(cfg);
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    pub fn build(self) -> Result<Qwen3VlVisionRunner> {
        let mmproj = self
            .mmproj
            .ok_or_else(|| anyhow!("mmproj GGUF path required (call .mmproj(...))"))?;
        let device = self.device.unwrap_or(Device::Cpu);

        let cfg = match (self.config, self.hf_config) {
            (Some(c), _) => c,
            (None, Some(p)) => Qwen3VlVisionConfig::from_hf_config_json(&p)
                .with_context(|| format!("rlx-qwen3-vl: parse {p:?}"))?,
            (None, None) => {
                return Err(anyhow!(
                    "rlx-qwen3-vl: either .config(...) or .hf_config(...) required \
                     so the vision tower hyperparameters are known"
                ));
            }
        };

        let mut wm = rlx_core::load_weight_map(&mmproj, &[])
            .with_context(|| format!("rlx-qwen3-vl: load weights {mmproj:?}"))?;
        let built: Qwen3VlVisionBuilt = build_qwen3_vl_vision(&cfg, &mut wm)?;
        let typed = built.model.typed_params.clone();
        let pre = built.preprocess;
        let (graph, params) = rlx_core::flow_util::graph_from_built(built.model)?;
        let opts =
            rlx_core::flow_bridge::compile_options_for_profile(&CompileProfile::encoder(), device);
        let mut compiled = Session::new(device).compile_with(graph, &opts);
        rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);

        Ok(Qwen3VlVisionRunner {
            compiled,
            cfg,
            preprocess: pre,
            device,
        })
    }
}

pub struct Qwen3VlVisionRunner {
    compiled: rlx_runtime::CompiledGraph,
    cfg: Qwen3VlVisionConfig,
    preprocess: Qwen3VlPreprocessWeights,
    device: Device,
}

impl Qwen3VlVisionRunner {
    pub fn builder() -> Qwen3VlVisionRunnerBuilder {
        Qwen3VlVisionRunnerBuilder::default()
    }

    pub fn config(&self) -> &Qwen3VlVisionConfig {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }

    pub fn preprocessor(&self) -> Qwen3VlImagePreprocessor {
        Qwen3VlImagePreprocessor {
            cfg: self.cfg.clone(),
        }
    }

    /// End-to-end image → LM-aligned embeddings.
    /// Output: `[num_patches, projector_output_dim]` row-major.
    pub fn embed_image_path(&mut self, path: &Path) -> Result<Vec<f32>> {
        let pp = self.preprocessor();
        let patches = pp.preprocess_path(path)?;
        self.embed_patches(&patches)
    }

    pub fn embed_image_bytes(&mut self, bytes: &[u8]) -> Result<Vec<f32>> {
        let pp = self.preprocessor();
        let patches = pp.preprocess_bytes(bytes)?;
        self.embed_patches(&patches)
    }

    pub fn embed_patches(&mut self, patches: &ImagePatches) -> Result<Vec<f32>> {
        let hidden = assemble_hidden(&self.preprocess, patches)?;
        let outputs = self.compiled.run(&[("hidden", hidden.as_slice())]);
        let flat = outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("qwen3-vl forward returned no output"))?;
        Ok(flat)
    }
}

impl VisionTower for Qwen3VlVisionRunner {
    fn embed(&mut self, patches: &ImagePatches) -> Result<Vec<f32>> {
        self.embed_patches(patches)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
}

/// Identity-projector: Qwen3-VL bakes the projector into the vision
/// graph, so `Projector::project` just passes through. Provided so
/// downstream code can use the `rlx_vlm_base::Projector` trait
/// uniformly across families.
pub struct Qwen3VlIdentityProjector {
    pub output_dim: usize,
}

impl Projector for Qwen3VlIdentityProjector {
    fn project(&mut self, vision_embed: &[f32], num_patches: usize) -> Result<Vec<f32>> {
        debug_assert_eq!(vision_embed.len(), num_patches * self.output_dim);
        Ok(vision_embed.to_vec())
    }
    fn output_dim(&self) -> usize {
        self.output_dim
    }
}
