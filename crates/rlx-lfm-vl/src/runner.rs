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

use anyhow::{Context, Result, anyhow};
use rlx_flow::CompileProfile;
use rlx_runtime::{Device, Session};
use rlx_vlm_base::{ImagePatches, ImagePreprocessor, Projector, VisionTower};
use std::path::{Path, PathBuf};

use super::config::LfmVlVisionConfig;
use super::flow::{LfmVlVisionBuilt, build_lfm_vl_vision};
use super::preprocess::{LfmVlImagePreprocessor, LfmVlPreprocessWeights, assemble_hidden};

#[derive(Debug, Clone, Default)]
pub struct LfmVlVisionRunnerBuilder {
    mmproj: Option<PathBuf>,
    hf_config: Option<PathBuf>,
    config: Option<LfmVlVisionConfig>,
    device: Option<Device>,
}

impl LfmVlVisionRunnerBuilder {
    pub fn mmproj(mut self, p: impl Into<PathBuf>) -> Self {
        self.mmproj = Some(p.into());
        self
    }
    pub fn hf_config(mut self, p: impl Into<PathBuf>) -> Self {
        self.hf_config = Some(p.into());
        self
    }
    pub fn config(mut self, c: LfmVlVisionConfig) -> Self {
        self.config = Some(c);
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    pub fn build(self) -> Result<LfmVlVisionRunner> {
        let mmproj = self.mmproj.ok_or_else(|| anyhow!("mmproj path required"))?;
        let device = self.device.unwrap_or(Device::Cpu);
        let cfg = match (self.config, self.hf_config) {
            (Some(c), _) => c,
            (None, Some(p)) => LfmVlVisionConfig::from_hf_config_json(&p)
                .with_context(|| format!("rlx-lfm-vl: parse {p:?}"))?,
            (None, None) => {
                return Err(anyhow!(
                    "rlx-lfm-vl: either .config(...) or .hf_config(...) required"
                ));
            }
        };
        let mut wm = rlx_core::load_weight_map(&mmproj, &[])
            .with_context(|| format!("rlx-lfm-vl: load weights {mmproj:?}"))?;
        let built: LfmVlVisionBuilt = build_lfm_vl_vision(&cfg, &mut wm)?;
        let typed = built.model.typed_params.clone();
        let pre = built.preprocess;
        let (graph, params) = rlx_core::flow_util::graph_from_built(built.model)?;
        let opts =
            rlx_core::flow_bridge::compile_options_for_profile(&CompileProfile::encoder(), device);
        let mut compiled = Session::new(device).compile_with(graph, &opts);
        rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);
        Ok(LfmVlVisionRunner {
            compiled,
            cfg,
            preprocess: pre,
            device,
        })
    }
}

pub struct LfmVlVisionRunner {
    compiled: rlx_runtime::CompiledGraph,
    cfg: LfmVlVisionConfig,
    preprocess: LfmVlPreprocessWeights,
    device: Device,
}

impl LfmVlVisionRunner {
    pub fn builder() -> LfmVlVisionRunnerBuilder {
        LfmVlVisionRunnerBuilder::default()
    }
    pub fn config(&self) -> &LfmVlVisionConfig {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn preprocessor(&self) -> LfmVlImagePreprocessor {
        LfmVlImagePreprocessor {
            cfg: self.cfg.clone(),
        }
    }
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
        outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("lfm-vl forward returned no output"))
    }
}

impl VisionTower for LfmVlVisionRunner {
    fn embed(&mut self, patches: &ImagePatches) -> Result<Vec<f32>> {
        self.embed_patches(patches)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
}

pub struct LfmVlIdentityProjector {
    pub output_dim: usize,
}
impl Projector for LfmVlIdentityProjector {
    fn project(&mut self, vision_embed: &[f32], num_patches: usize) -> Result<Vec<f32>> {
        debug_assert_eq!(vision_embed.len(), num_patches * self.output_dim);
        Ok(vision_embed.to_vec())
    }
    fn output_dim(&self) -> usize {
        self.output_dim
    }
}
