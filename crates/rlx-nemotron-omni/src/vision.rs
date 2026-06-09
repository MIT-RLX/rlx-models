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

//! Nemotron-Omni vision tower (SigLIP-variant ViT + projector).
//!
//! Mirrors the Qwen3-VL / LFM2.5-VL pattern but reads NVIDIA's
//! `vision_tower.…` / `mm_projector.…` weight names. The encoder
//! itself uses [`rlx_flow::blocks::siglip_layer_fused_with_prefix`].

use anyhow::{Context, Result, anyhow, ensure};
use image::imageops::FilterType;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::blocks::siglip_layer_fused_with_prefix;
use rlx_flow::{BuiltModel, CompileProfile, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::{Device, Session};
use rlx_vlm_base::{ImagePatches, ImagePreprocessor, Projector, VisionTower};
use std::path::{Path, PathBuf};

use super::config::NemotronOmniVisionConfig;

const SIGLIP_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const SIGLIP_STD: [f32; 3] = [0.5, 0.5, 0.5];

const ENCODER_PREFIX: &str = "vision_tower.vision_model.encoder";
const POST_LN_W: &str = "vision_tower.vision_model.post_layernorm.weight";
const POST_LN_B: &str = "vision_tower.vision_model.post_layernorm.bias";
const PATCH_EMBED_W: &str = "vision_tower.vision_model.embeddings.patch_embedding.weight";
const PATCH_EMBED_B: &str = "vision_tower.vision_model.embeddings.patch_embedding.bias";
const POS_EMBED: &str = "vision_tower.vision_model.embeddings.position_embedding.weight";
const PROJ_FC1_W: &str = "mm_projector.0.weight";
const PROJ_FC1_B: &str = "mm_projector.0.bias";
const PROJ_FC2_W: &str = "mm_projector.2.weight";
const PROJ_FC2_B: &str = "mm_projector.2.bias";

pub struct NemotronOmniPreprocessWeights {
    pub proj_w: Vec<f32>,
    pub proj_b: Vec<f32>,
    pub pos_embed: Vec<f32>,
    pub embed_dim: usize,
    pub patch_dim: usize,
    pub num_patches: usize,
}

fn extract_preprocess_weights(
    weights: &mut WeightMap,
    cfg: &NemotronOmniVisionConfig,
) -> Result<NemotronOmniPreprocessWeights> {
    let embed_dim = cfg.hidden_size;
    let patch_dim = cfg.patch_dim();
    let num_patches = cfg.num_patches();

    let (proj_raw, proj_shape) = weights.take(PATCH_EMBED_W)?;
    ensure!(
        proj_shape.len() == 4
            && proj_shape[0] == embed_dim
            && proj_shape[1] * proj_shape[2] * proj_shape[3] == patch_dim,
        "{PATCH_EMBED_W} expected [E={embed_dim}, 3, ps, ps] (patch_dim={patch_dim}), got {proj_shape:?}"
    );
    let mut proj_w = vec![0f32; embed_dim * patch_dim];
    for e in 0..embed_dim {
        for d in 0..patch_dim {
            proj_w[d * embed_dim + e] = proj_raw[e * patch_dim + d];
        }
    }
    let (proj_b, _) = weights.take(PATCH_EMBED_B)?;
    let (pos_embed, _) = weights.take(POS_EMBED)?;
    ensure!(
        pos_embed.len() == num_patches * embed_dim,
        "{POS_EMBED} length {} ≠ num_patches*E ({}*{})",
        pos_embed.len(),
        num_patches,
        embed_dim
    );

    Ok(NemotronOmniPreprocessWeights {
        proj_w,
        proj_b,
        pos_embed,
        embed_dim,
        patch_dim,
        num_patches,
    })
}

fn image_to_patch_tensor(
    img: &image::DynamicImage,
    cfg: &NemotronOmniVisionConfig,
) -> Result<ImagePatches> {
    let target = cfg.image_size as u32;
    let resized = img.resize_exact(target, target, FilterType::CatmullRom);
    let rgb = resized.to_rgb8();
    let ps = cfg.patch_size as u32;
    ensure!(
        target.is_multiple_of(ps),
        "image_size {target} not divisible by patch_size {ps}"
    );
    let grid = (target / ps) as usize;
    let num_patches = grid * grid;
    let patch_dim = cfg.num_channels * cfg.patch_size * cfg.patch_size;
    let mut patches = vec![0f32; num_patches * patch_dim];
    for gy in 0..grid {
        for gx in 0..grid {
            let row = gy * grid + gx;
            for py in 0..cfg.patch_size {
                for px in 0..cfg.patch_size {
                    let pix = rgb.get_pixel(
                        (gx * cfg.patch_size + px) as u32,
                        (gy * cfg.patch_size + py) as u32,
                    );
                    for c in 0..cfg.num_channels {
                        let v = (pix.0[c] as f32 / 255.0 - SIGLIP_MEAN[c]) / SIGLIP_STD[c];
                        let inner = c * cfg.patch_size * cfg.patch_size + py * cfg.patch_size + px;
                        patches[row * patch_dim + inner] = v;
                    }
                }
            }
        }
    }
    Ok(ImagePatches {
        patches,
        grid_h: grid,
        grid_w: grid,
        patch_h: cfg.patch_size,
        patch_w: cfg.patch_size,
        channels: cfg.num_channels,
    })
}

fn assemble_hidden(pp: &NemotronOmniPreprocessWeights, patches: &ImagePatches) -> Result<Vec<f32>> {
    ensure!(patches.num_patches() == pp.num_patches);
    ensure!(patches.patch_dim() == pp.patch_dim);
    let (n, e, d) = (pp.num_patches, pp.embed_dim, pp.patch_dim);
    let mut out = vec![0f32; n * e];
    for row in 0..n {
        for col in 0..e {
            let mut acc = pp.proj_b[col];
            for k in 0..d {
                acc += patches.patches[row * d + k] * pp.proj_w[k * e + col];
            }
            acc += pp.pos_embed[row * e + col];
            out[row * e + col] = acc;
        }
    }
    Ok(out)
}

pub struct NemotronOmniImagePreprocessor {
    pub cfg: NemotronOmniVisionConfig,
}
impl ImagePreprocessor for NemotronOmniImagePreprocessor {
    fn preprocess_path(&self, path: &Path) -> Result<ImagePatches> {
        let img =
            image::open(path).map_err(|e| anyhow!("rlx-nemotron-omni: open {path:?}: {e}"))?;
        image_to_patch_tensor(&img, &self.cfg)
    }
    fn preprocess_bytes(&self, bytes: &[u8]) -> Result<ImagePatches> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| anyhow!("rlx-nemotron-omni: decode image: {e}"))?;
        image_to_patch_tensor(&img, &self.cfg)
    }
}

pub struct NemotronOmniVisionBuilt {
    pub model: BuiltModel,
    pub preprocess: NemotronOmniPreprocessWeights,
}

pub fn build_nemotron_omni_vision(
    cfg: &NemotronOmniVisionConfig,
    weights: &mut WeightMap,
) -> Result<NemotronOmniVisionBuilt> {
    let preprocess = extract_preprocess_weights(weights, cfg)?;
    let batch = 1usize;
    let seq = cfg.seq_len();
    let h = cfg.hidden_size;
    let lm_h = cfg.projector_output_dim;
    let nh = cfg.num_attention_heads;
    let eps = cfg.layer_norm_eps as f32;
    let f = DType::F32;

    let mut flow = ModelFlow::new("nemotron_omni_vision")
        .with_profile(CompileProfile::encoder())
        .input("hidden", Shape::new(&[batch, seq, h], f))
        .attn_mask_ones(batch, seq);

    for i in 0..cfg.num_hidden_layers {
        let layer_prefix = format!("{ENCODER_PREFIX}.layers.{i}");
        flow = flow.named_layer(
            format!("layer{i}"),
            siglip_layer_fused_with_prefix(layer_prefix, i, h, nh, eps),
        );
    }

    flow = flow
        .layer_norm(POST_LN_W, POST_LN_B, eps)
        .plugin_named("nemotron_omni.projector", move |emit, hidden| {
            let v = hidden.ok_or_else(|| anyhow::anyhow!("projector requires hidden"))?;
            let l1_w = emit.load_param(PROJ_FC1_W, true)?;
            let l1_b = emit.load_param(PROJ_FC1_B, false)?;
            let l2_w = emit.load_param(PROJ_FC2_W, true)?;
            let l2_b = emit.load_param(PROJ_FC2_B, false)?;
            let mut gb = HirMut::new(emit.hir());
            let m1 = gb.mm(v.hir_id(), l1_w);
            let a1 = gb.add(m1, l1_b);
            let act = gb.gelu(a1);
            let m2 = gb.mm(act, l2_w);
            let out = gb.add(m2, l2_b);
            Ok(Some(
                emit.wrap(out, Shape::new(&[batch, seq, lm_h], DType::F32)),
            ))
        })
        .output("lm_embeds");

    Ok(NemotronOmniVisionBuilt {
        model: flow.build_with(&mut WeightMapSource(weights), None)?,
        preprocess,
    })
}

#[derive(Debug, Clone, Default)]
pub struct NemotronOmniVisionRunnerBuilder {
    mmproj: Option<PathBuf>,
    hf_config: Option<PathBuf>,
    config: Option<NemotronOmniVisionConfig>,
    device: Option<Device>,
}

impl NemotronOmniVisionRunnerBuilder {
    pub fn mmproj(mut self, p: impl Into<PathBuf>) -> Self {
        self.mmproj = Some(p.into());
        self
    }
    pub fn hf_config(mut self, p: impl Into<PathBuf>) -> Self {
        self.hf_config = Some(p.into());
        self
    }
    pub fn config(mut self, c: NemotronOmniVisionConfig) -> Self {
        self.config = Some(c);
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    pub fn build(self) -> Result<NemotronOmniVisionRunner> {
        let mmproj = self.mmproj.ok_or_else(|| anyhow!("mmproj path required"))?;
        let device = self.device.unwrap_or(Device::Cpu);
        let cfg = match (self.config, self.hf_config) {
            (Some(c), _) => c,
            (None, Some(p)) => NemotronOmniVisionConfig::from_hf_config_json(&p)
                .with_context(|| format!("rlx-nemotron-omni: parse {p:?}"))?,
            (None, None) => {
                return Err(anyhow!(
                    "rlx-nemotron-omni: either .config(...) or .hf_config(...) required"
                ));
            }
        };
        let mut wm = rlx_core::load_weight_map(&mmproj, &[])
            .with_context(|| format!("rlx-nemotron-omni: load weights {mmproj:?}"))?;
        let built = build_nemotron_omni_vision(&cfg, &mut wm)?;
        let typed = built.model.typed_params.clone();
        let pre = built.preprocess;
        let (graph, params) = rlx_core::flow_util::graph_from_built(built.model)?;
        let opts =
            rlx_core::flow_bridge::compile_options_for_profile(&CompileProfile::encoder(), device);
        let mut compiled = Session::new(device).compile_with(graph, &opts);
        rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);
        Ok(NemotronOmniVisionRunner {
            compiled,
            cfg,
            preprocess: pre,
            device,
        })
    }
}

pub struct NemotronOmniVisionRunner {
    compiled: rlx_runtime::CompiledGraph,
    cfg: NemotronOmniVisionConfig,
    preprocess: NemotronOmniPreprocessWeights,
    device: Device,
}

impl NemotronOmniVisionRunner {
    pub fn builder() -> NemotronOmniVisionRunnerBuilder {
        NemotronOmniVisionRunnerBuilder::default()
    }
    pub fn config(&self) -> &NemotronOmniVisionConfig {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn preprocessor(&self) -> NemotronOmniImagePreprocessor {
        NemotronOmniImagePreprocessor {
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
            .ok_or_else(|| anyhow!("nemotron-omni forward returned no output"))
    }
}

impl VisionTower for NemotronOmniVisionRunner {
    fn embed(&mut self, patches: &ImagePatches) -> Result<Vec<f32>> {
        self.embed_patches(patches)
    }
    #[allow(clippy::misnamed_getters)]
    fn hidden_size(&self) -> usize {
        self.cfg.projector_output_dim
    }
}

pub struct NemotronOmniIdentityProjector {
    pub output_dim: usize,
}
impl Projector for NemotronOmniIdentityProjector {
    fn project(&mut self, vision_embed: &[f32], num_patches: usize) -> Result<Vec<f32>> {
        debug_assert_eq!(vision_embed.len(), num_patches * self.output_dim);
        Ok(vision_embed.to_vec())
    }
    fn output_dim(&self) -> usize {
        self.output_dim
    }
}
