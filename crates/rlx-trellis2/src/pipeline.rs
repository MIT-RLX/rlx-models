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

//! End-to-end TRELLIS.2 image → mesh orchestration.
//!
//! Mirrors `Trellis2ImageTo3DPipeline.run` using host reference modules plus
//! [`rlx_dinov3`] for image conditioning. On non-CPU devices the generative
//! DiTs run through [`crate::dit_flow`] (AdaLN + backend SDPA); pass
//! `--eager-dit` / [`Trellis2RunnerBuilder::eager_dit`] to force the host path.

use crate::config::{
    Normalization, PipelineConfig, PipelineType, SamplerConfig as PipeSamplerConfig,
};
use crate::conv3d::Vol;
use crate::dit_flow::bucket_n_pos;
use crate::dit_host::dit_forward;
use crate::mesh::{Mesh, MeshWithPbr, dual_grid_to_mesh};
use crate::preprocess::{PreprocessOptions, PreprocessedImage, preprocess_image};
use crate::rng::SplitMix64;
use crate::rope::grid_coords;
use crate::sampler::{SamplerConfig, flow_euler_sample};
use crate::shape_decoder::{self, DecodedShape};
use crate::sparse::SparseTensor;
use crate::ss_decoder::{decode_occupancy, occupancy_to_coords};
use crate::weights::{
    CheckpointPaths, LoadedDit, LoadedSparseVae, LoadedSsDecoder, load_dit, load_sparse_vae,
    load_ss_decoder,
};
use anyhow::{Context, Result, bail};
use image::DynamicImage;
use rlx_dinov3::{DinoV3Config, DinoV3Runner};
use rlx_runtime::Device;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Builder for [`Trellis2Runner`].
pub struct Trellis2RunnerBuilder {
    model_dir: Option<PathBuf>,
    ss_decoder_dir: Option<PathBuf>,
    dinov3_weights: Option<PathBuf>,
    dinov3_config: Option<PathBuf>,
    device: Device,
    pipeline_type: Option<PipelineType>,
    shape_only: bool,
    max_num_tokens: usize,
    /// Override Euler steps for all samplers (`None` = pipeline.json).
    steps_override: Option<usize>,
    /// Skip DINOv3; use zero image conditioning (pipeline wiring / e2e without HF gated weights).
    no_dino: bool,
    /// Force host [`dit_forward`] even on Metal/MLX/CUDA.
    eager_dit: bool,
}

impl Default for Trellis2RunnerBuilder {
    fn default() -> Self {
        Self {
            model_dir: None,
            ss_decoder_dir: None,
            dinov3_weights: None,
            dinov3_config: None,
            device: Device::Cpu,
            pipeline_type: None,
            shape_only: false,
            max_num_tokens: 0,
            steps_override: None,
            no_dino: false,
            eager_dit: false,
        }
    }
}

impl Trellis2RunnerBuilder {
    pub fn model_dir(mut self, p: impl Into<PathBuf>) -> Self {
        self.model_dir = Some(p.into());
        self
    }
    pub fn ss_decoder_dir(mut self, p: impl Into<PathBuf>) -> Self {
        self.ss_decoder_dir = Some(p.into());
        self
    }
    pub fn dinov3_weights(mut self, p: impl Into<PathBuf>) -> Self {
        self.dinov3_weights = Some(p.into());
        self
    }
    pub fn dinov3_config(mut self, p: impl Into<PathBuf>) -> Self {
        self.dinov3_config = Some(p.into());
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = d;
        self
    }
    pub fn pipeline_type(mut self, t: PipelineType) -> Self {
        self.pipeline_type = Some(t);
        self
    }
    /// Skip texture DiT/VAE (shape mesh only).
    pub fn shape_only(mut self, v: bool) -> Self {
        self.shape_only = v;
        self
    }
    pub fn max_num_tokens(mut self, n: usize) -> Self {
        self.max_num_tokens = n;
        self
    }
    pub fn steps_override(mut self, n: usize) -> Self {
        self.steps_override = Some(n);
        self
    }
    /// Use zero DINOv3 features (no `--dinov3-weights` required).
    pub fn no_dino(mut self, v: bool) -> Self {
        self.no_dino = v;
        self
    }
    /// Force the host DiT path (skip compiled AdaLN/SDPA graph).
    pub fn eager_dit(mut self, v: bool) -> Self {
        self.eager_dit = v;
        self
    }

    pub fn build(self) -> Result<Trellis2Runner> {
        let model_dir = self
            .model_dir
            .ok_or_else(|| anyhow::anyhow!("model_dir is required"))?;
        if !self.no_dino && self.dinov3_weights.is_none() {
            bail!("dinov3_weights is required (or set no_dino)");
        }

        let (pipe, paths) = CheckpointPaths::resolve(&model_dir, self.ss_decoder_dir.as_deref())?;
        let pipeline_type = self.pipeline_type.unwrap_or_else(|| {
            PipelineType::parse(&pipe.args.default_pipeline_type)
                .unwrap_or(PipelineType::Cascade1024)
        });

        let need_tex = !self.shape_only;
        let missing = paths.missing_for(pipeline_type, need_tex);
        if !missing.is_empty() {
            bail!(
                "missing TRELLIS.2 checkpoints for {:?}:\n  - {}",
                pipeline_type,
                missing.join("\n  - ")
            );
        }

        let ss_flow = load_dit(&paths.sparse_structure_flow)?;
        let ss_dec = load_ss_decoder(&paths.sparse_structure_decoder)?;
        let shape_dec = load_sparse_vae(&paths.shape_slat_decoder)?;
        let shape_512 = load_dit(&paths.shape_slat_flow_512)?;
        let shape_1024 = paths
            .shape_slat_flow_1024
            .as_ref()
            .map(|p| load_dit(p))
            .transpose()?;
        let tex_dec = if need_tex {
            Some(load_sparse_vae(paths.tex_slat_decoder.as_ref().unwrap())?)
        } else {
            None
        };
        let tex_512 = if need_tex {
            paths
                .tex_slat_flow_512
                .as_ref()
                .map(|p| load_dit(p))
                .transpose()?
        } else {
            None
        };
        let tex_1024 = if need_tex {
            paths
                .tex_slat_flow_1024
                .as_ref()
                .map(|p| load_dit(p))
                .transpose()?
        } else {
            None
        };

        let dino_template = {
            let mut cfg = match &self.dinov3_config {
                Some(p) => DinoV3Config::from_file(p)
                    .with_context(|| format!("reading dinov3 config {}", p.display()))?,
                None => DinoV3Config::vit_l16(512),
            };
            // Trellis `DinoV3FeatureExtractor` uses non-affine F.layer_norm.
            cfg.final_layer_norm_affine = false;
            cfg
        };

        Ok(Trellis2Runner {
            pipe,
            paths,
            pipeline_type,
            shape_only: self.shape_only,
            max_num_tokens: if self.max_num_tokens == 0 {
                49_152
            } else {
                self.max_num_tokens
            },
            steps_override: self.steps_override,
            no_dino: self.no_dino,
            eager_dit: self.eager_dit,
            device: self.device,
            ss_flow,
            ss_dec,
            shape_dec,
            shape_512,
            shape_1024,
            tex_dec,
            tex_512,
            tex_1024,
            dino_weights: self.dinov3_weights.unwrap_or_default(),
            dino_template,
            dino_512: None,
            dino_1024: None,
        })
    }
}

/// Loaded TRELLIS.2 pipeline.
pub struct Trellis2Runner {
    pub pipe: PipelineConfig,
    pub paths: CheckpointPaths,
    pub pipeline_type: PipelineType,
    pub shape_only: bool,
    pub max_num_tokens: usize,
    pub steps_override: Option<usize>,
    pub no_dino: bool,
    pub eager_dit: bool,
    pub device: Device,
    ss_flow: LoadedDit,
    ss_dec: LoadedSsDecoder,
    shape_dec: LoadedSparseVae,
    shape_512: LoadedDit,
    shape_1024: Option<LoadedDit>,
    tex_dec: Option<LoadedSparseVae>,
    tex_512: Option<LoadedDit>,
    tex_1024: Option<LoadedDit>,
    dino_weights: PathBuf,
    dino_template: DinoV3Config,
    dino_512: Option<DinoV3Runner>,
    dino_1024: Option<DinoV3Runner>,
}

/// Generation inputs.
pub struct Trellis2Input<'a> {
    pub image: &'a DynamicImage,
    pub seed: u64,
    pub preprocess: PreprocessOptions,
}

/// Shape-only output.
pub struct Trellis2ShapeOutput {
    pub mesh: Mesh,
    pub resolution: usize,
    pub shape_slat: SparseTensor,
    pub subs: Vec<SparseTensor>,
}

/// Full (optional texture) output.
pub struct Trellis2Output {
    pub mesh: MeshWithPbr,
    pub resolution: usize,
    pub shape_slat: SparseTensor,
    pub tex_slat: Option<SparseTensor>,
}

impl Trellis2Runner {
    pub fn builder() -> Trellis2RunnerBuilder {
        Trellis2RunnerBuilder::default()
    }

    /// Image → shape mesh (and optional PBR voxels when texture weights loaded).
    pub fn generate(&mut self, input: Trellis2Input<'_>) -> Result<Trellis2Output> {
        let seed = input.seed;
        let preprocess = input.preprocess;
        let image = input.image;
        let shape = self.generate_shape(Trellis2Input {
            image,
            seed,
            preprocess,
        })?;
        if self.shape_only {
            return Ok(Trellis2Output {
                mesh: MeshWithPbr {
                    mesh: shape.mesh,
                    coords: Vec::new(),
                    attrs: Vec::new(),
                    grid_size: shape.resolution,
                },
                resolution: shape.resolution,
                shape_slat: shape.shape_slat,
                tex_slat: None,
            });
        }

        let pre = preprocess_image(image, preprocess)?;
        let cond = self.cond_for_pipeline(&pre)?;
        let mut rng = SplitMix64::new(seed.wrapping_add(0x7e57_0001));
        let tex_slat = match self.pipeline_type {
            PipelineType::Res512 => {
                let flow = self
                    .tex_512
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("tex_slat_flow_512 not loaded"))?;
                sample_tex_slat(
                    flow,
                    &shape.shape_slat,
                    &self.pipe.args.shape_slat_normalization,
                    &self.pipe.args.tex_slat_normalization,
                    &sampler_from_pipe(&self.pipe.args.tex_slat_sampler, self.steps_override),
                    &cond.pos,
                    &cond.neg,
                    cond.n_cond,
                    self.device,
                    self.eager_dit,
                    self.max_num_tokens,
                    &mut rng,
                )?
            }
            _ => {
                let flow = self
                    .tex_1024
                    .as_mut()
                    .or(self.tex_512.as_mut())
                    .ok_or_else(|| anyhow::anyhow!("tex_slat_flow not loaded"))?;
                sample_tex_slat(
                    flow,
                    &shape.shape_slat,
                    &self.pipe.args.shape_slat_normalization,
                    &self.pipe.args.tex_slat_normalization,
                    &sampler_from_pipe(&self.pipe.args.tex_slat_sampler, self.steps_override),
                    &cond.pos,
                    &cond.neg,
                    cond.n_cond,
                    self.device,
                    self.eager_dit,
                    self.max_num_tokens,
                    &mut rng,
                )?
            }
        };

        let tex_dec = self
            .tex_dec
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("texture decoder not loaded"))?;
        let decoded_tex =
            shape_decoder::decode(&tex_dec.cfg, &tex_dec.weights, &tex_slat, Some(&shape.subs))?;
        // attrs = output * 0.5 + 0.5
        let mut attrs = decoded_tex.voxels.feats.clone();
        for v in &mut attrs {
            *v = *v * 0.5 + 0.5;
        }

        Ok(Trellis2Output {
            mesh: MeshWithPbr {
                mesh: shape.mesh,
                coords: decoded_tex.voxels.coords,
                attrs,
                grid_size: shape.resolution,
            },
            resolution: shape.resolution,
            shape_slat: shape.shape_slat,
            tex_slat: Some(tex_slat),
        })
    }

    /// Image → dual-grid mesh (no texture).
    pub fn generate_shape(&mut self, input: Trellis2Input<'_>) -> Result<Trellis2ShapeOutput> {
        let pre = preprocess_image(input.image, input.preprocess)?;
        let mut rng = SplitMix64::new(input.seed);

        let cond_512 = self.get_cond(&pre, 512)?;
        let ss_res = self.pipeline_type.sparse_structure_res();
        let coords = self.sample_sparse_structure(&cond_512, ss_res, &mut rng)?;

        let (shape_slat, resolution) = match self.pipeline_type {
            PipelineType::Res512 => {
                let slat = sample_shape_slat(
                    &mut self.shape_512,
                    &coords,
                    &self.pipe.args.shape_slat_normalization,
                    &sampler_from_pipe(&self.pipe.args.shape_slat_sampler, self.steps_override),
                    &cond_512.pos,
                    &cond_512.neg,
                    cond_512.n_cond,
                    self.device,
                    self.eager_dit,
                    self.max_num_tokens,
                    &mut rng,
                )?;
                (slat, 512)
            }
            PipelineType::Res1024 => {
                let cond = self.get_cond(&pre, 1024)?;
                let flow = self
                    .shape_1024
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("shape_slat_flow_1024 not loaded"))?;
                let slat = sample_shape_slat(
                    flow,
                    &coords,
                    &self.pipe.args.shape_slat_normalization,
                    &sampler_from_pipe(&self.pipe.args.shape_slat_sampler, self.steps_override),
                    &cond.pos,
                    &cond.neg,
                    cond.n_cond,
                    self.device,
                    self.eager_dit,
                    self.max_num_tokens,
                    &mut rng,
                )?;
                (slat, 1024)
            }
            PipelineType::Cascade1024 => {
                self.sample_shape_cascade(&pre, &cond_512, &coords, 512, 1024, &mut rng)?
            }
            PipelineType::Cascade1536 => {
                self.sample_shape_cascade(&pre, &cond_512, &coords, 512, 1536, &mut rng)?
            }
        };

        let decoded = self.decode_shape(&shape_slat, resolution)?;
        let mesh = dual_grid_to_mesh(
            &decoded.voxels,
            resolution,
            [-0.5, -0.5, -0.5],
            [0.5, 0.5, 0.5],
        );
        Ok(Trellis2ShapeOutput {
            mesh,
            resolution,
            shape_slat,
            subs: decoded.subs,
        })
    }

    fn decode_shape(&self, slat: &SparseTensor, _resolution: usize) -> Result<DecodedShape> {
        shape_decoder::decode(&self.shape_dec.cfg, &self.shape_dec.weights, slat, None)
    }

    fn sample_sparse_structure(
        &mut self,
        cond: &CondBundle,
        target_res: usize,
        rng: &mut SplitMix64,
    ) -> Result<Vec<[i32; 3]>> {
        let res = self.ss_flow.cfg.args.resolution;
        let in_ch = self.ss_flow.cfg.args.in_channels;
        let out_ch = self.ss_flow.cfg.args.out_channels;
        let n_pos = res * res * res;
        let coords_rope = grid_coords(res);
        let noise = rng.gaussian_vec(in_ch * n_pos);
        let scfg = sampler_from_pipe(
            &self.pipe.args.sparse_structure_sampler,
            self.steps_override,
        );
        eprintln!(
            "rlx-trellis2: sampling sparse structure (steps={}, n_pos={}, target_res={})",
            scfg.steps, n_pos, target_res
        );

        let mut step_i = 0usize;
        let use_compiled = !self.eager_dit && self.device != Device::Cpu;
        let device = self.device;
        let n_cond = cond.n_cond;
        let model_v = |x_t: &[f32], t_scaled: f32, cnd: &[f32]| -> Vec<f32> {
            step_i += 1;
            let t0 = Instant::now();
            let tokens = cdhw_to_tokens(x_t, n_pos, in_ch);
            let out = if use_compiled {
                self.ss_flow
                    .forward_compiled(device, &tokens, &coords_rope, n_pos, cnd, n_cond, t_scaled)
                    .expect("ss compiled forward")
            } else {
                dit_forward(
                    &self.ss_flow.cfg,
                    &self.ss_flow.weights,
                    &tokens,
                    &coords_rope,
                    n_pos,
                    cnd,
                    n_cond,
                    t_scaled,
                    None,
                )
                .expect("ss dit forward")
            };
            eprintln!(
                "rlx-trellis2:   ss dit eval #{step_i} t={t_scaled:.1} {:.1}s{}",
                t0.elapsed().as_secs_f64(),
                if use_compiled {
                    " (compiled)"
                } else {
                    " (host)"
                }
            );
            tokens_to_cdhw(&out, n_pos, out_ch)
        };

        let sample = flow_euler_sample(model_v, &noise, &cond.pos, &cond.neg, &scfg);
        let latent = Vol {
            c: out_ch,
            d: res,
            h: res,
            w: res,
            data: sample,
        };
        let occ = decode_occupancy(&self.ss_dec.cfg, &self.ss_dec.weights, &latent)?;
        let with_batch = occupancy_to_coords(&occ, target_res);
        let mut coords: Vec<[i32; 3]> =
            with_batch.into_iter().map(|c| [c[1], c[2], c[3]]).collect();
        eprintln!(
            "rlx-trellis2: structure decode → {} active voxels @ {target_res}³",
            coords.len()
        );
        if coords.is_empty() {
            // Zero image cond / few Euler steps often yield an empty occupancy
            // field. Seed a small centered cube so shape DiT → VAE → mesh still
            // exercise the rest of the pipeline.
            let side = (target_res / 8).clamp(2, 4);
            let lo = ((target_res - side) / 2) as i32;
            eprintln!(
                "rlx-trellis2: empty occupancy — seeding {side}³ cube at ({lo},{lo},{lo}) for pipeline continuation"
            );
            for x in 0..side as i32 {
                for y in 0..side as i32 {
                    for z in 0..side as i32 {
                        coords.push([lo + x, lo + y, lo + z]);
                    }
                }
            }
        }
        Ok(coords)
    }

    fn sample_shape_cascade(
        &mut self,
        pre: &PreprocessedImage,
        lr_cond: &CondBundle,
        coords: &[[i32; 3]],
        lr_resolution: usize,
        hr_resolution: usize,
        rng: &mut SplitMix64,
    ) -> Result<(SparseTensor, usize)> {
        let lr = sample_shape_slat(
            &mut self.shape_512,
            coords,
            &self.pipe.args.shape_slat_normalization,
            &sampler_from_pipe(&self.pipe.args.shape_slat_sampler, self.steps_override),
            &lr_cond.pos,
            &lr_cond.neg,
            lr_cond.n_cond,
            self.device,
            self.eager_dit,
            self.max_num_tokens,
            rng,
        )?;

        let hr_coords =
            shape_decoder::upsample_coords(&self.shape_dec.cfg, &self.shape_dec.weights, &lr, 4)?;

        let mut hr_res = hr_resolution;
        let coords_hr = loop {
            let quant = quantize_cascade_coords(&hr_coords, lr_resolution, hr_res);
            let n = quant.len();
            if n < self.max_num_tokens || hr_res == 1024 {
                if hr_res != hr_resolution {
                    eprintln!(
                        "rlx-trellis2: token cap reduced cascade resolution to {hr_res} ({n} tokens)"
                    );
                }
                break quant;
            }
            hr_res -= 128;
        };

        let hr_cond = self.get_cond(pre, 1024)?;
        let device = self.device;
        let eager = self.eager_dit;
        let max_tok = self.max_num_tokens;
        let shape_norm = self.pipe.args.shape_slat_normalization.clone();
        let scfg = sampler_from_pipe(&self.pipe.args.shape_slat_sampler, self.steps_override);
        let flow = self
            .shape_1024
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("shape_slat_flow_1024 not loaded"))?;
        let slat = sample_shape_slat(
            flow,
            &coords_hr,
            &shape_norm,
            &scfg,
            &hr_cond.pos,
            &hr_cond.neg,
            hr_cond.n_cond,
            device,
            eager,
            max_tok,
            rng,
        )?;
        Ok((slat, hr_res))
    }

    fn cond_for_pipeline(&mut self, pre: &PreprocessedImage) -> Result<CondBundle> {
        match self.pipeline_type {
            PipelineType::Res512 => self.get_cond(pre, 512),
            _ => self.get_cond(pre, 1024),
        }
    }

    fn get_cond(&mut self, pre: &PreprocessedImage, resolution: usize) -> Result<CondBundle> {
        if self.no_dino {
            let mut cfg = self.dino_template.clone();
            cfg.image_size = resolution;
            let n_cond = cfg.seq_len();
            let dim = cfg.hidden_size;
            let pos = vec![0.0f32; n_cond * dim];
            let neg = pos.clone();
            return Ok(CondBundle { pos, neg, n_cond });
        }
        let runner = match resolution {
            512 => {
                if self.dino_512.is_none() {
                    self.dino_512 = Some(build_dino(
                        &self.dino_weights,
                        &self.dino_template,
                        512,
                        self.device,
                    )?);
                }
                self.dino_512.as_mut().unwrap()
            }
            1024 => {
                if self.dino_1024.is_none() {
                    self.dino_1024 = Some(build_dino(
                        &self.dino_weights,
                        &self.dino_template,
                        1024,
                        self.device,
                    )?);
                }
                self.dino_1024.as_mut().unwrap()
            }
            other => bail!("unsupported DINO resolution {other}"),
        };
        let out = runner.predict_image(&pre.rgb, pre.height, pre.width)?;
        let pos = out
            .tokens
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("dinov3 returned no tokens"))?;
        let n_cond = out.seq;
        let neg = vec![0.0f32; pos.len()];
        Ok(CondBundle { pos, neg, n_cond })
    }
}

struct CondBundle {
    pos: Vec<f32>,
    neg: Vec<f32>,
    n_cond: usize,
}

fn build_dino(
    weights: &Path,
    template: &DinoV3Config,
    img_size: usize,
    device: Device,
) -> Result<DinoV3Runner> {
    let mut cfg = template.clone();
    cfg.image_size = img_size;
    DinoV3Runner::builder()
        .weights(weights)
        .config(cfg)
        .device(device)
        .img_size(img_size)
        .build()
}

fn sampler_from_pipe(s: &PipeSamplerConfig, steps_override: Option<usize>) -> SamplerConfig {
    let mut cfg = SamplerConfig::from_params(s.sigma_min(), &s.params);
    if let Some(steps) = steps_override {
        cfg.steps = steps;
    }
    cfg
}

fn normalize_feats(feats: &[f32], n: usize, c: usize, norm: &Normalization) -> Vec<f32> {
    let mut out = vec![0.0f32; feats.len()];
    for i in 0..n {
        for k in 0..c {
            out[i * c + k] = (feats[i * c + k] - norm.mean[k]) / norm.std[k];
        }
    }
    out
}

fn denormalize_feats(feats: &[f32], n: usize, c: usize, norm: &Normalization) -> Vec<f32> {
    let mut out = vec![0.0f32; feats.len()];
    for i in 0..n {
        for k in 0..c {
            out[i * c + k] = feats[i * c + k] * norm.std[k] + norm.mean[k];
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn sample_shape_slat(
    flow: &mut LoadedDit,
    coords: &[[i32; 3]],
    norm: &Normalization,
    scfg: &SamplerConfig,
    cond: &[f32],
    neg_cond: &[f32],
    n_cond: usize,
    device: Device,
    eager_dit: bool,
    max_num_tokens: usize,
    rng: &mut SplitMix64,
) -> Result<SparseTensor> {
    let in_ch = flow.cfg.args.in_channels;
    let out_ch = flow.cfg.args.out_channels;
    let n = coords.len();
    if n == 0 {
        bail!("shape SLat sampling requires at least one active voxel");
    }
    let noise = rng.gaussian_vec(n * in_ch);
    let coords_f = coords_to_f32(coords);
    let use_compiled = !eager_dit && device != Device::Cpu;
    let n_bucket = if use_compiled {
        bucket_n_pos(n, max_num_tokens)
    } else {
        n
    };
    eprintln!(
        "rlx-trellis2: sampling shape SLat (n_tokens={n}, bucket={n_bucket}, steps={}, C={in_ch})",
        scfg.steps
    );

    let mut step_i = 0usize;
    let model_v = |x_t: &[f32], t_scaled: f32, cnd: &[f32]| -> Vec<f32> {
        step_i += 1;
        let t0 = Instant::now();
        let out = if use_compiled {
            flow.forward_compiled_padded(device, x_t, &coords_f, n, n_bucket, cnd, n_cond, t_scaled)
                .expect("shape compiled forward")
        } else {
            dit_forward(
                &flow.cfg,
                &flow.weights,
                x_t,
                &coords_f,
                n,
                cnd,
                n_cond,
                t_scaled,
                None,
            )
            .expect("shape dit forward")
        };
        eprintln!(
            "rlx-trellis2:   shape dit eval #{step_i} t={t_scaled:.1} {:.1}s{}",
            t0.elapsed().as_secs_f64(),
            if use_compiled {
                " (compiled)"
            } else {
                " (host)"
            }
        );
        out
    };
    let sample = flow_euler_sample(model_v, &noise, cond, neg_cond, scfg);
    let feats = denormalize_feats(&sample, n, out_ch, norm);
    Ok(SparseTensor::new(feats, coords.to_vec(), out_ch))
}

#[allow(clippy::too_many_arguments)]
fn sample_tex_slat(
    flow: &mut LoadedDit,
    shape_slat: &SparseTensor,
    shape_norm: &Normalization,
    tex_norm: &Normalization,
    scfg: &SamplerConfig,
    cond: &[f32],
    neg_cond: &[f32],
    n_cond: usize,
    device: Device,
    eager_dit: bool,
    max_num_tokens: usize,
    rng: &mut SplitMix64,
) -> Result<SparseTensor> {
    let in_ch = flow.cfg.args.in_channels;
    let out_ch = flow.cfg.args.out_channels;
    let shape_c = shape_slat.c;
    let noise_c = in_ch.checked_sub(shape_c).with_context(|| {
        format!("tex DiT in_channels={in_ch} must exceed shape channels={shape_c}")
    })?;
    let n = shape_slat.n();
    if n == 0 {
        bail!("texture SLat sampling requires at least one shape voxel");
    }
    let shape_n = normalize_feats(&shape_slat.feats, n, shape_c, shape_norm);
    let noise = rng.gaussian_vec(n * noise_c);
    let coords_f = coords_to_f32(&shape_slat.coords);
    let use_compiled = !eager_dit && device != Device::Cpu;
    let n_bucket = if use_compiled {
        bucket_n_pos(n, max_num_tokens)
    } else {
        n
    };
    eprintln!(
        "rlx-trellis2: sampling tex SLat (n_tokens={n}, bucket={n_bucket}, steps={}, C={out_ch})",
        scfg.steps
    );

    let mut step_i = 0usize;
    let model_v = |x_t: &[f32], t_scaled: f32, cnd: &[f32]| -> Vec<f32> {
        step_i += 1;
        let t0 = Instant::now();
        let mut tokens = vec![0.0f32; n * in_ch];
        for i in 0..n {
            tokens[i * in_ch..i * in_ch + noise_c]
                .copy_from_slice(&x_t[i * noise_c..(i + 1) * noise_c]);
            tokens[i * in_ch + noise_c..i * in_ch + in_ch]
                .copy_from_slice(&shape_n[i * shape_c..(i + 1) * shape_c]);
        }
        let out = if use_compiled {
            flow.forward_compiled_padded(
                device, &tokens, &coords_f, n, n_bucket, cnd, n_cond, t_scaled,
            )
            .expect("tex compiled forward")
        } else {
            dit_forward(
                &flow.cfg,
                &flow.weights,
                &tokens,
                &coords_f,
                n,
                cnd,
                n_cond,
                t_scaled,
                None,
            )
            .expect("tex dit forward")
        };
        eprintln!(
            "rlx-trellis2:   tex dit eval #{step_i} t={t_scaled:.1} {:.1}s{}",
            t0.elapsed().as_secs_f64(),
            if use_compiled {
                " (compiled)"
            } else {
                " (host)"
            }
        );
        out
    };
    let sample = flow_euler_sample(model_v, &noise, cond, neg_cond, scfg);
    let feats = denormalize_feats(&sample, n, out_ch, tex_norm);
    Ok(SparseTensor::new(feats, shape_slat.coords.clone(), out_ch))
}

fn coords_to_f32(coords: &[[i32; 3]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(coords.len() * 3);
    for c in coords {
        out.push(c[0] as f32);
        out.push(c[1] as f32);
        out.push(c[2] as f32);
    }
    out
}

/// `((coord + 0.5) / lr_resolution * (hr_resolution / 16)).floor()`, unique.
fn quantize_cascade_coords(
    hr_coords: &[[i32; 3]],
    lr_resolution: usize,
    hr_resolution: usize,
) -> Vec<[i32; 3]> {
    let scale = (hr_resolution / 16) as f32 / lr_resolution as f32;
    let mut set = BTreeSet::new();
    for c in hr_coords {
        let q = [
            ((c[0] as f32 + 0.5) * scale).floor() as i32,
            ((c[1] as f32 + 0.5) * scale).floor() as i32,
            ((c[2] as f32 + 0.5) * scale).floor() as i32,
        ];
        set.insert(q);
    }
    set.into_iter().collect()
}

fn cdhw_to_tokens(x: &[f32], n_pos: usize, ch: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; n_pos * ch];
    for c in 0..ch {
        for p in 0..n_pos {
            t[p * ch + c] = x[c * n_pos + p];
        }
    }
    t
}

fn tokens_to_cdhw(t: &[f32], n_pos: usize, ch: usize) -> Vec<f32> {
    let mut x = vec![0.0f32; n_pos * ch];
    for p in 0..n_pos {
        for c in 0..ch {
            x[c * n_pos + p] = t[p * ch + c];
        }
    }
    x
}
