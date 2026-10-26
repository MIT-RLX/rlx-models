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

//! End-to-end FLUX.1 text-to-image.
//!
//! Memory order matters at this size: the text encoders run first and are
//! dropped (T5-XXL alone is ~9.5 GB at bf16) before the 24 GB denoiser is
//! compiled; the denoiser then stays resident for every later prompt of the
//! same geometry.

use anyhow::{Context, Result, bail, ensure};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

use crate::clip::{ClipTextEncoder, clip_ids};
use crate::config::{ClipTextConfig, Flux1Config, T5Config};
use crate::lora::LoraSet;
use crate::sampler::{euler, gaussian_noise, latent_hw, pack, schedule, unpack};
use crate::t5::{T5Encoder, t5_ids};
use crate::transformer::{DitGeometry, Flux1Dit};
use crate::vae::{Flux1VaeDecoder, Flux1VaeEncoder, load_flux1_vae, rgb_to_u8};
use crate::weights::{DitSource, TensorFile};
use rlx_flux2::vae::{Flux2VaeConfig, Flux2VaeWeights};

/// Where each FLUX.1 component lives.
#[derive(Debug, Clone)]
pub struct Flux1Paths {
    /// Denoiser: a BFL-layout `.safetensors` file (or a directory of shards).
    pub transformer: PathBuf,
    /// T5 encoder directory (`config.json`, weights, `tokenizer.json`).
    pub t5: PathBuf,
    /// CLIP directory (`config.json`, weights, `tokenizer.json`).
    pub clip: PathBuf,
    /// `ae.safetensors`, or a diffusers `vae/` directory.
    pub vae: PathBuf,
}

impl Flux1Paths {
    /// Find the components under `root`, accepting both the layout
    /// `just fetch-flux1-dev` writes and a diffusers-style tree.
    pub fn discover(root: &Path) -> Result<Self> {
        let first = |cands: &[&str]| cands.iter().map(|c| root.join(c)).find(|p| p.exists());
        let transformer = if root.is_file() {
            root.to_path_buf()
        } else {
            first(&[
                "flux1-dev.safetensors",
                "flux1-schnell.safetensors",
                "transformer.safetensors",
            ])
            .or_else(|| {
                std::fs::read_dir(root)
                    .ok()?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .find(|p| {
                        p.extension().is_some_and(|x| x == "safetensors")
                            && p.file_name()
                                .is_some_and(|n| n.to_string_lossy().starts_with("flux1"))
                    })
            })
            .with_context(|| {
                format!(
                    "no FLUX.1 transformer (flux1-*.safetensors) under {}",
                    root.display()
                )
            })?
        };
        let base = if root.is_file() {
            root.parent().unwrap_or(Path::new("."))
        } else {
            root
        };
        let find = |cands: &[&str], what: &str| -> Result<PathBuf> {
            cands
                .iter()
                .map(|c| base.join(c))
                .find(|p| p.exists())
                .with_context(|| {
                    format!("no {what} under {} (looked for {cands:?})", base.display())
                })
        };
        Ok(Self {
            transformer,
            t5: find(&["t5", "text_encoder_2"], "T5 encoder")?,
            clip: find(&["clip", "text_encoder"], "CLIP text encoder")?,
            vae: find(&["vae/ae.safetensors", "ae.safetensors", "vae"], "VAE")?,
        })
    }
}

/// Load-time settings.
#[derive(Debug, Clone)]
pub struct Flux1Options {
    pub device: Device,
    /// `(adapter file, scale)`, merged into the denoiser on load.
    pub loras: Vec<(PathBuf, f32)>,
    /// T5 sequence length (default: 512 for `[dev]`, 256 for `[schnell]`).
    pub t5_len: Option<usize>,
    /// Graph chunking (host memory during build ≈ one chunk at f32).
    pub doubles_per_graph: usize,
    pub singles_per_graph: usize,
    pub t5_layers_per_graph: usize,
    /// Keep the text encoders compiled between prompts (costs ~10 GB).
    pub keep_text_encoders: bool,
    /// Quantize the denoiser's projections at load ([`crate::weights::WeightQuant`]).
    pub weight_quant: Option<crate::weights::WeightQuant>,
    /// Quantize the T5 encoder's projections at load.
    pub t5_quant: Option<crate::weights::WeightQuant>,
}

impl Default for Flux1Options {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            loras: Vec::new(),
            t5_len: None,
            doubles_per_graph: 4,
            singles_per_graph: 8,
            t5_layers_per_graph: 6,
            keep_text_encoders: false,
            weight_quant: None,
            t5_quant: None,
        }
    }
}

/// Per-image settings.
#[derive(Debug, Clone)]
pub struct Flux1Sample {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// Distilled guidance (`[dev]` only; BFL default 3.5).
    pub guidance: f32,
    pub seed: u64,
    /// Initial noise `[16, latent_h, latent_w]`; generated from `seed` when absent.
    pub noise: Option<Vec<f32>>,
}

impl Default for Flux1Sample {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            steps: 28,
            guidance: 3.5,
            seed: 0,
            noise: None,
        }
    }
}

/// Latent steering for [`Flux1Pipeline::sample_latents_guided`].
#[derive(Debug, Clone)]
pub struct LatentGuide {
    /// `[latent_h · latent_w]`: 1 = generated freely, 0 = pinned to `target`.
    pub keep: Vec<f32>,
    /// `[16, latent_h, latent_w]` target latents.
    pub target: Vec<f32>,
    /// Optional starting latents (`[16, h, w]`): the run begins at the first
    /// scheduled time ≤ `start_t` from `init` re-noised to that level, instead
    /// of from pure noise — composition kept, details re-rendered.
    pub init: Option<Vec<f32>>,
    pub start_t: f32,
}

/// Prompt conditioning: T5 states `[t5_len, 4096]` + pooled CLIP `[768]`.
#[derive(Debug, Clone)]
pub struct PromptEmbeds {
    pub t5: Vec<f32>,
    pub pooled: Vec<f32>,
}

/// An RGB8 image.
#[derive(Debug, Clone)]
pub struct Flux1Image {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

impl Flux1Image {
    pub fn to_image(&self) -> Result<image::RgbImage> {
        image::RgbImage::from_raw(self.width as u32, self.height as u32, self.rgb.clone())
            .context("RGB buffer size mismatch")
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.to_image()?
            .save(path)
            .with_context(|| format!("writing {}", path.display()))
    }
}

struct TextEncoders {
    t5: T5Encoder,
    t5_tok: tokenizers::Tokenizer,
    clip: ClipTextEncoder,
    clip_tok: tokenizers::Tokenizer,
}

/// FLUX.1 pipeline.
pub struct Flux1Pipeline {
    pub cfg: Flux1Config,
    pub paths: Flux1Paths,
    pub opts: Flux1Options,
    dit_file: TensorFile,
    lora: LoraSet,
    dit: Option<Flux1Dit>,
    text: Option<TextEncoders>,
    vae: (Flux2VaeConfig, Flux2VaeWeights),
    vae_dec: Option<Flux1VaeDecoder>,
}

fn load_tokenizer(dir: &Path, alt: &str) -> Result<tokenizers::Tokenizer> {
    let cands = [
        dir.join("tokenizer.json"),
        dir.parent().unwrap_or(dir).join(alt).join("tokenizer.json"),
    ];
    let path = cands
        .iter()
        .find(|p| p.is_file())
        .with_context(|| format!("no tokenizer.json in {} or ../{alt}", dir.display()))?;
    tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

fn read_config(dir: &Path) -> Result<String> {
    let p = dir.join("config.json");
    std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))
}

impl Flux1Pipeline {
    pub fn open(paths: Flux1Paths, opts: Flux1Options) -> Result<Self> {
        let dit_file =
            TensorFile::open(&paths.transformer)?.with_common_prefix("model.diffusion_model.");
        if dit_file.contains("transformer_blocks.0.attn.to_q.weight") {
            bail!(
                "{} is a diffusers-layout transformer; rlx-flux1 reads the BFL single-file layout \
                 (flux1-dev.safetensors)",
                paths.transformer.display()
            );
        }
        let cfg = Flux1Config::from_shapes(|k| dit_file.shape(k))?;
        let modules: Vec<String> = dit_file
            .keys()
            .into_iter()
            .filter_map(|k| k.strip_suffix(".weight").map(str::to_string))
            .collect();
        let mut lora = LoraSet::default();
        for (path, scale) in &opts.loras {
            let n = lora.add_file(path, *scale, &cfg, &modules)?;
            eprintln!("[rlx-flux1] LoRA {} ×{scale}: {n} modules", path.display());
        }
        let vae = load_flux1_vae(&paths.vae)?;
        Ok(Self {
            cfg,
            paths,
            opts,
            dit_file,
            lora,
            dit: None,
            text: None,
            vae,
            vae_dec: None,
        })
    }

    /// T5 sequence length in use.
    pub fn t5_len(&self) -> usize {
        self.opts
            .t5_len
            .unwrap_or(if self.cfg.guidance_embed { 512 } else { 256 })
    }

    /// T5 tokens `prompt` needs (with the end token), without loading the
    /// encoders. The denoiser attends to every padded position, so its cost
    /// grows with [`Self::t5_len`]: at 160 px the text is 84 % of a step's
    /// tokens at 512.
    pub fn t5_token_count(&self, prompt: &str) -> Result<usize> {
        let tok = load_tokenizer(&self.paths.t5, "tokenizer_2")?;
        let enc = tok
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("t5 tokenize: {e}"))?;
        Ok(enc.get_ids().len())
    }

    /// Build the text encoders now (otherwise the first encode does), e.g.
    /// while other work runs.
    pub fn load_text_encoders(&mut self) -> Result<()> {
        self.text_encoders().map(|_| ())
    }

    /// Change the T5 length; only before the text encoders are built.
    pub fn set_t5_len(&mut self, len: usize) -> Result<()> {
        ensure!(
            self.text.is_none(),
            "set_t5_len after the text encoders were built"
        );
        self.opts.t5_len = Some(len);
        Ok(())
    }

    fn text_encoders(&mut self) -> Result<&mut TextEncoders> {
        if self.text.is_none() {
            let t0 = std::time::Instant::now();
            let t5_cfg: T5Config = serde_json::from_str(&read_config(&self.paths.t5)?)?;
            let clip_cfg = ClipTextConfig::from_json(&read_config(&self.paths.clip)?)?;
            let t5 = T5Encoder::compile_quant(
                &self.paths.t5,
                t5_cfg,
                self.t5_len(),
                self.opts.device,
                self.opts.t5_layers_per_graph,
                self.opts.t5_quant,
            )?;
            let clip = ClipTextEncoder::compile(&self.paths.clip, clip_cfg, self.opts.device)?;
            self.text = Some(TextEncoders {
                t5,
                t5_tok: load_tokenizer(&self.paths.t5, "tokenizer_2")?,
                clip,
                clip_tok: load_tokenizer(&self.paths.clip, "tokenizer")?,
            });
            verbose(format!(
                "text encoders ready in {:.1}s",
                t0.elapsed().as_secs_f32()
            ));
        }
        Ok(self.text.as_mut().expect("just set"))
    }

    /// Token ids `(t5, clip)` for `prompt`.
    pub fn tokenize(&mut self, prompt: &str) -> Result<(Vec<u32>, Vec<u32>)> {
        let len = self.t5_len();
        let te = self.text_encoders()?;
        let t5 = t5_ids(&te.t5_tok, prompt, len)?;
        let c = &te.clip.cfg;
        let clip = clip_ids(
            &te.clip_tok,
            prompt,
            c.max_position_embeddings,
            c.eos_token_id,
        )?;
        Ok((t5, clip))
    }

    /// Encode `prompt` (frees the encoders afterwards unless
    /// [`Flux1Options::keep_text_encoders`]).
    pub fn encode_prompt(&mut self, prompt: &str) -> Result<PromptEmbeds> {
        let (t5_ids, clip_ids) = self.tokenize(prompt)?;
        let t0 = std::time::Instant::now();
        let te = self.text_encoders()?;
        let t5 = te.t5.encode(&t5_ids)?;
        let pooled = te.clip.pooled(&clip_ids)?;
        verbose(format!(
            "prompt encoded in {:.1}s",
            t0.elapsed().as_secs_f32()
        ));
        if !self.opts.keep_text_encoders {
            self.text = None;
        }
        Ok(PromptEmbeds { t5, pooled })
    }

    fn dit_for(&mut self, geo: DitGeometry) -> Result<&mut Flux1Dit> {
        if self.dit.as_ref().is_none_or(|d| d.geo != geo) {
            self.dit = None;
            let t0 = std::time::Instant::now();
            let mut src = DitSource {
                file: &self.dit_file,
                lora: &self.lora,
                cfg: &self.cfg,
                quant: self.opts.weight_quant,
            };
            let dit =
                crate::precision::with_component(crate::precision::Component::Denoiser, || {
                    Flux1Dit::compile(
                        &self.cfg,
                        geo,
                        &mut src,
                        self.opts.device,
                        self.opts.doubles_per_graph,
                        self.opts.singles_per_graph,
                    )
                })?;
            verbose(format!(
                "denoiser compiled in {:.1}s",
                t0.elapsed().as_secs_f32()
            ));
            self.dit = Some(dit);
        }
        Ok(self.dit.as_mut().expect("just set"))
    }

    /// Run the sampler; returns the final latents `[16, latent_h, latent_w]`.
    pub fn sample_latents(&mut self, embeds: &PromptEmbeds, s: &Flux1Sample) -> Result<Vec<f32>> {
        let (lh, lw) = latent_hw(s.height, s.width);
        let c = self.cfg.in_channels / 4;
        let geo = DitGeometry {
            txt_len: embeds.t5.len() / self.cfg.context_dim,
            grid_h: lh / 2,
            grid_w: lw / 2,
        };
        let noise = match &s.noise {
            Some(n) => {
                anyhow::ensure!(
                    n.len() == c * lh * lw,
                    "noise has {} values, expected {}",
                    n.len(),
                    c * lh * lw
                );
                n.clone()
            }
            None => gaussian_noise(c * lh * lw, s.seed),
        };
        let ts = schedule(s.steps, geo.img_len(), self.cfg.guidance_embed);
        let dit = self.dit_for(geo)?;
        let t0 = std::time::Instant::now();
        let x = euler(pack(&noise, c, lh, lw), &ts, |x, t, i| {
            let v = dit.forward(x, &embeds.t5, &embeds.pooled, t, s.guidance)?;
            verbose(format!(
                "step {}/{} t={t:.4} ({:.1}s)",
                i + 1,
                s.steps,
                t0.elapsed().as_secs_f32()
            ));
            Ok(v)
        })?;
        Ok(unpack(&x, c, lh, lw))
    }

    /// [`Self::sample_latents`], steering part of the image toward fixed
    /// latents: after every step, outside `guide.keep` (1 = free, 0 = pinned,
    /// per latent pixel) the state is replaced by the target at that noise
    /// level, `(1 - t)·target + t·noise`, with the run's own initial noise —
    /// the flow-matching form of RePaint-style inpainting. With the same seed
    /// and prompt the free region is re-rendered against the new surroundings.
    pub fn sample_latents_guided(
        &mut self,
        embeds: &PromptEmbeds,
        s: &Flux1Sample,
        guide: &LatentGuide,
    ) -> Result<Vec<f32>> {
        let (lh, lw) = latent_hw(s.height, s.width);
        let c = self.cfg.in_channels / 4;
        anyhow::ensure!(
            guide.keep.len() == lh * lw && guide.target.len() == c * lh * lw,
            "guide size mismatch"
        );
        let geo = DitGeometry {
            txt_len: embeds.t5.len() / self.cfg.context_dim,
            grid_h: lh / 2,
            grid_w: lw / 2,
        };
        let noise = match &s.noise {
            Some(n) => n.clone(),
            None => gaussian_noise(c * lh * lw, s.seed),
        };
        let ts = schedule(s.steps, geo.img_len(), self.cfg.guidance_embed);
        // Start: pure noise, or `(1 - t)·init + t·noise` at the first scheduled
        // time ≤ `start_t` (re-render only the late, detail-setting steps).
        let (ts, mut x) = match &guide.init {
            Some(init) => {
                let k = ts
                    .iter()
                    .position(|&t| t <= guide.start_t)
                    .unwrap_or(ts.len() - 1);
                let t0 = ts[k];
                let x: Vec<f32> = init
                    .iter()
                    .zip(&noise)
                    .map(|(a, n)| (1.0 - t0) * a + t0 * n)
                    .collect();
                (ts[k..].to_vec(), x)
            }
            None => (ts, noise.clone()),
        };
        let dit = self.dit_for(geo)?;
        for (i, w) in ts.windows(2).enumerate() {
            let (t, t_next) = (w[0], w[1]);
            let v = unpack(
                &dit.forward(
                    &pack(&x, c, lh, lw),
                    &embeds.t5,
                    &embeds.pooled,
                    t,
                    s.guidance,
                )?,
                c,
                lh,
                lw,
            );
            let dt = t_next - t;
            x.iter_mut().zip(&v).for_each(|(a, b)| *a += dt * b);
            for ch in 0..c {
                for p in 0..lh * lw {
                    let k = guide.keep[p];
                    let j = ch * lh * lw + p;
                    let pinned = (1.0 - t_next) * guide.target[j] + t_next * noise[j];
                    x[j] = k * x[j] + (1.0 - k) * pinned;
                }
            }
            verbose(format!("guided step {}/{} t={t:.4}", i + 1, ts.len() - 1));
        }
        Ok(x)
    }

    /// Encode an RGB8 image (`height × width`) to denoiser-space latents.
    pub fn encode_image(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<f32>> {
        let mut enc =
            Flux1VaeEncoder::compile(&self.vae.0, &self.vae.1, height, width, self.opts.device)?;
        enc.encode_rgb8(rgb)
    }

    /// Latents → RGB8.
    pub fn decode(&mut self, latents: &[f32], lh: usize, lw: usize) -> Result<Flux1Image> {
        if self
            .vae_dec
            .as_ref()
            .is_none_or(|d| (d.latent_h, d.latent_w) != (lh, lw))
        {
            self.vae_dec = Some(Flux1VaeDecoder::compile(
                &self.vae.0,
                &self.vae.1,
                lh,
                lw,
                self.opts.device,
            )?);
        }
        let rgb = self.vae_dec.as_mut().expect("just set").decode(latents)?;
        let (h, w) = (lh * 8, lw * 8);
        Ok(Flux1Image {
            width: w,
            height: h,
            rgb: rgb_to_u8(&rgb, h, w),
        })
    }

    /// Prompt → image.
    pub fn generate(&mut self, prompt: &str, s: &Flux1Sample) -> Result<Flux1Image> {
        let embeds = self.encode_prompt(prompt)?;
        self.generate_from_embeds(&embeds, s)
    }

    pub fn generate_from_embeds(
        &mut self,
        embeds: &PromptEmbeds,
        s: &Flux1Sample,
    ) -> Result<Flux1Image> {
        let latents = self.sample_latents(embeds, s)?;
        let (lh, lw) = latent_hw(s.height, s.width);
        let img = self.decode(&latents, lh, lw)?;
        Ok(crop(img, s.width, s.height))
    }

    /// Encode several prompts with one text-encoder load, then free the
    /// encoders — before the denoiser is compiled, so the two never coexist.
    pub fn encode_prompts(&mut self, prompts: &[&str]) -> Result<Vec<PromptEmbeds>> {
        let keep = self.opts.keep_text_encoders;
        self.opts.keep_text_encoders = true;
        let out = prompts
            .iter()
            .map(|p| self.encode_prompt(p))
            .collect::<Result<Vec<_>>>();
        self.opts.keep_text_encoders = keep;
        if !keep {
            self.text = None;
        }
        out
    }

    /// Drop the compiled text encoders.
    pub fn release_text_encoders(&mut self) {
        self.text = None;
    }

    /// Drop the compiled denoiser (frees its ~24 GB).
    pub fn release_denoiser(&mut self) {
        self.dit = None;
    }
}

/// Crop to the requested size (latents round the canvas up to 16 px).
fn crop(img: Flux1Image, w: usize, h: usize) -> Flux1Image {
    if img.width == w && img.height == h {
        return img;
    }
    let mut rgb = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        rgb.extend_from_slice(&img.rgb[y * img.width * 3..(y * img.width + w) * 3]);
    }
    Flux1Image {
        width: w,
        height: h,
        rgb,
    }
}

fn verbose(msg: String) {
    if rlx_ir::env::flag("RLX_FLUX1_VERBOSE") {
        eprintln!("[rlx-flux1] {msg}");
    }
}
