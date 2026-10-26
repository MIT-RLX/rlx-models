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

//! Prompt (+ reference images) → image, BFL `flux3/pipeline.py`.
//!
//! Components load one at a time so the machine never holds the text
//! encoder and the DiT on the device together: the prompts are encoded and
//! the encoder released, then the DiT is built, then the autoencoder.

use anyhow::{Context, Result, bail, ensure};
use image::RgbImage;
use rlx_flux1::weights::{TensorFile, WeightQuant};
use rlx_flux2::vae::{Flux2VaeConfig, Flux2VaeWeights};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

use crate::config::{Flux3Config, Mode, PackageConfig, SamplerDefaults};
use crate::dit::{ChainGeometry, MainChain, ModeStack, StepInputs, step_mods};
use crate::embed::{Conditioning, round_bf16};
use crate::rope::{latent_ids, text_ids};
use crate::sampler::{initial_noise, timesteps};
use crate::text::{TextEncoder, TextTokens};
use crate::vae::{AeDecoder, AeEncoder, DOWNSAMPLING_RATIO, load_ae, rgb_to_u8};
use crate::weights::DitSource;

/// At most this many reference images.
pub const MAX_REFERENCES: usize = 10;
/// The canvas of a single reference (padding included) stays under 4 MP.
pub const MATCH_MAX_PIXELS: usize = 4 * 1024 * 1024;
pub const MIN_SIDE: usize = 64;
pub const MAX_SIDE: usize = 4096;
/// Each reference of a multi-reference edit is fitted under 1 MP.
pub const MULTI_REFERENCE_MAX_PIXELS: usize = 1024 * 1024;

/// Where a package's components live.
#[derive(Debug, Clone)]
pub struct Flux3Paths {
    /// The package directory (holds `config.json`).
    pub root: PathBuf,
    /// BFL-layout DiT `model.safetensors`.
    pub dit: PathBuf,
    /// `text_encoder/` (Qwen3-VL-4B-Instruct).
    pub text_encoder: PathBuf,
    /// `ae.safetensors`, or a diffusers `vae/` directory.
    pub ae: PathBuf,
}

impl Flux3Paths {
    /// The layout of a `flux-3-image-klein[-base]` download. `root` may also
    /// be the DiT file itself, with the shared components next to it.
    pub fn discover(root: &Path) -> Result<Self> {
        let (base, dit) = if root.is_file() {
            (
                root.parent().unwrap_or(Path::new(".")).to_path_buf(),
                root.to_path_buf(),
            )
        } else {
            (root.to_path_buf(), root.join("model.safetensors"))
        };
        ensure!(
            dit.is_file(),
            "no DiT checkpoint {} (the BFL `model.safetensors`)",
            dit.display()
        );
        let text_encoder = base.join("text_encoder");
        ensure!(
            text_encoder.join("config.json").is_file(),
            "no text encoder under {}",
            text_encoder.display()
        );
        let ae = ["ae.safetensors", "vae"]
            .iter()
            .map(|c| base.join(c))
            .find(|p| p.exists())
            .with_context(|| format!("no autoencoder (ae.safetensors) under {}", base.display()))?;
        Ok(Self {
            root: base,
            dit,
            text_encoder,
            ae,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Flux3Options {
    pub device: Device,
    /// Transformer blocks per compiled graph (host memory during build ≈
    /// one chunk's weights).
    pub blocks_per_graph: usize,
    /// Quantize the DiT's projections at load.
    pub weight_quant: Option<WeightQuant>,
    /// Quantize the text encoder's projections at load.
    pub text_quant: Option<WeightQuant>,
    /// Keep the text encoder's graphs between prompts (~6.5 GB).
    pub keep_text_encoder: bool,
}

impl Default for Flux3Options {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            blocks_per_graph: 8,
            weight_quant: None,
            text_quant: None,
            keep_text_encoder: false,
        }
    }
}

/// Per-image settings; `None` takes the package default.
#[derive(Debug, Clone, Default)]
pub struct Flux3Sample {
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub steps: Option<usize>,
    /// CFG scale (base only).
    pub guidance: Option<f32>,
    /// Schedule time-shift.
    pub alpha: Option<f32>,
    pub seed: u64,
    /// Initial latents `[h·w, 128]`; drawn from `seed` when absent.
    pub noise: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct Flux3Image {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

impl Flux3Image {
    pub fn to_image(&self) -> Result<RgbImage> {
        RgbImage::from_raw(self.width as u32, self.height as u32, self.rgb.clone())
            .context("RGB buffer size mismatch")
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.to_image()?
            .save(path)
            .with_context(|| format!("writing {}", path.display()))
    }
}

/// A reference image ready for the autoencoder, and where the original sits
/// inside it (single references are padded to the 16-px grid).
#[derive(Debug, Clone)]
pub struct PreparedReference {
    pub image: RgbImage,
    /// `(left, top, right, bottom)` of the fitted image in a padded canvas.
    pub crop: Option<(u32, u32, u32, u32)>,
}

fn ceil16(v: usize) -> usize {
    v.div_ceil(DOWNSAMPLING_RATIO) * DOWNSAMPLING_RATIO
}

/// `(fitted size, canvas)` of a single reference: the canvas is the fitted
/// size rounded up to 16, at most 4 MP, sides 64–4096.
pub fn reference_canvas(width: usize, height: usize) -> ((usize, usize), (usize, usize)) {
    let (w, h) = (width as f64, height as f64);
    let mut scale = 1f64
        .min((MATCH_MAX_PIXELS as f64 / (w * h)).sqrt())
        .min(MAX_SIDE as f64 / w.max(h));
    loop {
        let fit = (
            ((w * scale).round() as usize).max(1),
            ((h * scale).round() as usize).max(1),
        );
        let canvas = (
            ceil16(fit.0).clamp(MIN_SIDE, MAX_SIDE),
            ceil16(fit.1).clamp(MIN_SIDE, MAX_SIDE),
        );
        if canvas.0 * canvas.1 <= MATCH_MAX_PIXELS {
            return (fit, canvas);
        }
        scale *= 0.995;
    }
}

/// numpy `pad` source index for `mode="reflect"` (`symmetric` repeats the edge).
fn pad_index(i: isize, n: isize, symmetric: bool) -> usize {
    if n == 1 {
        return 0;
    }
    let (period, mut j) = if symmetric {
        (2 * n, i)
    } else {
        (2 * n - 2, i)
    };
    j = j.rem_euclid(period);
    let k = if symmetric {
        if j < n { j } else { period - 1 - j }
    } else if j < n {
        j
    } else {
        period - j
    };
    k as usize
}

/// A single reference: fitted by [`reference_canvas`] and reflect-padded,
/// centred, to its canvas.
pub fn prepare_reference(image: &RgbImage) -> PreparedReference {
    let (fit, (cw, ch)) = reference_canvas(image.width() as usize, image.height() as usize);
    let img = if (fit.0 as u32, fit.1 as u32) != image.dimensions() {
        image::imageops::resize(
            image,
            fit.0 as u32,
            fit.1 as u32,
            image::imageops::FilterType::Lanczos3,
        )
    } else {
        image.clone()
    };
    let (w, h) = (img.width() as usize, img.height() as usize);
    if (cw, ch) == (w, h) {
        return PreparedReference {
            image: img,
            crop: None,
        };
    }
    let (left, top) = ((cw - w) / 2, (ch - h) / 2);
    let symmetric = w.min(h) <= DOWNSAMPLING_RATIO;
    let out = RgbImage::from_fn(cw as u32, ch as u32, |x, y| {
        let sx = pad_index(x as isize - left as isize, w as isize, symmetric);
        let sy = pad_index(y as isize - top as isize, h as isize, symmetric);
        *img.get_pixel(sx as u32, sy as u32)
    });
    PreparedReference {
        image: out,
        crop: Some((left as u32, top as u32, (left + w) as u32, (top + h) as u32)),
    }
}

/// Each reference of a multi-reference edit: under 1 MP (sides floored),
/// centre-cropped to the 16-px grid.
pub fn prepare_multi_reference(image: &RgbImage) -> RgbImage {
    let (mut w, mut h) = (image.width() as usize, image.height() as usize);
    let mut img = image.clone();
    if w * h > MULTI_REFERENCE_MAX_PIXELS {
        let s = (MULTI_REFERENCE_MAX_PIXELS as f64 / (w * h) as f64).sqrt();
        w = (w as f64 * s) as usize;
        h = (h as f64 * s) as usize;
        img = image::imageops::resize(
            image,
            w as u32,
            h as u32,
            image::imageops::FilterType::Lanczos3,
        );
    }
    let (cw, ch) = (
        w / DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO,
        h / DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO,
    );
    let (left, top) = ((w - cw) / 2, (h - ch) / 2);
    image::imageops::crop_imm(&img, left as u32, top as u32, cw as u32, ch as u32).to_image()
}

/// Encoded references: tokens `[Sc, 128]` (bf16-rounded) and their ids.
#[derive(Debug, Clone, Default)]
pub struct ReferenceTokens {
    pub tokens: Vec<f32>,
    pub ids: Vec<[f32; 4]>,
}

impl ReferenceTokens {
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Per-prompt state of a sampling run (see [`Flux3Pipeline::prepare`]).
pub struct Prepared {
    texts: Vec<TextTokens>,
    cond: ReferenceTokens,
    /// Kept only for a flow map, whose text/reference modulation moves.
    text_stacks: Vec<ModeStack>,
    cond_stack: Option<ModeStack>,
    /// `[B, St, D]` text early output, zero-filled per sample.
    txt: Vec<f32>,
    /// `[Sc, D]` reference early output.
    cond_h: Vec<f32>,
    mask: Vec<f32>,
    txt_len: usize,
    si: usize,
}

impl Prepared {
    fn run_stacks(&mut self, c: &Conditioning, d: usize, t_ref: Option<f32>) -> Result<()> {
        let v0 = c.vec(0.0, t_ref);
        let txt_mods = c.mode_mods(Mode::Text, &v0).early;
        let (b, st) = (self.texts.len(), self.txt_len);
        let mut txt = vec![0.0f32; b * st * d];
        for (i, (s, t)) in self.text_stacks.iter_mut().zip(&self.texts).enumerate() {
            let out = s.run(&t.tokens, &txt_mods)?;
            txt[i * st * d..i * st * d + out.len()].copy_from_slice(&out);
        }
        self.txt = txt;
        if let Some(s) = &mut self.cond_stack {
            let m = c.mode_mods(Mode::Cond, &v0).early;
            self.cond_h = s.run(&self.cond.tokens, &m)?;
        }
        Ok(())
    }
}

pub struct Flux3Pipeline {
    pub cfg: Flux3Config,
    pub pkg: PackageConfig,
    pub paths: Flux3Paths,
    pub opts: Flux3Options,
    dit_file: TensorFile,
    conditioning: Conditioning,
    text: Option<TextEncoder>,
    ae: (Flux2VaeConfig, Flux2VaeWeights),
    chain: Option<MainChain>,
    decoder: Option<AeDecoder>,
}

impl Flux3Pipeline {
    pub fn open(paths: Flux3Paths, opts: Flux3Options) -> Result<Self> {
        let pkg = PackageConfig::load(&paths.root)?;
        let dit_file = TensorFile::open(&paths.dit)?;
        let cfg = Flux3Config::from_shapes(|k| dit_file.shape(k))?.with_package(&pkg)?;
        let src = DitSource {
            file: &dit_file,
            cfg: &cfg,
            quant: None,
        };
        let conditioning = Conditioning::load(&src, &cfg)?;
        let ae = load_ae(&paths.ae)?;
        Ok(Self {
            cfg,
            pkg,
            paths,
            opts,
            dit_file,
            conditioning,
            text: None,
            ae,
            chain: None,
            decoder: None,
        })
    }

    /// The package's sampler defaults for this checkpoint.
    pub fn sampler_defaults(&self) -> SamplerDefaults {
        self.pkg.sampler(self.cfg.flow_map)
    }

    fn text_encoder(&mut self) -> Result<&mut TextEncoder> {
        if self.text.is_none() {
            self.text = Some(TextEncoder::open(
                &self.paths.text_encoder,
                self.pkg.text_layers(),
                self.pkg.pad_multiple(),
                self.pkg.max_tokens(),
                self.opts.device,
                self.opts.text_quant,
            )?);
        }
        Ok(self.text.as_mut().expect("opened above"))
    }

    /// Text tokens for `prompt`.
    pub fn encode_prompt(&mut self, prompt: &str) -> Result<TextTokens> {
        let t = self.text_encoder()?.encode(prompt)?;
        ensure!(
            t.tokens.len() == t.len * self.cfg.context_dim,
            "text width {} != DiT context width {}",
            t.tokens.len() / t.len.max(1),
            self.cfg.context_dim
        );
        Ok(t)
    }

    /// Free the text encoder's device memory (unless asked to keep it).
    pub fn release_text_encoder(&mut self) {
        if !self.opts.keep_text_encoder
            && let Some(t) = &mut self.text
        {
            t.release();
        }
    }

    /// Encode prepared references (sides multiples of 16).
    pub fn encode_references(&mut self, refs: &[RgbImage]) -> Result<ReferenceTokens> {
        ensure!(
            refs.len() <= MAX_REFERENCES,
            "at most {MAX_REFERENCES} references"
        );
        let mut out = ReferenceTokens::default();
        let (cfg, weights) = (&self.ae.0, &self.ae.1);
        for (i, r) in refs.iter().enumerate() {
            let (w, h) = (r.width() as usize, r.height() as usize);
            let mut enc = AeEncoder::compile(cfg, weights, h, w, self.opts.device)?;
            let t = enc.encode_rgb8(cfg, weights, r.as_raw())?;
            out.tokens.extend(t.into_iter().map(round_bf16));
            out.ids.extend(latent_ids(
                h / DOWNSAMPLING_RATIO,
                w / DOWNSAMPLING_RATIO,
                self.cfg.image_rope_scale,
                -(1.0 + i as f32),
            ));
        }
        Ok(out)
    }

    /// Prompt + raw reference images → image, with the reference's
    /// preparation (fit, pad, crop back) applied.
    pub fn generate(
        &mut self,
        prompt: &str,
        references: &[RgbImage],
        sample: &Flux3Sample,
    ) -> Result<Flux3Image> {
        ensure!(
            references.len() <= MAX_REFERENCES,
            "at most {MAX_REFERENCES} references"
        );
        let (refs, crop) = if references.len() > 1 {
            (
                references
                    .iter()
                    .map(prepare_multi_reference)
                    .collect::<Vec<_>>(),
                None,
            )
        } else {
            let p: Vec<PreparedReference> = references.iter().map(prepare_reference).collect();
            let crop = p
                .first()
                .and_then(|r| r.crop)
                .filter(|_| sample.width.is_none() && sample.height.is_none());
            (p.into_iter().map(|r| r.image).collect(), crop)
        };
        let d = self.sampler_defaults();
        let (dw, dh) = refs
            .first()
            .map(|r| (r.width() as usize, r.height() as usize))
            .unwrap_or((d.size[0], d.size[1]));
        let sample = Flux3Sample {
            width: Some(sample.width.unwrap_or(dw)),
            height: Some(sample.height.unwrap_or(dh)),
            ..sample.clone()
        };
        let guidance = sample.guidance.unwrap_or(d.guidance);
        let cfg_on = !self.cfg.flow_map && guidance != 1.0;
        let mut texts = Vec::new();
        if cfg_on {
            texts.push(self.encode_prompt("")?);
        }
        texts.push(self.encode_prompt(prompt)?);
        self.release_text_encoder();
        let cond = self.encode_references(&refs)?;
        let img = self.generate_from_tokens(&texts, &cond, &sample)?;
        Ok(match crop {
            Some((l, t, r, b)) => {
                let full = img.to_image()?;
                let c = image::imageops::crop_imm(&full, l, t, r - l, b - t).to_image();
                Flux3Image {
                    width: c.width() as usize,
                    height: c.height() as usize,
                    rgb: c.into_raw(),
                }
            }
            None => img,
        })
    }

    /// Sample from encoded inputs. `texts` is `[unconditional, conditional]`
    /// for CFG, or `[conditional]`.
    pub fn generate_from_tokens(
        &mut self,
        texts: &[TextTokens],
        cond: &ReferenceTokens,
        sample: &Flux3Sample,
    ) -> Result<Flux3Image> {
        let latents = self.denoise(texts, cond, sample)?;
        let d = self.sampler_defaults();
        let (w, h) = (
            sample.width.unwrap_or(d.size[0]),
            sample.height.unwrap_or(d.size[1]),
        );
        self.decode(&latents, h / DOWNSAMPLING_RATIO, w / DOWNSAMPLING_RATIO)
    }

    /// Compile and run what the sampling loop needs for these inputs: the
    /// text and reference early stacks (once, unless the checkpoint is a
    /// flow map) and the per-step chain for a `grid_h × grid_w` latent.
    pub fn prepare(
        &mut self,
        texts: &[TextTokens],
        cond: &ReferenceTokens,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Prepared> {
        let flow_map = self.cfg.flow_map;
        let batch = texts.len();
        ensure!(
            batch == 1 || (batch == 2 && !flow_map),
            "{batch} text samples: expected 1, or 2 for CFG"
        );
        let si = grid_h * grid_w;
        let quant = self.opts.weight_quant;
        let device = self.opts.device;
        let cfg = self.cfg.clone();
        let mut text_stacks = Vec::with_capacity(batch);
        let mut cond_stack = None;
        {
            let mut src = DitSource {
                file: &self.dit_file,
                cfg: &cfg,
                quant,
            };
            for t in texts {
                ensure!(
                    t.tokens.len() == t.len * cfg.context_dim,
                    "text tokens {} != {} x {}",
                    t.tokens.len(),
                    t.len,
                    cfg.context_dim
                );
                text_stacks.push(ModeStack::compile(
                    &cfg,
                    Mode::Text,
                    &text_ids(t.len),
                    &mut src,
                    device,
                )?);
            }
            if !cond.is_empty() {
                cond_stack = Some(ModeStack::compile(
                    &cfg,
                    Mode::Cond,
                    &cond.ids,
                    &mut src,
                    device,
                )?);
            }
        }
        let txt_len = texts.iter().map(|t| t.len).max().context("no text")?;
        let sc = cond.len();
        let masked = texts.iter().any(|t| t.len != txt_len);
        let mut mask = Vec::new();
        if masked {
            let s = si + sc + txt_len;
            mask = vec![1.0f32; batch * s];
            for (b, t) in texts.iter().enumerate() {
                for v in &mut mask[b * s + si + sc + t.len..(b + 1) * s] {
                    *v = 0.0;
                }
            }
        }
        let mut prep = Prepared {
            texts: texts.to_vec(),
            cond: cond.clone(),
            text_stacks,
            cond_stack,
            txt: Vec::new(),
            cond_h: Vec::new(),
            mask,
            txt_len,
            si,
        };
        prep.run_stacks(&self.conditioning, self.cfg.hidden, None)?;
        if !flow_map {
            prep.text_stacks.clear();
            prep.cond_stack = None;
        }

        let mut ids = latent_ids(grid_h, grid_w, self.cfg.image_rope_scale, 0.0);
        ids.extend_from_slice(&cond.ids);
        ids.extend(text_ids(txt_len));
        let geo = ChainGeometry {
            batch,
            grid_h,
            grid_w,
            cond_len: sc,
            txt_len,
            masked,
            ids,
        };
        if self.chain.as_ref().is_none_or(|c| c.geo != geo) {
            self.chain = None;
            let t0 = std::time::Instant::now();
            let mut src = DitSource {
                file: &self.dit_file,
                cfg: &cfg,
                quant,
            };
            self.chain = Some(MainChain::compile(
                &cfg,
                geo,
                &mut src,
                device,
                self.opts.blocks_per_graph,
            )?);
            if rlx_ir::env::flag("RLX_FLUX3_VERBOSE") {
                eprintln!(
                    "[rlx-flux3] DiT compiled in {:.1}s",
                    t0.elapsed().as_secs_f32()
                );
            }
        }
        Ok(prep)
    }

    /// Raw velocity `[B, h·w, 128]` of every sample at flow time `t` (and
    /// the flow map's jump target `t_ref`).
    pub fn velocity(
        &mut self,
        prep: &mut Prepared,
        x: &[f32],
        t: f32,
        t_ref: Option<f32>,
    ) -> Result<Vec<f32>> {
        if self.cfg.flow_map {
            prep.run_stacks(&self.conditioning, self.cfg.hidden, t_ref)?;
        }
        let m = step_mods(&self.conditioning, t, t_ref);
        let chain = self.chain.as_mut().context("call prepare first")?;
        let v = chain.run(&StepInputs {
            img: x,
            cond: &prep.cond_h,
            txt: &prep.txt,
            mask: &prep.mask,
            mod_early: &m.early,
            mod_single: &m.single,
            mod_final: &m.final_,
        })?;
        let n = prep.texts.len() * prep.si * self.cfg.in_channels;
        ensure!(v.len() == n, "velocity len {} != {n}", v.len());
        Ok(v)
    }

    /// The denoising loop: latent tokens `[h·w, 128]`.
    pub fn denoise(
        &mut self,
        texts: &[TextTokens],
        cond: &ReferenceTokens,
        sample: &Flux3Sample,
    ) -> Result<Vec<f32>> {
        let d = self.sampler_defaults();
        let (w, h) = (
            sample.width.unwrap_or(d.size[0]),
            sample.height.unwrap_or(d.size[1]),
        );
        if !w.is_multiple_of(DOWNSAMPLING_RATIO) || !h.is_multiple_of(DOWNSAMPLING_RATIO) {
            bail!("width and height must be multiples of {DOWNSAMPLING_RATIO} (got {w}x{h})");
        }
        let (gh, gw) = (h / DOWNSAMPLING_RATIO, w / DOWNSAMPLING_RATIO);
        let steps = sample.steps.unwrap_or(d.steps);
        let alpha = sample.alpha.unwrap_or(d.alpha);
        let guidance = sample.guidance.unwrap_or(d.guidance);
        let ic = self.cfg.in_channels;
        let si = gh * gw;
        let mut x = match &sample.noise {
            Some(n) => {
                ensure!(n.len() == si * ic, "noise len {} != {}", n.len(), si * ic);
                n.clone()
            }
            None => initial_noise(si * ic, sample.seed),
        };
        let verbose = rlx_ir::env::flag("RLX_FLUX3_VERBOSE");
        let mut prep = self.prepare(texts, cond, gh, gw)?;
        let flow_map = self.cfg.flow_map;
        for (i, w2) in timesteps(steps, alpha).windows(2).enumerate() {
            let (t, t_next) = (w2[0], w2[1]);
            let t0 = std::time::Instant::now();
            let v = self.velocity(&mut prep, &x, t, flow_map.then_some(t_next))?;
            let dt = t_next - t;
            if texts.len() == 2 {
                let (u, c) = v.split_at(si * ic);
                for ((xv, &uv), &cv) in x.iter_mut().zip(u).zip(c) {
                    *xv += dt * (uv + guidance * (cv - uv));
                }
            } else {
                for (xv, &vv) in x.iter_mut().zip(&v) {
                    *xv += dt * vv;
                }
            }
            if verbose {
                eprintln!(
                    "[rlx-flux3] step {}/{} t={t:.4} {:.2}s",
                    i + 1,
                    steps,
                    t0.elapsed().as_secs_f32()
                );
            }
        }
        Ok(x)
    }

    /// Free the DiT's device memory.
    pub fn release_dit(&mut self) {
        self.chain = None;
    }

    /// Latent tokens `[h·w, 128]` → image.
    pub fn decode(&mut self, tokens: &[f32], grid_h: usize, grid_w: usize) -> Result<Flux3Image> {
        if self
            .decoder
            .as_ref()
            .is_none_or(|d| (d.grid_h, d.grid_w) != (grid_h, grid_w))
        {
            self.decoder = None;
            self.decoder = Some(AeDecoder::compile(
                &self.ae.0,
                &self.ae.1,
                grid_h,
                grid_w,
                self.opts.device,
            )?);
        }
        let dec = self.decoder.as_mut().expect("compiled above");
        let rgb = dec.decode(&self.ae.0, &self.ae.1, tokens)?;
        let (h, w) = (grid_h * DOWNSAMPLING_RATIO, grid_w * DOWNSAMPLING_RATIO);
        Ok(Flux3Image {
            width: w,
            height: h,
            rgb: rgb_to_u8(&rgb, h, w),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canvas_rounds_up_to_the_grid() {
        assert_eq!(reference_canvas(1000, 700), ((1000, 700), (1008, 704)));
        assert_eq!(reference_canvas(1024, 1024), ((1024, 1024), (1024, 1024)));
        // 4000x3000 is 12 MP: scaled under 4 MP, canvas included.
        let ((fw, fh), (cw, ch)) = reference_canvas(4000, 3000);
        assert!(cw * ch <= MATCH_MAX_PIXELS && fw <= cw && fh <= ch);
        // Tiny images get the minimum canvas.
        assert_eq!(reference_canvas(10, 10).1, (64, 64));
    }

    #[test]
    fn padding_matches_numpy() {
        // numpy reflect of [0 1 2 3] by 2 each side: [2 1 | 0 1 2 3 | 2 1]
        let r: Vec<usize> = (-2..6).map(|i| pad_index(i, 4, false)).collect();
        assert_eq!(r, [2, 1, 0, 1, 2, 3, 2, 1]);
        // symmetric: [1 0 | 0 1 2 3 | 3 2]
        let s: Vec<usize> = (-2..6).map(|i| pad_index(i, 4, true)).collect();
        assert_eq!(s, [1, 0, 0, 1, 2, 3, 3, 2]);
    }

    #[test]
    fn single_reference_is_padded_and_centred() {
        let img = RgbImage::from_fn(100, 70, |x, y| image::Rgb([x as u8, y as u8, 0]));
        let p = prepare_reference(&img);
        assert_eq!(p.image.dimensions(), (112, 80));
        assert_eq!(p.crop, Some((6, 5, 106, 75)));
        assert_eq!(p.image.get_pixel(6, 5), img.get_pixel(0, 0));
        // Reflected column left of the image: x = 5 mirrors x = 1.
        assert_eq!(p.image.get_pixel(5, 5), img.get_pixel(1, 0));
        let m = prepare_multi_reference(&img);
        assert_eq!(m.dimensions(), (96, 64));
    }
}
