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

//! `rlx-genmoji` command line.

use anyhow::{Context, Result, bail};
use rlx_cli::{parse_standard_device, req};
use rlx_flux1::cli::{FLUX_FLAGS_HELP, FluxArgs};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

use crate::assistant::{
    DEFAULT_ASSISTANT_WEIGHTS, DEFAULT_TEMPERATURE, HttpAssistant, LocalAssistant, Metaprompt,
    PromptAssistant, assemble, classify_with, rewrite,
};
use crate::postprocess::{
    Cutout, apply_alpha_on_white, background_pockets, coarse_alpha, dilate, erode, frame_square,
    remove_white_background_hinted, remove_white_background_with, resample_mask, resize_rgb,
    resize_rgba, subject_box, triangulation_matte, white_prior,
};

const HELP: &str = "\
rlx-genmoji \"<description>\" [\"<description>\" …] [options]
rlx-genmoji --input generated.png [--input …]          (sticker step only)

genmoji:  --direct (skip the prompt rewrite)  --lora NAME|FILE (flux-dev | diverse-emoji; default flux-dev)
          --lora-scale F  --add-lora FILE[:S] (another LoRA on top, e.g. a few-step one)
          --fast (Hyper-SD 8-step LoRA, 6 steps, T5 at 128, packed Q6_K assistant)
          --export-web DIR (write the denoiser for genmoji.html in the browser: LoRAs merged, Q8_0 GGUF)
          --serve (keep every model loaded; one description per stdin line, `DESC<TAB>OUT.png`
          optional; prints each written path)  --genmoji-dir DIR (LoRAs + metaprompts; default weights/image/genmoji)
          --metaprompt NAME|FILE.md (default open-genmoji)  --rewrite-only (print the rewrite, exit)
          --free-rewrite (keep the LM's own add-on phrases instead of category assembly)
assistant: --assistant-weights DIR (local rlx LM; default weights/lm/qwen3.5-0.8b)
          --assistant-gguf FILE (its weights from a GGUF of the same model, packed)
          --sequential (rewrite, then load FLUX's text encoders: lower peak memory, slower)
          --assistant-device DEV (default: --device)
          --assistant-url URL [--assistant-model NAME --temperature T] (OpenAI-compatible server instead)
output:   -o FILE.png (one description) or DIR (several; default genmoji.png / genmoji/)
          --size PX (final side, default 800)  --sr NAME|FILE|none (super-resolution model;
          default RealESRGAN_x4plus_anime_6B from weights/image/upscale, else Lanczos)
          --keep-raw (also save the raw generations)  --opaque (keep the white)
          --two-pass (experimental: second render over black + triangulation matting; exact for
          white-on-white, but the re-render is still softer than the original, which halos edges)
          --rematte-strength T (how much of the trajectory the black render redoes, 0..1, default 0.5)
          --matte auto|lift|birefnet[:DIR]|classic (background removal; auto = the vendor's
          subject lifting from this Mac's Vision.framework, else BiRefNet_lite, else classic)
          --matte-size N (BiRefNet input side, default 512)
          --frame FILL|none (square framing: subject's long side = FILL of the canvas, centred,
          cut-off sides anchored, as the vendor's EmojiFramer; default 0.75)
          --strikes 40,64,96,160,320 (also write NAME@<size>.png at each emoji size)
          --segment FILE|none (classic only: SAM ViT-B for enclosed pockets; default
          weights/sam_v1/sam_vit_b_meta.safetensors when present)
          --cutout-flood D (background seed tolerance, default 10)  --cutout-band F (edge band, default 0.006)
          --cutout-pocket D (how grey an enclosed pocket may be, default 12)";

/// Resolve `--lora`: a file path, or a name under `<genmoji-dir>/lora/`.
fn resolve_lora(name: &str, dir: &Path) -> Result<PathBuf> {
    let p = PathBuf::from(name);
    if p.is_file() {
        return Ok(p);
    }
    let named = dir.join("lora").join(format!("{name}.safetensors"));
    if named.is_file() {
        return Ok(named);
    }
    bail!(
        "LoRA `{name}` not found (looked for {}); run `just fetch-genmoji`",
        named.display()
    )
}

/// `--fast`: Hyper-SD's 8-step FLUX.1-dev LoRA (under `--genmoji-dir`) and
/// its recommended scale.
const FAST_LORA: &str = "turbo/Hyper-FLUX.1-dev-8steps-lora.safetensors";
const FAST_LORA_SCALE: f32 = 0.125;
/// `--fast`: the assistant's weights as a packed Q6_K GGUF, when present
/// (Q8_0 calls "cat astronaut" a person).
const FAST_ASSISTANT_GGUF: &str = "weights/Qwen3.5-0.8B-gguf/Qwen3.5-0.8B-Q6_K.gguf";

/// Default super-resolution model, when present.
const DEFAULT_SR: &str = "weights/image/upscale/RealESRGAN_x4plus_anime_6B.pth";

fn resolve_sr(name: Option<&str>) -> Result<Option<PathBuf>> {
    match name {
        Some("none") | Some("lanczos") => Ok(None),
        Some(n) => {
            let p = PathBuf::from(n);
            if p.is_file() {
                return Ok(Some(p));
            }
            let named = Path::new("weights/image/upscale").join(format!("{n}.pth"));
            if named.is_file() {
                return Ok(Some(named));
            }
            bail!("super-resolution model `{n}` not found (run `just fetch-upscale-esrgan`)")
        }
        None => Ok(Some(PathBuf::from(DEFAULT_SR)).filter(|p| p.is_file())),
    }
}

/// Where the prompt rewrite runs.
enum AssistantSpec {
    Http {
        url: String,
        model: Option<String>,
        temperature: f32,
    },
    Local {
        dir: PathBuf,
        gguf: Option<PathBuf>,
        device: Device,
    },
}

impl AssistantSpec {
    fn open(self) -> Result<Box<dyn PromptAssistant>> {
        Ok(match self {
            Self::Http {
                url,
                model,
                temperature,
            } => {
                let mut a = HttpAssistant::new(&url);
                a.model = model;
                a.temperature = temperature;
                Box::new(a)
            }
            Self::Local { dir, gguf, device } => {
                Box::new(LocalAssistant::open_with(&dir, gguf.as_deref(), device)?)
            }
        })
    }
}

/// One description → the prompt FLUX sees.
/// `baseline` caches the classifier's calibration across descriptions.
fn rewrite_one(
    a: &mut dyn PromptAssistant,
    mp: &Metaprompt,
    p: &str,
    assemble_prompt: bool,
    baseline: &mut Option<Vec<f32>>,
) -> Result<String> {
    let t0 = std::time::Instant::now();
    let r = rewrite(a, mp, p)?;
    let r = if assemble_prompt {
        let t1 = std::time::Instant::now();
        let cat = classify_with(a, p, baseline)?;
        eprintln!(
            "[rlx-genmoji] {p:?} is {cat:?} (rewrite {:.2}s, classify {:.2}s)",
            (t1 - t0).as_secs_f32(),
            t1.elapsed().as_secs_f32()
        );
        assemble(&r, cat, p)
    } else {
        r
    };
    eprintln!("[rlx-genmoji] {p:?} → {r}");
    Ok(r)
}

/// File name for a description: lowercase words joined by `-`.
fn slug(s: &str) -> String {
    let mut out = String::new();
    for w in s
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .take(8)
    {
        if !out.is_empty() {
            out.push('-');
        }
        out.push_str(&w.to_lowercase());
    }
    if out.is_empty() {
        "genmoji".into()
    } else {
        out
    }
}

/// Default subject segmenter (SAM ViT-B, `just fetch-sam-v1`).
const DEFAULT_SAM: &str = "weights/sam_v1/sam_vit_b_meta.safetensors";

fn resolve_sam(name: Option<&str>) -> Result<Option<PathBuf>> {
    match name {
        Some("none") => Ok(None),
        Some(p) => {
            let p = PathBuf::from(p);
            if !p.is_file() {
                bail!(
                    "segmentation model {} not found (run `just fetch-sam-v1`)",
                    p.display()
                );
            }
            Ok(Some(p))
        }
        None => Ok(Some(PathBuf::from(DEFAULT_SAM)).filter(|p| p.is_file())),
    }
}

/// SAM with a box prompt around the subject: tells enclosed background
/// pockets (between an arm and a body, a head and a backpack) from the
/// subject, which a fill from the border cannot reach and a colour rule
/// cannot separate from pale parts of the subject.
pub struct SubjectSegmenter {
    sam: rlx_sam::Sam,
}

impl SubjectSegmenter {
    pub fn open(path: &Path, device: Device) -> Result<Self> {
        let p = path.to_str().context("non-utf8 SAM path")?;
        Ok(Self {
            sam: rlx_sam::Sam::from_safetensors_on(p, rlx_sam::SamConfig::vit_b(), device)?,
        })
    }

    /// Segment `hi` (the sharpest view of `lo`): enclosed background pockets at
    /// `lo`'s resolution, and the subject's soft mask at `hi`'s.
    pub fn segment(
        &mut self,
        hi: &image::RgbImage,
        lo: &image::RgbImage,
        c: Cutout,
    ) -> Result<Option<(Vec<bool>, Vec<f32>)>> {
        let Some((x0, y0, x1, y1)) = subject_box(lo, c) else {
            return Ok(None);
        };
        let (hw, hh) = (hi.width() as f32, hi.height() as f32);
        let (lw, lh) = (lo.width() as f32, lo.height() as f32);
        let side = self.sam.input_image_size() as f32;
        // SAM resizes the long side to `side` and pads bottom/right; prompts and
        // masks live in that frame.
        let s = side / hw.max(hh);
        let to_frame = |x: f32, lo_w: f32, hi_w: f32| x / lo_w * hi_w * s;
        let pad = 2.0;
        let bx = [
            to_frame((x0 as f32 - pad).max(0.0), lw, hw),
            to_frame((y0 as f32 - pad).max(0.0), lh, hh),
            to_frame((x1 as f32 + 1.0 + pad).min(lw), lw, hw),
            to_frame((y1 as f32 + 1.0 + pad).min(lh), lh, hh),
        ];
        let (pred, _) = self.sam.forward(
            hi.as_raw(),
            hh as usize,
            hw as usize,
            None,
            Some(&bx),
            None,
            true,
        )?;
        let m = pred.mask_side;
        let cell = side / m as f32;
        // Of SAM's candidate masks, the one covering most of what the border
        // fill left (the whole object, wings included — not just its body).
        let not_bg = crate::postprocess::not_background(lo, c);
        let frame_idx = |x: usize, y: usize| -> usize {
            let fx = ((x as f32 + 0.5) / lw * hw * s / cell) as usize;
            let fy = ((y as f32 + 0.5) / lh * hh * s / cell) as usize;
            fy.min(m - 1) * m + fx.min(m - 1)
        };
        let best = (0..pred.num_masks)
            .max_by_key(|&k| {
                let lg = &pred.mask_logits[k * m * m..(k + 1) * m * m];
                not_bg
                    .iter()
                    .enumerate()
                    .filter(|&(i, &nb)| {
                        nb && lg[frame_idx(i % lo.width() as usize, i / lo.width() as usize)] > 0.0
                    })
                    .count()
            })
            .unwrap_or(0);
        let logits = &pred.mask_logits[best * m * m..(best + 1) * m * m];
        let logit = |x: u32, y: u32| -> f32 {
            let fx = ((x as f32 + 0.5) / lw * hw * s / cell - 0.5).clamp(0.0, m as f32 - 1.0);
            let fy = ((y as f32 + 0.5) / lh * hh * s / cell - 0.5).clamp(0.0, m as f32 - 1.0);
            let (ix, iy) = (fx.floor() as usize, fy.floor() as usize);
            let (jx, jy) = ((ix + 1).min(m - 1), (iy + 1).min(m - 1));
            let (tx, ty) = (fx - ix as f32, fy - iy as f32);
            let v = |x: usize, y: usize| logits[y * m + x];
            (v(ix, iy) * (1.0 - tx) + v(jx, iy) * tx) * (1.0 - ty)
                + (v(ix, jy) * (1.0 - tx) + v(jx, jy) * tx) * ty
        };
        let pockets = background_pockets(lo, c, |x, y| logit(x, y) < -1.0);
        // Soft mask at `hi`: bilinear logits through a sigmoid.
        let soft: Vec<f32> = (0..(hw * hh) as usize)
            .map(|i| {
                let (x, y) = ((i % hw as usize) as f32, (i / hw as usize) as f32);
                let fx = ((x + 0.5) * s / cell - 0.5).clamp(0.0, m as f32 - 1.0);
                let fy = ((y + 0.5) * s / cell - 0.5).clamp(0.0, m as f32 - 1.0);
                let (ix, iy) = (fx.floor() as usize, fy.floor() as usize);
                let (jx, jy) = ((ix + 1).min(m - 1), (iy + 1).min(m - 1));
                let (tx, ty) = (fx - ix as f32, fy - iy as f32);
                let v = |x: usize, y: usize| logits[y * m + x];
                let l = (v(ix, iy) * (1.0 - tx) + v(jx, iy) * tx) * (1.0 - ty)
                    + (v(ix, jy) * (1.0 - tx) + v(jx, jy) * tx) * ty;
                1.0 / (1.0 + (-l).exp())
            })
            .collect();
        Ok(Some((pockets, soft)))
    }
}

/// What [`Sticker::plan_second_pass`] hands to [`Sticker::make_two`].
pub struct SecondPass {
    hi: image::RgbImage,
    alpha: Vec<f32>,
    pockets: Option<Vec<bool>>,
    /// Latent keep-mask for the black-background render.
    pub keep: Vec<f32>,
    /// Subject coverage per latent block (0..1), for its starting latents.
    pub cover: Vec<f32>,
}

/// A learned background matte.
pub enum Matter {
    /// The vendor's subject-lifting network, read from this Mac's Vision.framework
    /// (`rlx-subject-lift`).
    SubjectLift(rlx_subject_lift::SubjectLift),
    /// BiRefNet (MIT; `rlx-birefnet`) at a fixed input side.
    BiRefNet(rlx_birefnet::BiRefNet),
}

impl Matter {
    pub fn name(&self) -> String {
        match self {
            Matter::SubjectLift(_) => "subject lifting".into(),
            Matter::BiRefNet(m) => format!("BiRefNet @{}", m.side),
        }
    }

    /// Foreground alpha (0..1) at `img`'s size.
    pub fn alpha(&mut self, img: &image::RgbImage) -> Result<Vec<f32>> {
        match self {
            Matter::SubjectLift(m) => Ok(m.run(img)?.alpha(img.width(), img.height())),
            Matter::BiRefNet(m) => m.alpha(img),
        }
    }
}

/// T5 length for emoji prompts (see where it is applied in [`run`]).
const GENMOJI_T5_LEN: usize = 256;

/// Default open matting weights (`just fetch-birefnet`).
const DEFAULT_BIREFNET: &str = "weights/image/birefnet/BiRefNet_lite";

/// `--matte auto|lift|birefnet[:DIR]|classic`; `auto` prefers the vendor's model
/// (present on macOS 14+), then BiRefNet when downloaded, else the classic
/// white cut-out.
pub fn resolve_matter(spec: &str, side: usize, device: Device) -> Result<Option<Matter>> {
    let birefnet = |dir: &Path| -> Result<Matter> {
        Ok(Matter::BiRefNet(
            rlx_birefnet::BiRefNet::open(dir, side, device)
                .with_context(|| format!("loading BiRefNet from {}", dir.display()))?,
        ))
    };
    let (kind, arg) = spec
        .split_once(':')
        .map_or((spec, None), |(k, a)| (k, Some(a)));
    Ok(match kind {
        "classic" | "none" => None,
        "lift" => Some(Matter::SubjectLift(rlx_subject_lift::SubjectLift::open(
            arg.map(Path::new),
            device,
            Default::default(),
        )?)),
        "birefnet" => Some(birefnet(Path::new(arg.unwrap_or(DEFAULT_BIREFNET)))?),
        "auto" => {
            if Path::new(rlx_subject_lift::SYSTEM_MODEL).is_file() {
                Some(Matter::SubjectLift(rlx_subject_lift::SubjectLift::open(
                    None,
                    device,
                    Default::default(),
                )?))
            } else if Path::new(DEFAULT_BIREFNET)
                .join("model.safetensors")
                .is_file()
            {
                Some(birefnet(Path::new(DEFAULT_BIREFNET))?)
            } else {
                None
            }
        }
        k => bail!("--matte {k}: expected auto, lift, birefnet[:DIR] or classic"),
    })
}

/// The sticker step: super-resolve (or Lanczos), matte, resample to `size`.
pub struct Sticker {
    sr: Option<rlx_upscale::Upscaler>,
    seg: Option<SubjectSegmenter>,
    /// Learned matte; `None` = the classic white cut-out (+ SAM pockets).
    pub matter: Option<Matter>,
    /// Square framing fill (`None` = keep the generation's own framing).
    pub frame: Option<f32>,
    pub size: u32,
    pub cut: Cutout,
    pub opaque: bool,
}

impl Sticker {
    pub fn new(
        sr: Option<&Path>,
        sam: Option<&Path>,
        device: Device,
        size: u32,
        cut: Cutout,
        opaque: bool,
    ) -> Result<Self> {
        let sr = match sr {
            Some(p) => Some(
                rlx_upscale::Upscaler::open(p, device, rlx_upscale::UpscaleOptions::default())
                    .with_context(|| format!("loading super-resolution model {}", p.display()))?,
            ),
            None => None,
        };
        let seg = match sam {
            Some(p) if !opaque => Some(
                SubjectSegmenter::open(p, device)
                    .with_context(|| format!("loading SAM {}", p.display()))?,
            ),
            _ => None,
        };
        Ok(Self {
            sr,
            seg,
            matter: None,
            frame: Some(0.75),
            size,
            cut,
            opaque,
        })
    }

    fn super_resolve(&mut self, img: &image::RgbImage) -> Result<image::RgbImage> {
        let (w, h) = img.dimensions();
        let (tw, th) = (self.size, (self.size as u64 * h as u64 / w as u64) as u32);
        Ok(match &mut self.sr {
            Some(up) => {
                let (px, ow, oh) = up.upscale_rgb8(img.as_raw(), w as usize, h as usize)?;
                image::RgbImage::from_raw(ow as u32, oh as u32, px).context("SR output size")?
            }
            None => resize_rgb(img, tw, th),
        })
    }

    /// Enclosed background pockets (low resolution), if a segmenter is loaded.
    fn pockets(
        &mut self,
        hi: &image::RgbImage,
        lo: &image::RgbImage,
    ) -> Result<(Option<Vec<bool>>, Option<Vec<f32>>)> {
        Ok(match &mut self.seg {
            Some(seg) => match seg.segment(hi, lo, self.cut)? {
                Some((p, s)) => (Some(p), Some(s)),
                None => (None, None),
            },
            None => (None, None),
        })
    }

    /// Where a second, black-background render must be free to differ, at
    /// `lo`'s latent resolution (`/8`): every 8×8 block the subject touches,
    /// grown by one block. Also returns the coarse alpha and pockets for
    /// [`Self::make_two`].
    pub fn plan_second_pass(&mut self, lo: &image::RgbImage) -> Result<SecondPass> {
        let hi = self.super_resolve(lo)?;
        let (pockets, _) = self.pockets(&hi, lo)?;
        let alpha = coarse_alpha(lo, self.cut, pockets.as_deref());
        let (w, h) = (lo.width() as usize, lo.height() as usize);
        let touched: Vec<bool> = alpha.iter().map(|&a| a > 0.02).collect();
        let grown = dilate(&touched, w, h, 8);
        let (lw, lh) = (w.div_ceil(8), h.div_ceil(8));
        let keep: Vec<f32> = (0..lw * lh)
            .map(|b| {
                let (bx, by) = (b % lw * 8, b / lw * 8);
                let any =
                    (by..(by + 8).min(h)).any(|y| (bx..(bx + 8).min(w)).any(|x| grown[y * w + x]));
                if any { 1.0 } else { 0.0 }
            })
            .collect();
        // How much of each latent block is subject: the black render starts from
        // the white one only in proportion, so the band around the subject starts
        // black (not the white it was) and is still free to re-render the edge.
        let cover: Vec<f32> = (0..lw * lh)
            .map(|b| {
                let (bx, by) = (b % lw * 8, b / lw * 8);
                let (mut sum, mut n) = (0.0f32, 0usize);
                for y in by..(by + 8).min(h) {
                    for x in bx..(bx + 8).min(w) {
                        sum += alpha[y * w + x];
                        n += 1;
                    }
                }
                sum / n.max(1) as f32
            })
            .collect();
        Ok(SecondPass {
            hi,
            alpha,
            pockets,
            keep,
            cover,
        })
    }

    /// Sticker from two renders of the same subject (white and black
    /// background) by triangulation matting.
    pub fn make_two(
        &mut self,
        plan: SecondPass,
        lo_black: &image::RgbImage,
    ) -> Result<image::RgbaImage> {
        let (w, h) = (plan.hi.width() as usize, plan.hi.height() as usize);
        let (lw, lh) = (lo_black.width() as usize, lo_black.height() as usize);
        let hi_black = self.super_resolve(lo_black)?;
        let scale = (w / lw).max(1) as u32;
        let solid: Vec<bool> = plan.alpha.iter().map(|&a| a >= 0.999).collect();
        let sure = erode(&resample_mask(&solid, lw, lh, w, h), w, h, 2 * scale);
        let near: Vec<bool> = plan.alpha.iter().map(|&a| a > 0.0).collect();
        let near = dilate(&near, lw, lh, 3);
        let near: Vec<bool> = match &plan.pockets {
            Some(p) => near.iter().zip(p).map(|(&n, &pk)| n && !pk).collect(),
            None => near,
        };
        let free = resample_mask(&near, lw, lh, w, h);
        let (tw, th) = (self.size, (self.size as u64 * h as u64 / w as u64) as u32);
        Ok(resize_rgba(
            &triangulation_matte(&plan.hi, &hi_black, &sure, &free),
            tw,
            th,
        ))
    }

    pub fn make(&mut self, img: &image::RgbImage) -> Result<image::RgbaImage> {
        let (w, h) = img.dimensions();
        let (tw, th) = (self.size, (self.size as u64 * h as u64 / w as u64) as u32);
        let hi = match &mut self.sr {
            Some(up) => {
                let (px, ow, oh) = up.upscale_rgb8(img.as_raw(), w as usize, h as usize)?;
                image::RgbImage::from_raw(ow as u32, oh as u32, px).context("SR output size")?
            }
            None => resize_rgb(img, tw, th),
        };
        if self.opaque {
            return Ok(image::DynamicImage::ImageRgb8(resize_rgb(&hi, tw, th)).to_rgba8());
        }
        if let Some(m) = &mut self.matter {
            let mut a = m.alpha(&hi)?;
            if let Some(prior) = white_prior(&hi) {
                let mut added = 0usize;
                for (a, p) in a.iter_mut().zip(prior) {
                    if p > *a + 0.5 {
                        added += 1;
                    }
                    *a = a.max(p);
                }
                if added * 100 >= a.len() {
                    eprintln!(
                        "[rlx-genmoji] the white background showed {:.0}% more subject than the matte",
                        100.0 * added as f32 / a.len() as f32
                    );
                }
            }
            let rgba = apply_alpha_on_white(&hi, &a);
            return Ok(match self.frame {
                Some(fill) => frame_square(&rgba, tw.max(th), fill),
                None => resize_rgba(&rgba, tw, th),
            });
        }
        let (pockets, soft) = self.pockets(&hi, img)?;
        if let Some(p) = &pockets {
            let n = p.iter().filter(|&&b| b).count();
            if n > 0 {
                eprintln!("[rlx-genmoji] cleared {n} enclosed background pixels (SAM)");
            }
        }
        Ok(if hi.dimensions() == img.dimensions() {
            resize_rgba(
                &remove_white_background_with(&hi, self.cut, pockets.as_deref()),
                tw,
                th,
            )
        } else {
            resize_rgba(
                &remove_white_background_hinted(
                    &hi,
                    img,
                    self.cut,
                    pockets.as_deref(),
                    soft.as_deref(),
                ),
                tw,
                th,
            )
        })
    }
}

pub fn run(args: &[String]) -> Result<()> {
    let mut fa = FluxArgs::default();
    // Open Genmoji's settings: 160 px (the training size), 20 steps, seed 7.
    fa.sample.width = 160;
    fa.sample.height = 160;
    fa.sample.steps = 20;
    fa.sample.guidance = 5.0;
    fa.sample.seed = 7;
    let mut prompts: Vec<String> = Vec::new();
    let mut inputs: Vec<PathBuf> = Vec::new();
    let mut direct = false;
    let mut lora = "flux-dev".to_string();
    let mut lora_scale = 1.0f32;
    let mut genmoji_dir = PathBuf::from("weights/image/genmoji");
    let mut metaprompt = "open-genmoji".to_string();
    let mut url: Option<String> = None;
    let mut model: Option<String> = None;
    let mut local: Option<PathBuf> = None;
    let mut local_gguf: Option<PathBuf> = None;
    let mut sequential = false;
    let mut serve = false;
    let mut fast = false;
    let mut export_web: Option<PathBuf> = None;
    let mut assistant_device: Option<String> = None;
    let mut temperature = DEFAULT_TEMPERATURE;
    let mut out: Option<PathBuf> = None;
    let mut size = 800u32;
    let mut sr_name: Option<String> = None;
    let mut sam_name: Option<String> = None;
    let mut matte_spec = "auto".to_string();
    let mut matte_side = 512usize;
    let mut frame: Option<f32> = Some(0.75);
    let mut strikes: Vec<u32> = Vec::new();
    let mut opaque = false;
    let mut keep_raw = false;
    // Off by default: the black re-render is not yet sharp enough to match the
    // white one edge for edge, and triangulation turns every mismatch into a halo.
    let mut two_pass = false;
    let mut assemble_prompt = true;
    let mut rematte_t = 0.5f32;
    let mut rewrite_only = false;
    let mut cut = Cutout::default();
    let mut i = 0;
    while i < args.len() {
        let flag = |name: &str| args[i] == name;
        if [
            "--direct",
            "-d",
            "--opaque",
            "--rewrite-only",
            "--keep-raw",
            "--single-pass",
            "--two-pass",
            "--free-rewrite",
            "--sequential",
            "--serve",
            "--fast",
        ]
        .iter()
        .any(|f| flag(f))
        {
            match args[i].as_str() {
                "--serve" => serve = true,
                "--fast" => fast = true,
                "--single-pass" => two_pass = false,
                "--two-pass" => two_pass = true,
                "--opaque" => opaque = true,
                "--rewrite-only" => rewrite_only = true,
                "--keep-raw" => keep_raw = true,
                "--free-rewrite" => assemble_prompt = false,
                "--sequential" => sequential = true,
                _ => direct = true,
            }
            i += 1;
            continue;
        }
        match args[i].as_str() {
            "--lora" | "-l" => lora = req(args, &mut i)?,
            "--lora-scale" => lora_scale = req(args, &mut i)?.parse().context("--lora-scale")?,
            "--export-web" => export_web = Some(PathBuf::from(req(args, &mut i)?)),
            "--add-lora" => {
                // Stacked on the emoji LoRA, e.g. a few-step one.
                let v = req(args, &mut i)?;
                let (path, scale) = match v.rsplit_once(':') {
                    Some((p, s)) if s.parse::<f32>().is_ok() => (p.to_string(), s.parse()?),
                    _ => (v, 1.0),
                };
                fa.opts.loras.push((path.into(), scale));
            }
            "--genmoji-dir" => genmoji_dir = req(args, &mut i)?.into(),
            "--metaprompt" => metaprompt = req(args, &mut i)?,
            "--assistant-url" => url = Some(req(args, &mut i)?),
            "--assistant-model" => model = Some(req(args, &mut i)?),
            "--assistant-weights" => local = Some(req(args, &mut i)?.into()),
            "--assistant-gguf" => local_gguf = Some(req(args, &mut i)?.into()),
            "--assistant-device" => assistant_device = Some(req(args, &mut i)?),
            "--temperature" => temperature = req(args, &mut i)?.parse().context("--temperature")?,
            "--output" | "-o" => out = Some(req(args, &mut i)?.into()),
            "--input" | "-i" => inputs.push(req(args, &mut i)?.into()),
            "--size" => size = req(args, &mut i)?.parse().context("--size")?,
            "--upscale" | "-u" => {
                // Kept for the reference's spelling: final side = 160 × factor.
                let f: u32 = req(args, &mut i)?.parse().context("--upscale")?;
                size = fa.sample.width as u32 * f;
            }
            "--sr" => sr_name = Some(req(args, &mut i)?),
            "--rematte-strength" => {
                rematte_t = req(args, &mut i)?.parse().context("--rematte-strength")?
            }
            "--segment" => sam_name = Some(req(args, &mut i)?),
            "--matte" => matte_spec = req(args, &mut i)?,
            "--matte-size" => matte_side = req(args, &mut i)?.parse().context("--matte-size")?,
            "--cutout-pocket" => {
                cut.pocket = req(args, &mut i)?.parse().context("--cutout-pocket")?
            }
            "--cutout-flood" => cut.flood = req(args, &mut i)?.parse().context("--cutout-flood")?,
            "--frame" => {
                let v = req(args, &mut i)?;
                frame = if v == "none" {
                    None
                } else {
                    Some(v.parse().context("--frame")?)
                };
            }
            "--strikes" => {
                strikes = req(args, &mut i)?
                    .split(',')
                    .map(|v| v.trim().parse().context("--strikes"))
                    .collect::<Result<_>>()?;
            }
            "--cutout-band" => cut.band = req(args, &mut i)?.parse().context("--cutout-band")?,
            "--help" | "-h" => {
                eprintln!("{HELP}\n{FLUX_FLAGS_HELP}");
                return Ok(());
            }
            _ if fa.parse_flag(args, &mut i)? => {}
            s if !s.starts_with('-') => {
                prompts.push(s.to_string());
                i += 1;
            }
            other => bail!("unknown flag: {other} (see --help)"),
        }
    }
    if fast {
        // Few-step LoRA on top of the emoji one; 6 steps match 20 by eye.
        let turbo = genmoji_dir.join(FAST_LORA);
        if !turbo.is_file() {
            bail!(
                "--fast needs {} (`just fetch-genmoji-fast`)",
                turbo.display()
            );
        }
        fa.opts.loras.push((turbo, FAST_LORA_SCALE));
        if !args.iter().any(|a| a == "--steps") {
            fa.sample.steps = 6;
        }
        // 128 T5 positions cost what 64 do at 160 px (the matmul tiles
        // round up) and keep the composition of 256; 64 crops subjects.
        if fa.opts.t5_len.is_none() {
            fa.opts.t5_len = Some(128);
        }
        // Packed weights: ~5× faster rewrites than the f32 checkpoint.
        if local_gguf.is_none() && url.is_none() && Path::new(FAST_ASSISTANT_GGUF).is_file() {
            local_gguf = Some(FAST_ASSISTANT_GGUF.into());
        }
    }
    if let Some(dir) = &export_web {
        // Weights for the browser pipeline (`rlx-models-web`'s genmoji.html):
        // the denoiser as the native builder receives it, LoRAs merged, Q8_0.
        if fa.model_dir.is_none() && fa.transformer.is_none() {
            fa.model_dir = Some("weights/image/flux1-dev".into());
        }
        let mut loras = vec![(resolve_lora(&lora, &genmoji_dir)?, lora_scale)];
        loras.extend(fa.opts.loras.iter().cloned());
        std::fs::create_dir_all(dir)?;
        let name = if fast {
            "flux1-dev-genmoji-fast-q8_0.gguf"
        } else {
            "flux1-dev-genmoji-q8_0.gguf"
        };
        let out = dir.join(name);
        let t0 = std::time::Instant::now();
        let kept = out.is_file();
        let n = if kept {
            eprintln!("[rlx-genmoji] {} exists, kept", out.display());
            0
        } else {
            rlx_flux1::web::export_denoiser(
                &fa.paths()?.transformer,
                &loras,
                rlx_flux1::weights::WeightQuant::Q8_0,
                &out,
            )?
        };
        if !kept {
            eprintln!(
                "[rlx-genmoji] wrote {} ({n} tensors, {:.1} GB) in {:.0}s",
                out.display(),
                std::fs::metadata(&out)?.len() as f64 / 1e9,
                t0.elapsed().as_secs_f32()
            );
        }
        // CLIP and the VAE decoder, as stored (small).
        let out = dir.join("clip-vae.gguf");
        if !out.is_file() {
            let p = fa.paths()?;
            let n = rlx_flux1::web::export_aux(&p.clip, &p.vae, &out)?;
            eprintln!(
                "[rlx-genmoji] wrote {} ({n} tensors, {:.2} GB)",
                out.display(),
                std::fs::metadata(&out)?.len() as f64 / 1e9
            );
        }
        // T5 too, Q8_0: bf16 would need ~9.5 GB and six more workers.
        let out = dir.join("t5-v1_1-xxl-q8_0.gguf");
        if !out.is_file() {
            let t0 = std::time::Instant::now();
            let n = rlx_flux1::web::export_t5(
                &fa.paths()?.t5,
                rlx_flux1::weights::WeightQuant::Q8_0,
                &out,
            )?;
            eprintln!(
                "[rlx-genmoji] wrote {} ({n} tensors, {:.1} GB) in {:.0}s",
                out.display(),
                std::fs::metadata(&out)?.len() as f64 / 1e9,
                t0.elapsed().as_secs_f32()
            );
        }
        return Ok(());
    }
    let device = parse_standard_device("genmoji", fa.device.as_deref().unwrap_or("cpu"))?;
    let n_items = prompts.len() + inputs.len();
    if export_web.is_some() {
        // Exports take no descriptions.
    } else if serve {
        anyhow::ensure!(
            n_items == 0,
            "--serve reads descriptions from stdin, one per line"
        );
    } else if n_items == 0 {
        bail!("give a description, e.g. rlx-genmoji \"flying pig\" (or --input IMAGE)");
    }
    // One item: `-o` is the file. Several: `-o` is a directory.
    let target = |stem: &str| -> PathBuf {
        match (&out, n_items) {
            (Some(o), 1) => o.clone(),
            (Some(o), _) => o.join(format!("{stem}.png")),
            (None, 1) => PathBuf::from("genmoji.png"),
            (None, _) => PathBuf::from("genmoji").join(format!("{stem}.png")),
        }
    };
    let save = |img: &image::RgbaImage, path: &Path| -> Result<()> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        img.save(path)
            .with_context(|| format!("writing {}", path.display()))?;
        eprintln!("[rlx-genmoji] wrote {}", path.display());
        // Emoji "strikes": the sizes text renders at, each from the master.
        for &n in &strikes {
            let stem = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let p = path.with_file_name(format!("{stem}@{n}.png"));
            resize_rgba(img, n, n)
                .save(&p)
                .with_context(|| format!("writing {}", p.display()))?;
        }
        Ok(())
    };

    // 0. Sticker step only.
    if prompts.is_empty() && !serve {
        let mut sticker = Sticker::new(
            resolve_sr(sr_name.as_deref())?.as_deref(),
            resolve_sam(sam_name.as_deref())?.as_deref(),
            device,
            size,
            cut,
            opaque,
        )?;
        if !opaque {
            sticker.matter = resolve_matter(&matte_spec, matte_side, device)?;
        }
        sticker.frame = frame;
        eprintln!(
            "[rlx-genmoji] background removal: {}",
            sticker
                .matter
                .as_ref()
                .map_or("classic".into(), Matter::name)
        );
        for input in &inputs {
            let img = image::open(input)
                .with_context(|| format!("reading {}", input.display()))?
                .to_rgb8();
            let stem = input
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            save(&sticker.make(&img)?, &target(&format!("{stem}-sticker")))?;
        }
        return Ok(());
    }

    // 1. Prompt rewrite, on its own thread: the LM and FLUX's text encoders
    // load concurrently (they are independent until T5 encodes the result).
    let mp = match direct {
        true => None,
        false => Some(Metaprompt::load(
            &genmoji_dir.join("metaprompt"),
            &metaprompt,
        )?),
    };
    let mut spec = match (&url, direct) {
        (_, true) => None,
        (Some(u), _) => Some(AssistantSpec::Http {
            url: u.clone(),
            model: model.clone(),
            temperature,
        }),
        (None, _) => {
            let dir = local
                .clone()
                .unwrap_or_else(|| PathBuf::from(DEFAULT_ASSISTANT_WEIGHTS));
            if !dir.join("config.json").is_file() {
                bail!(
                    "assistant model {} not found: run `just fetch-qwen35-08b-instruct`, pass \
                     --assistant-weights DIR, or skip the rewrite with --direct",
                    dir.display()
                );
            }
            let device = parse_standard_device(
                "genmoji",
                assistant_device
                    .as_deref()
                    .or(fa.device.as_deref())
                    .unwrap_or("cpu"),
            )?;
            Some(AssistantSpec::Local {
                dir,
                gguf: local_gguf.clone(),
                device,
            })
        }
    };
    let rewrite_job: Option<std::thread::JoinHandle<Result<Vec<String>>>> = if direct || serve {
        None
    } else {
        let mp = mp.clone().expect("loaded unless --direct");
        let spec = spec.take().expect("resolved unless --direct");
        let prompts = prompts.clone();
        Some(
            std::thread::Builder::new()
                .name("genmoji-rewrite".into())
                .spawn(move || -> Result<Vec<String>> {
                    let mut assistant = spec.open()?;
                    let t0 = std::time::Instant::now();
                    let (mut v, mut baseline) = (Vec::new(), None);
                    for p in &prompts {
                        v.push(rewrite_one(
                            assistant.as_mut(),
                            &mp,
                            p,
                            assemble_prompt,
                            &mut baseline,
                        )?);
                    }
                    eprintln!(
                        "[rlx-genmoji] prompt rewrite took {:.1}s",
                        t0.elapsed().as_secs_f32()
                    );
                    Ok(v)
                })?,
        )
    };
    let finish =
        |job: Option<std::thread::JoinHandle<Result<Vec<String>>>>| -> Result<Vec<String>> {
            match job {
                Some(j) => j
                    .join()
                    .map_err(|_| anyhow::anyhow!("the prompt rewrite thread panicked"))?,
                None => Ok(prompts.clone()),
            }
        };
    if rewrite_only {
        for p in &finish(rewrite_job)? {
            println!("{p}");
        }
        return Ok(());
    }
    // A number format is set process-wide when FLUX opens; let the LM build
    // first so it is not rounded too.
    let mut rewrite_job = rewrite_job;
    let mut flux_prompts: Option<Vec<String>> = None;
    if fa.number_format.is_some() || rewrite_job.is_none() || sequential {
        flux_prompts = Some(finish(rewrite_job.take())?);
    }

    // 2. FLUX.1 [dev] + emoji LoRA, then 3. the sticker.
    fa.opts
        .loras
        .insert(0, (resolve_lora(&lora, &genmoji_dir)?, lora_scale));
    if fa.model_dir.is_none() && fa.transformer.is_none() {
        fa.model_dir = Some("weights/image/flux1-dev".into());
    }
    if !args.iter().any(|a| a == "--weight-quant") {
        // Q8_0 denoiser weights: renders match bf16 to 38 dB PSNR, each step
        // is faster on Metal, and the denoiser holds ~13 GB instead of 24.
        fa.opts.weight_quant = Some(rlx_flux1::weights::WeightQuant::Q8_0);
    }
    // Resident: the LM is built before FLUX (a number format, set when FLUX
    // opens, must not round it) and every stage stays compiled.
    let resident_lm = match serve {
        true => spec.take().map(AssistantSpec::open).transpose()?,
        false => None,
    };
    if serve {
        fa.opts.keep_text_encoders = true;
    }
    let mut pipe = fa.open("genmoji")?;
    // Emoji prompts are ~50 T5 tokens. 256 positions instead of FLUX.1
    // [dev]'s 512 halve each denoising step at 160 px and keep the
    // composition (128 changes it), so use 256 whenever everything fits.
    // While the rewrite is still running the length is provisional.
    let auto_len = fa.opts.t5_len.is_none();
    if auto_len {
        pipe.set_t5_len(GENMOJI_T5_LEN)?;
    }
    let new_sticker = || -> Result<Sticker> {
        let mut st = Sticker::new(
            resolve_sr(sr_name.as_deref())?.as_deref(),
            resolve_sam(sam_name.as_deref())?.as_deref(),
            device,
            size,
            cut,
            opaque,
        )?;
        if !opaque && !two_pass {
            st.matter = resolve_matter(&matte_spec, matte_side, device)?;
        }
        st.frame = frame;
        eprintln!(
            "[rlx-genmoji] background removal: {}",
            st.matter.as_ref().map_or("classic".into(), Matter::name)
        );
        Ok(st)
    };
    if serve {
        return serve_loop(
            &mut pipe,
            new_sticker()?,
            resident_lm,
            mp.as_ref(),
            assemble_prompt,
            &fa.sample,
            out.clone().unwrap_or_else(|| PathBuf::from("genmoji")),
            &save,
        );
    }
    if rewrite_job.is_some() {
        pipe.load_text_encoders()?;
    }
    let flux_prompts = match flux_prompts {
        Some(v) => v,
        None => finish(rewrite_job)?,
    };
    if auto_len {
        let mut longest = 0;
        for p in &flux_prompts {
            longest = longest.max(pipe.t5_token_count(p)?);
        }
        if longest > GENMOJI_T5_LEN {
            pipe.release_text_encoders();
            pipe.set_t5_len(512)?;
        }
    }
    // All prompts through the text encoders first, which are then freed: the
    // denoiser never shares memory with them.
    let refs: Vec<&str> = flux_prompts.iter().map(String::as_str).collect();
    let embeds = pipe.encode_prompts(&refs)?;
    let mut sticker: Option<Sticker> = None;
    let mut black_latent: Option<Vec<f32>> = None;
    for (user, emb) in prompts.iter().zip(&embeds) {
        let t0 = std::time::Instant::now();
        let (lh, lw) = rlx_flux1::sampler::latent_hw(fa.sample.height, fa.sample.width);
        let latents = pipe.sample_latents(emb, &fa.sample)?;
        let img = pipe.decode(&latents, lh, lw)?.to_image()?;
        eprintln!(
            "[rlx-genmoji] generated {}x{} in {:.1}s",
            img.width(),
            img.height(),
            t0.elapsed().as_secs_f32()
        );
        let path = target(&slug(user));
        if keep_raw {
            let raw = path.with_extension("raw.png");
            img.save(&raw)?;
            eprintln!("[rlx-genmoji] wrote {}", raw.display());
        }
        if sticker.is_none() {
            sticker = Some(new_sticker()?);
        }
        let st = sticker.as_mut().expect("just set");
        if two_pass && !opaque && img.width() % 16 == 0 && img.height() % 16 == 0 {
            // Second render of the same seed with the background pinned to black,
            // for triangulation matting.
            let t1 = std::time::Instant::now();
            let plan = st.plan_second_pass(&img)?;
            let (w, h) = (img.width() as usize, img.height() as usize);
            if black_latent.is_none() {
                black_latent = Some(pipe.encode_image(&vec![0u8; 3 * w * h], w, h)?);
            }
            // Composition from the white render, background black, the late half
            // of the trajectory re-rendered: same subject, edges over black.
            let black_z = black_latent.clone().expect("set");
            let plane = lh * lw;
            let init: Vec<f32> = latents
                .iter()
                .zip(&black_z)
                .enumerate()
                .map(|(j, (z, b))| {
                    let k = plan.cover[j % plane];
                    k * z + (1.0 - k) * b
                })
                .collect();
            let guide = rlx_flux1::LatentGuide {
                keep: plan.keep.clone(),
                target: black_z,
                init: Some(init),
                start_t: rematte_t,
            };
            let lat = pipe.sample_latents_guided(emb, &fa.sample, &guide)?;
            let black = pipe.decode(&lat, h / 8, w / 8)?.to_image()?;
            eprintln!(
                "[rlx-genmoji] black-background pass in {:.1}s",
                t1.elapsed().as_secs_f32()
            );
            if keep_raw {
                let raw = path.with_extension("raw-black.png");
                black.save(&raw)?;
                eprintln!("[rlx-genmoji] wrote {}", raw.display());
            }
            save(&st.make_two(plan, &black)?, &path)?;
            continue;
        }
        save(&st.make(&img)?, &path)?;
    }
    Ok(())
}

/// Warm-up description: compiles every stage before the first request.
const WARMUP: &str = "a red apple";

/// `--serve`: every model stays compiled and each stdin line is one request,
/// `DESCRIPTION` or `DESCRIPTION<TAB>OUT.png` (default `<dir>/<slug>.png`).
/// Prints each written path to stdout; a failed request is reported and the
/// loop goes on.
#[allow(clippy::too_many_arguments)]
fn serve_loop(
    pipe: &mut rlx_flux1::Flux1Pipeline,
    mut sticker: Sticker,
    mut lm: Option<Box<dyn PromptAssistant>>,
    mp: Option<&Metaprompt>,
    assemble_prompt: bool,
    sample: &rlx_flux1::Flux1Sample,
    dir: PathBuf,
    save: &dyn Fn(&image::RgbaImage, &Path) -> Result<()>,
) -> Result<()> {
    use std::io::{BufRead, Write};
    use std::time::Instant;
    let (lh, lw) = rlx_flux1::sampler::latent_hw(sample.height, sample.width);
    let mut baseline = None;
    // Two rewrites before the first request: the second is what lets the
    // assistant snapshot the metaprompt prefix they share.
    if let (Some(a), Some(mp)) = (lm.as_mut(), mp) {
        for d in [WARMUP, "a blue bird"] {
            rewrite_one(a.as_mut(), mp, d, assemble_prompt, &mut baseline)?;
        }
    }
    let mut handle = |desc: &str, path: Option<&Path>| -> Result<()> {
        let t0 = Instant::now();
        let prompt = match (lm.as_mut(), mp) {
            (Some(a), Some(mp)) => {
                rewrite_one(a.as_mut(), mp, desc, assemble_prompt, &mut baseline)?
            }
            _ => desc.to_string(),
        };
        let t1 = Instant::now();
        let emb = pipe.encode_prompt(&prompt)?;
        let t2 = Instant::now();
        let latents = pipe.sample_latents(&emb, sample)?;
        let t3 = Instant::now();
        let img = pipe.decode(&latents, lh, lw)?.to_image()?;
        let t4 = Instant::now();
        let rgba = sticker.make(&img)?;
        let t5 = Instant::now();
        if let Some(p) = path {
            save(&rgba, p)?;
            println!("{}", p.display());
            std::io::stdout().flush()?;
        }
        let s = |a: Instant, b: Instant| (b - a).as_secs_f32();
        eprintln!(
            "[rlx-genmoji] {desc:?} in {:.2}s: rewrite {:.2}, text {:.2}, {} steps {:.2}, \
             decode {:.2}, sticker {:.2}",
            s(t0, t5),
            s(t0, t1),
            s(t1, t2),
            sample.steps,
            s(t2, t3),
            s(t3, t4),
            s(t4, t5)
        );
        Ok(())
    };
    let t = Instant::now();
    handle(WARMUP, None)?;
    eprintln!(
        "[rlx-genmoji] ready ({:.1}s warm-up); one description per line",
        t.elapsed().as_secs_f32()
    );
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let (desc, path) = match line.split_once('\t') {
            Some((d, p)) => (d.trim(), PathBuf::from(p.trim())),
            None => (line.trim(), dir.join(format!("{}.png", slug(line.trim())))),
        };
        if desc.is_empty() {
            continue;
        }
        if let Err(e) = handle(desc, Some(&path)) {
            eprintln!("[rlx-genmoji] {desc:?} failed: {e:#}");
        }
    }
    Ok(())
}
