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

//! The prompt upsampler, `flux-3-image-klein-pup`.
//!
//! A Qwen3.5-9B fine-tune (`Qwen3_5ForConditionalGeneration`) that turns a
//! short request into the full scene caption the DiT expects. It runs on
//! `rlx-qwen35` (HF safetensors or a GGUF of the same model) with the
//! reference's decoding (BFL `flux3/pup.py`): the package's system prompt,
//! thinking off, greedy, presence penalty 1.5 for text-to-image and 0 for
//! edits, at most 3584 new tokens, stop on `<|im_end|>` / `<|endoftext|>`.

use crate::bbox::{Json, round_half_even};
use anyhow::{Context, Result, anyhow, bail};
use image::RgbImage;
use rlx_qwen35::{MultiImagePrompt, Qwen35Runner, Qwen35VisionEncoder, VisionEncodeOutput};
use rlx_runtime::Device;
use rlx_runtime::SampleOpts;
use rlx_text::chat::{ChatMessage, ChatRenderOptions, ChatTemplate};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Longest caption the reference lets the upsampler write.
pub const MAX_NEW_TOKENS: usize = 3584;
/// Presence penalty of a text-to-image request.
pub const T2I_PRESENCE_PENALTY: f32 = 1.5;

/// The upsampler's files: tokenizer, chat template and `system_prompts.json`
/// in `dir`, weights in `weights` (the same directory, or a GGUF).
#[derive(Debug, Clone)]
pub struct UpsamplerPaths {
    pub dir: PathBuf,
    pub weights: PathBuf,
}

impl UpsamplerPaths {
    /// The vision tower (`mmproj-*.gguf` in [`Self::dir`]; BF16 preferred),
    /// needed only to upsample edits.
    pub fn mmproj(&self) -> Option<PathBuf> {
        let mut found: Vec<PathBuf> = std::fs::read_dir(&self.dir)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension().is_some_and(|x| x == "gguf")
                    && p.file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with("mmproj"))
            })
            .collect();
        found.sort_by_key(|p| !p.to_string_lossy().contains("BF16"));
        found.into_iter().next()
    }

    /// `path` is the upsampler directory, or a GGUF whose directory holds
    /// the tokenizer files. A directory with a single `*.gguf` and no
    /// safetensors runs from the GGUF.
    pub fn discover(path: &Path) -> Result<Self> {
        if path.is_file() {
            let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
            return Ok(Self {
                dir,
                weights: path.to_path_buf(),
            });
        }
        let entries: Vec<PathBuf> = std::fs::read_dir(path)
            .with_context(|| format!("reading {}", path.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        let has_st = entries
            .iter()
            .any(|p| p.extension().is_some_and(|x| x == "safetensors"));
        let mut ggufs: Vec<&PathBuf> = entries
            .iter()
            .filter(|p| {
                p.extension().is_some_and(|x| x == "gguf")
                    && !p
                        .file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with("mmproj"))
            })
            .collect();
        ggufs.sort();
        let weights = if has_st {
            path.to_path_buf()
        } else {
            ggufs
                .first()
                .map(|p| (*p).clone())
                .with_context(|| format!("no upsampler weights in {}", path.display()))?
        };
        Ok(Self {
            dir: path.to_path_buf(),
            weights,
        })
    }
}

pub struct Upsampler {
    runner: Qwen35Runner,
    tokenizer: tokenizers::Tokenizer,
    template: ChatTemplate,
    prompts: HashMap<String, String>,
    stop: Vec<u32>,
    paths: UpsamplerPaths,
    device: Device,
    /// Opened with [`Self::open_for`]`(…, true)`.
    edits: bool,
    /// Loaded on the first edit.
    vision: Option<Qwen35VisionEncoder>,
    pub max_new_tokens: usize,
}

impl Upsampler {
    /// Open for text-to-image requests.
    pub fn open(paths: &UpsamplerPaths, device: Device) -> Result<Self> {
        Self::open_for(paths, device, false)
    }

    /// Open, with `edits` also compiling what [`Self::upsample_edit`] needs
    /// (the prefill over assembled text + image embeddings).
    pub fn open_for(paths: &UpsamplerPaths, device: Device, edits: bool) -> Result<Self> {
        let dir = &paths.dir;
        let tok = dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok)
            .map_err(|e| anyhow!("{}: {e}", tok.display()))?;
        let tcfg: serde_json::Value = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let source = match tcfg["chat_template"].as_str() {
            Some(t) => t.to_string(),
            None => std::fs::read_to_string(dir.join("chat_template.jinja"))
                .with_context(|| format!("no chat template in {}", dir.display()))?,
        };
        let template = ChatTemplate::from_source(source)?;
        let sp = dir.join("system_prompts.json");
        let prompts: HashMap<String, String> = serde_json::from_str(
            &std::fs::read_to_string(&sp).with_context(|| format!("reading {}", sp.display()))?,
        )
        .with_context(|| format!("parsing {}", sp.display()))?;
        let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| tokenizer.token_to_id(t))
            .collect();
        if stop.is_empty() {
            bail!("upsampler tokenizer has no <|im_end|>");
        }
        // The presence penalty is applied on the host, step by step, which
        // needs the host-fed decode; Metal and CUDA otherwise default to the
        // resident KV cache (an explicit setting wins).
        if rlx_ir::env::var_result("RLX_QWEN35_GPU_KV").is_err() {
            rlx_ir::env::set("RLX_QWEN35_GPU_KV", "0");
        }
        let gguf = paths.weights.extension().is_some_and(|e| e == "gguf");
        let mut b = Qwen35Runner::builder()
            .weights(&paths.weights)
            .device(device)
            .max_seq(8192)
            .skip_auto_mmproj(true)
            .hidden_prefill(edits);
        if gguf {
            b = b.packed_weights(true);
        }
        Ok(Self {
            runner: b.build()?,
            tokenizer,
            template,
            prompts,
            stop,
            paths: paths.clone(),
            device,
            edits,
            vision: None,
            max_new_tokens: MAX_NEW_TOKENS,
        })
    }

    /// The system prompt for `task` (`t2i`, `i2i`, `t2i_bbox`, `i2i_bbox`).
    pub fn system_prompt(&self, task: &str) -> Result<&str> {
        self.prompts
            .get(task)
            .map(String::as_str)
            .with_context(|| format!("system_prompts.json has no `{task}`"))
    }

    /// Token ids of the chat for `request` under `task`'s system prompt.
    pub fn prompt_ids(&self, task: &str, request: &str) -> Result<Vec<u32>> {
        let messages = [
            ChatMessage::system(self.system_prompt(task)?),
            ChatMessage::user(request),
        ];
        let text = self.template.render_with_options(
            &messages,
            ChatRenderOptions {
                add_generation_prompt: true,
                enable_thinking: false,
                ..ChatRenderOptions::default()
            },
        )?;
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("upsampler tokenize: {e}"))?
            .get_ids()
            .to_vec())
    }

    /// Generate from prompt ids with `presence_penalty` over the generated
    /// tokens, greedy; stops at the end of the turn.
    pub fn generate_ids(&mut self, ids: &[u32], presence_penalty: f32) -> Result<Vec<u32>> {
        let stop = self.stop.clone();
        let max = self
            .max_new_tokens
            .min(self.runner.max_seq().saturating_sub(ids.len() + 1));
        let mut on_token = |t: u32| !stop.contains(&t);
        let mut out = if presence_penalty != 0.0 {
            let opts = SampleOpts {
                presence_penalty,
                ..SampleOpts::greedy()
            };
            self.runner.set_sampling(Some((opts, 0)))?;
            let r = self.runner.generate_stepwise(ids, max, &mut on_token);
            self.runner.set_sampling(None)?;
            r?
        } else {
            self.runner.generate(ids, max, &mut on_token)?
        };
        if out.last().is_some_and(|t| stop.contains(t)) {
            out.pop();
        }
        Ok(out)
    }

    /// Text-to-image: a short request → one full scene caption.
    pub fn upsample(&mut self, request: &str) -> Result<String> {
        self.caption_t2i("t2i", request)
    }

    /// Any request, BFL `Upsampler.__call__`: text-to-image or an edit (with
    /// `images`), in layout mode when `boxes` (0-1 coordinates) are given
    /// or the request already ends in a box table, whose locked rows are then
    /// re-imposed on the answer ([`crate::bbox::reconcile`]).
    pub fn upsample_request(
        &mut self,
        request: &str,
        images: &[RgbImage],
        boxes: Option<&[Json]>,
    ) -> Result<String> {
        let plan = crate::bbox::plan(request, boxes, images.len())?;
        let caption = if images.is_empty() {
            self.caption_t2i(&plan.task, &plan.text)?
        } else {
            let scaffold = if plan.locked.is_some() {
                I2I_SCAFFOLD_LOCKED
            } else {
                I2I_SCAFFOLD
            };
            self.caption_i2i(
                &plan.task,
                &scaffold.replace("{instruction}", &plan.text),
                images,
            )?
        };
        Ok(match &plan.locked {
            Some(locked) => crate::bbox::reconcile(&caption, locked),
            None => caption,
        })
    }

    fn caption_t2i(&mut self, task: &str, text: &str) -> Result<String> {
        let ids = self.prompt_ids(task, text)?;
        let out = self.generate_ids(&ids, T2I_PRESENCE_PENALTY)?;
        let caption = self
            .tokenizer
            .decode(&out, true)
            .map_err(|e| anyhow!("upsampler decode: {e}"))?
            .trim()
            .to_string();
        if caption.is_empty() {
            bail!("the upsampler returned an empty caption");
        }
        Ok(caption)
    }

    /// Edit: a request plus the reference images → the full caption of the
    /// edited image (the `i2i` system prompt, greedy, no presence penalty).
    pub fn upsample_edit(&mut self, request: &str, images: &[RgbImage]) -> Result<String> {
        self.upsample_request(request, images, None)
    }

    /// An edit's caption from the user text after the images.
    fn caption_i2i(&mut self, task: &str, scaffold: &str, images: &[RgbImage]) -> Result<String> {
        if !self.edits {
            bail!("upsampler opened for text-to-image only; open it with `open_for(…, true)`");
        }
        if images.len() > MAX_REFERENCES {
            bail!("the upsampler takes at most {MAX_REFERENCES} references");
        }
        let refs: Vec<RgbImage> = images
            .iter()
            .map(|i| policy_reference(i, images.len()))
            .collect::<Result<_>>()?;
        let vision = self.encode_images(&refs)?;
        let user = format!("{}{scaffold}", rlx_qwen35::MEDIA_MARKER.repeat(refs.len()));
        let messages = [
            ChatMessage::system(self.system_prompt(task)?),
            ChatMessage::user(user),
        ];
        let text = self.template.render_with_options(
            &messages,
            ChatRenderOptions {
                add_generation_prompt: true,
                enable_thinking: false,
                ..ChatRenderOptions::default()
            },
        )?;
        let tokenizer = &self.tokenizer;
        let runner = &self.runner;
        let prefill = MultiImagePrompt {
            prompt: &text,
            images: &vision,
        }
        .assemble_with(
            |t| {
                Ok(tokenizer
                    .encode(t, false)
                    .map_err(|e| anyhow!("upsampler tokenize: {e}"))?
                    .get_ids()
                    .to_vec())
            },
            |ids| runner.embed_tokens(ids),
            self.runner.cfg().hidden_size,
            0,
        )?;
        let max = self
            .max_new_tokens
            .min(self.runner.max_seq().saturating_sub(prefill.seq.len() + 1));
        let stop = self.stop.clone();
        let mut out = self.runner.generate_from_assembled_with_opts(
            prefill,
            max,
            rlx_qwen35::SampleOpts::greedy(),
            |t| !stop.contains(&t),
        )?;
        if out.last().is_some_and(|t| stop.contains(t)) {
            out.pop();
        }
        let caption = self
            .tokenizer
            .decode(&out, true)
            .map_err(|e| anyhow!("upsampler decode: {e}"))?
            .trim()
            .to_string();
        if caption.is_empty() {
            bail!("the upsampler returned an empty caption");
        }
        Ok(caption)
    }

    /// Vision-tower embeddings of prepared references, with the
    /// checkpoint's own pixel bounds (`preprocessor_config.json`).
    fn encode_images(&mut self, refs: &[RgbImage]) -> Result<Vec<VisionEncodeOutput>> {
        if self.vision.is_none() {
            let mm = self.paths.mmproj().with_context(|| {
                format!(
                    "no mmproj-*.gguf (vision tower) in {} — `just fetch-flux3-pup` fetches it",
                    self.paths.dir.display()
                )
            })?;
            let (mut cfg, weights) = rlx_qwen35::MmProjWeights::from_gguf_path(
                mm.to_str().context("non-utf8 mmproj path")?,
            )?;
            if let Some((lo, hi)) = pixel_bounds(&self.paths.dir) {
                cfg.image_min_pixels = lo;
                cfg.image_max_pixels = hi;
            }
            let (w, h) = (refs[0].width() as usize, refs[0].height() as usize);
            self.vision = Some(Qwen35VisionEncoder::from_parts(
                cfg,
                weights,
                w,
                h,
                self.device,
            )?);
        }
        let enc = self.vision.as_mut().expect("loaded above");
        refs.iter()
            .map(|r| enc.encode_rgb(r.as_raw(), r.width() as usize, r.height() as usize))
            .collect()
    }

    /// Free the model (its weights are ~6-19 GB depending on the format).
    pub fn into_runner(self) -> Qwen35Runner {
        self.runner
    }
}

/// At most this many references in one request.
pub const MAX_REFERENCES: usize = 10;

/// The user turn of an edit after the images (BFL `I2I_SCAFFOLD[False]`).
pub const I2I_SCAFFOLD: &str = "Reference images follow in order (the first image, the second image, ...).\n\
User request: {instruction}\n\
Output the caption.";

/// The same with locked boxes (BFL `I2I_SCAFFOLD[True]`).
pub const I2I_SCAFFOLD_LOCKED: &str = "Reference images follow in attachment order: ref_image_0, ref_image_1, ...\n\
User editing instruction: {instruction}\n\
Decide the edit, imagine the target image, and output the annotation record.";

const REFERENCE_MULTIPLE: usize = 32;
const MAX_ASPECT_RATIO: f64 = 200.0;
const JPEG_QUALITY: u8 = 92;

/// Pixel budget of each reference for `n` references.
fn max_pixels(n: usize) -> usize {
    match n {
        1 => 1024 * 1024,
        2 => 724 * 724,
        _ => 512 * 512,
    }
}

/// BFL `reference_size`: sides on the 32-px grid within the per-count budget.
pub fn reference_size(width: usize, height: usize, n_refs: usize) -> Result<(usize, usize)> {
    let (wf, hf) = (width as f64, height as f64);
    if wf.max(hf) / wf.min(hf) > MAX_ASPECT_RATIO {
        bail!("aspect ratio of {width}x{height} exceeds {MAX_ASPECT_RATIO}:1");
    }
    let m = REFERENCE_MULTIPLE as f64;
    let floor = |v: f64| REFERENCE_MULTIPLE.max(((v / m).floor() as usize) * REFERENCE_MULTIPLE);
    let cap = max_pixels(n_refs);
    let mut w = REFERENCE_MULTIPLE.max(round_half_even(wf / m) as usize * REFERENCE_MULTIPLE);
    let mut h = REFERENCE_MULTIPLE.max(round_half_even(hf / m) as usize * REFERENCE_MULTIPLE);
    if w * h > cap {
        let beta = (wf * hf / cap as f64).sqrt();
        w = floor(wf / beta);
        h = floor(hf / beta);
    }
    if w * h > cap {
        if w == REFERENCE_MULTIPLE {
            h = floor(cap as f64 / w as f64);
        } else {
            w = floor(cap as f64 / h as f64);
        }
    }
    Ok((w, h))
}

/// BFL `policy_reference`: Lanczos to [`reference_size`], then a JPEG q92
/// round trip, as the upsampler saw its training references.
pub fn policy_reference(image: &RgbImage, n_refs: usize) -> Result<RgbImage> {
    let (w, h) = reference_size(image.width() as usize, image.height() as usize, n_refs)?;
    let img = if image.dimensions() != (w as u32, h as u32) {
        image::imageops::resize(
            image,
            w as u32,
            h as u32,
            image::imageops::FilterType::Lanczos3,
        )
    } else {
        image.clone()
    };
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, JPEG_QUALITY)
        .encode_image(&img)
        .context("JPEG encode")?;
    Ok(image::load_from_memory_with_format(&jpeg, image::ImageFormat::Jpeg)?.to_rgb8())
}

/// `(min, max)` pixels of the image processor (`size.shortest_edge` /
/// `size.longest_edge` in `preprocessor_config.json`, which Qwen's
/// processors read as pixel counts).
fn pixel_bounds(dir: &Path) -> Option<(usize, usize)> {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("preprocessor_config.json")).ok()?)
            .ok()?;
    Some((
        v.pointer("/size/shortest_edge")?.as_u64()? as usize,
        v.pointer("/size/longest_edge")?.as_u64()? as usize,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_sizes_follow_the_budget() {
        // One reference: ≤ 1 MP on the 32-px grid.
        assert_eq!(reference_size(1024, 1024, 1).unwrap(), (1024, 1024));
        assert_eq!(reference_size(1000, 700, 1).unwrap(), (992, 704));
        let (w, h) = reference_size(4000, 3000, 1).unwrap();
        assert!(w * h <= 1024 * 1024 && w % 32 == 0 && h % 32 == 0);
        let (w, h) = reference_size(1024, 1024, 3).unwrap();
        assert_eq!((w, h), (512, 512));
        assert!(reference_size(10_000, 10, 1).is_err());
        // Python rounds half to even: 80 / 32 = 2.5 → 2.
        assert_eq!(reference_size(80, 64, 1).unwrap(), (64, 64));
    }
}
