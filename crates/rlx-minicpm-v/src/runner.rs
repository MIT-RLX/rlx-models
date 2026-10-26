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

//! [`MiniCpmVRunner`]: vision tower + canvas positions in front of the
//! `rlx-qwen35` trunk, which loads the same HF directory for the LM half.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, ensure};
use rlx_core::SafetensorsCheckpoint;
use rlx_core::weight_map::WeightMap;
use rlx_ir::DType;
use rlx_qwen35::{MultimodalPrefill, Qwen35Runner, SampleOpts};
use rlx_runtime::{CompiledGraph, Device};

use crate::config::{MiniCpmVConfig, MropeMode};
use crate::gguf::GgufPair;
use crate::positions::{Pos, canvas_positions, linear_positions};
use crate::preprocess::{Crop, SlicedImage, slice_image};
use crate::prompt::{Media, chat_prompt_media, expand_media, image_placeholder, video_placeholder};
use crate::vision::{MERGER, PATCH_LINEAR, TOWER, build_vision, crop_feeds, pick_bucket};

/// An RGB8 image, row-major interleaved.
#[derive(Debug, Clone)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

impl RgbImage {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        let img = image::open(p)
            .with_context(|| format!("minicpm-v: open image {p:?}"))?
            .to_rgb8();
        Ok(Self {
            width: img.width() as usize,
            height: img.height() as usize,
            rgb: img.into_raw(),
        })
    }
}

#[derive(Default)]
pub struct MiniCpmVRunnerBuilder {
    model_dir: Option<PathBuf>,
    gguf: Option<GgufPair>,
    prefer: Option<String>,
    packed: Option<bool>,
    tokenizer: Option<PathBuf>,
    device: Option<Device>,
    max_seq: Option<usize>,
    bf16_vision: bool,
}

impl MiniCpmVRunnerBuilder {
    /// HF checkpoint directory (`config.json`, `model.safetensors`,
    /// `tokenizer.json`), or a directory of the official GGUFs (an LM GGUF
    /// plus `mmproj-*.gguf`; pick the quantization with [`Self::prefer`]).
    pub fn model_dir(mut self, p: impl Into<PathBuf>) -> Self {
        self.model_dir = Some(p.into());
        self
    }
    /// An explicit LM GGUF + vision mmproj pair.
    pub fn gguf(mut self, lm: impl Into<PathBuf>, mmproj: impl Into<PathBuf>) -> Self {
        self.gguf = Some(GgufPair {
            lm: lm.into(),
            mmproj: mmproj.into(),
        });
        self
    }
    /// LM quantization to pick from a GGUF directory (`"Q4_K_M"`, `"F16"`).
    pub fn prefer(mut self, q: impl Into<String>) -> Self {
        self.prefer = Some(q.into());
        self
    }
    /// Keep a GGUF trunk's quantized blocks packed (default: `rlx-qwen35`'s
    /// size-based choice).
    pub fn packed(mut self, on: bool) -> Self {
        self.packed = Some(on);
        self
    }
    /// `tokenizer.json` to use instead of the GGUF's embedded vocabulary.
    pub fn tokenizer(mut self, p: impl Into<PathBuf>) -> Self {
        self.tokenizer = Some(p.into());
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    /// Longest prompt + answer, in tokens (default 4096).
    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = Some(n);
        self
    }
    /// Keep the vision tower's matrices in bf16 instead of widening them to
    /// f32. Halves its ~2 GB; the math stays f32.
    pub fn bf16_vision(mut self, on: bool) -> Self {
        self.bf16_vision = on;
        self
    }

    pub fn build(self) -> Result<MiniCpmVRunner> {
        let device = self.device.unwrap_or(Device::Cpu);
        let gguf = match (self.gguf, self.model_dir.as_deref()) {
            (Some(pair), _) => Some(pair),
            (None, Some(d)) if GgufPair::is_gguf_dir(d) => {
                Some(GgufPair::from_dir(d, self.prefer.as_deref())?)
            }
            _ => None,
        };
        let (cfg, source, lm_weights, tok_weights, tok_explicit, eos) = match gguf {
            Some(pair) => {
                let cfg = crate::gguf::config_from_gguf(&pair)?;
                let eos = crate::gguf::eos_from_gguf(&pair.lm)?;
                eprintln!(
                    "[minicpm-v] GGUF: lm={:?} mmproj={:?}",
                    pair.lm.file_name().unwrap_or_default(),
                    pair.mmproj.file_name().unwrap_or_default()
                );
                (
                    cfg,
                    VisionSource::Gguf(pair.mmproj),
                    pair.lm.clone(),
                    pair.lm,
                    self.tokenizer,
                    eos,
                )
            }
            None => {
                let dir = self
                    .model_dir
                    .ok_or_else(|| anyhow!("MiniCpmVRunner: model_dir or gguf(...) is required"))?;
                let cfg = MiniCpmVConfig::from_dir(&dir)?;
                let tokenizer = self.tokenizer.unwrap_or_else(|| dir.join("tokenizer.json"));
                ensure!(tokenizer.is_file(), "minicpm-v: {tokenizer:?} not found");
                let eos = read_eos(&dir);
                (
                    cfg,
                    VisionSource::Hf(dir.clone()),
                    dir.clone(),
                    dir,
                    Some(tokenizer),
                    eos,
                )
            }
        };
        let mut b = Qwen35Runner::builder()
            .weights(&lm_weights)
            .device(device)
            .max_seq(self.max_seq.unwrap_or(4096))
            .runtime_mrope(cfg.mrope == MropeMode::Canvas)
            // A GGUF does not say so, but the trunk's M-RoPE is interleaved
            // (HF `mrope_interleaved`, llama.cpp IMROPE for `qwen35`).
            .mrope_interleaved(true)
            .force_host_embed(true)
            // Prefill takes embeddings, not ids: image rows are spliced in.
            .hidden_prefill(true)
            .skip_auto_mmproj(true)
            .skip_warm(true);
        if let Some(on) = self.packed {
            b = b.packed_weights(on);
        }
        let lm = b
            .build()
            .with_context(|| format!("minicpm-v: build Qwen3.5 trunk from {lm_weights:?}"))?;
        Ok(MiniCpmVRunner {
            cfg,
            source,
            device,
            lm,
            tok_weights,
            tok_explicit,
            eos,
            vision: HashMap::new(),
            vision_dtype: if self.bf16_vision {
                DType::BF16
            } else {
                DType::F32
            },
        })
    }
}

/// Where the vision tower's weights come from.
#[derive(Debug, Clone)]
enum VisionSource {
    /// HF checkpoint directory.
    Hf(PathBuf),
    /// Vision mmproj GGUF.
    Gguf(PathBuf),
}

/// `generation_config.json` `eos_token_id` (int or list); `<|im_end|>` when
/// the file is missing.
pub(crate) fn read_eos(dir: &Path) -> Vec<u32> {
    let v: Option<serde_json::Value> = std::fs::read_to_string(dir.join("generation_config.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());
    match v.as_ref().and_then(|v| v.get("eos_token_id")) {
        Some(serde_json::Value::Number(n)) => {
            n.as_u64().map(|x| vec![x as u32]).unwrap_or_default()
        }
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_u64())
            .map(|x| x as u32)
            .collect(),
        _ => vec![248046],
    }
}

/// One picture or clip of a prompt, already sliced.
#[derive(Debug, Clone)]
pub enum Visual {
    Image(SlicedImage),
    /// The sampled frames, each sliced like a picture.
    Video(Vec<SlicedImage>),
}

impl Visual {
    fn media(&self) -> Media {
        match self {
            Visual::Image(_) => Media::Image,
            Visual::Video(_) => Media::Video,
        }
    }

    fn crops(&self) -> Box<dyn Iterator<Item = &Crop> + '_> {
        match self {
            Visual::Image(im) => Box::new(im.crops.iter()),
            Visual::Video(frames) => Box::new(frames.iter().flat_map(|f| f.crops.iter())),
        }
    }
}

/// Settings for a clip (`MiniCPMV4_6VideoProcessor`; the model card
/// recommends one slice per frame for video).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoOpts {
    pub max_num_frames: usize,
    pub max_slice_nums: usize,
}

impl Default for VideoOpts {
    fn default() -> Self {
        Self {
            max_num_frames: 128,
            max_slice_nums: 1,
        }
    }
}

/// An assembled prompt: what the trunk prefills from.
#[derive(Debug, Clone)]
pub struct AssembledPrompt {
    pub ids: Vec<u32>,
    pub positions: Vec<Pos>,
    /// `[ids.len(), hidden]` input embeddings, image rows spliced in.
    pub embeds: Vec<f32>,
    /// Every crop's patch grid in prompt order.
    pub grids: Vec<(usize, usize)>,
}

pub struct MiniCpmVRunner {
    cfg: MiniCpmVConfig,
    source: VisionSource,
    device: Device,
    lm: Qwen35Runner,
    /// Tokenizer lookup: `tokenizer.json` when given, else the LM GGUF's
    /// embedded vocabulary.
    tok_weights: PathBuf,
    tok_explicit: Option<PathBuf>,
    eos: Vec<u32>,
    /// Compiled vision graphs by padded patch count.
    vision: HashMap<usize, CompiledGraph>,
    vision_dtype: DType,
}

impl MiniCpmVRunner {
    pub fn builder() -> MiniCpmVRunnerBuilder {
        MiniCpmVRunnerBuilder::default()
    }

    pub fn config(&self) -> &MiniCpmVConfig {
        &self.cfg
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn lm(&mut self) -> &mut Qwen35Runner {
        &mut self.lm
    }

    pub fn eos_ids(&self) -> &[u32] {
        &self.eos
    }

    /// Slice an image the way the HF processor does.
    pub fn slice(&self, img: &RgbImage) -> Result<SlicedImage> {
        slice_image(
            &img.rgb,
            img.width,
            img.height,
            &self.cfg.slice,
            self.cfg.vision.patch_size,
        )
    }

    fn vision_graph(&mut self, bucket: usize) -> Result<&mut CompiledGraph> {
        if !self.vision.contains_key(&bucket) {
            let t = web_time::Instant::now();
            let mut wm = match &self.source {
                VisionSource::Hf(dir) => load_vision_weights(dir)?,
                VisionSource::Gguf(mmproj) => crate::gguf::load_mmproj_weights(mmproj, &self.cfg)?,
            };
            let built = build_vision(&self.cfg, &mut wm, bucket, self.vision_dtype)?;
            let g = rlx_core::flow_util::compile_built(built, self.device)?;
            eprintln!(
                "[minicpm-v] vision graph for {bucket} patches compiled in {:.1}s",
                t.elapsed().as_secs_f64()
            );
            // A larger graph serves every smaller crop; drop the smaller ones
            // rather than hold a weight copy and an arena per bucket.
            self.vision.retain(|&b, _| b > bucket);
            self.vision.insert(bucket, g);
        }
        Ok(self.vision.get_mut(&bucket).expect("inserted above"))
    }

    /// Free the compiled vision graphs (weights and arena); the next crop
    /// compiles again. For a tight device: the trunk's prefill then has the
    /// memory to itself.
    pub fn release_vision_graphs(&mut self) {
        self.vision.clear();
    }

    /// `[n_tokens, text_hidden]` LM-space features of one crop.
    pub fn encode_crop(&mut self, crop: &Crop) -> Result<Vec<f32>> {
        let bucket = pick_bucket(self.vision.keys().copied(), crop.num_patches());
        let feeds = crop_feeds(&self.cfg, crop.grid_h, crop.grid_w, &crop.patches, bucket)?;
        let hidden = self.cfg.text_hidden_size;
        let out = self
            .vision_graph(bucket)?
            .run(&feeds.as_inputs())
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("minicpm-v: vision graph returned no output"))?;
        ensure!(
            out.len() >= feeds.n_tokens * hidden,
            "minicpm-v: vision output has {} values, need {}",
            out.len(),
            feeds.n_tokens * hidden
        );
        Ok(out[..feeds.n_tokens * hidden].to_vec())
    }

    /// Slice one video frame: the picture path with the clip's slice budget.
    pub fn slice_frame(&self, img: &RgbImage, max_slice_nums: usize) -> Result<SlicedImage> {
        let cfg = crate::config::SliceConfig {
            max_slice_nums,
            ..self.cfg.slice.clone()
        };
        slice_image(
            &img.rgb,
            img.width,
            img.height,
            &cfg,
            self.cfg.vision.patch_size,
        )
    }

    /// Sample, decode (with `ffmpeg`) and slice a clip.
    pub fn load_video(&self, path: &Path, opts: VideoOpts) -> Result<Visual> {
        let frames = crate::video::load_video(path, opts.max_num_frames)?;
        Ok(Visual::Video(
            frames
                .iter()
                .map(|f| self.slice_frame(f, opts.max_slice_nums))
                .collect::<Result<_>>()?,
        ))
    }

    /// Tokens per crop of `grid` patches.
    fn tokens_for(&self, grid: (usize, usize)) -> usize {
        let f = self.cfg.downsample.merge_factor();
        grid.0 * grid.1 / (f * f)
    }

    /// `(thumb_tokens, slice_tokens, grid)` of one sliced picture or frame.
    fn layout(&self, im: &SlicedImage) -> (usize, usize, Option<(usize, usize)>) {
        let th = self.tokens_for((im.thumbnail().grid_h, im.thumbnail().grid_w));
        let sl = im
            .slices()
            .first()
            .map_or(0, |c| self.tokens_for((c.grid_h, c.grid_w)));
        (th, sl, im.grid)
    }

    /// Expand `prompt` (with one [`crate::IMAGE_PAD`] per image), tokenize,
    /// run every crop through the tower, splice the features in and lay out
    /// the positions.
    pub fn assemble(&mut self, prompt: &str, images: &[SlicedImage]) -> Result<AssembledPrompt> {
        let visuals: Vec<Visual> = images.iter().cloned().map(Visual::Image).collect();
        self.assemble_media(prompt, &visuals)
    }

    /// [`Self::assemble`] for pictures and clips: `prompt` holds one
    /// [`crate::IMAGE_PAD`] or [`crate::VIDEO_PAD`] per visual, in order.
    pub fn assemble_media(&mut self, prompt: &str, visuals: &[Visual]) -> Result<AssembledPrompt> {
        let use_id = self.cfg.slice.use_image_id;
        let mut image_idx = 0usize;
        let mut placeholders = Vec::with_capacity(visuals.len());
        for v in visuals {
            let text = match v {
                Visual::Image(im) => {
                    let (th, sl, grid) = self.layout(im);
                    let id = use_id.then_some(image_idx);
                    image_idx += 1;
                    image_placeholder(id, th, sl, grid)
                }
                Visual::Video(frames) => {
                    let layout: Vec<_> = frames.iter().map(|f| self.layout(f)).collect();
                    video_placeholder(&layout)
                }
            };
            placeholders.push((v.media(), text));
        }
        let text = expand_media(prompt, &placeholders)?;
        let ids =
            rlx_qwen35::encode_prompt_auto(&self.tok_weights, self.tok_explicit.as_deref(), &text)?;
        let image_tok = self.cfg.image_token_id;
        let video_tok = self.cfg.video_token_id;
        if visuals.iter().any(|v| matches!(v, Visual::Video(_))) {
            ensure!(
                video_tok.is_some(),
                "minicpm-v: this checkpoint names no video_token_id"
            );
        }
        let modality: Vec<u8> = ids
            .iter()
            .map(|&t| {
                if t == image_tok {
                    1
                } else if Some(t) == video_tok {
                    2
                } else {
                    0
                }
            })
            .collect();

        let crops: Vec<&Crop> = visuals.iter().flat_map(|v| v.crops()).collect();
        let grids: Vec<(usize, usize)> = crops.iter().map(|c| (c.grid_h, c.grid_w)).collect();
        let hidden = self.cfg.text_hidden_size;
        let mut embeds = self.lm.embed_tokens(&ids)?;
        let mut rows = modality
            .iter()
            .enumerate()
            .filter(|(_, m)| **m != 0)
            .map(|(i, _)| i);
        let mut placed = 0usize;
        for c in &crops {
            let feats = self.encode_crop(c)?;
            for r in feats.chunks_exact(hidden) {
                let i = rows
                    .next()
                    .ok_or_else(|| anyhow!("minicpm-v: more visual features than placeholders"))?;
                embeds[i * hidden..(i + 1) * hidden].copy_from_slice(r);
                placed += 1;
            }
        }
        ensure!(
            rows.next().is_none(),
            "minicpm-v: {placed} visual features for more placeholders"
        );

        let positions = match self.cfg.mrope {
            MropeMode::Linear => linear_positions(ids.len()),
            MropeMode::Canvas => {
                let markers = self.cfg.markers.ok_or_else(|| {
                    anyhow!("minicpm-v: canvas M-RoPE needs the marker ids in config.json")
                })?;
                canvas_positions(
                    &ids,
                    &modality,
                    &grids,
                    self.cfg.downsample.merge_factor(),
                    &markers,
                )?
            }
        };
        Ok(AssembledPrompt {
            ids,
            positions,
            embeds,
            grids,
        })
    }

    /// Generate up to `max_new` tokens for an assembled prompt; stops at EOS
    /// (not included) or when `on_token` returns `false`.
    pub fn generate_assembled(
        &mut self,
        prompt: AssembledPrompt,
        max_new: usize,
        opts: SampleOpts,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let last = prompt.ids.len().saturating_sub(1);
        let prefill = MultimodalPrefill {
            hidden: prompt.embeds,
            mrope_sections: prompt.positions,
            last_token_idx: last,
            seq: prompt.ids,
        };
        let eos: HashSet<u32> = self.eos.iter().copied().collect();
        let mut out = Vec::new();
        self.lm
            .generate_from_assembled_with_opts(prefill, max_new, opts, |t| {
                if eos.contains(&t) {
                    return false;
                }
                out.push(t);
                on_token(t)
            })?;
        Ok(out)
    }

    /// Answer `question` about `images` (one chat turn) and return the text.
    pub fn chat(
        &mut self,
        images: &[RgbImage],
        question: &str,
        max_new: usize,
        opts: SampleOpts,
        thinking: bool,
        on_text: impl FnMut(&str),
    ) -> Result<String> {
        let visuals = images
            .iter()
            .map(|im| self.slice(im).map(Visual::Image))
            .collect::<Result<Vec<_>>>()?;
        self.chat_media(&visuals, question, max_new, opts, thinking, on_text)
    }

    /// [`Self::chat`] with pictures and clips (see [`Self::load_video`]),
    /// placed before `question` in `visuals` order.
    pub fn chat_media(
        &mut self,
        visuals: &[Visual],
        question: &str,
        max_new: usize,
        opts: SampleOpts,
        thinking: bool,
        mut on_text: impl FnMut(&str),
    ) -> Result<String> {
        let media: Vec<Media> = visuals.iter().map(Visual::media).collect();
        let prompt = chat_prompt_media(question, &media, thinking);
        let assembled = self.assemble_media(&prompt, visuals)?;
        let (tw, te) = (self.tok_weights.clone(), self.tok_explicit.clone());
        let mut ids: Vec<u32> = Vec::new();
        let mut shown = 0usize;
        self.generate_assembled(assembled, max_new, opts, |t| {
            ids.push(t);
            if let Ok(s) = rlx_qwen35::decode_ids_auto(&tw, te.as_deref(), &ids, true) {
                // Hold back a trailing replacement char (a split UTF-8 rune).
                if !s.ends_with('\u{FFFD}') && s.len() > shown {
                    on_text(&s[shown..]);
                    shown = s.len();
                }
            }
            true
        })?;
        self.decode(&ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        rlx_qwen35::decode_ids_auto(&self.tok_weights, self.tok_explicit.as_deref(), ids, true)
    }
}

/// The tower and merger tensors of an HF checkpoint, with the patch conv
/// reshaped to the linear the graph reads.
pub fn load_vision_weights(dir: &Path) -> Result<WeightMap> {
    let ckpt = SafetensorsCheckpoint::open(dir)
        .with_context(|| format!("minicpm-v: open safetensors in {dir:?}"))?;
    let merger_root = MERGER.trim_end_matches(".mlp.0");
    let want: HashSet<String> = ckpt
        .keys()
        .filter(|k| {
            k.starts_with(&format!("{TOWER}.")) || k.starts_with(&format!("{merger_root}."))
        })
        .map(str::to_string)
        .collect();
    ensure!(
        !want.is_empty(),
        "minicpm-v: no {TOWER}.* tensors in {dir:?}"
    );
    let mut wm = ckpt.load_selected(&want)?;
    let conv = format!("{TOWER}.embeddings.patch_embedding.weight");
    let (w, shape) = wm.take(&conv)?;
    ensure!(
        shape.len() == 4,
        "minicpm-v: {conv} is {shape:?}, expected a 4-D conv"
    );
    let cols = shape[1] * shape[2] * shape[3];
    wm.insert(PATCH_LINEAR, w, vec![shape[0], cols]);
    Ok(wm)
}
