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

//! The resampler generation (`model_type = minicpmv`, OpenBMB's remote
//! code): the same SigLIP-400M tower as 4.6 / 4.7, no window merger, then a
//! perceiver resampler that turns every crop into `query_num` (64) LM rows.
//!
//! The language model is the checkpoint's `llm` — Llama-3 (2.5), Llama with
//! LongRoPE (4.0) or Qwen3 (4.5) — run by `rlx-minicpm-sala`'s step graph:
//! one compiled graph for prompt chunks and decode, the KV cache threaded by
//! the host.
//!
//! The prompt is `chat()`'s: `(<image>./</image>)` per picture, then the
//! question, in the checkpoint's chat template; each tag becomes `<image>` +
//! `query_num` × `<unk>` + `</image>` plus the slices (laid out per
//! [`Generation`]), and the features replace the `<unk>` rows.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use rlx_core::SafetensorsCheckpoint;
use rlx_core::weight_map::WeightMap;
use rlx_ir::DType;
use rlx_minicpm_sala::{SalaConfig, SalaRunner};
use rlx_qwen35::SampleOpts;
use rlx_runtime::{CompiledGraph, Device};
use serde_json::Value;

use crate::config::{
    DownsampleMode, MiniCpmVConfig, MropeMode, Projector, SliceConfig, VisionConfig,
};
use crate::preprocess::{Crop, SlicedImage, resampler_gen::slice_image};
use crate::runner::RgbImage;
use crate::vision::{
    GroupFrame, PATCH_LINEAR, RESAMPLER, TOWER, VisionPart, build_vision, build_vision_part,
    crop_feeds_padded, group_resampler_feeds, pick_bucket, tower_feeds,
};

/// Tag `chat()` puts where a picture goes.
pub const IMAGE_TAG: &str = "(<image>./</image>)";

/// `true` when `dir/config.json` names a resampler-generation checkpoint.
pub fn is_model_dir(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .is_some_and(|v| v.get("model_type").and_then(Value::as_str) == Some("minicpmv"))
}

/// Which release: the trunk, chat template and slice layout follow it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Generation {
    /// MiniCPM-Llama3-V 2.5: Llama-3 template; slices as one `<slice>` block
    /// of `<image>…</image>` crops; no image ids.
    V2_5,
    /// MiniCPM-V 2.6: as 4.0, Qwen2 trunk (q/k/v biases); the template
    /// opens with a default system turn.
    V2_6,
    /// MiniCPM-V 4.0: ChatML; `<image_id>i</image_id>`, each slice its own
    /// `<slice>…</slice>`.
    V4_0,
    /// MiniCPM-V 4.5: as 4.0, Qwen3 trunk and template (`<think>` block).
    V4_5,
}

/// Vision side (tower + resampler) and the trunk's config.
#[derive(Debug, Clone)]
pub struct ResamplerConfig {
    pub vision: MiniCpmVConfig,
    pub lm: SalaConfig,
    pub generation: Generation,
}

impl ResamplerConfig {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let read = |name: &str| -> Result<Value> {
            let p = dir.join(name);
            serde_json::from_str(
                &std::fs::read_to_string(&p).with_context(|| format!("minicpm-v: read {p:?}"))?,
            )
            .with_context(|| format!("minicpm-v: parse {p:?}"))
        };
        let v = read("config.json")?;
        let pp = read("preprocessor_config.json").ok();
        Self::from_values(&v, pp.as_ref())
    }

    pub fn from_values(v: &Value, preprocessor: Option<&Value>) -> Result<Self> {
        let mt = v.get("model_type").and_then(Value::as_str);
        ensure!(
            mt == Some("minicpmv"),
            "minicpm-v: model_type {mt:?} is not the resampler generation"
        );
        // `version` is a number (4.0, 4.5) or a string ("2.5").
        let version = v
            .get("version")
            .and_then(|x| x.as_f64().or_else(|| x.as_str()?.parse().ok()))
            .unwrap_or(0.0);
        let generation = if version == 2.5 {
            Generation::V2_5
        } else if version == 2.6 {
            Generation::V2_6
        } else if version == 4.0 {
            Generation::V4_0
        } else if version == 4.5 {
            Generation::V4_5
        } else {
            bail!(
                "minicpm-v: resampler checkpoint version {version}: 2.5, 2.6, 4.0 and 4.5 are implemented"
            )
        };
        let us = |o: &Value, k: &str| o.get(k).and_then(Value::as_u64).map(|n| n as usize);
        let vc = v
            .get("vision_config")
            .ok_or_else(|| anyhow!("minicpm-v: config.json has no vision_config"))?;
        let need =
            |k: &str| us(vc, k).ok_or_else(|| anyhow!("minicpm-v: vision_config.{k} missing"));
        let patch_size = need("patch_size")?;
        let mut layers = need("num_hidden_layers")?;
        if v.get("drop_vision_last_layer").and_then(Value::as_bool) == Some(true) {
            layers -= 1;
        }
        let vision = VisionConfig {
            hidden_size: need("hidden_size")?,
            intermediate_size: need("intermediate_size")?,
            num_attention_heads: need("num_attention_heads")?,
            num_hidden_layers: layers,
            patch_size,
            num_patches_per_side: need("image_size")? / patch_size,
            layer_norm_eps: vc
                .get("layer_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
            insert_layer_id: 0,
            window: 2,
            flat_positions: generation == Generation::V2_5,
        };
        let mut slice = SliceConfig {
            use_image_id: generation != Generation::V2_5
                && v.get("use_image_id")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            slice_mode: v.get("slice_mode").and_then(Value::as_bool).unwrap_or(true),
            ..SliceConfig::default()
        };
        if let Some(sc) = v.get("slice_config") {
            if let Some(n) = us(sc, "max_slice_nums") {
                slice.max_slice_nums = n;
            }
            if let Some(n) = us(sc, "scale_resolution") {
                slice.scale_resolution = n;
            }
        }
        if let Some(pp) = preprocessor {
            let triple = |k: &str| -> Option<[f32; 3]> {
                let a = pp.get(k)?.as_array()?;
                (a.len() == 3).then(|| [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.0) as f32))
            };
            if let Some(m) = triple("norm_mean") {
                slice.image_mean = m;
            }
            if let Some(s) = triple("norm_std") {
                slice.image_std = s;
            }
        }
        let queries = us(v, "query_num").unwrap_or(64);
        let hidden =
            us(v, "hidden_size").ok_or_else(|| anyhow!("minicpm-v: config.hidden_size missing"))?;
        let mut lm = SalaConfig::from_llama_value(v, "llm.")?;
        lm.attn_qk_norm = generation == Generation::V4_5;
        // Qwen2 always carries q/k/v biases (its config does not say so).
        lm.attn_qkv_bias |= generation == Generation::V2_6;
        Ok(Self {
            generation,
            vision: MiniCpmVConfig {
                model_type: "minicpmv".into(),
                vision,
                slice,
                downsample: DownsampleMode::X4,
                mrope: MropeMode::Linear,
                merger_times: 1,
                merge_kernel: 2,
                text_hidden_size: hidden,
                markers: None,
                // Filled from the tokenizer by the runner.
                image_token_id: 0,
                video_token_id: None,
                projector: Projector::Resampler { queries },
            },
            lm,
        })
    }

    pub fn queries(&self) -> usize {
        match self.vision.projector {
            Projector::Resampler { queries } => queries,
            Projector::Merger => unreachable!("built with a resampler"),
        }
    }
}

/// The tower and resampler tensors renamed to what [`build_vision`] reads:
/// `vpm.*` under [`TOWER`], the patch conv as [`PATCH_LINEAR`], and the
/// resampler's packed `in_proj` split into `q_proj` / `k_proj` / `v_proj`.
pub fn load_vision_weights(dir: &Path) -> Result<WeightMap> {
    let ckpt = SafetensorsCheckpoint::open(dir)
        .with_context(|| format!("minicpm-v: open safetensors in {dir:?}"))?;
    let want: std::collections::HashSet<String> = ckpt
        .keys()
        .filter(|k| k.starts_with("vpm.") || k.starts_with(&format!("{RESAMPLER}.")))
        .map(str::to_string)
        .collect();
    ensure!(!want.is_empty(), "minicpm-v: no vpm.* tensors in {dir:?}");
    let mut wm = ckpt.load_selected(&want)?;
    for (k, (data, shape)) in wm.take_all() {
        let name = match k.strip_prefix("vpm.") {
            Some(rest) => format!("{TOWER}.{rest}"),
            None => k,
        };
        wm.insert(name, data, shape);
    }
    let conv = format!("{TOWER}.embeddings.patch_embedding.weight");
    let (w, shape) = wm.take(&conv)?;
    ensure!(shape.len() == 4, "minicpm-v: {conv} is {shape:?}");
    wm.insert(
        PATCH_LINEAR,
        w,
        vec![shape[0], shape[1] * shape[2] * shape[3]],
    );

    let a = format!("{RESAMPLER}.attn");
    let (w, shape) = wm.take(&format!("{a}.in_proj_weight"))?;
    let (b, _) = wm.take(&format!("{a}.in_proj_bias"))?;
    ensure!(
        shape.len() == 2 && shape[0] == 3 * shape[1],
        "minicpm-v: in_proj_weight is {shape:?}"
    );
    let d = shape[1];
    for (i, name) in ["q_proj", "k_proj", "v_proj"].iter().enumerate() {
        wm.insert(
            format!("{a}.{name}.weight"),
            w[i * d * d..(i + 1) * d * d].to_vec(),
            vec![d, d],
        );
        wm.insert(
            format!("{a}.{name}.bias"),
            b[i * d..(i + 1) * d].to_vec(),
            vec![d],
        );
    }
    Ok(wm)
}

/// Token ids the prompt layout uses.
#[derive(Debug, Clone, Copy)]
struct Markers {
    unk: u32,
    image_start: u32,
    image_end: u32,
    slice_start: u32,
    slice_end: u32,
}

#[derive(Default)]
pub struct ResamplerRunnerBuilder {
    dir: Option<PathBuf>,
    device: Option<Device>,
    max_seq: Option<usize>,
    chunk: Option<usize>,
    bf16_vision: bool,
}

impl ResamplerRunnerBuilder {
    pub fn model_dir(mut self, p: impl Into<PathBuf>) -> Self {
        self.dir = Some(p.into());
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    /// Longest prompt + answer (default 4096).
    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = Some(n);
        self
    }
    /// Prompt tokens per trunk run (default 16).
    pub fn chunk(mut self, n: usize) -> Self {
        self.chunk = Some(n);
        self
    }
    /// Keep the tower's and resampler's matrices in bf16.
    pub fn bf16_vision(mut self, on: bool) -> Self {
        self.bf16_vision = on;
        self
    }

    pub fn build(self) -> Result<ResamplerRunner> {
        let dir = self
            .dir
            .ok_or_else(|| anyhow!("minicpm-v: model_dir is required"))?;
        let device = self.device.unwrap_or(Device::Cpu);
        let mut cfg = ResamplerConfig::from_dir(&dir)?;
        let tok_path = dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow!("minicpm-v: load {tok_path:?}: {e}"))?;
        let id = |t: &str| {
            tokenizer
                .token_to_id(t)
                .ok_or_else(|| anyhow!("minicpm-v: token {t:?} not in {tok_path:?}"))
        };
        let markers = Markers {
            unk: id("<unk>")?,
            image_start: id("<image>")?,
            image_end: id("</image>")?,
            slice_start: id("<slice>")?,
            slice_end: id("</slice>")?,
        };
        cfg.vision.image_token_id = markers.unk;
        let mut eos = cfg.lm.eos_token_ids.clone();
        eos.extend(tokenizer.token_to_id("<|im_end|>"));
        eos.extend(tokenizer.token_to_id("<|eot_id|>"));
        let lm = SalaRunner::builder()
            .model_dir(&dir)
            .config(cfg.lm.clone())
            .device(device)
            .max_seq(self.max_seq.unwrap_or(4096))
            .chunk(self.chunk.unwrap_or(16))
            .build()?;
        Ok(ResamplerRunner {
            cfg,
            dir,
            device,
            lm,
            tokenizer,
            markers,
            eos,
            vision: HashMap::new(),
            towers: HashMap::new(),
            group_resamplers: HashMap::new(),
            vision_dtype: if self.bf16_vision {
                DType::BF16
            } else {
                DType::F32
            },
        })
    }
}

pub struct ResamplerRunner {
    cfg: ResamplerConfig,
    dir: PathBuf,
    device: Device,
    lm: SalaRunner,
    tokenizer: tokenizers::Tokenizer,
    markers: Markers,
    eos: Vec<u32>,
    vision: HashMap<usize, CompiledGraph>,
    /// 4.5 video: tower-only graphs by bucket, resampler-only graphs by rows.
    towers: HashMap<usize, CompiledGraph>,
    group_resamplers: HashMap<usize, CompiledGraph>,
    vision_dtype: DType,
}

/// A clip for 4.5's packed video: each frame a thumbnail crop with its
/// temporal id, consecutive frames grouped; a group becomes one set of
/// `query_num` LM rows.
#[derive(Debug, Clone)]
pub struct PackedVideo {
    pub groups: Vec<Vec<(Crop, usize)>>,
}

impl ResamplerRunner {
    pub fn builder() -> ResamplerRunnerBuilder {
        ResamplerRunnerBuilder::default()
    }

    pub fn config(&self) -> &ResamplerConfig {
        &self.cfg
    }

    pub fn lm(&mut self) -> &mut SalaRunner {
        &mut self.lm
    }

    /// Slice an image the way the checkpoint's processor does.
    pub fn slice(&self, img: &RgbImage) -> Result<SlicedImage> {
        slice_image(
            &img.rgb,
            img.width,
            img.height,
            &self.cfg.vision.slice,
            self.cfg.vision.vision.patch_size,
        )
    }

    /// Free the compiled vision graphs (weights and arena); the next crop
    /// compiles again. For a tight device: the trunk's prefill then has the
    /// memory to itself.
    pub fn release_vision_graphs(&mut self) {
        self.vision.clear();
        self.towers.clear();
        self.group_resamplers.clear();
    }

    /// Sample (the model card's `encode_video`), decode with `ffmpeg` and
    /// cut a clip into packed groups. 4.5 only: its resampler carries the
    /// temporal positions. Frames are thumbnails (`max_slice_nums = 1`, as
    /// the card runs video).
    pub fn load_packed_video(
        &self,
        path: &Path,
        o: &crate::video::PackOpts,
    ) -> Result<PackedVideo> {
        ensure!(
            self.cfg.generation == Generation::V4_5,
            "minicpm-v: packed video needs MiniCPM-V 4.5"
        );
        let meta = crate::video::probe(path)?;
        let (idx, groups) = crate::video::sample_packed(&meta, o);
        let frames = crate::video::decode_frames(path, &meta, &idx)?;
        self.pack_frames(&frames, &groups)
    }

    /// Group decoded `frames` by `temporal_groups` (one id per frame).
    pub fn pack_frames(
        &self,
        frames: &[RgbImage],
        temporal_groups: &[Vec<usize>],
    ) -> Result<PackedVideo> {
        let n: usize = temporal_groups.iter().map(Vec::len).sum();
        ensure!(
            n == frames.len(),
            "minicpm-v: {n} temporal ids for {} frames",
            frames.len()
        );
        let cfg = SliceConfig {
            max_slice_nums: 1,
            ..self.cfg.vision.slice.clone()
        };
        let mut it = frames.iter();
        let mut groups = Vec::with_capacity(temporal_groups.len());
        for ids in temporal_groups {
            let mut g = Vec::with_capacity(ids.len());
            for &t in ids {
                let f = it.next().expect("counted above");
                let sliced = slice_image(
                    &f.rgb,
                    f.width,
                    f.height,
                    &cfg,
                    self.cfg.vision.vision.patch_size,
                )?;
                g.push((sliced.crops.into_iter().next().expect("a thumbnail"), t));
            }
            groups.push(g);
        }
        Ok(PackedVideo { groups })
    }

    /// `[query_num, hidden]` LM rows of one packed group: the tower on each
    /// frame, then one resampler pass over all their patches.
    pub fn encode_group(&mut self, group: &[(Crop, usize)]) -> Result<Vec<f32>> {
        let c = self.cfg.vision.vision.hidden_size;
        let mut feats = Vec::with_capacity(group.len());
        for (crop, _) in group {
            let n = crop.num_patches();
            let bucket = pick_bucket(self.towers.keys().copied(), n);
            if !self.towers.contains_key(&bucket) {
                let g = self.compile_part(bucket, VisionPart::Tower)?;
                self.towers.retain(|&b, _| b > bucket);
                self.towers.insert(bucket, g);
            }
            let feeds = tower_feeds(&self.cfg.vision, crop, bucket)?;
            let inputs: Vec<(&str, &[f32])> =
                feeds.iter().map(|(k, v)| (*k, v.as_slice())).collect();
            let out = self
                .towers
                .get_mut(&bucket)
                .expect("inserted above")
                .run(&inputs);
            let out = out
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("minicpm-v: tower returned no output"))?;
            ensure!(out.len() >= n * c, "minicpm-v: tower output too short");
            feats.push(out[..n * c].to_vec());
        }
        let total: usize = group.iter().map(|(c, _)| c.num_patches()).sum();
        let rows = pick_bucket(self.group_resamplers.keys().copied(), total);
        if !self.group_resamplers.contains_key(&rows) {
            let g = self.compile_part(rows, VisionPart::Resampler)?;
            self.group_resamplers.retain(|&b, _| b > rows);
            self.group_resamplers.insert(rows, g);
        }
        let frames: Vec<GroupFrame<'_>> = group
            .iter()
            .zip(&feats)
            .map(|((crop, t), x)| GroupFrame {
                features: x,
                grid_h: crop.grid_h,
                grid_w: crop.grid_w,
                temporal_id: *t,
            })
            .collect();
        let feeds = group_resampler_feeds(&self.cfg.vision, &frames, rows)?;
        let inputs: Vec<(&str, &[f32])> = feeds.iter().map(|(k, v)| (*k, v.as_slice())).collect();
        let out = self
            .group_resamplers
            .get_mut(&rows)
            .expect("inserted above")
            .run(&inputs);
        let out = out
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("minicpm-v: resampler returned no output"))?;
        let n = self.cfg.queries() * self.cfg.vision.text_hidden_size;
        ensure!(out.len() >= n, "minicpm-v: resampler output too short");
        Ok(out[..n].to_vec())
    }

    fn compile_part(&self, s: usize, part: VisionPart) -> Result<CompiledGraph> {
        let t = web_time::Instant::now();
        let mut wm = load_vision_weights(&self.dir)?;
        let built = build_vision_part(&self.cfg.vision, &mut wm, s, self.vision_dtype, part)?;
        let g = rlx_core::flow_util::compile_built(built, self.device)?;
        eprintln!(
            "[minicpm-v] {part:?} graph for {s} rows compiled in {:.1}s",
            t.elapsed().as_secs_f64()
        );
        Ok(g)
    }

    /// `chat()`'s prompt for a packed clip (`use_image_id=False`): one
    /// `<image>…</image>` per group, groups joined by `\n`, then `question`.
    pub fn video_prompt(&self, question: &str, groups: usize, thinking: bool) -> String {
        let one = format!("<image>{}</image>", "<unk>".repeat(self.cfg.queries()));
        let mut s = format!(
            "<|im_start|>user\n{}\n{question}<|im_end|>\n<|im_start|>assistant\n",
            vec![one; groups].join("\n")
        );
        if self.cfg.generation == Generation::V4_5 {
            s.push_str(if thinking {
                "<think>\n"
            } else {
                "<think>\n\n</think>\n\n"
            });
        }
        s
    }

    /// Tokenize [`Self::video_prompt`] and splice every group's rows into its
    /// `<unk>` slots: `(ids, embeds)`.
    pub fn assemble_video(
        &mut self,
        question: &str,
        video: &PackedVideo,
        thinking: bool,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let text = self.video_prompt(question, video.groups.len(), thinking);
        let enc = self
            .tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| anyhow!("minicpm-v: tokenize: {e}"))?;
        let ids = enc.get_ids().to_vec();
        let slots: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == self.markers.unk)
            .map(|(i, _)| i)
            .collect();
        let h = self.cfg.vision.text_hidden_size;
        ensure!(
            slots.len() == video.groups.len() * self.cfg.queries(),
            "minicpm-v: {} slots for {} groups",
            slots.len(),
            video.groups.len()
        );
        let mut embeds = self.lm.embed(&ids)?;
        let mut slot = slots.into_iter();
        for g in &video.groups {
            for r in self.encode_group(g)?.chunks_exact(h) {
                let i = slot.next().expect("counted above");
                embeds[i * h..(i + 1) * h].copy_from_slice(r);
            }
        }
        Ok((ids, embeds))
    }

    /// `[query_num, hidden]` LM rows of one crop.
    pub fn encode_crop(&mut self, crop: &Crop) -> Result<Vec<f32>> {
        self.encode_crop_rows(crop, crop.num_patches())
    }

    /// The LM rows of every crop of a prompt. 2.5 runs its tower on the crops
    /// padded, unmasked, to the longest one (see
    /// [`crate::vision::crop_feeds_padded`]), so a crop's features depend on
    /// the others; later releases encode each crop on its own.
    pub fn encode_crops(&mut self, crops: &[&Crop]) -> Result<Vec<Vec<f32>>> {
        let longest = crops.iter().map(|c| c.num_patches()).max().unwrap_or(0);
        crops
            .iter()
            .map(|c| {
                let rows = if self.cfg.generation == Generation::V2_5 {
                    longest
                } else {
                    c.num_patches()
                };
                self.encode_crop_rows(c, rows)
            })
            .collect()
    }

    fn encode_crop_rows(&mut self, crop: &Crop, tower_rows: usize) -> Result<Vec<f32>> {
        let bucket = pick_bucket(self.vision.keys().copied(), tower_rows);
        let feeds = crop_feeds_padded(
            &self.cfg.vision,
            crop.grid_h,
            crop.grid_w,
            &crop.patches,
            bucket,
            tower_rows,
        )?;
        if !self.vision.contains_key(&bucket) {
            let t = web_time::Instant::now();
            let mut wm = load_vision_weights(&self.dir)?;
            let built = build_vision(&self.cfg.vision, &mut wm, bucket, self.vision_dtype)?;
            let g = rlx_core::flow_util::compile_built(built, self.device)?;
            eprintln!(
                "[minicpm-v] vision graph for {bucket} patches compiled in {:.1}s",
                t.elapsed().as_secs_f64()
            );
            self.vision.retain(|&b, _| b > bucket);
            self.vision.insert(bucket, g);
        }
        let n = feeds.n_tokens * self.cfg.vision.text_hidden_size;
        let out = self
            .vision
            .get_mut(&bucket)
            .expect("inserted above")
            .run(&feeds.as_inputs())
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("minicpm-v: vision graph returned no output"))?;
        ensure!(out.len() >= n, "minicpm-v: vision output too short");
        Ok(out[..n].to_vec())
    }

    /// `chat()`'s prompt: one [`IMAGE_TAG`] per picture, then `question`.
    /// With a Qwen3 template, `thinking` opens a `<think>` block; otherwise
    /// the empty one is closed (`enable_thinking=False`, `chat()`'s default).
    pub fn chat_prompt(&self, question: &str, n_images: usize, thinking: bool) -> String {
        let mut parts = vec![IMAGE_TAG; n_images];
        parts.push(question);
        let content = parts.join("\n");
        if self.cfg.generation == Generation::V2_5 {
            return format!(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n\n",
                content.trim()
            );
        }
        let mut s = String::new();
        if self.cfg.generation == Generation::V2_6 {
            s.push_str("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
        }
        s.push_str(&format!(
            "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
        ));
        if self.cfg.generation == Generation::V4_5 {
            s.push_str(if thinking {
                "<think>\n"
            } else {
                "<think>\n\n</think>\n\n"
            });
        }
        s
    }

    /// Placeholder text of one sliced picture.
    fn placeholder(&self, index: usize, im: &SlicedImage) -> String {
        let q = self.cfg.queries();
        let unk = "<unk>".repeat(q);
        let mut s = String::new();
        if self.cfg.vision.slice.use_image_id {
            s.push_str(&format!("<image_id>{index}</image_id>"));
        }
        s.push_str(&format!("<image>{unk}</image>"));
        if let Some((rows, cols)) = im.grid {
            if self.cfg.generation == Generation::V2_5 {
                let one = format!("<image>{unk}</image>");
                let grid = vec![one.repeat(cols); rows].join("\n");
                s.push_str(&format!("<slice>{grid}</slice>"));
            } else {
                let one = format!("<slice>{unk}</slice>");
                s.push_str(&vec![one.repeat(cols); rows].join("\n"));
            }
        }
        s
    }

    /// Expand `prompt`'s [`IMAGE_TAG`]s, tokenize and splice every crop's
    /// features into its `<unk>` rows: `(ids, embeds)`. The tokenizer adds
    /// BOS unless the template already wrote it (2.5's Llama-3 template).
    pub fn assemble(
        &mut self,
        prompt: &str,
        images: &[SlicedImage],
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let pieces: Vec<&str> = prompt.split(IMAGE_TAG).collect();
        ensure!(
            pieces.len() == images.len() + 1,
            "minicpm-v: prompt has {} image tags for {} images",
            pieces.len() - 1,
            images.len()
        );
        let mut text = pieces[0].to_string();
        for (i, (im, rest)) in images.iter().zip(&pieces[1..]).enumerate() {
            text.push_str(&self.placeholder(i, im));
            text.push_str(rest);
        }
        let enc = self
            .tokenizer
            .encode(text.as_str(), self.cfg.generation != Generation::V2_5)
            .map_err(|e| anyhow!("minicpm-v: tokenize: {e}"))?;
        let ids = enc.get_ids().to_vec();

        // `image_bound`: the rows between each `<image>`/`<slice>` and its
        // closing marker.
        let m = self.markers;
        let mut rows = Vec::new();
        let mut open = None;
        for (i, &t) in ids.iter().enumerate() {
            if t == m.image_start || t == m.slice_start {
                open = Some(i + 1);
            } else if (t == m.image_end || t == m.slice_end)
                && let Some(s) = open.take()
            {
                rows.extend(s..i);
            }
        }
        let h = self.cfg.vision.text_hidden_size;
        let mut embeds = self.lm.embed(&ids)?;
        let mut slot = rows.into_iter();
        let crops: Vec<&Crop> = images.iter().flat_map(|im| im.crops.iter()).collect();
        for feats in self.encode_crops(&crops)? {
            for r in feats.chunks_exact(h) {
                let i = slot
                    .next()
                    .ok_or_else(|| anyhow!("minicpm-v: more image features than slots"))?;
                embeds[i * h..(i + 1) * h].copy_from_slice(r);
            }
        }
        ensure!(
            slot.next().is_none(),
            "minicpm-v: image slots left without features"
        );
        Ok((ids, embeds))
    }

    /// Generate from assembled embeddings; stops at EOS (not returned) or
    /// when `on_token` returns `false`.
    pub fn generate_embeds(
        &mut self,
        embeds: &[f32],
        max_new: usize,
        opts: SampleOpts,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.lm.reset();
        let eos = self.eos.clone();
        let mut out = Vec::new();
        self.lm.generate_from_embeds(embeds, max_new, opts, |t| {
            if eos.contains(&t) {
                return false;
            }
            out.push(t);
            on_token(t)
        })?;
        Ok(out)
    }

    /// Answer `question` about `images` and return the text.
    pub fn chat(
        &mut self,
        images: &[RgbImage],
        question: &str,
        max_new: usize,
        opts: SampleOpts,
        thinking: bool,
        mut on_text: impl FnMut(&str),
    ) -> Result<String> {
        let sliced = images
            .iter()
            .map(|im| self.slice(im))
            .collect::<Result<Vec<_>>>()?;
        let prompt = self.chat_prompt(question, images.len(), thinking);
        let (_, embeds) = self.assemble(&prompt, &sliced)?;
        let tok = self.tokenizer.clone();
        let mut ids: Vec<u32> = Vec::new();
        let mut shown = 0usize;
        self.generate_embeds(&embeds, max_new, opts, |t| {
            ids.push(t);
            if let Ok(s) = tok.decode(&ids, true)
                && !s.ends_with('\u{FFFD}')
                && s.len() > shown
            {
                on_text(&s[shown..]);
                shown = s.len();
            }
            true
        })?;
        self.decode(&ids)
    }

    /// Answer `question` about a packed clip (see [`Self::load_packed_video`]).
    pub fn chat_video(
        &mut self,
        video: &PackedVideo,
        question: &str,
        max_new: usize,
        opts: SampleOpts,
        thinking: bool,
        mut on_text: impl FnMut(&str),
    ) -> Result<String> {
        let (_, embeds) = self.assemble_video(question, video, thinking)?;
        self.release_vision_graphs();
        let tok = self.tokenizer.clone();
        let mut ids: Vec<u32> = Vec::new();
        let mut shown = 0usize;
        self.generate_embeds(&embeds, max_new, opts, |t| {
            ids.push(t);
            if let Ok(s) = tok.decode(&ids, true)
                && !s.ends_with('\u{FFFD}')
                && s.len() > shown
            {
                on_text(&s[shown..]);
                shown = s.len();
            }
            true
        })?;
        self.decode(&ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow!("minicpm-v: decode: {e}"))
    }
}
