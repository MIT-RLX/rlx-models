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

use anyhow::{Context, Result, anyhow, bail};
use rlx_runtime::{CompiledGraph, Device, Session};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::multimodal::{
    AUDIO_MARKER, AUDIO_MARKER_HF, GemmaAudioConfig, GemmaMultimodalConfig, GemmaVisionConfig,
    IMAGE_MARKER, IMAGE_MARKER_HF, MediaSlot, VIDEO_MARKER, VIDEO_MARKER_HF,
    build_audio_projection_graph, build_vision_projection_graph, frame_audio_samples,
    fuse_multimodal_embeddings, load_image_patches, tokenize_with_media,
};
use crate::unified_preprocess::{
    compute_num_soft_tokens_from_size, factorized_pos_bias, load_unified_image,
    prepare_unified_audio_samples, strip_valid_vision_rows, unified_audio_token_count,
};
use crate::unified_projector::{
    build_unified_audio_graph, build_unified_vision_graph, is_unified_vision_weights,
};

/// Which HF projector weight layout is loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProjectorLayout {
    /// llama.cpp-style `vision_tower.*` + learned soft-token pool.
    #[default]
    LegacyPool,
    /// Native `google/gemma-4-12B-it` unified embedders.
    Unified,
}

/// One compiled projector — wraps a `CompiledGraph` + the static
/// dimensions it was specialised for.
struct VisionStage {
    compiled: CompiledGraph,
    num_patches: usize,
    output_dim: usize,
    unified: bool,
}

impl VisionStage {
    fn output_shape(&self) -> (usize, usize) {
        (self.num_patches, self.output_dim)
    }
}

struct AudioStage {
    compiled: CompiledGraph,
    num_frames: usize,
    output_dim: usize,
    unified: bool,
}

impl AudioStage {
    fn output_shape(&self) -> (usize, usize) {
        (self.num_frames, self.output_dim)
    }
}

/// End-to-end multimodal pipeline for Gemma 4 unified models.
///
/// Holds the compiled vision/audio projector graphs (constructed
/// once) and runs them on demand to produce hidden-aligned soft
/// tokens. The LM-side fusion is exposed as
/// [`GemmaMultimodalRunner::fuse_text_and_media`].
pub struct GemmaMultimodalRunner {
    cfg: GemmaMultimodalConfig,
    lm_hidden: usize,
    device: Device,
    layout: ProjectorLayout,
    max_soft_tokens: usize,
    max_audio_tokens: usize,
    vision: Option<VisionStage>,
    audio: Option<AudioStage>,
}

impl GemmaMultimodalRunner {
    /// Construct a runner for a fixed image-grid and audio-frame
    /// budget. Both `num_patches` and `num_frames` are graph-time
    /// constants — call [`Self::reconfigure`] if you need different
    /// dimensions later.
    pub fn new(
        cfg: GemmaMultimodalConfig,
        lm_hidden: usize,
        device: Device,
        max_soft_tokens: Option<usize>,
        max_audio_tokens: Option<usize>,
    ) -> Result<Self> {
        let max_soft = max_soft_tokens.unwrap_or_else(|| {
            cfg.vision
                .as_ref()
                .map(|v| v.num_soft_tokens)
                .unwrap_or(280)
        });
        Ok(Self {
            cfg,
            lm_hidden,
            device,
            layout: ProjectorLayout::LegacyPool,
            max_soft_tokens: max_soft,
            max_audio_tokens: max_audio_tokens.unwrap_or(750),
            vision: None,
            audio: None,
        })
    }

    /// Current vision projector dimensions, when one is compiled.
    /// `(num_patches, output_dim)` — output_dim equals the LM hidden
    /// size for the Gemma 4 12B reference.
    pub fn vision_output_shape(&self) -> Option<(usize, usize)> {
        self.vision.as_ref().map(|s| s.output_shape())
    }

    /// Current audio projector dimensions, when one is compiled.
    /// `(num_frames, output_dim)`.
    pub fn audio_output_shape(&self) -> Option<(usize, usize)> {
        self.audio.as_ref().map(|s| s.output_shape())
    }

    pub fn config(&self) -> &GemmaMultimodalConfig {
        &self.cfg
    }

    pub fn lm_hidden(&self) -> usize {
        self.lm_hidden
    }

    pub fn layout(&self) -> ProjectorLayout {
        self.layout
    }

    pub fn is_unified(&self) -> bool {
        self.layout == ProjectorLayout::Unified
    }

    fn sync_layout(&mut self, weights: &MultimodalWeights) {
        self.layout = weights.layout();
    }

    fn vision_cfg(&self) -> Result<&GemmaVisionConfig> {
        self.cfg
            .vision
            .as_ref()
            .ok_or_else(|| anyhow!("vision config missing — model is text/audio only"))
    }

    fn audio_cfg(&self) -> Result<&GemmaAudioConfig> {
        self.cfg
            .audio
            .as_ref()
            .ok_or_else(|| anyhow!("audio config missing — model is text/vision only"))
    }

    /// Compile (or recompile) the vision projector graph for a
    /// `num_patches` budget. Weights are NOT loaded yet — call
    /// [`Self::load_vision_weights`] before running.
    pub fn compile_vision(&mut self, num_patches: usize) -> Result<()> {
        let vcfg = self.vision_cfg()?.clone();
        let unified = self.layout == ProjectorLayout::Unified;
        let g = if unified {
            build_unified_vision_graph(num_patches, &vcfg)?
        } else {
            build_vision_projection_graph(1, num_patches, &vcfg)?
        };
        let session = Session::new(self.device);
        let compiled = session
            .compile_hir(g.hir)
            .map_err(|e| anyhow!("vision projector lower failed: {e:?}"))?;
        self.vision = Some(VisionStage {
            compiled,
            num_patches,
            output_dim: if unified {
                self.lm_hidden
            } else {
                vcfg.output_proj_dims
            },
            unified,
        });
        Ok(())
    }

    pub fn compile_audio(&mut self, num_frames: usize) -> Result<()> {
        let acfg = self.audio_cfg()?.clone();
        let unified = self.layout == ProjectorLayout::Unified;
        let g = if unified {
            build_unified_audio_graph(num_frames, &acfg, self.lm_hidden)?
        } else {
            build_audio_projection_graph(1, num_frames, &acfg, self.lm_hidden)?
        };
        let session = Session::new(self.device);
        let compiled = session
            .compile_hir(g.hir)
            .map_err(|e| anyhow!("audio projector lower failed: {e:?}"))?;
        self.audio = Some(AudioStage {
            compiled,
            num_frames,
            output_dim: self.lm_hidden,
            unified,
        });
        Ok(())
    }

    /// Reconfigure the vision projector for a new patch count. Calls
    /// [`Self::compile_vision`] followed by
    /// [`Self::load_vision_weights`].
    pub fn reconfigure(
        &mut self,
        num_patches: Option<usize>,
        num_frames: Option<usize>,
        weights: &MultimodalWeights,
    ) -> Result<()> {
        if let Some(n) = num_patches {
            self.compile_vision(n)?;
            self.load_vision_weights(weights)?;
        }
        if let Some(n) = num_frames {
            self.compile_audio(n)?;
            self.load_audio_weights(weights)?;
        }
        Ok(())
    }

    /// Bind vision-projector weights into the compiled graph.
    pub fn load_vision_weights(&mut self, weights: &MultimodalWeights) -> Result<()> {
        self.sync_layout(weights);
        let stage = self
            .vision
            .as_mut()
            .ok_or_else(|| anyhow!("vision projector not compiled — call compile_vision first"))?;
        if stage.unified {
            let vcfg = self.cfg.vision.as_ref().unwrap();
            let d = vcfg.mm_embed_dim;
            for key in [
                "model.vision_embedder.patch_ln1.weight",
                "model.vision_embedder.patch_ln1.bias",
                "model.vision_embedder.patch_dense.weight",
                "model.vision_embedder.patch_dense.bias",
                "model.vision_embedder.patch_ln2.weight",
                "model.vision_embedder.patch_ln2.bias",
                "model.vision_embedder.pos_norm.weight",
                "model.vision_embedder.pos_norm.bias",
                "model.embed_vision.embedding_projection.weight",
            ] {
                let data = weights.get_linear(key)?;
                stage.compiled.set_param(key, &data);
            }
            stage.compiled.set_param("unified.ones", &vec![1.0f32; d]);
            stage
                .compiled
                .set_param("unified.zero_beta", &vec![0.0f32; d]);
        } else {
            for key in [
                "vision_tower.embed.weight",
                "vision_tower.pos_embed.weight",
                "vision_tower.norm.weight",
                "vision_tower.soft_token.weight",
                "vision_tower.lm_proj.weight",
            ] {
                let data = weights.get(key)?;
                stage.compiled.set_param(key, data);
            }
            let vcfg = self.cfg.vision.as_ref().unwrap();
            stage
                .compiled
                .set_param("vision_tower.ones", &vec![1.0f32; vcfg.mm_embed_dim]);
            stage
                .compiled
                .set_param("vision_tower.zero_beta", &vec![0.0f32; vcfg.mm_embed_dim]);
        }
        Ok(())
    }

    /// Bind audio-projector weights into the compiled graph.
    pub fn load_audio_weights(&mut self, weights: &MultimodalWeights) -> Result<()> {
        self.sync_layout(weights);
        let stage = self
            .audio
            .as_mut()
            .ok_or_else(|| anyhow!("audio projector not compiled — call compile_audio first"))?;
        if stage.unified {
            let acfg = self.cfg.audio.as_ref().unwrap();
            let d = acfg.audio_embed_dim;
            let data = weights.get_linear("model.embed_audio.embedding_projection.weight")?;
            stage
                .compiled
                .set_param("model.embed_audio.embedding_projection.weight", &data);
            stage
                .compiled
                .set_param("unified.audio.ones", &vec![1.0f32; d]);
            stage
                .compiled
                .set_param("unified.audio.zero_beta", &vec![0.0f32; d]);
        } else {
            for key in [
                "audio_tower.embed.weight",
                "audio_tower.norm.weight",
                "audio_tower.lm_proj.weight",
            ] {
                let data = weights.get(key)?;
                stage.compiled.set_param(key, data);
            }
            let acfg = self.cfg.audio.as_ref().unwrap();
            stage
                .compiled
                .set_param("audio_tower.ones", &vec![1.0f32; acfg.audio_embed_dim]);
            stage
                .compiled
                .set_param("audio_tower.zero_beta", &vec![0.0f32; acfg.audio_embed_dim]);
        }
        Ok(())
    }

    /// Project a single image (already preprocessed to patches) into LM hidden rows.
    pub fn project_image_patches(&mut self, patches: &[f32]) -> Result<Vec<f32>> {
        self.project_image_with_pos(patches, None)
    }

    /// Unified vision path: `patches` + optional CPU `pos_bias`.
    pub fn project_image_with_pos(
        &mut self,
        patches: &[f32],
        pos_bias: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let stage = self
            .vision
            .as_mut()
            .ok_or_else(|| anyhow!("vision projector not compiled"))?;
        let outs = if stage.unified {
            let pos = pos_bias.ok_or_else(|| anyhow!("unified vision requires pos_bias"))?;
            stage
                .compiled
                .run(&[("patches", patches), ("pos_bias", pos)])
        } else {
            stage.compiled.run(&[("patches", patches)])
        };
        outs.into_iter()
            .next()
            .ok_or_else(|| anyhow!("vision projector returned no outputs"))
    }

    /// Convenience: load a JPEG/PNG, resize to fit the configured
    /// patch budget, run the projector. The vision graph is
    /// recompiled when the produced patch count differs from the
    /// current setting (so callers pinning to a single image size
    /// pay zero overhead).
    pub fn project_image_file(
        &mut self,
        path: impl AsRef<Path>,
        weights: &MultimodalWeights,
        max_side_patches: usize,
    ) -> Result<Vec<f32>> {
        self.sync_layout(weights);
        if self.layout == ProjectorLayout::Unified {
            let vcfg = self.vision_cfg()?.clone();
            let img = load_unified_image(
                path.as_ref(),
                vcfg.patch_size,
                vcfg.pooling_kernel_size,
                self.max_soft_tokens,
            )?;
            let num_slots = self.max_soft_tokens;
            let need_recompile = match &self.vision {
                Some(s) => s.num_patches != num_slots || !s.unified,
                None => true,
            };
            if need_recompile {
                self.compile_vision(num_slots)?;
                self.load_vision_weights(weights)?;
            }
            let pos_table = weights.get("model.vision_embedder.pos_embedding")?;
            let pos_bias = factorized_pos_bias(
                pos_table,
                vcfg.mm_posemb_size,
                vcfg.mm_embed_dim,
                &img.positions,
            );
            let projected = self.project_image_with_pos(&img.patches, Some(&pos_bias))?;
            return Ok(strip_valid_vision_rows(
                &projected,
                &img.positions,
                self.lm_hidden,
            ));
        }

        let vcfg = self.vision_cfg()?.clone();
        let (patches, grid_h, grid_w) =
            load_image_patches(path, vcfg.patch_size, max_side_patches)?;
        let num_patches = grid_h * grid_w;
        let need_recompile = match &self.vision {
            Some(s) => s.num_patches != num_patches || s.unified,
            None => true,
        };
        if need_recompile {
            self.compile_vision(num_patches)?;
            self.load_vision_weights(weights)?;
        }
        self.project_image_patches(&patches)
    }

    /// Project an audio waveform (already at 16 kHz mono f32) into
    /// `[num_frames, lm_hidden]` audio soft tokens.
    /// Project one video frame (unified layout, 70 soft tokens by default).
    pub fn project_video_frame(
        &mut self,
        path: impl AsRef<Path>,
        weights: &MultimodalWeights,
    ) -> Result<Vec<f32>> {
        let prev = self.max_soft_tokens;
        self.max_soft_tokens = 70;
        let out = self.project_image_file(path, weights, 32);
        self.max_soft_tokens = prev;
        out
    }

    /// Dynamic soft-token count for an on-disk image (unified layout).
    pub fn image_soft_token_count(&self, path: impl AsRef<Path>) -> Result<usize> {
        let vcfg = self.vision_cfg()?.clone();
        let img =
            image::open(path.as_ref()).map_err(|e| anyhow!("decode {:?}: {e}", path.as_ref()))?;
        let (w, h) = (img.width() as usize, img.height() as usize);
        compute_num_soft_tokens_from_size(
            h,
            w,
            vcfg.patch_size,
            vcfg.pooling_kernel_size,
            self.max_soft_tokens,
        )
    }

    pub fn project_audio_samples(
        &mut self,
        samples: &[f32],
        weights: &MultimodalWeights,
    ) -> Result<Vec<f32>> {
        self.sync_layout(weights);
        let acfg = self.audio_cfg()?.clone();
        let prepared = if self.layout == ProjectorLayout::Unified {
            prepare_unified_audio_samples(
                samples,
                acfg.audio_samples_per_token,
                self.max_audio_tokens,
            )
        } else {
            samples.to_vec()
        };
        let (frames, num_frames) = frame_audio_samples(&prepared, acfg.audio_samples_per_token)?;
        let effective_frames = if self.layout == ProjectorLayout::Unified {
            unified_audio_token_count(
                samples
                    .len()
                    .min(crate::unified_preprocess::MAX_AUDIO_SAMPLES),
                acfg.audio_samples_per_token,
                self.max_audio_tokens,
            )
        } else {
            num_frames
        };
        let need_recompile = match &self.audio {
            Some(s) => {
                s.num_frames != num_frames || s.unified != (self.layout == ProjectorLayout::Unified)
            }
            None => true,
        };
        if need_recompile {
            self.compile_audio(num_frames)?;
            self.load_audio_weights(weights)?;
        }
        let stage = self
            .audio
            .as_mut()
            .ok_or_else(|| anyhow!("audio projector not compiled"))?;
        let outs = stage.compiled.run(&[("frames", &frames[..])]);
        let projected = outs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("audio projector returned no outputs"))?;
        if self.layout == ProjectorLayout::Unified {
            Ok(projected[..effective_frames * self.lm_hidden].to_vec())
        } else {
            Ok(projected)
        }
    }

    /// Convenience: load and project a WAV file.
    pub fn project_audio_file(
        &mut self,
        path: impl AsRef<Path>,
        weights: &MultimodalWeights,
    ) -> Result<Vec<f32>> {
        self.sync_layout(weights);
        let samples = crate::multimodal::load_wav_mono_16khz(path)?;
        if self.audio.is_none() {
            self.compile_audio(
                samples
                    .len()
                    .div_ceil(self.audio_cfg()?.audio_samples_per_token),
            )?;
            self.load_audio_weights(weights)?;
        }
        let was_present = self.audio.is_some();
        let result = self.project_audio_samples(&samples, weights);
        if !was_present {
            self.load_audio_weights(weights)?;
        }
        result
    }

    pub fn fuse_text_and_media(
        &self,
        text_embeds: &mut [f32],
        token_ids: &[u32],
        image_embeds: &[f32],
        audio_embeds: &[f32],
        video_embeds: &[f32],
    ) -> Result<()> {
        fuse_multimodal_embeddings(
            text_embeds,
            token_ids,
            self.lm_hidden,
            &self.cfg,
            image_embeds,
            audio_embeds,
            video_embeds,
        )
    }

    /// Tokenize a multimodal prompt with per-media dynamic placeholder counts.
    pub fn tokenize_prompt<F>(
        &self,
        prompt: &str,
        image_soft_counts: &[usize],
        audio_sample_lengths: &[usize],
        video_soft_counts: &[usize],
        encode_fn: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(&str) -> Result<Vec<u32>>,
    {
        let audio_per = self
            .cfg
            .audio
            .as_ref()
            .map(|a| a.audio_samples_per_token)
            .unwrap_or(640);

        let mut slots: Vec<MediaSlot> = Vec::new();
        let mut img_idx = 0usize;
        let mut aud_idx = 0usize;
        let mut vid_idx = 0usize;
        let mut cursor = 0usize;
        while cursor <= prompt.len() {
            let remainder = &prompt[cursor..];
            let next = find_next_marker(remainder);
            match next {
                Some((off, kind, marker_len)) => {
                    match kind {
                        MarkerKind::Image => {
                            let count = *image_soft_counts.get(img_idx).ok_or_else(|| {
                                anyhow!(
                                    "not enough image_soft_counts for marker at offset {cursor}"
                                )
                            })?;
                            slots.push(MediaSlot::Image { count });
                            img_idx += 1;
                        }
                        MarkerKind::Audio => {
                            let n = *audio_sample_lengths.get(aud_idx).ok_or_else(|| {
                                anyhow!(
                                    "not enough audio_sample_lengths for marker at offset {cursor}"
                                )
                            })?;
                            let count = if audio_per == 640 {
                                unified_audio_token_count(
                                    n.min(crate::unified_preprocess::MAX_AUDIO_SAMPLES),
                                    audio_per,
                                    self.max_audio_tokens,
                                )
                            } else {
                                n.div_ceil(audio_per).max(1)
                            };
                            slots.push(MediaSlot::Audio { count });
                            aud_idx += 1;
                        }
                        MarkerKind::Video => {
                            let count = *video_soft_counts.get(vid_idx).ok_or_else(|| {
                                anyhow!(
                                    "not enough video_soft_counts for marker at offset {cursor}"
                                )
                            })?;
                            slots.push(MediaSlot::Video { count });
                            vid_idx += 1;
                        }
                    }
                    cursor += off + marker_len;
                }
                None => break,
            }
        }
        if img_idx != image_soft_counts.len() {
            bail!(
                "prompt has {img_idx} image markers but {} image_soft_counts supplied",
                image_soft_counts.len()
            );
        }
        if aud_idx != audio_sample_lengths.len() {
            bail!(
                "prompt has {aud_idx} audio markers but {} audio lengths supplied",
                audio_sample_lengths.len()
            );
        }
        if vid_idx != video_soft_counts.len() {
            bail!(
                "prompt has {vid_idx} video markers but {} video_soft_counts supplied",
                video_soft_counts.len()
            );
        }
        tokenize_with_media(prompt, &slots, &self.cfg, encode_fn)
    }
}

#[derive(Clone, Copy)]
enum MarkerKind {
    Image,
    Audio,
    Video,
}

fn find_next_marker(prompt: &str) -> Option<(usize, MarkerKind, usize)> {
    let candidates: &[(&str, MarkerKind)] = &[
        (IMAGE_MARKER_HF, MarkerKind::Image),
        (IMAGE_MARKER, MarkerKind::Image),
        (AUDIO_MARKER_HF, MarkerKind::Audio),
        (AUDIO_MARKER, MarkerKind::Audio),
        (VIDEO_MARKER_HF, MarkerKind::Video),
        (VIDEO_MARKER, MarkerKind::Video),
    ];
    let mut best: Option<(usize, MarkerKind, usize)> = None;
    for &(m, kind) in candidates {
        if let Some(i) = prompt.find(m) {
            if best.map(|(bi, _, _)| i < bi).unwrap_or(true) {
                best = Some((i, kind, m.len()));
            }
        }
    }
    best
}

// ── Safetensors weights loader ──────────────────────────────────

/// In-memory weight bundle for the multimodal projectors. Owns the
/// raw f32 data; lookup-by-key in `get()` returns a borrowed slice
/// that's lifetime-tied to the bundle.
pub struct MultimodalWeights {
    data: HashMap<String, Vec<f32>>,
    /// Original file path (just for diagnostics).
    pub source: Option<PathBuf>,
}

impl MultimodalWeights {
    pub fn empty() -> Self {
        Self {
            data: HashMap::new(),
            source: None,
        }
    }

    /// Load vision/audio projector tensors from a safetensors file.
    /// Supports legacy `vision_tower.*` / `audio_tower.*` and unified
    /// `model.vision_embedder.*` / `model.embed_*` keys.
    pub fn from_safetensors(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).with_context(|| format!("read {path:?}"))?;
        let st = SafeTensors::deserialize(&bytes)
            .map_err(|e| anyhow!("parse safetensors {path:?}: {e}"))?;
        let mut data = HashMap::new();
        for (name, view) in st.tensors() {
            if !is_multimodal_tensor_key(&name) {
                continue;
            }
            let shape: Vec<usize> = view.shape().to_vec();
            let mut f32_data = tensor_to_f32(&view).with_context(|| format!("decode {name}"))?;
            if should_transpose_hf_linear(&name, &shape) {
                f32_data = transpose_2d(&f32_data, shape[0], shape[1]);
            }
            data.insert(name, f32_data);
        }
        Ok(Self {
            data,
            source: Some(path.to_path_buf()),
        })
    }

    /// Load projector weights from a llama.cpp-style **`mmproj.gguf`**
    /// companion file. llama.cpp ships multimodal projector weights
    /// in a separate GGUF next to the LM `model.gguf` — this loader
    /// drains every tensor whose name starts with `vision_tower.` /
    /// `audio_tower.` and dequantizes it to F32.
    ///
    /// Tensors with other prefixes are ignored. The dequantization
    /// path supports every K-quant scheme `rlx_gguf` decodes
    /// (`Q8_0`, `Q4_0`, `Q4_K_M`, `Q5_K_S`, `Q6_K`, F16, BF16, F32).
    pub fn from_mmproj_gguf(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = rlx_gguf::GgufFile::from_path(path)
            .with_context(|| format!("open mmproj GGUF {path:?}"))?;
        let mut data = HashMap::new();
        let keys: Vec<String> = file.keys().map(str::to_string).collect();
        for name in keys {
            if !is_multimodal_tensor_key(&name) {
                continue;
            }
            let (decoded, shape) = file
                .dequant_f32(&name)
                .with_context(|| format!("dequant `{name}`"))?;
            let mut f32_data = decoded;
            if shape.len() == 2 && should_transpose_hf_linear(&name, &shape) {
                f32_data = transpose_2d(&f32_data, shape[0], shape[1]);
            }
            data.insert(name, f32_data);
        }
        Ok(Self {
            data,
            source: Some(path.to_path_buf()),
        })
    }

    /// Insert a custom-source tensor (mainly for tests + callers that
    /// pull weights from a quantized GGUF).
    pub fn insert(&mut self, key: impl Into<String>, data: Vec<f32>) {
        self.data.insert(key.into(), data);
    }

    pub fn get(&self, key: &str) -> Result<&[f32]> {
        self.data
            .get(key)
            .map(|v| v.as_slice())
            .ok_or_else(|| anyhow!("multimodal weight missing: `{key}`"))
    }

    /// Matmul-ready linear weight (HF `out×in` already transposed when loaded).
    pub fn get_linear(&self, key: &str) -> Result<Vec<f32>> {
        Ok(self.get(key)?.to_vec())
    }

    pub fn layout(&self) -> ProjectorLayout {
        if is_unified_vision_weights(self.keys()) {
            ProjectorLayout::Unified
        } else {
            ProjectorLayout::LegacyPool
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.data.keys().map(|s| s.as_str())
    }
}

fn is_multimodal_tensor_key(name: &str) -> bool {
    name.starts_with("vision_tower.")
        || name.starts_with("audio_tower.")
        || name.starts_with("model.vision_embedder.")
        || name.starts_with("model.embed_vision.")
        || name.starts_with("model.embed_audio.")
}

fn should_transpose_hf_linear(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2
        && (name.contains("patch_dense.weight") || name.contains("embedding_projection.weight"))
}

fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

fn tensor_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    use safetensors::tensor::Dtype;
    match view.dtype() {
        Dtype::F32 => {
            let raw = view.data();
            if raw.len() % 4 != 0 {
                bail!("F32 tensor data length not multiple of 4");
            }
            let mut out = Vec::with_capacity(raw.len() / 4);
            for ch in raw.chunks_exact(4) {
                out.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
            }
            Ok(out)
        }
        Dtype::F16 => {
            let raw = view.data();
            if raw.len() % 2 != 0 {
                bail!("F16 tensor data length not multiple of 2");
            }
            let mut out = Vec::with_capacity(raw.len() / 2);
            for ch in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([ch[0], ch[1]]);
                out.push(f16_to_f32(bits));
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let raw = view.data();
            if raw.len() % 2 != 0 {
                bail!("BF16 tensor data length not multiple of 2");
            }
            let mut out = Vec::with_capacity(raw.len() / 2);
            for ch in raw.chunks_exact(2) {
                // bf16 = top 16 bits of an f32.
                let bits = u32::from(u16::from_le_bytes([ch[0], ch[1]])) << 16;
                out.push(f32::from_bits(bits));
            }
            Ok(out)
        }
        other => bail!("unsupported tensor dtype for multimodal weight: {other:?}"),
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            (sign as u32) << 31
        } else {
            // Subnormal — normalize.
            let mut m = mant as u32;
            let mut e: i32 = -14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            ((sign as u32) << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        ((sign as u32) << 31) | (0xff << 23) | ((mant as u32) << 13)
    } else {
        ((sign as u32) << 31) | (((exp as i32 + 127 - 15) as u32) << 23) | ((mant as u32) << 13)
    };
    f32::from_bits(f32_bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multimodal::{GemmaAudioConfig, GemmaMultimodalConfig, GemmaVisionConfig};

    fn tiny_cfg() -> GemmaMultimodalConfig {
        GemmaMultimodalConfig {
            vision: Some(GemmaVisionConfig {
                patch_size: 2,
                model_patch_size: 4,
                mm_embed_dim: 8,
                mm_posemb_size: 16,
                num_soft_tokens: 4,
                output_proj_dims: 8,
                pooling_kernel_size: 1,
                rms_norm_eps: 1e-6,
            }),
            audio: Some(GemmaAudioConfig {
                hidden_size: 4,
                audio_embed_dim: 4,
                audio_samples_per_token: 8,
                output_proj_dims: 4,
                rms_norm_eps: 1e-6,
            }),
            image_token_id: Some(100),
            audio_token_id: Some(200),
            boi_token_id: Some(80),
            eoi_token_id: Some(81),
            boa_token_id: Some(90),
            eoa_token_index: Some(91),
            ..Default::default()
        }
    }

    fn synthetic_weights(num_patches: usize, num_frames: usize) -> MultimodalWeights {
        let v = GemmaVisionConfig {
            patch_size: 2,
            model_patch_size: 4,
            mm_embed_dim: 8,
            mm_posemb_size: 16,
            num_soft_tokens: 4,
            output_proj_dims: 8,
            pooling_kernel_size: 1,
            rms_norm_eps: 1e-6,
        };
        let a = GemmaAudioConfig {
            hidden_size: 4,
            audio_embed_dim: 4,
            audio_samples_per_token: 8,
            output_proj_dims: 4,
            rms_norm_eps: 1e-6,
        };
        let patch_features = v.patch_size * v.patch_size * 3;
        let mut w = MultimodalWeights::empty();
        // All-zero weights are fine for "graph runs end-to-end" — we
        // just need the matrices to have the right shapes.
        w.insert(
            "vision_tower.embed.weight",
            vec![0.01f32; patch_features * v.mm_embed_dim],
        );
        w.insert(
            "vision_tower.pos_embed.weight",
            vec![0.0f32; num_patches * v.mm_embed_dim],
        );
        w.insert("vision_tower.norm.weight", vec![0.0f32; v.mm_embed_dim]);
        w.insert(
            "vision_tower.soft_token.weight",
            // Patch-axis reducer: [num_patches, num_soft_tokens].
            vec![0.01f32; num_patches * v.num_soft_tokens],
        );
        w.insert(
            "vision_tower.lm_proj.weight",
            // Feature projection: [mm_embed_dim, output_proj_dims].
            vec![0.01f32; v.mm_embed_dim * v.output_proj_dims],
        );
        w.insert(
            "audio_tower.embed.weight",
            vec![0.01f32; a.audio_samples_per_token * a.audio_embed_dim],
        );
        w.insert("audio_tower.norm.weight", vec![0.0f32; a.audio_embed_dim]);
        // lm_hidden in the tiny test is 8 (= mm_embed_dim).
        w.insert(
            "audio_tower.lm_proj.weight",
            vec![0.01f32; a.audio_embed_dim * 8],
        );
        let _ = num_frames;
        w
    }

    #[test]
    fn runner_compiles_and_runs_vision_projector_on_cpu() {
        let cfg = tiny_cfg();
        let num_patches = 4;
        let mut runner = GemmaMultimodalRunner::new(cfg, 8, Device::Cpu, None, None).unwrap();
        let weights = synthetic_weights(num_patches, 0);
        runner.compile_vision(num_patches).unwrap();
        runner.load_vision_weights(&weights).unwrap();
        // 4 patches × 2*2*3 = 12 features per patch.
        let patches = vec![0.5f32; num_patches * 12];
        let out = runner.project_image_patches(&patches).unwrap();
        // Output shape: [num_soft_tokens=4, output_proj_dims=8] = 32.
        assert_eq!(out.len(), 4 * 8);
    }

    #[test]
    fn runner_compiles_and_runs_audio_projector_on_cpu() {
        let cfg = tiny_cfg();
        let num_frames = 2;
        let mut runner = GemmaMultimodalRunner::new(cfg, 8, Device::Cpu, None, None).unwrap();
        let weights = synthetic_weights(0, num_frames);
        runner.compile_audio(num_frames).unwrap();
        runner.load_audio_weights(&weights).unwrap();
        // 2 frames × 8 samples per frame.
        let frames = vec![0.1f32; num_frames * 8];
        let out = runner.project_audio_samples(&frames, &weights).unwrap();
        // [num_frames=2, lm_hidden=8] = 16 floats.
        assert_eq!(out.len(), 2 * 8);
    }

    /// Encode an f32 as its bf16 byte representation (top 16 bits).
    fn f32_to_bf16_bytes(f: f32) -> [u8; 2] {
        let bits = f.to_bits();
        let bf16 = ((bits >> 16) & 0xffff) as u16;
        bf16.to_le_bytes()
    }

    #[test]
    fn mmproj_gguf_loader_rejects_missing_file() {
        // Sanity: missing file surfaces as an open error rather than
        // panicking. Real round-trip is exercised by callers that
        // pass an actual llama.cpp-produced `mmproj.gguf`.
        match MultimodalWeights::from_mmproj_gguf("/nonexistent/mmproj.gguf") {
            Ok(_) => panic!("expected error for missing mmproj path"),
            Err(err) => {
                let msg = format!("{err:#}");
                assert!(
                    msg.contains("mmproj") || msg.contains("open") || msg.contains("No such file"),
                    "unexpected error message: {msg}"
                );
            }
        }
    }

    #[test]
    fn weights_loader_decodes_bf16_safetensors() {
        // bf16 keeps top 16 bits of an f32, so round-trip values are
        // exact at the bf16-precision rung. Pick values whose lower
        // 16 mantissa bits are zero to avoid quantization here.
        let originals: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5];
        let mut payload: Vec<u8> = Vec::with_capacity(originals.len() * 2);
        for &f in &originals {
            payload.extend_from_slice(&f32_to_bf16_bytes(f));
        }
        let header = format!(
            r#"{{"audio_tower.norm.weight":{{"dtype":"BF16","shape":[4],"data_offsets":[0,{}]}}}}"#,
            payload.len(),
        );
        let header_bytes = header.as_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(header_bytes);
        buf.extend_from_slice(&payload);
        let tmp = std::env::temp_dir().join("rlx_gemma_mm_bf16_test.safetensors");
        std::fs::write(&tmp, &buf).unwrap();
        let w = MultimodalWeights::from_safetensors(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();
        let decoded = w.get("audio_tower.norm.weight").unwrap();
        assert_eq!(decoded.len(), originals.len());
        for (got, want) in decoded.iter().zip(originals.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "bf16 decode: got {got} want {want}"
            );
        }
    }

    #[test]
    fn weights_loader_decodes_f32_safetensors() {
        // safetensors header is JSON; build one in-memory.
        let payload: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let header = format!(
            r#"{{"vision_tower.norm.weight":{{"dtype":"F32","shape":[4],"data_offsets":[0,{}]}}}}"#,
            payload.len(),
        );
        let header_bytes = header.as_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(header_bytes);
        buf.extend_from_slice(&payload);
        // Write to a tmp file and round-trip through from_safetensors.
        let tmp = std::env::temp_dir().join("rlx_gemma_multimodal_test.safetensors");
        std::fs::write(&tmp, &buf).unwrap();
        let w = MultimodalWeights::from_safetensors(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();
        assert_eq!(
            w.get("vision_tower.norm.weight").unwrap(),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn tokenize_prompt_derives_slot_counts() {
        let cfg = tiny_cfg();
        let runner = GemmaMultimodalRunner::new(cfg, 8, Device::Cpu, None, None).unwrap();
        // num_soft_tokens=4 (per vision config), audio_per=8.
        let encode = |s: &str| -> Result<Vec<u32>> { Ok(s.bytes().map(|b| b as u32).collect()) };
        // 16 audio samples → ceil(16/8)=2 audio tokens.
        let out = runner
            .tokenize_prompt("hi <image> see <audio>", &[4], &[16], &[], encode)
            .unwrap();
        // "hi " + boi + image*4 + eoi + " see " + boa + audio*2 + eoa
        let mut expected: Vec<u32> = b"hi ".iter().map(|b| *b as u32).collect();
        expected.extend([80, 100, 100, 100, 100, 81]);
        expected.extend(b" see ".iter().map(|b| *b as u32));
        expected.extend([90, 200, 200, 91]);
        assert_eq!(out, expected);
    }

    #[test]
    fn fuse_text_and_media_replaces_in_order() {
        let cfg = tiny_cfg();
        let runner = GemmaMultimodalRunner::new(cfg, 4, Device::Cpu, None, None).unwrap();
        let mut text = vec![0.0f32; 4 * 4]; // 4 tokens, hidden=4
        let ids = [10, 100, 200, 11];
        let img = vec![7.0f32; 4];
        let aud = vec![9.0f32; 4];
        runner
            .fuse_text_and_media(&mut text, &ids, &img, &aud, &[])
            .unwrap();
        assert_eq!(&text[4..8], &[7.0; 4]);
        assert_eq!(&text[8..12], &[9.0; 4]);
    }
}
