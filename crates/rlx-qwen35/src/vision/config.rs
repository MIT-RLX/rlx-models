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

//! Qwen3.5 VLM mmproj config — parsed from GGUF `clip.*` metadata keys
//! (llama.cpp `tools/mtmd/clip-impl.h`) or HF `vision_config`.

use anyhow::{Result, anyhow};
use rlx_gguf::{GgufFile, MetaValue};
use std::path::Path;

/// Vision / mmproj hyperparameters from a Qwen3-VL `mmproj` GGUF.
#[derive(Debug, Clone)]
pub struct MmProjConfig {
    pub patch_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub image_size: usize,
    pub image_min_pixels: usize,
    pub image_max_pixels: usize,
    /// Spatial merge factor (`clip.vision.spatial_merge_size` / `n_merge`).
    pub n_merge: usize,
    pub eps: f64,
    pub projector_type: String,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub spatial_merge_size: usize,
    /// LLM hidden size (`clip.vision.projection_dim`).
    pub llm_hidden_size: usize,
    pub n_ff: usize,
    /// Layer indices with deepstack side paths (optional).
    pub deepstack_layers: Vec<usize>,
}

impl MmProjConfig {
    /// Read from a mmproj GGUF (`general.architecture` is typically `clip`).
    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let u32k = |k: &str| -> Result<u32> {
            raw.metadata
                .get(k)
                .and_then(MetaValue::as_u32)
                .ok_or_else(|| anyhow!("missing clip metadata key: {k}"))
        };
        let u32k_opt = |k: &str| -> Option<u32> { raw.metadata.get(k).and_then(MetaValue::as_u32) };
        let f32k = |k: &str| -> Option<f32> {
            raw.metadata.get(k).and_then(|v| match v {
                MetaValue::F32(x) => Some(*x),
                _ => None,
            })
        };
        let strk = |k: &str| -> Result<String> {
            raw.metadata
                .get(k)
                .and_then(MetaValue::as_str)
                .map(str::to_string)
                .ok_or_else(|| anyhow!("missing clip metadata key: {k}"))
        };
        let arr_f32 = |k: &str| -> Result<[f32; 3]> {
            let arr = raw
                .metadata
                .get(k)
                .and_then(|v| match v {
                    MetaValue::Array(a) => Some(
                        a.iter()
                            .filter_map(|x| match x {
                                MetaValue::F32(f) => Some(*f),
                                MetaValue::F64(f) => Some(*f as f32),
                                _ => None,
                            })
                            .collect::<Vec<_>>(),
                    ),
                    _ => None,
                })
                .ok_or_else(|| anyhow!("missing clip metadata key: {k}"))?;
            if arr.len() != 3 {
                return Err(anyhow!("{k}: expected 3 floats, got {}", arr.len()));
            }
            Ok([arr[0], arr[1], arr[2]])
        };
        let arr_bool_layers = |k: &str| -> Vec<usize> {
            raw.metadata
                .get(k)
                .and_then(|v| match v {
                    MetaValue::Array(a) => Some(
                        a.iter()
                            .enumerate()
                            .filter_map(|(i, x)| match x {
                                MetaValue::Bool(b) if *b => Some(i),
                                MetaValue::U32(u) if *u != 0 => Some(i),
                                _ => None,
                            })
                            .collect(),
                    ),
                    _ => None,
                })
                .unwrap_or_default()
        };

        let patch_size = u32k("clip.vision.patch_size")? as usize;
        let n_embd = u32k("clip.vision.embedding_length")? as usize;
        let n_head = u32k("clip.vision.attention.head_count")? as usize;
        let n_layer = u32k("clip.vision.block_count")? as usize;
        let image_size = u32k("clip.vision.image_size")? as usize;
        let n_merge = u32k_opt("clip.vision.spatial_merge_size")
            .or_else(|| u32k_opt("clip.vision.projector.scale_factor"))
            .unwrap_or(2) as usize;
        let spatial_merge_size = n_merge;

        let image_min_pixels = u32k_opt("clip.vision.image_min_pixels")
            .map(|v| v as usize)
            .unwrap_or(1024 * n_merge * n_merge * patch_size * patch_size);
        let image_max_pixels = u32k_opt("clip.vision.image_max_pixels")
            .map(|v| v as usize)
            .unwrap_or(4096 * n_merge * n_merge * patch_size * patch_size);

        let mut cfg = Self {
            patch_size,
            n_embd,
            n_head,
            n_layer,
            image_size,
            image_min_pixels,
            image_max_pixels,
            n_merge,
            eps: f32k("clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-6) as f64,
            projector_type: strk("clip.projector_type")
                .or_else(|_| strk("clip.vision.projector_type"))
                .unwrap_or_else(|_| "qwen3vl".to_string()),
            image_mean: arr_f32("clip.vision.image_mean").unwrap_or([0.5, 0.5, 0.5]),
            image_std: arr_f32("clip.vision.image_std").unwrap_or([0.5, 0.5, 0.5]),
            spatial_merge_size,
            llm_hidden_size: u32k("clip.vision.projection_dim")? as usize,
            n_ff: u32k("clip.vision.feed_forward_length")? as usize,
            deepstack_layers: arr_bool_layers("clip.vision.is_deepstack_layers"),
        };
        // Honor the same low-mem pixel caps as the HF path. A large image
        // (e.g. Qwen3.6's ~1M-pixel default) yields ~1k vision tokens, which
        // blows past the compiled `max_seq` / Metal buffer limit on a 27B; a
        // lower cap keeps the multimodal prefill in memory.
        cfg.apply_pixel_env_overrides();
        Ok(cfg)
    }

    /// Apply `RLX_QWEN35_IMAGE_MIN_PIXELS` / `RLX_QWEN35_IMAGE_MAX_PIXELS`
    /// overrides (used by both the GGUF and HF config paths).
    fn apply_pixel_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("RLX_QWEN35_IMAGE_MIN_PIXELS") {
            if let Ok(n) = v.parse::<usize>() {
                self.image_min_pixels = n.max(1);
            }
        }
        if let Ok(v) = std::env::var("RLX_QWEN35_IMAGE_MAX_PIXELS") {
            if let Ok(n) = v.parse::<usize>() {
                self.image_max_pixels = n.max(self.image_min_pixels);
            }
        }
    }

    /// Read `vision_config` from a HuggingFace Qwen3.5 / Fara multimodal
    /// `config.json`. Image mean/std default to the Qwen3-VL preprocessor
    /// constants (`0.5`).
    pub fn from_hf_config_json(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("qwen35 vision: read {path:?}: {e}"))?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| anyhow!("qwen35 vision: parse {path:?}: {e}"))?;
        Self::from_hf_config_value(&v, path)
    }

    /// Parse vision hyperparameters from an already-decoded HF config.
    pub fn from_hf_config_value(v: &serde_json::Value, path: &Path) -> Result<Self> {
        let top = v
            .as_object()
            .ok_or_else(|| anyhow!("qwen35 vision: {path:?} is not a JSON object"))?;
        let vc = top
            .get("vision_config")
            .and_then(|c| c.as_object())
            .ok_or_else(|| anyhow!("qwen35 vision: missing vision_config in {path:?}"))?;
        let u =
            |k: &str| -> Option<usize> { vc.get(k).and_then(|x| x.as_u64()).map(|n| n as usize) };
        let f = |k: &str| -> Option<f64> { vc.get(k).and_then(|x| x.as_f64()) };

        let patch_size = u("patch_size").unwrap_or(16);
        let n_embd = u("hidden_size")
            .ok_or_else(|| anyhow!("qwen35 vision: missing hidden_size in {path:?}"))?;
        let n_head = u("num_heads")
            .or_else(|| u("num_attention_heads"))
            .unwrap_or(16);
        let n_layer = u("depth")
            .or_else(|| u("num_hidden_layers"))
            .ok_or_else(|| anyhow!("qwen35 vision: missing depth in {path:?}"))?;
        let n_merge = u("spatial_merge_size").unwrap_or(2);
        let n_ff = u("intermediate_size").unwrap_or(n_embd * 4);
        let llm_hidden_size = u("out_hidden_size").unwrap_or(n_embd);
        let num_pos = u("num_position_embeddings").unwrap_or(2304);
        // `num_position_embeddings` is a square grid (e.g. 48² = 2304).
        let grid = (num_pos as f64).sqrt().round() as usize;
        let image_size = grid.saturating_mul(patch_size).max(patch_size);

        let deepstack_layers = vc
            .get("deepstack_visual_indexes")
            .and_then(|a| a.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let mut cfg = Self {
            patch_size,
            n_embd,
            n_head,
            n_layer,
            image_size,
            image_min_pixels: 1024 * n_merge * n_merge * patch_size * patch_size,
            image_max_pixels: 4096 * n_merge * n_merge * patch_size * patch_size,
            n_merge,
            eps: f("rms_norm_eps")
                .or_else(|| f("layer_norm_eps"))
                .unwrap_or(1e-6),
            projector_type: "qwen3vl".into(),
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            spatial_merge_size: n_merge,
            llm_hidden_size,
            n_ff,
            deepstack_layers,
        };
        // Low-mem / short-ctx overrides (Fara BF16 otherwise upscales to ≥1M pixels).
        cfg.apply_pixel_env_overrides();
        Ok(cfg)
    }

    /// Alignment factor for smart resize (`patch_size * n_merge`).
    pub fn align_size(&self) -> usize {
        self.patch_size * self.n_merge
    }

    /// Output vision token grid after dual-conv patch embed (llama.cpp `clip_n_output_tokens_*`).
    pub fn output_grid(&self, img_w: usize, img_h: usize) -> (usize, usize) {
        let gx = (img_w / self.patch_size) / 2;
        let gy = (img_h / self.patch_size) / 2;
        (gx.max(1), gy.max(1))
    }

    /// Number of LLM-side vision tokens after mm 4× merge.
    pub fn n_out_tokens(&self, img_w: usize, img_h: usize) -> usize {
        let (gx, gy) = self.output_grid(img_w, img_h);
        gx * gy
    }

    /// Patch-grid token count inside the ViT trunk (before mm merge).
    pub fn n_patch_tokens(&self, img_w: usize, img_h: usize) -> usize {
        let px = img_w / self.patch_size;
        let py = img_h / self.patch_size;
        px * py
    }
}
