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
//! (llama.cpp `tools/mtmd/clip-impl.h`).

use anyhow::{Result, anyhow};
use rlx_gguf::{GgufFile, MetaValue};

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

        Ok(Self {
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
        })
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
