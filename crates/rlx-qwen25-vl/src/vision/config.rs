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

//! Qwen2.5-VL mmproj config — parsed from GGUF `clip.*` metadata
//! (`tools/mtmd/models/qwen2vl.cpp`, projector `qwen2.5vl_merger`).

use anyhow::{Result, anyhow};
use rlx_gguf::{GgufFile, MetaValue};

/// Vision / mmproj hyperparameters from a Qwen2.5-VL `mmproj` GGUF.
#[derive(Debug, Clone)]
pub struct MmProjConfig {
    pub patch_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub image_size: usize,
    pub image_min_pixels: usize,
    pub image_max_pixels: usize,
    pub n_merge: usize,
    pub eps: f64,
    pub projector_type: String,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub spatial_merge_size: usize,
    pub llm_hidden_size: usize,
    pub n_ff: usize,
    /// Window-attention pattern length (`clip.vision.n_wa_pattern`, Qwen2.5-VL).
    pub n_wa_pattern: usize,
    /// Merger uses SiLU when true (Qwen2.5-VL); GELU for Qwen2-VL.
    pub use_silu: bool,
    /// RMS norm in ViT when true (Qwen2.5-VL); LayerNorm for Qwen2-VL.
    pub use_rms_norm: bool,
}

impl MmProjConfig {
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
        let boolk = |k: &str| -> bool {
            raw.metadata
                .get(k)
                .and_then(|v| match v {
                    MetaValue::Bool(b) => Some(*b),
                    MetaValue::U32(u) => Some(*u != 0),
                    _ => None,
                })
                .unwrap_or(false)
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

        let patch_size = u32k("clip.vision.patch_size")? as usize;
        let n_embd = u32k("clip.vision.embedding_length")? as usize;
        let n_head = u32k("clip.vision.attention.head_count")? as usize;
        let n_layer = u32k("clip.vision.block_count")? as usize;
        let image_size = u32k("clip.vision.image_size")? as usize;
        let n_merge = u32k_opt("clip.vision.spatial_merge_size")
            .or_else(|| u32k_opt("clip.vision.projector.scale_factor"))
            .unwrap_or(2) as usize;

        let image_min_pixels = u32k_opt("clip.vision.image_min_pixels")
            .map(|v| v as usize)
            .unwrap_or(1024 * n_merge * n_merge * patch_size * patch_size);
        let image_max_pixels = u32k_opt("clip.vision.image_max_pixels")
            .map(|v| v as usize)
            .unwrap_or(4096 * n_merge * n_merge * patch_size * patch_size);

        let projector_type = strk("clip.projector_type")
            .or_else(|_| strk("clip.vision.projector_type"))
            .unwrap_or_else(|_| "qwen2.5vl_merger".to_string());
        // mtmd mmproj uses `clip.use_silu`; older dumps may use `clip.vision.use_silu`.
        let use_silu = boolk("clip.use_silu")
            || boolk("clip.vision.use_silu")
            || projector_type.contains("2.5")
            || projector_type.contains("25");
        let use_rms_norm = projector_type.contains("2.5") || projector_type.contains("25");

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
            projector_type,
            image_mean: arr_f32("clip.vision.image_mean").unwrap_or([0.5, 0.5, 0.5]),
            image_std: arr_f32("clip.vision.image_std").unwrap_or([0.5, 0.5, 0.5]),
            spatial_merge_size: n_merge,
            llm_hidden_size: u32k("clip.vision.projection_dim")? as usize,
            n_ff: u32k("clip.vision.feed_forward_length").unwrap_or(0) as usize,
            n_wa_pattern: u32k_opt("clip.vision.n_wa_pattern").unwrap_or(0) as usize,
            use_silu,
            use_rms_norm,
        })
    }

    pub fn align_size(&self) -> usize {
        self.patch_size * self.n_merge
    }

    pub fn output_grid(&self, img_w: usize, img_h: usize) -> (usize, usize) {
        let gx = (img_w / self.patch_size) / 2;
        let gy = (img_h / self.patch_size) / 2;
        (gx.max(1), gy.max(1))
    }

    pub fn n_out_tokens(&self, img_w: usize, img_h: usize) -> usize {
        let (gx, gy) = self.output_grid(img_w, img_h);
        gx * gy
    }
}
