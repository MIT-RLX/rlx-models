// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Pixtral / Ministral mmproj vision config from GGUF `clip.*` keys.

use anyhow::{Context, Result, bail};
use rlx_gguf::{GgufFile, MetaValue};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct PixtralVisionConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub layer_norm_eps: f32,
    pub projector_output_dim: usize,
    pub spatial_merge_size: usize,
    pub rope_theta: f32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub use_silu: bool,
}

impl Default for PixtralVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            image_size: 1540,
            patch_size: 14,
            num_channels: 3,
            layer_norm_eps: 1e-5,
            projector_output_dim: 5120,
            spatial_merge_size: 2,
            rope_theta: 10_000.0,
            image_mean: [0.48145467, 0.4578275, 0.40821073],
            image_std: [0.26862955, 0.2613026, 0.2757771],
            use_silu: true,
        }
    }
}

impl PixtralVisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn align_size(&self) -> usize {
        self.patch_size * self.spatial_merge_size.max(1)
    }

    pub fn from_mmproj_gguf(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = GgufFile::header_from_path(path)
            .with_context(|| format!("rlx-mistral-vl: open mmproj header {path:?}"))?;
        Self::from_gguf(&raw)
    }

    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let get = |k: &str| raw.metadata.get(k);
        let u32k =
            |k: &str| -> Option<usize> { get(k).and_then(MetaValue::as_u32).map(|v| v as usize) };
        let f32k = |k: &str| -> Option<f32> {
            get(k).and_then(|v| match v {
                MetaValue::F32(x) => Some(*x),
                _ => None,
            })
        };
        let boolk = |k: &str| -> Option<bool> {
            get(k).and_then(|v| match v {
                MetaValue::Bool(b) => Some(*b),
                _ => None,
            })
        };
        let proj = get("clip.projector_type")
            .and_then(MetaValue::as_str)
            .unwrap_or("pixtral");
        if proj != "pixtral" && proj != "lightonocr" {
            bail!("rlx-mistral-vl: expected clip.projector_type=pixtral, got {proj}");
        }

        let mut cfg = Self::default();
        if let Some(v) = u32k("clip.vision.embedding_length") {
            cfg.hidden_size = v;
        }
        if let Some(v) = u32k("clip.vision.block_count") {
            cfg.num_hidden_layers = v;
        }
        if let Some(v) = u32k("clip.vision.attention.head_count") {
            cfg.num_attention_heads = v;
        }
        if let Some(v) = u32k("clip.vision.feed_forward_length") {
            cfg.intermediate_size = v;
        }
        if let Some(v) = u32k("clip.vision.image_size") {
            cfg.image_size = v;
        }
        if let Some(v) = u32k("clip.vision.patch_size") {
            cfg.patch_size = v;
        }
        if let Some(v) = u32k("clip.vision.projection_dim") {
            cfg.projector_output_dim = v;
        }
        if let Some(v) = u32k("clip.vision.spatial_merge_size") {
            cfg.spatial_merge_size = v.max(1);
        }
        if let Some(v) = f32k("clip.vision.attention.layer_norm_epsilon") {
            cfg.layer_norm_eps = v;
        }
        if let Some(v) = boolk("clip.use_silu") {
            cfg.use_silu = v;
        }
        if let Some(arr) = get("clip.vision.image_mean").and_then(as_f32_array3) {
            cfg.image_mean = arr;
        }
        if let Some(arr) = get("clip.vision.image_std").and_then(as_f32_array3) {
            cfg.image_std = arr;
        }
        Ok(cfg)
    }
}

fn as_f32_array3(v: &MetaValue) -> Option<[f32; 3]> {
    match v {
        MetaValue::Array(items) if items.len() >= 3 => {
            let mut out = [0f32; 3];
            for (i, it) in items.iter().take(3).enumerate() {
                out[i] = match it {
                    MetaValue::F32(x) => *x,
                    MetaValue::F64(x) => *x as f32,
                    _ => return None,
                };
            }
            Some(out)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::PixtralVisionConfig;

    #[test]
    fn config_derived_dims() {
        let c = PixtralVisionConfig::default();
        assert_eq!(c.head_dim(), c.hidden_size / c.num_attention_heads);
        assert_eq!(c.head_dim(), 64);
        // align_size = patch_size * spatial_merge_size (aspect-alignment stride).
        assert_eq!(c.align_size(), c.patch_size * c.spatial_merge_size);
        assert_eq!(c.align_size(), 28);
    }
}
