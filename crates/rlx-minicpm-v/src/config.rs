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

//! `config.json` + `preprocessor_config.json` of a MiniCPM-V 4.6 / 4.7
//! checkpoint (`model_type = minicpmv4_6 | minicpmv4_7`).
//!
//! Only the vision half and the multimodal glue live here; the
//! `text_config` (a Qwen3.5 trunk) is parsed by `rlx-qwen35`.

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use std::path::Path;

/// Model types this crate runs.
pub const MODEL_TYPES: [&str; 2] = ["minicpmv4_6", "minicpmv4_7"];

/// How many patches become one LLM token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownsampleMode {
    /// Window merger (2×2) inside the tower, then the 2×2 merger: 16 patches
    /// per token. The checkpoints' default.
    X16,
    /// Skip the window merger: 4 patches per token.
    X4,
}

impl DownsampleMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "16x" => Ok(Self::X16),
            "4x" => Ok(Self::X4),
            other => bail!("minicpm-v: downsample_mode {other:?} (expected \"16x\" or \"4x\")"),
        }
    }

    /// Patches per LLM token along one axis (`merge_factor` in HF).
    pub fn merge_factor(self) -> usize {
        match self {
            Self::X16 => 4,
            Self::X4 => 2,
        }
    }

    pub fn uses_window_merger(self) -> bool {
        self == Self::X16
    }
}

/// Position layout fed to the trunk's M-RoPE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MropeMode {
    /// 4.6: plain 1-D positions (`t = h = w`).
    Linear,
    /// 4.7: thumbnail and slices laid out on one 2-D canvas.
    Canvas,
}

/// What turns the tower's patch features into LM rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projector {
    /// 4.6 / 4.7: 2×2 concat merger, one LM row per 4 (or 16) patches.
    Merger,
    /// 2.x–4.5: perceiver resampler, `queries` LM rows per crop whatever its
    /// size, cross-attending to the patches with 2-D sin-cos positions.
    Resampler { queries: usize },
}

/// SigLIP-400M tower with the 4.6 window merger.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
    /// Side of the learned position grid (`image_size / patch_size`).
    pub num_patches_per_side: usize,
    pub layer_norm_eps: f32,
    /// Encoder layer after which the window merger runs.
    pub insert_layer_id: usize,
    /// Window side of the window merger (square, HF `window_kernel_size`).
    pub window: usize,
    /// Position ids as if the crop were one row of `h·w` patches: 2.5 calls
    /// transformers' Idefics2 tower with a flattened patch mask, from which
    /// it reads the grid as `1 × n`.
    pub flat_positions: bool,
}

/// Image slicing (`MiniCPMV4_6ImageProcessor`).
#[derive(Debug, Clone, PartialEq)]
pub struct SliceConfig {
    pub max_slice_nums: usize,
    pub scale_resolution: usize,
    pub slice_mode: bool,
    pub use_image_id: bool,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Default for SliceConfig {
    fn default() -> Self {
        Self {
            max_slice_nums: 9,
            scale_resolution: 448,
            slice_mode: true,
            use_image_id: true,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
        }
    }
}

/// Special token ids the canvas layout is anchored on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarkerIds {
    pub image: u32,
    pub image_start: u32,
    pub image_end: u32,
    pub slice_start: u32,
    pub slice_end: u32,
    pub newline: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MiniCpmVConfig {
    pub model_type: String,
    pub vision: VisionConfig,
    pub slice: SliceConfig,
    pub downsample: DownsampleMode,
    pub mrope: MropeMode,
    /// Merge rounds in the final merger (HF `merger_times`).
    pub merger_times: usize,
    /// Merge kernel side of the final merger.
    pub merge_kernel: usize,
    /// LM hidden size (`text_config.hidden_size`), the merger's output width.
    pub text_hidden_size: usize,
    /// `None` on 4.6 checkpoints, which do not list them; the markers are
    /// then looked up in the tokenizer.
    pub markers: Option<MarkerIds>,
    pub image_token_id: u32,
    /// `<|video_pad|>`, the placeholder of video frames; `None` when the
    /// checkpoint does not name one.
    pub video_token_id: Option<u32>,
    pub projector: Projector,
}

impl MiniCpmVConfig {
    /// Read `config.json` (and `preprocessor_config.json` when present) from
    /// a checkpoint directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let cfg_path = dir.join("config.json");
        let v: Value = serde_json::from_str(
            &std::fs::read_to_string(&cfg_path)
                .with_context(|| format!("minicpm-v: read {cfg_path:?}"))?,
        )
        .with_context(|| format!("minicpm-v: parse {cfg_path:?}"))?;
        let pp_path = dir.join("preprocessor_config.json");
        let pp = if pp_path.is_file() {
            Some(serde_json::from_str::<Value>(&std::fs::read_to_string(
                &pp_path,
            )?)?)
        } else {
            None
        };
        Self::from_values(&v, pp.as_ref())
    }

    pub fn from_values(v: &Value, preprocessor: Option<&Value>) -> Result<Self> {
        let model_type = v
            .get("model_type")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("minicpm-v: config.json has no model_type"))?;
        if !MODEL_TYPES.contains(&model_type) {
            bail!(
                "minicpm-v: model_type {model_type:?} is not one of {MODEL_TYPES:?} \
                 (MiniCPM-V 2.x–4.5 use a perceiver resampler and are not this crate)"
            );
        }
        let vc = v
            .get("vision_config")
            .ok_or_else(|| anyhow!("minicpm-v: config.json has no vision_config"))?;
        let tc = v
            .get("text_config")
            .ok_or_else(|| anyhow!("minicpm-v: config.json has no text_config"))?;
        let us = |o: &Value, k: &str| o.get(k).and_then(Value::as_u64).map(|n| n as usize);
        let need = |o: &Value, k: &str| {
            us(o, k).ok_or_else(|| anyhow!("minicpm-v: vision_config.{k} missing"))
        };

        let patch_size = need(vc, "patch_size")?;
        let image_size = need(vc, "image_size")?;
        let window = vc
            .get("window_kernel_size")
            .and_then(Value::as_array)
            .map(|a| {
                let w: Vec<usize> = a
                    .iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as usize)
                    .collect();
                match w.as_slice() {
                    [h, w] if h == w => Ok(*h),
                    _ => Err(anyhow!(
                        "minicpm-v: window_kernel_size {a:?} must be square"
                    )),
                }
            })
            .transpose()?
            .unwrap_or(2);
        let vision = VisionConfig {
            hidden_size: need(vc, "hidden_size")?,
            intermediate_size: need(vc, "intermediate_size")?,
            num_attention_heads: need(vc, "num_attention_heads")?,
            num_hidden_layers: need(vc, "num_hidden_layers")?,
            patch_size,
            num_patches_per_side: image_size / patch_size,
            layer_norm_eps: vc
                .get("layer_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
            // The top-level key wins: HF copies it into the vision config.
            insert_layer_id: us(v, "insert_layer_id")
                .or_else(|| us(vc, "insert_layer_id"))
                .unwrap_or(6),
            window,
            flat_positions: false,
        };
        if vision.insert_layer_id >= vision.num_hidden_layers {
            bail!(
                "minicpm-v: insert_layer_id {} >= {} vision layers",
                vision.insert_layer_id,
                vision.num_hidden_layers
            );
        }

        let mut slice = SliceConfig::default();
        let mut mrope_key = v.get("mrope_mode").and_then(Value::as_str);
        if let Some(pp) = preprocessor {
            if let Some(n) = us(pp, "max_slice_nums") {
                slice.max_slice_nums = n;
            }
            if let Some(n) = us(pp, "scale_resolution") {
                slice.scale_resolution = n;
            }
            if let Some(b) = pp.get("slice_mode").and_then(Value::as_bool) {
                slice.slice_mode = b;
            }
            if let Some(b) = pp.get("use_image_id").and_then(Value::as_bool) {
                slice.use_image_id = b;
            }
            let triple = |k: &str| -> Option<[f32; 3]> {
                let a = pp.get(k)?.as_array()?;
                (a.len() == 3).then(|| [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.0) as f32))
            };
            if let Some(m) = triple("image_mean") {
                slice.image_mean = m;
            }
            if let Some(s) = triple("image_std") {
                slice.image_std = s;
            }
            mrope_key = mrope_key.or_else(|| pp.get("mrope_mode").and_then(Value::as_str));
        }

        let mrope = match mrope_key {
            Some("canvas") => MropeMode::Canvas,
            None if model_type == "minicpmv4_7" => MropeMode::Canvas,
            None => MropeMode::Linear,
            Some(other) => bail!("minicpm-v: unknown mrope_mode {other:?}"),
        };

        let id = |k: &str| v.get(k).and_then(Value::as_u64).map(|n| n as u32);
        let image_token_id =
            id("image_token_id").ok_or_else(|| anyhow!("minicpm-v: image_token_id missing"))?;
        let markers = match (
            id("image_start_id"),
            id("image_end_id"),
            id("slice_start_id"),
            id("slice_end_id"),
            id("newline_id"),
        ) {
            (
                Some(image_start),
                Some(image_end),
                Some(slice_start),
                Some(slice_end),
                Some(newline),
            ) => Some(MarkerIds {
                image: image_token_id,
                image_start,
                image_end,
                slice_start,
                slice_end,
                newline,
            }),
            _ => None,
        };

        let merge_kernel = v
            .get("merge_kernel_size")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(2);
        Ok(Self {
            model_type: model_type.to_string(),
            vision,
            slice,
            downsample: DownsampleMode::parse(
                v.get("downsample_mode")
                    .and_then(Value::as_str)
                    .unwrap_or("16x"),
            )?,
            mrope,
            merger_times: us(v, "merger_times").unwrap_or(1),
            merge_kernel,
            text_hidden_size: us(tc, "hidden_size")
                .ok_or_else(|| anyhow!("minicpm-v: text_config.hidden_size missing"))?,
            markers,
            image_token_id,
            video_token_id: id("video_token_id"),
            projector: Projector::Merger,
        })
    }

    /// `true` when `dir/config.json` names a MiniCPM-V 4.6 / 4.7 checkpoint.
    pub fn is_model_dir(dir: &Path) -> bool {
        std::fs::read_to_string(dir.join("config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .and_then(|v| {
                v.get("model_type")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .is_some_and(|t| MODEL_TYPES.contains(&t.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped MiniCPM-V-4.7-1B `config.json`, minus the text layer list.
    fn v47_1b() -> Value {
        serde_json::json!({
            "architectures": ["MiniCPMV4_7ForConditionalGeneration"],
            "model_type": "minicpmv4_7",
            "text_config": {"hidden_size": 1024, "model_type": "qwen3_5_text"},
            "insert_layer_id": 6,
            "downsample_mode": "16x",
            "mrope_mode": "canvas",
            "vision_config": {
                "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1152, "image_size": 980,
                "intermediate_size": 4304, "layer_norm_eps": 1e-6,
                "model_type": "minicpmv4_7_vision", "num_attention_heads": 16,
                "num_channels": 3, "num_hidden_layers": 27, "patch_size": 14
            },
            "image_token_id": 248056, "video_token_id": 248057,
            "image_start_id": 248078, "image_end_id": 248079,
            "slice_start_id": 248088, "slice_end_id": 248089, "newline_id": 198,
            "version": 4.7
        })
    }

    #[test]
    fn parses_v47_1b() {
        let pp = serde_json::json!({"max_slice_nums": 9, "image_mean": [0.5,0.5,0.5],
            "image_std": [0.5,0.5,0.5], "mrope_mode": "canvas"});
        let c = MiniCpmVConfig::from_values(&v47_1b(), Some(&pp)).unwrap();
        assert_eq!(c.vision.num_patches_per_side, 70);
        assert_eq!(c.vision.insert_layer_id, 6);
        assert_eq!(c.vision.window, 2);
        assert_eq!(c.downsample, DownsampleMode::X16);
        assert_eq!(c.mrope, MropeMode::Canvas);
        assert_eq!(c.text_hidden_size, 1024);
        assert_eq!(c.markers.unwrap().slice_end, 248089);
        assert_eq!(c.slice.scale_resolution, 448);
    }

    #[test]
    fn v46_has_linear_positions() {
        let mut v = v47_1b();
        v["model_type"] = "minicpmv4_6".into();
        let o = v.as_object_mut().unwrap();
        o.remove("mrope_mode");
        for k in [
            "image_start_id",
            "image_end_id",
            "slice_start_id",
            "slice_end_id",
            "newline_id",
        ] {
            o.remove(k);
        }
        let c = MiniCpmVConfig::from_values(&v, None).unwrap();
        assert_eq!(c.mrope, MropeMode::Linear);
        assert!(c.markers.is_none());
    }

    #[test]
    fn rejects_resampler_generation() {
        let mut v = v47_1b();
        v["model_type"] = "minicpmv".into();
        assert!(MiniCpmVConfig::from_values(&v, None).is_err());
    }
}
