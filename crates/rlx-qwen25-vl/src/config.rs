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

//! Top-level Qwen2.5-VL configuration — HF JSON + derived MRoPE layout.

use crate::vision::MmProjConfig;
use anyhow::{Context, Result, ensure};
use rlx_qwen3::Qwen3Config;
use serde::Deserialize;
use std::path::Path;

/// LM-side settings plus multimodal RoPE section layout.
#[derive(Debug, Clone)]
pub struct Qwen25VlLmConfig {
    pub lm: Qwen3Config,
    /// Four MRoPE section widths (llama.cpp `ggml_rope_multi`), e.g.
    /// `[16, 24, 24, 0]` for Qwen2.5-VL-7B.
    pub mrope_sections: [usize; 4],
    /// Rotary dim count (`rope_scaling` / GGUF `*.rope.dimension_count`).
    pub rope_dim_count: usize,
}

impl Qwen25VlLmConfig {
    pub fn head_half(&self) -> usize {
        self.lm.head_dim / 2
    }

    pub fn n_rot(&self) -> usize {
        self.rope_dim_count
    }
}

/// Full model config (text + optional vision hyperparameters from HF).
#[derive(Debug, Clone)]
pub struct Qwen25VlConfig {
    pub lm: Qwen25VlLmConfig,
    pub vision: Option<Qwen25VlVisionHfConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen25VlVisionHfConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    #[serde(default)]
    pub depth: usize,
    #[serde(default)]
    pub patch_size: usize,
    #[serde(default)]
    pub spatial_merge_size: usize,
    #[serde(default)]
    pub temporal_patch_size: usize,
    #[serde(default)]
    pub tokens_per_second: f64,
    #[serde(default)]
    pub window_size: usize,
    #[serde(default)]
    pub fullatt_block_indexes: Vec<usize>,
    #[serde(default)]
    pub out_hidden_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub r#type: String,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen25VlHfConfig {
    pub model_type: String,
    pub text_config: Qwen3Config,
    #[serde(default)]
    pub vision_config: Option<Qwen25VlVisionHfConfig>,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
}

impl Qwen25VlHfConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).with_context(|| format!("read {path:?}"))?;
        Ok(serde_json::from_str(&data).with_context(|| format!("parse {path:?}"))?)
    }

    pub fn into_runtime(self) -> Result<Qwen25VlConfig> {
        ensure!(
            crate::ACCEPTED_HF_MODEL_TYPES.contains(&self.model_type.as_str()),
            "unsupported model_type `{}`, expected one of {:?}",
            self.model_type,
            crate::ACCEPTED_HF_MODEL_TYPES
        );
        let mrope_sections = mrope_sections_from_hf(self.rope_scaling.as_ref());
        let rope_dim_count = self.text_config.head_dim;
        Ok(Qwen25VlConfig {
            lm: Qwen25VlLmConfig {
                lm: self.text_config,
                mrope_sections,
                rope_dim_count,
            },
            vision: self.vision_config,
        })
    }
}

/// Parse HF `config.json` into a runtime config bundle.
pub fn config_from_hf_json(path: &Path) -> Result<Qwen25VlConfig> {
    Qwen25VlHfConfig::from_file(path)?.into_runtime()
}

/// Validate an on-disk mmproj GGUF matches Qwen2 / 2.5-VL expectations.
pub fn validate_mmproj_config(cfg: &MmProjConfig) -> Result<()> {
    ensure!(
        crate::ACCEPTED_MMPROJ_TYPES.contains(&cfg.projector_type.as_str()),
        "mmproj projector_type `{}` not supported (expected {:?})",
        cfg.projector_type,
        crate::ACCEPTED_MMPROJ_TYPES
    );
    Ok(())
}

pub fn mrope_sections_from_hf(scaling: Option<&RopeScaling>) -> [usize; 4] {
    let Some(rs) = scaling else {
        return [16, 24, 24, 0];
    };
    if rs.mrope_section.len() >= 3 {
        let a = rs.mrope_section[0];
        let b = rs.mrope_section.get(1).copied().unwrap_or(a);
        let c = rs.mrope_section.get(2).copied().unwrap_or(b);
        return [a, b, c, 0];
    }
    [16, 24, 24, 0]
}

pub fn mrope_sections_from_gguf(raw: &rlx_gguf::GgufFile) -> [usize; 4] {
    use rlx_gguf::MetaValue;
    let arch = raw
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        .unwrap_or("qwen2");
    for key in [
        format!("{arch}.rope.dimension_sections"),
        "qwen2.rope.dimension_sections".to_string(),
        "qwen3.rope.dimension_sections".to_string(),
    ] {
        if let Some(arr) = raw.metadata.get(&key).and_then(|v| match v {
            MetaValue::Array(a) => Some(
                a.iter()
                    .filter_map(|x| x.as_u32().map(|u| u as usize))
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        }) {
            return crate::mrope::mrope_sections4(&arr);
        }
    }
    [16, 24, 24, 0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_mrope_section_maps_to_four_sections() {
        let sec = mrope_sections_from_hf(Some(&RopeScaling {
            r#type: "mrope".into(),
            mrope_section: vec![16, 24, 24],
        }));
        assert_eq!(sec, [16, 24, 24, 0]);
    }
}
