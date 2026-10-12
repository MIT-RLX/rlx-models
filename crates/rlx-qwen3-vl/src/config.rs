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

//! Qwen3-VL vision-tower configuration.
//!
//! Maps to the `vision_config` block of the HuggingFace
//! `Qwen3VLConfig` and the corresponding `vision.*` keys in the
//! Qwen3-VL mmproj GGUF.

use anyhow::{Result, anyhow, bail};
use serde::Deserialize;
use std::path::Path;

/// SigLIP-variant vision-tower config used by Qwen3-VL.
#[derive(Debug, Clone)]
pub struct Qwen3VlVisionConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub layer_norm_eps: f64,
    /// Output dim of the multimodal projector (matches the LM's hidden
    /// size — i.e. `Qwen3Config::hidden_size`).
    pub projector_output_dim: usize,
    /// Side of the patch block the projector merges into one LM embedding
    /// (Qwen3-VL: 2, i.e. 2x2 patches per token). `1` is no merging.
    ///
    /// Read rather than assumed because getting it wrong is silent: the
    /// tower still produces embeddings, there are just `merge^2` times too
    /// many of them, and the image/text splice then puts the text in the
    /// wrong place.
    pub spatial_merge_size: usize,
    /// Frames per temporal patch (Qwen3-VL: 2). Only meaningful for video.
    pub temporal_patch_size: usize,
    /// Rows in the tower's learned position-embedding table, when it has
    /// one. Qwen3-VL interpolates this table to the actual grid.
    pub num_position_embeddings: Option<usize>,
    /// Layers whose intermediate output is also fed to the LM ("deepstack").
    ///
    /// Qwen3-VL states `[8, 16, 24]`: those three layers' features are
    /// injected into the language model in addition to the final ones. A
    /// tower that ignores them computes a different function.
    pub deepstack_visual_indexes: Vec<usize>,
}

impl Qwen3VlVisionConfig {
    pub fn num_patches(&self) -> usize {
        let p = self.image_size / self.patch_size;
        p * p
    }
    pub fn patch_dim(&self) -> usize {
        self.num_channels * self.patch_size * self.patch_size
    }
    pub fn seq_len(&self) -> usize {
        // SigLIP has no CLS — `seq == num_patches`.
        self.num_patches()
    }
}

/// The `vision_config` block, under either spelling.
///
/// Qwen2.5-VL and the earlier exports use the transformers-wide names
/// (`num_hidden_layers`, `num_attention_heads`, `num_channels`). The Qwen3-VL
/// release renamed them (`depth`, `num_heads`, `in_channels`) and dropped
/// `image_size` entirely — the tower takes any resolution and derives its grid
/// from the patch count. Aliases rather than a second struct, because the
/// fields mean the same thing and a model that fails to load over a renamed
/// key is the least useful way to find that out.
#[derive(Debug, Deserialize)]
struct HfVisionConfig {
    hidden_size: usize,
    #[serde(alias = "depth")]
    num_hidden_layers: usize,
    #[serde(alias = "num_heads")]
    num_attention_heads: usize,
    intermediate_size: usize,
    /// Absent on Qwen3-VL, whose tower is resolution-agnostic. The default is
    /// the preprocessor's own, and only sizes the fixed grid helpers.
    #[serde(default = "default_image_size")]
    image_size: usize,
    patch_size: usize,
    #[serde(default = "default_channels", alias = "in_channels")]
    num_channels: usize,
    #[serde(default = "default_eps")]
    layer_norm_eps: f64,
    /// Qwen3-VL states the projector's output here; older exports leave it to
    /// be taken from the language model's hidden size.
    #[serde(default)]
    out_hidden_size: Option<usize>,
    #[serde(default = "default_merge")]
    spatial_merge_size: usize,
    #[serde(default = "default_merge")]
    temporal_patch_size: usize,
    #[serde(default)]
    num_position_embeddings: Option<usize>,
    #[serde(default)]
    deepstack_visual_indexes: Vec<usize>,
}

fn default_channels() -> usize {
    3
}
fn default_eps() -> f64 {
    1e-6
}
fn default_merge() -> usize {
    // Pre-Qwen3-VL exports state neither and merged nothing.
    1
}
fn default_image_size() -> usize {
    // 384 is SigLIP's, and what every earlier export states explicitly.
    384
}

#[derive(Debug, Deserialize)]
struct HfTopConfig {
    vision_config: HfVisionConfig,
    text_config: HfTextConfig,
}

#[derive(Debug, Deserialize)]
struct HfTextConfig {
    hidden_size: usize,
}

impl Qwen3VlVisionConfig {
    /// Read from a HuggingFace `config.json` (the top-level Qwen3-VL
    /// model config with embedded `vision_config` + `text_config`).
    pub fn from_hf_config_json(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("rlx-qwen3-vl: read {path:?}: {e}"))?;
        let cfg: HfTopConfig =
            serde_json::from_str(&raw).map_err(|e| anyhow!("rlx-qwen3-vl: parse {path:?}: {e}"))?;
        Ok(Self {
            hidden_size: cfg.vision_config.hidden_size,
            num_hidden_layers: cfg.vision_config.num_hidden_layers,
            num_attention_heads: cfg.vision_config.num_attention_heads,
            intermediate_size: cfg.vision_config.intermediate_size,
            image_size: cfg.vision_config.image_size,
            patch_size: cfg.vision_config.patch_size,
            num_channels: cfg.vision_config.num_channels,
            layer_norm_eps: cfg.vision_config.layer_norm_eps,
            // The export's own figure when it states one. It has to match the
            // language model's hidden size for the splice to work, and a
            // mismatch is caught at prefill with both numbers named.
            projector_output_dim: cfg
                .vision_config
                .out_hidden_size
                .unwrap_or(cfg.text_config.hidden_size),
            spatial_merge_size: cfg.vision_config.spatial_merge_size,
            temporal_patch_size: cfg.vision_config.temporal_patch_size,
            num_position_embeddings: cfg.vision_config.num_position_embeddings,
            deepstack_visual_indexes: cfg.vision_config.deepstack_visual_indexes,
        })
    }

    /// Refuse a checkpoint whose vision tower this crate would compute
    /// incorrectly rather than not at all.
    ///
    /// Both of these are silent when unchecked, which is the point. A
    /// Qwen3-VL export states `spatial_merge_size: 2` and
    /// `deepstack_visual_indexes: [8, 16, 24]`; this crate's graph
    /// ([`crate::build_qwen3_vl_vision`]) implements neither — it emits one
    /// embedding per patch from the final layer only. Run it anyway and the
    /// tower produces `merge^2` times too many embeddings, drawn from the
    /// wrong features, and the model answers confidently about an image it
    /// effectively did not see. `rlx-qwen4exp`'s reader already refuses a
    /// deepstack mmproj for the same reason.
    ///
    /// The implementation to port from is `rlx-qwen35`'s `vision/` module,
    /// which has the real Qwen3-VL tower (merge-block permute before the
    /// transformer, deepstack taps).
    pub fn validate_supported(&self) -> Result<()> {
        if self.spatial_merge_size > 1 {
            bail!(
                "rlx-qwen3-vl: this checkpoint merges {m}x{m} patches per LM token, which \
                 this crate's vision graph does not implement (it emits one embedding per \
                 patch). Running it would hand the language model {n}x too many vision \
                 tokens. Port the tower from rlx-qwen35's `vision` module.",
                m = self.spatial_merge_size,
                n = self.spatial_merge_size * self.spatial_merge_size
            );
        }
        if !self.deepstack_visual_indexes.is_empty() {
            bail!(
                "rlx-qwen3-vl: this checkpoint feeds layers {:?} to the language model in \
                 addition to the final ones (deepstack), which this crate's vision graph \
                 does not implement. Port the tower from rlx-qwen35's `vision` module.",
                self.deepstack_visual_indexes
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests run in parallel in one process, so the filename has to be unique
    /// per call or two of them read each other's config.
    fn write(json: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "rlx-vl-cfg-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, json).unwrap();
        path
    }

    /// The Qwen3-VL release renamed half of these. A model that fails to load
    /// over a renamed key is the least useful way to discover that.
    #[test]
    fn the_qwen3_vl_spelling_of_the_vision_block_is_read() {
        let path = write(
            r#"{
                "text_config": { "hidden_size": 2048 },
                "vision_config": {
                    "hidden_size": 1152,
                    "depth": 27,
                    "num_heads": 16,
                    "intermediate_size": 4304,
                    "patch_size": 16,
                    "in_channels": 3,
                    "out_hidden_size": 2048
                }
            }"#,
        );
        let cfg = Qwen3VlVisionConfig::from_hf_config_json(&path).expect("parses");
        assert_eq!(cfg.num_hidden_layers, 27);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_channels, 3);
        assert_eq!(cfg.projector_output_dim, 2048);
        // Absent from the Qwen3-VL export, whose tower is resolution-agnostic.
        assert_eq!(cfg.image_size, 384);
    }

    /// And the older spelling still parses exactly as it did.
    #[test]
    fn the_earlier_spelling_still_parses() {
        let path = write(
            r#"{
                "text_config": { "hidden_size": 1536 },
                "vision_config": {
                    "hidden_size": 1152,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "intermediate_size": 4304,
                    "image_size": 384,
                    "patch_size": 14
                }
            }"#,
        );
        let cfg = Qwen3VlVisionConfig::from_hf_config_json(&path).expect("parses");
        assert_eq!(cfg.num_hidden_layers, 27);
        assert_eq!(cfg.image_size, 384);
        assert_eq!(cfg.patch_size, 14);
        // No `out_hidden_size`, so the language model's hidden size stands in.
        assert_eq!(cfg.projector_output_dim, 1536);
    }
}
