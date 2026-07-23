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

//! Checkpoint tensor names + mmap-backed safetensors access for
//! `baidu/Unlimited-OCR`.
//!
//! Key spelling confirmed against the published
//! `model.safetensors.index.json` (2710 tensors, single shard):
//!
//! ```text
//! lm_head.weight
//! model.embed_tokens.weight
//! model.norm.weight
//! model.image_newline                        # nn.Parameter, no `.weight` suffix
//! model.view_seperator                       # nn.Parameter; HF's spelling (sic), not "separator"
//! model.sam_model.{patch_embed,pos_embed,blocks.N.*,neck.*,net_2,net_3}
//! model.vision_model.{embeddings,pre_layrnorm,transformer.layers.N.*}
//! model.projector.layers.{weight,bias}        # single nn.Linear (projector_type="linear")
//! model.layers.N.{input_layernorm,post_attention_layernorm,self_attn.*}
//! model.layers.N.mlp.{gate_proj,up_proj,down_proj}          # dense layers (< first_k_dense_replace)
//! model.layers.N.mlp.gate.weight                            # MoE router (dense layers have none)
//! model.layers.N.mlp.shared_experts.{gate_proj,up_proj,down_proj}
//! model.layers.N.mlp.experts.E.{gate_proj,up_proj,down_proj}
//! ```

use anyhow::Result;
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use rlx_core::weight_map::WeightMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const PREFIX_SAM_MODEL: &str = "model.sam_model.";
pub const PREFIX_VISION_MODEL: &str = "model.vision_model.";
pub const PREFIX_PROJECTOR: &str = "model.projector.";
pub const PREFIX_LM_LAYERS: &str = "model.layers.";

pub const EMBED_TOKENS: &str = "model.embed_tokens.weight";
pub const LM_NORM: &str = "model.norm.weight";
pub const LM_HEAD: &str = "lm_head.weight";
/// `nn.Parameter` (no weight/bias suffix) — the row-separator embedding
/// appended after each row of the global view's query grid.
pub const IMAGE_NEWLINE: &str = "model.image_newline";
/// `nn.Parameter` (no weight/bias suffix). HF's checkpoint spells this
/// "seperator" (not "separator") — preserved verbatim, it's the literal key.
pub const VIEW_SEPARATOR: &str = "model.view_seperator";

/// HF weight-name helpers for `baidu/Unlimited-OCR`'s SAM + CLIP + projector
/// + DeepSeek-V2-MoE decoder checkpoint.
#[derive(Debug, Clone, Copy)]
pub struct UnlimitedOcrWeightPrefix;

impl UnlimitedOcrWeightPrefix {
    // -- SAM-ViT-B tower (`model.sam_model.*`) --------------------------

    pub fn sam_patch_embed_w() -> &'static str {
        "model.sam_model.patch_embed.proj.weight"
    }
    pub fn sam_patch_embed_b() -> &'static str {
        "model.sam_model.patch_embed.proj.bias"
    }
    pub fn sam_pos_embed() -> &'static str {
        "model.sam_model.pos_embed"
    }
    /// `suffix` ∈ `{attn.qkv, attn.proj}.{weight,bias}`, `attn.rel_pos_{h,w}`,
    /// `{mlp.lin1,mlp.lin2}.{weight,bias}`, `{norm1,norm2}.{weight,bias}`.
    pub fn sam_block(i: usize, suffix: &str) -> String {
        format!("{PREFIX_SAM_MODEL}blocks.{i}.{suffix}")
    }
    /// `idx` ∈ `0..=3`: `neck.0` (Conv2d), `neck.1` (LayerNorm2d), `neck.2`
    /// (Conv2d), `neck.3` (LayerNorm2d). `suffix` ∈ `{weight, bias}` (convs
    /// with `bias=False` have no `.bias` tensor: `neck.0`, `neck.2`).
    pub fn sam_neck(idx: usize, suffix: &str) -> String {
        format!("{PREFIX_SAM_MODEL}neck.{idx}.{suffix}")
    }
    pub fn sam_net2_w() -> &'static str {
        "model.sam_model.net_2.weight"
    }
    pub fn sam_net3_w() -> &'static str {
        "model.sam_model.net_3.weight"
    }

    // -- CLIP-L/14-224 tower (`model.vision_model.*`) -------------------

    pub fn clip_class_embedding() -> &'static str {
        "model.vision_model.embeddings.class_embedding"
    }
    pub fn clip_patch_embedding_w() -> &'static str {
        "model.vision_model.embeddings.patch_embedding.weight"
    }
    pub fn clip_position_embedding_w() -> &'static str {
        "model.vision_model.embeddings.position_embedding.weight"
    }
    pub fn clip_pre_layernorm_w() -> &'static str {
        "model.vision_model.pre_layrnorm.weight"
    }
    pub fn clip_pre_layernorm_b() -> &'static str {
        "model.vision_model.pre_layrnorm.bias"
    }
    /// `suffix` ∈ `{layer_norm1,layer_norm2}.{weight,bias}`,
    /// `mlp.{fc1,fc2}.{weight,bias}`,
    /// `self_attn.{qkv_proj,out_proj}.{weight,bias}`.
    pub fn clip_block(i: usize, suffix: &str) -> String {
        format!("{PREFIX_VISION_MODEL}transformer.layers.{i}.{suffix}")
    }

    // -- Projector (`model.projector.*`) --------------------------------

    pub fn projector_w() -> &'static str {
        "model.projector.layers.weight"
    }
    pub fn projector_b() -> &'static str {
        "model.projector.layers.bias"
    }

    // -- DeepSeek-V2-MoE decoder (`model.layers.*`) ---------------------

    pub fn embed_tokens() -> &'static str {
        EMBED_TOKENS
    }
    pub fn lm_norm() -> &'static str {
        LM_NORM
    }
    pub fn lm_head() -> &'static str {
        LM_HEAD
    }
    pub fn image_newline() -> &'static str {
        IMAGE_NEWLINE
    }
    pub fn view_separator() -> &'static str {
        VIEW_SEPARATOR
    }

    pub fn lm_input_layernorm(i: usize) -> String {
        format!("{PREFIX_LM_LAYERS}{i}.input_layernorm.weight")
    }
    pub fn lm_post_attention_layernorm(i: usize) -> String {
        format!("{PREFIX_LM_LAYERS}{i}.post_attention_layernorm.weight")
    }
    /// `proj` ∈ `{q,k,v,o}_proj`.
    pub fn lm_attn(i: usize, proj: &str) -> String {
        format!("{PREFIX_LM_LAYERS}{i}.self_attn.{proj}.weight")
    }
    /// Dense-layer FFN (`layer_idx < first_k_dense_replace`). `proj` ∈
    /// `{gate,up,down}_proj`.
    pub fn lm_dense_mlp(i: usize, proj: &str) -> String {
        format!("{PREFIX_LM_LAYERS}{i}.mlp.{proj}.weight")
    }
    /// MoE router (`layer_idx >= first_k_dense_replace`).
    pub fn lm_moe_gate(i: usize) -> String {
        format!("{PREFIX_LM_LAYERS}{i}.mlp.gate.weight")
    }
    /// Always-on shared expert(s). `proj` ∈ `{gate,up,down}_proj`.
    pub fn lm_moe_shared_expert(i: usize, proj: &str) -> String {
        format!("{PREFIX_LM_LAYERS}{i}.mlp.shared_experts.{proj}.weight")
    }
    /// Routed expert `e` (0-based, `< n_routed_experts`). `proj` ∈
    /// `{gate,up,down}_proj`.
    pub fn lm_moe_expert(i: usize, e: usize, proj: &str) -> String {
        format!("{PREFIX_LM_LAYERS}{i}.mlp.experts.{e}.{proj}.weight")
    }
}

/// mmap-backed checkpoint handle — lists/loads tensors without a full-RAM snapshot.
///
/// Opens either a sharded `model.safetensors.index.json` or a flat directory
/// of `*.safetensors` files (single-shard checkpoints, like the published
/// `baidu/Unlimited-OCR` release, have no index and are scanned directly).
pub struct UnlimitedOcrWeightStore {
    dir: PathBuf,
    checkpoint: Arc<SafetensorsCheckpoint>,
    all_keys: Arc<HashSet<String>>,
}

impl UnlimitedOcrWeightStore {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let checkpoint = Arc::new(SafetensorsCheckpoint::open(model_dir)?);
        let all_keys = Arc::new(checkpoint.keys().map(str::to_string).collect());
        Ok(Self {
            dir: model_dir.to_path_buf(),
            checkpoint,
            all_keys,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.dir
    }

    pub fn keys(&self) -> &HashSet<String> {
        &self.all_keys
    }

    pub fn contains(&self, key: &str) -> bool {
        self.all_keys.contains(key)
    }

    pub fn count_keys_with_prefix(&self, prefix: &str) -> usize {
        self.all_keys
            .iter()
            .filter(|k| k.starts_with(prefix))
            .count()
    }

    /// Number of routed-expert slots present for decoder layer `i` (0 for
    /// dense layers, `n_routed_experts` for MoE layers).
    pub fn count_experts(&self, layer_idx: usize) -> usize {
        let prefix = format!("{PREFIX_LM_LAYERS}{layer_idx}.mlp.experts.");
        let mut ids = HashSet::new();
        for key in self.all_keys.iter() {
            if let Some(rest) = key.strip_prefix(&prefix) {
                if let Some(e) = rest.split('.').next() {
                    ids.insert(e.to_string());
                }
            }
        }
        ids.len()
    }

    /// Whether decoder layer `i` has a MoE router (i.e. is not dense).
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.contains(&UnlimitedOcrWeightPrefix::lm_moe_gate(layer_idx))
    }

    pub fn load_keys(&self, keys: &[&str]) -> Result<WeightMap> {
        let want: HashSet<String> = keys.iter().map(|k| (*k).to_string()).collect();
        self.checkpoint.load_selected(&want)
    }

    pub fn load_owned_keys(&self, keys: impl IntoIterator<Item = String>) -> Result<WeightMap> {
        let want: HashSet<String> = keys.into_iter().collect();
        self.checkpoint.load_selected(&want)
    }

    pub fn load_prefixes(&self, prefixes: &[&str]) -> Result<WeightMap> {
        let want: HashSet<String> = self
            .all_keys
            .iter()
            .filter(|k| prefixes.iter().any(|p| k.starts_with(p)))
            .cloned()
            .collect();
        self.checkpoint.load_selected(&want)
    }

    /// `model.sam_model.*`.
    pub fn load_sam_tower(&self) -> Result<WeightMap> {
        self.load_prefixes(&[PREFIX_SAM_MODEL])
    }

    /// `model.vision_model.*`.
    pub fn load_clip_tower(&self) -> Result<WeightMap> {
        self.load_prefixes(&[PREFIX_VISION_MODEL])
    }

    /// `model.projector.*`.
    pub fn load_projector(&self) -> Result<WeightMap> {
        self.load_prefixes(&[PREFIX_PROJECTOR])
    }

    /// One decoder layer's tensors (`model.layers.{i}.*`), dense or MoE.
    pub fn load_lm_layer(&self, layer_idx: usize) -> Result<WeightMap> {
        self.load_prefixes(&[format!("{PREFIX_LM_LAYERS}{layer_idx}.").as_str()])
    }

    /// `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`,
    /// `model.image_newline`, `model.view_seperator`.
    pub fn load_lm_globals(&self) -> Result<WeightMap> {
        self.load_keys(&[
            EMBED_TOKENS,
            LM_NORM,
            LM_HEAD,
            IMAGE_NEWLINE,
            VIEW_SEPARATOR,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_helpers_format_real_checkpoint_names() {
        assert_eq!(
            UnlimitedOcrWeightPrefix::sam_block(3, "attn.qkv.weight"),
            "model.sam_model.blocks.3.attn.qkv.weight"
        );
        assert_eq!(
            UnlimitedOcrWeightPrefix::clip_block(0, "self_attn.qkv_proj.weight"),
            "model.vision_model.transformer.layers.0.self_attn.qkv_proj.weight"
        );
        assert_eq!(
            UnlimitedOcrWeightPrefix::lm_attn(0, "q_proj"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            UnlimitedOcrWeightPrefix::lm_dense_mlp(0, "gate_proj"),
            "model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            UnlimitedOcrWeightPrefix::lm_moe_gate(1),
            "model.layers.1.mlp.gate.weight"
        );
        assert_eq!(
            UnlimitedOcrWeightPrefix::lm_moe_expert(1, 63, "down_proj"),
            "model.layers.1.mlp.experts.63.down_proj.weight"
        );
        assert_eq!(
            UnlimitedOcrWeightPrefix::lm_moe_shared_expert(1, "up_proj"),
            "model.layers.1.mlp.shared_experts.up_proj.weight"
        );
        assert_eq!(UnlimitedOcrWeightPrefix::lm_head(), "lm_head.weight");
        assert_eq!(UnlimitedOcrWeightPrefix::embed_tokens(), EMBED_TOKENS);
        assert_eq!(UnlimitedOcrWeightPrefix::image_newline(), IMAGE_NEWLINE);
        assert_eq!(UnlimitedOcrWeightPrefix::view_separator(), VIEW_SEPARATOR);
        assert_eq!(
            UnlimitedOcrWeightPrefix::projector_w(),
            "model.projector.layers.weight"
        );
    }
}
