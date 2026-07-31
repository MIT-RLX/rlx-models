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

//! Configuration for the VLASH π₀ / π₀.₅ Vision-Language-Action policies.
//!
//! Both policies wrap a **PaliGemma** backbone (SigLIP-So400m/14 @224 vision
//! tower + a Gemma-2B text model) and a smaller **Gemma-300M action expert**.
//! The two Gemma stacks run in lock-step through 18 *joint* transformer layers
//! that share one attention over the concatenated `[prefix ++ suffix]`
//! sequence. Actions are produced by flow matching (10 Euler denoise steps).
//!
//! The dimensions below are fixed by the published `lerobot/pi0_base` /
//! `lerobot/pi05_base` checkpoints (see the reference `configuration_pi0.py` /
//! `configuration_pi05.py`).

use serde::{Deserialize, Serialize};

/// LayerNorm epsilon for the SigLIP vision tower.
pub const VISION_LN_EPS: f32 = 1e-6;
/// RMSNorm epsilon for both Gemma stacks.
pub const GEMMA_RMS_EPS: f32 = 1e-6;

/// Which VLASH policy family a config describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VlashVariant {
    /// π₀ — state is a suffix token, time is fused into the action embeddings,
    /// standard Gemma RMSNorm (no conditioning).
    Pi0,
    /// π₀.₅ — action-only suffix; state + time drive adaptive RMSNorm (adaRMS)
    /// in the action-expert stream.
    Pi05,
}

impl VlashVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            VlashVariant::Pi0 => "pi0",
            VlashVariant::Pi05 => "pi05",
        }
    }
}

/// SigLIP vision tower dimensions (PaliGemma So400m/14 @224).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VisionConfig {
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub patch_size: usize,
    pub image_size: usize,
    /// Output width after the `multi_modal_projector.linear` (== Gemma hidden).
    pub projection_dim: usize,
    pub ln_eps: f32,
}

impl VisionConfig {
    /// SigLIP-So400m/14 @224 as used by `google/paligemma-3b-pt-224`.
    pub fn so400m_patch14_224() -> Self {
        VisionConfig {
            width: 1152,
            layers: 27,
            heads: 16,
            head_dim: 72,
            intermediate: 4304,
            patch_size: 14,
            image_size: 224,
            projection_dim: 2048,
            ln_eps: VISION_LN_EPS,
        }
    }

    /// Patches per side (`image_size / patch_size`).
    pub fn n_side(&self) -> usize {
        self.image_size / self.patch_size
    }
    /// Total patches (no CLS token in SigLIP).
    pub fn num_patches(&self) -> usize {
        self.n_side() * self.n_side()
    }
    /// Unfolded patch vector length `3 · ps · ps`.
    pub fn patch_dim(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }
}

/// One Gemma-v1 transformer stack (the VLM text model or the action expert).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GemmaConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    pub intermediate: usize,
    pub rope_theta: f64,
    pub rms_eps: f32,
    /// Whether this stack uses adaptive RMSNorm (π₀.₅ action expert only).
    pub use_adarms: bool,
}

impl GemmaConfig {
    /// The Gemma-2B VLM text backbone inside PaliGemma.
    pub fn vlm_2b() -> Self {
        GemmaConfig {
            hidden: 2048,
            layers: 18,
            heads: 8,
            head_dim: 256,
            num_kv_heads: 1,
            intermediate: 16_384,
            rope_theta: 10_000.0,
            rms_eps: GEMMA_RMS_EPS,
            use_adarms: false,
        }
    }

    /// The Gemma-300M action expert.
    pub fn expert_300m(use_adarms: bool) -> Self {
        GemmaConfig {
            hidden: 1024,
            layers: 18,
            heads: 8,
            head_dim: 256,
            num_kv_heads: 1,
            intermediate: 4096,
            rope_theta: 10_000.0,
            rms_eps: GEMMA_RMS_EPS,
            use_adarms,
        }
    }

    /// GQA replication factor (`heads / num_kv_heads`).
    pub fn kv_group(&self) -> usize {
        self.heads / self.num_kv_heads
    }
    /// Full attention projection width (`heads · head_dim`).
    pub fn attn_dim(&self) -> usize {
        self.heads * self.head_dim
    }
    /// K/V projection width (`num_kv_heads · head_dim`).
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
    /// Attention score scale = `head_dim^-0.5` (Gemma v1; no query_pre_attn_scalar).
    pub fn score_scale(&self) -> f32 {
        (self.head_dim as f32).powf(-0.5)
    }
}

/// Full VLASH policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VlashConfig {
    pub variant: VlashVariant,
    pub vision: VisionConfig,
    /// The VLM (PaliGemma Gemma-2B) backbone stack.
    pub vlm: GemmaConfig,
    /// The action-expert (Gemma-300M) stack.
    pub expert: GemmaConfig,

    // --- action prediction ---
    pub max_state_dim: usize,
    pub max_action_dim: usize,
    pub chunk_size: usize,
    pub n_action_steps: usize,

    // --- flow matching ---
    pub num_inference_steps: usize,
    pub min_period: f32,
    pub max_period: f32,

    // --- tokenization / images ---
    pub tokenizer_max_length: usize,
    pub image_size: usize,

    /// π₀.₅ only: fold robot state into the adaRMS conditioning signal.
    pub state_cond: bool,
}

impl VlashConfig {
    /// Default π₀ configuration (matches `lerobot/pi0_base`).
    pub fn pi0() -> Self {
        VlashConfig {
            variant: VlashVariant::Pi0,
            vision: VisionConfig::so400m_patch14_224(),
            vlm: GemmaConfig::vlm_2b(),
            expert: GemmaConfig::expert_300m(false),
            max_state_dim: 32,
            max_action_dim: 32,
            chunk_size: 50,
            n_action_steps: 50,
            num_inference_steps: 10,
            min_period: 4e-3,
            max_period: 4.0,
            tokenizer_max_length: 200,
            image_size: 224,
            state_cond: false,
        }
    }

    /// Default π₀.₅ configuration (matches `lerobot/pi05_base`, `state_cond=true`).
    pub fn pi05() -> Self {
        VlashConfig {
            variant: VlashVariant::Pi05,
            vision: VisionConfig::so400m_patch14_224(),
            vlm: GemmaConfig::vlm_2b(),
            expert: GemmaConfig::expert_300m(true),
            max_state_dim: 32,
            max_action_dim: 32,
            chunk_size: 50,
            n_action_steps: 50,
            num_inference_steps: 10,
            min_period: 4e-3,
            max_period: 4.0,
            tokenizer_max_length: 200,
            image_size: 224,
            state_cond: true,
        }
    }

    /// Config for `variant` with policy defaults.
    pub fn for_variant(variant: VlashVariant) -> Self {
        match variant {
            VlashVariant::Pi0 => Self::pi0(),
            VlashVariant::Pi05 => Self::pi05(),
        }
    }

    /// The Gemma hidden dimension shared by the prefix (== VLM hidden).
    pub fn prefix_width(&self) -> usize {
        self.vlm.hidden
    }
    /// The action-expert hidden dimension (suffix width).
    pub fn suffix_width(&self) -> usize {
        self.expert.hidden
    }
    /// The shared attention head dimension (identical for both stacks).
    pub fn head_dim(&self) -> usize {
        self.vlm.head_dim
    }
    /// Number of attention heads (identical for both stacks).
    pub fn heads(&self) -> usize {
        self.vlm.heads
    }
    /// π₀ suffix length (state token + action tokens); π₀.₅ = action tokens only.
    pub fn suffix_len(&self) -> usize {
        match self.variant {
            VlashVariant::Pi0 => 1 + self.chunk_size,
            VlashVariant::Pi05 => self.chunk_size,
        }
    }
}
