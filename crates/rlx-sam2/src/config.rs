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

//! SAM 2 model configuration. Mirrors Meta's `segment-anything-2` (a.k.a.
//! `facebookresearch/sam2`) reference exactly, so the published
//! `sam2_hiera_{t,s,b+,l}.pt` checkpoints can load without remapping.
//!
//! The image encoder is `Hiera` (Ryali et al. 2023) — a hierarchical,
//! multi-scale ViT with mask-unit attention and Q-pooling — wrapped by
//! an FPN-style neck that emits feature maps at strides 4/8/16/32. The
//! prompt encoder + mask decoder are similar in spirit to SAM v1 but
//! add an object-pointer token, an object-score head, and a high-res
//! mask path that consumes the FPN's stride-4 / stride-8 features.
//!
//! Phase split (mirrors `crate::sam`):
//!   - **Phase 1 (this commit)**: Hiera image encoder graph + FpnNeck.
//!   - Phase 2: prompt encoder + mask decoder + IoU/object-score heads.
//!   - Phase 3: memory attention + memory encoder for video tracking.

use serde::Deserialize;

/// SAM 2 normalises pixels with ImageNet stats *after* /255 scaling
/// (note: this differs from SAM v1, which normalises raw 0..255
/// pixel values directly). Matches `sam2/utils/transforms.py`.
pub const SAM2_PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const SAM2_PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Target image side after preprocessing. SAM 2 always operates at
/// 1024×1024 internally (same as SAM v1).
pub const SAM2_IMG_SIZE: usize = 1024;

/// Hiera patch embedding parameters — Conv2d(in=3, out=embed_dim,
/// k=7, s=4, p=3). The /4 stride gives a 256×256 grid at stage 0.
pub const SAM2_PATCH_KERNEL: usize = 7;
pub const SAM2_PATCH_STRIDE: usize = 4;
pub const SAM2_PATCH_PADDING: usize = 3;

/// Spatial resolution emitted by the patch embedding (stage 0 input).
pub const SAM2_PATCH_GRID: usize = SAM2_IMG_SIZE / SAM2_PATCH_STRIDE; // 256

/// Number of Q-pooling stages — fixed at 3 in the reference for every
/// Hiera variant. After each Q-pool the spatial sequence is downsampled
/// 2× along each spatial axis (4× area reduction) and the channel
/// dimension + head count each double.
pub const SAM2_Q_POOL_COUNT: usize = 3;
pub const SAM2_Q_STRIDE: usize = 2;

/// Channel count of the embeddings emitted by the FPN neck and consumed
/// by the prompt encoder + mask decoder.
pub const SAM2_PROMPT_EMBED_DIM: usize = 256;

/// Hiera image-encoder configuration — Tiny, Small, Base+ or Large.
///
/// Field names mirror Hiera's Python kwargs so the values map 1:1 to
/// the published checkpoints.
#[derive(Debug, Clone, Deserialize)]
pub struct Sam2HieraConfig {
    /// Stage-0 embedding dimension. Doubles after each Q-pool.
    pub embed_dim: usize,
    /// Stage-0 head count. Doubles after each Q-pool.
    pub num_heads: usize,
    /// Number of blocks per stage. `stages.len() == 4` for every
    /// published Hiera variant (Tiny/Small/Base+/Large).
    pub stages: Vec<usize>,
    /// Indices (in the *flattened* block enumeration across stages) of
    /// blocks that use *global* attention rather than mask-unit
    /// (windowed) attention. Always exactly 3 entries in published
    /// configs.
    pub global_att_blocks: Vec<usize>,
    /// Background-pos-embed spatial size: `[Ph, Pw]` for the
    /// learned-pos table that gets bilinear-interpolated to the
    /// current grid each forward.
    pub window_pos_embed_bkg_spatial_size: [usize; 2],
    /// Mask-unit window size per stage (in units of post-Q-pool tokens
    /// at that stage). Always 4 entries, one per stage.
    pub window_spec: [usize; 4],
    /// LayerNorm eps used throughout the encoder.
    pub layer_norm_eps: f64,
    /// MLP expansion ratio (FFN hidden = `mlp_ratio · embed_dim`).
    pub mlp_ratio: f64,
    /// QKV linear bias toggle (always true in published configs).
    pub qkv_bias: bool,
    /// Output channels per FPN level (256 for every published config).
    pub fpn_out_chans: usize,
}

impl Sam2HieraConfig {
    /// `sam2_hiera_tiny` — ~30 M params.
    pub fn tiny() -> Self {
        Self {
            embed_dim: 96,
            num_heads: 1,
            stages: vec![1, 2, 7, 2],
            global_att_blocks: vec![5, 7, 9],
            window_pos_embed_bkg_spatial_size: [7, 7],
            window_spec: [8, 4, 14, 7],
            layer_norm_eps: 1e-6,
            mlp_ratio: 4.0,
            qkv_bias: true,
            fpn_out_chans: SAM2_PROMPT_EMBED_DIM,
        }
    }
    /// `sam2_hiera_small` — ~46 M params.
    pub fn small() -> Self {
        Self {
            stages: vec![1, 2, 11, 2],
            global_att_blocks: vec![7, 10, 13],
            ..Self::tiny()
        }
    }
    /// `sam2_hiera_base_plus` — ~80 M params, the recommended default.
    pub fn base_plus() -> Self {
        Self {
            embed_dim: 112,
            num_heads: 2,
            stages: vec![2, 3, 16, 3],
            global_att_blocks: vec![12, 16, 20],
            window_pos_embed_bkg_spatial_size: [14, 14],
            window_spec: [8, 4, 14, 7],
            layer_norm_eps: 1e-6,
            mlp_ratio: 4.0,
            qkv_bias: true,
            fpn_out_chans: SAM2_PROMPT_EMBED_DIM,
        }
    }
    /// `sam2_hiera_large` — ~224 M params. The YAML overrides
    /// `window_pos_embed_bkg_spatial_size` to `[7, 7]` (vs `[14, 14]`
    /// for base+).
    pub fn large() -> Self {
        Self {
            embed_dim: 144,
            num_heads: 2,
            stages: vec![2, 6, 36, 4],
            global_att_blocks: vec![23, 33, 43],
            window_pos_embed_bkg_spatial_size: [7, 7],
            window_spec: [8, 4, 16, 8],
            ..Self::base_plus()
        }
    }

    /// Total number of transformer blocks across all stages.
    pub fn total_blocks(&self) -> usize {
        self.stages.iter().sum()
    }

    /// Indices (in the flattened block enumeration) where a Q-pool
    /// happens — at the *first* block of every stage after stage 0.
    ///
    /// Reference: in `Hiera.__init__` the Q-pool boundaries are
    /// `cumulative_sum(stages)[:-1]`, i.e. `[s0, s0+s1, s0+s1+s2]`.
    pub fn q_pool_block_indices(&self) -> Vec<usize> {
        let mut acc = 0usize;
        let mut out = Vec::with_capacity(SAM2_Q_POOL_COUNT);
        for &n in &self.stages[..self.stages.len() - 1] {
            acc += n;
            out.push(acc);
        }
        out
    }

    /// Stage index for the i-th flattened block.
    pub fn stage_of_block(&self, block_idx: usize) -> usize {
        let mut acc = 0usize;
        for (si, &n) in self.stages.iter().enumerate() {
            acc += n;
            if block_idx < acc {
                return si;
            }
        }
        self.stages.len() - 1
    }

    /// Embedding dimension at stage `s` (doubles per Q-pool).
    pub fn embed_dim_at_stage(&self, s: usize) -> usize {
        self.embed_dim * (1 << s)
    }
    /// Number of heads at stage `s` (doubles per Q-pool).
    pub fn num_heads_at_stage(&self, s: usize) -> usize {
        self.num_heads * (1 << s)
    }
    /// Mask-unit window size at stage `s`.
    pub fn window_size_at_stage(&self, s: usize) -> usize {
        self.window_spec[s]
    }
    /// Per-axis spatial size of the token grid at stage `s` (before any
    /// Q-pool *inside* the stage — i.e. the size at the stage's first
    /// post-Q-pool block, for s>0, or stage-0 patch grid for s=0).
    pub fn grid_size_at_stage(&self, s: usize) -> usize {
        SAM2_PATCH_GRID / (1 << s)
    }
}

/// FPN neck configuration. Mirrors `FpnNeck` in the reference.
///
/// SAM 2's neck takes the per-stage outputs from Hiera (finest →
/// coarsest, i.e. stage 0..3) and runs a top-down pyramid:
///   1. Each level gets a 1×1 lateral conv to `d_model=256`.
///   2. Going coarse → fine, levels listed in `fpn_top_down_levels`
///      *also* receive a nearest-neighbour ×2 upsample of the next-
///      coarser level summed in.
///
/// The published `_b+` / `_l` configs use `fpn_top_down_levels=[2, 3]`,
/// meaning only the two coarsest levels actually fuse with their
/// neighbours; the two finest levels are emitted as plain laterals.
///
/// Note on indexing: `backbone_channel_list` and `fpn_top_down_levels`
/// are stored **coarse-to-fine** (i.e. `[stage3_dim, …, stage0_dim]`)
/// to match the reference YAML and let conv weight keys
/// (`image_encoder.neck.convs.{n-i}…`) line up 1:1 with the
/// checkpoint.
#[derive(Debug, Clone)]
pub struct Sam2FpnConfig {
    pub d_model: usize,
    /// Per-stage Hiera output channels, **coarse → fine** order:
    /// `[stage3_dim, stage2_dim, stage1_dim, stage0_dim]`.
    pub backbone_channel_list: Vec<usize>,
    /// Backbone stage indices (in the same coarse-to-fine ordering as
    /// `backbone_channel_list`) that participate in the top-down sum.
    /// `[2, 3]` in every published config, i.e. only the two coarsest
    /// levels (note: indices into the *reversed* level enumeration the
    /// reference uses — kept as-is here for checkpoint compatibility).
    pub fpn_top_down_levels: Vec<usize>,
    /// Interpolation mode for the top-down upsample. Reference uses
    /// `"nearest"`, which we lower as cheap host-side replicate.
    pub interpolation_nearest: bool,
}

impl Sam2FpnConfig {
    pub fn for_hiera(cfg: &Sam2HieraConfig) -> Self {
        // Coarsest stage first → finest last, matching the reference's
        // YAML ordering and the `image_encoder.neck.convs.{i}` key layout.
        let channels: Vec<usize> = (0..cfg.stages.len())
            .rev()
            .map(|s| cfg.embed_dim_at_stage(s))
            .collect();
        // Preflight: B+ should give [896, 448, 224, 112].
        debug_assert!(
            channels.first().copied().unwrap_or(0) >= channels.last().copied().unwrap_or(0),
            "backbone_channel_list must be coarse → fine"
        );
        Self {
            d_model: cfg.fpn_out_chans,
            backbone_channel_list: channels,
            fpn_top_down_levels: vec![2, 3],
            interpolation_nearest: true,
        }
    }
}

/// Mask decoder configuration. Field names + defaults mirror
/// `sam2/modeling/sam/mask_decoder.py::MaskDecoder.__init__` and the
/// published `sam2_hiera_*.yaml` `model.sam_mask_decoder_extra_args`.
#[derive(Debug, Clone)]
pub struct Sam2DecoderConfig {
    pub transformer_dim: usize,
    pub transformer_depth: usize,
    pub transformer_num_heads: usize,
    pub transformer_mlp_dim: usize,
    /// 4 = 1 best-mask token + 3 multimask tokens (`num_multimask_outputs=3`
    /// in the YAML; total tokens = `num_multimask_outputs + 1`).
    pub num_mask_tokens: usize,
    pub iou_head_depth: usize,
    pub iou_head_hidden_dim: usize,
    /// `iou_prediction_use_sigmoid` flag (true in the published YAML).
    pub iou_prediction_use_sigmoid: bool,
    /// SAM 2 emits an additional object-pointer token. Always true for
    /// the published video configs.
    pub use_object_pointer: bool,
    /// If true, `obj_ptr_proj` is a 3-layer MLP; else a plain Linear.
    /// True in every published config.
    pub use_mlp_for_obj_ptr_proj: bool,
    /// Predict an object-score logit (whether an object is present).
    /// True in every published config.
    pub pred_obj_scores: bool,
    /// If true, `pred_obj_score_head` is a 3-layer MLP; else a Linear.
    /// True in every published config.
    pub pred_obj_scores_mlp: bool,
    /// When multimask is selected, use the three multimask tokens
    /// (rather than the best-mask token) for the object pointer.
    /// True in every published config.
    pub use_multimask_token_for_obj_ptr: bool,
    /// Use the FpnNeck's stride-4 + stride-8 features to refine the
    /// upscaled mask. True in every published config.
    pub use_high_res_features: bool,
    /// Fall back to the best multimask output when the single-mask
    /// token's stability score is below threshold. True in every
    /// published video config (`dynamic_multimask_via_stability=True`).
    pub dynamic_multimask_via_stability: bool,
    pub dynamic_multimask_stability_delta: f32,
    pub dynamic_multimask_stability_thresh: f32,
    pub layer_norm_eps: f64,
}

impl Default for Sam2DecoderConfig {
    fn default() -> Self {
        Self {
            transformer_dim: SAM2_PROMPT_EMBED_DIM,
            transformer_depth: 2,
            transformer_num_heads: 8,
            transformer_mlp_dim: 2048,
            num_mask_tokens: 4,
            iou_head_depth: 3,
            iou_head_hidden_dim: SAM2_PROMPT_EMBED_DIM,
            iou_prediction_use_sigmoid: true,
            use_object_pointer: true,
            use_mlp_for_obj_ptr_proj: true,
            pred_obj_scores: true,
            pred_obj_scores_mlp: true,
            use_multimask_token_for_obj_ptr: true,
            use_high_res_features: true,
            dynamic_multimask_via_stability: true,
            dynamic_multimask_stability_delta: 0.05,
            dynamic_multimask_stability_thresh: 0.98,
            layer_norm_eps: 1e-6,
        }
    }
}

/// Memory-encoder configuration. Mirrors
/// `sam2/modeling/memory_encoder.py::MemoryEncoder` + its
/// `MaskDownSampler` and `Fuser`. Defaults match every published
/// `sam2_hiera_*.yaml` `memory_encoder:` block.
#[derive(Debug, Clone)]
pub struct Sam2MemoryEncoderConfig {
    /// Input feature dim from the FpnNeck stride-16 level.
    pub in_dim: usize,
    /// Output memory token dim (smaller than `in_dim` in published
    /// configs: 64 vs 256, so memory bank tokens are cheap to store).
    pub out_dim: usize,
    /// MaskDownSampler: per-step kernel/stride/padding + total stride.
    /// total_stride must be a power of `stride`; reference uses
    /// kernel=3 stride=2 padding=1 total=16 → 4 down-sampling levels.
    pub mask_downsampler_kernel: usize,
    pub mask_downsampler_stride: usize,
    pub mask_downsampler_padding: usize,
    pub mask_downsampler_total_stride: usize,
    /// Fuser: `num_layers` × CXBlock(dim, kernel, padding, ls_init).
    pub fuser_num_layers: usize,
    pub fuser_dim: usize,
    pub fuser_kernel: usize,
    pub fuser_padding: usize,
    pub fuser_layer_scale_init_value: f32,
    pub fuser_use_dwconv: bool,
    pub fuser_input_projection: bool,
    /// Memory-encoder output PE: `num_pos_feats * 2` is the channel
    /// count of the emitted position encoding.
    pub pe_num_pos_feats: usize,
    pub pe_temperature: f32,
}

impl Default for Sam2MemoryEncoderConfig {
    fn default() -> Self {
        Self {
            in_dim: SAM2_PROMPT_EMBED_DIM,
            out_dim: 64,
            mask_downsampler_kernel: 3,
            mask_downsampler_stride: 2,
            mask_downsampler_padding: 1,
            mask_downsampler_total_stride: 16,
            fuser_num_layers: 2,
            fuser_dim: SAM2_PROMPT_EMBED_DIM,
            fuser_kernel: 7,
            fuser_padding: 3,
            fuser_layer_scale_init_value: 1e-6,
            fuser_use_dwconv: true,
            fuser_input_projection: false,
            // num_pos_feats * 2 == out_dim so PE shape matches memory
            // tokens for `memory + pos` addition in MemoryAttention.
            pe_num_pos_feats: 32,
            pe_temperature: 10000.0,
        }
    }
}

/// Memory-attention configuration (video path).
///
/// `d_model` (256) is the working dim of memory-attention layers;
/// memory bank tokens come in at `kv_in_dim=out_dim` of the memory
/// encoder (64 in published configs) and are projected up by the
/// cross-attention's k/v linear layers.
#[derive(Debug, Clone)]
pub struct Sam2MemoryConfig {
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub dim_feedforward: usize,
    pub layer_norm_eps: f64,
    /// Memory bank kv channel dim — matches memory-encoder out_dim.
    pub kv_in_dim: usize,
    pub rope_theta: f32,
    /// Spatial size used to build the RoPE table for the image
    /// features (32×32 for the stride-32 path in published configs).
    pub rope_feat_size: [usize; 2],
    pub rope_k_repeat: bool,
    /// Whether to add the input PE to the queries at layer input
    /// (`pos_enc_at_input` in the reference YAML; true).
    pub pos_enc_at_input: bool,
    pub pos_enc_at_attn: bool,
    pub pos_enc_at_cross_attn_keys: bool,
    pub pos_enc_at_cross_attn_queries: bool,
    /// Maximum number of object pointers preserved across frames.
    pub max_obj_ptrs_in_encoder: usize,
    /// Fuse each memory-attention layer into one graph with [`Op::AxialRope2d`]
    /// (faster than the default five-graph + host-RoPE path). Requires axial RoPE
    /// on the compile device (CPU today; Metal uses host fallback when enabled).
    pub mem_attn_in_graph_rope: bool,
    /// Number of *temporal-position* embeddings packed with each
    /// object-pointer token. Reference YAML: 64.
    pub mem_dim: usize,
}

impl Default for Sam2MemoryConfig {
    fn default() -> Self {
        Self {
            d_model: SAM2_PROMPT_EMBED_DIM,
            num_layers: 4,
            num_heads: 1,
            dim_feedforward: 2048,
            layer_norm_eps: 1e-5,
            kv_in_dim: 64,
            rope_theta: 10000.0,
            // Published YAMLs use feat_sizes: [64, 64] for memory
            // attention RoPE — matches the stride-16 feature grid.
            rope_feat_size: [64, 64],
            rope_k_repeat: true,
            pos_enc_at_input: true,
            pos_enc_at_attn: false,
            pos_enc_at_cross_attn_keys: true,
            pos_enc_at_cross_attn_queries: false,
            max_obj_ptrs_in_encoder: 16,
            mem_dim: 64,
            mem_attn_in_graph_rope: false,
        }
    }
}

/// Top-level SAM 2 configuration — Hiera + FPN + decoder + memory
/// (encoder + attention) for the video path. Mirrors `SAM2Base` in
/// the reference.
#[derive(Debug, Clone)]
pub struct Sam2Config {
    pub hiera: Sam2HieraConfig,
    pub fpn: Sam2FpnConfig,
    pub decoder: Sam2DecoderConfig,
    pub memory: Sam2MemoryConfig,
    pub memory_encoder: Sam2MemoryEncoderConfig,
}

impl Sam2Config {
    pub fn hiera_tiny() -> Self {
        let hiera = Sam2HieraConfig::tiny();
        let fpn = Sam2FpnConfig::for_hiera(&hiera);
        Self {
            hiera,
            fpn,
            decoder: Sam2DecoderConfig::default(),
            memory: Sam2MemoryConfig::default(),
            memory_encoder: Sam2MemoryEncoderConfig::default(),
        }
    }
    pub fn hiera_small() -> Self {
        let hiera = Sam2HieraConfig::small();
        let fpn = Sam2FpnConfig::for_hiera(&hiera);
        Self {
            hiera,
            fpn,
            decoder: Sam2DecoderConfig::default(),
            memory: Sam2MemoryConfig::default(),
            memory_encoder: Sam2MemoryEncoderConfig::default(),
        }
    }
    pub fn hiera_base_plus() -> Self {
        let hiera = Sam2HieraConfig::base_plus();
        let fpn = Sam2FpnConfig::for_hiera(&hiera);
        Self {
            hiera,
            fpn,
            decoder: Sam2DecoderConfig::default(),
            memory: Sam2MemoryConfig::default(),
            memory_encoder: Sam2MemoryEncoderConfig::default(),
        }
    }
    pub fn hiera_large() -> Self {
        let hiera = Sam2HieraConfig::large();
        let fpn = Sam2FpnConfig::for_hiera(&hiera);
        Self {
            hiera,
            fpn,
            decoder: Sam2DecoderConfig::default(),
            memory: Sam2MemoryConfig::default(),
            memory_encoder: Sam2MemoryEncoderConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q_pool_indices_match_reference() {
        // Reference (sam2/modeling/backbones/hieradet.py): q_pool
        // happens at `cumulative_sum(stages)[:-1]`.
        assert_eq!(
            Sam2HieraConfig::tiny().q_pool_block_indices(),
            vec![1, 3, 10]
        );
        assert_eq!(
            Sam2HieraConfig::small().q_pool_block_indices(),
            vec![1, 3, 14]
        );
        assert_eq!(
            Sam2HieraConfig::base_plus().q_pool_block_indices(),
            vec![2, 5, 21]
        );
        assert_eq!(
            Sam2HieraConfig::large().q_pool_block_indices(),
            vec![2, 8, 44]
        );
    }

    #[test]
    fn stage_dim_and_head_doubling() {
        let cfg = Sam2HieraConfig::base_plus();
        assert_eq!(cfg.embed_dim_at_stage(0), 112);
        assert_eq!(cfg.embed_dim_at_stage(1), 224);
        assert_eq!(cfg.embed_dim_at_stage(2), 448);
        assert_eq!(cfg.embed_dim_at_stage(3), 896);
        assert_eq!(cfg.num_heads_at_stage(3), 16);
    }

    #[test]
    fn grid_halves_per_stage() {
        let cfg = Sam2HieraConfig::base_plus();
        assert_eq!(cfg.grid_size_at_stage(0), 256);
        assert_eq!(cfg.grid_size_at_stage(1), 128);
        assert_eq!(cfg.grid_size_at_stage(2), 64);
        assert_eq!(cfg.grid_size_at_stage(3), 32);
    }
}
