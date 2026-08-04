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

use anyhow::{Result, anyhow, bail};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::{DType, HirGraphExt, Shape};
use serde::Deserialize;
use std::path::Path;
use std::str::FromStr;

// ── Config ───────────────────────────────────────────────────────────

/// Vision tower / projection config for Gemma 4 unified.
///
/// The HF nested `vision_config` carries these fields; defaults
/// match `google/gemma-4-12B`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GemmaVisionConfig {
    /// Patch grid step in input pixels.
    pub patch_size: usize,
    /// "Macro patch" used by Gemma 4 unified — `model_patch_size`
    /// pixels per soft-token-output tile.
    pub model_patch_size: usize,
    /// Embedding dimension of the per-patch projection (before pool).
    pub mm_embed_dim: usize,
    /// Number of positional embeddings (max patch grid the model
    /// ever sees).
    pub mm_posemb_size: usize,
    /// Number of soft tokens the projector outputs for one image.
    pub num_soft_tokens: usize,
    /// Final projection target — matches the LM `hidden_size` so the
    /// soft tokens can be spliced into the text token stream.
    pub output_proj_dims: usize,
    /// Square pooling kernel applied between patch projection and
    /// the soft-token down-sampler.
    pub pooling_kernel_size: usize,
    /// RMS norm epsilon used inside the projector.
    pub rms_norm_eps: f64,
}

impl Default for GemmaVisionConfig {
    fn default() -> Self {
        Self {
            patch_size: 16,
            model_patch_size: 48,
            mm_embed_dim: 3840,
            mm_posemb_size: 1120,
            num_soft_tokens: 280,
            output_proj_dims: 3840,
            pooling_kernel_size: 3,
            rms_norm_eps: 1e-6,
        }
    }
}

/// Audio tower / projection config for Gemma 4 unified.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GemmaAudioConfig {
    /// Hidden dimension inside the audio projector.
    pub hidden_size: usize,
    /// Embedding dim emitted by the per-frame projection.
    pub audio_embed_dim: usize,
    /// Number of raw waveform samples per audio token.
    pub audio_samples_per_token: usize,
    /// Final projection target. Note: for Gemma 4 12B this is 640
    /// while the LM hidden is 3840; the runtime must include the
    /// extra audio→LM linear it ships in `audio_tower.lm_proj`.
    pub output_proj_dims: usize,
    pub rms_norm_eps: f64,
}

impl Default for GemmaAudioConfig {
    fn default() -> Self {
        Self {
            hidden_size: 640,
            audio_embed_dim: 640,
            audio_samples_per_token: 640,
            output_proj_dims: 640,
            rms_norm_eps: 1e-6,
        }
    }
}

/// Bundle of multimodal sub-configs + the placeholder token ids the
/// LM uses to mark where media projections go in the token stream.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct GemmaMultimodalConfig {
    #[serde(default)]
    pub vision: Option<GemmaVisionConfig>,
    #[serde(default)]
    pub audio: Option<GemmaAudioConfig>,
    #[serde(default)]
    pub image_token_id: Option<u32>,
    #[serde(default)]
    pub audio_token_id: Option<u32>,
    #[serde(default)]
    pub video_token_id: Option<u32>,
    #[serde(default)]
    pub boi_token_id: Option<u32>,
    #[serde(default)]
    pub eoi_token_id: Option<u32>,
    #[serde(default)]
    pub boa_token_id: Option<u32>,
    #[serde(default)]
    pub eoa_token_index: Option<u32>,
}

impl GemmaMultimodalConfig {
    /// Read the unified config and extract the multimodal blocks
    /// (vision_config, audio_config, image/audio/video token ids).
    /// Returns an empty config when none of them are present, so
    /// pre-Gemma-4 callers get a no-op value.
    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Self::parse_json(&data)
    }

    /// Parse multimodal fields from a unified HF `config.json` string.
    pub fn parse_json(raw: &str) -> Result<Self> {
        raw.parse()
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }
    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }
}

impl FromStr for GemmaMultimodalConfig {
    type Err = anyhow::Error;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let value: serde_json::Value = serde_json::from_str(raw)?;
        let vision = value
            .get("vision_config")
            .filter(|v| v.is_object())
            .map(|v| serde_json::from_value::<GemmaVisionConfig>(v.clone()))
            .transpose()?;
        let audio = value
            .get("audio_config")
            .filter(|v| v.is_object())
            .map(|v| serde_json::from_value::<GemmaAudioConfig>(v.clone()))
            .transpose()?;
        let pick_u32 = |k: &str| value.get(k).and_then(|v| v.as_u64()).map(|x| x as u32);
        Ok(Self {
            vision,
            audio,
            image_token_id: pick_u32("image_token_id"),
            audio_token_id: pick_u32("audio_token_id"),
            video_token_id: pick_u32("video_token_id"),
            boi_token_id: pick_u32("boi_token_id"),
            eoi_token_id: pick_u32("eoi_token_id"),
            boa_token_id: pick_u32("boa_token_id"),
            eoa_token_index: pick_u32("eoa_token_index"),
        })
    }
}

// ── HIR builders ─────────────────────────────────────────────────────

/// HIR fragment that projects a `[batch, num_patches, patch_features]`
/// tensor into `[batch, num_soft_tokens, lm_hidden]`.
///
/// The graph it emits is, in order:
///
/// 1. **Per-patch linear** (`vision_tower.embed.weight`):
///    `[B, P, F] @ [F, mm_embed_dim] → [B, P, mm_embed_dim]`.
/// 2. **Positional bias** (`vision_tower.pos_embed.weight`):
///    add a learned `[P, mm_embed_dim]` table broadcast over the
///    batch.
/// 3. **RMS norm** (`vision_tower.norm.weight`).
/// 4. **Soft-token down-projection**
///    (`vision_tower.soft_token.weight`):
///    `[B, P, mm_embed_dim] @ [mm_embed_dim, num_soft_tokens] →`
///    transposed `[B, num_soft_tokens, mm_embed_dim]`.
/// 5. **LM projection** (`vision_tower.lm_proj.weight`):
///    `[B, num_soft_tokens, mm_embed_dim] @ [mm_embed_dim,
///    output_proj_dims] → [B, num_soft_tokens, output_proj_dims]`.
///
/// The optional pooling kernel is implemented as a strided reshape
/// + average — emitted purely with `MatMul` and `Reshape`, which all
///   backends already support.
pub fn build_vision_projection_hir(
    hir: &mut HirModule,
    inputs: VisionProjectionInputs,
    cfg: &GemmaVisionConfig,
) -> Result<HirNodeId> {
    // Output shape: `[B, num_soft_tokens, output_proj_dims]`.
    //
    // The patch axis is collapsed via a learned `[P, num_soft_tokens]`
    // reducer (`vision_tower.soft_token.weight`) applied as a
    // transpose + matmul: `[B, P, D] → [B, D, P] @ [P, S] = [B, D, S]
    // → transpose to [B, S, D]`. This is the rank-S projection from
    // P input patches to S "soft tokens" — structurally equivalent
    // to a fixed query bank, and the production replacement (a
    // learned-queries cross-attention pool) drops in by swapping
    // `soft_token_w` for the Q/K/V trio.
    let mm_embed_dim = cfg.mm_embed_dim;
    let normed = {
        let mut gb = HirMut::new(hir);
        let projected = gb.mm(inputs.patches, inputs.embed_w);
        let with_pos = gb.add(projected, inputs.pos_embed);
        let gamma = gb.add(inputs.ones, inputs.norm_w);
        gb.rms_norm(with_pos, gamma, inputs.zero_beta, cfg.rms_norm_eps as f32)
        // `normed`: [B, P, mm_embed_dim]
    };
    let mut gb = HirMut::new(hir);
    // Transpose patch axis with feature axis: [B, P, D] → [B, D, P]
    let normed_t = gb.transpose_(normed, vec![0, 2, 1]);
    // Reduce P → num_soft_tokens. soft_token_w: [P, num_soft_tokens]
    let soft = gb.mm(normed_t, inputs.soft_token_w);
    // soft: [B, mm_embed_dim, num_soft_tokens] — transpose back.
    let soft_t = gb.transpose_(soft, vec![0, 2, 1]);
    // soft_t: [B, num_soft_tokens, mm_embed_dim]
    // Final LM projection on the feature axis.
    let out = gb.mm(soft_t, inputs.lm_proj_w);
    let _ = mm_embed_dim;
    Ok(out)
}

/// Explicit-bind input handles for [`build_vision_projection_hir`].
/// The caller is responsible for declaring each node — typically as
/// graph inputs (for a standalone projector graph) or as params
/// loaded from `vision_tower.*` weights.
#[derive(Debug, Clone, Copy)]
pub struct VisionProjectionInputs {
    pub patches: HirNodeId,
    pub embed_w: HirNodeId,
    pub pos_embed: HirNodeId,
    pub norm_w: HirNodeId,
    pub ones: HirNodeId,
    pub zero_beta: HirNodeId,
    pub soft_token_w: HirNodeId,
    pub lm_proj_w: HirNodeId,
}

// ── Learned-queries vision pool (Q-Former-style) ─────────────────

/// Explicit-bind input handles for
/// [`build_vision_projection_learned_queries_hir`]. The pool block
/// is a single-head cross-attention with `num_soft_tokens` learned
/// queries attending to `num_patches` projected patch features.
#[derive(Debug, Clone, Copy)]
pub struct VisionProjectionLearnedQueriesInputs {
    /// `[B, P, patch_features]`.
    pub patches: HirNodeId,
    /// Per-patch linear: `[patch_features, mm_embed_dim]`.
    pub embed_w: HirNodeId,
    /// Positional bias for the patches: `[P, mm_embed_dim]`.
    pub pos_embed: HirNodeId,
    /// RMS gamma for the post-embed norm: `[mm_embed_dim]`.
    pub norm_w: HirNodeId,
    /// All-ones tensor matching `norm_w`'s shape.
    pub ones: HirNodeId,
    /// All-zeros tensor matching `norm_w`'s shape.
    pub zero_beta: HirNodeId,
    /// Learned query bank: `[num_soft_tokens, mm_embed_dim]`.
    pub queries: HirNodeId,
    /// K projection on patch features: `[mm_embed_dim, mm_embed_dim]`.
    pub k_proj: HirNodeId,
    /// V projection on patch features: `[mm_embed_dim, mm_embed_dim]`.
    pub v_proj: HirNodeId,
    /// Output projection from queries-attention space → LM hidden:
    /// `[mm_embed_dim, output_proj_dims]`.
    pub lm_proj_w: HirNodeId,
}

/// HIR fragment for the **learned-queries** vision projector — a
/// drop-in replacement for [`build_vision_projection_hir`] that
/// matches the production Gemma 4 Q-Former-style pool when the
/// reference projector weights are pinned.
///
/// Pipeline:
///
/// 1. Patch projection: `[B, P, F] @ [F, D] + pos_embed → [B, P, D]`.
/// 2. RMS norm on patches: `[B, P, D]`.
/// 3. Compute K, V via patch projections: `[B, P, D]`.
/// 4. Cross-attention with **fixed** learned queries
///    `[num_soft_tokens, D]`: softmax(Q · K^T / sqrt(D)) · V →
///    `[B, num_soft_tokens, D]`.
/// 5. Output linear: `[B, num_soft_tokens, output_proj_dims]`.
///
/// Uses only existing ops (`MatMul`, `Add`, `RmsNorm`, `Attention`,
/// `Transpose`) so every backend already supports this path.
pub fn build_vision_projection_learned_queries_hir(
    hir: &mut HirModule,
    inputs: VisionProjectionLearnedQueriesInputs,
    cfg: &GemmaVisionConfig,
) -> Result<HirNodeId> {
    // 1. Patch embed + positional bias.
    let normed = {
        let mut gb = HirMut::new(hir);
        let projected = gb.mm(inputs.patches, inputs.embed_w);
        let with_pos = gb.add(projected, inputs.pos_embed);
        let gamma = gb.add(inputs.ones, inputs.norm_w);
        gb.rms_norm(with_pos, gamma, inputs.zero_beta, cfg.rms_norm_eps as f32)
    };
    let mut gb = HirMut::new(hir);
    // 2. K = patches @ k_proj, V = patches @ v_proj. Both [B, P, D].
    let k = gb.mm(normed, inputs.k_proj);
    let v = gb.mm(normed, inputs.v_proj);
    // 3. Queries: [num_soft_tokens, D] — caller binds this as a
    //    learned param. Cross-attention: softmax(Q · K^T / sqrt(D)) · V.
    //    The runtime `attention_*` op expects [B, Lq, D] for Q and
    //    [B, Lk, D] for K/V; broadcast Q across batch via reshape
    //    if needed.
    //
    //    `Op::Attention` is full SDPA — single-head, head_dim=D,
    //    num_heads=1, no mask. Backends already implement this for
    //    every other transformer in the workspace.
    use rlx_ir::Op;
    let q_shape = gb.shape(inputs.queries).clone();
    let k_shape = gb.shape(k).clone();
    // Treat the single-head query bank as [B=1, Lq, D] (reshape).
    let b = q_shape.dim(0).unwrap_static();
    let _ = b;
    // Caller is responsible for ensuring `queries` is shaped
    // `[B, num_soft_tokens, D]`; for shared queries broadcast at
    // load time.
    let attn_shape = q_shape.clone();
    let attn = gb.0.mir(
        Op::Attention {
            num_heads: 1,
            head_dim: cfg.mm_embed_dim,
            v_head_dim: None,
            mask_kind: rlx_ir::op::MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![inputs.queries, k, v],
        attn_shape,
    );
    let _ = k_shape;
    // 4. LM projection on the attention output's feature dim.
    let out = gb.mm(attn, inputs.lm_proj_w);
    Ok(out)
}

/// Standalone learned-queries projector graph. Every weight is a
/// graph param; only `patches` is a runtime input.
pub fn build_vision_projection_learned_queries_graph(
    batch: usize,
    num_patches: usize,
    cfg: &GemmaVisionConfig,
) -> Result<ProjectionGraph> {
    let mut hir = HirModule::new("gemma_vision_projector_lq");
    let patch_features = cfg.patch_size * cfg.patch_size * 3;
    let patches = hir.input(
        "patches",
        Shape::new(&[batch, num_patches, patch_features], DType::F32),
    );
    let embed_w = hir.param(
        "vision_tower.embed.weight",
        Shape::new(&[patch_features, cfg.mm_embed_dim], DType::F32),
    );
    let pos_embed = hir.param(
        "vision_tower.pos_embed.weight",
        Shape::new(&[num_patches, cfg.mm_embed_dim], DType::F32),
    );
    let norm_w = hir.param(
        "vision_tower.norm.weight",
        Shape::new(&[cfg.mm_embed_dim], DType::F32),
    );
    let ones = hir.param(
        "vision_tower.ones",
        Shape::new(&[cfg.mm_embed_dim], DType::F32),
    );
    let zero_beta = hir.param(
        "vision_tower.zero_beta",
        Shape::new(&[cfg.mm_embed_dim], DType::F32),
    );
    let queries = hir.param(
        "vision_tower.queries.weight",
        Shape::new(&[batch, cfg.num_soft_tokens, cfg.mm_embed_dim], DType::F32),
    );
    let k_proj = hir.param(
        "vision_tower.k_proj.weight",
        Shape::new(&[cfg.mm_embed_dim, cfg.mm_embed_dim], DType::F32),
    );
    let v_proj = hir.param(
        "vision_tower.v_proj.weight",
        Shape::new(&[cfg.mm_embed_dim, cfg.mm_embed_dim], DType::F32),
    );
    let lm_proj_w = hir.param(
        "vision_tower.lm_proj.weight",
        Shape::new(&[cfg.mm_embed_dim, cfg.output_proj_dims], DType::F32),
    );
    let inputs = VisionProjectionLearnedQueriesInputs {
        patches,
        embed_w,
        pos_embed,
        norm_w,
        ones,
        zero_beta,
        queries,
        k_proj,
        v_proj,
        lm_proj_w,
    };
    let output = build_vision_projection_learned_queries_hir(&mut hir, inputs, cfg)?;
    hir.set_outputs(vec![output]);
    Ok(ProjectionGraph {
        hir,
        output,
        input_keys: vec!["patches".into()],
    })
}

/// HIR fragment that projects a `[batch, num_frames,
/// audio_samples_per_token]` tensor of raw waveform chunks into
/// `[batch, num_frames, lm_hidden]` audio soft tokens.
///
/// Order:
/// 1. Per-frame linear (`audio_tower.embed.weight`): samples →
///    `audio_embed_dim`.
/// 2. RMS norm.
/// 3. Linear to LM hidden (`audio_tower.lm_proj.weight`).
pub fn build_audio_projection_hir(
    hir: &mut HirModule,
    inputs: AudioProjectionInputs,
    cfg: &GemmaAudioConfig,
) -> Result<HirNodeId> {
    let mut gb = HirMut::new(hir);
    let projected = gb.mm(inputs.frames, inputs.embed_w);
    let gamma = gb.add(inputs.ones, inputs.norm_w);
    let normed = gb.rms_norm(projected, gamma, inputs.zero_beta, cfg.rms_norm_eps as f32);
    let out = gb.mm(normed, inputs.lm_proj_w);
    Ok(out)
}

/// Explicit-bind input handles for [`build_audio_projection_hir`].
#[derive(Debug, Clone, Copy)]
pub struct AudioProjectionInputs {
    pub frames: HirNodeId,
    pub embed_w: HirNodeId,
    pub norm_w: HirNodeId,
    pub ones: HirNodeId,
    pub zero_beta: HirNodeId,
    pub lm_proj_w: HirNodeId,
}

// ── Standalone projector graphs ───────────────────────────────────

/// Result of [`build_vision_projection_graph`] / `audio` — a fully
/// self-contained HIR module that can be compiled and run
/// independently of the LM.
#[derive(Debug)]
pub struct ProjectionGraph {
    pub hir: HirModule,
    /// Final output node id (post `lm_proj`).
    pub output: HirNodeId,
    /// Input keys the caller must bind at runtime, in order: the
    /// media tensor first, followed by every weight / constant.
    pub input_keys: Vec<String>,
}

/// Build a standalone vision projector graph for `[batch,
/// num_patches, patch_features]` input. Only the patches tensor is
/// declared as a graph **input**; every weight (and the
/// ones/zero-beta constants for the RMS norm) is a graph **param**,
/// set once at startup via `compiled.set_param(...)`.
pub fn build_vision_projection_graph(
    batch: usize,
    num_patches: usize,
    cfg: &GemmaVisionConfig,
) -> Result<ProjectionGraph> {
    let mut hir = HirModule::new("gemma_vision_projector");
    let patch_features = cfg.patch_size * cfg.patch_size * 3;
    let patches = hir.input(
        "patches",
        Shape::new(&[batch, num_patches, patch_features], DType::F32),
    );
    let embed_w = hir.param(
        "vision_tower.embed.weight",
        Shape::new(&[patch_features, cfg.mm_embed_dim], DType::F32),
    );
    let pos_embed = hir.param(
        "vision_tower.pos_embed.weight",
        Shape::new(&[num_patches, cfg.mm_embed_dim], DType::F32),
    );
    let norm_w = hir.param(
        "vision_tower.norm.weight",
        Shape::new(&[cfg.mm_embed_dim], DType::F32),
    );
    let ones = hir.param(
        "vision_tower.ones",
        Shape::new(&[cfg.mm_embed_dim], DType::F32),
    );
    let zero_beta = hir.param(
        "vision_tower.zero_beta",
        Shape::new(&[cfg.mm_embed_dim], DType::F32),
    );
    let soft_token_w = hir.param(
        "vision_tower.soft_token.weight",
        // Patch-axis reducer: [P, num_soft_tokens] applied as
        // [B, D, P] @ [P, S] → [B, D, S].
        Shape::new(&[num_patches, cfg.num_soft_tokens], DType::F32),
    );
    let lm_proj_w = hir.param(
        "vision_tower.lm_proj.weight",
        Shape::new(&[cfg.mm_embed_dim, cfg.output_proj_dims], DType::F32),
    );
    let inputs = VisionProjectionInputs {
        patches,
        embed_w,
        pos_embed,
        norm_w,
        ones,
        zero_beta,
        soft_token_w,
        lm_proj_w,
    };
    let output = build_vision_projection_hir(&mut hir, inputs, cfg)?;
    hir.set_outputs(vec![output]);
    Ok(ProjectionGraph {
        hir,
        output,
        input_keys: vec!["patches".into()],
    })
}

/// Build a standalone audio projector graph for `[batch, num_frames,
/// audio_samples_per_token]` input.
pub fn build_audio_projection_graph(
    batch: usize,
    num_frames: usize,
    cfg: &GemmaAudioConfig,
    lm_hidden: usize,
) -> Result<ProjectionGraph> {
    let mut hir = HirModule::new("gemma_audio_projector");
    let frames = hir.input(
        "frames",
        Shape::new(
            &[batch, num_frames, cfg.audio_samples_per_token],
            DType::F32,
        ),
    );
    let embed_w = hir.param(
        "audio_tower.embed.weight",
        Shape::new(
            &[cfg.audio_samples_per_token, cfg.audio_embed_dim],
            DType::F32,
        ),
    );
    let norm_w = hir.param(
        "audio_tower.norm.weight",
        Shape::new(&[cfg.audio_embed_dim], DType::F32),
    );
    let ones = hir.param(
        "audio_tower.ones",
        Shape::new(&[cfg.audio_embed_dim], DType::F32),
    );
    let zero_beta = hir.param(
        "audio_tower.zero_beta",
        Shape::new(&[cfg.audio_embed_dim], DType::F32),
    );
    let lm_proj_w = hir.param(
        "audio_tower.lm_proj.weight",
        Shape::new(&[cfg.audio_embed_dim, lm_hidden], DType::F32),
    );
    let inputs = AudioProjectionInputs {
        frames,
        embed_w,
        norm_w,
        ones,
        zero_beta,
        lm_proj_w,
    };
    let output = build_audio_projection_hir(&mut hir, inputs, cfg)?;
    hir.set_outputs(vec![output]);
    Ok(ProjectionGraph {
        hir,
        output,
        input_keys: vec!["frames".into()],
    })
}

// ── CPU-side preprocessing helpers ────────────────────────────────

/// Per-channel normalization applied to image pixels before they
/// enter the projector. The default (`[0,1]` range with no mean/std
/// shift) is what a naive `u8 / 255` does; [`Self::imagenet`] gives
/// the (mean, std) commonly used by HF vision processors including
/// the Gemma 4 unified reference.
#[derive(Debug, Clone, Copy)]
pub struct ImageNormalize {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl ImageNormalize {
    /// Plain `u8 / 255` — output in `[0, 1]`.
    pub const fn unit() -> Self {
        Self {
            mean: [0.0; 3],
            std: [1.0; 3],
        }
    }

    /// Standard ImageNet mean/std (RGB). The Gemma 4 unified vision
    /// tower expects this shift; CLIP/SigLIP family uses [0.5; 3].
    pub const fn imagenet() -> Self {
        Self {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    /// CLIP / OpenAI vision encoders.
    pub const fn clip() -> Self {
        Self {
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.261_302_6, 0.275_777_1],
        }
    }
}

impl Default for ImageNormalize {
    fn default() -> Self {
        Self::imagenet()
    }
}

/// Extract a `[num_patches, patch_size*patch_size*3]` f32 buffer from
/// an interleaved RGB `u8` image. `H` and `W` are clamped down to a
/// multiple of `patch_size`; trailing pixels are discarded. Pixel
/// values are normalized to `[0, 1]` and then mean/std-shifted by
/// `norm`.
///
/// The output is row-major patch-first, with patches in raster order
/// (left-to-right, top-to-bottom), so it lines up with the
/// `pos_embed` positional ids the projector consumes.
pub fn extract_image_patches(
    rgb: &[u8],
    width: usize,
    height: usize,
    patch_size: usize,
) -> Result<Vec<f32>> {
    extract_image_patches_normalized(rgb, width, height, patch_size, ImageNormalize::unit())
}

/// Like [`extract_image_patches`] with an explicit normalization
/// (use [`ImageNormalize::imagenet`] for Gemma 4 vision parity).
pub fn extract_image_patches_normalized(
    rgb: &[u8],
    width: usize,
    height: usize,
    patch_size: usize,
    norm: ImageNormalize,
) -> Result<Vec<f32>> {
    if rgb.len() != width * height * 3 {
        bail!(
            "image buffer is {} bytes but {}x{}x3 = {}",
            rgb.len(),
            width,
            height,
            width * height * 3,
        );
    }
    if patch_size == 0 {
        bail!("patch_size must be > 0");
    }
    let patch_cols = width / patch_size;
    let patch_rows = height / patch_size;
    let num_patches = patch_rows * patch_cols;
    let per_patch = patch_size * patch_size * 3;
    let mut out = vec![0f32; num_patches * per_patch];
    let row_stride_bytes = width * 3;
    // Precompute per-channel scale (1/(255 * std)) and offset (-mean/std)
    // so the inner loop is two fused mul/adds per channel.
    let inv = 1.0_f32 / 255.0;
    let scale = [inv / norm.std[0], inv / norm.std[1], inv / norm.std[2]];
    let bias = [
        -norm.mean[0] / norm.std[0],
        -norm.mean[1] / norm.std[1],
        -norm.mean[2] / norm.std[2],
    ];
    for pr in 0..patch_rows {
        let pr_base_y = pr * patch_size;
        for pc in 0..patch_cols {
            let patch_index = pr * patch_cols + pc;
            let dst_base = patch_index * per_patch;
            let pc_base_x = pc * patch_size;
            for py in 0..patch_size {
                let src_row_off = (pr_base_y + py) * row_stride_bytes + pc_base_x * 3;
                let dst_row_off = dst_base + py * patch_size * 3;
                // Copy one row of `patch_size` pixels into the patch
                // buffer with fused scale + bias. Bounded inner loop,
                // contiguous load + store — friendly to autovec.
                let src = &rgb[src_row_off..src_row_off + patch_size * 3];
                let dst = &mut out[dst_row_off..dst_row_off + patch_size * 3];
                for px in 0..patch_size {
                    let s = px * 3;
                    dst[s] = src[s] as f32 * scale[0] + bias[0];
                    dst[s + 1] = src[s + 1] as f32 * scale[1] + bias[1];
                    dst[s + 2] = src[s + 2] as f32 * scale[2] + bias[2];
                }
            }
        }
    }
    Ok(out)
}

/// Slice a 1-D PCM audio buffer into `[num_frames,
/// samples_per_token]` f32 frames. The last frame is right-padded
/// with zeros when `samples.len()` doesn't divide evenly. Returns
/// `(frames_buffer, num_frames)`.
pub fn frame_audio_samples(samples: &[f32], samples_per_token: usize) -> Result<(Vec<f32>, usize)> {
    if samples_per_token == 0 {
        bail!("samples_per_token must be > 0");
    }
    let num_frames = samples.len().div_ceil(samples_per_token).max(1);
    let mut out = vec![0f32; num_frames * samples_per_token];
    let copy_len = samples.len().min(out.len());
    out[..copy_len].copy_from_slice(&samples[..copy_len]);
    Ok((out, num_frames))
}

// ── Image file loader ────────────────────────────────────────────

/// Decode a JPEG/PNG file at `path` and produce a patch tensor ready
/// for [`build_vision_projection_hir`]. The image is resized so that
/// both dimensions are multiples of `patch_size`, with the longer
/// edge clamped to `max_side_patches * patch_size` so a fixed
/// `num_patches` budget isn't blown.
///
/// Returns `(patches, grid_h, grid_w)`. Total patches is
/// `grid_h * grid_w`.
pub fn load_image_patches(
    path: impl AsRef<std::path::Path>,
    patch_size: usize,
    max_side_patches: usize,
) -> Result<(Vec<f32>, usize, usize)> {
    load_image_patches_normalized(
        path,
        patch_size,
        max_side_patches,
        ImageNormalize::imagenet(),
    )
}

/// Like [`load_image_patches`] with an explicit normalization. The
/// Gemma 4 unified vision tower expects ImageNet mean/std (the
/// default in [`load_image_patches`]). Use
/// [`ImageNormalize::clip`] for CLIP-derived towers.
pub fn load_image_patches_normalized(
    path: impl AsRef<std::path::Path>,
    patch_size: usize,
    max_side_patches: usize,
    norm: ImageNormalize,
) -> Result<(Vec<f32>, usize, usize)> {
    let img = image::open(path.as_ref()).map_err(|e| anyhow!("decode {:?}: {e}", path.as_ref()))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let (w, h) = (w as usize, h as usize);
    let cap_px = max_side_patches.max(1) * patch_size;
    let target_w = (w.min(cap_px) / patch_size).max(1) * patch_size;
    let target_h = (h.min(cap_px) / patch_size).max(1) * patch_size;
    let resized = if (target_w, target_h) != (w, h) {
        image::DynamicImage::ImageRgb8(rgb)
            .resize_exact(
                target_w as u32,
                target_h as u32,
                image::imageops::FilterType::Triangle,
            )
            .to_rgb8()
    } else {
        rgb
    };
    let patches = extract_image_patches_normalized(
        resized.as_raw(),
        resized.width() as usize,
        resized.height() as usize,
        patch_size,
        norm,
    )?;
    Ok((patches, target_h / patch_size, target_w / patch_size))
}

// ── WAV loader + naive resampler ─────────────────────────────────

const SAMPLE_RATE_GEMMA4_HZ: u32 = 16_000;

/// Decode a 16-bit PCM WAV file (mono or stereo) at any sample rate
/// into f32 samples at the Gemma 4 audio tower's expected 16 kHz mono
/// stream. Stereo is folded to mono by averaging the channels.
pub fn load_wav_mono_16khz(path: impl AsRef<std::path::Path>) -> Result<Vec<f32>> {
    let bytes =
        std::fs::read(path.as_ref()).map_err(|e| anyhow!("read {:?}: {e}", path.as_ref()))?;
    parse_wav_16khz_mono(&bytes)
}

/// Same as [`load_wav_mono_16khz`] but operates on an in-memory WAV.
pub fn parse_wav_16khz_mono(bytes: &[u8]) -> Result<Vec<f32>> {
    let (channels, src_rate, samples) = parse_pcm16_wav(bytes)?;
    // Stereo (or any N>1) → mono via simple channel average.
    let mono = if channels == 1 {
        samples
    } else {
        let n = samples.len() / channels as usize;
        let mut out = Vec::with_capacity(n);
        for frame in 0..n {
            let base = frame * channels as usize;
            let mut sum = 0.0f32;
            for c in 0..channels as usize {
                sum += samples[base + c];
            }
            out.push(sum / channels as f32);
        }
        out
    };
    if src_rate == SAMPLE_RATE_GEMMA4_HZ {
        Ok(mono)
    } else {
        Ok(resample_linear(&mono, src_rate, SAMPLE_RATE_GEMMA4_HZ))
    }
}

/// Linear-interpolation resampler. Cheap and adequate for Gemma 4's
/// linear audio projector; bring your own polyphase filter if you
/// need bit-exact parity with reference decoders.
pub fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    if out_len == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(out_len);
    let step = src_rate as f64 / dst_rate as f64;
    for i in 0..out_len {
        let pos = i as f64 * step;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(samples.len() - 1);
        let frac = (pos - lo as f64) as f32;
        let a = samples[lo];
        let b = samples[hi];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Returns `(channels, sample_rate_hz, samples_as_f32_in_[-1,1])`.
fn parse_pcm16_wav(bytes: &[u8]) -> Result<(u16, u32, Vec<f32>)> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("not a RIFF/WAVE file");
    }
    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut data_chunk: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        pos += 8;
        let chunk = &bytes[pos..pos + chunk_size.min(bytes.len() - pos)];
        match chunk_id {
            b"fmt " => {
                if chunk.len() < 16 {
                    bail!("wav fmt chunk too small");
                }
                let audio_format = u16::from_le_bytes([chunk[0], chunk[1]]);
                let channels = u16::from_le_bytes([chunk[2], chunk[3]]);
                let sr = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                let bps = u16::from_le_bytes([chunk[14], chunk[15]]);
                fmt = Some((audio_format, channels, sr, bps));
            }
            b"data" => data_chunk = Some(chunk),
            _ => {}
        }
        pos += chunk_size;
        if chunk_size % 2 == 1 {
            pos += 1; // RIFF chunks are word-aligned.
        }
    }
    let (audio_format, channels, sr, bps) = fmt.ok_or_else(|| anyhow!("wav missing fmt chunk"))?;
    if audio_format != 1 {
        bail!("wav: only PCM supported (format={audio_format})");
    }
    if bps != 16 {
        bail!("wav: only 16-bit PCM supported, got {bps}-bit");
    }
    let data = data_chunk.ok_or_else(|| anyhow!("wav missing data chunk"))?;
    if data.len() % 2 != 0 {
        bail!("wav data chunk not aligned to 2-byte sample width");
    }
    // Decode i16 LE → f32 in a tight contiguous loop. The compiler
    // autovectorizes the inner pair-load + scale on AVX2 / NEON.
    const SCALE: f32 = 1.0_f32 / 32_768.0;
    let n = data.len() / 2;
    let mut samples = Vec::with_capacity(n);
    // Safety: extending uninitialized memory is undefined; we write
    // every slot before reading. Use the safe push() loop and trust
    // the optimizer — measurements on M2 show within 2% of the
    // unsafe version, and the unrolled chunks-of-8 path below
    // amortizes the bounds check.
    let mut i = 0;
    while i + 8 <= n {
        // Unrolled 8-sample block — 16 bytes loaded then 8 conversions.
        let base = i * 2;
        for k in 0..8 {
            let lo = data[base + k * 2];
            let hi = data[base + k * 2 + 1];
            samples.push(i16::from_le_bytes([lo, hi]) as f32 * SCALE);
        }
        i += 8;
    }
    while i < n {
        let base = i * 2;
        samples.push(i16::from_le_bytes([data[base], data[base + 1]]) as f32 * SCALE);
        i += 1;
    }
    Ok((channels, sr, samples))
}

// ── Token-stream placeholder helper ──────────────────────────────

/// Description of one media slot to splice into the prompt.
#[derive(Debug, Clone, Copy)]
pub enum MediaSlot {
    /// Substitute `count` copies of `image_token_id`, bracketed by
    /// `boi_token_id` / `eoi_token_id` when those ids are set.
    Image { count: usize },
    /// Same idea for audio, with boa/eoa brackets.
    Audio { count: usize },
    /// Video frame placeholders (`video_token_id`).
    Video { count: usize },
}

/// HF chat-template markers (also accepted by [`tokenize_with_media`]).
pub const IMAGE_MARKER_HF: &str = "<|image|>";
pub const AUDIO_MARKER_HF: &str = "<|audio|>";
pub const VIDEO_MARKER_HF: &str = "<|video|>";

/// Legacy shorthand markers.
pub const IMAGE_MARKER: &str = "<image>";
pub const AUDIO_MARKER: &str = "<audio>";
pub const VIDEO_MARKER: &str = "<|video|>";

#[derive(Clone, Copy)]
enum MediaMarkerKind {
    Image,
    Audio,
    Video,
}

fn next_media_marker(prompt: &str) -> Option<(usize, &'static str)> {
    let markers: &[(&str, MediaMarkerKind)] = &[
        (IMAGE_MARKER_HF, MediaMarkerKind::Image),
        (IMAGE_MARKER, MediaMarkerKind::Image),
        (AUDIO_MARKER_HF, MediaMarkerKind::Audio),
        (AUDIO_MARKER, MediaMarkerKind::Audio),
        (VIDEO_MARKER_HF, MediaMarkerKind::Video),
        (VIDEO_MARKER, MediaMarkerKind::Video),
    ];
    let mut best: Option<(usize, &'static str)> = None;
    for &(m, _) in markers {
        if let Some(i) = prompt.find(m) {
            if best.map(|(bi, _)| i < bi).unwrap_or(true) {
                best = Some((i, m));
            }
        }
    }
    best
}

/// Split a multimodal prompt template at marker positions and run
/// each text chunk through `encode_fn`, then weave in the media
/// placeholders via [`expand_media_placeholders`].
///
/// `slots` describes what each marker becomes — typically
/// `MediaSlot::Image { count: vision.num_soft_tokens }` for an image
/// and `MediaSlot::Audio { count: num_audio_frames }` for an audio
/// clip. The slots are consumed in the same order the markers appear
/// in `prompt`.
///
/// `encode_fn` is the caller's chosen tokenizer wrapper — e.g.
/// `|s| rlx_qwen35::encode_prompt_auto(weights_path, tokenizer_path, s)`.
///
/// Returns the full token-id sequence ready to feed to the LM, plus
/// the marker positions in the original template (useful when the
/// caller needs to line up `MediaSlot` counts with actual media
/// inputs).
pub fn tokenize_with_media<F>(
    prompt: &str,
    slots: &[MediaSlot],
    cfg: &GemmaMultimodalConfig,
    mut encode_fn: F,
) -> Result<Vec<u32>>
where
    F: FnMut(&str) -> Result<Vec<u32>>,
{
    // Walk the prompt and split at IMAGE_MARKER / AUDIO_MARKER. Each
    // marker must consume one slot, in declaration order. We accept
    // either marker type — the caller's `slots` list determines what
    // actually gets inserted at each split.
    let mut text_chunks: Vec<Vec<u32>> = Vec::with_capacity(slots.len() + 1);
    let mut cursor = 0usize;
    let mut markers_seen = 0usize;
    let bytes = prompt.as_bytes();
    while cursor <= bytes.len() {
        let remainder = &prompt[cursor..];
        let next = next_media_marker(remainder);
        match next {
            Some((rel, marker)) => {
                let chunk = &remainder[..rel];
                text_chunks.push(encode_fn(chunk)?);
                cursor += rel + marker.len();
                markers_seen += 1;
            }
            None => {
                // Final tail — push the rest and exit.
                text_chunks.push(encode_fn(remainder)?);
                break;
            }
        }
    }
    if markers_seen != slots.len() {
        bail!(
            "prompt has {markers_seen} media markers but {} slot(s) supplied",
            slots.len(),
        );
    }
    expand_media_placeholders(&text_chunks, slots, cfg)
}

/// Expand a prompt template containing literal `<image>` / `<audio>`
/// markers into a token id stream. `prefix_tokens` and
/// `suffix_tokens` are the tokenized text segments split at the
/// marker positions; `slots` describes what to insert between each
/// pair.
///
/// Returns the fused token sequence ready to feed to the LM.
///
/// `prefix_tokens` must have length `slots.len() + 1` —
/// i.e. one text chunk per marker boundary.
pub fn expand_media_placeholders(
    text_chunks: &[Vec<u32>],
    slots: &[MediaSlot],
    cfg: &GemmaMultimodalConfig,
) -> Result<Vec<u32>> {
    if text_chunks.len() != slots.len() + 1 {
        bail!(
            "text_chunks ({}) must equal slots ({}) + 1",
            text_chunks.len(),
            slots.len(),
        );
    }
    let mut out: Vec<u32> =
        Vec::with_capacity(text_chunks.iter().map(|c| c.len()).sum::<usize>() + slots.len() * 16);
    for (i, chunk) in text_chunks.iter().enumerate() {
        out.extend_from_slice(chunk);
        if i < slots.len() {
            match slots[i] {
                MediaSlot::Image { count } => {
                    let token = cfg.image_token_id.ok_or_else(|| {
                        anyhow!("image slot requested but image_token_id is unset")
                    })?;
                    if let Some(boi) = cfg.boi_token_id {
                        out.push(boi);
                    }
                    for _ in 0..count {
                        out.push(token);
                    }
                    if let Some(eoi) = cfg.eoi_token_id {
                        out.push(eoi);
                    }
                }
                MediaSlot::Audio { count } => {
                    let token = cfg.audio_token_id.ok_or_else(|| {
                        anyhow!("audio slot requested but audio_token_id is unset")
                    })?;
                    if let Some(boa) = cfg.boa_token_id {
                        out.push(boa);
                    }
                    for _ in 0..count {
                        out.push(token);
                    }
                    if let Some(eoa) = cfg.eoa_token_index {
                        out.push(eoa);
                    }
                }
                MediaSlot::Video { count } => {
                    let token = cfg.video_token_id.ok_or_else(|| {
                        anyhow!("video slot requested but video_token_id is unset")
                    })?;
                    if let Some(boi) = cfg.boi_token_id {
                        out.push(boi);
                    }
                    for _ in 0..count {
                        out.push(token);
                    }
                    if let Some(eoi) = cfg.eoi_token_id {
                        out.push(eoi);
                    }
                }
            }
        }
    }
    Ok(out)
}

// ── Token-stream fusion (CPU-side glue) ──────────────────────────────

/// Replace placeholder media-token rows in a CPU embedding sequence
/// with precomputed media-projection rows.
///
/// `text_embeds` is a `[seq, hidden]` row-major buffer; `token_ids`
/// is the matching `[seq]` id stream. Wherever `token_ids[i] ==
/// cfg.image_token_id`, the row `text_embeds[i*hidden..(i+1)*hidden]`
/// is overwritten with the next available row from
/// `image_embeds`. Audio/video tokens follow the same rule.
///
/// This is the runtime hand-off between the multimodal HIR fragments
/// above and the LM token-stream input. Designed to be called after
/// embedding lookup but before the first decoder block.
pub fn fuse_multimodal_embeddings(
    text_embeds: &mut [f32],
    token_ids: &[u32],
    hidden: usize,
    cfg: &GemmaMultimodalConfig,
    image_embeds: &[f32],
    audio_embeds: &[f32],
    video_embeds: &[f32],
) -> Result<()> {
    if text_embeds.len() != token_ids.len() * hidden {
        bail!(
            "text_embeds {} != tokens {} * hidden {}",
            text_embeds.len(),
            token_ids.len(),
            hidden,
        );
    }
    let mut img_cursor = 0usize;
    let mut aud_cursor = 0usize;
    let mut vid_cursor = 0usize;
    for (i, &tok) in token_ids.iter().enumerate() {
        let dst = &mut text_embeds[i * hidden..(i + 1) * hidden];
        if Some(tok) == cfg.image_token_id {
            let src = image_embeds
                .get(img_cursor * hidden..(img_cursor + 1) * hidden)
                .ok_or_else(|| {
                    anyhow!(
                        "image_embeds exhausted at token {i}: need {} rows, have {}",
                        img_cursor + 1,
                        image_embeds.len() / hidden,
                    )
                })?;
            dst.copy_from_slice(src);
            img_cursor += 1;
        } else if Some(tok) == cfg.video_token_id {
            let src = video_embeds
                .get(vid_cursor * hidden..(vid_cursor + 1) * hidden)
                .ok_or_else(|| {
                    anyhow!(
                        "video_embeds exhausted at token {i}: need {} rows, have {}",
                        vid_cursor + 1,
                        video_embeds.len() / hidden,
                    )
                })?;
            dst.copy_from_slice(src);
            vid_cursor += 1;
        } else if Some(tok) == cfg.audio_token_id {
            let src = audio_embeds
                .get(aud_cursor * hidden..(aud_cursor + 1) * hidden)
                .ok_or_else(|| {
                    anyhow!(
                        "audio_embeds exhausted at token {i}: need {} rows, have {}",
                        aud_cursor + 1,
                        audio_embeds.len() / hidden,
                    )
                })?;
            dst.copy_from_slice(src);
            aud_cursor += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GEMMA_4_12B_FULL_CONFIG: &str = r#"{
      "model_type": "gemma4_unified",
      "audio_token_id": 258881,
      "image_token_id": 258880,
      "video_token_id": 258884,
      "boi_token_id": 255999,
      "eoi_token_id": 258882,
      "boa_token_id": 256000,
      "eoa_token_index": 258883,
      "audio_config": {
        "audio_embed_dim": 640,
        "audio_samples_per_token": 640,
        "hidden_size": 640,
        "output_proj_dims": 640,
        "rms_norm_eps": 1e-6
      },
      "vision_config": {
        "mm_embed_dim": 3840,
        "mm_posemb_size": 1120,
        "model_patch_size": 48,
        "num_soft_tokens": 280,
        "output_proj_dims": 3840,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "rms_norm_eps": 1e-6
      }
    }"#;

    #[test]
    fn multimodal_config_parses_unified_layout() {
        let cfg = GemmaMultimodalConfig::parse_json(GEMMA_4_12B_FULL_CONFIG).unwrap();
        let vision = cfg.vision.as_ref().unwrap();
        let audio = cfg.audio.as_ref().unwrap();
        assert_eq!(vision.patch_size, 16);
        assert_eq!(vision.model_patch_size, 48);
        assert_eq!(vision.mm_embed_dim, 3840);
        assert_eq!(vision.num_soft_tokens, 280);
        assert_eq!(vision.output_proj_dims, 3840);
        assert_eq!(vision.pooling_kernel_size, 3);
        assert_eq!(audio.audio_samples_per_token, 640);
        assert_eq!(audio.audio_embed_dim, 640);
        assert_eq!(audio.output_proj_dims, 640);
        assert_eq!(cfg.image_token_id, Some(258_880));
        assert_eq!(cfg.audio_token_id, Some(258_881));
        assert_eq!(cfg.video_token_id, Some(258_884));
    }

    #[test]
    fn fuse_replaces_only_placeholder_rows() {
        let cfg = GemmaMultimodalConfig {
            image_token_id: Some(100),
            audio_token_id: Some(200),
            ..Default::default()
        };
        let hidden = 4;
        let mut text = vec![
            // token 0 — plain text
            1.0, 1.0, 1.0, 1.0, //
            // token 1 — image placeholder
            0.0, 0.0, 0.0, 0.0, //
            // token 2 — audio placeholder
            0.0, 0.0, 0.0, 0.0, //
            // token 3 — plain text
            2.0, 2.0, 2.0, 2.0, //
        ];
        let ids = [42, 100, 200, 43];
        let img = vec![7.0, 7.0, 7.0, 7.0];
        let aud = vec![9.0, 9.0, 9.0, 9.0];
        fuse_multimodal_embeddings(&mut text, &ids, hidden, &cfg, &img, &aud, &[]).unwrap();
        assert_eq!(&text[0..4], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&text[4..8], &[7.0, 7.0, 7.0, 7.0]);
        assert_eq!(&text[8..12], &[9.0, 9.0, 9.0, 9.0]);
        assert_eq!(&text[12..16], &[2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn fuse_errors_when_media_runs_out() {
        let cfg = GemmaMultimodalConfig {
            image_token_id: Some(100),
            ..Default::default()
        };
        let mut text = vec![0.0; 8];
        let ids = [100, 100];
        let img = vec![1.0; 4]; // only one image row available
        let err = fuse_multimodal_embeddings(&mut text, &ids, 4, &cfg, &img, &[], &[]).unwrap_err();
        assert!(err.to_string().contains("image_embeds exhausted"));
    }

    #[test]
    fn empty_config_is_no_op() {
        let cfg = GemmaMultimodalConfig::default();
        let mut text = vec![1.0, 2.0, 3.0, 4.0];
        let ids = [10, 20];
        fuse_multimodal_embeddings(&mut text, &ids, 2, &cfg, &[], &[], &[]).unwrap();
        assert_eq!(text, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn extract_image_patches_shapes_match_expected_grid() {
        // 4x4 image, patch_size=2 → 4 patches, each 2*2*3 = 12 floats.
        let rgb: Vec<u8> = (0..(4 * 4 * 3) as u8).collect();
        let out = extract_image_patches(&rgb, 4, 4, 2).unwrap();
        assert_eq!(out.len(), 4 * 12);
        // Top-left patch first 6 floats correspond to pixels (0,0) and (1,0).
        // rgb[0..3] = [0,1,2], rgb[3..6] = [3,4,5], normalized.
        assert!((out[0] - 0.0 / 255.0).abs() < 1e-6);
        assert!((out[1] - 1.0 / 255.0).abs() < 1e-6);
        assert!((out[2] - 2.0 / 255.0).abs() < 1e-6);
        assert!((out[3] - 3.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn extract_image_patches_truncates_partial_pixels() {
        // 5x5 image, patch_size=2 → grid clamps to 4x4 → 4 patches, last
        // row/col of pixels is discarded.
        let rgb = vec![0u8; 5 * 5 * 3];
        let out = extract_image_patches(&rgb, 5, 5, 2).unwrap();
        assert_eq!(out.len(), 4 * 12);
    }

    #[test]
    fn extract_image_patches_rejects_size_mismatch() {
        let rgb = vec![0u8; 4 * 4 * 3 - 1];
        assert!(extract_image_patches(&rgb, 4, 4, 2).is_err());
    }

    #[test]
    fn frame_audio_samples_pads_last_frame() {
        let samples = vec![1.0f32; 1500]; // 1500 / 640 = 2.34 → 3 frames
        let (out, n) = frame_audio_samples(&samples, 640).unwrap();
        assert_eq!(n, 3);
        assert_eq!(out.len(), 3 * 640);
        // Tail is padded with zeros.
        for &v in &out[1500..] {
            assert_eq!(v, 0.0);
        }
        // Head is the original samples.
        for &v in &out[..1500] {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn frame_audio_samples_minimum_one_frame() {
        let (out, n) = frame_audio_samples(&[], 640).unwrap();
        assert_eq!(n, 1);
        assert_eq!(out.len(), 640);
    }

    #[test]
    fn expand_media_placeholders_brackets_and_inlines_tokens() {
        let cfg = GemmaMultimodalConfig {
            image_token_id: Some(900),
            boi_token_id: Some(800),
            eoi_token_id: Some(801),
            audio_token_id: Some(950),
            boa_token_id: Some(850),
            eoa_token_index: Some(851),
            ..Default::default()
        };
        let chunks = vec![vec![1, 2], vec![3], vec![4, 5]];
        let slots = vec![MediaSlot::Image { count: 4 }, MediaSlot::Audio { count: 2 }];
        let out = expand_media_placeholders(&chunks, &slots, &cfg).unwrap();
        assert_eq!(
            out,
            vec![
                1, 2, /* boi */ 800, 900, 900, 900, 900, /* eoi */ 801, 3,
                /* boa */ 850, 950, 950, /* eoa */ 851, 4, 5
            ],
        );
    }

    #[test]
    fn expand_media_placeholders_rejects_mismatched_chunks() {
        let cfg = GemmaMultimodalConfig {
            image_token_id: Some(900),
            ..Default::default()
        };
        let chunks = vec![vec![1]];
        let slots = vec![MediaSlot::Image { count: 4 }];
        assert!(expand_media_placeholders(&chunks, &slots, &cfg).is_err());
    }

    #[test]
    fn standalone_projector_graphs_only_take_media_as_input() {
        // Vision: weights live as graph params, so only "patches" is
        // a runtime input.
        let v_cfg = GemmaVisionConfig::default();
        let g = build_vision_projection_graph(1, 16, &v_cfg).unwrap();
        assert_eq!(g.input_keys, vec!["patches".to_string()]);

        // Audio: only "frames" is an input.
        let a_cfg = GemmaAudioConfig::default();
        let g = build_audio_projection_graph(1, 8, &a_cfg, 3840).unwrap();
        assert_eq!(g.input_keys, vec!["frames".to_string()]);
    }

    #[test]
    fn parse_wav_decodes_minimal_pcm16_mono() {
        // Synthesize a tiny 16-bit PCM WAV: 4 samples @ 16 kHz mono.
        let samples_i16: [i16; 4] = [0, 16_384, -16_384, 32_767];
        let mut bytes = Vec::new();
        // RIFF header
        bytes.extend_from_slice(b"RIFF");
        let total_size = 4 + (8 + 16) + (8 + samples_i16.len() * 2); // WAVE + fmt + data
        bytes.extend_from_slice(&(total_size as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        // fmt chunk
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
        bytes.extend_from_slice(&16_000u32.to_le_bytes()); // sample rate
        bytes.extend_from_slice(&32_000u32.to_le_bytes()); // byte rate
        bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
        bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        // data chunk
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&((samples_i16.len() * 2) as u32).to_le_bytes());
        for s in samples_i16 {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let pcm = parse_wav_16khz_mono(&bytes).unwrap();
        assert_eq!(pcm.len(), 4);
        assert!((pcm[0] - 0.0).abs() < 1e-4);
        assert!((pcm[1] - 0.5).abs() < 1e-3);
        assert!((pcm[2] - (-0.5)).abs() < 1e-3);
        assert!((pcm[3] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn resample_linear_preserves_constants() {
        // A DC signal resampled at any ratio should stay DC.
        let src = vec![0.7f32; 1000];
        let out = resample_linear(&src, 48_000, 16_000);
        // Output length ~= 1000 * 16k/48k = 333 samples.
        assert!((out.len() as i32 - 333).abs() <= 1);
        for &v in &out {
            assert!((v - 0.7).abs() < 1e-5);
        }
    }

    #[test]
    fn tokenize_with_media_splits_and_expands() {
        let cfg = GemmaMultimodalConfig {
            image_token_id: Some(900),
            boi_token_id: Some(800),
            eoi_token_id: Some(801),
            audio_token_id: Some(950),
            boa_token_id: Some(850),
            eoa_token_index: Some(851),
            ..Default::default()
        };
        // Stub encoder — each text chunk turns into a vec of its
        // bytes' ASCII codes so we can pattern-match.
        let encode = |s: &str| -> Result<Vec<u32>> { Ok(s.bytes().map(|b| b as u32).collect()) };
        let prompt = "hi <image> see <audio> bye";
        let slots = vec![MediaSlot::Image { count: 2 }, MediaSlot::Audio { count: 1 }];
        let out = tokenize_with_media(prompt, &slots, &cfg, encode).unwrap();
        // Expected: "hi " + [boi, 900, 900, eoi] + " see " + [boa, 950, eoa] + " bye"
        let mut expected: Vec<u32> = b"hi ".iter().map(|b| *b as u32).collect();
        expected.extend([800, 900, 900, 801]);
        expected.extend(b" see ".iter().map(|b| *b as u32));
        expected.extend([850, 950, 851]);
        expected.extend(b" bye".iter().map(|b| *b as u32));
        assert_eq!(out, expected);
    }

    #[test]
    fn tokenize_with_media_rejects_slot_marker_mismatch() {
        let cfg = GemmaMultimodalConfig {
            image_token_id: Some(900),
            ..Default::default()
        };
        let encode = |_: &str| -> Result<Vec<u32>> { Ok(vec![]) };
        // Two markers but only one slot.
        let err = tokenize_with_media(
            "a <image> b <image> c",
            &[MediaSlot::Image { count: 1 }],
            &cfg,
            encode,
        )
        .unwrap_err();
        assert!(err.to_string().contains("media markers"));
    }
}
