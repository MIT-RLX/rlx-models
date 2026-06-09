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

//! HIR builders for Gemma 4 **unified** vision + audio embedders.
//!
//! Weight keys match HuggingFace `google/gemma-4-12B-it`:
//! - `model.vision_embedder.*`
//! - `model.embed_vision.embedding_projection.weight`
//! - `model.embed_audio.embedding_projection.weight`

use crate::multimodal::{GemmaAudioConfig, GemmaVisionConfig, ProjectionGraph};
use anyhow::Result;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::{DType, HirGraphExt, Op, Shape};

const LN_EPS: f32 = 1e-6;

fn layer_norm(
    gb: &mut HirMut<'_>,
    x: HirNodeId,
    gamma: HirNodeId,
    beta: HirNodeId,
    out_shape: Shape,
) -> HirNodeId {
    gb.0.mir(
        Op::LayerNorm {
            axis: -1,
            eps: LN_EPS,
        },
        vec![x, gamma, beta],
        out_shape,
    )
}

fn rms_no_scale(
    gb: &mut HirMut<'_>,
    x: HirNodeId,
    ones: HirNodeId,
    zero_beta: HirNodeId,
    eps: f32,
) -> HirNodeId {
    gb.rms_norm(x, ones, zero_beta, eps)
}

/// Inputs for [`build_unified_vision_hir`].
#[derive(Debug, Clone, Copy)]
pub struct UnifiedVisionInputs {
    pub patches: HirNodeId,
    pub pos_bias: HirNodeId,
    pub patch_ln1_w: HirNodeId,
    pub patch_ln1_b: HirNodeId,
    pub patch_dense_w: HirNodeId,
    pub patch_dense_b: HirNodeId,
    pub patch_ln2_w: HirNodeId,
    pub patch_ln2_b: HirNodeId,
    pub pos_norm_w: HirNodeId,
    pub pos_norm_b: HirNodeId,
    pub embed_proj_w: HirNodeId,
    pub ones: HirNodeId,
    pub zero_beta: HirNodeId,
}

/// Unified vision: LN→Dense→LN→+pos→LN→RMS→Linear.
pub fn build_unified_vision_hir(
    hir: &mut HirModule,
    inputs: UnifiedVisionInputs,
    cfg: &GemmaVisionConfig,
    num_slots: usize,
) -> Result<HirNodeId> {
    let d = cfg.mm_embed_dim;
    let patch_dim = cfg.model_patch_size * cfg.model_patch_size * 3;
    let out_shape = Shape::new(&[1, num_slots, d], DType::F32);
    let patch_shape = Shape::new(&[1, num_slots, patch_dim], DType::F32);

    let mut gb = HirMut::new(hir);
    let h = gb.0.mir(
        Op::LayerNorm {
            axis: -1,
            eps: LN_EPS,
        },
        vec![inputs.patches, inputs.patch_ln1_w, inputs.patch_ln1_b],
        patch_shape,
    );
    let h = gb.mm(h, inputs.patch_dense_w);
    let h = gb.add(h, inputs.patch_dense_b);
    let h = layer_norm(
        &mut gb,
        h,
        inputs.patch_ln2_w,
        inputs.patch_ln2_b,
        Shape::new(&[1, num_slots, d], DType::F32),
    );
    let h = gb.add(h, inputs.pos_bias);
    let h = layer_norm(
        &mut gb,
        h,
        inputs.pos_norm_w,
        inputs.pos_norm_b,
        out_shape.clone(),
    );
    let h = rms_no_scale(
        &mut gb,
        h,
        inputs.ones,
        inputs.zero_beta,
        cfg.rms_norm_eps as f32,
    );
    let out = gb.mm(h, inputs.embed_proj_w);
    Ok(out)
}

pub fn build_unified_vision_graph(
    num_slots: usize,
    cfg: &GemmaVisionConfig,
) -> Result<ProjectionGraph> {
    let patch_dim = cfg.model_patch_size * cfg.model_patch_size * 3;
    let d = cfg.mm_embed_dim;
    let mut hir = HirModule::new("gemma_unified_vision");
    let patches = hir.input(
        "patches",
        Shape::new(&[1, num_slots, patch_dim], DType::F32),
    );
    let pos_bias = hir.input("pos_bias", Shape::new(&[1, num_slots, d], DType::F32));
    let patch_ln1_w = hir.param(
        "model.vision_embedder.patch_ln1.weight",
        Shape::new(&[patch_dim], DType::F32),
    );
    let patch_ln1_b = hir.param(
        "model.vision_embedder.patch_ln1.bias",
        Shape::new(&[patch_dim], DType::F32),
    );
    let patch_dense_w = hir.param(
        "model.vision_embedder.patch_dense.weight",
        Shape::new(&[patch_dim, d], DType::F32),
    );
    let patch_dense_b = hir.param(
        "model.vision_embedder.patch_dense.bias",
        Shape::new(&[d], DType::F32),
    );
    let patch_ln2_w = hir.param(
        "model.vision_embedder.patch_ln2.weight",
        Shape::new(&[d], DType::F32),
    );
    let patch_ln2_b = hir.param(
        "model.vision_embedder.patch_ln2.bias",
        Shape::new(&[d], DType::F32),
    );
    let pos_norm_w = hir.param(
        "model.vision_embedder.pos_norm.weight",
        Shape::new(&[d], DType::F32),
    );
    let pos_norm_b = hir.param(
        "model.vision_embedder.pos_norm.bias",
        Shape::new(&[d], DType::F32),
    );
    let embed_proj_w = hir.param(
        "model.embed_vision.embedding_projection.weight",
        Shape::new(&[d, d], DType::F32),
    );
    let ones = hir.param("unified.ones", Shape::new(&[d], DType::F32));
    let zero_beta = hir.param("unified.zero_beta", Shape::new(&[d], DType::F32));
    let inputs = UnifiedVisionInputs {
        patches,
        pos_bias,
        patch_ln1_w,
        patch_ln1_b,
        patch_dense_w,
        patch_dense_b,
        patch_ln2_w,
        patch_ln2_b,
        pos_norm_w,
        pos_norm_b,
        embed_proj_w,
        ones,
        zero_beta,
    };
    let output = build_unified_vision_hir(&mut hir, inputs, cfg, num_slots)?;
    hir.set_outputs(vec![output]);
    Ok(ProjectionGraph {
        hir,
        output,
        input_keys: vec!["patches".into(), "pos_bias".into()],
    })
}

#[derive(Debug, Clone, Copy)]
pub struct UnifiedAudioInputs {
    pub frames: HirNodeId,
    pub embed_proj_w: HirNodeId,
    pub ones: HirNodeId,
    pub zero_beta: HirNodeId,
}

pub fn build_unified_audio_hir(
    hir: &mut HirModule,
    inputs: UnifiedAudioInputs,
    cfg: &GemmaAudioConfig,
    lm_hidden: usize,
    num_frames: usize,
) -> Result<HirNodeId> {
    let mut gb = HirMut::new(hir);
    let h = rms_no_scale(
        &mut gb,
        inputs.frames,
        inputs.ones,
        inputs.zero_beta,
        cfg.rms_norm_eps as f32,
    );
    let out = gb.mm(h, inputs.embed_proj_w);
    let _ = (lm_hidden, num_frames);
    Ok(out)
}

pub fn build_unified_audio_graph(
    num_frames: usize,
    cfg: &GemmaAudioConfig,
    lm_hidden: usize,
) -> Result<ProjectionGraph> {
    let d = cfg.audio_embed_dim;
    let samples = cfg.audio_samples_per_token;
    let mut hir = HirModule::new("gemma_unified_audio");
    let frames = hir.input("frames", Shape::new(&[1, num_frames, samples], DType::F32));
    let embed_proj_w = hir.param(
        "model.embed_audio.embedding_projection.weight",
        Shape::new(&[d, lm_hidden], DType::F32),
    );
    let ones = hir.param("unified.audio.ones", Shape::new(&[d], DType::F32));
    let zero_beta = hir.param("unified.audio.zero_beta", Shape::new(&[d], DType::F32));
    let inputs = UnifiedAudioInputs {
        frames,
        embed_proj_w,
        ones,
        zero_beta,
    };
    let output = build_unified_audio_hir(&mut hir, inputs, cfg, lm_hidden, num_frames)?;
    hir.set_outputs(vec![output]);
    Ok(ProjectionGraph {
        hir,
        output,
        input_keys: vec!["frames".into()],
    })
}

pub fn is_unified_vision_weights(keys: impl IntoIterator<Item = impl AsRef<str>>) -> bool {
    keys.into_iter()
        .any(|k| k.as_ref().starts_with("model.vision_embedder."))
}
