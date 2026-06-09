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

//! Compile SAM v1 two-way transformer to IR.

use super::transformer::{AttentionWeights, TwoWayAttentionBlockWeights, TwoWayTransformerWeights};
use anyhow::Result;
use rlx_flow::CompileProfile;
use rlx_runtime::Device;
use rlx_sam_ir::twoway_transformer_ir::TwoWayTransformerCompiled;
use rlx_sam_ir::twoway_transformer_ir::{AttentionSpec, TwoWayBlockSpec, TwoWayTransformerSpec};

fn attn_spec(w: &AttentionWeights) -> AttentionSpec {
    AttentionSpec {
        q_w: w.q_w.clone(),
        q_b: w.q_b.clone(),
        k_w: w.k_w.clone(),
        k_b: w.k_b.clone(),
        v_w: w.v_w.clone(),
        v_b: w.v_b.clone(),
        out_w: w.out_w.clone(),
        out_b: w.out_b.clone(),
        num_heads: w.num_heads,
        embed_dim: w.embed_dim,
        internal_dim: w.internal_dim,
    }
}

fn block_spec(w: &TwoWayAttentionBlockWeights) -> TwoWayBlockSpec {
    TwoWayBlockSpec {
        self_attn: attn_spec(&w.self_attn),
        norm1_g: w.norm1_g.clone(),
        norm1_b: w.norm1_b.clone(),
        cross_token_to_image: attn_spec(&w.cross_token_to_image),
        norm2_g: w.norm2_g.clone(),
        norm2_b: w.norm2_b.clone(),
        mlp_lin1_w: w.mlp_lin1_w.clone(),
        mlp_lin1_b: w.mlp_lin1_b.clone(),
        mlp_lin2_w: w.mlp_lin2_w.clone(),
        mlp_lin2_b: w.mlp_lin2_b.clone(),
        norm3_g: w.norm3_g.clone(),
        norm3_b: w.norm3_b.clone(),
        cross_image_to_token: attn_spec(&w.cross_image_to_token),
        norm4_g: w.norm4_g.clone(),
        norm4_b: w.norm4_b.clone(),
        skip_first_layer_pe: w.skip_first_layer_pe,
    }
}

pub fn transformer_spec(w: &TwoWayTransformerWeights) -> TwoWayTransformerSpec {
    TwoWayTransformerSpec {
        layers: w.layers.iter().map(block_spec).collect(),
        final_attn: attn_spec(&w.final_attn_token_to_image),
        norm_final_g: w.norm_final_g.clone(),
        norm_final_b: w.norm_final_b.clone(),
        embed_dim: w.embed_dim,
    }
}

/// Compile with padded slots for sparse prompts ([`rlx_sam_ir::twoway_transformer_ir::MAX_SPARSE_PROMPT_TOKENS`]).
pub fn compile_two_way_transformer(
    w: &TwoWayTransformerWeights,
    base_q_n: usize,
    grid: usize,
    device: Device,
) -> Result<TwoWayTransformerCompiled> {
    compile_two_way_transformer_with_profile(
        w,
        base_q_n,
        grid,
        device,
        &CompileProfile::sam_encoder(),
    )
}

pub fn compile_two_way_transformer_with_profile(
    w: &TwoWayTransformerWeights,
    base_q_n: usize,
    grid: usize,
    device: Device,
    profile: &CompileProfile,
) -> Result<TwoWayTransformerCompiled> {
    TwoWayTransformerCompiled::compile_with_sparse_slots_profile(
        &transformer_spec(w),
        base_q_n,
        grid * grid,
        device,
        profile,
    )
}
