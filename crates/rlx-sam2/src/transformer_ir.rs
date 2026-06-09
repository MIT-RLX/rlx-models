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

//! Compile SAM2 two-way transformer to IR.

use super::transformer::{
    Sam2AttentionWeights, Sam2TwoWayAttentionBlockWeights, Sam2TwoWayTransformerWeights,
};
use rlx_sam_ir::twoway_transformer_ir::{AttentionSpec, TwoWayBlockSpec, TwoWayTransformerSpec};

use anyhow::Result;
use rlx_flow::CompileProfile;
use rlx_runtime::Device;
pub use rlx_sam_ir::twoway_transformer_ir::TwoWayTransformerCompiled;

fn attn_spec(w: &Sam2AttentionWeights) -> AttentionSpec {
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

fn block_spec(w: &Sam2TwoWayAttentionBlockWeights) -> TwoWayBlockSpec {
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

pub fn transformer_spec(w: &Sam2TwoWayTransformerWeights) -> TwoWayTransformerSpec {
    TwoWayTransformerSpec {
        layers: w.layers.iter().map(block_spec).collect(),
        final_attn: attn_spec(&w.final_attn_token_to_image),
        norm_final_g: w.norm_final_g.clone(),
        norm_final_b: w.norm_final_b.clone(),
        embed_dim: w.embed_dim,
    }
}

/// `base_q_n`: output tokens without sparse prompts; +[`rlx_sam_ir::twoway_transformer_ir::MAX_SPARSE_PROMPT_TOKENS`] padded slots.
pub fn compile_two_way_transformer(
    w: &Sam2TwoWayTransformerWeights,
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
    w: &Sam2TwoWayTransformerWeights,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::{
        Sam2AttentionWeights, Sam2TwoWayAttentionBlockWeights, Sam2TwoWayTransformerWeights,
        two_way_transformer_forward,
    };

    fn synth_attn(e: usize, down: usize) -> Sam2AttentionWeights {
        let id = e / down;
        let nh = 8;
        Sam2AttentionWeights {
            q_w: vec![0.01; id * e],
            q_b: vec![0.0; id],
            k_w: vec![0.02; id * e],
            k_b: vec![0.0; id],
            v_w: vec![0.03; id * e],
            v_b: vec![0.0; id],
            out_w: vec![0.04; e * id],
            out_b: vec![0.0; e],
            num_heads: nh,
            embed_dim: e,
            internal_dim: id,
        }
    }

    fn synth_block(e: usize) -> Sam2TwoWayAttentionBlockWeights {
        Sam2TwoWayAttentionBlockWeights {
            self_attn: synth_attn(e, 1),
            norm1_g: vec![1.0; e],
            norm1_b: vec![0.0; e],
            cross_token_to_image: synth_attn(e, 2),
            norm2_g: vec![1.0; e],
            norm2_b: vec![0.0; e],
            mlp_lin1_w: vec![0.01; 2048 * e],
            mlp_lin1_b: vec![0.0; 2048],
            mlp_lin2_w: vec![0.02; e * 2048],
            mlp_lin2_b: vec![0.0; e],
            norm3_g: vec![1.0; e],
            norm3_b: vec![0.0; e],
            cross_image_to_token: synth_attn(e, 2),
            norm4_g: vec![1.0; e],
            norm4_b: vec![0.0; e],
            skip_first_layer_pe: true,
        }
    }

    fn synth_transformer(e: usize) -> Sam2TwoWayTransformerWeights {
        Sam2TwoWayTransformerWeights {
            layers: vec![synth_block(e)],
            final_attn_token_to_image: synth_attn(e, 2),
            norm_final_g: vec![1.0; e],
            norm_final_b: vec![0.0; e],
            embed_dim: e,
        }
    }

    fn assert_two_way_parity(
        w: &Sam2TwoWayTransformerWeights,
        tokens: &[f32],
        image: &[f32],
        image_pe: &[f32],
        e: usize,
        g_grid: usize,
        q_n: usize,
        compile_sparse: Option<usize>,
    ) {
        let k_n = g_grid * g_grid;
        let host =
            two_way_transformer_forward(w, image, image_pe, tokens, 1, e, g_grid, g_grid, q_n);
        let spec = transformer_spec(w);
        let mut ir = match compile_sparse {
            Some(base_q) => TwoWayTransformerCompiled::compile_with_sparse_slots(
                &spec,
                base_q,
                k_n,
                Device::Cpu,
            )
            .unwrap(),
            None => TwoWayTransformerCompiled::compile(&spec, q_n, k_n, Device::Cpu).unwrap(),
        };
        let got = if compile_sparse.is_some() {
            ir.run_nchw_masked(tokens, q_n, image, image_pe, g_grid)
                .unwrap()
        } else {
            ir.run_nchw(tokens, image, image_pe, g_grid).unwrap()
        };
        let fd_q = host
            .0
            .iter()
            .zip(&got.0)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let fd_k = host
            .1
            .iter()
            .zip(&got.1)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(fd_q < 5e-3, "two-way queries max |Δ| = {fd_q:.3e}");
        assert!(fd_k < 5e-3, "two-way keys max |Δ| = {fd_k:.3e}");
    }

    #[test]
    fn two_way_ir_matches_host_small_grid() {
        let e = 256usize;
        let g_grid = 8usize;
        let q_n = 6usize;
        let w = synth_transformer(e);
        let tokens: Vec<f32> = (0..q_n * e).map(|i| (i as f32) * 0.001).collect();
        let image: Vec<f32> = (0..e * g_grid * g_grid)
            .map(|i| (i as f32) * 0.0001)
            .collect();
        let image_pe = image.clone();
        assert_two_way_parity(&w, &tokens, &image, &image_pe, e, g_grid, q_n, None);
    }

    #[test]
    fn two_way_masked_sparse_slots_match_host() {
        let e = 256usize;
        let g_grid = 8usize;
        let base_q = 5usize;
        let active_q = 7usize;
        let w = synth_transformer(e);
        let tokens: Vec<f32> = (0..active_q * e).map(|i| (i as f32) * 0.001).collect();
        let image: Vec<f32> = (0..e * g_grid * g_grid)
            .map(|i| (i as f32) * 0.0001)
            .collect();
        let image_pe = image.clone();
        assert_two_way_parity(
            &w,
            &tokens,
            &image,
            &image_pe,
            e,
            g_grid,
            active_q,
            Some(base_q),
        );
    }

    #[test]
    fn two_way_ir_matches_host_sam2_grid() {
        let e = 256usize;
        let g_grid = 32usize;
        let q_n = 8usize;
        let w = synth_transformer(e);
        let tokens: Vec<f32> = (0..q_n * e).map(|i| (i as f32) * 0.001).collect();
        let image: Vec<f32> = (0..e * g_grid * g_grid)
            .map(|i| (i as f32) * 0.0001)
            .collect();
        let image_pe = image.clone();
        assert_two_way_parity(&w, &tokens, &image, &image_pe, e, g_grid, q_n, None);
    }
}
