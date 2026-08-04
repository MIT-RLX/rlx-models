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

//! DaViT (Dual-Attention Vision Transformer) backbone + image projection.
//!
//! Mirrors `DaViT.forward_features_unpool` and `_encode_image` in
//! `modeling_florence2.py`. The forward keeps activations as `[B, N, C]`
//! token tensors between blocks and converts to `[B, C, H, W]` only inside
//! the depthwise convolutions and patch-embed stems.

use crate::builder::{Florence2Builder, conv_out};
use crate::config::Florence2Config;
use crate::weights::vision as vk;
use anyhow::{Result, ensure};
use rlx_core::vision_ops_ir::conv2d_bias_groups;
use rlx_ir::hir::{HirGraphExt, HirNodeId};
use rlx_ir::op::MaskKind;
use rlx_ir::{Shape, ops::attention::attention_kind_op};

impl Florence2Builder<'_> {
    /// Full image encoder: `pixel_values [B,3,H,W]` → `image_features
    /// [B, image_seq, projection_dim]` (577 × 1024 for 768² inputs).
    pub(crate) fn emit_vision(
        &mut self,
        cfg: &Florence2Config,
        pixel: HirNodeId,
        img_h: usize,
        img_w: usize,
    ) -> Result<HirNodeId> {
        let v = &cfg.vision;
        let mut h = img_h;
        let mut w = img_w;
        let mut cur_c = 3usize;
        let mut tokens: Option<HirNodeId> = None;

        for stage in 0..v.num_stages() {
            let k = v.patch_size[stage];
            let s = v.patch_stride[stage];
            let p = v.patch_padding[stage];
            let out_c = v.dim_embed[stage];
            let prenorm = v.patch_prenorm[stage];

            // ConvEmbed input as NCHW.
            let conv_in = if stage == 0 {
                pixel
            } else {
                let mut t = tokens.unwrap();
                if prenorm {
                    t = self.layer_norm(t, &vk::conv_norm_w(stage), &vk::conv_norm_b(stage))?;
                }
                self.bnc_to_nchw(t, h, w, cur_c)
            };

            let out_h = conv_out(h, k, s, p);
            let out_w = conv_out(w, k, s, p);
            let cw = self.load_param(&vk::conv_proj_w(stage), false)?;
            let cb = self.load_param(&vk::conv_proj_b(stage), false)?;
            let batch = self.batch;
            let y = conv2d_bias_groups(
                &mut self.g(),
                conv_in,
                cw,
                cb,
                batch,
                out_c,
                k,
                k,
                [s, s],
                [p, p],
                1,
                out_h,
                out_w,
            );
            let mut toks = self.nchw_to_bnc(y, out_c, out_h, out_w);
            if !prenorm {
                toks = self.layer_norm(toks, &vk::conv_norm_w(stage), &vk::conv_norm_b(stage))?;
            }

            h = out_h;
            w = out_w;
            cur_c = out_c;

            let ws = v.window_size;
            let stop_stage = std::env::var("RLX_FLORENCE2_VISION_STOP_STAGE")
                .ok()
                .and_then(|s| s.parse::<usize>().ok());
            let stop_depth = std::env::var("RLX_FLORENCE2_VISION_STOP_DEPTH")
                .ok()
                .and_then(|s| s.parse::<usize>().ok());
            let stop_phase = std::env::var("RLX_FLORENCE2_VISION_STOP_PHASE").ok();
            for depth in 0..v.depths[stage] {
                toks =
                    self.spatial_block(stage, depth, toks, h, w, cur_c, v.num_heads[stage], ws)?;
                if stop_stage == Some(stage)
                    && stop_depth == Some(depth)
                    && stop_phase.as_deref() == Some("spatial")
                {
                    return Ok(toks);
                }
                toks = self.channel_block(stage, depth, toks, h, w, cur_c, v.num_groups[stage])?;
                if stop_stage == Some(stage)
                    && stop_depth == Some(depth)
                    && stop_phase.as_deref() == Some("channel")
                {
                    return Ok(toks);
                }
            }
            tokens = Some(toks);
            // Debug bisection hook: emit the post-stage token tensor.
            if stop_stage == Some(stage) && stop_depth.is_none() {
                return Ok(toks);
            }
        }

        let x = tokens.unwrap();
        // Positional + temporal embeddings (precomputed host-side bias).
        let x = self.add_vision_pos_embed(cfg, x, h, w, cur_c)?;
        // Pool + project to the language model dimension.
        self.image_projection(cfg, x, h * w, cur_c)
    }

    fn spatial_block(
        &mut self,
        stage: usize,
        depth: usize,
        x: HirNodeId,
        h: usize,
        w: usize,
        c: usize,
        num_heads: usize,
        ws: usize,
    ) -> Result<HirNodeId> {
        let block = vk::block(stage, depth, "spatial_block");
        // conv1 (depthwise) residual
        let dw1 = self.depthwise_conv(&block, "conv1", x, h, w, c)?;
        let mut x = self.g().add(x, dw1);
        // window attention residual
        let ln = self.layer_norm(
            x,
            &vk::attn_norm_w(&block, "window_attn"),
            &vk::attn_norm_b(&block, "window_attn"),
        )?;
        let attn = self.window_attention(&block, ln, h, w, c, num_heads, ws)?;
        x = self.g().add(x, attn);
        // conv2 (depthwise) residual
        let dw2 = self.depthwise_conv(&block, "conv2", x, h, w, c)?;
        x = self.g().add(x, dw2);
        // ffn residual
        let ln2 = self.layer_norm(x, &vk::ffn_norm_w(&block), &vk::ffn_norm_b(&block))?;
        let mlp = self.vision_mlp(&block, ln2)?;
        Ok(self.g().add(x, mlp))
    }

    fn channel_block(
        &mut self,
        stage: usize,
        depth: usize,
        x: HirNodeId,
        h: usize,
        w: usize,
        c: usize,
        groups: usize,
    ) -> Result<HirNodeId> {
        let block = vk::block(stage, depth, "channel_block");
        let dw1 = self.depthwise_conv(&block, "conv1", x, h, w, c)?;
        let mut x = self.g().add(x, dw1);
        let ln = self.layer_norm(
            x,
            &vk::attn_norm_w(&block, "channel_attn"),
            &vk::attn_norm_b(&block, "channel_attn"),
        )?;
        let attn = self.channel_attention(&block, ln, h * w, c, groups)?;
        x = self.g().add(x, attn);
        let dw2 = self.depthwise_conv(&block, "conv2", x, h, w, c)?;
        x = self.g().add(x, dw2);
        let ln2 = self.layer_norm(x, &vk::ffn_norm_w(&block), &vk::ffn_norm_b(&block))?;
        let mlp = self.vision_mlp(&block, ln2)?;
        Ok(self.g().add(x, mlp))
    }

    /// Depthwise 3×3 conv (stride 1, pad 1, groups = channels), token in/out.
    fn depthwise_conv(
        &mut self,
        block: &str,
        conv: &str,
        x: HirNodeId,
        h: usize,
        w: usize,
        c: usize,
    ) -> Result<HirNodeId> {
        let nchw = self.bnc_to_nchw(x, h, w, c);
        let cw = self.load_param(&vk::dw_w(block, conv), false)?;
        let cb = self.load_param(&vk::dw_b(block, conv), false)?;
        let batch = self.batch;
        let y = conv2d_bias_groups(
            &mut self.g(),
            nchw,
            cw,
            cb,
            batch,
            c,
            3,
            3,
            [1, 1],
            [1, 1],
            c,
            h,
            w,
        );
        Ok(self.nchw_to_bnc(y, c, h, w))
    }

    fn vision_mlp(&mut self, block: &str, x: HirNodeId) -> Result<HirNodeId> {
        let h1 = self.linear(x, &vk::ffn_fc1_w(block), Some(&vk::ffn_fc1_b(block)))?;
        let h1 = self.g().gelu(h1);
        self.linear(h1, &vk::ffn_fc2_w(block), Some(&vk::ffn_fc2_b(block)))
    }

    /// Window multi-head self-attention. Assumes `H`,`W` divisible by the
    /// window size (true for the 768² → 24² Florence-2 feature maps).
    fn window_attention(
        &mut self,
        block: &str,
        x: HirNodeId,
        h: usize,
        w: usize,
        c: usize,
        num_heads: usize,
        ws: usize,
    ) -> Result<HirNodeId> {
        ensure!(
            h.is_multiple_of(ws) && w.is_multiple_of(ws),
            "window attention needs H,W divisible by {ws} (got {h}x{w})"
        );
        let b = self.batch as i64;
        let nh = (h / ws) as i64;
        let nw = (w / ws) as i64;
        let m = (ws * ws) as i64;
        let cc = c as i64;
        let bw = (self.batch * (h / ws) * (w / ws)) as i64;

        // partition: [B,N,C] -> windows [B*nWin, ws*ws, C]
        let x4 = self.g().reshape_(x, vec![b, h as i64, w as i64, cc]);
        let x6 = self
            .g()
            .reshape_(x4, vec![b, nh, ws as i64, nw, ws as i64, cc]);
        let x6p = self.g().transpose_(x6, vec![0, 1, 3, 2, 4, 5]);
        let win = self.g().reshape_(x6p, vec![bw, m, cc]);

        // qkv
        let qkv = self.linear(
            win,
            &vk::attn_qkv_w(block, "window_attn"),
            Some(&vk::attn_qkv_b(block, "window_attn")),
        )?;
        let q = self.g().narrow_(qkv, 2, 0, c);
        let k = self.g().narrow_(qkv, 2, c, c);
        let vv = self.g().narrow_(qkv, 2, 2 * c, c);
        let head_dim = c / num_heads;
        let out_shape = Shape::new(&[bw as usize, (ws * ws), c], self.f);
        let attn = self.g().add_node(
            attention_kind_op(num_heads, head_dim, None, MaskKind::None, None, None),
            vec![q, k, vv],
            out_shape,
        );
        let proj = self.linear(
            attn,
            &vk::attn_proj_w(block, "window_attn"),
            Some(&vk::attn_proj_b(block, "window_attn")),
        )?;

        // reverse: windows -> [B,N,C]
        let w6 = self
            .g()
            .reshape_(proj, vec![b, nh, nw, ws as i64, ws as i64, cc]);
        let w6p = self.g().transpose_(w6, vec![0, 1, 3, 2, 4, 5]);
        Ok(self.g().reshape_(w6p, vec![b, (h * w) as i64, cc]))
    }

    /// Channel (transposed) multi-head attention over grouped channels.
    fn channel_attention(
        &mut self,
        block: &str,
        x: HirNodeId,
        n: usize,
        c: usize,
        groups: usize,
    ) -> Result<HirNodeId> {
        let b = self.batch as i64;
        let nn = n as i64;
        let cc = c as i64;
        let g = groups as i64;
        let cg = (c / groups) as i64;

        let qkv = self.linear(
            x,
            &vk::attn_qkv_w(block, "channel_attn"),
            Some(&vk::attn_qkv_b(block, "channel_attn")),
        )?;
        // [B,N,3,groups,Cg] -> permute(2,0,3,1,4) -> [3,B,groups,N,Cg]
        let qkv5 = self.g().reshape_(qkv, vec![b, nn, 3, g, cg]);
        let qkv5p = self.g().transpose_(qkv5, vec![2, 0, 3, 1, 4]);
        // Fold (B·groups) into a single batch dim — the channel attention is a
        // batched matmul over [B·groups], kept 3-D for portable GPU coverage.
        let bg = b * g;
        let q = self.slice_first_3d(qkv5p, 0, bg, nn, cg);
        let k = self.slice_first_3d(qkv5p, 1, bg, nn, cg);
        let vv = self.slice_first_3d(qkv5p, 2, bg, nn, cg);

        // q *= N^-0.5
        let scale = (n as f32).powf(-0.5);
        let scale_key = format!("{block}.channel_attn.scale");
        let scale_p = self.register_param(&scale_key, vec![scale], &[1])?;
        let q = self.g().mul(q, scale_p);

        // attention = q.transpose(-1,-2) @ k -> [Bg,Cg,Cg]
        let q_t = self.g().transpose_(q, vec![0, 2, 1]);
        let attn = self.g().mm(q_t, k);
        let attn = self.g().sm(attn, -1);
        // (attn @ v.transpose(-1,-2)).transpose(-1,-2) -> [Bg,N,Cg]
        let v_t = self.g().transpose_(vv, vec![0, 2, 1]);
        let out = self.g().mm(attn, v_t);
        let out = self.g().transpose_(out, vec![0, 2, 1]); // [Bg,N,Cg]
        // Merge heads → [B,N,C] (group-major): [B,groups,N,Cg] → [B,N,groups,Cg].
        let out4 = self.g().reshape_(out, vec![b, g, nn, cg]);
        let out4 = self.g().transpose_(out4, vec![0, 2, 1, 3]);
        let out = self.g().reshape_(out4, vec![b, nn, cc]);
        let _ = groups;
        self.linear(
            out,
            &vk::attn_proj_w(block, "channel_attn"),
            Some(&vk::attn_proj_b(block, "channel_attn")),
        )
    }

    /// Pick index `i` along a leading axis-0 of size 3 from `[3,B,groups,N,Cg]`
    /// and flatten `(B·groups)` into one batch dim → `[B·groups, N, Cg]`.
    fn slice_first_3d(&mut self, x: HirNodeId, i: usize, bg: i64, n: i64, cg: i64) -> HirNodeId {
        let sl = self.g().narrow_(x, 0, i, 1);
        self.g().reshape_(sl, vec![bg, n, cg])
    }

    /// Add the (precomputed) 2D position + temporal cosine embeddings.
    fn add_vision_pos_embed(
        &mut self,
        cfg: &Florence2Config,
        x: HirNodeId,
        h: usize,
        w: usize,
        c: usize,
    ) -> Result<HirNodeId> {
        let bias = self.vision_pos_bias(cfg, h, w, c)?;
        let n = (h * w) as i64;
        let bias3 = self.g().reshape_(bias, vec![1, n, c as i64]);
        Ok(self.g().add(x, bias3))
    }

    /// Compute `pos2d[i,j] = concat(col_emb[j], row_emb[i])` plus
    /// `temporal[0]`, flattened to `[N, C]`, host-side from the embedding
    /// tables in the checkpoint.
    fn vision_pos_bias(
        &mut self,
        _cfg: &Florence2Config,
        h: usize,
        w: usize,
        c: usize,
    ) -> Result<HirNodeId> {
        let (row, _row_shape) = self.weights.take(vk::POS_ROW, false)?; // [50, C/2]
        let (col, _col_shape) = self.weights.take(vk::POS_COL, false)?; // [50, C/2]
        let (temporal, _t_shape) = self.weights.take(vk::TEMPORAL, false)?; // [100, C]
        let half = c / 2;
        let mut bias = vec![0f32; h * w * c];
        for i in 0..h {
            for j in 0..w {
                let base = (i * w + j) * c;
                // first half: column embedding indexed by width j
                for d in 0..half {
                    bias[base + d] = col[j * half + d] + temporal[d];
                }
                // second half: row embedding indexed by height i
                for d in 0..half {
                    bias[base + half + d] = row[i * half + d] + temporal[half + d];
                }
            }
        }
        self.register_param("vision.pos_bias", bias, &[h * w, c])
    }

    /// `spatial_avg_pool ++ temporal_avg_pool` → `@ image_projection` →
    /// `image_proj_norm`.
    fn image_projection(
        &mut self,
        _cfg: &Florence2Config,
        x: HirNodeId,
        n: usize,
        _c: usize,
    ) -> Result<HirNodeId> {
        // spatial_avg_pool: mean over the N tokens, keepdim -> [B,1,C]
        let spatial = self.g().mean(x, vec![1], true);
        // temporal_avg_pool with T=1 is identity -> [B,N,C]
        // concat([spatial, x], dim=1) -> [B, 1+N, C]
        let cat = self.g().concat_(vec![spatial, x], 1);
        let _ = n;
        // image_projection is a bare [C, projection_dim] parameter (in,out).
        let proj_w = self.load_param(vk::IMAGE_PROJECTION, false)?;
        let projected = self.g().mm(cat, proj_w);
        self.layer_norm(projected, vk::IMAGE_PROJ_NORM_W, vk::IMAGE_PROJ_NORM_B)
    }
}
