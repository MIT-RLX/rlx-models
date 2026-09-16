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

//! Mixing transformer (host reference).

use crate::config::TimesFM3Config;
use crate::host::math::{
    add_attn_bias, attn_weighted, linear2, matmul_attn_scores, merge_heads, per_dim_scale,
    relu_array, rms_norm, softmax_last_axis4, split_heads,
};
use crate::rope::{apply_rope_tables, build_attn_bias, build_rope_tables, repeat_heads};
use crate::weights::{MhaWeight, MixingLayerWeight};
use ndarray::{Array2, Array4, ArrayView2, ArrayView3, ArrayView4, s};

pub struct MixingStack {
    layers: Vec<MixingLayerWeight>,
    cfg: TimesFM3Config,
}

impl MixingStack {
    pub fn new(cfg: TimesFM3Config, layers: Vec<MixingLayerWeight>) -> Self {
        Self { layers, cfg }
    }

    pub fn forward(&self, x: ArrayView4<f32>, patch_mask: ArrayView3<bool>) -> Array4<f32> {
        let mut h = x.to_owned();
        for layer in &self.layers {
            h = self.mixing_layer(layer, h.view(), patch_mask);
        }
        h
    }

    fn mixing_layer(
        &self,
        w: &MixingLayerWeight,
        x: ArrayView4<f32>,
        patch_mask: ArrayView3<bool>,
    ) -> Array4<f32> {
        let (b, v, n, d) = x.dim();
        let seq_in = flatten_bvnp(x.view(), d);
        let seq_ln = rms_norm(seq_in.view(), w.pre_seq_ln.view());
        let mask_flat = flatten_bv_mask(patch_mask);
        let seq_out = self.mha(&w.seq_attn, seq_ln.view(), b * v, n, true, &mask_flat);
        let seq_out = rms_norm(seq_out.view(), w.post_seq_ln.view());
        let h1 = add_residual_4d(x, unflatten_bvnp(seq_out.view(), b, v, n, d).view());

        // Variate attention
        let h2 = if let (Some(pre), Some(post), Some(var)) =
            (&w.pre_var_ln, &w.post_var_ln, &w.var_attn)
        {
            let var_in = permute_bvn_to_bnv(h1.view());
            let var_ln = rms_norm(var_in.view(), pre.view());
            let var_mask = flatten_var_mask(patch_mask);
            let var_out = self.mha(var, var_ln.view(), b * n, v, false, &var_mask);
            let var_out = rms_norm(var_out.view(), post.view());
            let var_4d = permute_bnv_to_bvn((&var_in + &var_out).view(), b, v, n, d);
            add_residual_4d(h1.view(), var_4d.view())
        } else {
            h1
        };

        // FFN — per patch.
        let ff_in = flatten_bvnp(h2.view(), d);
        let ff_ln = rms_norm(ff_in.view(), w.pre_ff_ln.view());
        let mut hidden = linear2(ff_ln.view(), w.ff0.view(), None);
        relu_array(&mut hidden);
        let ff_out = linear2(hidden.view(), w.ff1.view(), None);
        let ff_out = rms_norm(ff_out.view(), w.post_ff_ln.view());
        add_residual_4d(h2.view(), unflatten_bvnp(ff_out.view(), b, v, n, d).view())
    }

    fn mha(
        &self,
        w: &MhaWeight,
        x: ArrayView2<f32>,
        batch: usize,
        seq: usize,
        causal: bool,
        patch_mask: &Array2<bool>,
    ) -> Array2<f32> {
        let d = self.cfg.model_dims();
        let h = self.num_heads();
        let hd = self.head_dim();

        let q = linear2(x, w.query.view(), None);
        let k = linear2(x, w.key.view(), None);
        let v = linear2(x, w.value.view(), None);

        let mut q4 = split_heads(q.view(), batch, seq, h, hd);
        let mut k4 = split_heads(k.view(), batch, seq, h, hd);
        let v4 = split_heads(v.view(), batch, seq, h, hd);

        let mut positions = Array2::<i32>::zeros((batch, seq));
        for bi in 0..batch {
            let front: i32 = patch_mask.row(bi).iter().take_while(|&&m| m).count() as i32;
            for ni in 0..seq {
                positions[[bi, ni]] = ni as i32 - front;
            }
        }

        let (cos, sin) = build_rope_tables(&positions, hd);
        let (cos, sin) = repeat_heads(&cos, &sin, batch, seq, hd / 2, h);
        let mut q3 = ndarray::Array3::zeros((batch * h, seq, hd));
        let mut k3 = ndarray::Array3::zeros((batch * h, seq, hd));
        for bi in 0..batch {
            for hi in 0..h {
                for si in 0..seq {
                    for di in 0..hd {
                        q3[[bi * h + hi, si, di]] = q4[[bi, si, hi, di]];
                        k3[[bi * h + hi, si, di]] = k4[[bi, si, hi, di]];
                    }
                }
            }
        }
        q3 = apply_rope_tables(q3.view(), &cos, &sin, hd);
        k3 = apply_rope_tables(k3.view(), &cos, &sin, hd);
        for bi in 0..batch {
            for hi in 0..h {
                for si in 0..seq {
                    for di in 0..hd {
                        q4[[bi, si, hi, di]] = q3[[bi * h + hi, si, di]];
                        k4[[bi, si, hi, di]] = k3[[bi * h + hi, si, di]];
                    }
                }
            }
        }

        // QK norm + per-dim scale on flattened head slices
        let mut q_flat = merge_heads(q4.view(), d);
        let mut k_flat = merge_heads(k4.view(), d);
        for bi in 0..batch * seq {
            let mut q_row = q_flat.slice_mut(s![bi, ..]);
            let mut k_row = k_flat.slice_mut(s![bi, ..]);
            let qr = q_row.to_owned().into_shape_with_order((h, hd)).unwrap();
            let kr = k_row.to_owned().into_shape_with_order((h, hd)).unwrap();
            for hi in 0..h {
                let mut q_h = qr
                    .slice(s![hi, ..])
                    .to_owned()
                    .insert_axis(ndarray::Axis(0));
                let mut k_h = kr
                    .slice(s![hi, ..])
                    .to_owned()
                    .insert_axis(ndarray::Axis(0));
                q_h = rms_norm(q_h.view(), w.query_ln.view());
                k_h = rms_norm(k_h.view(), w.key_ln.view());
                q_row
                    .slice_mut(s![hi * hd..(hi + 1) * hd])
                    .assign(&q_h.row(0));
                k_row
                    .slice_mut(s![hi * hd..(hi + 1) * hd])
                    .assign(&k_h.row(0));
            }
        }
        q4 = split_heads(q_flat.view(), batch, seq, h, hd);
        let k4 = split_heads(k_flat.view(), batch, seq, h, hd);

        let q4 = per_dim_scale(q4.view(), w.per_dim_scale.view(), hd);

        // (b, h, s, d)
        let q_t = q4.permuted_axes([0, 2, 1, 3]);
        let k_t = k4.permuted_axes([0, 2, 1, 3]);
        let v_t = v4.permuted_axes([0, 2, 1, 3]);

        let mut scores = matmul_attn_scores(q_t.view(), k_t.view(), 1.0);
        let bias = build_attn_bias(patch_mask, seq, causal);
        add_attn_bias(&mut scores, &bias, batch, seq);
        softmax_last_axis4(&mut scores);
        let ctx = attn_weighted(v_t.view(), scores.view());

        let ctx_bn = ctx.permuted_axes([0, 2, 1, 3]);
        let merged = merge_heads(ctx_bn.view(), d);
        linear2(merged.view(), w.out.view(), None)
    }

    fn num_heads(&self) -> usize {
        self.cfg.num_heads()
    }

    fn head_dim(&self) -> usize {
        self.cfg.head_dim()
    }
}

fn flatten_bvnp(x: ArrayView4<f32>, d: usize) -> Array2<f32> {
    let (b, v, n, _) = x.dim();
    let mut out = Array2::zeros((b * v * n, d));
    for bi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                for di in 0..d {
                    out[[bi * v * n + vi * n + ni, di]] = x[[bi, vi, ni, di]];
                }
            }
        }
    }
    out
}

fn unflatten_bvnp(x: ArrayView2<f32>, b: usize, v: usize, n: usize, d: usize) -> Array4<f32> {
    let mut out = Array4::zeros((b, v, n, d));
    for bi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                for di in 0..d {
                    out[[bi, vi, ni, di]] = x[[bi * v * n + vi * n + ni, di]];
                }
            }
        }
    }
    out
}

fn flatten_bv_mask(mask: ArrayView3<bool>) -> Array2<bool> {
    let (b, v, n) = mask.dim();
    let mut out = Array2::from_elem((b * v, n), false);
    for bi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                out[[bi * v + vi, ni]] = mask[[bi, vi, ni]];
            }
        }
    }
    out
}

fn flatten_var_mask(mask: ArrayView3<bool>) -> Array2<bool> {
    let (b, v, n) = mask.dim();
    let mut out = Array2::from_elem((b * n, v), false);
    for bi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                out[[bi * n + ni, vi]] = mask[[bi, vi, ni]];
            }
        }
    }
    out
}

fn permute_bvn_to_bnv(x: ArrayView4<f32>) -> Array2<f32> {
    let (b, v, n, d) = x.dim();
    let mut out = Array2::zeros((b * n * v, d));
    for bi in 0..b {
        for ni in 0..n {
            for vi in 0..v {
                for di in 0..d {
                    out[[bi * n * v + ni * v + vi, di]] = x[[bi, vi, ni, di]];
                }
            }
        }
    }
    out
}

fn permute_bnv_to_bvn(x: ArrayView2<f32>, b: usize, v: usize, n: usize, d: usize) -> Array4<f32> {
    let mut out = Array4::zeros((b, v, n, d));
    for bi in 0..b {
        for ni in 0..n {
            for vi in 0..v {
                for di in 0..d {
                    out[[bi, vi, ni, di]] = x[[bi * n * v + ni * v + vi, di]];
                }
            }
        }
    }
    out
}

fn add_residual_4d(base: ArrayView4<f32>, delta: ArrayView4<f32>) -> Array4<f32> {
    &base + &delta
}
