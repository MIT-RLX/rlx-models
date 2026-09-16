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

use std::collections::HashMap;

use anyhow::Result;
use ndarray::{Array1, Array2};
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::op::{Activation, MaskKind};
use rlx_ir::ops::attention::attention_kind_op;
use rlx_ir::{DType, Shape};

use crate::config::TimesFM3Config;
use crate::host::math::{RMS_EPS, per_dim_factors};
use crate::weights::{MhaWeight, MixingLayerWeight, TimesFM3Weights};

const F32: DType = DType::F32;

/// Static batch / variate / patch counts baked into a compiled core graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoreGraphDims {
    pub b: usize,
    pub v: usize,
    pub n: usize,
}

impl CoreGraphDims {
    pub fn key(self) -> u64 {
        ((self.b as u64) << 32) | ((self.v as u64) << 16) | (self.n as u64)
    }
}

/// Build the compiled core: resblock → mixing stack → output head.
pub fn build_core_hir(
    cfg: &TimesFM3Config,
    weights: &TimesFM3Weights,
    dims: CoreGraphDims,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    let mut hir = HirModule::new("timesfm3_core");
    let mut params = HashMap::new();
    let mut b = Builder {
        cfg,
        dims,
        params: &mut params,
    };

    let in_dim = cfg.resblock_input_dim();
    let rows = dims.b * dims.v * dims.n;
    let hd = cfg.head_dim();
    let nh = cfg.num_heads();

    let res_in = hir.input("res_in", Shape::new(&[rows, in_dim], F32));
    let rope_cos_seq = hir.input(
        "rope_cos_seq",
        Shape::new(&[dims.b * dims.v * nh, dims.n, hd / 2], F32),
    );
    let rope_sin_seq = hir.input(
        "rope_sin_seq",
        Shape::new(&[dims.b * dims.v * nh, dims.n, hd / 2], F32),
    );
    let attn_bias_seq = hir.input(
        "attn_bias_seq",
        Shape::new(&[dims.b * dims.v, dims.n, dims.n], F32),
    );
    let (rope_cos_var, rope_sin_var, attn_bias_var) = if cfg.use_variate_attention {
        (
            Some(hir.input(
                "rope_cos_var",
                Shape::new(&[dims.b * dims.n * nh, dims.v, hd / 2], F32),
            )),
            Some(hir.input(
                "rope_sin_var",
                Shape::new(&[dims.b * dims.n * nh, dims.v, hd / 2], F32),
            )),
            Some(hir.input(
                "attn_bias_var",
                Shape::new(&[dims.b * dims.n, dims.v, dims.v], F32),
            )),
        )
    } else {
        (None, None, None)
    };

    let mut h = b.resblock(&mut hir, res_in, &weights.resblock)?;
    for (i, layer) in weights.layers.iter().enumerate() {
        h = b.mixing_layer(
            &mut hir,
            i,
            layer,
            h,
            rope_cos_seq,
            rope_sin_seq,
            attn_bias_seq,
            rope_cos_var,
            rope_sin_var,
            attn_bias_var,
        )?;
    }

    let logits = b.output_head(
        &mut hir,
        h,
        &weights.output_head.w,
        weights.output_head.b.as_ref(),
    )?;
    hir.outputs = vec![logits];
    Ok((hir, params))
}

/// Resblock-only graph for dev parity isolation.
#[cfg(feature = "dev")]
pub fn build_resblock_hir(
    cfg: &TimesFM3Config,
    weights: &TimesFM3Weights,
    dims: CoreGraphDims,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    let mut hir = HirModule::new("timesfm3_resblock");
    let mut params = HashMap::new();
    let mut b = Builder {
        cfg,
        dims,
        params: &mut params,
    };
    let in_dim = cfg.resblock_input_dim();
    let rows = dims.b * dims.v * dims.n;
    let res_in = hir.input("res_in", Shape::new(&[rows, in_dim], F32));
    let out = b.resblock(&mut hir, res_in, &weights.resblock)?;
    hir.outputs = vec![out];
    Ok((hir, params))
}

struct Builder<'a> {
    cfg: &'a TimesFM3Config,
    dims: CoreGraphDims,
    params: &'a mut HashMap<String, Vec<f32>>,
}

impl Builder<'_> {
    fn rows(&self) -> usize {
        self.dims.b * self.dims.v * self.dims.n
    }

    fn d(&self) -> usize {
        self.cfg.model_dims()
    }

    fn hd(&self) -> usize {
        self.cfg.head_dim()
    }

    fn nh(&self) -> usize {
        self.cfg.num_heads()
    }

    fn g<'a>(&mut self, hir: &'a mut HirModule) -> HirMut<'a> {
        HirMut::new(hir)
    }

    fn register(
        &mut self,
        hir: &mut HirModule,
        key: &str,
        data: Vec<f32>,
        shape: &[usize],
    ) -> HirNodeId {
        let id = hir.param(key, Shape::new(shape, F32));
        self.params.insert(key.to_string(), data);
        id
    }

    fn linear_w(&mut self, hir: &mut HirModule, key: &str, w: &Array2<f32>) -> HirNodeId {
        let (out, inp) = w.dim();
        let mut data = vec![0.0f32; out * inp];
        for o in 0..out {
            for i in 0..inp {
                data[i * out + o] = w[[o, i]];
            }
        }
        self.register(hir, key, data, &[inp, out])
    }

    fn rms_norm(
        &mut self,
        hir: &mut HirModule,
        x: HirNodeId,
        weight: &Array1<f32>,
        key: &str,
    ) -> HirNodeId {
        let g = self.register(hir, key, weight.to_vec(), &[weight.len()]);
        let zb = self.register(
            hir,
            &format!("{key}.zb"),
            vec![0.0; weight.len()],
            &[weight.len()],
        );
        self.g(hir).rms_norm(x, g, zb, RMS_EPS)
    }

    fn linear(
        &mut self,
        hir: &mut HirModule,
        x: HirNodeId,
        w_key: &str,
        w: &Array2<f32>,
    ) -> HirNodeId {
        let wt = self.linear_w(hir, w_key, w);
        self.g(hir).mm(x, wt)
    }

    fn add_bias_1d(
        &mut self,
        hir: &mut HirModule,
        x: HirNodeId,
        b: &Array1<f32>,
        key: &str,
    ) -> HirNodeId {
        let bias = self.register(hir, key, b.to_vec(), &[b.len()]);
        self.g(hir).add(x, bias)
    }

    fn resblock(
        &mut self,
        hir: &mut HirModule,
        x: HirNodeId,
        w: &crate::weights::ResidualBlockWeight,
    ) -> Result<HirNodeId> {
        let d_out = self.d();
        let mut h = x;
        if let Some(ref ln) = w.pre_norm {
            h = self.rms_norm(hir, h, ln, "pre_transformer_resblock.pre_norm.weight");
        }
        let mut hidden = self.linear(hir, h, "pre_transformer_resblock.hidden", &w.hidden);
        let rows = self.rows();
        let hidden_dim = w.hidden.nrows();
        hidden = self.g(hir).activation(
            Activation::Relu,
            hidden,
            Shape::new(&[rows, hidden_dim], F32),
        );
        let out = self.linear(hir, hidden, "pre_transformer_resblock.output", &w.output);
        let res = self.linear(hir, h, "pre_transformer_resblock.residual", &w.residual);
        let _ = d_out;
        Ok(self.g(hir).add(out, res))
    }

    fn output_head(
        &mut self,
        hir: &mut HirModule,
        x: HirNodeId,
        w: &Array2<f32>,
        b: Option<&Array1<f32>>,
    ) -> Result<HirNodeId> {
        let mut y = self.linear(hir, x, "output_head", w);
        if let Some(bias) = b {
            y = self.add_bias_1d(hir, y, bias, "output_head.bias");
        }
        Ok(y)
    }

    fn mixing_layer(
        &mut self,
        hir: &mut HirModule,
        idx: usize,
        w: &MixingLayerWeight,
        x: HirNodeId,
        rope_cos_seq: HirNodeId,
        rope_sin_seq: HirNodeId,
        attn_bias_seq: HirNodeId,
        rope_cos_var: Option<HirNodeId>,
        rope_sin_var: Option<HirNodeId>,
        attn_bias_var: Option<HirNodeId>,
    ) -> Result<HirNodeId> {
        let p = format!("layer{idx}");
        let seq_ln = self.rms_norm(hir, x, &w.pre_seq_ln, &format!("{p}.pre_seq_ln"));
        let seq_out = self.mha(
            hir,
            &w.seq_attn,
            &format!("{p}.seq"),
            seq_ln,
            self.dims.b * self.dims.v,
            self.dims.n,
            true,
            rope_cos_seq,
            rope_sin_seq,
            attn_bias_seq,
        )?;
        let seq_out = self.rms_norm(hir, seq_out, &w.post_seq_ln, &format!("{p}.post_seq_ln"));
        let h1 = self.g(hir).add(x, seq_out);

        let h2 = if let (Some(pre), Some(post), Some(var), Some(rc), Some(rs), Some(ab)) = (
            w.pre_var_ln.as_ref(),
            w.post_var_ln.as_ref(),
            w.var_attn.as_ref(),
            rope_cos_var,
            rope_sin_var,
            attn_bias_var,
        ) {
            let var_in = self.bvn_to_bnv(hir, h1)?;
            let var_ln = self.rms_norm(hir, var_in, pre, &format!("{p}.pre_var_ln"));
            let var_out = self.mha(
                hir,
                var,
                &format!("{p}.var"),
                var_ln,
                self.dims.b * self.dims.n,
                self.dims.v,
                false,
                rc,
                rs,
                ab,
            )?;
            let var_out = self.rms_norm(hir, var_out, post, &format!("{p}.post_var_ln"));
            let merged = self.g(hir).add(var_in, var_out);
            let var_4d = self.bnv_to_bvn(hir, merged)?;
            self.g(hir).add(h1, var_4d)
        } else {
            h1
        };

        let ff_ln = self.rms_norm(hir, h2, &w.pre_ff_ln, &format!("{p}.pre_ff_ln"));
        let mut hidden = self.linear(hir, ff_ln, &format!("{p}.ff0"), &w.ff0);
        let rows = self.rows();
        let ff_hidden = w.ff0.nrows();
        hidden = self.g(hir).activation(
            Activation::Relu,
            hidden,
            Shape::new(&[rows, ff_hidden], F32),
        );
        let ff_out = self.linear(hir, hidden, &format!("{p}.ff1"), &w.ff1);
        let ff_out = self.rms_norm(hir, ff_out, &w.post_ff_ln, &format!("{p}.post_ff_ln"));
        Ok(self.g(hir).add(h2, ff_out))
    }

    /// `[B*V*N, D]` → `[B*N*V, D]`.
    fn bvn_to_bnv(&mut self, hir: &mut HirModule, x: HirNodeId) -> Result<HirNodeId> {
        let (b, v, n, d) = (self.dims.b, self.dims.v, self.dims.n, self.d());
        let x = self
            .g(hir)
            .reshape_(x, vec![b as i64, v as i64, n as i64, d as i64]);
        let x = self.g(hir).transpose_(x, vec![0, 2, 1, 3]);
        Ok(self.g(hir).reshape_(x, vec![(b * n * v) as i64, d as i64]))
    }

    /// `[B*N*V, D]` → `[B*V*N, D]`.
    fn bnv_to_bvn(&mut self, hir: &mut HirModule, x: HirNodeId) -> Result<HirNodeId> {
        let (b, v, n, d) = (self.dims.b, self.dims.v, self.dims.n, self.d());
        let x = self
            .g(hir)
            .reshape_(x, vec![b as i64, n as i64, v as i64, d as i64]);
        let x = self.g(hir).transpose_(x, vec![0, 2, 1, 3]);
        Ok(self.g(hir).reshape_(x, vec![(b * v * n) as i64, d as i64]))
    }

    #[allow(clippy::too_many_arguments)]
    fn mha(
        &mut self,
        hir: &mut HirModule,
        w: &MhaWeight,
        tag: &str,
        x: HirNodeId,
        batch: usize,
        seq: usize,
        causal: bool,
        rope_cos: HirNodeId,
        rope_sin: HirNodeId,
        attn_bias: HirNodeId,
    ) -> Result<HirNodeId> {
        let d = self.d();
        let nh = self.nh();
        let hd = self.hd();
        let _ = causal;

        let q = self.linear(hir, x, &format!("{tag}.q"), &w.query);
        let k = self.linear(hir, x, &format!("{tag}.k"), &w.key);
        let v = self.linear(hir, x, &format!("{tag}.v"), &w.value);

        let q = self
            .g(hir)
            .reshape_(q, vec![batch as i64, seq as i64, nh as i64, hd as i64]);
        let k = self
            .g(hir)
            .reshape_(k, vec![batch as i64, seq as i64, nh as i64, hd as i64]);
        let v = self
            .g(hir)
            .reshape_(v, vec![batch as i64, seq as i64, nh as i64, hd as i64]);

        let q = self
            .g(hir)
            .reshape_(q, vec![(batch * nh) as i64, seq as i64, hd as i64]);
        let k = self
            .g(hir)
            .reshape_(k, vec![(batch * nh) as i64, seq as i64, hd as i64]);

        let q = self.apply_host_rope(hir, q, rope_cos, rope_sin, hd)?;
        let k = self.apply_host_rope(hir, k, rope_cos, rope_sin, hd)?;

        // RoPE runs on `[batch*nh, seq, hd]`; QK-RMS must match host row order
        // `(batch, seq, head)` → flat `(batch*seq*nh, hd)`.
        let q = self
            .g(hir)
            .reshape_(q, vec![batch as i64, seq as i64, nh as i64, hd as i64]);
        let k = self
            .g(hir)
            .reshape_(k, vec![batch as i64, seq as i64, nh as i64, hd as i64]);
        let q = self
            .g(hir)
            .reshape_(q, vec![(batch * seq * nh) as i64, hd as i64]);
        let k = self
            .g(hir)
            .reshape_(k, vec![(batch * seq * nh) as i64, hd as i64]);

        let q = self.qk_rms(hir, q, &w.query_ln, &format!("{tag}.qln"), batch * seq * nh)?;
        let k = self.qk_rms(hir, k, &w.key_ln, &format!("{tag}.kln"), batch * seq * nh)?;

        let q = self
            .g(hir)
            .reshape_(q, vec![batch as i64, seq as i64, nh as i64, hd as i64]);
        let k = self
            .g(hir)
            .reshape_(k, vec![batch as i64, seq as i64, nh as i64, hd as i64]);

        let q = self.per_dim_scale(
            hir,
            q,
            &w.per_dim_scale,
            &format!("{tag}.pds"),
            batch,
            seq,
            nh,
        )?;
        let q = self
            .g(hir)
            .reshape_(q, vec![batch as i64, nh as i64, seq as i64, hd as i64]);
        let k = self
            .g(hir)
            .reshape_(k, vec![batch as i64, nh as i64, seq as i64, hd as i64]);
        let v = self
            .g(hir)
            .reshape_(v, vec![batch as i64, nh as i64, seq as i64, hd as i64]);

        let k_t = self.g(hir).transpose_(k, vec![0, 1, 3, 2]);
        let mut scores = self.g(hir).mm(q, k_t);
        let bias = self
            .g(hir)
            .reshape_(attn_bias, vec![batch as i64, 1, seq as i64, seq as i64]);
        scores = self.g(hir).add(scores, bias);
        let attn = self.g(hir).sm(scores, -1);
        let ctx = self.g(hir).mm(attn, v);
        let ctx = self.g(hir).transpose_(ctx, vec![0, 2, 1, 3]);
        let ctx = self
            .g(hir)
            .reshape_(ctx, vec![(batch * seq) as i64, d as i64]);
        Ok(self.linear(hir, ctx, &format!("{tag}.o"), &w.out))
    }

    /// RoPE matching `rope::apply_rope_tables` (same tables as the host path).
    fn apply_host_rope(
        &mut self,
        hir: &mut HirModule,
        x: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
        hd: usize,
    ) -> Result<HirNodeId> {
        let half = hd / 2;
        let x0 = self.g(hir).narrow_(x, 2, 0, half);
        let x1 = self.g(hir).narrow_(x, 2, half, half);
        let q0c = self.g(hir).mul(x0, cos);
        let q1s = self.g(hir).mul(x1, sin);
        let out0 = self.g(hir).sub(q0c, q1s);
        let q1c = self.g(hir).mul(x1, cos);
        let q0s = self.g(hir).mul(x0, sin);
        let out1 = self.g(hir).add(q1c, q0s);
        Ok(self.g(hir).concat_(vec![out0, out1], 2))
    }

    /// SDPA with additive mask; scale `1.0` matches the host reference (not `1/sqrt(d)`).
    #[allow(dead_code)]
    fn attention_bias_scaled(
        &mut self,
        hir: &mut HirModule,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        bias: HirNodeId,
        nh: usize,
        hd: usize,
    ) -> Result<HirNodeId> {
        let shape = self.g(hir).shape(q).clone();
        Ok(self.g(hir).0.mir(
            attention_kind_op(nh, hd, None, MaskKind::Bias, Some(1.0), None),
            vec![q, k, v, bias],
            shape,
        ))
    }

    fn qk_rms(
        &mut self,
        hir: &mut HirModule,
        x: HirNodeId,
        weight: &Array1<f32>,
        key: &str,
        rows: usize,
    ) -> Result<HirNodeId> {
        let hd = self.hd();
        let x = self.g(hir).reshape_(x, vec![rows as i64, hd as i64]);
        let y = self.rms_norm(hir, x, weight, key);
        Ok(self.g(hir).reshape_(y, vec![rows as i64, hd as i64]))
    }

    fn per_dim_scale(
        &mut self,
        hir: &mut HirModule,
        q: HirNodeId,
        scale: &Array1<f32>,
        key: &str,
        batch: usize,
        seq: usize,
        nh: usize,
    ) -> Result<HirNodeId> {
        let hd = self.hd();
        let factors = per_dim_factors(scale.view(), hd);
        let mul = self.register(hir, key, factors, &[hd]);
        let mul = self.g(hir).reshape_(mul, vec![1, 1, 1, hd as i64]);
        let q = self
            .g(hir)
            .reshape_(q, vec![batch as i64, seq as i64, nh as i64, hd as i64]);
        Ok(self.g(hir).mul(q, mul))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TimesFM3Config;

    #[test]
    fn core_hir_lowers() {
        let cfg = TimesFM3Config::synth_tiny();
        let w = TimesFM3Weights::synth(&cfg, 1);
        let dims = CoreGraphDims { b: 1, v: 1, n: 4 };
        let (hir, _params) = build_core_hir(&cfg, &w, dims).unwrap();
        hir.lower_to_mir().expect("lower");
    }
}
