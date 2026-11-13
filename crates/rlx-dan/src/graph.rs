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

//! DAN as two RLX graphs.
//!
//! **Encoder** (`image [1, 3, H, W]`): `FCN_Encoder` — six `ConvBlock`s
//! (conv → ReLU → conv → ReLU → InstanceNorm → strided conv → ReLU) and
//! four depthwise-separable `DSCBlock`s (residual where the shape allows)
//! — then the 2-D positional encoding and, since cross-attention keys and
//! values never change during decoding, every decoder layer's
//! `att.in_proj_{k,v}` of the result. Outputs `features [1, 256, h, w]`,
//! then `cross_k_i`, `cross_v_i [1, h·w, 256]`.
//!
//! **Step** (one token): a `GlobalDecoderLayer` stack for one position.
//! DAN's decoder caches each layer's *outputs* and recomputes keys and
//! values from them over the last 100 tokens; keys and values are a fixed
//! function of those outputs, so caching them instead is the same
//! computation. Inputs `x [1, 1, 256]` (token embedding + 1-D position),
//! `mask [1, W]` and `past_k_i` / `past_v_i [1, W − 1, 256]` (`W` = the
//! window, rows in any order: attention does not see order); the cross
//! K/V and the feature padding mask are parameters set once per page.
//! Outputs `logits [1, 1, vocab + 1]`, then the new `k_i`, `v_i` rows.

use crate::model::{CONV_BLOCKS, DIM, DSC_BLOCKS, HEADS, LAYERS, NORM_EPS, TensorTable, WINDOW};
use anyhow::{Result, anyhow, ensure};
use rlx_core::vision_ops_ir::{add_bias_nchw, nchw_shape};
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::{MaskKind, Op};
use rlx_ir::ops::attention::attention_kind_op;
use rlx_ir::{DType, HirGraphExt, HirModule, Shape};
use std::collections::HashMap;

pub const IMAGE: &str = "image";
pub const STEP_X: &str = "x";
pub const STEP_MASK: &str = "mask";
pub const MEMORY_MASK: &str = "@memory_mask";

pub fn cross_k_name(i: usize) -> String {
    format!("@cross_k_{i}")
}

pub fn cross_v_name(i: usize) -> String {
    format!("@cross_v_{i}")
}

pub struct BuiltGraph {
    pub hir: HirModule,
    pub params: HashMap<String, Vec<f32>>,
}

struct Builder<'t> {
    hir: HirModule,
    params: HashMap<String, Vec<f32>>,
    t: &'t TensorTable,
}

impl Builder<'_> {
    fn m(&mut self) -> HirMut<'_> {
        HirMut::new(&mut self.hir)
    }

    fn put(&mut self, name: &str, shape: &[usize], data: Vec<f32>) -> HirNodeId {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>(), "{name}");
        let id = self.m().param(name, Shape::new(shape, DType::F32));
        self.params.insert(name.to_string(), data);
        id
    }

    fn get(&self, key: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (d, s) = self
            .t
            .get(key)
            .ok_or_else(|| anyhow!("checkpoint is missing {key}"))?;
        ensure!(s == shape, "{key}: shape {s:?}, expected {shape:?}");
        Ok(d.clone())
    }

    fn tensor(&mut self, key: &str, shape: &[usize]) -> Result<HirNodeId> {
        let d = self.get(key, shape)?;
        Ok(self.put(key, shape, d))
    }

    /// `Conv2d` with bias, NCHW, padding 1 for 3×3.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        x: HirNodeId,
        prefix: &str,
        cin: usize,
        cout: usize,
        k: usize,
        stride: [usize; 2],
        groups: usize,
        hw: [usize; 2],
    ) -> Result<(HirNodeId, [usize; 2])> {
        let w = self.tensor(&format!("{prefix}.weight"), &[cout, cin / groups, k, k])?;
        let b = self.tensor(&format!("{prefix}.bias"), &[cout])?;
        let pad = k / 2;
        let out = [0, 1].map(|d| (hw[d] + 2 * pad - k) / stride[d] + 1);
        let y = self.m().add_node(
            Op::Conv {
                kernel_size: vec![k, k],
                stride: stride.to_vec(),
                padding: vec![pad, pad],
                dilation: vec![1, 1],
                groups,
            },
            vec![x, w],
            nchw_shape(1, cout, out[0], out[1], DType::F32),
        );
        Ok((
            add_bias_nchw(&mut self.m(), y, b, 1, cout, out[0], out[1]),
            out,
        ))
    }

    /// `DepthSepConv2D`: depthwise 3×3 (stride on it), then pointwise.
    fn ds_conv(
        &mut self,
        x: HirNodeId,
        prefix: &str,
        cin: usize,
        cout: usize,
        hw: [usize; 2],
    ) -> Result<HirNodeId> {
        let (y, hw2) = self.conv(
            x,
            &format!("{prefix}.depth_conv"),
            cin,
            cin,
            3,
            [1, 1],
            cin,
            hw,
        )?;
        let (y, _) = self.conv(
            y,
            &format!("{prefix}.point_conv"),
            cin,
            cout,
            1,
            [1, 1],
            1,
            hw2,
        )?;
        Ok(y)
    }

    /// `InstanceNorm2d(c, eps=1e-3)` without affine: GroupNorm with one
    /// group per channel, unit scale and zero shift.
    fn instance_norm(&mut self, x: HirNodeId, c: usize, tag: &str) -> HirNodeId {
        let g = self.put(&format!("{tag}.norm.one"), &[c], vec![1.0; c]);
        let b = self.put(&format!("{tag}.norm.zero"), &[c], vec![0.0; c]);
        self.m().group_norm(x, g, b, c, NORM_EPS)
    }

    fn linear(&mut self, x: HirNodeId, prefix: &str, out: usize, inp: usize) -> Result<HirNodeId> {
        let w = self.get(&format!("{prefix}.weight"), &[out, inp])?;
        let wt = self.put(
            &format!("{prefix}.weight^T"),
            &[inp, out],
            transpose(&w, out, inp),
        );
        let b = self.tensor(&format!("{prefix}.bias"), &[out])?;
        let y = self.m().mm(x, wt);
        Ok(self.m().add(y, b))
    }

    fn layer_norm(&mut self, x: HirNodeId, prefix: &str) -> Result<HirNodeId> {
        let g = self.tensor(&format!("{prefix}.weight"), &[DIM])?;
        let b = self.tensor(&format!("{prefix}.bias"), &[DIM])?;
        Ok(self.m().ln(x, g, b, 1e-5))
    }

    fn attend(&mut self, q: HirNodeId, k: HirNodeId, v: HirNodeId, mask: HirNodeId) -> HirNodeId {
        let shape = self.hir.node(q).shape.clone();
        // Default scale head_dim^-0.5 = `CustomMultiHeadAttention.scale_factor`.
        self.m().add_node(
            attention_kind_op(HEADS, DIM / HEADS, None, MaskKind::Custom, None, None),
            vec![q, k, v, mask],
            shape,
        )
    }
}

fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0f32; m.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = m[r * cols + c];
        }
    }
    t
}

/// Feature-map size the encoder gives for an `[h, w]` image.
pub fn feature_size(h: usize, w: usize) -> [usize; 2] {
    CONV_BLOCKS.iter().fold([h, w], |s, (_, _, st)| {
        [0, 1].map(|d| (s[d] - 1) / st[d] + 1)
    })
}

/// The encoder for an `[h, w]` page, plus cross K/V for every layer.
pub fn build_encoder(
    enc: &TensorTable,
    dec: &TensorTable,
    h: usize,
    w: usize,
) -> Result<BuiltGraph> {
    let mut b = Builder {
        hir: HirModule::new("dan_encoder"),
        params: HashMap::new(),
        t: enc,
    };
    let mut x = b.m().input(IMAGE, nchw_shape(1, 3, h, w, DType::F32));
    let mut hw = [h, w];
    for (i, &(cin, cout, stride)) in CONV_BLOCKS.iter().enumerate() {
        let p = format!("init_blocks.{i}");
        let (y, s) = b.conv(x, &format!("{p}.conv1"), cin, cout, 3, [1, 1], 1, hw)?;
        let y = b.m().relu(y);
        let (y, s) = b.conv(y, &format!("{p}.conv2"), cout, cout, 3, [1, 1], 1, s)?;
        let y = b.m().relu(y);
        let y = b.instance_norm(y, cout, &p);
        let (y, s) = b.conv(y, &format!("{p}.conv3"), cout, cout, 3, stride, 1, s)?;
        x = b.m().relu(y);
        hw = s;
    }
    for (i, &(cin, cout)) in DSC_BLOCKS.iter().enumerate() {
        let p = format!("blocks.{i}");
        let y = b.ds_conv(x, &format!("{p}.conv1"), cin, cout, hw)?;
        let y = b.m().relu(y);
        let y = b.ds_conv(y, &format!("{p}.conv2"), cout, cout, hw)?;
        let y = b.m().relu(y);
        let y = b.instance_norm(y, cout, &p);
        let y = b.ds_conv(y, &format!("{p}.conv3"), cout, cout, hw)?;
        x = if cin == cout { b.m().add(x, y) } else { y };
    }
    let [fh, fw] = hw;
    let features = x;
    let pe = b.put("pe_2d", &[1, DIM, fh, fw], crate::model::pe_2d(fh, fw));
    let pos = b.m().add(features, pe);
    let flat = b.m().reshape_(pos, vec![1, DIM as i64, (fh * fw) as i64]);
    let mem = b.m().transpose_(flat, vec![0, 2, 1]); // [1, h·w, C]
    let mut outs = vec![features];
    b.t = dec;
    for i in 0..LAYERS {
        let p = format!("att_decoder.decoder_layers.{i}.att");
        outs.push(b.linear(mem, &format!("{p}.in_proj_k"), DIM, DIM)?);
        outs.push(b.linear(mem, &format!("{p}.in_proj_v"), DIM, DIM)?);
    }
    b.m().set_outputs(outs);
    Ok(BuiltGraph {
        hir: b.hir,
        params: b.params,
    })
}

/// One decoder step against `memory` feature positions and a window of
/// `WINDOW − 1` cached tokens.
pub fn build_step(dec: &TensorTable, memory: usize, vocab: usize) -> Result<BuiltGraph> {
    let past = WINDOW - 1;
    let f = DType::F32;
    let mut b = Builder {
        hir: HirModule::new("dan_step"),
        params: HashMap::new(),
        t: dec,
    };
    let mut x = b.m().input(STEP_X, Shape::new(&[1, 1, DIM], f));
    let mask = b.m().input(STEP_MASK, Shape::new(&[1, WINDOW], f));
    let mem_mask = b.m().param(MEMORY_MASK, Shape::new(&[1, memory], f));
    let names = rlx_core::autoregressive::past_kv_input_names(LAYERS);
    let mut new_kv = Vec::with_capacity(2 * LAYERS);
    for i in 0..LAYERS {
        let pk = b.m().input(&names[2 * i], Shape::new(&[1, past, DIM], f));
        let pv = b
            .m()
            .input(&names[2 * i + 1], Shape::new(&[1, past, DIM], f));
        let ck = b
            .m()
            .param(cross_k_name(i), Shape::new(&[1, memory, DIM], f));
        let cv = b
            .m()
            .param(cross_v_name(i), Shape::new(&[1, memory, DIM], f));
        let lp = format!("att_decoder.decoder_layers.{i}");

        let sp = format!("{lp}.self_att");
        let q = b.linear(x, &format!("{sp}.in_proj_q"), DIM, DIM)?;
        let kn = b.linear(x, &format!("{sp}.in_proj_k"), DIM, DIM)?;
        let vn = b.linear(x, &format!("{sp}.in_proj_v"), DIM, DIM)?;
        let k = b.m().concat_(vec![pk, kn], 1);
        let v = b.m().concat_(vec![pv, vn], 1);
        new_kv.push(kn);
        new_kv.push(vn);
        let a = b.attend(q, k, v, mask);
        let o = b.linear(a, &format!("{sp}.out_proj"), DIM, DIM)?;
        let r = b.m().add(x, o);
        x = b.layer_norm(r, &format!("{lp}.norm1"))?;

        let cp = format!("{lp}.att");
        let q = b.linear(x, &format!("{cp}.in_proj_q"), DIM, DIM)?;
        let a = b.attend(q, ck, cv, mem_mask);
        let o = b.linear(a, &format!("{cp}.out_proj"), DIM, DIM)?;
        let r = b.m().add(x, o);
        x = b.layer_norm(r, &format!("{lp}.norm2"))?;

        let h1 = b.linear(x, &format!("{lp}.linear1"), crate::model::FFN, DIM)?;
        let h1 = b.m().relu(h1);
        let h2 = b.linear(h1, &format!("{lp}.linear2"), DIM, crate::model::FFN)?;
        let r = b.m().add(x, h2);
        x = b.layer_norm(r, &format!("{lp}.norm3"))?;
    }
    // `end_conv` is a 1×1 Conv1d over relu(output): a linear layer.
    let classes = vocab + 1;
    let w = b.get("end_conv.weight", &[classes, DIM, 1])?;
    let wt = b.put(
        "end_conv.weight^T",
        &[DIM, classes],
        transpose(&w, classes, DIM),
    );
    let bias = b.tensor("end_conv.bias", &[classes])?;
    let r = b.m().relu(x);
    let y = b.m().mm(r, wt);
    let logits = b.m().add(y, bias);
    let mut outs = vec![logits];
    outs.extend(new_kv);
    b.m().set_outputs(outs);
    Ok(BuiltGraph {
        hir: b.hir,
        params: b.params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_geometry() {
        // ceil-halving: H five times, W three times.
        assert_eq!(feature_size(792, 1240), [25, 155]);
        assert_eq!(feature_size(1754, 1240), [55, 155]);
    }
}
