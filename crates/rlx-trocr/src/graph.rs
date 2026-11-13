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

//! The two graphs generation runs.
//!
//! **Encoder** (`pixels [1, 3, S, S]`): ViT/DeiT — patch conv, prefix
//! tokens, learned positions, pre-norm blocks, final LayerNorm — and then,
//! because the decoder's cross-attention keys and values depend only on the
//! image, every decoder layer's `encoder_attn.{k,v}_proj` of the result.
//! Outputs `encoder [1, S_enc, d_enc]`, then `cross_k_i`, `cross_v_i`.
//!
//! **Step** (one token): HF `TrOCRDecoder` for a single position against a
//! fixed-capacity self-attention cache. Inputs: `x [1, 1, d]` (token
//! embedding × scale + learned position, gathered on the host), `mask
//! [1, C + 1]` ([`rlx_runtime::attn_mask::bucket_decode_mask`]),
//! `past_k_i` / `past_v_i [1, C, d]`; the cross K/V are parameters set once
//! per image. Post-norm layers (attn → +res → LN, cross-attn → +res → LN,
//! FFN → +res → LN), then the output projection. Outputs `logits
//! [1, 1, vocab]`, then the new `k_i`, `v_i` rows `[1, 1, d]`.

use crate::config::{DecoderAct, EncoderKind, TrOcrConfig};
use anyhow::{Result, anyhow, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::MaskKind;
use rlx_ir::ops::attention::attention_kind_op;
use rlx_ir::{DType, HirGraphExt, HirModule, Shape};
use std::collections::HashMap;

pub const PIXELS: &str = "pixels";
pub const STEP_X: &str = "x";
pub const STEP_MASK: &str = "mask";

/// Decoder tensors live under this prefix in a `VisionEncoderDecoderModel`.
pub const DEC: &str = "decoder.model.decoder";

pub fn cross_k_name(i: usize) -> String {
    format!("@cross_k_{i}")
}

pub fn cross_v_name(i: usize) -> String {
    format!("@cross_v_{i}")
}

/// A graph plus the parameter tensors it reads.
pub struct BuiltGraph {
    pub hir: HirModule,
    pub params: HashMap<String, Vec<f32>>,
}

struct Builder<'w> {
    hir: HirModule,
    params: HashMap<String, Vec<f32>>,
    w: &'w mut WeightMap,
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

    /// A checkpoint tensor as a parameter of its own shape.
    fn tensor(&mut self, key: &str) -> Result<HirNodeId> {
        let (data, shape) = self.w.take(key)?;
        Ok(self.put(key, &shape, data))
    }

    /// `x · Wᵀ (+ b)` for an HF `[out, in]` weight.
    fn linear(&mut self, x: HirNodeId, prefix: &str, bias: bool) -> Result<HirNodeId> {
        let (w, shape) = self.w.take(&format!("{prefix}.weight"))?;
        ensure!(shape.len() == 2, "{prefix}.weight is not a matrix");
        let (out, inp) = (shape[0], shape[1]);
        let wt = self.put(
            &format!("{prefix}.weight^T"),
            &[inp, out],
            transpose(&w, out, inp),
        );
        let y = self.m().mm(x, wt);
        if !bias {
            return Ok(y);
        }
        let b = self.tensor(&format!("{prefix}.bias"))?;
        Ok(self.m().add(y, b))
    }

    fn layer_norm(&mut self, x: HirNodeId, prefix: &str, eps: f32) -> Result<HirNodeId> {
        let g = self.tensor(&format!("{prefix}.weight"))?;
        let b = self.tensor(&format!("{prefix}.bias"))?;
        Ok(self.m().ln(x, g, b, eps))
    }

    fn attend(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        mask: Option<HirNodeId>,
        heads: usize,
        head_dim: usize,
    ) -> HirNodeId {
        let shape = self.hir.node(q).shape.clone();
        let (kind, inputs) = match mask {
            Some(m) => (MaskKind::Custom, vec![q, k, v, m]),
            None => (MaskKind::None, vec![q, k, v]),
        };
        // Default scale head_dim^-0.5, which is TrOCR's / ViT's `scaling`.
        self.m().add_node(
            attention_kind_op(heads, head_dim, None, kind, None, None),
            inputs,
            shape,
        )
    }
}

/// Row-major `[rows, cols]` → `[cols, rows]`.
pub fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0f32; m.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = m[r * cols + c];
        }
    }
    t
}

/// The encoder plus every decoder layer's cross-attention K/V.
pub fn build_encoder(cfg: &TrOcrConfig, w: &mut WeightMap) -> Result<BuiltGraph> {
    let e = &cfg.encoder;
    let (h, g, p) = (e.hidden, e.grid(), e.patch_size);
    let seq = e.seq();
    let mut b = Builder {
        hir: HirModule::new("trocr_encoder"),
        params: HashMap::new(),
        w,
    };
    let f = DType::F32;
    let px = b.m().input(
        PIXELS,
        Shape::new(&[1, e.channels, e.image_size, e.image_size], f),
    );
    let pw = b.tensor("encoder.embeddings.patch_embeddings.projection.weight")?;
    let pb = b.tensor("encoder.embeddings.patch_embeddings.projection.bias")?;
    let conv = b.m().conv2d(
        px,
        pw,
        [p, p],
        [p, p],
        [0, 0],
        1,
        Shape::new(&[1, h, g, g], f),
    );
    let conv = rlx_core::vision_ops_ir::add_bias_nchw(&mut b.m(), conv, pb, 1, h, g, g);
    let flat = b.m().reshape_(conv, vec![1, h as i64, (g * g) as i64]);
    let patches = b.m().transpose_(flat, vec![0, 2, 1]);
    let mut prefix = vec![b.tensor("encoder.embeddings.cls_token")?];
    if e.kind == EncoderKind::Deit {
        prefix.push(b.tensor("encoder.embeddings.distillation_token")?);
    }
    prefix.push(patches);
    let tokens = b.m().concat_(prefix, 1);
    let (pos, pshape) = b.w.take("encoder.embeddings.position_embeddings")?;
    ensure!(
        pshape == [1, seq, h],
        "position_embeddings {pshape:?}, expected [1, {seq}, {h}] (interpolated positions \
         are not supported)"
    );
    let pos = b.put("encoder.embeddings.position_embeddings", &pshape, pos);
    let mut x = b.m().add(tokens, pos);

    let hd = h / e.heads;
    for i in 0..e.layers {
        let lp = format!("encoder.encoder.layer.{i}");
        let n1 = b.layer_norm(x, &format!("{lp}.layernorm_before"), e.eps)?;
        let ap = format!("{lp}.attention.attention");
        let q = b.linear(n1, &format!("{ap}.query"), e.qkv_bias)?;
        let k = b.linear(n1, &format!("{ap}.key"), e.qkv_bias)?;
        let v = b.linear(n1, &format!("{ap}.value"), e.qkv_bias)?;
        let a = b.attend(q, k, v, None, e.heads, hd);
        let o = b.linear(a, &format!("{lp}.attention.output.dense"), true)?;
        x = b.m().add(x, o);
        let n2 = b.layer_norm(x, &format!("{lp}.layernorm_after"), e.eps)?;
        let f1 = b.linear(n2, &format!("{lp}.intermediate.dense"), true)?;
        let f1 = b.m().gelu(f1);
        let f2 = b.linear(f1, &format!("{lp}.output.dense"), true)?;
        x = b.m().add(x, f2);
    }
    let enc = b.layer_norm(x, "encoder.layernorm", e.eps)?;
    // DeiT/ViT checkpoints saved from a pooled model carry a pooler the
    // encoder-decoder never runs.
    for k in ["encoder.pooler.dense.weight", "encoder.pooler.dense.bias"] {
        if b.w.has(k) {
            b.w.take(k)?;
        }
    }

    let mut outs = vec![enc];
    for i in 0..cfg.decoder.layers {
        let ap = format!("{DEC}.layers.{i}.encoder_attn");
        outs.push(b.linear(enc, &format!("{ap}.k_proj"), true)?);
        outs.push(b.linear(enc, &format!("{ap}.v_proj"), true)?);
    }
    b.m().set_outputs(outs);
    Ok(BuiltGraph {
        hir: b.hir,
        params: b.params,
    })
}

/// One decoder step against `capacity` cached positions.
pub fn build_step(cfg: &TrOcrConfig, w: &mut WeightMap, capacity: usize) -> Result<BuiltGraph> {
    let dc = &cfg.decoder;
    let (d, n) = (dc.d_model, dc.layers);
    let s_enc = cfg.encoder.seq();
    let hd = d / dc.heads;
    let f = DType::F32;
    let mut b = Builder {
        hir: HirModule::new("trocr_step"),
        params: HashMap::new(),
        w,
    };
    let mut x = b.m().input(STEP_X, Shape::new(&[1, 1, d], f));
    let mask = b.m().input(STEP_MASK, Shape::new(&[1, capacity + 1], f));
    let names = rlx_core::autoregressive::past_kv_input_names(n);
    let mut past = Vec::with_capacity(n);
    for i in 0..n {
        let k = b.m().input(&names[2 * i], Shape::new(&[1, capacity, d], f));
        let v = b
            .m()
            .input(&names[2 * i + 1], Shape::new(&[1, capacity, d], f));
        let ck = b.m().param(cross_k_name(i), Shape::new(&[1, s_enc, d], f));
        let cv = b.m().param(cross_v_name(i), Shape::new(&[1, s_enc, d], f));
        past.push((k, v, ck, cv));
    }
    if dc.layernorm_embedding {
        x = b.layer_norm(x, &format!("{DEC}.layernorm_embedding"), 1e-5)?;
    }
    let mut new_kv = Vec::with_capacity(2 * n);
    for (i, &(pk, pv, ck, cv)) in past.iter().enumerate() {
        let lp = format!("{DEC}.layers.{i}");
        let sp = format!("{lp}.self_attn");
        let q = b.linear(x, &format!("{sp}.q_proj"), true)?;
        let kn = b.linear(x, &format!("{sp}.k_proj"), true)?;
        let vn = b.linear(x, &format!("{sp}.v_proj"), true)?;
        let k = b.m().concat_(vec![pk, kn], 1);
        let v = b.m().concat_(vec![pv, vn], 1);
        new_kv.push(kn);
        new_kv.push(vn);
        let a = b.attend(q, k, v, Some(mask), dc.heads, hd);
        let o = b.linear(a, &format!("{sp}.out_proj"), true)?;
        let r = b.m().add(x, o);
        x = b.layer_norm(r, &format!("{lp}.self_attn_layer_norm"), 1e-5)?;

        let cp = format!("{lp}.encoder_attn");
        let q = b.linear(x, &format!("{cp}.q_proj"), true)?;
        let a = b.attend(q, ck, cv, None, dc.heads, hd);
        let o = b.linear(a, &format!("{cp}.out_proj"), true)?;
        let r = b.m().add(x, o);
        x = b.layer_norm(r, &format!("{lp}.encoder_attn_layer_norm"), 1e-5)?;

        let h1 = b.linear(x, &format!("{lp}.fc1"), true)?;
        let h1 = match dc.act {
            DecoderAct::Relu => b.m().relu(h1),
            DecoderAct::Gelu => b.m().gelu(h1),
        };
        let h2 = b.linear(h1, &format!("{lp}.fc2"), true)?;
        let r = b.m().add(x, h2);
        x = b.layer_norm(r, &format!("{lp}.final_layer_norm"), 1e-5)?;
    }
    let (out_w, shape) = output_projection(b.w)?;
    ensure!(
        shape == [dc.vocab, d],
        "output projection {shape:?}, expected [{}, {d}]",
        dc.vocab
    );
    let wt = b.put(
        "output_projection^T",
        &[d, dc.vocab],
        transpose(&out_w, dc.vocab, d),
    );
    let logits = b.m().mm(x, wt);
    let mut outs = vec![logits];
    outs.extend(new_kv);
    b.m().set_outputs(outs);
    Ok(BuiltGraph {
        hir: b.hir,
        params: b.params,
    })
}

/// `decoder.output_projection.weight` (the runner inserts the token
/// embeddings under that name for tied checkpoints).
fn output_projection(w: &mut WeightMap) -> Result<(Vec<f32>, Vec<usize>)> {
    w.take("decoder.output_projection.weight")
        .map_err(|_| anyhow!("no decoder.output_projection.weight"))
}
