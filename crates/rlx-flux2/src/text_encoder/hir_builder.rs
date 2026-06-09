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

//! Qwen3-shaped FLUX.2 text encoder HIR (causal LM trunk → multi-layer prompt embeds).

use super::super::builder::Flux2GraphParams;
use super::weights::{
    Flux2TextEncoderAttnWeights, Flux2TextEncoderLayerWeights, Flux2TextEncoderMlpWeights,
    Flux2TextEncoderWeights,
};
use crate::weights::{LinearWeights, RmsNormWeight};
use anyhow::{Result, ensure};
use rlx_ir::hir::{FusionPolicy, HirModule, HirNodeId};
use rlx_ir::op::{Activation, BinaryOp, MaskKind};
use rlx_ir::{DType, Op, Shape};
use rlx_qwen3::Qwen3Config;
use rlx_runtime::Device;

pub struct Flux2TextEncoderGraph {
    pub hir: HirModule,
    pub params: Flux2GraphParams,
    pub joint_dim: usize,
}

pub fn build_flux2_text_encoder_hir(
    cfg: &Qwen3Config,
    weights: &Flux2TextEncoderWeights,
    batch: usize,
    seq: usize,
    hidden_state_layers: &[usize],
) -> Result<Flux2TextEncoderGraph> {
    ensure!(
        cfg.num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads),
        "num_attention_heads must divide num_key_value_heads"
    );
    let joint_dim = cfg.hidden_size * hidden_state_layers.len();
    let f = DType::F32;
    let mut hir = HirModule::new("flux2_text_encoder").with_fusion_policy(FusionPolicy::Direct);
    let mut params = Flux2GraphParams::new();
    let ids = hir.input("input_ids", Shape::new(&[batch, seq], f));
    let mut b =
        TextEncoderHirBuilder::from_emit_parts(&mut hir, &mut params, cfg, weights, batch, seq);
    let mut hidden = b.emit_embed(ids)?;
    let mut checkpoints = vec![hidden];
    let (cos, sin) = b.rope_tables()?;
    for (li, layer) in weights.layers.iter().enumerate() {
        hidden = b.layer_forward(layer, li, hidden, cos, sin)?;
        checkpoints.push(hidden);
    }
    let out = b.emit_joint_output(&checkpoints, hidden_state_layers, joint_dim)?;
    hir.outputs = vec![out];
    Ok(Flux2TextEncoderGraph {
        hir,
        params,
        joint_dim,
    })
}

pub fn compile_flux2_text_encoder_hir(
    cfg: &Qwen3Config,
    weights: &Flux2TextEncoderWeights,
    batch: usize,
    seq: usize,
    hidden_state_layers: &[usize],
    device: Device,
    aot: Option<&rlx_runtime::AotCache>,
) -> Result<(rlx_runtime::CompiledGraph, Flux2GraphParams)> {
    use crate::compile_util::{compile_hir_cached, flux2_text_encoder_aot_key};

    crate::device::assert_flux2_device_available(device)?;
    let g = build_flux2_text_encoder_hir(cfg, weights, batch, seq, hidden_state_layers)?;
    let key = flux2_text_encoder_aot_key(device, batch, seq);
    let mut compiled = compile_hir_cached(
        device,
        aot,
        &key,
        g.hir,
        &crate::compile_util::flux2_compile_profile(),
    )?;
    for (name, data) in &g.params {
        compiled.set_param(name, data);
    }
    Ok((compiled, g.params))
}

pub(crate) struct TextEncoderHirBuilder<'a> {
    hir: &'a mut HirModule,
    params: &'a mut Flux2GraphParams,
    cfg: &'a Qwen3Config,
    weights: &'a Flux2TextEncoderWeights,
    batch: usize,
    seq: usize,
    f: DType,
    eps: f32,
}

impl<'a> TextEncoderHirBuilder<'a> {
    pub(crate) fn from_emit_parts(
        hir: &'a mut HirModule,
        params: &'a mut Flux2GraphParams,
        cfg: &'a Qwen3Config,
        weights: &'a Flux2TextEncoderWeights,
        batch: usize,
        seq: usize,
    ) -> Self {
        Self {
            hir,
            params,
            cfg,
            weights,
            batch,
            seq,
            f: DType::F32,
            eps: cfg.rms_norm_eps as f32,
        }
    }

    pub(crate) fn emit_embed(&mut self, ids: HirNodeId) -> Result<HirNodeId> {
        let h = self.cfg.hidden_size;
        let (embed_data, vocab, _) = &self.weights.embed_tokens;
        let embed = self.register_param(
            "embed_tokens.weight",
            embed_data.clone(),
            Shape::new(&[*vocab, h], self.f),
        );
        Ok(self
            .hir
            .mir(Op::Gather { axis: 0 }, vec![embed, ids], self.bsh()))
    }

    pub(crate) fn emit_joint_output(
        &mut self,
        checkpoints: &[HirNodeId],
        hidden_state_layers: &[usize],
        joint_dim: usize,
    ) -> Result<HirNodeId> {
        let h = self.cfg.hidden_size;
        let mut out_pieces: Vec<HirNodeId> = Vec::with_capacity(hidden_state_layers.len());
        for (i, &layer_idx) in hidden_state_layers.iter().enumerate() {
            ensure!(
                layer_idx < checkpoints.len(),
                "hidden_state_layers[{i}]={layer_idx} out of range (len={})",
                checkpoints.len()
            );
            out_pieces.push(checkpoints[layer_idx]);
        }
        let rows = self.batch * self.seq;
        let mut flat_parts: Vec<HirNodeId> = Vec::with_capacity(out_pieces.len());
        for p in &out_pieces {
            flat_parts.push(self.reshape(*p, vec![rows as i64, h as i64]));
        }
        let flat = if flat_parts.len() == 1 {
            flat_parts[0]
        } else {
            self.concat(flat_parts, 1, Shape::new(&[rows, joint_dim], self.f))
        };
        Ok(self.reshape(
            flat,
            vec![self.batch as i64, self.seq as i64, joint_dim as i64],
        ))
    }

    fn bsh(&self) -> Shape {
        Shape::new(&[self.batch, self.seq, self.cfg.hidden_size], self.f)
    }

    fn bsh_heads(&self, heads: usize) -> Shape {
        Shape::new(&[self.batch, self.seq, heads * self.cfg.head_dim], self.f)
    }

    fn register_param(&mut self, name: &str, data: Vec<f32>, shape: Shape) -> HirNodeId {
        let id = self.hir.param(name, shape);
        self.params.insert(name.to_string(), data);
        id
    }

    fn linear(
        &mut self,
        x: HirNodeId,
        lw: &LinearWeights,
        name: &str,
        out_shape: Shape,
    ) -> Result<HirNodeId> {
        let w = self.register_param(
            &format!("{name}.weight"),
            lw.w_t.clone(),
            Shape::new(&[lw.in_dim, lw.out_dim], self.f),
        );
        let bias = if lw.bias.iter().all(|&v| v == 0.0) {
            None
        } else {
            let b = self.register_param(
                &format!("{name}.bias"),
                lw.bias.clone(),
                Shape::new(&[lw.out_dim], self.f),
            );
            Some(b)
        };
        Ok(self.hir.linear(x, w, bias, None, out_shape))
    }

    fn rms_norm(
        &mut self,
        x: HirNodeId,
        gamma: &RmsNormWeight,
        name: &str,
        shape: Shape,
    ) -> HirNodeId {
        let g = self.register_param(
            &format!("{name}.weight"),
            gamma.scale.clone(),
            Shape::new(&[gamma.scale.len()], self.f),
        );
        let beta = self.register_param(
            &format!("{name}.beta"),
            vec![0.0f32; gamma.scale.len()],
            Shape::new(&[gamma.scale.len()], self.f),
        );
        self.hir.mir(
            Op::RmsNorm {
                axis: -1,
                eps: self.eps,
            },
            vec![x, g, beta],
            shape,
        )
    }

    fn per_head_rms(
        &mut self,
        x: HirNodeId,
        gamma: &RmsNormWeight,
        name: &str,
        heads: usize,
    ) -> HirNodeId {
        let hd = self.cfg.head_dim;
        let flat = self.reshape(x, vec![(self.batch * self.seq * heads) as i64, hd as i64]);
        let n = self.rms_norm(
            flat,
            gamma,
            name,
            Shape::new(&[self.batch * self.seq * heads, hd], self.f),
        );
        self.reshape(
            n,
            vec![self.batch as i64, self.seq as i64, (heads * hd) as i64],
        )
    }

    pub(crate) fn layer_forward(
        &mut self,
        layer: &Flux2TextEncoderLayerWeights,
        li: usize,
        x: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
    ) -> Result<HirNodeId> {
        let lp = format!("layers.{li}");
        let shape = self.bsh();
        let normed = self.rms_norm(
            x,
            &layer.input_layernorm,
            &format!("{lp}.in_ln"),
            shape.clone(),
        );
        let attn_out = self.attn_forward(&layer.attn, &format!("{lp}.attn"), normed, cos, sin)?;
        let post_attn = self.add(x, attn_out, shape.clone());
        let mlp_out = self.mlp_forward(
            &layer.mlp,
            &layer.post_attention_layernorm,
            &format!("{lp}.mlp"),
            post_attn,
        )?;
        Ok(self.add(post_attn, mlp_out, shape))
    }

    fn attn_forward(
        &mut self,
        attn: &Flux2TextEncoderAttnWeights,
        tag: &str,
        x: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
    ) -> Result<HirNodeId> {
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        let group = nh / nkv;
        let shape = self.bsh();

        let q = self.linear(x, &attn.q, &format!("{tag}.q"), self.bsh_heads(nh))?;
        let k = self.linear(x, &attn.k, &format!("{tag}.k"), self.bsh_heads(nkv))?;
        let v = self.linear(x, &attn.v, &format!("{tag}.v"), self.bsh_heads(nkv))?;

        let q = self.per_head_rms(q, &attn.q_norm, &format!("{tag}.nq"), nh);
        let k = self.per_head_rms(k, &attn.k_norm, &format!("{tag}.nk"), nkv);

        let qh = self.bsh_heads(nh);
        let q = self.rope(q, cos, sin, qh.clone());
        let k = self.rope(k, cos, sin, self.bsh_heads(nkv));
        let k_rep = self.repeat_kv(k, nkv, hd, group);
        let v_rep = self.repeat_kv(v, nkv, hd, group);

        let attn_out =
            self.hir
                .attention(q, k_rep, v_rep, None, nh, hd, MaskKind::Causal, qh.clone());
        self.linear(attn_out, &attn.o, &format!("{tag}.o"), shape)
    }

    fn mlp_forward(
        &mut self,
        mlp: &Flux2TextEncoderMlpWeights,
        post_ln: &RmsNormWeight,
        tag: &str,
        x: HirNodeId,
    ) -> Result<HirNodeId> {
        let rows = self.batch * self.seq;
        let h = self.cfg.hidden_size;
        let ff = self.cfg.intermediate_size;
        let flat = self.reshape(x, vec![rows as i64, h as i64]);
        let flat = self.rms_norm(
            flat,
            post_ln,
            &format!("{tag}.post_ln"),
            Shape::new(&[rows, h], self.f),
        );
        let gate = self.linear(
            flat,
            &mlp.gate,
            &format!("{tag}.gate"),
            Shape::new(&[rows, ff], self.f),
        )?;
        let up = self.linear(
            flat,
            &mlp.up,
            &format!("{tag}.up"),
            Shape::new(&[rows, ff], self.f),
        )?;
        let gate3 = self.reshape(gate, vec![self.batch as i64, self.seq as i64, ff as i64]);
        let up3 = self.reshape(up, vec![self.batch as i64, self.seq as i64, ff as i64]);
        let silu = self.hir.mir(
            Op::Activation(Activation::Silu),
            vec![gate3],
            Shape::new(&[self.batch, self.seq, ff], self.f),
        );
        let prod = self.mul(silu, up3, Shape::new(&[self.batch, self.seq, ff], self.f));
        let prod_flat = self.reshape(prod, vec![rows as i64, ff as i64]);
        self.linear(
            prod_flat,
            &mlp.down,
            &format!("{tag}.down"),
            Shape::new(&[rows, h], self.f),
        )
        .map(|o| self.reshape(o, vec![self.batch as i64, self.seq as i64, h as i64]))
    }

    fn mul(&mut self, a: HirNodeId, b: HirNodeId, shape: Shape) -> HirNodeId {
        self.hir.mir(Op::Binary(BinaryOp::Mul), vec![a, b], shape)
    }

    fn repeat_kv(&mut self, x: HirNodeId, nkv: usize, hd: usize, group: usize) -> HirNodeId {
        if group == 1 {
            return x;
        }
        let last = 2;
        let slice_shape = Shape::new(&[self.batch, self.seq, hd], self.f);
        let out_shape = Shape::new(&[self.batch, self.seq, nkv * group * hd], self.f);
        let mut pieces = Vec::with_capacity(nkv * group);
        for h in 0..nkv {
            let slice = self.narrow(x, last, h * hd, hd, slice_shape.clone());
            for _ in 0..group {
                pieces.push(slice);
            }
        }
        self.concat(pieces, last, out_shape)
    }

    pub(crate) fn rope_tables(&mut self) -> Result<(HirNodeId, HirNodeId)> {
        let dh = self.cfg.head_dim;
        let half = dh / 2;
        let max_pos = self.cfg.max_position_embeddings;
        let mut cos_data = vec![0f32; max_pos * dh];
        let mut sin_data = vec![0f32; max_pos * dh];
        for pos in 0..max_pos {
            for i in 0..half {
                let freq = 1.0 / self.cfg.rope_theta.powf((2 * i) as f64 / dh as f64);
                let angle = pos as f64 * freq;
                let (s, c) = angle.sin_cos();
                cos_data[pos * dh + 2 * i] = c as f32;
                cos_data[pos * dh + 2 * i + 1] = c as f32;
                sin_data[pos * dh + 2 * i] = s as f32;
                sin_data[pos * dh + 2 * i + 1] = s as f32;
            }
        }
        let cos = self.register_param("rope.cos", cos_data, Shape::new(&[max_pos, dh], self.f));
        let sin = self.register_param("rope.sin", sin_data, Shape::new(&[max_pos, dh], self.f));
        Ok((cos, sin))
    }

    fn rope(&mut self, x: HirNodeId, cos: HirNodeId, sin: HirNodeId, shape: Shape) -> HirNodeId {
        self.hir.mir(
            Op::Rope {
                head_dim: self.cfg.head_dim,
                n_rot: self.cfg.head_dim,
            },
            vec![x, cos, sin],
            shape,
        )
    }

    fn reshape(&mut self, x: HirNodeId, new_shape: Vec<i64>) -> HirNodeId {
        let shape = Shape::new(
            &new_shape.iter().map(|&d| d as usize).collect::<Vec<_>>(),
            self.f,
        );
        self.hir.mir(Op::Reshape { new_shape }, vec![x], shape)
    }

    fn narrow(
        &mut self,
        x: HirNodeId,
        axis: usize,
        start: usize,
        len: usize,
        shape: Shape,
    ) -> HirNodeId {
        self.hir
            .mir(Op::Narrow { axis, start, len }, vec![x], shape)
    }

    fn concat(&mut self, inputs: Vec<HirNodeId>, axis: usize, shape: Shape) -> HirNodeId {
        self.hir.mir(Op::Concat { axis }, inputs, shape)
    }

    fn add(&mut self, a: HirNodeId, b: HirNodeId, shape: Shape) -> HirNodeId {
        self.hir.mir(Op::Binary(BinaryOp::Add), vec![a, b], shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_encoder::{
        TINY_TEXT_ENCODER_LAYERS, encode_prompt_embeds, synthetic_text_encoder_weights,
        tiny_text_encoder_config,
    };
    use rlx_runtime::Device;

    #[test]
    fn text_encoder_hir_lowers() {
        let cfg = tiny_text_encoder_config();
        let w = synthetic_text_encoder_weights(&cfg);
        let g = build_flux2_text_encoder_hir(&cfg, &w, 1, 4, TINY_TEXT_ENCODER_LAYERS).unwrap();
        g.hir.lower_to_mir().expect("lower");
    }

    #[test]
    fn compiled_single_layer_hidden_matches_native() {
        let cfg = tiny_text_encoder_config();
        let w = synthetic_text_encoder_weights(&cfg);
        let layers = [1usize];
        let batch = 1usize;
        let seq = 4usize;
        let ids: Vec<u32> = (0..seq as u32).collect();
        let ids_f32: Vec<f32> = ids.iter().map(|&x| x as f32).collect();
        let native = encode_prompt_embeds(&w, &cfg, &ids, batch, seq, &layers).unwrap();
        let (mut compiled, _) =
            compile_flux2_text_encoder_hir(&cfg, &w, batch, seq, &layers, Device::Cpu, None)
                .unwrap();
        let out = compiled.run(&[("input_ids", ids_f32.as_slice())]).remove(0);
        assert_eq!(out.len(), native.prompt_embeds.len());
        let max = out
            .iter()
            .zip(&native.prompt_embeds)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 2e-2, "single layer max_abs_diff={max}");
    }

    #[test]
    fn compiled_text_encoder_matches_native() {
        let cfg = tiny_text_encoder_config();
        let w = synthetic_text_encoder_weights(&cfg);
        let batch = 1usize;
        let seq = 4usize;
        let ids: Vec<u32> = (0..seq as u32).collect();
        let ids_f32: Vec<f32> = ids.iter().map(|&x| x as f32).collect();
        let layers = TINY_TEXT_ENCODER_LAYERS;

        let native = encode_prompt_embeds(&w, &cfg, &ids, batch, seq, layers).unwrap();

        let (mut compiled, _) =
            compile_flux2_text_encoder_hir(&cfg, &w, batch, seq, layers, Device::Cpu, None)
                .unwrap();
        let out = compiled.run(&[("input_ids", ids_f32.as_slice())]).remove(0);

        assert_eq!(out.len(), native.prompt_embeds.len());
        let max = out
            .iter()
            .zip(&native.prompt_embeds)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 2e-2, "HIR vs native max_abs_diff={max}");
    }
}
