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

//! Compiled acoustic velocity stack (3-token bidirectional transformer, no attention RoPE).

use crate::acoustic::FM_SEQ;
use crate::config::AcousticTransformerArgs;
use anyhow::{Result, ensure};
use rlx_core::flow_util::built_from_hir;
use rlx_core::weight_map::WeightMap;
use rlx_flow::BuiltModel;
use rlx_flow::WeightSource;
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

struct AcousticStackBuilder<'a> {
    hir: &'a mut HirModule,
    params: &'a mut HashMap<String, Vec<f32>>,
    weights: &'a mut dyn WeightSource,
    batch: usize,
    seq: usize,
    f: DType,
}

impl<'a> AcousticStackBuilder<'a> {
    fn g(&mut self) -> HirMut<'_> {
        HirMut::new(self.hir)
    }

    fn register_param(&mut self, name: &str, data: Vec<f32>, shape: &[usize]) -> Result<HirNodeId> {
        let f = self.f;
        let id = self.g().param(name, Shape::new(shape, f));
        self.params.insert(name.to_string(), data);
        Ok(id)
    }

    fn load_weight(&mut self, key: &str, transpose: bool) -> Result<HirNodeId> {
        let (data, shape) = self.weights.take(key, transpose)?;
        self.register_param(key, data, &shape)
    }

    fn linear(&mut self, x: HirNodeId, w_key: &str) -> Result<HirNodeId> {
        let w = self.load_weight(w_key, true)?;
        Ok(self.g().mm(x, w))
    }

    fn rms_norm(
        &mut self,
        x: HirNodeId,
        w_key: &str,
        eps: f32,
        zero_beta: HirNodeId,
    ) -> Result<HirNodeId> {
        let gamma = self.load_weight(w_key, false)?;
        Ok(self.g().rms_norm(x, gamma, zero_beta, eps))
    }

    fn repeat_kv(
        &mut self,
        x: HirNodeId,
        num_kv_heads: usize,
        head_dim: usize,
        group: usize,
    ) -> HirNodeId {
        if group == 1 {
            return x;
        }
        let last_ax = self.g().shape(x).rank() - 1;
        let mut pieces = Vec::with_capacity(num_kv_heads * group);
        for h in 0..num_kv_heads {
            let slice = self.g().narrow_(x, last_ax, h * head_dim, head_dim);
            for _ in 0..group {
                pieces.push(slice);
            }
        }
        self.g().concat_(pieces, last_ax)
    }

    fn mha(
        &mut self,
        x: HirNodeId,
        layer: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<HirNodeId> {
        let lp = format!("layers.{layer}");
        let q_dim = n_heads * head_dim;
        let q = self.linear(x, &format!("{lp}.attention.wq.weight"))?;
        let k = self.linear(x, &format!("{lp}.attention.wk.weight"))?;
        let v = self.linear(x, &format!("{lp}.attention.wv.weight"))?;
        let group = n_heads / n_kv_heads;
        let k_rep = self.repeat_kv(k, n_kv_heads, head_dim, group);
        let v_rep = self.repeat_kv(v, n_kv_heads, head_dim, group);
        let out_shape = Shape::new(&[self.batch, self.seq, q_dim], self.f);
        let attn = self.g().attention_kind(
            q,
            k_rep,
            v_rep,
            n_heads,
            head_dim,
            MaskKind::None,
            out_shape,
        );
        self.linear(attn, &format!("{lp}.attention.wo.weight"))
    }

    fn swiglu_ffn(
        &mut self,
        x: HirNodeId,
        layer: usize,
        _hidden: usize,
        _inter: usize,
    ) -> Result<HirNodeId> {
        let lp = format!("layers.{layer}");
        let gate = self.linear(x, &format!("{lp}.feed_forward.w1.weight"))?;
        let up = self.linear(x, &format!("{lp}.feed_forward.w3.weight"))?;
        let gate_act = self.g().silu(gate);
        let swiglu = self.g().mul(gate_act, up);
        self.linear(swiglu, &format!("{lp}.feed_forward.w2.weight"))
    }

    fn layer(
        &mut self,
        layer: usize,
        x: HirNodeId,
        args: &AcousticTransformerArgs,
    ) -> Result<HirNodeId> {
        let hidden = args.dim;
        let eps = args.norm_eps as f32;
        let zero = self.register_param(
            &format!("acoustic.zero_beta.{layer}"),
            vec![0.0; hidden],
            &[hidden],
        )?;
        let zero_bc = self.g().reshape_(zero, vec![1, 1, hidden as i64]);

        let normed = self.rms_norm(
            x,
            &format!("layers.{layer}.attention_norm.weight"),
            eps,
            zero_bc,
        )?;
        let attn = self.mha(normed, layer, args.n_heads, args.n_kv_heads, args.head_dim)?;
        let post_attn = self.g().add(x, attn);

        let normed_ff = self.rms_norm(
            post_attn,
            &format!("layers.{layer}.ffn_norm.weight"),
            eps,
            zero_bc,
        )?;
        let ff = self.swiglu_ffn(normed_ff, layer, hidden, args.hidden_dim)?;
        Ok(self.g().add(post_attn, ff))
    }

    fn emit_stack(
        &mut self,
        args: &AcousticTransformerArgs,
        hidden_in: HirNodeId,
    ) -> Result<HirNodeId> {
        let mut x = hidden_in;
        for i in 0..args.n_layers {
            x = self.layer(i, x, args)?;
        }
        let hidden = args.dim;
        let eps = args.norm_eps as f32;
        let zero = self.register_param("acoustic.zero_beta.final", vec![0.0; hidden], &[hidden])?;
        let zero_bc = self.g().reshape_(zero, vec![1, 1, hidden as i64]);
        let normed = self.rms_norm(x, "norm.weight", eps, zero_bc)?;
        let seq = self.seq;
        let all = self
            .g()
            .reshape_(normed, vec![1, seq as i64, hidden as i64]);
        let row0 = self.g().narrow_(all, 1, 0, 1);
        let flat = self.g().reshape_(row0, vec![1, hidden as i64]);
        self.linear(flat, "acoustic_codebook_output.weight")
    }
}

fn build_acoustic_velocity_hir(
    args: &AcousticTransformerArgs,
    weights: &mut dyn WeightSource,
    batch: usize,
    seq: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    ensure!(batch > 0 && seq > 0, "batch/seq must be positive");
    ensure!(seq == FM_SEQ, "acoustic FM stack expects seq={FM_SEQ}");
    let hidden = args.dim;
    ensure!(hidden > 0, "acoustic dim must be > 0");

    let f = DType::F32;
    let mut hir = HirModule::new("voxtral_tts_acoustic_velocity");
    let mut params = HashMap::new();
    let hidden_shape = Shape::new(&[batch, seq, hidden], f);
    let hidden_in = hir.input("hidden", hidden_shape);

    let mut b = AcousticStackBuilder {
        hir: &mut hir,
        params: &mut params,
        weights,
        batch,
        seq,
        f,
    };
    let out = b.emit_stack(args, hidden_in)?;
    hir.outputs = vec![out];
    Ok((hir, params))
}

pub fn build_acoustic_velocity_built(
    args: &AcousticTransformerArgs,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
) -> Result<BuiltModel> {
    use rlx_core::flow_util::WeightMapSource;
    let (hir, params) =
        build_acoustic_velocity_hir(args, &mut WeightMapSource(weights), batch, seq)?;
    built_from_hir(hir, params)
}
