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

//! Voxtral audio encoder HIR (Whisper-style conv + transformer).

use crate::config::VoxtralAudioConfig;
use crate::weights::VoxtralWeightPrefix;
use anyhow::{Result, ensure};
use rlx_flow::WeightSource;
use rlx_ir::hir::{FusionPolicy, HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Op, Shape};
use std::collections::HashMap;

const LN_EPS: f32 = 1e-5;

pub(crate) struct AudioEncoderBuilder<'a> {
    pub hir: &'a mut HirModule,
    pub params: &'a mut HashMap<String, Vec<f32>>,
    pub weights: &'a mut dyn WeightSource,
    pub batch: usize,
    pub f: DType,
}

impl<'a> AudioEncoderBuilder<'a> {
    fn g(&mut self) -> HirMut<'_> {
        HirMut::new(self.hir)
    }

    pub(crate) fn emit_encoder_through_conv1(
        &mut self,
        cfg: &VoxtralAudioConfig,
        mel: HirNodeId,
        mel_frames: usize,
        gelu: bool,
    ) -> Result<HirNodeId> {
        if gelu {
            self.conv1d_gelu(
                mel,
                VoxtralWeightPrefix::enc_conv1_w(),
                VoxtralWeightPrefix::enc_conv1_b(),
                cfg.num_mel_bins,
                cfg.d_model,
                mel_frames,
                3,
                1,
                1,
            )
        } else {
            self.conv1d(
                mel,
                VoxtralWeightPrefix::enc_conv1_w(),
                VoxtralWeightPrefix::enc_conv1_b(),
                cfg.num_mel_bins,
                cfg.d_model,
                mel_frames,
                3,
                1,
                1,
            )
        }
    }

    pub(crate) fn emit_encoder_through_conv2(
        &mut self,
        cfg: &VoxtralAudioConfig,
        mel: HirNodeId,
        mel_frames: usize,
    ) -> Result<HirNodeId> {
        let d = cfg.d_model;
        let x = self.emit_encoder_through_conv1(cfg, mel, mel_frames, true)?;
        self.conv1d_gelu(
            x,
            VoxtralWeightPrefix::enc_conv2_w(),
            VoxtralWeightPrefix::enc_conv2_b(),
            d,
            d,
            mel_frames,
            3,
            2,
            1,
        )
    }

    pub(crate) fn emit_encoder_preamble(
        &mut self,
        cfg: &VoxtralAudioConfig,
        mel: HirNodeId,
        mel_frames: usize,
        enc_seq: usize,
    ) -> Result<HirNodeId> {
        let d = cfg.d_model;
        let mut x = self.emit_encoder_through_conv2(cfg, mel, mel_frames)?;
        x = self.g().transpose_(x, vec![0, 2, 1]);
        let pos_w = self.load_param(VoxtralWeightPrefix::enc_embed_positions(), false)?;
        let pos = self.g().narrow_(pos_w, 0, 0, enc_seq);
        let pos_bc = self.broadcast_pos(pos, enc_seq, d)?;
        Ok(self.g().add(x, pos_bc))
    }

    pub(crate) fn emit_encoder_inner(
        &mut self,
        cfg: &VoxtralAudioConfig,
        mel: HirNodeId,
        mel_frames: usize,
        enc_seq: usize,
    ) -> Result<HirNodeId> {
        let d = cfg.d_model;
        let mut x = self.emit_encoder_preamble(cfg, mel, mel_frames, enc_seq)?;
        for i in 0..cfg.encoder_layers {
            x = self.residual_block(cfg, i, x, enc_seq, cfg.encoder_attention_heads)?;
        }
        self.layer_norm(
            x,
            VoxtralWeightPrefix::enc_ln_post_w(),
            VoxtralWeightPrefix::enc_ln_post_b(),
            enc_seq,
            d,
        )
    }

    fn residual_block(
        &mut self,
        cfg: &VoxtralAudioConfig,
        layer: usize,
        x: HirNodeId,
        seq: usize,
        n_head: usize,
    ) -> Result<HirNodeId> {
        let d = cfg.d_model;
        let hd = d / n_head;
        let attn_ln_w = VoxtralWeightPrefix::enc_layer(layer, "self_attn_layer_norm.weight");
        let attn_ln_b = VoxtralWeightPrefix::enc_layer(layer, "self_attn_layer_norm.bias");
        let ln_x = self.layer_norm(x, &attn_ln_w, &attn_ln_b, seq, d)?;
        let layer_pfx = |suffix: &str| VoxtralWeightPrefix::enc_layer(layer, suffix);
        let sa = self.mha(
            ln_x,
            ln_x,
            &layer_pfx("self_attn.q_proj.weight"),
            Some(layer_pfx("self_attn.q_proj.bias").as_str()),
            &layer_pfx("self_attn.k_proj.weight"),
            None,
            &layer_pfx("self_attn.v_proj.weight"),
            layer_pfx("self_attn.v_proj.bias").as_str(),
            &layer_pfx("self_attn.out_proj.weight"),
            layer_pfx("self_attn.out_proj.bias").as_str(),
            seq,
            n_head,
            hd,
            MaskKind::None,
        )?;
        let x = self.g().add(x, sa);
        let mlp_ln_w = layer_pfx("final_layer_norm.weight");
        let mlp_ln_b = layer_pfx("final_layer_norm.bias");
        let ln_x = self.layer_norm(x, &mlp_ln_w, &mlp_ln_b, seq, d)?;
        let mlp = self.mlp(
            cfg,
            ln_x,
            &layer_pfx("fc1.weight"),
            layer_pfx("fc1.bias").as_str(),
            &layer_pfx("fc2.weight"),
            layer_pfx("fc2.bias").as_str(),
            seq,
            d,
        )?;
        Ok(self.g().add(x, mlp))
    }

    fn mha(
        &mut self,
        x: HirNodeId,
        kv_src: HirNodeId,
        qw: &str,
        qb: Option<&str>,
        kw: &str,
        kb: Option<&str>,
        vw: &str,
        vb: &str,
        ow: &str,
        ob: &str,
        seq: usize,
        n_head: usize,
        head_dim: usize,
        mask: MaskKind,
    ) -> Result<HirNodeId> {
        let d = n_head * head_dim;
        let q = self.linear(x, qw, qb, seq, d, d)?;
        let k = self.linear(kv_src, kw, kb, self.kv_seq(kv_src), d, d)?;
        let v = self.linear(kv_src, vw, Some(vb), self.kv_seq(kv_src), d, d)?;
        let out_shape = Shape::new(&[self.batch, seq, d], self.f);
        let attn = self
            .g()
            .attention_kind(q, k, v, n_head, head_dim, mask, out_shape);
        self.linear(attn, ow, Some(ob), seq, d, d)
    }

    fn kv_seq(&self, x: HirNodeId) -> usize {
        self.hir.node(x).shape.dim(1).unwrap_static()
    }

    fn mlp(
        &mut self,
        cfg: &VoxtralAudioConfig,
        x: HirNodeId,
        w1: &str,
        b1: &str,
        w2: &str,
        b2: &str,
        seq: usize,
        d: usize,
    ) -> Result<HirNodeId> {
        let mlp_dim = cfg.intermediate_size;
        let h1 = self.linear(x, w1, Some(b1), seq, mlp_dim, d)?;
        let h1 = self.g().gelu(h1);
        self.linear(h1, w2, Some(b2), seq, d, mlp_dim)
    }

    fn linear(
        &mut self,
        x: HirNodeId,
        w_key: &str,
        b_key: Option<&str>,
        _seq: usize,
        out_f: usize,
        _in_f: usize,
    ) -> Result<HirNodeId> {
        let w = self.load_param(w_key, true)?;
        let mut y = self.g().mm(x, w);
        if let Some(bk) = b_key {
            let b = self.load_param(bk, false)?;
            let b3 = self.g().reshape_(b, vec![1, 1, out_f as i64]);
            y = self.g().add(y, b3);
        }
        Ok(y)
    }

    fn layer_norm(
        &mut self,
        x: HirNodeId,
        w: &str,
        b: &str,
        _seq: usize,
        _d: usize,
    ) -> Result<HirNodeId> {
        let gamma = self.load_param(w, false)?;
        let beta = self.load_param(b, false)?;
        Ok(self.g().ln(x, gamma, beta, LN_EPS))
    }

    fn broadcast_bias(&mut self, bias: HirNodeId, out_c: usize, t_out: usize) -> Result<HirNodeId> {
        let batch = self.batch;
        let bias3 = self.g().reshape_(bias, vec![1, out_c as i64, 1]);
        let ones = self.register_param(
            &format!("conv_bias_batch_{out_c}_{t_out}"),
            vec![1.0; batch],
            &[batch],
        )?;
        let ones2 = self.g().reshape_(ones, vec![batch as i64, 1, 1]);
        let bias_bc = self.g().mul(bias3, ones2);
        let time = self.register_param(
            &format!("conv_bias_time_{t_out}"),
            vec![1.0; t_out],
            &[t_out],
        )?;
        let time3 = self.g().reshape_(time, vec![1, 1, t_out as i64]);
        Ok(self.g().mul(bias_bc, time3))
    }

    fn conv1d(
        &mut self,
        input: HirNodeId,
        w_key: &str,
        b_key: &str,
        in_c: usize,
        out_c: usize,
        t_in: usize,
        k: usize,
        stride: usize,
        pad: usize,
    ) -> Result<HirNodeId> {
        let batch = self.batch;
        let f = self.f;
        let t_out = (t_in + 2 * pad - k) / stride + 1;
        let nchw = self
            .g()
            .reshape_(input, vec![batch as i64, in_c as i64, t_in as i64, 1]);
        let (w_data, _) = self.weights.take(w_key, false)?;
        let w = self.register_param(
            w_key,
            pack_conv1d_weight(&w_data, out_c, in_c, k),
            &[out_c, in_c, k, 1],
        )?;
        let conv = self.g().add_node(
            Op::Conv {
                kernel_size: vec![k, 1],
                stride: vec![stride, 1],
                padding: vec![pad, 0],
                dilation: vec![1, 1],
                groups: 1,
            },
            vec![nchw, w],
            Shape::new(&[batch, out_c, t_out, 1], f),
        );
        let mut out = self
            .g()
            .reshape_(conv, vec![batch as i64, out_c as i64, t_out as i64]);
        let (b_data, _) = self.weights.take(b_key, false)?;
        let bias = self.register_param(b_key, b_data, &[out_c])?;
        let b3 = self.broadcast_bias(bias, out_c, t_out)?;
        out = self.g().add(out, b3);
        Ok(out)
    }

    fn conv1d_gelu(
        &mut self,
        input: HirNodeId,
        w_key: &str,
        b_key: &str,
        in_c: usize,
        out_c: usize,
        t_in: usize,
        k: usize,
        stride: usize,
        pad: usize,
    ) -> Result<HirNodeId> {
        let x = self.conv1d(input, w_key, b_key, in_c, out_c, t_in, k, stride, pad)?;
        Ok(self.g().gelu(x))
    }

    fn broadcast_pos(&mut self, pos: HirNodeId, seq: usize, d: usize) -> Result<HirNodeId> {
        let batch = self.batch;
        let pos3 = self.g().reshape_(pos, vec![1, seq as i64, d as i64]);
        let ones =
            self.register_param(&format!("pos_broadcast_{seq}"), vec![1.0; batch], &[batch])?;
        let ones2 = self.g().reshape_(ones, vec![batch as i64, 1, 1]);
        Ok(self.g().mul(pos3, ones2))
    }

    fn load_param(&mut self, key: &str, transpose: bool) -> Result<HirNodeId> {
        let (data, shape) = self.weights.take(key, transpose)?;
        let id = self.hir.param(key, Shape::new(&shape, self.f));
        self.params.insert(key.to_string(), data);
        Ok(id)
    }

    fn register_param(
        &mut self,
        key: &str,
        data: Vec<f32>,
        shape_dims: &[usize],
    ) -> Result<HirNodeId> {
        let id = self.hir.param(key, Shape::new(shape_dims, self.f));
        self.params.insert(key.to_string(), data);
        Ok(id)
    }
}

fn pack_conv1d_weight(raw: &[f32], _out_c: usize, _in_c: usize, _k: usize) -> Vec<f32> {
    raw.to_vec()
}

pub fn build_voxtral_encoder_conv1_built(
    cfg: &VoxtralAudioConfig,
    weights: &mut rlx_core::weight_map::WeightMap,
    batch: usize,
    mel_frames: usize,
    gelu: bool,
) -> Result<rlx_flow::BuiltModel> {
    use rlx_core::flow_util::WeightMapSource;
    validate_cfg(cfg)?;
    let f = DType::F32;
    let mut hir = HirModule::new("voxtral_encoder_conv1").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let mel = hir.input("mel", Shape::new(&[batch, cfg.num_mel_bins, mel_frames], f));
    let mut b = AudioEncoderBuilder {
        hir: &mut hir,
        params: &mut params,
        weights: &mut WeightMapSource(weights),
        batch,
        f,
    };
    let hidden = b.emit_encoder_through_conv1(cfg, mel, mel_frames, gelu)?;
    hir.outputs = vec![hidden];
    rlx_core::flow_util::built_from_hir(hir, params)
}

pub fn build_voxtral_encoder_conv2_built(
    cfg: &VoxtralAudioConfig,
    weights: &mut rlx_core::weight_map::WeightMap,
    batch: usize,
    mel_frames: usize,
) -> Result<rlx_flow::BuiltModel> {
    use rlx_core::flow_util::WeightMapSource;
    validate_cfg(cfg)?;
    let f = DType::F32;
    let mut hir = HirModule::new("voxtral_encoder_conv2").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let mel = hir.input("mel", Shape::new(&[batch, cfg.num_mel_bins, mel_frames], f));
    let mut b = AudioEncoderBuilder {
        hir: &mut hir,
        params: &mut params,
        weights: &mut WeightMapSource(weights),
        batch,
        f,
    };
    let hidden = b.emit_encoder_through_conv2(cfg, mel, mel_frames)?;
    hir.outputs = vec![hidden];
    let (hir, params) = (hir, params);
    rlx_core::flow_util::built_from_hir(hir, params)
}

pub fn build_voxtral_encoder_stem_hir(
    cfg: &VoxtralAudioConfig,
    weights: &mut dyn WeightSource,
    batch: usize,
    mel_frames: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;
    let enc_seq = cfg.encoder_seq_len(mel_frames);
    ensure!(
        enc_seq <= cfg.max_source_positions,
        "mel frames {mel_frames} -> encoder seq {enc_seq} exceeds max_source_positions {}",
        cfg.max_source_positions
    );
    let f = DType::F32;
    let mut hir = HirModule::new("voxtral_encoder_stem").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let mel = hir.input("mel", Shape::new(&[batch, cfg.num_mel_bins, mel_frames], f));
    let mut b = AudioEncoderBuilder {
        hir: &mut hir,
        params: &mut params,
        weights,
        batch,
        f,
    };
    let hidden = b.emit_encoder_preamble(cfg, mel, mel_frames, enc_seq)?;
    hir.outputs = vec![hidden];
    Ok((hir, params))
}

pub fn build_voxtral_encoder_stem_built(
    cfg: &VoxtralAudioConfig,
    weights: &mut rlx_core::weight_map::WeightMap,
    batch: usize,
    mel_frames: usize,
) -> Result<rlx_flow::BuiltModel> {
    use rlx_core::flow_util::WeightMapSource;
    let (hir, params) =
        build_voxtral_encoder_stem_hir(cfg, &mut WeightMapSource(weights), batch, mel_frames)?;
    rlx_core::flow_util::built_from_hir(hir, params)
}

pub fn build_voxtral_encoder_hir(
    cfg: &VoxtralAudioConfig,
    weights: &mut dyn WeightSource,
    batch: usize,
    mel_frames: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;
    let enc_seq = cfg.encoder_seq_len(mel_frames);
    ensure!(
        enc_seq <= cfg.max_source_positions,
        "mel frames {mel_frames} -> encoder seq {enc_seq} exceeds max_source_positions {}",
        cfg.max_source_positions
    );
    ensure!(
        enc_seq.is_multiple_of(4),
        "encoder seq {enc_seq} must be divisible by 4 for the multimodal projector"
    );
    let f = DType::F32;
    let mut hir = HirModule::new("voxtral_encoder").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let mel = hir.input("mel", Shape::new(&[batch, cfg.num_mel_bins, mel_frames], f));
    let mut b = AudioEncoderBuilder {
        hir: &mut hir,
        params: &mut params,
        weights,
        batch,
        f,
    };
    let hidden = b.emit_encoder_inner(cfg, mel, mel_frames, enc_seq)?;
    hir.outputs = vec![hidden];
    Ok((hir, params))
}

pub fn build_voxtral_encoder_built(
    cfg: &VoxtralAudioConfig,
    weights: &mut rlx_core::weight_map::WeightMap,
    batch: usize,
    mel_frames: usize,
) -> Result<rlx_flow::BuiltModel> {
    use rlx_core::flow_util::WeightMapSource;
    let (hir, params) =
        build_voxtral_encoder_hir(cfg, &mut WeightMapSource(weights), batch, mel_frames)?;
    rlx_core::flow_util::built_from_hir(hir, params)
}

fn validate_cfg(cfg: &VoxtralAudioConfig) -> Result<()> {
    ensure!(cfg.d_model > 0, "d_model must be > 0");
    ensure!(cfg.encoder_layers > 0, "encoder_layers must be > 0");
    ensure!(
        cfg.d_model.is_multiple_of(cfg.encoder_attention_heads),
        "encoder head dim"
    );
    Ok(())
}
