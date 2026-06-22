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

//! End-to-end smoke tests with synthetic weights: every model graph is built,
//! compiled, and run on each enabled backend, asserting finite outputs of the
//! right shape. This exercises the SAN-M / FSMN / CIF / decoder / conv paths
//! and confirms the internal key/shape contracts are consistent.

// Tests start from a default config and override a few fields for a tiny model.
#![allow(clippy::field_reassign_with_default)]

use std::collections::HashMap;

use rlx_core::weight_map::WeightMap;
use rlx_funasr::config::{
    CamPlusConfig, CtTransformerConfig, FsmnVadConfig, ParaformerConfig, SanmEncoderConfig,
    SenseVoiceConfig,
};
use rlx_funasr::paraformer::Paraformer;
use rlx_funasr::punc::CtTransformer;
use rlx_funasr::sensevoice::SenseVoice;
use rlx_funasr::speaker::CamPlus;
use rlx_funasr::vad::FsmnVad;
use rlx_runtime::Device;

type Tensors = HashMap<String, (Vec<f32>, Vec<usize>)>;

fn devices() -> Vec<Device> {
    #[allow(unused_mut)] // additional devices are pushed only under backend features
    let mut d = vec![Device::Cpu];
    #[cfg(feature = "metal")]
    d.push(Device::Metal);
    #[cfg(feature = "mlx")]
    d.push(Device::Mlx);
    #[cfg(feature = "gpu")]
    d.push(Device::Gpu);
    d
}

fn put(m: &mut Tensors, key: &str, shape: &[usize]) {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (((i % 17) as f32) - 8.0) * 0.02).collect();
    m.insert(key.to_string(), (data, shape.to_vec()));
}

fn bn1(m: &mut Tensors, prefix: &str, c: usize, affine: bool) {
    put(m, &format!("{prefix}.running_mean"), &[c]);
    put(m, &format!("{prefix}.running_var"), &[c]); // small positive via |x|+1 below
    // ensure variance is positive
    let key = format!("{prefix}.running_var");
    if let Some((d, _)) = m.get_mut(&key) {
        for v in d.iter_mut() {
            *v = v.abs() + 1.0;
        }
    }
    if affine {
        put(m, &format!("{prefix}.weight"), &[c]);
        put(m, &format!("{prefix}.bias"), &[c]);
    }
}

fn tiny_sanm() -> SanmEncoderConfig {
    SanmEncoderConfig {
        input_size: 20,
        output_size: 16,
        n_heads: 2,
        linear_units: 32,
        num_blocks: 2,
        tp_blocks: 1,
        kernel_size: 3,
        sanm_shfit: 0,
        ln_eps: 1e-12,
    }
}

fn sanm_layer_keys(m: &mut Tensors, p: &str, in_size: usize, size: usize, units: usize, k: usize) {
    put(m, &format!("{p}.norm1.weight"), &[in_size]);
    put(m, &format!("{p}.norm1.bias"), &[in_size]);
    put(
        m,
        &format!("{p}.self_attn.linear_q_k_v.weight"),
        &[3 * size, in_size],
    );
    put(m, &format!("{p}.self_attn.linear_q_k_v.bias"), &[3 * size]);
    put(
        m,
        &format!("{p}.self_attn.linear_out.weight"),
        &[size, size],
    );
    put(m, &format!("{p}.self_attn.linear_out.bias"), &[size]);
    put(
        m,
        &format!("{p}.self_attn.fsmn_block.weight"),
        &[size, 1, k],
    );
    put(m, &format!("{p}.feed_forward.w_1.weight"), &[units, size]);
    put(m, &format!("{p}.feed_forward.w_1.bias"), &[units]);
    put(m, &format!("{p}.feed_forward.w_2.weight"), &[size, units]);
    put(m, &format!("{p}.feed_forward.w_2.bias"), &[size]);
    put(m, &format!("{p}.norm2.weight"), &[size]);
    put(m, &format!("{p}.norm2.bias"), &[size]);
}

fn sanm_encoder_keys(m: &mut Tensors, prefix: &str, cfg: &SanmEncoderConfig, use_tp: bool) {
    let d = cfg.output_size;
    let k = cfg.kernel_size;
    let u = cfg.linear_units;
    sanm_layer_keys(m, &format!("{prefix}.encoders0.0"), cfg.input_size, d, u, k);
    for i in 0..cfg.num_blocks.saturating_sub(1) {
        sanm_layer_keys(m, &format!("{prefix}.encoders.{i}"), d, d, u, k);
    }
    put(m, &format!("{prefix}.after_norm.weight"), &[d]);
    put(m, &format!("{prefix}.after_norm.bias"), &[d]);
    if use_tp {
        for i in 0..cfg.tp_blocks {
            sanm_layer_keys(m, &format!("{prefix}.tp_encoders.{i}"), d, d, u, k);
        }
        put(m, &format!("{prefix}.tp_norm.weight"), &[d]);
        put(m, &format!("{prefix}.tp_norm.bias"), &[d]);
    }
}

#[test]
fn sensevoice_runs_all_backends() {
    let mut cfg = SenseVoiceConfig::default();
    cfg.encoder = tiny_sanm();
    cfg.vocab_size = 12;
    let d = cfg.encoder.output_size;
    let mut m = Tensors::new();
    sanm_encoder_keys(&mut m, "encoder", &cfg.encoder, true);
    put(&mut m, "ctc.ctc_lo.weight", &[cfg.vocab_size, d]);
    put(&mut m, "ctc.ctc_lo.bias", &[cfg.vocab_size]);

    let t = 6usize;
    let feats: Vec<f32> = (0..t * cfg.encoder.input_size)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
        .collect();
    for dev in devices() {
        let model = SenseVoice::from_parts(cfg.clone(), WeightMap::from_tensors(m.clone()), dev);
        let logits = model.run_logits(&feats, t).expect("sensevoice run");
        assert_eq!(logits.len(), t * cfg.vocab_size, "device {dev:?}");
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "device {dev:?} non-finite"
        );
    }
}

#[test]
fn paraformer_runs_all_backends() {
    let mut cfg = ParaformerConfig::default();
    cfg.encoder = SanmEncoderConfig {
        tp_blocks: 0,
        ..tiny_sanm()
    };
    let d = cfg.encoder.output_size;
    cfg.predictor.idim = d;
    cfg.decoder.dim = d;
    cfg.decoder.n_heads = 2;
    cfg.decoder.linear_units = 32;
    cfg.decoder.num_blocks = 2;
    cfg.decoder.att_layer_num = 1; // 1 cross + 1 fsmn-only + 1 ff-only
    cfg.decoder.self_kernel = 3;
    cfg.vocab_size = 12;

    let mut m = Tensors::new();
    sanm_encoder_keys(&mut m, "encoder", &cfg.encoder, false);
    // predictor head
    let kc = cfg.predictor.l_order + cfg.predictor.r_order + 1;
    put(&mut m, "predictor.cif_conv1d.weight", &[d, d, kc]);
    put(&mut m, "predictor.cif_conv1d.bias", &[d]);
    put(&mut m, "predictor.cif_output.weight", &[1, d]);
    put(&mut m, "predictor.cif_output.bias", &[1]);
    // decoder
    decoder_keys(&mut m, &cfg);

    let t = 8usize;
    let feats: Vec<f32> = (0..t * cfg.encoder.input_size)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
        .collect();
    for dev in devices() {
        let model = Paraformer::from_parts(cfg.clone(), WeightMap::from_tensors(m.clone()), dev);
        let enc = model.encode(&feats, t).expect("encode");
        assert_eq!(enc.len(), t * d);
        assert!(
            enc.iter().all(|x| x.is_finite()),
            "device {dev:?} encoder non-finite"
        );
        // decode with a synthetic 3-token acoustic sequence
        let l = 3usize;
        let acoustic: Vec<f32> = (0..l * d).map(|i| ((i % 7) as f32 - 3.0) * 0.04).collect();
        let logits = model.decode_logits(&enc, t, &acoustic, l).expect("decode");
        assert_eq!(logits.len(), l * cfg.vocab_size);
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "device {dev:?} decoder non-finite"
        );
    }
}

fn decoder_keys(m: &mut Tensors, cfg: &ParaformerConfig) {
    let d = cfg.decoder.dim;
    let u = cfg.decoder.linear_units;
    let k = cfg.decoder.self_kernel;
    let ff = |m: &mut Tensors, p: &str| {
        put(m, &format!("{p}.feed_forward.w_1.weight"), &[u, d]);
        put(m, &format!("{p}.feed_forward.w_1.bias"), &[u]);
        put(m, &format!("{p}.feed_forward.norm.weight"), &[u]);
        put(m, &format!("{p}.feed_forward.norm.bias"), &[u]);
        put(m, &format!("{p}.feed_forward.w_2.weight"), &[d, u]);
        put(m, &format!("{p}.norm1.weight"), &[d]);
        put(m, &format!("{p}.norm1.bias"), &[d]);
    };
    // cross-attn layers
    for i in 0..cfg.decoder.att_layer_num {
        let p = format!("decoder.decoders.{i}");
        ff(m, &p);
        put(m, &format!("{p}.norm2.weight"), &[d]);
        put(m, &format!("{p}.norm2.bias"), &[d]);
        put(m, &format!("{p}.self_attn.fsmn_block.weight"), &[d, 1, k]);
        put(m, &format!("{p}.norm3.weight"), &[d]);
        put(m, &format!("{p}.norm3.bias"), &[d]);
        put(m, &format!("{p}.src_attn.linear_q.weight"), &[d, d]);
        put(m, &format!("{p}.src_attn.linear_q.bias"), &[d]);
        put(m, &format!("{p}.src_attn.linear_k_v.weight"), &[2 * d, d]);
        put(m, &format!("{p}.src_attn.linear_k_v.bias"), &[2 * d]);
        put(m, &format!("{p}.src_attn.linear_out.weight"), &[d, d]);
        put(m, &format!("{p}.src_attn.linear_out.bias"), &[d]);
    }
    // fsmn-only layers
    for i in 0..cfg
        .decoder
        .num_blocks
        .saturating_sub(cfg.decoder.att_layer_num)
    {
        let p = format!("decoder.decoders2.{i}");
        ff(m, &p);
        put(m, &format!("{p}.norm2.weight"), &[d]);
        put(m, &format!("{p}.norm2.bias"), &[d]);
        put(m, &format!("{p}.self_attn.fsmn_block.weight"), &[d, 1, k]);
    }
    // ff-only layer
    ff(m, "decoder.decoders3.0");
    put(m, "decoder.after_norm.weight", &[d]);
    put(m, "decoder.after_norm.bias", &[d]);
    put(m, "decoder.output_layer.weight", &[cfg.vocab_size, d]);
    put(m, "decoder.output_layer.bias", &[cfg.vocab_size]);
}

#[test]
fn fsmn_vad_runs_all_backends() {
    let mut cfg = FsmnVadConfig::default();
    cfg.input_dim = 20;
    cfg.input_affine_dim = 12;
    cfg.fsmn_layers = 2;
    cfg.linear_dim = 16;
    cfg.proj_dim = 8;
    cfg.lorder = 3;
    cfg.lstride = 1;
    cfg.output_affine_dim = 12;
    cfg.output_dim = 4;

    let mut m = Tensors::new();
    put(
        &mut m,
        "encoder.in_linear1.linear.weight",
        &[cfg.input_affine_dim, cfg.input_dim],
    );
    put(
        &mut m,
        "encoder.in_linear1.linear.bias",
        &[cfg.input_affine_dim],
    );
    put(
        &mut m,
        "encoder.in_linear2.linear.weight",
        &[cfg.linear_dim, cfg.input_affine_dim],
    );
    put(&mut m, "encoder.in_linear2.linear.bias", &[cfg.linear_dim]);
    for i in 0..cfg.fsmn_layers {
        let p = format!("encoder.fsmn.{i}");
        put(
            &mut m,
            &format!("{p}.linear.linear.weight"),
            &[cfg.proj_dim, cfg.linear_dim],
        );
        put(
            &mut m,
            &format!("{p}.fsmn_block.conv_left.weight"),
            &[cfg.proj_dim, 1, cfg.lorder, 1],
        );
        put(
            &mut m,
            &format!("{p}.affine.linear.weight"),
            &[cfg.linear_dim, cfg.proj_dim],
        );
        put(
            &mut m,
            &format!("{p}.affine.linear.bias"),
            &[cfg.linear_dim],
        );
    }
    put(
        &mut m,
        "encoder.out_linear1.linear.weight",
        &[cfg.output_affine_dim, cfg.linear_dim],
    );
    put(
        &mut m,
        "encoder.out_linear1.linear.bias",
        &[cfg.output_affine_dim],
    );
    put(
        &mut m,
        "encoder.out_linear2.linear.weight",
        &[cfg.output_dim, cfg.output_affine_dim],
    );
    put(&mut m, "encoder.out_linear2.linear.bias", &[cfg.output_dim]);

    let t = 10usize;
    let feats: Vec<f32> = (0..t * cfg.input_dim)
        .map(|i| ((i % 9) as f32 - 4.0) * 0.05)
        .collect();
    for dev in devices() {
        let model = FsmnVad::from_parts(cfg.clone(), WeightMap::from_tensors(m.clone()), dev);
        let out = model.run_logits(&feats, t).expect("vad run");
        assert_eq!(out.len(), t * cfg.output_dim, "device {dev:?}");
        assert!(
            out.iter().all(|x| x.is_finite()),
            "device {dev:?} non-finite"
        );
    }
}

#[test]
fn ct_transformer_runs_cpu() {
    let mut cfg = CtTransformerConfig::default();
    cfg.embed_unit = 16;
    cfg.encoder = SanmEncoderConfig {
        input_size: 16,
        ..tiny_sanm()
    };
    cfg.encoder.tp_blocks = 0;
    let d = cfg.encoder.output_size;
    let punc = cfg.punc_list.len();

    let mut m = Tensors::new();
    sanm_encoder_keys(&mut m, "encoder", &cfg.encoder, false);
    put(&mut m, "decoder.weight", &[punc, d]);
    put(&mut m, "decoder.bias", &[punc]);

    let t = 5usize;
    let embed: Vec<f32> = (0..t * cfg.embed_unit)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let model = CtTransformer::from_parts(cfg.clone(), WeightMap::from_tensors(m), Device::Cpu);
    let punc_ids = model.run_punc(&embed, t).expect("punc run");
    assert_eq!(punc_ids.len(), t);
    assert!(punc_ids.iter().all(|&p| (p as usize) < punc));
}

#[test]
fn campplus_runs_cpu() {
    let mut cfg = CamPlusConfig::default();
    cfg.feat_dim = 8; // F: 8 -> 1 after /8
    cfg.growth_rate = 4;
    cfg.bn_size = 1;
    cfg.blocks = vec![(2, 3, 1), (2, 3, 2)];
    let bn_c = cfg.bn_size * cfg.growth_rate;
    let eps_affine = true;

    let mut m = Tensors::new();
    // FCM
    put(&mut m, "head.conv1.weight", &[32, 1, 3, 3]);
    bn1(&mut m, "head.bn1", 32, eps_affine);
    res_block_keys(&mut m, "head.layer1.0", 32, 32, true);
    res_block_keys(&mut m, "head.layer1.1", 32, 32, false);
    res_block_keys(&mut m, "head.layer2.0", 32, 32, true);
    res_block_keys(&mut m, "head.layer2.1", 32, 32, false);
    put(&mut m, "head.conv2.weight", &[32, 32, 3, 3]);
    bn1(&mut m, "head.bn2", 32, eps_affine);
    // tdnn
    let head_out = 32 * (cfg.feat_dim / 8);
    put(&mut m, "xvector.tdnn.linear.weight", &[128, head_out, 5]);
    bn1(&mut m, "xvector.tdnn.nonlinear.batchnorm", 128, true);
    let mut ch = 128usize;
    for (bi, &(nl, k, _dil)) in cfg.blocks.iter().enumerate() {
        let bp = format!("xvector.block{}", bi + 1);
        for i in 0..nl {
            let lp = format!("{bp}.tdnnd{}", i + 1);
            let in_ch = ch + i * cfg.growth_rate;
            bn1(&mut m, &format!("{lp}.nonlinear1.batchnorm"), in_ch, true);
            put(&mut m, &format!("{lp}.linear1.weight"), &[bn_c, in_ch, 1]);
            bn1(&mut m, &format!("{lp}.nonlinear2.batchnorm"), bn_c, true);
            let red = (bn_c / 2).max(1);
            put(
                &mut m,
                &format!("{lp}.cam_layer.linear_local.weight"),
                &[cfg.growth_rate, bn_c, k],
            );
            put(
                &mut m,
                &format!("{lp}.cam_layer.linear1.weight"),
                &[red, bn_c, 1],
            );
            put(&mut m, &format!("{lp}.cam_layer.linear1.bias"), &[red]);
            put(
                &mut m,
                &format!("{lp}.cam_layer.linear2.weight"),
                &[cfg.growth_rate, red, 1],
            );
            put(
                &mut m,
                &format!("{lp}.cam_layer.linear2.bias"),
                &[cfg.growth_rate],
            );
        }
        ch += nl * cfg.growth_rate;
        let tp = format!("xvector.transit{}", bi + 1);
        bn1(&mut m, &format!("{tp}.nonlinear.batchnorm"), ch, true);
        put(&mut m, &format!("{tp}.linear.weight"), &[ch / 2, ch, 1]);
        ch /= 2;
    }
    bn1(&mut m, "xvector.out_nonlinear.batchnorm", ch, true);
    put(
        &mut m,
        "xvector.dense.linear.weight",
        &[cfg.embedding_size, 2 * ch, 1],
    );
    bn1(
        &mut m,
        "xvector.dense.nonlinear.batchnorm",
        cfg.embedding_size,
        false,
    );

    let t = 24usize;
    let feats: Vec<f32> = (0..t * cfg.feat_dim)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
        .collect();
    let model = CamPlus::from_parts(cfg.clone(), WeightMap::from_tensors(m), Device::Cpu);
    let emb = model.run_embedding(&feats, t).expect("campplus run");
    assert_eq!(emb.len(), cfg.embedding_size);
    assert!(emb.iter().all(|x| x.is_finite()), "campplus non-finite");
}

fn res_block_keys(m: &mut Tensors, p: &str, in_c: usize, out_c: usize, downsample: bool) {
    put(m, &format!("{p}.conv1.weight"), &[out_c, in_c, 3, 3]);
    bn1(m, &format!("{p}.bn1"), out_c, true);
    put(m, &format!("{p}.conv2.weight"), &[out_c, out_c, 3, 3]);
    bn1(m, &format!("{p}.bn2"), out_c, true);
    if downsample {
        put(m, &format!("{p}.shortcut.0.weight"), &[out_c, in_c, 1, 1]);
        bn1(m, &format!("{p}.shortcut.1"), out_c, true);
    }
}
