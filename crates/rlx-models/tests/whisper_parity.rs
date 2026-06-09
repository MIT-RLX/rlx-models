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

//! Whisper encoder / decoder parity: RLX vs candle-transformers (synthetic tiny config).
//!
//! ```sh
//! cargo test -p rlx-models --test whisper_parity --features parity-candle whisper_synthetic --release
//! ```

#![cfg(feature = "parity-candle")]

use anyhow::Result;
use candle_core::{DType as CDType, Device as CDevice, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as candle_whisper, Config as CandleConfig};
use rlx_core::flow_util::compile_built;
use rlx_models::weight_map::WeightMap;
use rlx_models::whisper::{
    WhisperConfig, WhisperWeightPrefix, build_whisper_decoder_built, build_whisper_encoder_built,
};
use rlx_runtime::Device;
use std::collections::HashMap;

const MAX_ABS: f32 = 5e-2;
const MEAN_ABS: f32 = 2e-2;

fn tiny_rlx_cfg() -> WhisperConfig {
    WhisperConfig {
        num_mel_bins: 4,
        max_source_positions: 16,
        d_model: 8,
        encoder_attention_heads: 2,
        encoder_layers: 1,
        decoder_layers: 1,
        vocab_size: 32,
        max_target_positions: 16,
        decoder_attention_heads: 2,
        suppress_tokens: vec![],
        begin_suppress_tokens: vec![],
    }
}

fn tiny_candle_cfg() -> CandleConfig {
    CandleConfig {
        num_mel_bins: 4,
        max_source_positions: 16,
        d_model: 8,
        encoder_attention_heads: 2,
        encoder_layers: 1,
        decoder_layers: 1,
        vocab_size: 32,
        max_target_positions: 16,
        decoder_attention_heads: 2,
        suppress_tokens: vec![],
    }
}

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + 1.0) * scale * 0.01).sin())
        .collect()
}

fn synth_weights(cfg: &WhisperConfig, pfx: &WhisperWeightPrefix) -> WeightMap {
    let d = cfg.d_model;
    let m = cfg.num_mel_bins;
    let v = cfg.vocab_size;
    let mlp = d * 4;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert(pfx.enc_conv1_w(), (ramp(d * m * 3, 1.0), vec![d, m, 3]));
    t.insert(pfx.enc_conv1_b(), (ramp(d, 0.1), vec![d]));
    t.insert(pfx.enc_conv2_w(), (ramp(d * d * 3, 1.0), vec![d, d, 3]));
    t.insert(pfx.enc_conv2_b(), (ramp(d, 0.1), vec![d]));
    t.insert(pfx.enc_ln_post_w(), (ramp(d, 0.1), vec![d]));
    t.insert(pfx.enc_ln_post_b(), (ramp(d, 0.1), vec![d]));
    for i in 0..cfg.encoder_layers {
        for name in ["self_attn.q_proj", "self_attn.out_proj", "self_attn.v_proj"] {
            t.insert(
                pfx.enc_layer(i, &format!("{name}.weight")),
                (ramp(d * d, 0.2), vec![d, d]),
            );
            t.insert(
                pfx.enc_layer(i, &format!("{name}.bias")),
                (ramp(d, 0.05), vec![d]),
            );
        }
        t.insert(
            pfx.enc_layer(i, "self_attn.k_proj.weight"),
            (ramp(d * d, 0.2), vec![d, d]),
        );
        t.insert(
            pfx.enc_layer(i, "fc1.weight"),
            (ramp(mlp * d, 0.2), vec![mlp, d]),
        );
        t.insert(pfx.enc_layer(i, "fc1.bias"), (ramp(mlp, 0.05), vec![mlp]));
        t.insert(
            pfx.enc_layer(i, "fc2.weight"),
            (ramp(d * mlp, 0.2), vec![d, mlp]),
        );
        t.insert(pfx.enc_layer(i, "fc2.bias"), (ramp(d, 0.05), vec![d]));
        for n in ["self_attn_layer_norm", "final_layer_norm"] {
            t.insert(
                pfx.enc_layer(i, &format!("{n}.weight")),
                (ramp(d, 0.1), vec![d]),
            );
            t.insert(
                pfx.enc_layer(i, &format!("{n}.bias")),
                (ramp(d, 0.1), vec![d]),
            );
        }
    }
    t.insert(pfx.dec_embed_tokens(), (ramp(v * d, 0.15), vec![v, d]));
    t.insert(
        pfx.dec_embed_positions(),
        (
            ramp(cfg.max_target_positions * d, 0.05),
            vec![cfg.max_target_positions, d],
        ),
    );
    t.insert(pfx.dec_ln_w(), (ramp(d, 0.1), vec![d]));
    t.insert(pfx.dec_ln_b(), (ramp(d, 0.1), vec![d]));
    for i in 0..cfg.decoder_layers {
        for name in [
            "self_attn.q_proj",
            "self_attn.out_proj",
            "self_attn.v_proj",
            "encoder_attn.q_proj",
            "encoder_attn.out_proj",
            "encoder_attn.v_proj",
        ] {
            t.insert(
                pfx.dec_layer(i, &format!("{name}.weight")),
                (ramp(d * d, 0.2), vec![d, d]),
            );
            t.insert(
                pfx.dec_layer(i, &format!("{name}.bias")),
                (ramp(d, 0.05), vec![d]),
            );
        }
        t.insert(
            pfx.dec_layer(i, "self_attn.k_proj.weight"),
            (ramp(d * d, 0.2), vec![d, d]),
        );
        t.insert(
            pfx.dec_layer(i, "encoder_attn.k_proj.weight"),
            (ramp(d * d, 0.2), vec![d, d]),
        );
        t.insert(
            pfx.dec_layer(i, "fc1.weight"),
            (ramp(mlp * d, 0.2), vec![mlp, d]),
        );
        t.insert(pfx.dec_layer(i, "fc1.bias"), (ramp(mlp, 0.05), vec![mlp]));
        t.insert(
            pfx.dec_layer(i, "fc2.weight"),
            (ramp(d * mlp, 0.2), vec![d, mlp]),
        );
        t.insert(pfx.dec_layer(i, "fc2.bias"), (ramp(d, 0.05), vec![d]));
        for n in [
            "self_attn_layer_norm",
            "encoder_attn_layer_norm",
            "final_layer_norm",
        ] {
            t.insert(
                pfx.dec_layer(i, &format!("{n}.weight")),
                (ramp(d, 0.1), vec![d]),
            );
            t.insert(
                pfx.dec_layer(i, &format!("{n}.bias")),
                (ramp(d, 0.1), vec![d]),
            );
        }
    }
    WeightMap::from_tensors(t)
}

fn max_mean_abs(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut max = 0f32;
    let mut sum = 0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        sum += d as f64;
        max = max.max(d);
    }
    (max, (sum / a.len() as f64) as f32)
}

fn weight_map_to_vb(map: &WeightMap, dev: &CDevice) -> Result<VarBuilder<'static>> {
    let tensors: HashMap<String, Tensor> = map
        .keys()
        .map(|k| {
            let (data, shape) = map.get(k).expect("key");
            Ok((k.to_string(), Tensor::from_vec(data.to_vec(), shape, dev)?))
        })
        .collect::<Result<_>>()?;
    Ok(VarBuilder::from_tensors(tensors, CDType::F32, dev))
}

fn assert_close(a: &[f32], b: &[f32], label: &str) {
    let (mx, mn) = max_mean_abs(a, b);
    assert!(
        mx <= MAX_ABS && mn <= MEAN_ABS,
        "{label}: max_abs={mx} mean_abs={mn} (limits {MAX_ABS}/{MEAN_ABS})"
    );
}

#[test]
fn whisper_synthetic_encoder_parity() -> Result<()> {
    let rlx_cfg = tiny_rlx_cfg();
    let candle_cfg = tiny_candle_cfg();
    let mel_frames = 8;
    let batch = 1;
    let pfx = WhisperWeightPrefix {
        encoder: "model.encoder".into(),
        decoder: "model.decoder".into(),
        hf_embed_names: true,
    };
    let mut wm = synth_weights(&rlx_cfg, &pfx);

    let enc_built = build_whisper_encoder_built(&rlx_cfg, &mut wm, &pfx, batch, mel_frames)?;
    let params = enc_built.params().clone();
    let mut rlx = compile_built(enc_built, Device::Cpu)?;
    for (n, d) in &params {
        rlx.set_param(n, d);
    }
    let mel = ramp(batch * rlx_cfg.num_mel_bins * mel_frames, 0.3);
    let rlx_out = rlx.run(&[("mel", &mel)]).into_iter().next().unwrap();

    let dev = CDevice::Cpu;
    let map = synth_weights(&rlx_cfg, &pfx);
    let vb = weight_map_to_vb(&map, &dev)?;
    let mut candle_model = candle_whisper::model::Whisper::load(&vb, candle_cfg)?;
    let mel_t = Tensor::from_vec(mel.clone(), (batch, rlx_cfg.num_mel_bins, mel_frames), &dev)?;
    let enc_out = candle_model.encoder.forward(&mel_t, true)?;
    let candle_flat = enc_out.flatten_all()?.to_vec1()?;

    assert_close(&rlx_out, &candle_flat, "encoder");
    Ok(())
}

#[test]
fn whisper_synthetic_decoder_parity() -> Result<()> {
    let rlx_cfg = tiny_rlx_cfg();
    let candle_cfg = tiny_candle_cfg();
    let mel_frames = 8;
    let dec_seq = 4;
    let batch = 1;
    let enc_seq = rlx_cfg.encoder_seq_len(mel_frames);
    let pfx = WhisperWeightPrefix {
        encoder: "model.encoder".into(),
        decoder: "model.decoder".into(),
        hf_embed_names: true,
    };

    let mut wm_enc = synth_weights(&rlx_cfg, &pfx);
    let enc_built = build_whisper_encoder_built(&rlx_cfg, &mut wm_enc, &pfx, batch, mel_frames)?;
    let enc_params = enc_built.params().clone();
    let mut enc_c = compile_built(enc_built, Device::Cpu)?;
    for (n, d) in &enc_params {
        enc_c.set_param(n, d);
    }
    let mel = ramp(batch * rlx_cfg.num_mel_bins * mel_frames, 0.3);
    let enc_hidden = enc_c.run(&[("mel", &mel)]).into_iter().next().unwrap();

    let tokens: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let mut wm_dec = synth_weights(&rlx_cfg, &pfx);
    let dec_built =
        build_whisper_decoder_built(&rlx_cfg, &mut wm_dec, &pfx, batch, dec_seq, enc_seq)?;
    let dec_params = dec_built.params().clone();
    let mut rlx_dec = compile_built(dec_built, Device::Cpu)?;
    for (n, d) in &dec_params {
        rlx_dec.set_param(n, d);
    }
    let rlx_logits = rlx_dec
        .run(&[("token_ids", &tokens), ("encoder_hidden", &enc_hidden)])
        .into_iter()
        .next()
        .unwrap();

    let dev = CDevice::Cpu;
    let map = synth_weights(&rlx_cfg, &pfx);
    let vb = weight_map_to_vb(&map, &dev)?;
    let mut candle_model = candle_whisper::model::Whisper::load(&vb, candle_cfg)?;
    let mel_t = Tensor::from_vec(mel, (batch, rlx_cfg.num_mel_bins, mel_frames), &dev)?;
    let enc_out = candle_model.encoder.forward(&mel_t, true)?;
    let tok = Tensor::from_vec(
        tokens.iter().map(|&x| x as u32).collect::<Vec<_>>(),
        (batch, dec_seq),
        &dev,
    )?;
    let dec_hidden = candle_model.decoder.forward(&tok, &enc_out, true)?;
    let candle_logits = candle_model.decoder.final_linear(&dec_hidden)?;
    let candle_flat = candle_logits.flatten_all()?.to_vec1()?;

    assert_close(&rlx_logits, &candle_flat, "decoder");
    Ok(())
}
