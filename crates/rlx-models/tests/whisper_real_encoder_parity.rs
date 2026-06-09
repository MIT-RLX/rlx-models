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

//! RLX vs candle encoder on real whisper-tiny weights + JFK wav.
#![cfg(feature = "parity-candle")]

use anyhow::Result;
use candle_core::{DType as CDType, Device as CDevice, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as candle_whisper, Config as CandleConfig};
use rlx_core::flow_util::compile_built;
use rlx_models::weight_map::WeightMap;
use rlx_models::whisper::{
    WhisperConfig, WhisperWeightPrefix, build_whisper_encoder_built, load_wav_mono_f32, pcm_to_mel,
};
use rlx_runtime::Device;
use std::path::PathBuf;

fn cache_dir() -> PathBuf {
    std::env::var("RLX_WHISPER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/whisper-tiny")
        })
}

fn jfk_wav() -> PathBuf {
    std::env::var("RLX_WHISPER_WAV")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/whisper-bench/jfk_16k.wav")
        })
}

fn max_mean_abs(a: &[f32], b: &[f32]) -> (f32, f32) {
    let n = a.len().min(b.len());
    let mut max = 0f32;
    let mut sum = 0f64;
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        max = max.max(d);
        sum += d as f64;
    }
    (max, (sum / n.max(1) as f64) as f32)
}

#[test]
fn whisper_tiny_jfk_encoder_matches_candle() -> Result<()> {
    let dir = cache_dir();
    let weights_path = dir.join("model.safetensors");
    let wav = jfk_wav();
    if !weights_path.is_file() || !wav.is_file() {
        eprintln!("skip: missing weights or wav");
        return Ok(());
    }

    let cfg = WhisperConfig::from_file(&dir.join("config.json"))?;
    let pcm = load_wav_mono_f32(&wav)?;
    let mel = pcm_to_mel(&cfg, &pcm);
    let mel_frames = mel.n_frames;
    let batch = 1usize;

    let wt = weights_path.to_str().unwrap();
    let mut wm = WeightMap::from_tensors(WeightMap::snapshot_from_path(wt)?);
    let pfx = WhisperWeightPrefix::detect(&wm);
    let enc_built = build_whisper_encoder_built(&cfg, &mut wm, &pfx, batch, mel_frames)?;
    let enc_params = enc_built.params().clone();
    let mut rlx_enc = compile_built(enc_built, Device::Cpu)?;
    for (n, d) in &enc_params {
        rlx_enc.set_param(n, d);
    }
    let rlx_out = rlx_enc
        .run(&[("mel", &mel.data)])
        .into_iter()
        .next()
        .unwrap();

    let candle_cfg: CandleConfig =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let dev = CDevice::Cpu;
    let tensors: std::collections::HashMap<String, Tensor> = WeightMap::snapshot_from_path(wt)?
        .into_iter()
        .map(|(k, (data, shape))| Ok((k, Tensor::from_vec(data, shape, &dev)?)))
        .collect::<Result<_>>()?;
    let vb = VarBuilder::from_tensors(tensors, CDType::F32, &dev);
    let mut candle_model = candle_whisper::model::Whisper::load(&vb, candle_cfg)?;
    let mel_t = Tensor::from_vec(
        mel.data.clone(),
        (batch, cfg.num_mel_bins, mel_frames),
        &dev,
    )?;
    let candle_enc = candle_model.encoder.forward(&mel_t, true)?;
    let candle_flat = candle_enc.flatten_all()?.to_vec1()?;

    let (mx, mn) = max_mean_abs(&rlx_out, &candle_flat);
    eprintln!("encoder parity max_abs={mx:.6} mean_abs={mn:.6}");
    // Mean tracks well; a few positions spike vs candle on long JFK encodes (pos ~400).
    assert!(mn < 0.01, "encoder mean_abs={mn}");
    assert!(mx < 0.45, "encoder max_abs={mx}");
    Ok(())
}
