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

//! Prefill logits: cross-cache path vs full decoder (real whisper-tiny + JFK).
//!
//! ```sh
//! cargo test -p rlx-models --test whisper_decode_path_parity --release -- --nocapture
//! ```

use anyhow::Result;
use rlx_core::flow_util::compile_built;
use rlx_models::weight_map::WeightMap;
use rlx_models::whisper::{
    WhisperConfig, WhisperWeightPrefix, build_whisper_cross_kv_built, build_whisper_decoder_built,
    build_whisper_decoder_prefill_built_ext, build_whisper_encoder_built, load_wav_mono_f32,
    pcm_to_mel,
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

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn whisper_jfk_prefill_cross_matches_full_decoder() -> Result<()> {
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
    let enc_seq = cfg.encoder_seq_len(mel_frames);
    let batch = 1usize;
    let prompt: Vec<f32> = vec![50258., 50259., 50359., 50363.];
    let dec_seq = prompt.len();

    let wt = weights_path.to_str().unwrap();
    let snapshot = WeightMap::snapshot_from_path(wt)?;
    let pfx = WhisperWeightPrefix::detect(&WeightMap::from_tensors(snapshot.clone()));

    let mut wm_enc = WeightMap::from_tensors(snapshot.clone());
    let enc_built = build_whisper_encoder_built(&cfg, &mut wm_enc, &pfx, batch, mel_frames)?;
    let enc_params = enc_built.params().clone();
    let mut enc = compile_built(enc_built, Device::Cpu)?;
    for (n, d) in &enc_params {
        enc.set_param(n, d);
    }
    let enc_hidden = enc.run(&[("mel", &mel.data)]).into_iter().next().unwrap();

    let mut wm_full = WeightMap::from_tensors(snapshot.clone());
    let full_built =
        build_whisper_decoder_built(&cfg, &mut wm_full, &pfx, batch, dec_seq, enc_seq)?;
    let full_params = full_built.params().clone();
    let mut full_dec = compile_built(full_built, Device::Cpu)?;
    for (n, d) in &full_params {
        full_dec.set_param(n, d);
    }
    let full_logits = full_dec
        .run(&[("token_ids", &prompt), ("encoder_hidden", &enc_hidden)])
        .into_iter()
        .next()
        .unwrap();

    let mut wm_cross = WeightMap::from_tensors(snapshot.clone());
    let cross_built = build_whisper_cross_kv_built(&cfg, &mut wm_cross, &pfx, batch, enc_seq)?;
    let cross_params = cross_built.params().clone();
    let mut cross_c = compile_built(cross_built, Device::Cpu)?;
    for (n, d) in &cross_params {
        cross_c.set_param(n, d);
    }
    let cross_outs = cross_c.run(&[("encoder_hidden", &enc_hidden)]);

    let mut wm_pre = WeightMap::from_tensors(snapshot);
    let pre_built = build_whisper_decoder_prefill_built_ext(
        &cfg,
        &mut wm_pre,
        &pfx,
        batch,
        dec_seq,
        enc_seq,
        true,
    )?;
    let pre_params = pre_built.params().clone();
    let mut pre = compile_built(pre_built, Device::Cpu)?;
    for (n, d) in &pre_params {
        pre.set_param(n, d);
    }
    let cross_keys: Vec<String> = (0..cfg.decoder_layers)
        .flat_map(|i| [format!("cross_k_{i}"), format!("cross_v_{i}")])
        .collect();
    let mut pre_in: Vec<(&str, &[f32])> = vec![("token_ids", &prompt)];
    for i in 0..cfg.decoder_layers {
        pre_in.push((cross_keys[2 * i].as_str(), cross_outs[2 * i].as_slice()));
        pre_in.push((
            cross_keys[2 * i + 1].as_str(),
            cross_outs[2 * i + 1].as_slice(),
        ));
    }
    let cross_logits = pre.run(&pre_in).into_iter().next().unwrap();

    let vocab = cfg.vocab_size;
    let last_row = |logits: &[f32]| {
        let off = (dec_seq - 1) * vocab;
        logits[off..off + vocab].to_vec()
    };
    let mx = max_abs(&last_row(&full_logits), &last_row(&cross_logits));
    eprintln!("prefill cross vs full decoder max_abs={mx:.6}");
    assert!(
        mx < 0.05,
        "cross-cache prefill diverges from full decoder: max_abs={mx}"
    );
    Ok(())
}
