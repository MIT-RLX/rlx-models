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

//! wgpu GPU-KV: decode graph must expose `past_k_*` / `past_v_*` for `bind_gpu_handle`.
//!
//! ```sh
//! cargo test -p rlx-models --test whisper_wgpu_gpu_kv --features gpu --release
//! ```

#![cfg(feature = "gpu")]

use anyhow::Result;
use rlx_core::autoregressive::KvCacheState;
use rlx_core::flow_util::bucket_cache_ensure_built;
use rlx_core::install_gpu_kv_handles;
use rlx_core::weight_map::WeightMap;
use rlx_models::whisper::{
    FusedDecoderWeights, FusedEncoderWeights, N_FRAMES, WhisperConfig, WhisperGraphCtx,
    WhisperGraphOpts, WhisperWeightPrefix,
};
use rlx_runtime::Device;
use rlx_whisper::backend::{WhisperCompileOpts, decode_bucket_ladder};
use std::path::PathBuf;
use std::sync::Arc;

fn cache_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache")
}

fn tiny_ctx() -> Option<WhisperGraphCtx> {
    let dir = cache_root().join("whisper-tiny");
    let weights = dir.join("model.safetensors");
    if !weights.is_file() {
        return None;
    }
    let cfg = WhisperConfig::from_file(&dir.join("config.json")).ok()?;
    let wt = weights.to_str()?;
    let mut weights_cache = WeightMap::snapshot_from_path(wt).ok()?;
    let wm = WeightMap::from_tensors(weights_cache.clone());
    let pfx = WhisperWeightPrefix::detect(&wm);
    let fused = FusedDecoderWeights::from_checkpoint(&weights_cache, &cfg, &pfx).ok()?;
    let fused_enc = FusedEncoderWeights::from_checkpoint(&weights_cache, &cfg, &pfx).ok()?;
    fused.merge_into_tensors(&mut weights_cache);
    fused_enc.merge_into_tensors(&mut weights_cache);
    let enc_seq = cfg.encoder_seq_len(N_FRAMES);
    Some(WhisperGraphCtx {
        cfg,
        pfx,
        weights: Arc::new(weights_cache),
        enc_seq,
        mel_frames: N_FRAMES,
        graph_opts: WhisperGraphOpts::default(),
        fused: Some(fused),
        fused_enc: Some(fused_enc),
    })
}

#[test]
fn whisper_wgpu_decode_binds_past_kv() -> Result<()> {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("skip: wgpu unavailable");
        return Ok(());
    }
    let Some(ctx) = tiny_ctx() else {
        eprintln!("skip: no whisper-tiny weights");
        return Ok(());
    };
    let upper = 4u64;
    let key = 4u64;
    let layers = ctx.cfg.decoder_layers;
    let d_model = ctx.cfg.d_model;
    let weights = cache_root().join("whisper-tiny/model.safetensors");
    let opts = WhisperCompileOpts::new(Device::Gpu, false, &weights).decode;
    let mut cache = decode_bucket_ladder(Device::Gpu, 448);
    let compiled = bucket_cache_ensure_built(
        &mut cache,
        key,
        |u| ctx.build_decode_step(1, u as usize),
        &opts,
    );
    let Some((_upper, cg)) = compiled else {
        anyhow::bail!("bucket {key} missing");
    };
    for name in ["past_k_0", "past_v_0", "token_id", "pos_ix", "cross_k_0"] {
        eprintln!("{name}: has_gpu_handle={}", cg.has_gpu_handle(name));
    }
    let ok = cg.bind_gpu_handle("past_k_0", &vec![0.0f32; d_model]);
    assert!(ok, "bind_gpu_handle past_k_0 on wgpu decode graph");
    let kv = KvCacheState {
        past_len: 1,
        layers_k: vec![vec![0.0; d_model]; layers],
        layers_v: vec![vec![0.0; d_model]; layers],
        layers_kv_base: vec![0; layers],
    };
    install_gpu_kv_handles(cg, &kv, 1, upper, d_model, layers)?;
    Ok(())
}

#[test]
fn whisper_wgpu_greedy_decode_with_gpu_kv() -> Result<()> {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("skip: wgpu unavailable");
        return Ok(());
    }
    let dir = cache_root().join("whisper-tiny");
    let weights = dir.join("model.safetensors");
    if !weights.is_file() {
        eprintln!("skip: need weights + wav");
        return Ok(());
    }
    let (wav, reference) = match rlx_models::whisper::ensure_jfk_fixture() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("skip: {e}");
            return Ok(());
        }
    };
    let pcm = rlx_models::whisper::load_wav_mono_f32(&wav)?;
    let mut runner = rlx_models::whisper::WhisperRunner::builder()
        .weights(&weights)
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(Device::Gpu)
        .language("en")
        .build()?;
    assert!(runner.uses_gpu_kv(), "wgpu runner should use GPU-KV");
    let (_, text) = runner.bench_greedy_pipeline(&pcm, 32, 0)?;
    eprintln!("wgpu gpu-kv transcript: {text:?}");
    rlx_models::whisper::assert_transcript_matches_reference(&text, &reference);
    Ok(())
}
