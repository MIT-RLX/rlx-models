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

//! wgpu checks for TTS LM graphs.
//!
//! The full 4B F32 LM exceeds wgpu's per-buffer cap (~4 GiB). Layer sharding
//! (default 4 layers/shard) keeps each compiled graph under the limit.

use std::path::PathBuf;

#[allow(dead_code)]
fn model_dir() -> Option<PathBuf> {
    std::env::var("RLX_VOXTRAL_TTS_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.join("consolidated.safetensors").is_file())
}

#[test]
#[cfg(feature = "gpu")]
fn wgpu_shard_hir_builds_on_real_weights() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_VOXTRAL_TTS_DIR");
        return;
    };
    let cfg = VoxtralTtsConfig::from_model_dir(&dir).expect("config");
    let n_layers = cfg.text_config.num_hidden_layers;
    let store = VoxtralTtsWeightStore::open(&dir).expect("store");
    let mut wm = store.load_backbone().expect("wm");
    build_tts_backbone_prefill_shard_hir_dynamic_ext(
        &cfg.text_config,
        &mut wm,
        1,
        8,
        0,
        4.min(n_layers),
        true,
    )
    .expect("prefill shard HIR");
    let mut wm2 = store.load_backbone().expect("wm");
    let opts = Llama32DecodeOpts {
        batch: 1,
        past_seq: 4,
        dynamic_past: false,
        use_custom_mask: true,
        profile: None,
    };
    build_tts_backbone_decode_shard_built_opts(
        &cfg.text_config,
        &mut wm2,
        &opts,
        0,
        4.min(n_layers),
    )
    .expect("decode shard HIR");
    eprintln!("wgpu shard HIR built (layers 0..4 of {n_layers})");
}

#[test]
#[cfg(feature = "gpu")]
fn wgpu_shard_decode_graph_compiles() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_VOXTRAL_TTS_DIR");
        return;
    };
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("skip: wgpu (Device::Gpu) not available");
        return;
    }
    let cfg = VoxtralTtsConfig::from_model_dir(&dir).expect("config");
    let store = VoxtralTtsWeightStore::open(&dir).expect("store");
    let mut wm = store.load_backbone().expect("wm");
    let opts = Llama32DecodeOpts {
        batch: 1,
        past_seq: 4,
        dynamic_past: false,
        use_custom_mask: true,
        profile: None,
    };
    let built = build_tts_backbone_decode_shard_built_opts(
        &cfg.text_config,
        &mut wm,
        &opts,
        0,
        4.min(cfg.text_config.num_hidden_layers),
    )
    .expect("decode shard built");
    let params = built.params().clone();
    let mut compiled = compile_built(built, Device::Gpu).expect("wgpu shard compile");
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    eprintln!("wgpu decode shard compiled (layers 0..4, past_len=4)");
}
