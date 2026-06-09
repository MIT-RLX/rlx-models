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

//! Code-predictor eager vs compiled (MLX/Metal when available) on one greedy step.

use ndarray::Array1;
use rlx_qwen3_tts::Qwen3TtsConfig;
use rlx_qwen3_tts::code_predictor::{CodePredictorEngine, CpCompiledEngine, CpEagerModel};
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;
use rlx_qwen3_tts::prompt::{build_custom_voice_prompt, load_text_tokenizer};
use rlx_qwen3_tts::talker::engine::TalkerEngine;
use rlx_qwen3_tts::text_embed::TextEmbedder;
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;

fn pick_gpu_device() -> Option<Device> {
    if is_available(Device::Mlx) {
        Some(Device::Mlx)
    } else if is_available(Device::Metal) {
        Some(Device::Metal)
    } else if is_available(Device::Cuda) {
        Some(Device::Cuda)
    } else {
        None
    }
}

#[test]
fn cp_eager_matches_compiled_groups_for_g0_1995() {
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: RLX_QWEN3_TTS_PARITY=1");
        return;
    }
    let device = match std::env::var("RLX_QWEN3_TTS_CP_COMPILED").ok().as_deref() {
        Some("1") => pick_gpu_device().expect("RLX_QWEN3_TTS_CP_COMPILED=1 needs MLX/Metal/CUDA"),
        _ => {
            eprintln!("skip: set RLX_QWEN3_TTS_CP_COMPILED=1 to run compiled CP parity");
            return;
        }
    };

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
    let cp_cfg = cfg.code_predictor();

    let talker_snap = store
        .tensor_snapshot(&["talker.model.codec_embedding.weight"])
        .unwrap();
    let (tc_data, tc_shape) = talker_snap["talker.model.codec_embedding.weight"].clone();
    let talker_codec =
        ndarray::Array2::from_shape_vec((tc_shape[0], tc_shape[1]), tc_data).unwrap();

    let mut group_embeds = Vec::new();
    let mut lm_heads = Vec::new();
    for i in 0..cp_cfg.num_code_groups - 1 {
        let key = format!("talker.code_predictor.model.codec_embedding.{i}.weight");
        let (data, shape) = store.tensor_snapshot(&[&key]).unwrap()[&key].clone();
        group_embeds.push(ndarray::Array2::from_shape_vec((shape[0], shape[1]), data).unwrap());
        let hkey = format!("talker.code_predictor.lm_head.{i}.weight");
        let (data, shape) = store.tensor_snapshot(&[&hkey]).unwrap()[&hkey].clone();
        lm_heads.push(ndarray::Array2::from_shape_vec((shape[0], shape[1]), data).unwrap());
    }

    let tokenizer = load_text_tokenizer(&model_dir).unwrap();
    let text_embedder = TextEmbedder::open(&store).unwrap();
    let prompt = build_custom_voice_prompt(
        &cfg,
        &store,
        &text_embedder,
        &tokenizer,
        "Hi.",
        "vivian",
        "english",
    )
    .unwrap();

    // Fixed talker hidden from CPU compiled path (parity reference); compare CP backends on `device`.
    let mut talker = TalkerEngine::open(&store, cfg.talker(), Device::Cpu).unwrap();
    let prefill_rows = prompt.embeds.nrows().max(8);
    talker.warmup(prefill_rows).unwrap();
    let hidden = talker.prefill(prompt.embeds.view()).unwrap();
    let h_last: Array1<f32> = hidden.row(hidden.nrows() - 1).to_owned();

    let mut eager = CpEagerModel::open(&store, cp_cfg).unwrap();
    let eager_groups = eager
        .predict_groups(&talker_codec, &group_embeds, &lm_heads, h_last.view(), 1995)
        .unwrap();

    let cp_device = if device == Device::Metal
        && std::env::var("RLX_QWEN3_TTS_CP_METAL").ok().as_deref() != Some("1")
    {
        Device::Cpu
    } else {
        device
    };
    let mut compiled =
        CpCompiledEngine::open(store.model_dir(), &store, cp_cfg, cp_device).unwrap();
    compiled.warmup(22).unwrap();
    let compiled_groups = compiled
        .predict_groups(&talker_codec, &group_embeds, &lm_heads, h_last.view(), 1995)
        .unwrap();

    eprintln!(
        "cp eager vs compiled (talker {device:?}, cp {cp_device:?}): eager={eager_groups:?} compiled={compiled_groups:?}"
    );
    if cp_device == device {
        assert_eq!(
            eager_groups, compiled_groups,
            "code predictor groups diverged (talker {device:?}, cp {cp_device:?})"
        );
    }

    let mut engine = CodePredictorEngine::open(&store, cp_cfg, device).unwrap();
    engine.warmup(22).unwrap();
    let engine_groups = engine.predict_groups(h_last.view(), 1995).unwrap();
    eprintln!("cp engine (talker {device:?}) = {engine_groups:?}");
    assert_eq!(
        eager_groups, engine_groups,
        "CodePredictorEngine diverged from eager on talker {device:?}"
    );
}
