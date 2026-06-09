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

//! Compare eager vs compiled talker prefill/decode (`RLX_QWEN3_TTS_DIR`).

use ndarray::ArrayView1;
use rlx_core::KvCacheState;
use rlx_qwen3_tts::Qwen3TtsConfig;
use rlx_qwen3_tts::compile_opts::ensure_metal_lowering_env;
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;
use rlx_qwen3_tts::prompt::{build_custom_voice_prompt, load_text_tokenizer};
use rlx_qwen3_tts::talker::eager::TalkerEagerModel;
use rlx_qwen3_tts::talker::engine::TalkerEngine;
use rlx_qwen3_tts::talker::math::{linear_logits, sample_greedy_talker_codec};
use rlx_qwen3_tts::text_embed::TextEmbedder;
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;

fn golden_frame0_codec_emb(store: &Qwen3TtsWeightStore, hidden: usize) -> Vec<f32> {
    let golden0 = [
        1995u32, 1642, 988, 1088, 246, 1543, 1579, 437, 1356, 86, 1042, 248, 1555, 781, 1772, 374,
    ];
    let snap = store
        .tensor_snapshot(&["talker.model.codec_embedding.weight"])
        .expect("codec emb");
    let (tc, sh) = snap.get("talker.model.codec_embedding.weight").unwrap();
    let codec = ndarray::Array2::from_shape_vec((sh[0], sh[1]), tc.clone()).unwrap();
    let mut emb = vec![0f32; hidden];
    for (gi, &tok) in golden0.iter().enumerate() {
        let row = if gi == 0 {
            codec.row(tok as usize).to_vec()
        } else {
            let key = format!(
                "talker.code_predictor.model.codec_embedding.{}.weight",
                gi - 1
            );
            let (data, shape) = store.tensor_snapshot(&[&key]).expect("e")[&key].clone();
            let table = ndarray::Array2::from_shape_vec((shape[0], shape[1]), data).unwrap();
            table.row(tok as usize).to_vec()
        };
        for (j, v) in row.iter().enumerate() {
            emb[j] += *v;
        }
    }
    emb
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn eager_prefill_last_row_near_compiled() {
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
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

    let mut compiled = TalkerEngine::open(&store, cfg.talker(), rlx_runtime::Device::Cpu).unwrap();
    let h_comp = compiled.prefill(prompt.embeds.view()).unwrap();
    let last_c: Vec<f32> = h_comp.row(h_comp.nrows() - 1).iter().copied().collect();

    let mut eager = TalkerEagerModel::open(&store, cfg.talker()).unwrap();
    let h_eag = eager.prefill(prompt.embeds.view()).unwrap();
    let last_e: Vec<f32> = h_eag.row(h_eag.nrows() - 1).iter().copied().collect();

    let mut max_d = 0f32;
    for (a, b) in last_c.iter().zip(last_e.iter()) {
        max_d = max_d.max((a - b).abs());
    }
    eprintln!("eager vs compiled prefill last max_abs={max_d}");
    assert!(
        max_d < 0.05,
        "eager and compiled talker prefill diverged (max_abs={max_d})"
    );
}

#[test]
fn metal_session_eager_prefill_kv_matches_eager() {
    if !is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: RLX_QWEN3_TTS_PARITY=1");
        return;
    }
    if std::env::var("RLX_QWEN3_TTS_METAL_COMPILED")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!("skip: unset RLX_QWEN3_TTS_METAL_COMPILED for eager prefill kv test");
        return;
    }

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
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

    let mut eager = TalkerEagerModel::open(&store, cfg.talker()).unwrap();
    eager.prefill(prompt.embeds.view()).unwrap();
    let kv_e = eager.kv_cache_state();

    let mut metal = TalkerEngine::open(&store, cfg.talker(), Device::Metal).unwrap();
    assert!(metal.is_eager());
    metal.prefill(prompt.embeds.view()).unwrap();
    let kv_m = metal.kv_state();

    assert_eq!(kv_e.past_len, kv_m.past_len, "prefill past_len mismatch");
    let mut max_d = 0f32;
    for (le, lm) in kv_e.layers_k.iter().zip(kv_m.layers_k.iter()) {
        max_d = max_d.max(max_abs(le, lm));
    }
    eprintln!(
        "metal session eager vs eager prefill K max_abs={max_d} past={}",
        kv_e.past_len
    );
    assert!(max_d < 1e-3, "prefill K diverged (max_abs={max_d})");
}

#[test]
fn metal_decode_with_eager_kv_matches_eager() {
    if !is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: RLX_QWEN3_TTS_PARITY=1");
        return;
    }
    if std::env::var("RLX_QWEN3_TTS_METAL_COMPILED")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("skip: RLX_QWEN3_TTS_METAL_COMPILED=1");
        return;
    }

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
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
    let hidden = cfg.talker().hidden_size;
    let emb = golden_frame0_codec_emb(&store, hidden);

    let mut eager = TalkerEagerModel::open(&store, cfg.talker()).unwrap();
    eager.prefill(prompt.embeds.view()).unwrap();
    let kv_e: KvCacheState = eager.kv_cache_state();
    let mut h_eag = vec![0f32; hidden];
    eager
        .decode_step_into(ArrayView1::from(&emb), &mut h_eag)
        .unwrap();

    let rope_delta = eager.rope_delta();
    ensure_metal_lowering_env(Device::Metal);
    let mut metal = TalkerEngine::open(&store, cfg.talker(), Device::Metal).unwrap();
    metal.restore_kv_state(kv_e, rope_delta);
    let mut h_m = vec![0f32; hidden];
    metal
        .decode_hidden_into(ArrayView1::from(&emb), &mut h_m)
        .unwrap();

    let d = max_abs(&h_eag, &h_m);
    eprintln!("metal decode (eager KV) vs eager hidden max_abs={d}");
    if std::env::var("RLX_QWEN3_TTS_DECODE_DEBUG").ok().as_deref() == Some("1") {
        eprintln!("eager[:8] = {:?}", &h_eag[..8.min(hidden)]);
        eprintln!("metal[:8] = {:?}", &h_m[..8.min(hidden)]);
        eprintln!("past_len={}", metal.past_len());
    }
    assert!(
        d < 0.05,
        "metal decode diverged even with eager KV (max_abs={d})"
    );
}

/// Native Metal decode + GPU-resident K/V (`RLX_QWEN3_TTS_METAL_DECODE_NATIVE=1`).
#[test]
fn metal_native_decode_with_gpu_kv_matches_eager() {
    if !is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: RLX_QWEN3_TTS_PARITY=1");
        return;
    }
    if std::env::var("RLX_QWEN3_TTS_METAL_COMPILED")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("skip: RLX_QWEN3_TTS_METAL_COMPILED=1");
        return;
    }
    if std::env::var("RLX_QWEN3_TTS_METAL_DECODE_NATIVE")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("skip: RLX_QWEN3_TTS_METAL_DECODE_NATIVE=1");
        return;
    }
    if std::env::var("RLX_QWEN3_TTS_GPU_KV").ok().as_deref() == Some("0") {
        eprintln!("skip: unset RLX_QWEN3_TTS_GPU_KV=0 for GPU KV test");
        return;
    }

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
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
    let hidden = cfg.talker().hidden_size;
    let emb = golden_frame0_codec_emb(&store, hidden);

    let mut eager = TalkerEagerModel::open(&store, cfg.talker()).unwrap();
    eager.prefill(prompt.embeds.view()).unwrap();
    let kv_e = eager.kv_cache_state();
    let mut h_eag = vec![0f32; hidden];
    eager
        .decode_step_into(ArrayView1::from(&emb), &mut h_eag)
        .unwrap();

    let rope_delta = eager.rope_delta();
    ensure_metal_lowering_env(Device::Metal);
    let mut metal = TalkerEngine::open(&store, cfg.talker(), Device::Metal).unwrap();
    assert!(
        !metal.is_eager(),
        "compiled talker required for GPU KV (set RLX_QWEN3_TTS_METAL_COMPILED=1)"
    );
    assert!(
        metal.uses_gpu_kv(),
        "GPU KV should be enabled on native Metal decode"
    );
    metal.restore_kv_state(kv_e, rope_delta);
    metal.preinstall_gpu_kv_current().unwrap();
    let mut h_m = vec![0f32; hidden];
    metal
        .decode_hidden_into(ArrayView1::from(&emb), &mut h_m)
        .unwrap();

    let d = max_abs(&h_eag, &h_m);
    eprintln!("metal native decode (GPU KV) vs eager hidden max_abs={d}");
    assert!(
        d < 0.05,
        "metal GPU KV decode diverged from eager (max_abs={d})"
    );

    // Second step within the same bucket — K/V stays on GPU (no re-bind).
    let mut h_eag2 = vec![0f32; hidden];
    eager
        .decode_step_into(ArrayView1::from(&emb), &mut h_eag2)
        .unwrap();
    let mut h_m2 = vec![0f32; hidden];
    metal
        .decode_hidden_into(ArrayView1::from(&emb), &mut h_m2)
        .unwrap();
    let d2 = max_abs(&h_eag2, &h_m2);
    eprintln!("metal native decode step2 (GPU KV) vs eager max_abs={d2}");
    assert!(d2 < 0.05, "metal GPU KV step2 diverged (max_abs={d2})");
}

#[test]
fn cpu_compiled_decode_with_eager_kv_matches_eager() {
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_METAL_COMPILED")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!("skip: unset RLX_QWEN3_TTS_METAL_COMPILED for cpu compiled + eager kv test");
        return;
    }

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
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
    let hidden = cfg.talker().hidden_size;
    let emb = golden_frame0_codec_emb(&store, hidden);

    let mut eager = TalkerEagerModel::open(&store, cfg.talker()).unwrap();
    eager.prefill(prompt.embeds.view()).unwrap();
    let kv_e = eager.kv_cache_state();
    let mut h_eag = vec![0f32; hidden];
    eager
        .decode_step_into(ArrayView1::from(&emb), &mut h_eag)
        .unwrap();

    let rope_delta = eager.rope_delta();
    let mut compiled = TalkerEngine::open(&store, cfg.talker(), Device::Cpu).unwrap();
    assert!(!compiled.is_eager());
    compiled.restore_kv_state(kv_e, rope_delta);
    let mut h_c = vec![0f32; hidden];
    compiled
        .decode_hidden_into(ArrayView1::from(&emb), &mut h_c)
        .unwrap();

    let d = max_abs(&h_eag, &h_c);
    eprintln!("cpu compiled decode (eager KV) vs eager hidden max_abs={d}");
    assert!(
        d < 0.05,
        "cpu compiled decode diverged with eager KV (max_abs={d})"
    );
}

#[test]
fn metal_compiled_decode_matches_cpu_compiled_with_shared_kv() {
    if !is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: RLX_QWEN3_TTS_PARITY=1");
        return;
    }
    if std::env::var("RLX_QWEN3_TTS_METAL_COMPILED")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("skip: RLX_QWEN3_TTS_METAL_COMPILED=1");
        return;
    }

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
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
    let hidden = cfg.talker().hidden_size;
    let emb = golden_frame0_codec_emb(&store, hidden);

    let mut cpu = TalkerEngine::open(&store, cfg.talker(), Device::Cpu).unwrap();
    assert!(!cpu.is_eager());
    cpu.prefill(prompt.embeds.view()).unwrap();
    let kv = cpu.kv_state();
    let rope_delta = cpu.rope_delta();
    let mut h_cpu = vec![0f32; hidden];
    cpu.decode_hidden_into(ArrayView1::from(&emb), &mut h_cpu)
        .unwrap();

    let mut metal = TalkerEngine::open(&store, cfg.talker(), Device::Metal).unwrap();
    assert!(!metal.is_eager());
    metal.restore_kv_state(kv, rope_delta);
    let mut h_metal = vec![0f32; hidden];
    metal
        .decode_hidden_into(ArrayView1::from(&emb), &mut h_metal)
        .unwrap();

    let d = max_abs(&h_cpu, &h_metal);
    eprintln!("metal vs cpu-compiled decode hidden max_abs={d}");
    assert!(
        d < 0.05,
        "metal decode diverged from cpu compiled (max_abs={d})"
    );
}

#[test]
fn metal_compiled_decode_after_golden_frame0_matches_eager() {
    if !is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: RLX_QWEN3_TTS_PARITY=1");
        return;
    }
    if std::env::var("RLX_QWEN3_TTS_METAL_COMPILED")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("skip: RLX_QWEN3_TTS_METAL_COMPILED=1");
        return;
    }

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
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
    let hidden = cfg.talker().hidden_size;
    let emb = golden_frame0_codec_emb(&store, hidden);

    let head_engine = TalkerEngine::open(&store, cfg.talker(), Device::Cpu).unwrap();
    let codec_head = head_engine.codec_head();

    let mut eager = TalkerEagerModel::open(&store, cfg.talker()).unwrap();
    eager.prefill(prompt.embeds.view()).unwrap();
    let mut h_eag = vec![0f32; hidden];
    eager
        .decode_step_into(ArrayView1::from(&emb), &mut h_eag)
        .unwrap();
    let logits_e = linear_logits(ArrayView1::from(&h_eag), codec_head).unwrap();
    let g_e = sample_greedy_talker_codec(
        &logits_e,
        cfg.talker().vocab_size,
        cfg.talker().codec_eos_token_id,
    );

    let mut metal = TalkerEngine::open(&store, cfg.talker(), Device::Metal).unwrap();
    metal.prefill(prompt.embeds.view()).unwrap();
    let mut h_m = vec![0f32; hidden];
    metal
        .decode_hidden_into(ArrayView1::from(&emb), &mut h_m)
        .unwrap();
    let logits_m = linear_logits(ArrayView1::from(&h_m), codec_head).unwrap();
    let g_m = sample_greedy_talker_codec(
        &logits_m,
        cfg.talker().vocab_size,
        cfg.talker().codec_eos_token_id,
    );

    let d = max_abs(&h_eag, &h_m);
    eprintln!("metal compiled vs eager decode hidden max_abs={d}");
    eprintln!("g_eager={g_e} g_metal={g_m} expect_g0=215");
    assert!(d < 0.05, "metal decode hidden diverged (max_abs={d})");
    assert_eq!(g_m, 215, "metal decode g0 after golden frame-0");
    assert_eq!(g_m, g_e, "metal and eager g0 mismatch");
}

#[test]
fn cpu_compiled_decode_after_golden_frame0_matches_eager() {
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    if std::env::var("RLX_QWEN3_TTS_METAL_COMPILED")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!("skip: unset RLX_QWEN3_TTS_METAL_COMPILED for cpu compiled decode test");
        return;
    }

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
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
    let hidden = cfg.talker().hidden_size;
    let emb = golden_frame0_codec_emb(&store, hidden);
    let head_engine = TalkerEngine::open(&store, cfg.talker(), Device::Cpu).unwrap();
    let codec_head = head_engine.codec_head();

    let mut eager = TalkerEagerModel::open(&store, cfg.talker()).unwrap();
    eager.prefill(prompt.embeds.view()).unwrap();
    let mut h_eag = vec![0f32; hidden];
    eager
        .decode_step_into(ArrayView1::from(&emb), &mut h_eag)
        .unwrap();

    let mut compiled = TalkerEngine::open(&store, cfg.talker(), Device::Cpu).unwrap();
    assert!(!compiled.is_eager());
    compiled.prefill(prompt.embeds.view()).unwrap();
    let mut h_c = vec![0f32; hidden];
    compiled
        .decode_hidden_into(ArrayView1::from(&emb), &mut h_c)
        .unwrap();

    let d = max_abs(&h_eag, &h_c);
    let g_c = sample_greedy_talker_codec(
        &linear_logits(ArrayView1::from(&h_c), codec_head).unwrap(),
        cfg.talker().vocab_size,
        cfg.talker().codec_eos_token_id,
    );
    eprintln!("cpu compiled vs eager decode hidden max_abs={d} g_compiled={g_c}");
    assert!(d < 0.05, "cpu compiled decode diverged (max_abs={d})");
    assert_eq!(g_c, 215);
}
