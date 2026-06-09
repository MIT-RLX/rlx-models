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

//! Layer-count bisect for Metal CP prefill vs CPU eager.
//!
//! ```bash
//! export RLX_QWEN3_TTS_DIR=… RLX_QWEN3_TTS_PARITY=1
//! cargo test -p rlx-models --test qwen3_tts_cp_metal_bisect --release --features metal -- --nocapture
//! ```

use ndarray::{Array2, ArrayView1};
use rlx_core::KvCacheState;
use rlx_core::autoregressive::run_bucketed_kv_decode;
use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_qwen3::qwen3_profile_near_weights;
use rlx_qwen3_tts::Qwen3TtsConfig;
use rlx_qwen3_tts::code_predictor::{CpCompiledEngine, CpEagerModel};
use rlx_qwen3_tts::codec_frame::talker_decode_graph_parts;
use rlx_qwen3_tts::compile_opts::{
    ensure_metal_lowering_env, metal_mpsgraph_run_guard, talker_compile_options,
    talker_decode_compile_options,
};
use rlx_qwen3_tts::load::{Qwen3TtsWeightStore, remap_code_predictor_weights};
use rlx_qwen3_tts::mrope::talker_decode_rope_into;
use rlx_qwen3_tts::prompt::{build_custom_voice_prompt, load_text_tokenizer};
use rlx_qwen3_tts::talker::eager::TalkerEagerModel;
use rlx_qwen3_tts::talker::engine::TalkerEngine;
use rlx_qwen3_tts::talker::rope::build_inv_freq;
use rlx_qwen3_tts::text_embed::TextEmbedder;
use rlx_runtime::attn_mask::bucket_decode_mask;
use rlx_runtime::compile_cache::{BucketedCompileCache, CacheRunInput, pad_rows};
use rlx_runtime::{Device, Precision, Session, is_available};
use std::path::PathBuf;
use std::sync::Arc;

const PREFILL_SEQ: usize = 2;

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(
            0f32,
            |m, d| if d.is_nan() { f32::INFINITY } else { m.max(d) },
        )
}

fn synthetic_prefill_embeds(hidden: usize) -> Array2<f32> {
    let mut v = vec![0f32; PREFILL_SEQ * hidden];
    for t in 0..PREFILL_SEQ {
        for j in 0..hidden {
            v[t * hidden + j] = ((t + 1) as f32) * 1e-3 + (j as f32) * 1e-6;
        }
    }
    Array2::from_shape_vec((PREFILL_SEQ, hidden), v).unwrap()
}

#[test]
fn cp_metal_layer_count_bisect() {
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

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
    let base_cp = cfg.code_predictor();
    let embeds = synthetic_prefill_embeds(base_cp.hidden_size);

    for n_layers in 1..=base_cp.num_hidden_layers {
        let mut cp_cfg = base_cp.clone();
        cp_cfg.num_hidden_layers = n_layers;

        let mut eager = CpEagerModel::open(&store, &cp_cfg).unwrap();
        let eager_h = eager.forward(embeds.view()).unwrap();
        let eager_last: Vec<f32> = eager_h.row(eager_h.nrows() - 1).iter().copied().collect();

        let mut compiled =
            CpCompiledEngine::open(store.model_dir(), &store, &cp_cfg, Device::Metal).unwrap();
        compiled.warmup(8).unwrap();
        let compiled_h = compiled.prefill(embeds.view()).unwrap();
        let compiled_last: Vec<f32> = compiled_h
            .row(compiled_h.nrows() - 1)
            .iter()
            .copied()
            .collect();

        let d = max_abs(&eager_last, &compiled_last);
        eprintln!("cp metal bisect layers={n_layers} max_abs={d}");
        if d < 0.05 {
            eprintln!("  -> parity OK at {n_layers} layers");
        }
    }
}

#[test]
fn talker_vs_cp_weights_one_layer_metal() {
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

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
    let mut talker_cfg = cfg.talker().clone();
    talker_cfg.num_hidden_layers = 1;
    let embeds = synthetic_prefill_embeds(talker_cfg.hidden_size);

    let mut eager = TalkerEagerModel::open(&store, &talker_cfg).unwrap();
    let eager_h = eager.prefill(embeds.view()).unwrap();
    let eager_last: Vec<f32> = eager_h.row(eager_h.nrows() - 1).iter().copied().collect();

    let mut compiled = TalkerEngine::open(&store, &talker_cfg, Device::Metal).unwrap();
    compiled.warmup(PREFILL_SEQ).unwrap();
    let compiled_h = compiled.prefill(embeds.view()).unwrap();
    let compiled_last: Vec<f32> = compiled_h
        .row(compiled_h.nrows() - 1)
        .iter()
        .copied()
        .collect();
    let d_talker = max_abs(&eager_last, &compiled_last);
    eprintln!("talker 1L metal eager vs compiled max_abs={d_talker}");

    let mut cp_wm = store.load_code_predictor_backbone().unwrap();
    let cp_weights = remap_code_predictor_weights(&mut cp_wm).unwrap();
    let mut compiled_cp_w = TalkerEngine::open_with_weights(
        store.model_dir(),
        &store,
        &talker_cfg,
        cp_weights,
        Device::Metal,
    )
    .unwrap();
    compiled_cp_w.warmup(PREFILL_SEQ).unwrap();
    let cpw_h = compiled_cp_w.prefill(embeds.view()).unwrap();
    let cpw_last: Vec<f32> = cpw_h.row(cpw_h.nrows() - 1).iter().copied().collect();
    let d_cp_on_talker = max_abs(&eager_last, &cpw_last);
    let d_cp_graph = max_abs(&compiled_last, &cpw_last);
    eprintln!("CP weights via talker 1L metal vs talker eager max_abs={d_cp_on_talker}");
    eprintln!("CP weights via talker 1L metal vs talker compiled max_abs={d_cp_graph}");
}

#[test]
fn talker_layer_count_metal_parity() {
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

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
    let base = cfg.talker().clone();
    let embeds = synthetic_prefill_embeds(base.hidden_size);

    for n_layers in [1usize, 2, 4, 8, 14, 28] {
        if n_layers > base.num_hidden_layers {
            continue;
        }
        let mut talker_cfg = base.clone();
        talker_cfg.num_hidden_layers = n_layers;

        let mut eager = TalkerEagerModel::open(&store, &talker_cfg).unwrap();
        let eager_h = eager.prefill(embeds.view()).unwrap();
        let eager_last: Vec<f32> = eager_h.row(eager_h.nrows() - 1).iter().copied().collect();

        let mut compiled = TalkerEngine::open(&store, &talker_cfg, Device::Metal).unwrap();
        compiled.warmup(PREFILL_SEQ).unwrap();
        let compiled_h = compiled.prefill(embeds.view()).unwrap();
        let compiled_last: Vec<f32> = compiled_h
            .row(compiled_h.nrows() - 1)
            .iter()
            .copied()
            .collect();
        let d = max_abs(&eager_last, &compiled_last);
        eprintln!("talker metal layers={n_layers} synthetic prefill max_abs={d}");
    }
}

#[test]
fn talker_28l_metal_real_hi_prompt() {
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
    eprintln!("hi prompt prefill rows = {}", prompt.embeds.nrows());

    let mut eager = TalkerEagerModel::open(&store, cfg.talker()).unwrap();
    let eager_h = eager.prefill(prompt.embeds.view()).unwrap();
    let eager_last: Vec<f32> = eager_h.row(eager_h.nrows() - 1).iter().copied().collect();

    let mut compiled = TalkerEngine::open(&store, cfg.talker(), Device::Metal).unwrap();
    compiled.warmup(prompt.embeds.nrows()).unwrap();
    let compiled_h = compiled.prefill(prompt.embeds.view()).unwrap();
    let compiled_last: Vec<f32> = compiled_h
        .row(compiled_h.nrows() - 1)
        .iter()
        .copied()
        .collect();
    let d = max_abs(&eager_last, &compiled_last);
    eprintln!("talker 28L metal real Hi prompt eager vs metal max_abs={d}");

    let mut cpu_compiled = TalkerEngine::open(&store, cfg.talker(), Device::Cpu).unwrap();
    cpu_compiled.warmup(prompt.embeds.nrows()).unwrap();
    let cpu_h = cpu_compiled.prefill(prompt.embeds.view()).unwrap();
    let cpu_last: Vec<f32> = cpu_h.row(cpu_h.nrows() - 1).iter().copied().collect();
    let d_cpu_eager = max_abs(&eager_last, &cpu_last);
    let d_metal_cpu = max_abs(&compiled_last, &cpu_last);
    eprintln!("talker 28L cpu compiled vs eager max_abs={d_cpu_eager}");
    eprintln!("talker 28L metal compiled vs cpu compiled max_abs={d_metal_cpu}");
}

/// Golden frame-0 codec embed (matches `qwen3_tts_talker_eager_vs_compiled`).
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

/// Layer0 input RMS only (real talker weights + golden embed).
#[test]
fn talker_layer0_input_rms_cpu_vs_metal() {
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
    ensure_metal_lowering_env(Device::Metal);

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
    let hidden = cfg.talker().hidden_size;
    let emb = golden_frame0_codec_emb(&store, hidden);
    let mut wm = store.load_talker_backbone().unwrap();
    let weights = rlx_qwen3_tts::load::remap_talker_weights(&mut wm).unwrap();
    let gamma = weights
        .get("model.layers.0.input_layernorm.weight")
        .expect("ln weight")
        .0
        .clone();

    let f = DType::F32;
    let mut g = Graph::new("input_rms");
    let x = g.input("x", Shape::new(&[1, 1, hidden], f));
    let g_w = g.param("gamma", Shape::new(&[hidden], f));
    let beta = g.param("beta", Shape::new(&[hidden], f));
    let out = g.rms_norm(x, g_w, beta, cfg.talker().rms_norm_eps as f32);
    g.set_outputs(vec![out]);

    let mut cpu = Session::new(Device::Cpu).compile(g.clone());
    let mut metal = metal_mpsgraph_run_guard(Device::Metal, || {
        let mut m = Session::new(Device::Metal).compile(g);
        m.set_param("gamma", &gamma);
        m
    });
    cpu.set_param("gamma", &gamma);
    let z = vec![0f32; hidden];
    cpu.set_param("beta", &z);
    metal.set_param("beta", &z);

    let inputs = [("x", emb.as_slice())];
    let cpu_out = cpu.run(&inputs)[0].clone();
    let metal_out = metal_mpsgraph_run_guard(Device::Metal, || metal.run(&inputs))[0].clone();
    let d = max_abs(&cpu_out, &metal_out);
    eprintln!("layer0 input_rms cpu vs metal max_abs={d:.6}");
    if d >= 1e-4 {
        eprintln!("cpu[:4]={:?}", &cpu_out[..4]);
        eprintln!("metal[:4]={:?}", &metal_out[..4]);
    }
    assert!(d < 1e-4, "input_rms diverged (max_abs={d})");
}

/// Same IR graph compiled on CPU vs Metal (1L talker decode, real weights/KV).
#[test]
fn talker_1l_decode_graph_cpu_vs_metal_session() {
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

    ensure_metal_lowering_env(Device::Metal);
    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();

    let mut talker_cfg = cfg.talker().clone();
    talker_cfg.num_hidden_layers = 1;
    let qwen3 = talker_cfg.to_qwen3_config();
    let hidden = talker_cfg.hidden_size;
    let kv_dim = qwen3.kv_proj_dim();
    let head_half = talker_cfg.head_dim / 2;
    let inv_freq = build_inv_freq(talker_cfg.head_dim, talker_cfg.rope_theta);
    let emb = golden_frame0_codec_emb(&store, hidden);
    let prefill_embeds = synthetic_prefill_embeds(hidden);

    let mut eager = TalkerEagerModel::open(&store, &talker_cfg).unwrap();
    eager.prefill(prefill_embeds.view()).unwrap();
    let kv: KvCacheState = eager.kv_cache_state();
    let rope_delta = eager.rope_delta();
    let past_seq = kv.past_len;
    let upper = 16usize;

    let mut wm = store.load_talker_backbone().unwrap();
    let weights = rlx_qwen3_tts::load::remap_talker_weights(&mut wm).unwrap();
    let decode_profile = qwen3_profile_near_weights(&model_dir, true);
    let (graph, params) =
        talker_decode_graph_parts(&qwen3, &weights, &decode_profile, upper as u64).unwrap();

    let mut cpu_opts = talker_compile_options(&decode_profile, Device::Cpu);
    let mut metal_opts = talker_compile_options(&decode_profile, Device::Metal);
    cpu_opts.precision = Precision::F32;
    cpu_opts.policy = None;
    metal_opts.precision = Precision::F32;
    metal_opts.policy = None;
    metal_opts.fusion_opts.skip_fusion = true;
    cpu_opts.fusion_opts.skip_fusion = true;
    let mut cpu = Session::new(Device::Cpu).compile_with(graph.clone(), &cpu_opts);
    for (name, data) in &params {
        cpu.set_param(name, data);
    }
    let mut metal = metal_mpsgraph_run_guard(Device::Metal, || {
        let mut m = Session::new(Device::Metal).compile_with(graph, &metal_opts);
        for (name, data) in &params {
            m.set_param(name, data);
        }
        m
    });

    let mut rope_cos = vec![0f32; head_half];
    let mut rope_sin = vec![0f32; head_half];
    talker_decode_rope_into(
        &talker_cfg,
        &inv_freq,
        past_seq,
        rope_delta,
        &mut rope_cos,
        &mut rope_sin,
    );
    let mut mask = vec![0f32; upper + 1];
    for (i, slot) in mask.iter_mut().enumerate().take(upper + 1) {
        *slot = if i < past_seq || i == upper { 1.0 } else { 0.0 };
    }
    let padded_k = pad_rows(kv.layers_k[0].as_slice(), kv_dim, upper as u64);
    let padded_v = pad_rows(kv.layers_v[0].as_slice(), kv_dim, upper as u64);
    let inputs = [
        ("inputs_embeds", emb.as_slice()),
        ("rope_cos", rope_cos.as_slice()),
        ("rope_sin", rope_sin.as_slice()),
        ("mask", mask.as_slice()),
        ("past_k_0", padded_k.as_slice()),
        ("past_v_0", padded_v.as_slice()),
    ];
    let cpu_out = cpu.run(&inputs)[0].clone();
    let metal_out = metal_mpsgraph_run_guard(Device::Metal, || metal.run(&inputs))[0].clone();
    let n = hidden.min(cpu_out.len()).min(metal_out.len());
    let d = max_abs(&cpu_out[..n], &metal_out[..n]);
    eprintln!(
        "1L decode graph cpu vs metal max_abs={d} out_lens cpu={} metal={}",
        cpu_out.len(),
        metal_out.len()
    );
    if d >= 0.05 {
        eprintln!("cpu[:8]={:?}", &cpu_out[..8.min(n)]);
        eprintln!("metal[:8]={:?}", &metal_out[..8.min(n)]);
        let off = upper * hidden;
        if cpu_out.len() >= off + n && metal_out.len() >= off + n {
            let d_upper = max_abs(&cpu_out[off..off + n], &metal_out[..n]);
            eprintln!("cpu@upper vs metal@0 max_abs={d_upper:.6}");
            eprintln!("cpu@upper[:8]={:?}", &cpu_out[off..off + 8.min(n)]);
        }
    }
    assert!(
        d < 0.05,
        "1L decode graph metal diverged from cpu (max_abs={d})"
    );
}

/// Direct Session: 2L decode graph CPU vs Metal (isolates multi-layer graph from BucketedCompileCache).
#[test]
fn talker_2l_decode_graph_cpu_vs_metal_session() {
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

    ensure_metal_lowering_env(Device::Metal);
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

    let mut talker_cfg = cfg.talker().clone();
    talker_cfg.num_hidden_layers = 2;
    let qwen3 = talker_cfg.to_qwen3_config();
    let hidden = talker_cfg.hidden_size;
    let kv_dim = qwen3.kv_proj_dim();
    let head_half = talker_cfg.head_dim / 2;
    let inv_freq = build_inv_freq(talker_cfg.head_dim, talker_cfg.rope_theta);
    let emb = golden_frame0_codec_emb(&store, hidden);

    let mut eager = TalkerEagerModel::open(&store, &talker_cfg).unwrap();
    eager.prefill(prompt.embeds.view()).unwrap();
    let kv: KvCacheState = eager.kv_cache_state();
    let rope_delta = eager.rope_delta();
    let past_seq = kv.past_len;
    let upper = 16usize;

    let mut wm = store.load_talker_backbone().unwrap();
    let weights = rlx_qwen3_tts::load::remap_talker_weights(&mut wm).unwrap();
    let decode_profile = qwen3_profile_near_weights(&model_dir, true);
    let (graph, params) =
        talker_decode_graph_parts(&qwen3, &weights, &decode_profile, upper as u64).unwrap();

    let mut cpu_opts = talker_compile_options(&decode_profile, Device::Cpu);
    let mut metal_opts = talker_compile_options(&decode_profile, Device::Metal);
    cpu_opts.precision = Precision::F32;
    cpu_opts.policy = None;
    metal_opts.precision = Precision::F32;
    metal_opts.policy = None;
    metal_opts.fusion_opts.skip_fusion = true;
    cpu_opts.fusion_opts.skip_fusion = true;
    let mut cpu = Session::new(Device::Cpu).compile_with(graph.clone(), &cpu_opts);
    for (name, data) in &params {
        cpu.set_param(name, data);
    }
    let mut metal = metal_mpsgraph_run_guard(Device::Metal, || {
        let mut m = Session::new(Device::Metal).compile_with(graph, &metal_opts);
        for (name, data) in &params {
            m.set_param(name, data);
        }
        m
    });

    let mut rope_cos = vec![0f32; head_half];
    let mut rope_sin = vec![0f32; head_half];
    talker_decode_rope_into(
        &talker_cfg,
        &inv_freq,
        past_seq,
        rope_delta,
        &mut rope_cos,
        &mut rope_sin,
    );
    let mut mask = vec![0f32; upper + 1];
    for (i, slot) in mask.iter_mut().enumerate().take(upper + 1) {
        *slot = if i < past_seq || i == upper { 1.0 } else { 0.0 };
    }
    let padded_k0 = pad_rows(kv.layers_k[0].as_slice(), kv_dim, upper as u64);
    let padded_v0 = pad_rows(kv.layers_v[0].as_slice(), kv_dim, upper as u64);
    let padded_k1 = pad_rows(kv.layers_k[1].as_slice(), kv_dim, upper as u64);
    let padded_v1 = pad_rows(kv.layers_v[1].as_slice(), kv_dim, upper as u64);
    let inputs = [
        ("inputs_embeds", emb.as_slice()),
        ("rope_cos", rope_cos.as_slice()),
        ("rope_sin", rope_sin.as_slice()),
        ("mask", mask.as_slice()),
        ("past_k_0", padded_k0.as_slice()),
        ("past_v_0", padded_v0.as_slice()),
        ("past_k_1", padded_k1.as_slice()),
        ("past_v_1", padded_v1.as_slice()),
    ];
    let cpu_out = cpu.run(&inputs)[0].clone();
    let metal_out = metal_mpsgraph_run_guard(Device::Metal, || metal.run(&inputs))[0].clone();
    let n = hidden.min(cpu_out.len()).min(metal_out.len());
    let d = max_abs(&cpu_out[..n], &metal_out[..n]);
    eprintln!(
        "2L decode graph cpu vs metal max_abs={d} out_lens cpu={} metal={}",
        cpu_out.len(),
        metal_out.len()
    );
    assert!(
        d < 0.05,
        "2L decode graph metal diverged from cpu (max_abs={d})"
    );
}

/// `run_bucketed_kv_decode` + `BucketedCompileCache` (isolates cache path from `TalkerEngine`).
#[test]
fn talker_2l_bucketed_decode_cpu_vs_metal() {
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

    ensure_metal_lowering_env(Device::Metal);
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

    let mut talker_cfg = cfg.talker().clone();
    talker_cfg.num_hidden_layers = 2;
    let qwen3 = talker_cfg.to_qwen3_config();
    let hidden = talker_cfg.hidden_size;
    let kv_dim = qwen3.kv_proj_dim();
    let head_half = talker_cfg.head_dim / 2;
    let inv_freq = build_inv_freq(talker_cfg.head_dim, talker_cfg.rope_theta);
    let emb = golden_frame0_codec_emb(&store, hidden);

    let mut eager = TalkerEagerModel::open(&store, &talker_cfg).unwrap();
    eager.prefill(prompt.embeds.view()).unwrap();
    let kv: KvCacheState = eager.kv_cache_state();
    let rope_delta = eager.rope_delta();
    let past_seq = kv.past_len;
    let max_past = talker_cfg.max_position_embeddings.min(8192).min(8192) as u64;

    let mut wm = store.load_talker_backbone().unwrap();
    let weights = Arc::new(rlx_qwen3_tts::load::remap_talker_weights(&mut wm).unwrap());
    let decode_profile = qwen3_profile_near_weights(&model_dir, true);

    let mut rope_cos = vec![0f32; head_half];
    let mut rope_sin = vec![0f32; head_half];
    talker_decode_rope_into(
        &talker_cfg,
        &inv_freq,
        past_seq,
        rope_delta,
        &mut rope_cos,
        &mut rope_sin,
    );
    let upper = BucketedCompileCache::power_of_two_ladder(Device::Metal, 1, max_past)
        .bucket_for(past_seq as u64)
        .and_then(|idx| {
            BucketedCompileCache::power_of_two_ladder(Device::Metal, 1, max_past)
                .buckets()
                .nth(idx)
                .map(|r| (r.end - 1) as usize)
        })
        .unwrap_or(past_seq);
    let mask = bucket_decode_mask(past_seq, upper);
    let fixed = [
        CacheRunInput {
            name: "inputs_embeds",
            data: emb.as_slice(),
            row_inner: None,
        },
        CacheRunInput {
            name: "rope_cos",
            data: rope_cos.as_slice(),
            row_inner: None,
        },
        CacheRunInput {
            name: "rope_sin",
            data: rope_sin.as_slice(),
            row_inner: None,
        },
        CacheRunInput {
            name: "mask",
            data: mask.as_slice(),
            row_inner: None,
        },
    ];

    let qwen3_cpu = qwen3.clone();
    let qwen3_metal = qwen3;
    let profile_cpu = decode_profile.clone();
    let profile_metal = decode_profile.clone();
    let weights_cpu = Arc::clone(&weights);
    let weights_metal = Arc::clone(&weights);
    let cpu_opts = talker_decode_compile_options(&profile_cpu, Device::Cpu);
    let metal_opts = talker_decode_compile_options(&profile_metal, Device::Metal);

    let mut cpu_cache = BucketedCompileCache::power_of_two_ladder(Device::Cpu, 1, max_past);
    let (h_cpu, _, _) = run_bucketed_kv_decode(
        &mut cpu_cache,
        past_seq,
        &kv,
        kv_dim,
        2,
        &fixed,
        move |upper| {
            talker_decode_graph_parts(&qwen3_cpu, weights_cpu.as_ref(), &profile_cpu, upper)
                .unwrap()
        },
        &cpu_opts,
    )
    .unwrap();

    let mut metal_cache = BucketedCompileCache::power_of_two_ladder(Device::Metal, 1, max_past);
    let (h_metal, _, _) = metal_mpsgraph_run_guard(Device::Metal, || {
        run_bucketed_kv_decode(
            &mut metal_cache,
            past_seq,
            &kv,
            kv_dim,
            2,
            &fixed,
            move |upper| {
                talker_decode_graph_parts(
                    &qwen3_metal,
                    weights_metal.as_ref(),
                    &profile_metal,
                    upper,
                )
                .unwrap()
            },
            &metal_opts,
        )
    })
    .unwrap();

    let n = hidden.min(h_cpu.len()).min(h_metal.len());
    let d = max_abs(&h_cpu[..n], &h_metal[..n]);
    eprintln!(
        "2L bucketed decode cpu vs metal max_abs={d} lens cpu={} metal={} past={past_seq} upper={upper}",
        h_cpu.len(),
        h_metal.len()
    );
    assert!(d < 0.05, "2L bucketed decode metal diverged (max_abs={d})");
}

/// `TalkerEngine` 2L native decode vs inline `run_bucketed_kv_decode` (same KV/embed).
#[test]
fn talker_engine_2l_vs_inline_bucketed_metal() {
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
        || std::env::var("RLX_QWEN3_TTS_METAL_DECODE_NATIVE")
            .ok()
            .as_deref()
            != Some("1")
    {
        eprintln!("skip: RLX_QWEN3_TTS_METAL_COMPILED=1 RLX_QWEN3_TTS_METAL_DECODE_NATIVE=1");
        return;
    }

    ensure_metal_lowering_env(Device::Metal);
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

    let mut talker_cfg = cfg.talker().clone();
    talker_cfg.num_hidden_layers = 2;
    let qwen3 = talker_cfg.to_qwen3_config();
    let hidden = talker_cfg.hidden_size;
    let kv_dim = qwen3.kv_proj_dim();
    let head_half = talker_cfg.head_dim / 2;
    let inv_freq = build_inv_freq(talker_cfg.head_dim, talker_cfg.rope_theta);
    let emb = golden_frame0_codec_emb(&store, hidden);

    let mut eager = TalkerEagerModel::open(&store, &talker_cfg).unwrap();
    eager.prefill(prompt.embeds.view()).unwrap();
    let kv = eager.kv_cache_state();
    let rope_delta = eager.rope_delta();
    let past_seq = kv.past_len;

    let mut metal = TalkerEngine::open(&store, &talker_cfg, Device::Metal).unwrap();
    metal.restore_kv_state(kv.clone(), rope_delta);
    let mut h_eng = vec![0f32; hidden];
    metal
        .decode_hidden_into(ArrayView1::from(&emb), &mut h_eng)
        .unwrap();

    // Inline bucketed path with same inputs as engine would use.
    let mut wm = store.load_talker_backbone().unwrap();
    let weights = Arc::new(rlx_qwen3_tts::load::remap_talker_weights(&mut wm).unwrap());
    let decode_profile = qwen3_profile_near_weights(&model_dir, true);
    let mut rope_cos = vec![0f32; head_half];
    let mut rope_sin = vec![0f32; head_half];
    talker_decode_rope_into(
        &talker_cfg,
        &inv_freq,
        past_seq,
        rope_delta,
        &mut rope_cos,
        &mut rope_sin,
    );
    let max_past = talker_cfg.max_position_embeddings.min(8192) as u64;
    let upper = BucketedCompileCache::power_of_two_ladder(Device::Metal, 1, max_past)
        .bucket_for(past_seq as u64)
        .and_then(|idx| {
            BucketedCompileCache::power_of_two_ladder(Device::Metal, 1, max_past)
                .buckets()
                .nth(idx)
                .map(|r| (r.end - 1) as usize)
        })
        .unwrap_or(past_seq);
    let mask = bucket_decode_mask(past_seq, upper);
    let fixed = [
        CacheRunInput {
            name: "inputs_embeds",
            data: emb.as_slice(),
            row_inner: None,
        },
        CacheRunInput {
            name: "rope_cos",
            data: rope_cos.as_slice(),
            row_inner: None,
        },
        CacheRunInput {
            name: "rope_sin",
            data: rope_sin.as_slice(),
            row_inner: None,
        },
        CacheRunInput {
            name: "mask",
            data: mask.as_slice(),
            row_inner: None,
        },
    ];
    let qwen3_m = qwen3.clone();
    let profile_m = decode_profile.clone();
    let weights_m = Arc::clone(&weights);
    let metal_opts = talker_decode_compile_options(&decode_profile, Device::Metal);
    let mut metal_cache = BucketedCompileCache::power_of_two_ladder(Device::Metal, 1, max_past);
    let (h_inline, _, _) = metal_mpsgraph_run_guard(Device::Metal, || {
        run_bucketed_kv_decode(
            &mut metal_cache,
            past_seq,
            &kv,
            kv_dim,
            2,
            &fixed,
            move |upper| {
                talker_decode_graph_parts(&qwen3_m, weights_m.as_ref(), &profile_m, upper).unwrap()
            },
            &metal_opts,
        )
    })
    .unwrap();

    let n = hidden.min(h_eng.len()).min(h_inline.len());
    let d = max_abs(&h_eng[..n], &h_inline[..n]);
    eprintln!(
        "TalkerEngine vs inline bucketed 2L max_abs={d} eng[:4]={:?} inline[:4]={:?}",
        &h_eng[..4.min(n)],
        &h_inline[..4.min(n)]
    );
    assert!(
        d < 0.05,
        "TalkerEngine diverged from inline bucketed (max_abs={d})"
    );
}

/// Bisect: `qk_norm` / per-head RMS path in 1L decode graph.
#[test]
fn talker_1l_decode_graph_qk_norm_off_cpu_vs_metal() {
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

    ensure_metal_lowering_env(Device::Metal);
    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
    let mut talker_cfg = cfg.talker().clone();
    talker_cfg.num_hidden_layers = 1;
    talker_cfg.qk_norm = false;
    let qwen3 = talker_cfg.to_qwen3_config();
    let hidden = talker_cfg.hidden_size;
    let kv_dim = qwen3.kv_proj_dim();
    let head_half = talker_cfg.head_dim / 2;
    let inv_freq = build_inv_freq(talker_cfg.head_dim, talker_cfg.rope_theta);

    let mut eager = TalkerEagerModel::open(&store, &talker_cfg).unwrap();
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
    eager.prefill(prompt.embeds.view()).unwrap();
    let kv: KvCacheState = eager.kv_cache_state();
    let rope_delta = eager.rope_delta();
    let past_seq = kv.past_len;
    let upper = 16usize;
    let emb = golden_frame0_codec_emb(&store, hidden);

    let mut wm = store.load_talker_backbone().unwrap();
    let weights = rlx_qwen3_tts::load::remap_talker_weights(&mut wm).unwrap();
    let decode_profile = qwen3_profile_near_weights(&model_dir, true);
    let (graph, params) =
        talker_decode_graph_parts(&qwen3, &weights, &decode_profile, upper as u64).unwrap();

    let cpu_opts = talker_compile_options(&decode_profile, Device::Cpu);
    let metal_opts = talker_compile_options(&decode_profile, Device::Metal);
    let mut cpu = Session::new(Device::Cpu).compile_with(graph.clone(), &cpu_opts);
    for (name, data) in &params {
        cpu.set_param(name, data);
    }
    let mut metal = metal_mpsgraph_run_guard(Device::Metal, || {
        let mut m = Session::new(Device::Metal).compile_with(graph, &metal_opts);
        for (name, data) in &params {
            m.set_param(name, data);
        }
        m
    });

    let mut rope_cos = vec![0f32; head_half];
    let mut rope_sin = vec![0f32; head_half];
    talker_decode_rope_into(
        &talker_cfg,
        &inv_freq,
        past_seq,
        rope_delta,
        &mut rope_cos,
        &mut rope_sin,
    );
    let mut mask = vec![0f32; upper + 1];
    for (i, slot) in mask.iter_mut().enumerate().take(upper + 1) {
        *slot = if i < past_seq || i == upper { 1.0 } else { 0.0 };
    }
    let padded_k = pad_rows(kv.layers_k[0].as_slice(), kv_dim, upper as u64);
    let padded_v = pad_rows(kv.layers_v[0].as_slice(), kv_dim, upper as u64);
    let inputs = [
        ("inputs_embeds", emb.as_slice()),
        ("rope_cos", rope_cos.as_slice()),
        ("rope_sin", rope_sin.as_slice()),
        ("mask", mask.as_slice()),
        ("past_k_0", padded_k.as_slice()),
        ("past_v_0", padded_v.as_slice()),
    ];

    let cpu_out = cpu.run(&inputs)[0].clone();
    let metal_out = metal_mpsgraph_run_guard(Device::Metal, || metal.run(&inputs))[0].clone();
    let n = hidden.min(cpu_out.len()).min(metal_out.len());
    let d = max_abs(&cpu_out[..n], &metal_out[..n]);
    eprintln!("1L decode qk_norm=false cpu vs metal max_abs={d}");
    assert!(d < 0.05, "qk_norm off still diverged (max_abs={d})");
}

/// Native Metal bucketed decode bisect (`RLX_QWEN3_TTS_METAL_DECODE_NATIVE=1`).
#[test]
fn talker_decode_layer_bisect_metal_native() {
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
    if std::env::var("RLX_QWEN3_TTS_METAL_DECODE_NATIVE")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("skip: RLX_QWEN3_TTS_METAL_DECODE_NATIVE=1");
        return;
    }

    ensure_metal_lowering_env(Device::Metal);
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
    let base = cfg.talker().clone();

    for n_layers in [1usize, 2, 4, 8, 14, 28] {
        if n_layers > base.num_hidden_layers {
            continue;
        }
        let mut talker_cfg = base.clone();
        talker_cfg.num_hidden_layers = n_layers;

        let mut eager = TalkerEagerModel::open(&store, &talker_cfg).unwrap();
        eager.prefill(prompt.embeds.view()).unwrap();
        let kv = eager.kv_cache_state();
        let rope_delta = eager.rope_delta();
        let mut h_eag = vec![0f32; hidden];
        eager
            .decode_step_into(ArrayView1::from(&emb), &mut h_eag)
            .unwrap();

        let mut metal = TalkerEngine::open(&store, &talker_cfg, Device::Metal).unwrap();
        metal.restore_kv_state(kv.clone(), rope_delta);
        let mut h_m = vec![0f32; hidden];
        metal
            .decode_hidden_into(ArrayView1::from(&emb), &mut h_m)
            .unwrap();

        let d = max_abs(&h_eag, &h_m);
        eprintln!("talker decode native metal layers={n_layers} eager vs native max_abs={d}");

        // CPU-compiled decode graphs (default hybrid), same injected KV.
        let mut cpu_eng = TalkerEngine::open(&store, &talker_cfg, Device::Cpu).unwrap();
        cpu_eng.restore_kv_state(kv.clone(), rope_delta);
        let mut h_cpu = vec![0f32; hidden];
        cpu_eng
            .decode_hidden_into(ArrayView1::from(&emb), &mut h_cpu)
            .unwrap();
        let d_cpu_eager = max_abs(&h_eag, &h_cpu);
        let d_metal_cpu = max_abs(&h_m, &h_cpu);
        eprintln!(
            "  layers={n_layers} cpu_compiled vs eager={d_cpu_eager:.6} native_metal vs cpu_compiled={d_metal_cpu:.6}"
        );
    }
}
