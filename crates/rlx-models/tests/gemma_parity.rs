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

//! Gemma numerical + functional parity: RLX vs candle (PyTorch-aligned) and optional HF weights.
//!
//! Synthetic (no download):
//!   cargo test -p rlx-models --test gemma_parity --features parity-candle gemma_synthetic --release -- --nocapture
//!   cargo test -p rlx-models --test gemma_parity --features parity-candle gemma2_synthetic --release -- --nocapture
//!
//! Real checkpoint (safetensors + config.json):
//!   RLX_GEMMA_WEIGHTS=/path/model.safetensors RLX_GEMMA_CONFIG=/path/config.json \
//!   cargo test -p rlx-models --features parity-candle gemma_real --release -- --nocapture
//!
//! PyTorch / transformers reference (optional):
//!   RLX_GEMMA_WEIGHTS=... RLX_GEMMA_CONFIG=... \
//!   cargo test -p rlx-models --features parity-pytorch gemma_pytorch --release -- --nocapture

#![cfg(any(feature = "parity-candle", feature = "parity-pytorch"))]

mod compile_support;

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(feature = "parity-candle")]
use candle_core::{DType as CDType, Device as CDevice, Tensor};
#[cfg(feature = "parity-candle")]
use candle_nn::Activation;
#[cfg(feature = "parity-candle")]
use candle_nn::VarBuilder;
#[cfg(feature = "parity-candle")]
use candle_transformers::models::gemma as candle_gemma;
#[cfg(feature = "parity-candle")]
use candle_transformers::models::gemma2 as candle_gemma2;
#[cfg(feature = "parity-candle")]
use rlx_models::flow_util::compile_built;
#[cfg(feature = "parity-candle")]
use rlx_models::gemma::config::GemmaRopeMap;
#[cfg(feature = "parity-candle")]
use rlx_models::gemma::{
    GemmaArch, GemmaConfig, GemmaPrefillOpts, build_gemma_graph_sized_last_logits,
    build_gemma_prefill_built,
};
#[cfg(feature = "parity-candle")]
use rlx_models::weight_map::WeightMap;
#[cfg(feature = "parity-candle")]
use rlx_runtime::Device;

const LOGIT_MAX_ABS: f32 = 2e-2;
const LOGIT_MEAN_ABS: f32 = 5e-3;
const COSINE_MIN: f32 = 0.9995;
const COSINE_DIST_MAX: f64 = 5e-4;

fn weights_path() -> Option<String> {
    rlx_ir::env::var("RLX_GEMMA_WEIGHTS")
}

fn config_path() -> Option<String> {
    rlx_ir::env::var("RLX_GEMMA_CONFIG")
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / na.sqrt() / nb.sqrt()) as f32
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    (1.0 - cosine_similarity(a, b) as f64).max(0.0)
}

fn max_mean_abs_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut max = 0f32;
    let mut sum = 0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        sum += d as f64;
        if d > max {
            max = d;
        }
    }
    (max, (sum / a.len() as f64) as f32)
}

fn argmax(xs: &[f32]) -> usize {
    xs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(feature = "parity-candle")]
fn tiny_cfg() -> (GemmaConfig, candle_gemma::Config) {
    let rlx = GemmaConfig {
        arch: GemmaArch::Gemma,
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings: true,
        attention_bias: false,
        head_dim: Some(8),
        attn_logit_softcapping: None,
        final_logit_softcapping: None,
        sliding_window: None,
        query_pre_attn_scalar: None,
        effective_num_layers: None,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        expert_weights_scale: 1.0,
        // Gemma-3n / MoE / AltUp additions — inert for this plain-Gemma tiny
        // config (values mirror `GemmaConfig::tiny_test`).
        layer_types: Vec::new(),
        rope_parameters: GemmaRopeMap::default(),
        global_head_dim: None,
        num_global_key_value_heads: None,
        attention_k_eq_v: false,
        use_bidirectional_attention: None,
        hidden_size_per_layer_input: 0,
        vocab_size_per_layer_input: 0,
        num_kv_shared_layers: 0,
        use_double_wide_mlp: false,
        enable_moe_block: false,
        eog_token_ids: Vec::new(),
        activation_sparsity_pattern: Vec::new(),
        altup_num_inputs: 0,
        altup_active_idx: 0,
        altup_coef_clip: None,
        altup_correct_scale: false,
        laurel_rank: 0,
        rope_local_base_freq: 10_000.0,
    };
    let candle = candle_gemma::Config {
        attention_bias: false,
        head_dim: 8,
        hidden_act: Some(candle_nn::Activation::Gelu),
        hidden_activation: None,
        hidden_size: 32,
        intermediate_size: 64,
        num_attention_heads: 4,
        num_hidden_layers: 2,
        num_key_value_heads: 2,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        vocab_size: 64,
        max_position_embeddings: 64,
    };
    (rlx, candle)
}

/// Tiny Gemma 2 config aligned with candle-transformers eager attention:
/// `1/sqrt(head_dim)` scale, full causal mask (no sliding window in parity).
#[cfg(feature = "parity-candle")]
fn tiny_gemma2_cfg() -> (GemmaConfig, candle_gemma2::Config) {
    let (mut rlx, _) = tiny_cfg();
    rlx.arch = GemmaArch::Gemma2;
    rlx.attn_logit_softcapping = Some(50.0);
    rlx.final_logit_softcapping = Some(30.0);
    rlx.sliding_window = None;
    rlx.query_pre_attn_scalar = None;

    let candle = candle_gemma2::Config {
        attention_bias: false,
        head_dim: 8,
        hidden_activation: Activation::GeluPytorchTanh,
        hidden_size: 32,
        intermediate_size: 64,
        num_attention_heads: 4,
        num_hidden_layers: 2,
        num_key_value_heads: 2,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        vocab_size: 64,
        max_position_embeddings: 64,
        attn_logit_softcapping: Some(50.0),
        final_logit_softcapping: Some(30.0),
        query_pre_attn_scalar: 256,
        sliding_window: None,
    };
    (rlx, candle)
}

#[cfg(feature = "parity-candle")]
fn ramp(n: usize, scale: f32, salt: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(salt)) >> 8;
            (x as f32 / (1u32 << 24) as f32 - 0.5) * scale
        })
        .collect()
}

#[cfg(feature = "parity-candle")]
fn synthetic_tensors(cfg: &GemmaConfig) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let h = cfg.hidden_size;
    let q_dim = cfg.q_proj_dim();
    let kv_dim = cfg.kv_proj_dim();
    let int_dim = cfg.intermediate_size;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    t.insert(
        "model.embed_tokens.weight".into(),
        (ramp(cfg.vocab_size * h, 0.02, 1), vec![cfg.vocab_size, h]),
    );
    for layer in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{layer}");
        let salt = layer as u32 * 17;
        t.insert(
            format!("{lp}.input_layernorm.weight"),
            (ramp(h, 0.001, salt), vec![h]),
        );
        t.insert(
            format!("{lp}.post_attention_layernorm.weight"),
            (ramp(h, 0.001, salt + 1), vec![h]),
        );
        if cfg.arch == GemmaArch::Gemma2 {
            t.insert(
                format!("{lp}.pre_feedforward_layernorm.weight"),
                (ramp(h, 0.001, salt + 10), vec![h]),
            );
            t.insert(
                format!("{lp}.post_feedforward_layernorm.weight"),
                (ramp(h, 0.001, salt + 11), vec![h]),
            );
        }
        t.insert(
            format!("{lp}.self_attn.q_proj.weight"),
            (ramp(q_dim * h, 0.01, salt + 2), vec![q_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (ramp(kv_dim * h, 0.01, salt + 3), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (ramp(kv_dim * h, 0.01, salt + 4), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.o_proj.weight"),
            (ramp(h * q_dim, 0.01, salt + 5), vec![h, q_dim]),
        );
        t.insert(
            format!("{lp}.mlp.gate_proj.weight"),
            (ramp(int_dim * h, 0.01, salt + 6), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.up_proj.weight"),
            (ramp(int_dim * h, 0.01, salt + 7), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.down_proj.weight"),
            (ramp(h * int_dim, 0.01, salt + 8), vec![h, int_dim]),
        );
    }
    t.insert("model.norm.weight".into(), (ramp(h, 0.001, 99), vec![h]));
    t
}

#[cfg(feature = "parity-candle")]
fn candle_tensors_to_vb(
    tensors: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    device: &CDevice,
) -> candle_core::Result<VarBuilder<'static>> {
    let mut map = HashMap::new();
    for (k, data) in tensors {
        let shape = shapes
            .get(k)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing shape for {k}")))?;
        let t = Tensor::from_vec(data.clone(), shape.as_slice(), device)?.to_dtype(CDType::F32)?;
        map.insert(k.clone(), t);
    }
    Ok(VarBuilder::from_tensors(map, CDType::F32, device))
}

#[cfg(feature = "parity-candle")]
fn flat_from_wm(wm: &WeightMap) -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>) {
    let mut data = HashMap::new();
    let mut shapes = HashMap::new();
    for k in wm.keys() {
        let (v, s) = wm.get(k).expect("key present");
        data.insert(k.to_string(), v.to_vec());
        shapes.insert(k.to_string(), s.to_vec());
    }
    (data, shapes)
}

#[cfg(feature = "parity-candle")]
fn run_candle_last_logits(
    cfg: &candle_gemma::Config,
    tensors: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    batch: usize,
    seq: usize,
    ids: &[u32],
) -> Result<Vec<f32>> {
    let device = CDevice::Cpu;
    let vb = candle_tensors_to_vb(tensors, shapes, &device).map_err(anyhow::Error::from)?;
    let mut model = candle_gemma::Model::new(false, cfg, vb).map_err(anyhow::Error::from)?;
    let input =
        Tensor::from_vec(ids.to_vec(), (batch, seq), &device).map_err(anyhow::Error::from)?;
    let logits = model.forward(&input, 0).map_err(anyhow::Error::from)?;
    logits
        .flatten_all()
        .map_err(anyhow::Error::from)?
        .to_vec1()
        .map_err(anyhow::Error::from)
}

#[cfg(feature = "parity-candle")]
fn run_candle_gemma2_last_logits(
    cfg: &candle_gemma2::Config,
    tensors: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    batch: usize,
    seq: usize,
    ids: &[u32],
) -> Result<Vec<f32>> {
    let device = CDevice::Cpu;
    let vb = candle_tensors_to_vb(tensors, shapes, &device).map_err(anyhow::Error::from)?;
    let mut model = candle_gemma2::Model::new(false, cfg, vb).map_err(anyhow::Error::from)?;
    let input =
        Tensor::from_vec(ids.to_vec(), (batch, seq), &device).map_err(anyhow::Error::from)?;
    let logits = model.forward(&input, 0).map_err(anyhow::Error::from)?;
    logits
        .flatten_all()
        .map_err(anyhow::Error::from)?
        .to_vec1()
        .map_err(anyhow::Error::from)
}

#[cfg(feature = "parity-candle")]
#[allow(dead_code)] // kept as the compile_built reference path
fn run_rlx_gemma2_last_logits_compile_built(
    cfg: &GemmaConfig,
    wm: &mut WeightMap,
    batch: usize,
    seq: usize,
    ids: &[u32],
) -> Result<Vec<f32>> {
    let opts = GemmaPrefillOpts {
        batch,
        seq,
        dynamic_seq: false,
        prefill_hidden: false,
        media_attn_bias: false,
        with_lm_head: true,
        with_kv_outputs: false,
        last_logits_only: true,
        profile: None,
    };
    let built = build_gemma_prefill_built(cfg, wm, &opts)?;
    let mut compiled = compile_built(built, Device::Cpu)?;
    let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let outs = compiled.run(&[("input_ids", ids_f32.as_slice())]);
    Ok(outs.into_iter().next().unwrap_or_default())
}

#[cfg(feature = "parity-candle")]
fn run_rlx_last_logits(
    cfg: &GemmaConfig,
    wm: &mut WeightMap,
    batch: usize,
    seq: usize,
    ids: &[u32],
) -> Result<Vec<f32>> {
    let profile = if cfg.arch == GemmaArch::Gemma2 {
        rlx_flow::CompileProfile::gemma_prefill()
    } else {
        rlx_flow::CompileProfile::llama32_prefill()
    };
    run_rlx_last_logits_profile(cfg, wm, batch, seq, ids, &profile)
}

#[cfg(feature = "parity-candle")]
fn run_rlx_last_logits_profile(
    cfg: &GemmaConfig,
    wm: &mut WeightMap,
    batch: usize,
    seq: usize,
    ids: &[u32],
    profile: &rlx_flow::CompileProfile,
) -> Result<Vec<f32>> {
    let (graph, params) = build_gemma_graph_sized_last_logits(cfg, wm, batch, seq, false)?;
    let mut compiled = compile_support::compile_with_profile(Device::Cpu, graph, params, profile);
    let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let outs = compiled.run(&[("input_ids", ids_f32.as_slice())]);
    Ok(outs.into_iter().next().unwrap_or_default())
}

#[cfg(feature = "parity-candle")]
fn assert_logits_parity(rlx: &[f32], candle: &[f32], label: &str) {
    let vocab = rlx.len().min(candle.len());
    let rlx = &rlx[..vocab];
    let candle = &candle[..vocab];
    let (max_d, mean_d) = max_mean_abs_diff(rlx, candle);
    let cos = cosine_similarity(rlx, candle);
    let cos_dist = cosine_distance(rlx, candle);
    eprintln!(
        "{label}: max_abs={max_d:.6} mean_abs={mean_d:.6} cosine={cos:.8} cos_dist={cos_dist:.8}"
    );
    assert!(cos >= COSINE_MIN, "{label}: cosine {cos:.8} < {COSINE_MIN}");
    assert!(
        cos_dist <= COSINE_DIST_MAX,
        "{label}: cosine distance {cos_dist:.8} > {COSINE_DIST_MAX}"
    );
    assert!(
        max_d <= LOGIT_MAX_ABS,
        "{label}: max_abs {max_d} > {LOGIT_MAX_ABS}"
    );
    assert!(
        mean_d <= LOGIT_MEAN_ABS,
        "{label}: mean_abs {mean_d} > {LOGIT_MEAN_ABS}"
    );
    assert_eq!(
        argmax(rlx),
        argmax(candle),
        "{label}: top-1 token {} vs {}",
        argmax(rlx),
        argmax(candle)
    );
}

#[cfg(feature = "parity-candle")]
fn synthetic_weight_map(cfg: &GemmaConfig) -> WeightMap {
    WeightMap::from_tensors(synthetic_tensors(cfg))
}

#[cfg(feature = "parity-candle")]
#[test]
fn gemma_synthetic_last_logits_match_candle() -> Result<()> {
    let (rlx_cfg, candle_cfg) = tiny_cfg();
    let batch = 1usize;
    let seq = 6usize;
    let ids: Vec<u32> = vec![2, 5, 11, 17, 23, 31];

    let wm_tensors = synthetic_tensors(&rlx_cfg);
    let (flat, shape_map) = {
        let mut data = HashMap::new();
        let mut shapes = HashMap::new();
        for (k, (v, s)) in &wm_tensors {
            data.insert(k.clone(), v.clone());
            shapes.insert(k.clone(), s.clone());
        }
        (data, shapes)
    };
    let wm = WeightMap::from_tensors(wm_tensors);

    let candle_logits = run_candle_last_logits(&candle_cfg, &flat, &shape_map, batch, seq, &ids)?;
    let mut wm = wm;
    let rlx_logits = run_rlx_last_logits(&rlx_cfg, &mut wm, batch, seq, &ids)?;

    assert_logits_parity(&rlx_logits, &candle_logits, "synthetic B=1 L=6");
    Ok(())
}

#[cfg(feature = "parity-candle")]
#[test]
fn gemma2_synthetic_last_logits_match_candle() -> Result<()> {
    let (rlx_cfg, candle_cfg) = tiny_gemma2_cfg();
    let batch = 1usize;
    let seq = 6usize;
    let ids: Vec<u32> = vec![2, 5, 11, 17, 23, 31];

    let wm_tensors = synthetic_tensors(&rlx_cfg);
    let (flat, shape_map) = {
        let mut data = HashMap::new();
        let mut shapes = HashMap::new();
        for (k, (v, s)) in &wm_tensors {
            data.insert(k.clone(), v.clone());
            shapes.insert(k.clone(), s.clone());
        }
        (data, shapes)
    };
    let mut wm = WeightMap::from_tensors(wm_tensors);

    let candle_logits =
        run_candle_gemma2_last_logits(&candle_cfg, &flat, &shape_map, batch, seq, &ids)?;
    let rlx_logits = run_rlx_last_logits(&rlx_cfg, &mut wm, batch, seq, &ids)?;

    assert_logits_parity(&rlx_logits, &candle_logits, "gemma2 synthetic B=1 L=6");
    Ok(())
}

#[cfg(feature = "parity-candle")]
#[test]
fn debug_parity_g2_attention_ops() -> Result<()> {
    use rlx_ir::Op;
    let (cfg, _) = tiny_gemma2_cfg();
    let mut wm = synthetic_weight_map(&cfg);
    let (graph, _) = build_gemma_graph_sized_last_logits(&cfg, &mut wm, 1, 6, false)?;
    for n in graph.nodes() {
        if let Op::Attention {
            head_dim,
            v_head_dim: None,
            score_scale,
            attn_logit_softcap,
            mask_kind,
            ..
        } = &n.op
        {
            eprintln!(
                "Attention head_dim={head_dim} scale={score_scale:?} softcap={attn_logit_softcap:?} mask={mask_kind:?}"
            );
        }
    }
    Ok(())
}

#[cfg(feature = "parity-candle")]
#[test]
fn debug_rlx_gemma1_vs_gemma2_logits() -> Result<()> {
    let (g1, _) = tiny_cfg();
    let (g2, _) = tiny_gemma2_cfg();
    let batch = 1usize;
    let seq = 6usize;
    let ids: Vec<u32> = vec![2, 5, 11, 17, 23, 31];
    let mut profile = rlx_flow::CompileProfile::llama32_prefill();
    profile.passes.dce = false;
    profile.fusion.skip = true;
    {
        let mut wm1 = synthetic_weight_map(&g1);
        let mut wm2 = synthetic_weight_map(&g2);
        let (g1_graph, _) = build_gemma_graph_sized_last_logits(&g1, &mut wm1, batch, seq, false)?;
        let (g2_graph, _) = build_gemma_graph_sized_last_logits(&g2, &mut wm2, batch, seq, false)?;
        eprintln!("graph nodes g1={} g2={}", g1_graph.len(), g2_graph.len());
    }
    let mut wm1 = synthetic_weight_map(&g1);
    let mut wm2 = synthetic_weight_map(&g2);
    let l1 = run_rlx_last_logits_profile(&g1, &mut wm1, batch, seq, &ids, &profile)?;
    let l2 = run_rlx_last_logits_profile(&g2, &mut wm2, batch, seq, &ids, &profile)?;
    let (max_d, _) = max_mean_abs_diff(&l1, &l2);
    let cos = cosine_similarity(&l1, &l2);
    let (g1c, c1) = tiny_cfg();
    let (_, candle2) = tiny_gemma2_cfg();
    let wm1t = synthetic_tensors(&g1c);
    let (flat1, shapes1) = {
        let mut data = HashMap::new();
        let mut shapes = HashMap::new();
        for (k, (v, s)) in &wm1t {
            data.insert(k.clone(), v.clone());
            shapes.insert(k.clone(), s.clone());
        }
        (data, shapes)
    };
    let wm2t = synthetic_tensors(&g2);
    let (flat2, shapes2) = {
        let mut data = HashMap::new();
        let mut shapes = HashMap::new();
        for (k, (v, s)) in &wm2t {
            data.insert(k.clone(), v.clone());
            shapes.insert(k.clone(), s.clone());
        }
        (data, shapes)
    };
    let cl1 = run_candle_last_logits(&c1, &flat1, &shapes1, batch, seq, &ids)?;
    let cl2 = run_candle_gemma2_last_logits(&candle2, &flat2, &shapes2, batch, seq, &ids)?;
    let (cmax, _) = max_mean_abs_diff(&cl1, &cl2);
    eprintln!(
        "rlx g1 vs g2: max_abs={max_d:.8} cosine={cos:.8}; candle g1 vs g2: max_abs={cmax:.8}"
    );
    Ok(())
}

#[cfg(feature = "parity-candle")]
#[test]
fn gemma2_synthetic_no_softcap_last_logits_match_candle() -> Result<()> {
    let (mut rlx_cfg, mut candle_cfg) = tiny_gemma2_cfg();
    rlx_cfg.attn_logit_softcapping = None;
    rlx_cfg.final_logit_softcapping = None;
    candle_cfg.attn_logit_softcapping = None;
    candle_cfg.final_logit_softcapping = None;

    let batch = 1usize;
    let seq = 6usize;
    let ids: Vec<u32> = vec![2, 5, 11, 17, 23, 31];
    let wm_tensors = synthetic_tensors(&rlx_cfg);
    let (flat, shape_map) = {
        let mut data = HashMap::new();
        let mut shapes = HashMap::new();
        for (k, (v, s)) in &wm_tensors {
            data.insert(k.clone(), v.clone());
            shapes.insert(k.clone(), s.clone());
        }
        (data, shapes)
    };
    let mut wm = WeightMap::from_tensors(wm_tensors);
    let candle_logits =
        run_candle_gemma2_last_logits(&candle_cfg, &flat, &shape_map, batch, seq, &ids)?;
    let rlx_logits = run_rlx_last_logits(&rlx_cfg, &mut wm, batch, seq, &ids)?;
    assert_logits_parity(&rlx_logits, &candle_logits, "gemma2 no-softcap B=1 L=6");
    Ok(())
}

#[cfg(feature = "parity-candle")]
fn gemma_cfg_from_json(path: &str) -> Result<(GemmaConfig, candle_gemma::Config)> {
    let data = std::fs::read_to_string(path)?;
    let v: serde_json::Value = serde_json::from_str(&data)?;
    let head_dim = v
        .get("head_dim")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or_else(|| {
            let h = v["hidden_size"].as_u64().unwrap() as usize;
            let heads = v["num_attention_heads"].as_u64().unwrap() as usize;
            h / heads
        });
    let rlx = GemmaConfig {
        arch: GemmaArch::Gemma,
        vocab_size: v["vocab_size"].as_u64().unwrap() as usize,
        hidden_size: v["hidden_size"].as_u64().unwrap() as usize,
        intermediate_size: v["intermediate_size"].as_u64().unwrap() as usize,
        num_hidden_layers: v["num_hidden_layers"].as_u64().unwrap() as usize,
        num_attention_heads: v["num_attention_heads"].as_u64().unwrap() as usize,
        num_key_value_heads: v["num_key_value_heads"].as_u64().unwrap() as usize,
        max_position_embeddings: v["max_position_embeddings"].as_u64().unwrap() as usize,
        rms_norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6),
        rope_theta: v["rope_theta"].as_f64().unwrap_or(10_000.0),
        tie_word_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(true),
        attention_bias: v["attention_bias"].as_bool().unwrap_or(false),
        head_dim: Some(head_dim),
        attn_logit_softcapping: None,
        final_logit_softcapping: None,
        sliding_window: None,
        query_pre_attn_scalar: None,
        effective_num_layers: None,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        expert_weights_scale: 1.0,
        // Gemma-3n / MoE / AltUp additions — inert for this plain-Gemma tiny
        // config (values mirror `GemmaConfig::tiny_test`).
        layer_types: Vec::new(),
        rope_parameters: GemmaRopeMap::default(),
        global_head_dim: None,
        num_global_key_value_heads: None,
        attention_k_eq_v: false,
        use_bidirectional_attention: None,
        hidden_size_per_layer_input: 0,
        vocab_size_per_layer_input: 0,
        num_kv_shared_layers: 0,
        use_double_wide_mlp: false,
        enable_moe_block: false,
        eog_token_ids: Vec::new(),
        activation_sparsity_pattern: Vec::new(),
        altup_num_inputs: 0,
        altup_active_idx: 0,
        altup_coef_clip: None,
        altup_correct_scale: false,
        laurel_rank: 0,
        rope_local_base_freq: 10_000.0,
    };
    let candle = candle_gemma::Config {
        attention_bias: rlx.attention_bias,
        head_dim,
        hidden_act: Some(candle_nn::Activation::Gelu),
        hidden_activation: None,
        hidden_size: rlx.hidden_size,
        intermediate_size: rlx.intermediate_size,
        num_attention_heads: rlx.num_attention_heads,
        num_hidden_layers: rlx.num_hidden_layers,
        num_key_value_heads: rlx.num_key_value_heads,
        rms_norm_eps: rlx.rms_norm_eps,
        rope_theta: rlx.rope_theta,
        vocab_size: rlx.vocab_size,
        max_position_embeddings: rlx.max_position_embeddings,
    };
    Ok((rlx, candle))
}

#[cfg(feature = "parity-candle")]
#[test]
fn gemma_real_weights_last_logits_match_candle() -> Result<()> {
    let (weights, config) = match (weights_path(), config_path()) {
        (Some(w), Some(c)) => (w, c),
        _ => {
            eprintln!("skip gemma_real: set RLX_GEMMA_WEIGHTS + RLX_GEMMA_CONFIG");
            return Ok(());
        }
    };
    if !Path::new(&weights).exists() {
        eprintln!("skip gemma_real: weights not found at {weights}");
        return Ok(());
    }

    let (rlx_cfg, candle_cfg) = gemma_cfg_from_json(&config)?;
    let batch = 1usize;
    let seq = 8usize;
    let ids: Vec<u32> = vec![2, 106, 164, 207, 417, 521, 897, 128009];

    let mut wm = WeightMap::from_file(&weights)?;
    let (flat, shapes) = flat_from_wm(&wm);

    let candle_logits = run_candle_last_logits(&candle_cfg, &flat, &shapes, batch, seq, &ids)?;
    let rlx_logits = run_rlx_last_logits(&rlx_cfg, &mut wm, batch, seq, &ids)?;

    assert_logits_parity(&rlx_logits, &candle_logits, "real weights B=1 L=8");
    Ok(())
}

#[cfg(feature = "parity-pytorch")]
fn reference_script() -> std::path::PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/gemma_parity_reference.py")
}

#[cfg(all(feature = "parity-pytorch", feature = "parity-candle"))]
#[test]
fn gemma_pytorch_reference_last_logits() -> Result<()> {
    use std::process::Command;

    let (weights, config) = match (weights_path(), config_path()) {
        (Some(w), Some(c)) => (w, c),
        _ => {
            eprintln!("skip gemma_pytorch: set RLX_GEMMA_WEIGHTS + RLX_GEMMA_CONFIG");
            return Ok(());
        }
    };
    if !Path::new(&weights).exists() {
        eprintln!("skip gemma_pytorch: weights not found");
        return Ok(());
    }

    let script = reference_script();
    if !script.is_file() {
        anyhow::bail!("missing reference script at {}", script.display());
    }

    let out = Command::new("python3")
        .arg(&script)
        .arg(&weights)
        .arg(&config)
        .output()
        .context("running gemma_parity_reference.py")?;
    if !out.status.success() {
        anyhow::bail!("reference failed: {}", String::from_utf8_lossy(&out.stderr));
    }

    let line = std::str::from_utf8(&out.stdout)?
        .lines()
        .last()
        .context("empty reference stdout")?;
    let v: serde_json::Value = serde_json::from_str(line)?;
    let pt_logits: Vec<f32> = v["logits"]
        .as_array()
        .context("logits array")?
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let ids: Vec<u32> = v["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let seq = ids.len();

    let rlx_cfg = {
        #[cfg(feature = "parity-candle")]
        {
            gemma_cfg_from_json(&config)?.0
        }
        #[cfg(not(feature = "parity-candle"))]
        {
            let _ = &config;
            anyhow::bail!("gemma_pytorch test requires parity-candle for RLX forward");
        }
    };

    let mut wm = WeightMap::from_file(&weights)?;
    let rlx_logits = run_rlx_last_logits(&rlx_cfg, &mut wm, 1, seq, &ids)?;

    let (max_d, mean_d) = max_mean_abs_diff(&rlx_logits, &pt_logits);
    let cos = cosine_similarity(&rlx_logits, &pt_logits);
    let cos_dist = cosine_distance(&rlx_logits, &pt_logits);
    eprintln!(
        "pytorch reference: max_abs={max_d:.6} mean_abs={mean_d:.6} cosine={cos:.8} cos_dist={cos_dist:.8}"
    );
    assert!(cos >= COSINE_MIN, "pytorch cosine {cos:.8}");
    assert!(
        cos_dist <= COSINE_DIST_MAX * 10.0,
        "pytorch cos_dist {cos_dist:.8}"
    );
    assert_eq!(argmax(&rlx_logits), argmax(&pt_logits), "top-1 vs pytorch");
    Ok(())
}
