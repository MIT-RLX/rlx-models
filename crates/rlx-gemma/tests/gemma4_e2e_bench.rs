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

//! End-to-end workload benchmark for Gemma 4 **12B-shaped** graphs.
//!
//! Uses synthetic F32 weights at production dimensions (`hidden=3840`,
//! stride-6 layer pattern). Prompt workloads mirror real chat, code
//! review, document QA, and multimodal captioning — measured as
//! compile / prefill / decode with tok/s.
//!
//! ```bash
//! cargo test -p rlx-gemma --release --features apple-silicon \
//!     --test gemma4_e2e_bench bench_e2e_all_backends -- --nocapture --test-threads=1
//! ```
//!
//! Uses dynamic prefill + bucketed decode compile caches; reports compile
//! warmup separately from hot prefill / TTFT / steady decode tok/s.
//!
//! # Lite (fast Metal smoke, ~2 min):
//! RLX_GEMMA4_BENCH_LITE=1 cargo test -p rlx-gemma --release --features apple-silicon \
//!     --test gemma4_e2e_bench bench_synthetic_metal -- --nocapture --test-threads=1
//!
//! # Heavier (12/48 layers, slower compile):
//! RLX_GEMMA4_BENCH_LAYERS=12 cargo test -p rlx-gemma --release \
//!     --features apple-silicon --test gemma4_e2e_bench -- --nocapture
//!
//! # Real weights when available:
//! RLX_GEMMA4_FIXTURE=/path/to/fixture cargo test -p rlx-gemma --release \
//!     --features apple-silicon --test gemma4_e2e_bench bench_real_weights -- --nocapture
//! ```

mod gemma4_bench_common;

use anyhow::{Result, bail};
use rlx_core::weight_map::WeightMap;
use rlx_gemma::config::{
    GemmaArch, GemmaConfig, GemmaLayerType, GemmaRopeKind, GemmaRopeMap, GemmaRopeParameters,
};
use rlx_gemma::generator::GemmaGenerator;
use rlx_gemma::multimodal::{GemmaAudioConfig, GemmaVisionConfig};
use rlx_gemma::multimodal::{GemmaMultimodalConfig, MediaSlot, expand_media_placeholders};
use rlx_gemma::multimodal_embed::build_multimodal_inputs_embeds;
use rlx_gemma::multimodal_mask::build_multimodal_prefill_attn_bias;
use rlx_gemma::unified_projector::{build_unified_audio_graph, build_unified_vision_graph};
use rlx_qwen3::SampleOpts;
use rlx_runtime::{Device, Session, device_ext::is_available};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ── Workloads (representative prompts) ───────────────────────────

#[derive(Clone, Copy)]
struct LmWorkload {
    name: &'static str,
    prompt_hint: &'static str,
    prefill_len: usize,
    decode_steps: usize,
}

const LM_WORKLOADS: &[LmWorkload] = &[
    LmWorkload {
        name: "short_chat",
        prompt_hint: "User: What is quantum entanglement? Answer in two sentences.\nAssistant:",
        prefill_len: 32,
        decode_steps: 32,
    },
    LmWorkload {
        name: "code_review",
        prompt_hint: "Review this Rust async function for data races and suggest fixes:\n```rust\nasync fn fetch_all(urls: Vec<String>) -> Vec<Response> {\n    urls.into_iter().map(|u| reqwest::get(u)).collect()\n}\n```\n",
        prefill_len: 128,
        decode_steps: 32,
    },
    LmWorkload {
        name: "doc_summary",
        prompt_hint: "Summarize the main policy arguments in this article on grid-scale battery storage and interconnection reform. Focus on FERC Order 2023 impacts.\n\n[document body…]\n\nSummary:",
        prefill_len: 256,
        decode_steps: 32,
    },
    LmWorkload {
        name: "creative_write",
        prompt_hint: "Write the opening paragraph of a mystery novel set on a research station orbiting Europa. Establish tension without revealing the culprit.\n\n",
        prefill_len: 128,
        decode_steps: 32,
    },
];

#[derive(Clone, Copy)]
struct MmWorkload {
    name: &'static str,
    prompt_hint: &'static str,
    text_prefill_len: usize,
    vision_soft_tokens: usize,
    decode_steps: usize,
}

const MM_WORKLOADS: &[MmWorkload] = &[
    MmWorkload {
        name: "image_caption",
        prompt_hint: "You are a helpful vision assistant. <|image|> Describe this photograph in detail: subject, lighting, composition, and mood.",
        text_prefill_len: 48,
        vision_soft_tokens: 70,
        decode_steps: 32,
    },
    MmWorkload {
        name: "image_vqa",
        prompt_hint: "User: <|image|> How many people are visible and what are they doing?\nAssistant:",
        text_prefill_len: 48,
        vision_soft_tokens: 70,
        decode_steps: 32,
    },
];

// ── 12B-shaped config + weights ──────────────────────────────────

fn bench_layers() -> usize {
    std::env::var("RLX_GEMMA4_BENCH_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6)
}

fn bench_hidden() -> usize {
    std::env::var("RLX_GEMMA4_BENCH_HIDDEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024)
}

fn bench_full_scale() -> bool {
    std::env::var("RLX_GEMMA4_BENCH_FULL")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn bench_lite() -> bool {
    std::env::var("RLX_GEMMA4_BENCH_LITE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
        .unwrap_or(false)
}

fn active_lm_workloads() -> &'static [LmWorkload] {
    if bench_lite() {
        &LM_WORKLOADS[..1]
    } else {
        LM_WORKLOADS
    }
}

fn active_mm_workloads() -> &'static [MmWorkload] {
    if bench_lite() {
        &MM_WORKLOADS[..1]
    } else {
        MM_WORKLOADS
    }
}

fn scaled_gemma4_cfg(num_layers: usize) -> GemmaConfig {
    let hidden = if bench_full_scale() {
        3840
    } else {
        bench_hidden()
    };
    let int_dim = hidden * 4;
    let head_dim = if hidden >= 3840 { 256 } else { 128 };
    let nh = if hidden >= 3840 { 16 } else { 8 };
    let nkv = if hidden >= 3840 { 8 } else { 4 };
    let global_hd = if hidden >= 3840 {
        Some(512)
    } else {
        Some(head_dim * 2)
    };
    let global_nkv = if hidden >= 3840 { Some(1) } else { Some(2) };
    let layer_types: Vec<GemmaLayerType> = (0..num_layers)
        .map(|i| {
            if (i + 1) % 6 == 0 {
                GemmaLayerType::FullAttention
            } else {
                GemmaLayerType::SlidingAttention
            }
        })
        .collect();
    GemmaConfig {
        arch: GemmaArch::Gemma4,
        vocab_size: 8192,
        hidden_size: hidden,
        intermediate_size: int_dim,
        num_hidden_layers: num_layers,
        num_attention_heads: nh,
        num_key_value_heads: nkv,
        max_position_embeddings: 4096,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings: true,
        attention_bias: false,
        head_dim: Some(head_dim),
        attn_logit_softcapping: None,
        final_logit_softcapping: Some(30.0),
        sliding_window: Some(1024),
        query_pre_attn_scalar: None,
        effective_num_layers: None,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        expert_weights_scale: 1.0,
        layer_types,
        rope_parameters: GemmaRopeMap {
            sliding_attention: Some(GemmaRopeParameters {
                rope_theta: Some(10_000.0),
                rope_type: Some(GemmaRopeKind::Default),
                partial_rotary_factor: None,
            }),
            full_attention: Some(GemmaRopeParameters {
                rope_theta: Some(1_000_000.0),
                rope_type: Some(GemmaRopeKind::Proportional),
                partial_rotary_factor: Some(0.25),
            }),
        },
        global_head_dim: global_hd,
        num_global_key_value_heads: global_nkv,
        attention_k_eq_v: true,
        use_bidirectional_attention: Some("vision".into()),
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
    }
}

fn ramp(n: usize, scale: f32, salt: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(salt)) >> 8;
            (x as f32 / (1u32 << 24) as f32 - 0.5) * scale
        })
        .collect()
}

fn synthetic_weights(cfg: &GemmaConfig) -> WeightMap {
    let h = cfg.hidden_size;
    let int_dim = cfg.intermediate_size;
    let nh = cfg.num_attention_heads;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert(
        "model.embed_tokens.weight".into(),
        (ramp(cfg.vocab_size * h, 0.02, 1), vec![cfg.vocab_size, h]),
    );
    for layer in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{layer}");
        let salt = layer as u32 * 17;
        let dh = cfg.layer_head_dim(layer);
        let kv = cfg.layer_num_kv_heads(layer);
        let q_dim = nh * dh;
        let kv_dim = kv * dh;
        t.insert(
            format!("{lp}.input_layernorm.weight"),
            (ramp(h, 0.001, salt), vec![h]),
        );
        t.insert(
            format!("{lp}.pre_feedforward_layernorm.weight"),
            (ramp(h, 0.001, salt + 10), vec![h]),
        );
        t.insert(
            format!("{lp}.post_feedforward_layernorm.weight"),
            (ramp(h, 0.001, salt + 11), vec![h]),
        );
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
    WeightMap::from_tensors(t)
}

fn prompt_tokens(hint: &str, len: usize, vocab: usize) -> Vec<u32> {
    let mut seed = hint
        .bytes()
        .fold(0u32, |a, b| a.wrapping_mul(31).wrapping_add(b as u32));
    (0..len)
        .map(|i| {
            seed = seed.wrapping_mul(2654435761).wrapping_add(i as u32);
            ((seed as usize) % vocab.max(1)) as u32 + 1
        })
        .collect()
}

#[derive(Debug)]
struct StageTiming {
    prefill_ms: f64,
    ttft_ms: f64,
    steady_decode_ms: f64,
    decode_steps: usize,
    prefill_tokens: usize,
}

impl StageTiming {
    fn e2e_tok_s(&self) -> f64 {
        if self.ttft_ms + self.steady_decode_ms > 0.0 {
            self.decode_steps as f64 / ((self.ttft_ms + self.steady_decode_ms) / 1000.0)
        } else {
            0.0
        }
    }

    fn steady_tok_s(&self) -> f64 {
        let steady_steps = self.decode_steps.saturating_sub(1);
        if steady_steps > 0 && self.steady_decode_ms > 0.0 {
            steady_steps as f64 / (self.steady_decode_ms / 1000.0)
        } else {
            0.0
        }
    }

    fn prefill_tok_s(&self) -> f64 {
        if self.prefill_ms > 0.0 {
            self.prefill_tokens as f64 / (self.prefill_ms / 1000.0)
        } else {
            0.0
        }
    }
}

/// Longest prefix + decode horizon for bucketed decode cache sizing.
fn cache_horizon(cfg: &GemmaConfig) -> usize {
    lm_cache_horizon(cfg).max(mm_cache_horizon(cfg))
}

fn lm_cache_horizon(cfg: &GemmaConfig) -> usize {
    active_lm_workloads()
        .iter()
        .map(|w| w.prefill_len + w.decode_steps)
        .max()
        .unwrap_or(128)
        .max(64)
        .min(cfg.max_position_embeddings)
}

fn mm_cache_horizon(cfg: &GemmaConfig) -> usize {
    active_mm_workloads()
        .iter()
        .map(|w| w.text_prefill_len + w.vision_soft_tokens + 8 + w.decode_steps)
        .max()
        .unwrap_or(128)
        .max(64)
        .min(cfg.max_position_embeddings)
}

fn make_bench_generator(
    cfg: GemmaConfig,
    wm: &mut WeightMap,
    device: Device,
) -> Result<GemmaGenerator> {
    let horizon = lm_cache_horizon(&cfg);
    Ok(GemmaGenerator::from_loader(cfg, wm, device)?.with_inference_caches(horizon))
}

fn make_mm_bench_generator(
    cfg: GemmaConfig,
    wm: &mut WeightMap,
    device: Device,
) -> Result<GemmaGenerator> {
    let horizon = cache_horizon(&cfg);
    Ok(GemmaGenerator::from_loader(cfg, wm, device)?.with_inference_caches(horizon))
}

/// Compile all prefill lengths + decode buckets once (not timed).
fn warm_lm_graphs(g: &mut GemmaGenerator, cfg: &GemmaConfig) -> Result<f64> {
    let t0 = Instant::now();
    for w in active_lm_workloads() {
        let ids = prompt_tokens(w.prompt_hint, w.prefill_len, cfg.vocab_size);
        g.prefill_get_last_logits(&ids)?;
        g.prefill(&ids);
        let _ = g.generate_cached(w.decode_steps, SampleOpts::greedy())?;
    }
    g.sync_device();
    Ok(t0.elapsed().as_secs_f64() * 1000.0)
}

fn bench_lm_with_generator(
    g: &mut GemmaGenerator,
    cfg: &GemmaConfig,
    w: &LmWorkload,
) -> Result<StageTiming> {
    let ids = prompt_tokens(w.prompt_hint, w.prefill_len, cfg.vocab_size);

    // Untimed scratch (hits warm cache).
    g.prefill(&ids);
    let _ = g.generate_cached(w.decode_steps.min(4), SampleOpts::greedy())?;

    g.prefill(&ids);
    let t_prefill = Instant::now();
    g.prefill_get_last_logits(&ids)?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

    g.prefill(&ids);
    let t_ttft = Instant::now();
    let _ = g.generate_cached(1, SampleOpts::greedy())?;
    let ttft_ms = t_ttft.elapsed().as_secs_f64() * 1000.0;

    let steady_steps = w.decode_steps.saturating_sub(1);
    for _ in 0..steady_steps {
        g.step_cached(SampleOpts::greedy())?;
    }
    g.prefill(&ids);
    let _ = g.generate_cached(1, SampleOpts::greedy())?;
    let t_steady = Instant::now();
    for _ in 0..steady_steps {
        g.step_cached(SampleOpts::greedy())?;
    }
    let steady_decode_ms = t_steady.elapsed().as_secs_f64() * 1000.0;

    Ok(StageTiming {
        prefill_ms,
        ttft_ms,
        steady_decode_ms,
        decode_steps: w.decode_steps,
        prefill_tokens: ids.len(),
    })
}

#[allow(dead_code)]
fn bench_lm(device: Device, cfg: &GemmaConfig, w: &LmWorkload) -> Result<StageTiming> {
    let mut wm = synthetic_weights(cfg);
    let mut g = make_bench_generator(cfg.clone(), &mut wm, device)?;
    bench_lm_with_generator(&mut g, cfg, w)
}

fn unified_vision_cfg() -> GemmaVisionConfig {
    GemmaVisionConfig {
        patch_size: 16,
        model_patch_size: 48,
        mm_embed_dim: 3840,
        mm_posemb_size: 70,
        num_soft_tokens: 280,
        output_proj_dims: 3840,
        pooling_kernel_size: 3,
        rms_norm_eps: 1e-6,
    }
}

fn unified_audio_cfg() -> GemmaAudioConfig {
    GemmaAudioConfig {
        hidden_size: 3840,
        audio_embed_dim: 3840,
        audio_samples_per_token: 640,
        output_proj_dims: 3840,
        rms_norm_eps: 1e-6,
    }
}

fn bench_unified_vision(device: Device, num_slots: usize) -> Result<(f64, f64)> {
    let cfg = unified_vision_cfg();
    let patch_dim = cfg.model_patch_size * cfg.model_patch_size * 3;
    let d = cfg.mm_embed_dim;
    let g = build_unified_vision_graph(num_slots, &cfg)?;
    let t0 = Instant::now();
    let mut compiled = Session::new(device)
        .compile_hir(g.hir)
        .map_err(|e| anyhow::anyhow!("compile unified vision on {device:?}: {e:?}"))?;
    let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;
    compiled.set_param(
        "model.vision_embedder.patch_ln1.weight",
        &ramp(patch_dim, 0.01, 1),
    );
    compiled.set_param(
        "model.vision_embedder.patch_ln1.bias",
        &ramp(patch_dim, 0.01, 2),
    );
    compiled.set_param(
        "model.vision_embedder.patch_dense.weight",
        &ramp(patch_dim * d, 0.01, 3),
    );
    compiled.set_param("model.vision_embedder.patch_dense.bias", &ramp(d, 0.01, 4));
    compiled.set_param("model.vision_embedder.patch_ln2.weight", &ramp(d, 0.01, 5));
    compiled.set_param("model.vision_embedder.patch_ln2.bias", &ramp(d, 0.01, 6));
    compiled.set_param("model.vision_embedder.pos_norm.weight", &ramp(d, 0.01, 7));
    compiled.set_param("model.vision_embedder.pos_norm.bias", &ramp(d, 0.01, 8));
    compiled.set_param(
        "model.embed_vision.embedding_projection.weight",
        &ramp(d * d, 0.01, 9),
    );
    compiled.set_param("unified.ones", &vec![1.0f32; d]);
    compiled.set_param("unified.zero_beta", &vec![0.0f32; d]);
    let patches = ramp(num_slots * patch_dim, 0.001, 10);
    let pos = ramp(num_slots * d, 0.001, 11);
    for _ in 0..2 {
        compiled.run(&[("patches", &patches), ("pos_bias", &pos)]);
    }
    let t1 = Instant::now();
    compiled.run(&[("patches", &patches), ("pos_bias", &pos)]);
    let run_ms = t1.elapsed().as_secs_f64() * 1000.0;
    Ok((compile_ms, run_ms))
}

#[allow(dead_code)]
fn bench_unified_audio(device: Device, num_frames: usize) -> Result<(f64, f64)> {
    let cfg = unified_audio_cfg();
    let lm_hidden = 3840;
    let samples = cfg.audio_samples_per_token;
    let d = cfg.audio_embed_dim;
    let g = build_unified_audio_graph(num_frames, &cfg, lm_hidden)?;
    let t0 = Instant::now();
    let mut compiled = Session::new(device)
        .compile_hir(g.hir)
        .map_err(|e| anyhow::anyhow!("compile unified audio on {device:?}: {e:?}"))?;
    let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;
    compiled.set_param(
        "model.embed_audio.embedding_projection.weight",
        &ramp(d * lm_hidden, 0.01, 20),
    );
    compiled.set_param("unified.audio.ones", &vec![1.0f32; d]);
    compiled.set_param("unified.audio.zero_beta", &vec![0.0f32; d]);
    let frames = ramp(num_frames * samples, 0.001, 21);
    for _ in 0..2 {
        compiled.run(&[("frames", &frames)]);
    }
    let t1 = Instant::now();
    compiled.run(&[("frames", &frames)]);
    let run_ms = t1.elapsed().as_secs_f64() * 1000.0;
    Ok((compile_ms, run_ms))
}

fn mm_cfg_bench() -> GemmaMultimodalConfig {
    GemmaMultimodalConfig::parse_json(
        r#"{
        "image_token_id": 100,
        "audio_token_id": 101,
        "video_token_id": 102,
        "boi_token_id": 103,
        "eoi_token_id": 104,
        "boa_token_id": 105,
        "eoa_token_id": 106,
        "vision_config": {
            "patch_size": 16,
            "model_patch_size": 48,
            "mm_embed_dim": 3840,
            "mm_posemb_size": 70,
            "num_soft_tokens": 280,
            "output_proj_dims": 3840,
            "pooling_kernel_size": 3
        },
        "audio_config": {
            "hidden_size": 3840,
            "audio_embed_dim": 3840,
            "audio_samples_per_token": 640,
            "output_proj_dims": 3840
        }
    }"#,
    )
    .expect("mm config")
}

#[allow(dead_code)]
fn mm_cfg_12b() -> GemmaMultimodalConfig {
    GemmaMultimodalConfig::parse_json(
        r#"{
        "image_token_id": 258880,
        "audio_token_id": 258881,
        "video_token_id": 258884,
        "boi_token_id": 255999,
        "eoi_token_id": 256000,
        "boa_token_id": 256001,
        "eoa_token_id": 256002,
        "vision_config": {
            "patch_size": 16,
            "model_patch_size": 48,
            "mm_embed_dim": 3840,
            "mm_posemb_size": 70,
            "num_soft_tokens": 280,
            "output_proj_dims": 3840,
            "pooling_kernel_size": 3
        },
        "audio_config": {
            "hidden_size": 3840,
            "audio_embed_dim": 3840,
            "audio_samples_per_token": 640,
            "output_proj_dims": 3840
        }
    }"#,
    )
    .expect("mm config")
}

fn mm_token_ids(
    w: &MmWorkload,
    cfg: &GemmaConfig,
    mm_cfg: &GemmaMultimodalConfig,
) -> Result<Vec<u32>> {
    let parts: Vec<&str> = w.prompt_hint.split("<|image|>").collect();
    let has_image = parts.len() > 1;
    let mut chunks: Vec<Vec<u32>> = Vec::new();
    if !has_image {
        chunks.push(prompt_tokens(
            w.prompt_hint,
            w.text_prefill_len,
            cfg.vocab_size,
        ));
    } else {
        let per = (w.text_prefill_len / parts.len()).max(8);
        for p in &parts {
            chunks.push(prompt_tokens(p, per, cfg.vocab_size));
        }
    }
    let slots = if has_image {
        vec![MediaSlot::Image {
            count: w.vision_soft_tokens,
        }]
    } else {
        vec![]
    };
    if slots.is_empty() {
        Ok(chunks.into_iter().next().unwrap_or_default())
    } else {
        expand_media_placeholders(&chunks, &slots, mm_cfg)
    }
}

fn bench_multimodal_with_generator(
    g: &mut GemmaGenerator,
    cfg: &GemmaConfig,
    w: &MmWorkload,
) -> Result<(StageTiming, f64)> {
    let mm_cfg = mm_cfg_bench();
    let token_ids = mm_token_ids(w, cfg, &mm_cfg)?;
    let hidden = cfg.hidden_size;
    let image_embeds = ramp(w.vision_soft_tokens * hidden, 0.01, 42);

    let t_fuse = Instant::now();
    let embeds = build_multimodal_inputs_embeds(
        g.weights_cache(),
        cfg,
        &mm_cfg,
        &token_ids,
        &image_embeds,
        &[],
        &[],
    )?;
    let fuse_ms = t_fuse.elapsed().as_secs_f64() * 1000.0;
    let attn_bias = build_multimodal_prefill_attn_bias(&token_ids, cfg, &mm_cfg, 1);

    // Scratch run on warm cache.
    g.generate_from_embeds_with_bias(
        &token_ids,
        &embeds,
        attn_bias.clone(),
        1,
        SampleOpts::greedy(),
    )?;

    let t_ttft = Instant::now();
    g.generate_from_embeds_with_bias(
        &token_ids,
        &embeds,
        attn_bias.clone(),
        1,
        SampleOpts::greedy(),
    )?;
    let ttft_ms = t_ttft.elapsed().as_secs_f64() * 1000.0;

    let steady_steps = w.decode_steps.saturating_sub(1);
    let t_steady = Instant::now();
    for _ in 0..steady_steps {
        g.step_cached(SampleOpts::greedy())?;
    }
    let steady_decode_ms = t_steady.elapsed().as_secs_f64() * 1000.0;

    Ok((
        StageTiming {
            prefill_ms: 0.0,
            ttft_ms,
            steady_decode_ms,
            decode_steps: w.decode_steps,
            prefill_tokens: token_ids.len(),
        },
        fuse_ms,
    ))
}

fn print_lm_table(
    device: Device,
    layers: usize,
    hidden: usize,
    rows: &[(&str, &str, StageTiming)],
) {
    eprintln!("\n[gemma4 bench] LM workloads — {device:?}, {layers}/48 layers, hidden={hidden}");
    eprintln!(
        "{:<16} {:>7} {:>9} {:>9} {:>9} {:>9}",
        "workload", "prefill", "prefill", "TTFT", "steady", "e2e"
    );
    eprintln!(
        "{:<16} {:>7} {:>9} {:>9} {:>9} {:>9}",
        "", "tok", "tok/s", "ms", "tok/s", "tok/s"
    );
    for (name, hint, t) in rows {
        eprintln!(
            "{:<16} {:>7} {:>9.1} {:>9.1} {:>9.1} {:>9.1}",
            name,
            t.prefill_tokens,
            t.prefill_tok_s(),
            t.ttft_ms,
            t.steady_tok_s(),
            t.e2e_tok_s(),
        );
        eprintln!("  prompt: {}", hint.lines().next().unwrap_or(hint));
    }
}

fn run_synthetic_suite(device: Device) -> Result<()> {
    if !is_available(device) {
        eprintln!("[gemma4 bench] {device:?} unavailable — skip");
        return Ok(());
    }
    let layers = bench_layers();
    let cfg = scaled_gemma4_cfg(layers);
    eprintln!(
        "[gemma4 bench] synthetic 12B-shaped graph: layers={layers}/48 hidden={} vocab={} full_scale={} lite={}",
        cfg.hidden_size,
        cfg.vocab_size,
        bench_full_scale(),
        bench_lite(),
    );

    let mut lm_rows = Vec::new();
    let mut wm = synthetic_weights(&cfg);
    let mut g = make_bench_generator(cfg.clone(), &mut wm, device)?;
    let warm_ms = warm_lm_graphs(&mut g, &cfg)?;
    eprintln!(
        "[gemma4 bench] compile warmup {device:?}: {warm_ms:.1} ms (dynamic prefill + decode buckets through seq={})",
        cache_horizon(&cfg)
    );

    for w in active_lm_workloads() {
        let t = bench_lm_with_generator(&mut g, &cfg, w)?;
        lm_rows.push((w.name, w.prompt_hint, t));
    }
    g.sync_device();
    print_lm_table(device, layers, cfg.hidden_size, &lm_rows);
    // Release LM compile caches before standalone vision / multimodal graphs.
    drop(g);

    let (v_compile, v_run) = bench_unified_vision(device, 70)?;
    eprintln!(
        "[gemma4 bench] unified vision projector (70 slots, d=3840) {device:?}: \
         compile={v_compile:.1} ms run={v_run:.1} ms"
    );

    eprintln!(
        "[gemma4 bench] unified audio projector: skipped (raw PCM embed path not in standalone graph yet)"
    );

    eprintln!("\n[gemma4 bench] multimodal LM (fused embeds + vision bias)");
    let mut wm_mm = synthetic_weights(&cfg);
    let mut g = make_mm_bench_generator(cfg.clone(), &mut wm_mm, device)?;
    if let Some(w) = MM_WORKLOADS.first() {
        let mm_cfg = mm_cfg_bench();
        let token_ids = mm_token_ids(w, &cfg, &mm_cfg)?;
        let image_embeds = ramp(w.vision_soft_tokens * cfg.hidden_size, 0.01, 42);
        let embeds = build_multimodal_inputs_embeds(
            g.weights_cache(),
            &cfg,
            &mm_cfg,
            &token_ids,
            &image_embeds,
            &[],
            &[],
        )?;
        let attn_bias = build_multimodal_prefill_attn_bias(&token_ids, &cfg, &mm_cfg, 1);
        g.generate_from_embeds_with_bias(&token_ids, &embeds, attn_bias, 4, SampleOpts::greedy())?;
    }
    for w in active_mm_workloads() {
        let (t, fuse_ms) = bench_multimodal_with_generator(&mut g, &cfg, w)?;
        eprintln!(
            "  {:<16} seq={:<4} fuse={:.1} ms TTFT={:.1} ms steady={:.1} tok/s e2e={:.1} tok/s",
            w.name,
            t.prefill_tokens,
            fuse_ms,
            t.ttft_ms,
            t.steady_tok_s(),
            t.e2e_tok_s(),
        );
        eprintln!("    {}", w.prompt_hint);
    }

    Ok(())
}

#[cfg(not(any(
    all(target_os = "macos", feature = "metal"),
    all(target_os = "macos", feature = "mlx"),
    feature = "gpu"
)))]
fn optional_gemma4_backends() -> Vec<Device> {
    Vec::new()
}

#[cfg(any(
    all(target_os = "macos", feature = "metal"),
    all(target_os = "macos", feature = "mlx"),
    feature = "gpu"
))]
fn optional_gemma4_backends() -> Vec<Device> {
    let mut out = Vec::new();
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if is_available(Device::Metal) {
        out.push(Device::Metal);
    }
    #[cfg(all(target_os = "macos", feature = "mlx"))]
    if is_available(Device::Mlx) {
        out.push(Device::Mlx);
    }
    #[cfg(feature = "gpu")]
    if is_available(Device::Gpu) {
        out.push(Device::Gpu);
    }
    out
}

fn backends_to_bench() -> Vec<Device> {
    std::iter::once(Device::Cpu)
        .chain(optional_gemma4_backends())
        .collect()
}

// ── Optional real-weight bench ───────────────────────────────────

fn fixture_dir() -> Option<PathBuf> {
    std::env::var_os("RLX_GEMMA4_FIXTURE").map(PathBuf::from)
}

fn fixture_weights_path(dir: &Path) -> PathBuf {
    if let Ok(p) = std::env::var("RLX_GEMMA4_WEIGHTS") {
        return PathBuf::from(p);
    }
    for name in [
        "gemma-4-12b-it-Q4_K_M.gguf",
        "model.gguf",
        "model.safetensors",
    ] {
        let p = dir.join(name);
        if p.is_file() {
            return p;
        }
    }
    dir.join("model.safetensors")
}

fn estimate_f32_weight_gb(path: &Path) -> f64 {
    path.metadata()
        .map(|m| m.len() as f64 / (1024.0 * 1024.0 * 1024.0))
        .unwrap_or(0.0)
}

const REAL_PROMPTS: &[(&str, &str)] = &[
    (
        "short_chat",
        "User: What is quantum entanglement? Answer in two sentences.\nAssistant:",
    ),
    (
        "code_review",
        "Review this Rust async function for data races:\n```rust\nasync fn fetch_all(urls: Vec<String>) -> Vec<Response> { urls.into_iter().map(reqwest::get).collect().await }\n```\n",
    ),
    (
        "reasoning",
        "A farmer has 17 sheep. All but 9 die. How many are left? Think step by step.\n",
    ),
];

fn bench_real_weights_on(device: Device) -> Result<()> {
    let Some(dir) = fixture_dir() else {
        eprintln!("[gemma4 bench] RLX_GEMMA4_FIXTURE unset — skip real weights");
        return Ok(());
    };
    let weights = fixture_weights_path(&dir);
    if !weights.is_file() {
        eprintln!("[gemma4 bench] {:?} missing — skip real weights", weights);
        return Ok(());
    }
    if !is_available(device) {
        return Ok(());
    }

    let is_gguf = weights.extension().and_then(|s| s.to_str()) == Some("gguf");
    let file_gb = estimate_f32_weight_gb(&weights);
    let est_f32_gb = if is_gguf {
        file_gb * 6.0
    } else {
        file_gb * 2.0
    };
    eprintln!(
        "[gemma4 bench] weights {:?} ({file_gb:.1} GB on disk; ~{est_f32_gb:.0} GB peak if drained to F32)",
        weights.file_name().unwrap_or_default()
    );
    if !is_gguf && est_f32_gb > 56.0 {
        eprintln!(
            "[gemma4 bench] warning: F32 safetensors load likely needs ~{est_f32_gb:.0} GB RAM; \
             use Q4_K_M GGUF on 64 GB hosts (`just fetch-gemma4-12b-it-gguf`)"
        );
    }

    use rlx_gemma::{GemmaConfig, GemmaGenerator, GemmaRunner, encode_prompt_auto};
    use rlx_qwen3::SampleOpts;

    let config_path = dir.join("config.json");
    if !config_path.is_file() {
        bail!("missing config.json in fixture dir");
    }
    let cfg = GemmaConfig::from_file(&config_path)?;
    let tokenizer = dir.join("tokenizer.json");
    let weights_str = weights
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-utf8 weights path"))?;
    let decode_steps = std::env::var("RLX_GEMMA4_REAL_DECODE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32usize);

    eprintln!(
        "\n[gemma4 bench] REAL weights — {device:?} layers={} hidden={} decode_steps={decode_steps} gguf={is_gguf}",
        cfg.num_hidden_layers, cfg.hidden_size
    );

    if is_gguf {
        use rlx_gemma::GemmaConfigSource;
        let max_seq = std::env::var("RLX_GEMMA4_MAX_SEQ")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(128usize);
        let t_session = Instant::now();
        let mut runner = GemmaRunner::builder()
            .weights(&weights)
            .device(device)
            .max_seq(max_seq)
            .stream(false)
            .sample(SampleOpts::greedy())
            .packed_weights(true)
            .config(GemmaConfigSource::JsonFile(config_path))
            .build()?;
        let session_ms = t_session.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[gemma4 bench] GGUF packed path max_seq={max_seq} (prefill + bucketed decode KV cache) session_build={session_ms:.0} ms"
        );
        for (name, prompt) in REAL_PROMPTS {
            if std::env::var("RLX_GEMMA4_BENCH_ONE").is_ok() && *name != "short_chat" {
                continue;
            }
            let ids = encode_prompt_auto(&weights, Some(tokenizer.as_path()), prompt)?;

            let t_e2e = Instant::now();
            let mut n = 0usize;
            runner.generate(&ids, decode_steps, |_| n += 1)?;
            let e2e_ms = t_e2e.elapsed().as_secs_f64() * 1000.0;
            let e2e_tok_s = n as f64 / (e2e_ms / 1000.0);

            eprintln!(
                "  {:<14} prefill={} tok e2e={:.1} tok/s ({} tok in {:.0} ms; includes first prefill + decode)",
                name,
                ids.len(),
                e2e_tok_s,
                n,
                e2e_ms,
            );
        }
        return Ok(());
    }

    let horizon = lm_cache_horizon(&cfg).max(decode_steps + 256);
    let mut g =
        GemmaGenerator::from_path(cfg.clone(), weights_str, device)?.with_inference_caches(horizon);

    let t_warm = Instant::now();
    for (name, prompt) in REAL_PROMPTS {
        let ids = encode_prompt_auto(&weights, Some(tokenizer.as_path()), prompt)?;
        g.prefill_get_last_logits(&ids)?;
        g.prefill(&ids);
        let _ = g.generate_cached(decode_steps.min(8), SampleOpts::greedy())?;
        let _ = (name, prompt);
    }
    g.sync_device();
    eprintln!(
        "[gemma4 bench] compile warmup {device:?}: {:.1} ms (real weights, {} prompts)",
        t_warm.elapsed().as_secs_f64() * 1000.0,
        REAL_PROMPTS.len()
    );

    let mut rows: Vec<(&str, &str, StageTiming)> = Vec::new();
    for (name, prompt) in REAL_PROMPTS {
        let ids = encode_prompt_auto(&weights, Some(tokenizer.as_path()), prompt)?;

        g.prefill(&ids);
        let _ = g.generate_cached(decode_steps.min(4), SampleOpts::greedy())?;

        g.prefill(&ids);
        let t_prefill = Instant::now();
        g.prefill_get_last_logits(&ids)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

        g.prefill(&ids);
        let t_ttft = Instant::now();
        let _ = g.generate_cached(1, SampleOpts::greedy())?;
        let ttft_ms = t_ttft.elapsed().as_secs_f64() * 1000.0;

        let steady_steps = decode_steps.saturating_sub(1);
        g.prefill(&ids);
        let _ = g.generate_cached(1, SampleOpts::greedy())?;
        let t_steady = Instant::now();
        for _ in 0..steady_steps {
            g.step_cached(SampleOpts::greedy())?;
        }
        let steady_decode_ms = t_steady.elapsed().as_secs_f64() * 1000.0;

        rows.push((
            name,
            prompt,
            StageTiming {
                prefill_ms,
                ttft_ms,
                steady_decode_ms,
                decode_steps,
                prefill_tokens: ids.len(),
            },
        ));
    }
    print_lm_table(device, cfg.num_hidden_layers, cfg.hidden_size, &rows);
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────

/// End-to-end across every available RLX backend (sequential — safe for GPU).
#[test]
fn bench_e2e_all_backends() {
    eprintln!("\n========== Gemma 4 e2e bench (all backends) ==========");
    for device in backends_to_bench() {
        eprintln!("\n---------- {device:?} ----------");
        run_synthetic_suite(device).unwrap_or_else(|e| {
            panic!("{device:?} synthetic bench: {e:#}");
        });
    }
    let _ = bench_real_weights_on(Device::Cpu);
    eprintln!("\n========== done ==========");
}

#[test]
fn bench_synthetic_cpu() {
    run_synthetic_suite(Device::Cpu).expect("CPU synthetic bench");
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn bench_synthetic_metal() {
    run_synthetic_suite(Device::Metal).expect("Metal synthetic bench");
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn bench_synthetic_mlx() {
    run_synthetic_suite(Device::Mlx).expect("MLX synthetic bench");
}

#[cfg(feature = "gpu")]
#[test]
fn bench_synthetic_wgpu() {
    run_synthetic_suite(Device::Gpu).expect("wgpu synthetic bench");
}

#[test]
fn bench_real_weights() {
    for device in gemma4_bench_common::bench_devices_from_env() {
        bench_real_weights_on(device).unwrap_or_else(|e| panic!("real weights {device:?}: {e}"));
    }
}
