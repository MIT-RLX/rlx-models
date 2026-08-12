// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Both DiffusionGemma graphs on tiny synthetic weights: the causal MoE encoder
//! with its K/V taps, and the bidirectional denoiser that consumes them.
//!
//! The config keeps every structural relationship of the real checkpoint —
//! mixed sliding/full layers with *different* head widths and KV-head counts,
//! `v_proj` present only on sliding layers, proportional RoPE with a NoPE tail,
//! a two-branch FFN and top-k routing — just scaled down enough to compile fast.

use std::collections::HashMap;

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_diffusiongemma::config::LayerType;
use rlx_diffusiongemma::flow::{
    CANVAS_INPUT, EncoderCacheLens, INPUTS_EMBEDS_INPUT, SC_SIGNAL_INPUT, TEMPERATURE_INPUT,
    build_decoder_flow, build_encoder_flow, build_encoder_flow_embeds, enc_k_name, enc_v_name,
};
use rlx_diffusiongemma::vision::{
    PIXELS_INPUT, POOL_INPUT, POS_X_INPUT, POS_Y_INPUT, ROPE_COS_INPUT, ROPE_SIN_INPUT,
    SOFT_TOKENS_OUTPUT, VALID_INPUT,
};
use rlx_diffusiongemma::{
    DiffusionGemmaConfig, build_vision_flow, grid_positions, prepare_checkpoint,
    vision_pool_matrix, vision_rope_tables,
};
use rlx_runtime::Device;

fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.2
        })
        .collect()
}

/// 4 layers: 3 sliding (4 heads × 8, 2 KV heads, with `v_proj`) and a final
/// full-attention layer (4 heads × 16, 1 KV head, V aliased to K).
pub fn tiny_config() -> DiffusionGemmaConfig {
    DiffusionGemmaConfig::from_json(
        r#"{"model_type":"diffusion_gemma","canvas_length":4,
            "text_config":{
              "vocab_size":32,"hidden_size":16,"intermediate_size":12,
              "num_hidden_layers":4,"num_attention_heads":4,
              "num_key_value_heads":2,"num_global_key_value_heads":1,
              "head_dim":8,"global_head_dim":16,
              "layer_types":["sliding_attention","sliding_attention","sliding_attention","full_attention"],
              "sliding_window":4,"rms_norm_eps":1e-6,
              "final_logit_softcapping":30.0,
              "num_experts":4,"top_k_experts":2,"moe_intermediate_size":6,
              "rope_parameters":{
                "full_attention":{"partial_rotary_factor":0.25,"rope_theta":1000000.0,"rope_type":"proportional"},
                "sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}}}}"#,
    )
    .expect("parse tiny config")
}

pub fn weights(cfg: &DiffusionGemmaConfig) -> WeightMap {
    let t = &cfg.text_config;
    let h = t.hidden_size;
    let mut m: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1u64;
    let mut put =
        |m: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            seed += 3;
            m.insert(k, (fill(n, seed), shape));
        };

    put(
        &mut m,
        "model.decoder.embed_tokens.weight".into(),
        vec![t.vocab_size, h],
    );
    put(&mut m, "model.decoder.norm.weight".into(), vec![h]);
    let sc = "model.decoder.self_conditioning";
    put(&mut m, format!("{sc}.pre_norm.weight"), vec![h]);
    put(
        &mut m,
        format!("{sc}.gate_proj.weight"),
        vec![t.intermediate_size, h],
    );
    put(
        &mut m,
        format!("{sc}.up_proj.weight"),
        vec![t.intermediate_size, h],
    );
    put(
        &mut m,
        format!("{sc}.down_proj.weight"),
        vec![h, t.intermediate_size],
    );

    for l in 0..t.num_hidden_layers {
        let p = format!("model.decoder.layers.{l}");
        let dh = t.layer_head_dim(l);
        let q_dim = t.num_attention_heads * dh;
        let kv_dim = t.layer_kv_heads(l) * dh;
        for n in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
        ] {
            put(&mut m, format!("{p}.{n}.weight"), vec![h]);
        }
        // The encoder and decoder stacks differ only here.
        m.insert(format!("{p}.layer_scalar"), (vec![1.0], vec![1]));
        m.insert(
            format!("model.encoder.language_model.layers.{l}.layer_scalar"),
            (vec![0.9], vec![1]),
        );

        put(
            &mut m,
            format!("{p}.self_attn.q_proj.weight"),
            vec![q_dim, h],
        );
        put(
            &mut m,
            format!("{p}.self_attn.k_proj.weight"),
            vec![kv_dim, h],
        );
        if !t.layer_k_eq_v(l) {
            put(
                &mut m,
                format!("{p}.self_attn.v_proj.weight"),
                vec![kv_dim, h],
            );
        }
        put(&mut m, format!("{p}.self_attn.q_norm.weight"), vec![dh]);
        put(&mut m, format!("{p}.self_attn.k_norm.weight"), vec![dh]);
        put(
            &mut m,
            format!("{p}.self_attn.o_proj.weight"),
            vec![h, q_dim],
        );

        let i = t.intermediate_size;
        put(&mut m, format!("{p}.mlp.gate_proj.weight"), vec![i, h]);
        put(&mut m, format!("{p}.mlp.up_proj.weight"), vec![i, h]);
        put(&mut m, format!("{p}.mlp.down_proj.weight"), vec![h, i]);

        let (e, mi) = (t.num_experts, t.moe_intermediate_size);
        put(&mut m, format!("{p}.router.proj.weight"), vec![e, h]);
        put(&mut m, format!("{p}.router.scale"), vec![h]);
        m.insert(
            format!("{p}.router.per_expert_scale"),
            (vec![1.0; e], vec![e]),
        );
        put(
            &mut m,
            format!("{p}.experts.gate_up_proj"),
            vec![e, 2 * mi, h],
        );
        put(&mut m, format!("{p}.experts.down_proj"), vec![e, h, mi]);
    }
    WeightMap::from_tensors(m)
}

fn ids(n: usize, vocab: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 + 3) % vocab) as f32).collect()
}

/// Run the encoder and return `(hidden, [(k, v); layers])`.
pub fn run_encoder(
    cfg: &DiffusionGemmaConfig,
    wm: &WeightMap,
    seq: usize,
) -> (Vec<f32>, Vec<(Vec<f32>, Vec<f32>)>) {
    let t = &cfg.text_config;
    let built = build_encoder_flow(cfg, wm, seq).expect("build encoder");
    let names: Vec<String> = built.output_names().to_vec();
    let mut compiled = compile_built(built, dev()).expect("compile encoder");

    let (cos_s, sin_s) = t.rope_tables(0, 0, seq);
    let (cos_f, sin_f) = t.rope_tables(3, 0, seq);
    let input_ids = ids(seq, t.vocab_size);
    let outs = compiled.run(&[
        ("input_ids", input_ids.as_slice()),
        ("rope_cos_sliding", cos_s.as_slice()),
        ("rope_sin_sliding", sin_s.as_slice()),
        ("rope_cos_full", cos_f.as_slice()),
        ("rope_sin_full", sin_f.as_slice()),
    ]);
    let by_name: HashMap<&str, &Vec<f32>> =
        names.iter().map(|s| s.as_str()).zip(outs.iter()).collect();
    let hidden = by_name["hidden"].clone();
    let kv = (0..t.num_hidden_layers)
        .map(|l| {
            (
                by_name[enc_k_name(l).as_str()].clone(),
                by_name[enc_v_name(l).as_str()].clone(),
            )
        })
        .collect();
    (hidden, kv)
}

#[test]
fn encoder_compiles_runs_and_taps_kv() {
    let cfg = tiny_config();
    let t = &cfg.text_config;
    // Guard the structure the rest of the test depends on.
    assert_eq!(t.layer_type(0), LayerType::Sliding);
    assert_eq!(t.layer_type(3), LayerType::Full);
    assert_eq!((t.layer_head_dim(0), t.layer_head_dim(3)), (8, 16));
    assert_eq!((t.layer_kv_heads(0), t.layer_kv_heads(3)), (2, 1));
    assert!(!t.layer_k_eq_v(0) && t.layer_k_eq_v(3));

    let mut wm = weights(&cfg);
    prepare_checkpoint(&cfg, &mut wm).expect("prepare");
    let seq = 6usize;
    let (hidden, kv) = run_encoder(&cfg, &wm, seq);

    assert_eq!(hidden.len(), seq * t.hidden_size);
    assert!(
        hidden.iter().all(|v| v.is_finite()),
        "hidden must be finite"
    );
    assert!(hidden.iter().any(|v| v.abs() > 1e-9));

    assert_eq!(kv.len(), t.num_hidden_layers);
    for (l, (k, v)) in kv.iter().enumerate() {
        let kv_dim = t.layer_kv_heads(l) * t.layer_head_dim(l);
        assert_eq!(k.len(), seq * kv_dim, "layer {l} K tap shape");
        assert_eq!(v.len(), seq * kv_dim, "layer {l} V tap shape");
        assert!(k.iter().chain(v.iter()).all(|x| x.is_finite()));
    }
    // The full-attention layer aliases V to K, but they still differ: K is
    // `k_norm`-ed and RoPE'd, V only gets the scale-free `v_norm`.
    let (k3, v3) = &kv[3];
    assert!(
        k3.iter().zip(v3).any(|(a, b)| (a - b).abs() > 1e-4),
        "K and V must diverge even when V aliases K"
    );
}

/// The encoder is causal, so extending the prompt must not change the hidden
/// states already computed for earlier positions. This is the cheapest check
/// that the masks and the RoPE offsets run in the same direction.
#[test]
fn encoder_is_causal() {
    let cfg = tiny_config();
    let h = cfg.text_config.hidden_size;
    let mut wm = weights(&cfg);
    prepare_checkpoint(&cfg, &mut wm).expect("prepare");
    // The sliding window is 4, so compare only positions every layer can see
    // identically in both runs.
    let (short, _) = run_encoder(&cfg, &wm, 4);
    let (long, _) = run_encoder(&cfg, &wm, 6);
    for pos in 0..4 {
        for c in 0..h {
            let (a, b) = (short[pos * h + c], long[pos * h + c]);
            assert!(
                (a - b).abs() <= 2e-4 * a.abs().max(1.0),
                "position {pos} channel {c} moved when the prompt grew: {a} vs {b}"
            );
        }
    }
}

#[test]
fn decoder_compiles_and_runs_against_the_encoder_cache() {
    let cfg = tiny_config();
    let t = &cfg.text_config;
    let (seq, canvas) = (6usize, cfg.canvas_length);
    let mut wm = weights(&cfg);
    prepare_checkpoint(&cfg, &mut wm).expect("prepare");
    let (_, kv) = run_encoder(&cfg, &wm, seq);

    let cache = EncoderCacheLens::for_prompt(t, seq);
    assert_eq!(cache.sliding, 4, "windowed to sliding_window");
    assert_eq!(cache.full, 6);

    let built = build_decoder_flow(&cfg, &wm, canvas, cache).expect("build decoder");
    let names: Vec<String> = built.output_names().to_vec();
    let mut compiled = compile_built(built, dev()).expect("compile decoder");

    // Canvas positions continue after the prompt.
    let (cos_s, sin_s) = t.rope_tables(0, seq, canvas);
    let (cos_f, sin_f) = t.rope_tables(3, seq, canvas);
    let canvas_ids = ids(canvas, t.vocab_size);
    let sc_signal = vec![0f32; canvas * t.hidden_size];
    let temperature = vec![0.8f32];

    // Slice each layer's tap to the length that layer's cache keeps.
    let sliced: Vec<(Vec<f32>, Vec<f32>)> = (0..t.num_hidden_layers)
        .map(|l| {
            let kv_dim = t.layer_kv_heads(l) * t.layer_head_dim(l);
            let keep = cache.for_layer(t, l);
            let start = (seq - keep) * kv_dim;
            (kv[l].0[start..].to_vec(), kv[l].1[start..].to_vec())
        })
        .collect();

    let mut inputs: Vec<(&str, &[f32])> = vec![
        (CANVAS_INPUT, canvas_ids.as_slice()),
        (SC_SIGNAL_INPUT, sc_signal.as_slice()),
        (TEMPERATURE_INPUT, temperature.as_slice()),
        ("rope_cos_sliding", cos_s.as_slice()),
        ("rope_sin_sliding", sin_s.as_slice()),
        ("rope_cos_full", cos_f.as_slice()),
        ("rope_sin_full", sin_f.as_slice()),
    ];
    let kn: Vec<String> = (0..t.num_hidden_layers).map(enc_k_name).collect();
    let vn: Vec<String> = (0..t.num_hidden_layers).map(enc_v_name).collect();
    for l in 0..t.num_hidden_layers {
        inputs.push((kn[l].as_str(), sliced[l].0.as_slice()));
        inputs.push((vn[l].as_str(), sliced[l].1.as_slice()));
    }
    let outs = compiled.run(&inputs);
    let by_name: HashMap<&str, &Vec<f32>> =
        names.iter().map(|s| s.as_str()).zip(outs.iter()).collect();

    let logits = by_name["logits"];
    assert_eq!(logits.len(), canvas * t.vocab_size);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits must be finite"
    );
    assert!(logits.iter().any(|v| v.abs() > 1e-9));
    // Soft cap then temperature: |logits| <= softcap / temperature.
    let bound = t.final_logit_softcapping / temperature[0];
    assert!(
        logits.iter().all(|v| v.abs() <= bound + 1e-3),
        "soft cap not applied: max {}",
        logits.iter().fold(0f32, |a, b| a.max(b.abs()))
    );

    let soft = by_name["soft_embeds"];
    assert_eq!(soft.len(), canvas * t.hidden_size);
    assert!(soft.iter().all(|v| v.is_finite()));
    assert!(soft.iter().any(|v| v.abs() > 1e-9));
}

/// The denoiser is bidirectional: every canvas position sees every other, so
/// changing the *last* canvas token must move the *first* position's logits.
/// A causal decoder would leave them untouched — this is the check that catches
/// a stray causal mask.
#[test]
fn decoder_is_bidirectional_over_the_canvas() {
    let cfg = tiny_config();
    let t = &cfg.text_config;
    let (seq, canvas) = (6usize, cfg.canvas_length);
    let mut wm = weights(&cfg);
    prepare_checkpoint(&cfg, &mut wm).expect("prepare");
    let (_, kv) = run_encoder(&cfg, &wm, seq);
    let cache = EncoderCacheLens::for_prompt(t, seq);

    let built = build_decoder_flow(&cfg, &wm, canvas, cache).expect("build decoder");
    let names: Vec<String> = built.output_names().to_vec();
    let mut compiled = compile_built(built, dev()).expect("compile decoder");

    let (cos_s, sin_s) = t.rope_tables(0, seq, canvas);
    let (cos_f, sin_f) = t.rope_tables(3, seq, canvas);
    let sc_signal = vec![0f32; canvas * t.hidden_size];
    let temperature = vec![1.0f32];
    let sliced: Vec<(Vec<f32>, Vec<f32>)> = (0..t.num_hidden_layers)
        .map(|l| {
            let kv_dim = t.layer_kv_heads(l) * t.layer_head_dim(l);
            let start = (seq - cache.for_layer(t, l)) * kv_dim;
            (kv[l].0[start..].to_vec(), kv[l].1[start..].to_vec())
        })
        .collect();
    let kn: Vec<String> = (0..t.num_hidden_layers).map(enc_k_name).collect();
    let vn: Vec<String> = (0..t.num_hidden_layers).map(enc_v_name).collect();

    let mut run_with = |canvas_ids: &[f32]| -> Vec<f32> {
        let mut inputs: Vec<(&str, &[f32])> = vec![
            (CANVAS_INPUT, canvas_ids),
            (SC_SIGNAL_INPUT, sc_signal.as_slice()),
            (TEMPERATURE_INPUT, temperature.as_slice()),
            ("rope_cos_sliding", cos_s.as_slice()),
            ("rope_sin_sliding", sin_s.as_slice()),
            ("rope_cos_full", cos_f.as_slice()),
            ("rope_sin_full", sin_f.as_slice()),
        ];
        for l in 0..t.num_hidden_layers {
            inputs.push((kn[l].as_str(), sliced[l].0.as_slice()));
            inputs.push((vn[l].as_str(), sliced[l].1.as_slice()));
        }
        let outs = compiled.run(&inputs);
        let idx = names.iter().position(|n| n == "logits").unwrap();
        outs[idx].clone()
    };

    let base = run_with(&[1.0, 2.0, 3.0, 4.0]);
    // Baseline: the graph is deterministic, so an identical input reproduces
    // the logits bit for bit. Any difference below is therefore real
    // dependence, not numerical noise — no magnitude threshold needed.
    let repeat = run_with(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(base, repeat, "decoder must be deterministic");

    let tweaked = run_with(&[1.0, 2.0, 3.0, 29.0]); // only the LAST token differs
    let v = t.vocab_size;
    let moved: f32 = (0..v).map(|c| (base[c] - tweaked[c]).abs()).sum();
    assert!(
        moved > 0.0,
        "position 0 did not react to a change at the end of the canvas; \
         the denoiser must be bidirectional"
    );
    // And the change must reach every position, not just the one that moved.
    for pos in 0..canvas {
        let d: f32 = (0..v)
            .map(|c| (base[pos * v + c] - tweaked[pos * v + c]).abs())
            .sum();
        assert!(d > 0.0, "position {pos} did not see the canvas edit");
    }
}

/// The self-conditioning signal must actually reach the output — a dropped
/// `sc_signal` input would silently make every denoising step identical.
#[test]
fn self_conditioning_signal_changes_the_output() {
    let cfg = tiny_config();
    let t = &cfg.text_config;
    let (seq, canvas) = (6usize, cfg.canvas_length);
    let mut wm = weights(&cfg);
    prepare_checkpoint(&cfg, &mut wm).expect("prepare");
    let (_, kv) = run_encoder(&cfg, &wm, seq);
    let cache = EncoderCacheLens::for_prompt(t, seq);

    let built = build_decoder_flow(&cfg, &wm, canvas, cache).expect("build decoder");
    let names: Vec<String> = built.output_names().to_vec();
    let mut compiled = compile_built(built, dev()).expect("compile decoder");

    let (cos_s, sin_s) = t.rope_tables(0, seq, canvas);
    let (cos_f, sin_f) = t.rope_tables(3, seq, canvas);
    let canvas_ids = ids(canvas, t.vocab_size);
    let temperature = vec![1.0f32];
    let sliced: Vec<(Vec<f32>, Vec<f32>)> = (0..t.num_hidden_layers)
        .map(|l| {
            let kv_dim = t.layer_kv_heads(l) * t.layer_head_dim(l);
            let start = (seq - cache.for_layer(t, l)) * kv_dim;
            (kv[l].0[start..].to_vec(), kv[l].1[start..].to_vec())
        })
        .collect();
    let kn: Vec<String> = (0..t.num_hidden_layers).map(enc_k_name).collect();
    let vn: Vec<String> = (0..t.num_hidden_layers).map(enc_v_name).collect();

    let mut run_with = |sc: &[f32]| -> Vec<f32> {
        let mut inputs: Vec<(&str, &[f32])> = vec![
            (CANVAS_INPUT, canvas_ids.as_slice()),
            (SC_SIGNAL_INPUT, sc),
            (TEMPERATURE_INPUT, temperature.as_slice()),
            ("rope_cos_sliding", cos_s.as_slice()),
            ("rope_sin_sliding", sin_s.as_slice()),
            ("rope_cos_full", cos_f.as_slice()),
            ("rope_sin_full", sin_f.as_slice()),
        ];
        for l in 0..t.num_hidden_layers {
            inputs.push((kn[l].as_str(), sliced[l].0.as_slice()));
            inputs.push((vn[l].as_str(), sliced[l].1.as_slice()));
        }
        let outs = compiled.run(&inputs);
        let idx = names.iter().position(|n| n == "logits").unwrap();
        outs[idx].clone()
    };

    let zeros = run_with(&vec![0f32; canvas * t.hidden_size]);
    let signal = run_with(&vec![0.7f32; canvas * t.hidden_size]);
    let delta: f32 = zeros.iter().zip(&signal).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        delta > 1e-4,
        "self-conditioning signal had no effect (sum |Δ| = {delta})"
    );
}

/// The multimodal encoder entry point must agree exactly with the id path when
/// it is fed the same embeddings the id path would have built. This is what
/// makes splicing vision soft tokens safe: nothing else about the stack changes.
#[test]
fn encoder_from_embeds_matches_the_id_path() {
    let cfg = tiny_config();
    let t = &cfg.text_config;
    let seq = 6usize;
    let mut wm = weights(&cfg);
    prepare_checkpoint(&cfg, &mut wm).expect("prepare");
    let (want, _) = run_encoder(&cfg, &wm, seq);

    // Build the same embeddings by hand: table lookup times sqrt(hidden).
    let ids = ids(seq, t.vocab_size);
    let (table, shape) = wm.get(rlx_diffusiongemma::EMBED_KEY).unwrap();
    assert_eq!(shape, &[t.vocab_size, t.hidden_size]);
    let scale = t.embed_scale();
    let mut embeds = vec![0f32; seq * t.hidden_size];
    for (pos, &id) in ids.iter().enumerate() {
        let row = id as usize;
        for c in 0..t.hidden_size {
            embeds[pos * t.hidden_size + c] = table[row * t.hidden_size + c] * scale;
        }
    }

    let built = build_encoder_flow_embeds(&cfg, &wm, seq).expect("build embeds encoder");
    let names: Vec<String> = built.output_names().to_vec();
    let mut compiled = compile_built(built, dev()).expect("compile embeds encoder");
    let (cos_s, sin_s) = t.rope_tables(0, 0, seq);
    let (cos_f, sin_f) = t.rope_tables(3, 0, seq);
    let outs = compiled.run(&[
        (INPUTS_EMBEDS_INPUT, embeds.as_slice()),
        ("rope_cos_sliding", cos_s.as_slice()),
        ("rope_sin_sliding", sin_s.as_slice()),
        ("rope_cos_full", cos_f.as_slice()),
        ("rope_sin_full", sin_f.as_slice()),
    ]);
    let idx = names.iter().position(|n| n == "hidden").unwrap();
    let got = &outs[idx];
    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!(
            (a - b).abs() <= 1e-5 * a.abs().max(1.0),
            "channel {i} diverged: {a} vs {b}"
        );
    }
}

/// The vision tower compiles and runs, and its soft tokens actually depend on
/// the pixels — a dropped input would still produce a plausible-looking block.
#[test]
fn vision_tower_compiles_and_reacts_to_pixels() {
    let cfg = vision_tiny_config();
    let v = cfg.vision_config.as_ref().unwrap();
    let wm = vision_weights(&cfg);
    let (grid, k) = (4usize, v.pooling_kernel_size);
    let positions = grid_positions(grid, grid);
    let patches = positions.len();
    let soft_len = patches / (k * k);

    let built = build_vision_flow(&cfg, &wm, patches, soft_len).expect("build vision");
    let names: Vec<String> = built.output_names().to_vec();
    let mut compiled = compile_built(built, dev()).expect("compile vision");

    let (cos, sin) = vision_rope_tables(v, &positions);
    let pool = vision_pool_matrix(&positions, k, soft_len);
    let pos_x: Vec<f32> = positions.iter().map(|p| p.0 as f32).collect();
    let pos_y: Vec<f32> = positions.iter().map(|p| p.1 as f32).collect();
    let valid = vec![1f32; patches];
    let patch_dim = 3 * v.patch_size * v.patch_size;

    let mut run = |pixels: &[f32]| -> Vec<f32> {
        let outs = compiled.run(&[
            (PIXELS_INPUT, pixels),
            (POS_X_INPUT, pos_x.as_slice()),
            (POS_Y_INPUT, pos_y.as_slice()),
            (ROPE_COS_INPUT, cos.as_slice()),
            (ROPE_SIN_INPUT, sin.as_slice()),
            (VALID_INPUT, valid.as_slice()),
            (POOL_INPUT, pool.as_slice()),
        ]);
        let idx = names.iter().position(|n| n == SOFT_TOKENS_OUTPUT).unwrap();
        outs[idx].clone()
    };

    let a = run(&fill(patches * patch_dim, 11));
    assert_eq!(a.len(), soft_len * cfg.text_config.hidden_size);
    assert!(
        a.iter().all(|x| x.is_finite()),
        "soft tokens must be finite"
    );
    assert!(a.iter().any(|x| x.abs() > 1e-9));

    let b = run(&fill(patches * patch_dim, 77));
    assert_ne!(a, b, "soft tokens must depend on the pixels");
}

/// The tiny text config plus a 2-layer vision tower.
pub fn vision_tiny_config() -> DiffusionGemmaConfig {
    let raw = r#"{"model_type":"diffusion_gemma","canvas_length":4,
        "vision_config":{
          "hidden_size":24,"num_hidden_layers":2,"num_attention_heads":2,
          "head_dim":12,"intermediate_size":20,"patch_size":2,
          "pooling_kernel_size":2,"position_embedding_size":32,
          "rms_norm_eps":1e-6,"standardize":true,"use_clipped_linears":false,
          "rope_parameters":{"rope_theta":100.0,"rope_type":"default"}},
        "text_config":{
          "vocab_size":32,"hidden_size":16,"intermediate_size":12,
          "num_hidden_layers":4,"num_attention_heads":4,
          "num_key_value_heads":2,"num_global_key_value_heads":1,
          "head_dim":8,"global_head_dim":16,
          "layer_types":["sliding_attention","sliding_attention","sliding_attention","full_attention"],
          "sliding_window":4,"rms_norm_eps":1e-6,"final_logit_softcapping":30.0,
          "num_experts":4,"top_k_experts":2,"moe_intermediate_size":6,
          "rope_parameters":{
            "full_attention":{"partial_rotary_factor":0.25,"rope_theta":1000000.0,"rope_type":"proportional"},
            "sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}}}}"#;
    DiffusionGemmaConfig::from_json(raw).expect("parse vision tiny config")
}

/// Text weights plus the vision tower and its projector.
pub fn vision_weights(cfg: &DiffusionGemmaConfig) -> WeightMap {
    let v = cfg.vision_config.as_ref().unwrap();
    let mut m: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    // Reuse the text weights, then add the tower.
    let text = weights(cfg);
    for k in text.keys().map(|s| s.to_string()).collect::<Vec<_>>() {
        let (d, sh) = text.get(&k).unwrap();
        m.insert(k, (d.to_vec(), sh.to_vec()));
    }
    let mut seed = 5000u64;
    let mut put =
        |m: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            seed += 7;
            m.insert(k, (fill(n, seed), shape));
        };
    let vp = "model.encoder.vision_tower";
    let h = v.hidden_size;
    put(
        &mut m,
        format!("{vp}.patch_embedder.input_proj.weight"),
        vec![h, 3 * v.patch_size * v.patch_size],
    );
    put(
        &mut m,
        format!("{vp}.patch_embedder.position_embedding_table"),
        vec![2, v.position_embedding_size, h],
    );
    put(&mut m, format!("{vp}.std_bias"), vec![h]);
    put(&mut m, format!("{vp}.std_scale"), vec![h]);
    for i in 0..v.num_hidden_layers {
        let p = format!("{vp}.encoder.layers.{i}");
        for n in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            put(&mut m, format!("{p}.{n}.weight"), vec![h]);
        }
        for n in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            put(
                &mut m,
                format!("{p}.self_attn.{n}.linear.weight"),
                vec![h, h],
            );
        }
        put(
            &mut m,
            format!("{p}.self_attn.q_norm.weight"),
            vec![v.head_dim],
        );
        put(
            &mut m,
            format!("{p}.self_attn.k_norm.weight"),
            vec![v.head_dim],
        );
        put(
            &mut m,
            format!("{p}.mlp.gate_proj.linear.weight"),
            vec![v.intermediate_size, h],
        );
        put(
            &mut m,
            format!("{p}.mlp.up_proj.linear.weight"),
            vec![v.intermediate_size, h],
        );
        put(
            &mut m,
            format!("{p}.mlp.down_proj.linear.weight"),
            vec![h, v.intermediate_size],
        );
    }
    put(
        &mut m,
        "model.encoder.embed_vision.embedding_projection.weight".into(),
        vec![cfg.text_config.hidden_size, h],
    );
    WeightMap::from_tensors(m)
}

/// The in-graph reduction must agree with the host sampler on the same logits:
/// identical argmax, entropy to f32 tolerance, and a `sampled` draw that is a
/// valid token. Sampling itself uses Gumbel-max rather than inverse-CDF, so the
/// individual draws are not expected to match the host RNG — only the
/// distribution is the same.
#[test]
fn in_graph_reduction_matches_the_host_sampler() {
    use rlx_diffusiongemma::flow::{
        ARGMAX_OUTPUT, DecoderOutputs, ENTROPY_OUTPUT, SAMPLED_OUTPUT, build_decoder_flow_with,
    };
    use rlx_diffusiongemma::sampler::{Rng, StepScores};

    let cfg = tiny_config();
    let t = &cfg.text_config;
    let (seq, canvas) = (6usize, cfg.canvas_length);
    let mut wm = weights(&cfg);
    prepare_checkpoint(&cfg, &mut wm).expect("prepare");
    let (_, kv) = run_encoder(&cfg, &wm, seq);
    let cache = EncoderCacheLens::for_prompt(t, seq);

    let (cos_s, sin_s) = t.rope_tables(0, seq, canvas);
    let (cos_f, sin_f) = t.rope_tables(3, seq, canvas);
    let canvas_ids = ids(canvas, t.vocab_size);
    let sc = vec![0f32; canvas * t.hidden_size];
    let temp = vec![0.8f32];
    let sliced: Vec<(Vec<f32>, Vec<f32>)> = (0..t.num_hidden_layers)
        .map(|l| {
            let kv_dim = t.layer_kv_heads(l) * t.layer_head_dim(l);
            let start = (seq - cache.for_layer(t, l)) * kv_dim;
            (kv[l].0[start..].to_vec(), kv[l].1[start..].to_vec())
        })
        .collect();
    let kn: Vec<String> = (0..t.num_hidden_layers).map(enc_k_name).collect();
    let vn: Vec<String> = (0..t.num_hidden_layers).map(enc_v_name).collect();

    let inputs = |extra: &mut Vec<(&'static str, Vec<f32>)>| {
        extra.push((CANVAS_INPUT, canvas_ids.clone()));
        extra.push((SC_SIGNAL_INPUT, sc.clone()));
        extra.push((TEMPERATURE_INPUT, temp.clone()));
        extra.push(("rope_cos_sliding", cos_s.clone()));
        extra.push(("rope_sin_sliding", sin_s.clone()));
        extra.push(("rope_cos_full", cos_f.clone()));
        extra.push(("rope_sin_full", sin_f.clone()));
    };

    let run = |outputs: DecoderOutputs| -> (Vec<String>, Vec<Vec<f32>>) {
        let built = build_decoder_flow_with(&cfg, &wm, canvas, cache, outputs).expect("build");
        let names: Vec<String> = built.output_names().to_vec();
        let mut compiled = compile_built(built, dev()).expect("compile");
        let mut base: Vec<(&'static str, Vec<f32>)> = Vec::new();
        inputs(&mut base);
        let mut refs: Vec<(&str, &[f32])> = base.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        for l in 0..t.num_hidden_layers {
            refs.push((kn[l].as_str(), sliced[l].0.as_slice()));
            refs.push((vn[l].as_str(), sliced[l].1.as_slice()));
        }
        (names, compiled.run(&refs))
    };

    // Ground truth: full logits reduced on the host.
    let (names_l, outs_l) = run(DecoderOutputs::Logits);
    let logits = &outs_l[names_l.iter().position(|n| n == "logits").unwrap()];
    let mut rng = Rng::seed_from_u64(4);
    let host = StepScores::from_logits(logits, canvas, t.vocab_size, &mut rng);

    let (names_r, outs_r) = run(DecoderOutputs::Reduced { seed: 4 });
    let by: HashMap<&str, &Vec<f32>> = names_r
        .iter()
        .map(|s| s.as_str())
        .zip(outs_r.iter())
        .collect();
    let entropy = by[ENTROPY_OUTPUT];
    let argmax = by[ARGMAX_OUTPUT];
    let sampled = by[SAMPLED_OUTPUT];
    assert_eq!(entropy.len(), canvas);
    assert_eq!(argmax.len(), canvas);
    assert_eq!(sampled.len(), canvas);

    for c in 0..canvas {
        assert_eq!(
            argmax[c] as u32, host.argmax[c],
            "position {c}: in-graph argmax disagrees"
        );
        assert!(
            (entropy[c] - host.entropy[c]).abs() < 2e-4,
            "position {c}: entropy {} vs host {}",
            entropy[c],
            host.entropy[c]
        );
        let s = sampled[c];
        assert!(
            s >= 0.0 && (s as usize) < t.vocab_size && s.fract() == 0.0,
            "position {c}: sampled {s} is not a token id"
        );
    }
    // Entropy is a real quantity, not an artifact of an all-zero reduction.
    assert!(entropy.iter().all(|e| e.is_finite() && *e >= 0.0));
    assert!(entropy.iter().any(|e| *e > 1e-3));
}
