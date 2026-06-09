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

#![allow(dead_code)]

use anyhow::Result;
use rlx_core::flow_util::{compile_built, graph_from_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::BuiltModel;
use rlx_locateanything::compile_support::{lm_decode_compile_options, metal_lm_compile_guard};
use rlx_locateanything::config::{LocateAnythingConfig, LocateAnythingTextConfig, MoonVitConfig};
use rlx_locateanything::lm_flow::{
    build_locateanything_decode_built, build_locateanything_prefill_built, compute_rope_slice,
};
use rlx_locateanything::moonvit::MoonVitCache;
use rlx_locateanything::preprocess::preprocess_image;
use rlx_locateanything::projector::build_projector_built;
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const BATCH: usize = 1;
const SEQ: usize = 4;
const N_VISION: usize = 2;

pub fn tiny_cfg() -> LocateAnythingConfig {
    LocateAnythingConfig {
        model_type: "locateanything".into(),
        image_token_index: 99,
        box_start_token_id: 1,
        box_end_token_id: 2,
        coord_start_token_id: 3,
        coord_end_token_id: 4,
        ref_start_token_id: 5,
        ref_end_token_id: 6,
        none_token_id: 7,
        mlp_connector_layers: 2,
        mlp_checkpoint: false,
        text_config: LocateAnythingTextConfig {
            vocab_size: 32,
            // head_dim = hidden / heads must be 8 for GPU attention reshapes (see qwen3_common).
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            block_size: 6,
            causal_attn: true,
            bos_token_id: 0,
            eos_token_id: 1,
            null_token_id: None,
            switch_token_id: None,
            text_mask_token_id: None,
        },
        vision_config: MoonVitConfig {
            model_type: "moonvit".into(),
            hidden_size: 16,
            intermediate_size: 32,
            num_attention_heads: 4,
            num_hidden_layers: 1,
            patch_size: 14,
            merge_kernel_size: [2, 2],
            init_pos_emb_height: 4,
            init_pos_emb_width: 4,
        },
        preprocessor: Default::default(),
    }
}

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| 0.001 + scale * (i as f32) * 0.01).collect()
}

pub fn synthetic_projector_weights(cfg: &LocateAnythingConfig) -> WeightMap {
    let in_dim = cfg.projector_input_dim();
    let out_dim = cfg.text_config.hidden_size;
    let z = |n: usize| vec![0.02f32; n];
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert("mlp1.0.weight".into(), (z(in_dim), vec![in_dim]));
    t.insert("mlp1.0.bias".into(), (z(in_dim), vec![in_dim]));
    t.insert(
        "mlp1.1.weight".into(),
        (ramp(out_dim * in_dim, 0.01), vec![out_dim, in_dim]),
    );
    t.insert("mlp1.1.bias".into(), (z(out_dim), vec![out_dim]));
    t.insert(
        "mlp1.3.weight".into(),
        (ramp(out_dim * out_dim, 0.01), vec![out_dim, out_dim]),
    );
    t.insert("mlp1.3.bias".into(), (z(out_dim), vec![out_dim]));
    WeightMap::from_tensors(t)
}

pub fn synthetic_lm_weights(cfg: &LocateAnythingConfig) -> WeightMap {
    let tc = &cfg.text_config;
    let h = tc.hidden_size;
    let q_dim = tc.num_attention_heads * tc.head_dim();
    let kv_dim = tc.num_key_value_heads * tc.head_dim();
    let int_dim = tc.intermediate_size;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    t.insert(
        "language_model.model.embed_tokens.weight".into(),
        (ramp(tc.vocab_size * h, 0.001), vec![tc.vocab_size, h]),
    );
    for i in 0..tc.num_hidden_layers {
        let lp = format!("language_model.model.layers.{i}");
        t.insert(
            format!("{lp}.input_layernorm.weight"),
            (vec![1.0; h], vec![h]),
        );
        t.insert(
            format!("{lp}.post_attention_layernorm.weight"),
            (vec![1.0; h], vec![h]),
        );
        t.insert(
            format!("{lp}.self_attn.q_proj.weight"),
            (ramp(q_dim * h, 0.01), vec![q_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.q_proj.bias"),
            (ramp(q_dim, 0.001), vec![q_dim]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (ramp(kv_dim * h, 0.01), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.bias"),
            (ramp(kv_dim, 0.001), vec![kv_dim]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (ramp(kv_dim * h, 0.01), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.bias"),
            (ramp(kv_dim, 0.001), vec![kv_dim]),
        );
        t.insert(
            format!("{lp}.self_attn.o_proj.weight"),
            (ramp(h * q_dim, 0.01), vec![h, q_dim]),
        );
        t.insert(
            format!("{lp}.mlp.gate_proj.weight"),
            (ramp(int_dim * h, 0.01), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.up_proj.weight"),
            (ramp(int_dim * h, 0.01), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.down_proj.weight"),
            (ramp(h * int_dim, 0.01), vec![h, int_dim]),
        );
    }
    t.insert(
        "language_model.model.norm.weight".into(),
        (vec![1.0; h], vec![h]),
    );
    t.insert(
        "language_model.lm_head.weight".into(),
        (ramp(tc.vocab_size * h, 0.001), vec![tc.vocab_size, h]),
    );
    WeightMap::from_tensors(t)
}

struct CachedProjector {
    compiled: CompiledGraph,
    vision: Vec<f32>,
}

fn per_device_projector_cache() -> &'static Mutex<HashMap<Device, CachedProjector>> {
    static CACHE: OnceLock<Mutex<HashMap<Device, CachedProjector>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn compile_projector(device: Device) -> CachedProjector {
    let cfg = tiny_cfg();
    let in_dim = cfg.projector_input_dim();
    let mut wm = synthetic_projector_weights(&cfg);
    let built = build_projector_built(&cfg, &mut wm, BATCH, N_VISION).expect("projector build");
    let params = built.model.params().clone();
    let mut compiled = compile_built(built.model, device).expect("projector compile");
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    CachedProjector {
        vision: ramp(N_VISION * in_dim, 0.05),
        compiled,
    }
}

pub fn run_projector_on_device(device: Device) {
    let mut cache = per_device_projector_cache().lock().unwrap();
    let entry = cache
        .entry(device)
        .or_insert_with(|| compile_projector(device));
    let out = entry
        .compiled
        .run(&[("vision", entry.vision.as_slice())])
        .into_iter()
        .next()
        .expect("projector output");
    let cfg = tiny_cfg();
    assert_eq!(out.len(), N_VISION * cfg.text_config.hidden_size);
    assert!(out.iter().all(|v| v.is_finite()));
}

pub fn run_projector_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip locateanything projector {device:?}: backend not available");
        return;
    }
    run_projector_on_device(device);
}

pub fn synthetic_vision_weights(cfg: &LocateAnythingConfig) -> WeightMap {
    let h = cfg.vision_config.hidden_size;
    let ps = cfg.vision_config.patch_size;
    let z = |n: usize| vec![0.02f32; n];
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert(
        "vision_model.patch_embed.proj.weight".into(),
        (z(h * 3 * ps * ps), vec![h, 3, ps, ps]),
    );
    t.insert("vision_model.patch_embed.proj.bias".into(), (z(h), vec![h]));
    t.insert(
        "vision_model.patch_embed.pos_emb.weight".into(),
        (z(4 * 4 * h), vec![4, 4, h]),
    );
    let lp = "vision_model.encoder.blocks.0";
    t.insert(format!("{lp}.norm0.weight"), (z(h), vec![h]));
    t.insert(format!("{lp}.norm0.bias"), (z(h), vec![h]));
    t.insert(format!("{lp}.wqkv.weight"), (z(h * 3 * h), vec![h * 3, h]));
    t.insert(format!("{lp}.wqkv.bias"), (z(h * 3), vec![h * 3]));
    t.insert(format!("{lp}.wo.weight"), (z(h * h), vec![h, h]));
    t.insert(format!("{lp}.wo.bias"), (z(h), vec![h]));
    t.insert(format!("{lp}.norm1.weight"), (z(h), vec![h]));
    t.insert(format!("{lp}.norm1.bias"), (z(h), vec![h]));
    t.insert(format!("{lp}.mlp.fc0.weight"), (z(32 * h), vec![32, h]));
    t.insert(format!("{lp}.mlp.fc0.bias"), (z(32), vec![32]));
    t.insert(format!("{lp}.mlp.fc1.weight"), (z(h * 32), vec![h, 32]));
    t.insert(format!("{lp}.mlp.fc1.bias"), (z(h), vec![h]));
    t.insert(
        "vision_model.encoder.final_layernorm.weight".into(),
        (z(h), vec![h]),
    );
    t.insert(
        "vision_model.encoder.final_layernorm.bias".into(),
        (z(h), vec![h]),
    );
    WeightMap::from_tensors(t)
}

pub fn run_moonvit_on_device(device: Device) {
    let cfg = tiny_cfg();
    let img = image::RgbImage::new(28, 28);
    let prep = preprocess_image(&image::DynamicImage::ImageRgb8(img), &cfg).expect("preprocess");
    let mut wm = synthetic_vision_weights(&cfg);
    let mut cache = MoonVitCache::default();
    let out = cache
        .encode(&cfg.vision_config, Some(&mut wm), &prep, device)
        .expect("moonvit encode");
    let expect_len = prep.num_patches() * cfg.vision_config.hidden_size;
    assert_eq!(out.len(), expect_len);
    assert!(out.iter().all(|v| v.is_finite()));
}

pub fn run_moonvit_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip locateanything moonvit {device:?}: backend not available");
        return;
    }
    run_moonvit_on_device(device);
}

struct CachedPrefill {
    compiled: CompiledGraph,
    inputs: Vec<f32>,
}

fn per_device_prefill_cache() -> &'static Mutex<HashMap<Device, CachedPrefill>> {
    static CACHE: OnceLock<Mutex<HashMap<Device, CachedPrefill>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn compile_prefill(device: Device) -> CachedPrefill {
    let cfg = tiny_cfg();
    let h = cfg.text_config.hidden_size;
    let mut wm = synthetic_lm_weights(&cfg);
    let built = build_locateanything_prefill_built(&cfg, &mut wm, BATCH, SEQ, false, true)
        .expect("prefill build");
    let params = built.params().clone();
    let mut compiled = compile_built(built, device).expect("prefill compile");
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    CachedPrefill {
        inputs: ramp(BATCH * SEQ * h, 0.02),
        compiled,
    }
}

pub fn run_prefill_last_logits_on_device(device: Device) {
    let mut cache = per_device_prefill_cache().lock().unwrap();
    let entry = cache
        .entry(device)
        .or_insert_with(|| compile_prefill(device));
    let logits = entry
        .compiled
        .run(&[("inputs_embeds", entry.inputs.as_slice())])
        .into_iter()
        .next()
        .expect("logits");
    assert_eq!(logits.len(), tiny_cfg().text_config.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

pub fn run_prefill_last_logits_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip locateanything prefill {device:?}: backend not available");
        return;
    }
    run_prefill_last_logits_on_device(device);
}

/// Prefill + KV-carry decode with dynamic past (production path on all backends).
pub fn run_decode_step(device: Device) {
    let cfg = tiny_cfg();
    let layers = cfg.text_config.num_hidden_layers;
    let vocab = cfg.text_config.vocab_size;
    let qcfg = cfg.text_config.to_qwen3_config();

    let mut wm = synthetic_lm_weights(&cfg);
    let pre_built =
        build_locateanything_prefill_built(&cfg, &mut wm, BATCH, SEQ, true, true).expect("prefill");
    let pre_params = pre_built.params().clone();
    let mut pre = compile_built(pre_built, device).expect("prefill compile");
    for (n, d) in &pre_params {
        pre.set_param(n, d);
    }
    let h = cfg.text_config.hidden_size;
    let inputs = ramp(BATCH * SEQ * h, 0.02);
    let pre_out = pre.run(&[("inputs_embeds", inputs.as_slice())]);
    assert_eq!(pre_out[0].len(), vocab);
    let kv = &pre_out[1..];

    let mut wm_dec = synthetic_lm_weights(&cfg);
    let dec_built =
        build_locateanything_decode_built(&cfg, &mut wm_dec, BATCH, SEQ, false).expect("decode");
    let dec_params = dec_built.params().clone();
    let mut dec = compile_decode_built(dec_built, device).expect("decode compile");
    for (n, d) in &dec_params {
        dec.set_param(n, d);
    }
    let token = [7f32];
    let (cos, sin) = compute_rope_slice(&qcfg, SEQ);
    let key_past: Vec<String> = (0..layers)
        .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
        .collect();
    let mut dec_in: Vec<(&str, &[f32])> = vec![
        ("input_ids", &token),
        ("rope_cos", cos.as_slice()),
        ("rope_sin", sin.as_slice()),
    ];
    for i in 0..layers {
        dec_in.push((key_past[2 * i].as_str(), kv[2 * i].as_slice()));
        dec_in.push((key_past[2 * i + 1].as_str(), kv[2 * i + 1].as_slice()));
    }
    let step = dec.run(&dec_in);
    assert_eq!(step[0].len(), vocab);
    assert!(step[0].iter().all(|v| v.is_finite()));
    assert_eq!(step.len(), 1 + 2 * layers);
}

fn compile_decode_built(built: BuiltModel, device: Device) -> Result<CompiledGraph> {
    let options = lm_decode_compile_options(device);
    let (graph, params) = graph_from_built(built)?;
    metal_lm_compile_guard(device, || {
        let mut compiled = Session::new(device).compile_with(graph, &options);
        for (name, data) in params {
            compiled.set_param(&name, &data);
        }
        Ok(compiled)
    })
}

pub fn run_decode_step_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip locateanything decode {device:?}: backend not available");
        return;
    }
    run_decode_step(device);
}
