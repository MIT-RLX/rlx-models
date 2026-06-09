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

//! Shared helpers for V-JEPA2 cross-backend check / parity tests.

#![allow(dead_code)]

use rlx_models::vjepa2::{
    Vjepa2Config, Vjepa2GraphParams, Vjepa2Masks, build_vjepa2_encoder_graph_sized,
    build_vjepa2_pooler_graph_sized, build_vjepa2_predictor_graph_sized, conv3d_patch_embed,
    extract_model_weights, predictor_mask_rows, prepare_predictor_layout,
};
use rlx_models::weight_map::WeightMap;
use rlx_runtime::{Device, first_unsupported_op};
use std::collections::HashMap;

pub struct PipelineOut {
    pub enc: Vec<f32>,
    pub pred: Vec<f32>,
    pub pool: Vec<f32>,
}

pub fn tiny_cfg() -> Vjepa2Config {
    Vjepa2Config {
        hidden_size: 64,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        crop_size: 32,
        patch_size: 16,
        tubelet_size: 2,
        frames_per_clip: 4,
        mlp_ratio: 4.0,
        layer_norm_eps: 1e-6,
        in_chans: 3,
        pred_hidden_size: 32,
        pred_num_attention_heads: 4,
        pred_num_hidden_layers: 1,
        pred_mlp_ratio: 4.0,
        pred_num_mask_tokens: 2,
        pred_zero_init_mask_tokens: true,
        num_pooler_layers: 1,
        num_classes: 0,
    }
}

fn synthetic_encoder_weights(cfg: &Vjepa2Config) -> WeightMap {
    let e = cfg.hidden_size;
    let h = cfg.intermediate_size();
    let c = cfg.in_chans;
    let ts = cfg.tubelet_size;
    let ps = cfg.patch_size;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let ramp = |n: usize| {
        (0..n)
            .map(|i| 0.001 + (i as f32) * 0.0001)
            .collect::<Vec<_>>()
    };

    t.insert(
        "encoder.embeddings.patch_embeddings.proj.weight".into(),
        (ramp(e * c * ts * ps * ps), vec![e, c, ts, ps, ps]),
    );
    t.insert(
        "encoder.embeddings.patch_embeddings.proj.bias".into(),
        (ramp(e), vec![e]),
    );
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("encoder.layer.{i}");
        t.insert(format!("{lp}.norm1.weight"), (ramp(e), vec![e]));
        t.insert(format!("{lp}.norm1.bias"), (ramp(e), vec![e]));
        t.insert(format!("{lp}.norm2.weight"), (ramp(e), vec![e]));
        t.insert(format!("{lp}.norm2.bias"), (ramp(e), vec![e]));
        for name in ["query", "key", "value"] {
            t.insert(
                format!("{lp}.attention.{name}.weight"),
                (ramp(e * e), vec![e, e]),
            );
            t.insert(format!("{lp}.attention.{name}.bias"), (ramp(e), vec![e]));
        }
        t.insert(
            format!("{lp}.attention.proj.weight"),
            (ramp(e * e), vec![e, e]),
        );
        t.insert(format!("{lp}.attention.proj.bias"), (ramp(e), vec![e]));
        t.insert(format!("{lp}.mlp.fc1.weight"), (ramp(h * e), vec![h, e]));
        t.insert(format!("{lp}.mlp.fc1.bias"), (ramp(h), vec![h]));
        t.insert(format!("{lp}.mlp.fc2.weight"), (ramp(e * h), vec![e, h]));
        t.insert(format!("{lp}.mlp.fc2.bias"), (ramp(e), vec![e]));
    }
    t.insert("encoder.layernorm.weight".into(), (ramp(e), vec![e]));
    t.insert("encoder.layernorm.bias".into(), (ramp(e), vec![e]));
    WeightMap::from_tensors(t)
}

fn synthetic_predictor_weights(cfg: &Vjepa2Config) -> WeightMap {
    let enc = cfg.hidden_size;
    let pred = cfg.pred_hidden_size;
    let h = cfg.pred_intermediate_size();
    let mut wm = synthetic_encoder_weights(cfg);
    let mut all: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for k in wm.keys().map(|s| s.to_string()).collect::<Vec<_>>() {
        all.insert(k.clone(), wm.take(&k).unwrap());
    }
    let ramp = |n: usize| {
        (0..n)
            .map(|i| 0.002 + (i as f32) * 0.0001)
            .collect::<Vec<_>>()
    };
    all.insert(
        "predictor.embeddings.predictor_embeddings.weight".into(),
        (ramp(enc * pred), vec![pred, enc]),
    );
    all.insert(
        "predictor.embeddings.predictor_embeddings.bias".into(),
        (ramp(pred), vec![pred]),
    );
    all.insert(
        "predictor.embeddings.mask_tokens".into(),
        (
            ramp(cfg.pred_num_mask_tokens * pred),
            vec![cfg.pred_num_mask_tokens, 1, 1, pred],
        ),
    );
    for i in 0..cfg.pred_num_hidden_layers {
        let lp = format!("predictor.layer.{i}");
        for suffix in ["norm1", "norm2"] {
            all.insert(format!("{lp}.{suffix}.weight"), (ramp(pred), vec![pred]));
            all.insert(format!("{lp}.{suffix}.bias"), (ramp(pred), vec![pred]));
        }
        for name in ["query", "key", "value"] {
            all.insert(
                format!("{lp}.attention.{name}.weight"),
                (ramp(pred * pred), vec![pred, pred]),
            );
            all.insert(
                format!("{lp}.attention.{name}.bias"),
                (ramp(pred), vec![pred]),
            );
        }
        all.insert(
            format!("{lp}.attention.proj.weight"),
            (ramp(pred * pred), vec![pred, pred]),
        );
        all.insert(
            format!("{lp}.attention.proj.bias"),
            (ramp(pred), vec![pred]),
        );
        all.insert(
            format!("{lp}.mlp.fc1.weight"),
            (ramp(h * pred), vec![h, pred]),
        );
        all.insert(format!("{lp}.mlp.fc1.bias"), (ramp(h), vec![h]));
        all.insert(
            format!("{lp}.mlp.fc2.weight"),
            (ramp(pred * h), vec![pred, h]),
        );
        all.insert(format!("{lp}.mlp.fc2.bias"), (ramp(pred), vec![pred]));
    }
    all.insert(
        "predictor.layernorm.weight".into(),
        (ramp(pred), vec![pred]),
    );
    all.insert("predictor.layernorm.bias".into(), (ramp(pred), vec![pred]));
    all.insert(
        "predictor.proj.weight".into(),
        (ramp(pred * enc), vec![enc, pred]),
    );
    all.insert("predictor.proj.bias".into(), (ramp(enc), vec![enc]));
    WeightMap::from_tensors(all)
}

fn synthetic_pooler_keys(cfg: &Vjepa2Config) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let e = cfg.hidden_size;
    let h = cfg.intermediate_size();
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let ramp = |n: usize| {
        (0..n)
            .map(|i| 0.003 + (i as f32) * 0.0001)
            .collect::<Vec<_>>()
    };
    t.insert("pooler.query_tokens".into(), (ramp(e), vec![1, 1, e]));
    for i in 0..cfg.num_pooler_layers {
        let lp = format!("pooler.self_attention_layers.{i}");
        for (suffix, dim) in [("layer_norm1", e), ("layer_norm2", e)] {
            t.insert(format!("{lp}.{suffix}.weight"), (ramp(dim), vec![dim]));
            t.insert(format!("{lp}.{suffix}.bias"), (ramp(dim), vec![dim]));
        }
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            t.insert(
                format!("{lp}.self_attn.{name}.weight"),
                (ramp(e * e), vec![e, e]),
            );
            t.insert(format!("{lp}.self_attn.{name}.bias"), (ramp(e), vec![e]));
        }
        t.insert(format!("{lp}.mlp.fc1.weight"), (ramp(h * e), vec![h, e]));
        t.insert(format!("{lp}.mlp.fc1.bias"), (ramp(h), vec![h]));
        t.insert(format!("{lp}.mlp.fc2.weight"), (ramp(e * h), vec![e, h]));
        t.insert(format!("{lp}.mlp.fc2.bias"), (ramp(e), vec![e]));
    }
    let lp = "pooler.cross_attention_layer";
    for (suffix, dim) in [("layer_norm1", e), ("layer_norm2", e)] {
        t.insert(format!("{lp}.{suffix}.weight"), (ramp(dim), vec![dim]));
        t.insert(format!("{lp}.{suffix}.bias"), (ramp(dim), vec![dim]));
    }
    for name in ["q_proj", "k_proj", "v_proj"] {
        t.insert(
            format!("{lp}.cross_attn.{name}.weight"),
            (ramp(e * e), vec![e, e]),
        );
        t.insert(format!("{lp}.cross_attn.{name}.bias"), (ramp(e), vec![e]));
    }
    t.insert(format!("{lp}.mlp.fc1.weight"), (ramp(h * e), vec![h, e]));
    t.insert(format!("{lp}.mlp.fc1.bias"), (ramp(h), vec![h]));
    t.insert(format!("{lp}.mlp.fc2.weight"), (ramp(e * h), vec![e, h]));
    t.insert(format!("{lp}.mlp.fc2.bias"), (ramp(e), vec![e]));
    t
}

pub fn combined_weights(cfg: &Vjepa2Config) -> WeightMap {
    let mut pred_wm = synthetic_predictor_weights(cfg);
    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for k in pred_wm.keys().map(|s| s.to_string()).collect::<Vec<_>>() {
        tensors.insert(k.clone(), pred_wm.take(&k).unwrap());
    }
    for (k, v) in synthetic_pooler_keys(cfg) {
        tensors.insert(k, v);
    }
    WeightMap::from_tensors(tensors)
}

pub fn default_masks(cfg: &Vjepa2Config) -> Vjepa2Masks {
    let seq = cfg.num_patches();
    Vjepa2Masks {
        context: (0..seq / 2).collect(),
        target: (seq / 2..seq).collect(),
        mask_index: 0,
    }
}

pub fn assert_graph_supported(device: Device, graph: &rlx_ir::Graph, label: &str) {
    if let Some((idx, op)) = first_unsupported_op(device, graph) {
        panic!("{label}: {device:?} cannot lower node {idx}: {op:?}");
    }
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

pub fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    let d = max_abs_diff(a, b);
    assert!(
        d <= tol,
        "{label}: max abs diff {d} > {tol} (len {})",
        a.len()
    );
    assert!(a.iter().all(|v| v.is_finite()), "{label}: non-finite ref");
    assert!(b.iter().all(|v| v.is_finite()), "{label}: non-finite out");
}

/// Run encoder → predictor → pooler through compiled IR on `device`.
pub fn run_compiled_pipeline(device: Device) -> PipelineOut {
    let cfg = tiny_cfg();
    let batch = 1;
    let mut wm = combined_weights(&cfg);
    let model = extract_model_weights(&mut wm, &cfg).expect("weights");
    let masks = default_masks(&cfg);
    let seq = cfg.num_patches();

    let video: Vec<f32> = (0..3 * cfg.frames_per_clip * cfg.crop_size * cfg.crop_size)
        .map(|i| 0.01 + (i as f32) * 0.00001)
        .collect();
    let patch = conv3d_patch_embed(
        &model.encoder.patch,
        &video,
        cfg.frames_per_clip,
        cfg.crop_size,
        cfg.crop_size,
    )
    .expect("patch embed");

    let (enc_g, enc_p, _) =
        build_vjepa2_encoder_graph_sized(&cfg, &model.encoder, batch).expect("enc graph");
    assert_graph_supported(device, &enc_g, "encoder");
    let mut enc_c =
        rlx_models::flow_bridge::compile_graph_encoder(device, enc_g).expect("enc compile");
    Vjepa2GraphParams::from_f32(enc_p).load(&mut enc_c);
    let enc = enc_c
        .run(&[("hidden", patch.as_slice())])
        .into_iter()
        .next()
        .expect("enc out");

    let pred_w = model.predictor.as_ref().expect("predictor");
    let layout = prepare_predictor_layout(&cfg, &masks, batch).expect("layout");
    let mask_rows = predictor_mask_rows(pred_w, &cfg, &masks, batch);
    let (pred_g, pred_params) =
        build_vjepa2_predictor_graph_sized(&cfg, pred_w, &layout, &mask_rows, batch)
            .expect("pred graph");
    assert_graph_supported(device, &pred_g, "predictor");
    let mut pred_c =
        rlx_models::flow_bridge::compile_graph_encoder(device, pred_g).expect("pred compile");
    pred_params.load(&mut pred_c);
    let pred = pred_c
        .run(&[("encoder", enc.as_slice())])
        .into_iter()
        .next()
        .expect("pred out");

    let pool_w = model.pooler.as_ref().expect("pooler");
    let (pool_g, pool_params) =
        build_vjepa2_pooler_graph_sized(&cfg, pool_w, batch).expect("pool graph");
    assert_graph_supported(device, &pool_g, "pooler");
    let mut pool_c =
        rlx_models::flow_bridge::compile_graph_encoder(device, pool_g).expect("pool compile");
    pool_params.load(&mut pool_c);
    let pool = pool_c
        .run(&[("encoder", enc.as_slice())])
        .into_iter()
        .next()
        .expect("pool out");

    assert_eq!(enc.len(), seq * cfg.hidden_size);
    assert_eq!(pred.len(), masks.target.len() * cfg.hidden_size);
    assert_eq!(pool.len(), cfg.hidden_size);

    PipelineOut { enc, pred, pool }
}

pub fn assert_pipeline_close(cpu: &PipelineOut, other: &PipelineOut, device: Device, tol: f32) {
    assert_close(&cpu.enc, &other.enc, tol, "encoder");
    assert_close(&cpu.pred, &other.pred, tol, "predictor");
    assert_close(&cpu.pool, &other.pool, tol, "pooler");
    let _ = device;
}
