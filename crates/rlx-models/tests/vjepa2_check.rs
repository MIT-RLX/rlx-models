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

//! V-JEPA2 ViT-G basic tests.

mod compile_support;

use rlx_models::weight_map::WeightMap;
use rlx_models::{
    Vjepa2Config, Vjepa2Masks, build_vjepa2_encoder_graph_sized, encode_video_native,
    extract_encoder_weights, extract_model_weights, predict_native,
};
use std::collections::HashMap;

fn tiny_cfg() -> Vjepa2Config {
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
    let z = |n: usize| vec![0.0f32; n];
    t.insert(
        "encoder.embeddings.patch_embeddings.proj.weight".into(),
        (z(e * c * ts * ps * ps), vec![e, c, ts, ps, ps]),
    );
    t.insert(
        "encoder.embeddings.patch_embeddings.proj.bias".into(),
        (z(e), vec![e]),
    );
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("encoder.layer.{i}");
        t.insert(format!("{lp}.norm1.weight"), (z(e), vec![e]));
        t.insert(format!("{lp}.norm1.bias"), (z(e), vec![e]));
        t.insert(format!("{lp}.norm2.weight"), (z(e), vec![e]));
        t.insert(format!("{lp}.norm2.bias"), (z(e), vec![e]));
        for name in ["query", "key", "value"] {
            t.insert(
                format!("{lp}.attention.{name}.weight"),
                (z(e * e), vec![e, e]),
            );
            t.insert(format!("{lp}.attention.{name}.bias"), (z(e), vec![e]));
        }
        t.insert(
            format!("{lp}.attention.proj.weight"),
            (z(e * e), vec![e, e]),
        );
        t.insert(format!("{lp}.attention.proj.bias"), (z(e), vec![e]));
        t.insert(format!("{lp}.mlp.fc1.weight"), (z(h * e), vec![h, e]));
        t.insert(format!("{lp}.mlp.fc1.bias"), (z(h), vec![h]));
        t.insert(format!("{lp}.mlp.fc2.weight"), (z(e * h), vec![e, h]));
        t.insert(format!("{lp}.mlp.fc2.bias"), (z(e), vec![e]));
    }
    t.insert("encoder.layernorm.weight".into(), (z(e), vec![e]));
    t.insert("encoder.layernorm.bias".into(), (z(e), vec![e]));
    WeightMap::from_tensors(t)
}

#[test]
fn vit_g_384_config_dims() {
    let cfg = Vjepa2Config::vit_g_384();
    assert_eq!(cfg.num_patches(), 18432);
    assert_eq!(cfg.pred_num_hidden_layers, 12);
}

#[test]
fn tiny_encoder_forward_check() {
    let cfg = tiny_cfg();
    let mut wm = synthetic_encoder_weights(&cfg);
    let enc = extract_encoder_weights(&mut wm, &cfg).unwrap();
    let video = vec![0.0f32; 3 * cfg.frames_per_clip * cfg.crop_size * cfg.crop_size];
    let out = encode_video_native(&enc, &cfg, &video, 1).unwrap();
    assert_eq!(out.tokens.len(), cfg.num_patches() * cfg.hidden_size);
}

#[test]
fn encoder_graph_builds() {
    let cfg = tiny_cfg();
    let mut wm = synthetic_encoder_weights(&cfg);
    let enc = extract_encoder_weights(&mut wm, &cfg).unwrap();
    let (g, _, _) = build_vjepa2_encoder_graph_sized(&cfg, &enc, 1).unwrap();
    assert_eq!(g.outputs.len(), 1);
}

#[test]
fn predictor_check() {
    let cfg = tiny_cfg();
    let mut wm = synthetic_encoder_weights(&cfg);
    let enc_dim = cfg.hidden_size;
    let pred = cfg.pred_hidden_size;
    let h = cfg.pred_intermediate_size();
    let keys: Vec<String> = wm.keys().map(|s| s.to_string()).collect();
    let mut all: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for k in keys {
        all.insert(k.clone(), wm.take(&k).unwrap());
    }
    all.insert(
        "predictor.embeddings.predictor_embeddings.weight".into(),
        (vec![0f32; enc_dim * pred], vec![pred, enc_dim]),
    );
    all.insert(
        "predictor.embeddings.predictor_embeddings.bias".into(),
        (vec![0f32; pred], vec![pred]),
    );
    all.insert(
        "predictor.embeddings.mask_tokens".into(),
        (
            vec![0f32; cfg.pred_num_mask_tokens * pred],
            vec![cfg.pred_num_mask_tokens, 1, 1, pred],
        ),
    );
    let lp = "predictor.layer.0";
    for suffix in ["norm1", "norm2"] {
        all.insert(
            format!("{lp}.{suffix}.weight"),
            (vec![0f32; pred], vec![pred]),
        );
        all.insert(
            format!("{lp}.{suffix}.bias"),
            (vec![0f32; pred], vec![pred]),
        );
    }
    for name in ["query", "key", "value"] {
        all.insert(
            format!("{lp}.attention.{name}.weight"),
            (vec![0f32; pred * pred], vec![pred, pred]),
        );
        all.insert(
            format!("{lp}.attention.{name}.bias"),
            (vec![0f32; pred], vec![pred]),
        );
    }
    all.insert(
        format!("{lp}.attention.proj.weight"),
        (vec![0f32; pred * pred], vec![pred, pred]),
    );
    all.insert(
        format!("{lp}.attention.proj.bias"),
        (vec![0f32; pred], vec![pred]),
    );
    all.insert(
        format!("{lp}.mlp.fc1.weight"),
        (vec![0f32; h * pred], vec![h, pred]),
    );
    all.insert(format!("{lp}.mlp.fc1.bias"), (vec![0f32; h], vec![h]));
    all.insert(
        format!("{lp}.mlp.fc2.weight"),
        (vec![0f32; pred * h], vec![pred, h]),
    );
    all.insert(format!("{lp}.mlp.fc2.bias"), (vec![0f32; pred], vec![pred]));
    all.insert(
        "predictor.layernorm.weight".into(),
        (vec![0f32; pred], vec![pred]),
    );
    all.insert(
        "predictor.layernorm.bias".into(),
        (vec![0f32; pred], vec![pred]),
    );
    all.insert(
        "predictor.proj.weight".into(),
        (vec![0f32; pred * enc_dim], vec![enc_dim, pred]),
    );
    all.insert(
        "predictor.proj.bias".into(),
        (vec![0f32; enc_dim], vec![enc_dim]),
    );
    let mut wm2 = WeightMap::from_tensors(all);
    let model = extract_model_weights(&mut wm2, &cfg).unwrap();
    let video = vec![0.0f32; 3 * cfg.frames_per_clip * cfg.crop_size * cfg.crop_size];
    let enc = encode_video_native(&model.encoder, &cfg, &video, 1).unwrap();
    let seq = cfg.num_patches();
    let masks = Vjepa2Masks {
        context: (0..seq / 2).collect(),
        target: (seq / 2..seq).collect(),
        mask_index: 0,
    };
    let pred_w = model.predictor.as_ref().unwrap();
    let out = predict_native(&enc.tokens, pred_w, &cfg, 1, seq, &masks).unwrap();
    assert_eq!(out.tokens.len(), masks.target.len() * enc_dim);
}
