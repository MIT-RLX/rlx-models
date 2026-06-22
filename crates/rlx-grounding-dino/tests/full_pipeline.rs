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

//! Full Grounding DINO pipeline smoke test on a tiny synthetic checkpoint:
//! preprocess → Swin → neck → text BERT → enhancer → query selection →
//! decoder → postprocess. Validates the whole graph composes with consistent
//! shapes and produces finite boxes/scores.

use rlx_core::weight_map::WeightMap;
use rlx_grounding_dino::GroundingDino;
use rlx_grounding_dino::config::{GroundingDinoConfig, SwinConfig, TextConfig};
use rlx_grounding_dino::tokenizer::text_tokens_from_ids;
use std::collections::HashMap;

type Tensors = HashMap<String, (Vec<f32>, Vec<usize>)>;

fn det(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 31 + seed * 17) % 23) as f32 - 11.0) * 0.03)
        .collect()
}

struct Builder {
    t: Tensors,
}
impl Builder {
    fn new() -> Self {
        Self { t: HashMap::new() }
    }
    fn w(&mut self, k: &str, shape: Vec<usize>, seed: usize) {
        let n: usize = shape.iter().product();
        self.t.insert(k.to_string(), (det(n, seed), shape));
    }
    fn ones(&mut self, k: &str, n: usize) {
        self.t.insert(k.to_string(), (vec![1.0; n], vec![n]));
    }
    fn zeros(&mut self, k: &str, n: usize) {
        self.t.insert(k.to_string(), (vec![0.0; n], vec![n]));
    }
    fn idx(&mut self, k: &str, shape: Vec<usize>, modulo: usize) {
        let n: usize = shape.iter().product();
        self.t.insert(
            k.to_string(),
            ((0..n).map(|i| (i % modulo) as f32).collect(), shape),
        );
    }
}

fn tiny_cfg() -> GroundingDinoConfig {
    GroundingDinoConfig {
        backbone_config: SwinConfig {
            embed_dim: 2,
            depths: vec![1, 1],
            num_heads: vec![1, 2],
            window_size: 2,
            image_size: 8,
            patch_size: 2,
            mlp_ratio: 4.0,
            num_channels: 3,
            out_indices: vec![1, 2],
            layer_norm_eps: 1e-5,
            qkv_bias: true,
        },
        text_config: TextConfig {
            vocab_size: 1100,
            hidden_size: 6,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 12,
            max_position_embeddings: 64,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".into(),
        },
        d_model: 8,
        encoder_layers: 1,
        decoder_layers: 1,
        encoder_attention_heads: 2,
        decoder_attention_heads: 2,
        encoder_ffn_dim: 16,
        decoder_ffn_dim: 16,
        num_feature_levels: 2,
        num_queries: 5,
        max_text_len: 16,
        encoder_n_points: 2,
        decoder_n_points: 2,
        activation_function: "relu".into(),
        position_embedding_type: "sine".into(),
        positional_embedding_temperature: 20.0,
    }
}

fn build_checkpoint(cfg: &GroundingDinoConfig) -> WeightMap {
    let mut b = Builder::new();
    let d = cfg.d_model;
    let bi = cfg.encoder_ffn_dim / 2;

    // ---- Swin backbone ----
    let sp = "model.backbone.conv_encoder.model.";
    let sw = &cfg.backbone_config;
    let ps = sw.patch_size;
    let ws = sw.window_size;
    let ws2 = ws * ws;
    let rel_rows = (2 * ws - 1) * (2 * ws - 1);
    b.w(
        &format!("{sp}embeddings.patch_embeddings.projection.weight"),
        vec![sw.embed_dim, 3, ps, ps],
        1,
    );
    b.zeros(
        &format!("{sp}embeddings.patch_embeddings.projection.bias"),
        sw.embed_dim,
    );
    b.ones(&format!("{sp}embeddings.norm.weight"), sw.embed_dim);
    b.zeros(&format!("{sp}embeddings.norm.bias"), sw.embed_dim);
    for s in 0..sw.num_stages() {
        let dim = sw.stage_dim(s);
        let heads = sw.num_heads[s];
        let inter = dim * 4;
        for bl in 0..sw.depths[s] {
            let bp = format!("{sp}encoder.layers.{s}.blocks.{bl}.");
            b.ones(&format!("{bp}layernorm_before.weight"), dim);
            b.zeros(&format!("{bp}layernorm_before.bias"), dim);
            for (nm, seed) in [("query", 10), ("key", 20), ("value", 30)] {
                b.w(
                    &format!("{bp}attention.self.{nm}.weight"),
                    vec![dim, dim],
                    seed + s,
                );
                b.zeros(&format!("{bp}attention.self.{nm}.bias"), dim);
            }
            b.w(
                &format!("{bp}attention.self.relative_position_bias_table"),
                vec![rel_rows, heads],
                40,
            );
            b.idx(
                &format!("{bp}attention.self.relative_position_index"),
                vec![ws2, ws2],
                rel_rows,
            );
            b.w(
                &format!("{bp}attention.output.dense.weight"),
                vec![dim, dim],
                50,
            );
            b.zeros(&format!("{bp}attention.output.dense.bias"), dim);
            b.ones(&format!("{bp}layernorm_after.weight"), dim);
            b.zeros(&format!("{bp}layernorm_after.bias"), dim);
            b.w(
                &format!("{bp}intermediate.dense.weight"),
                vec![inter, dim],
                60,
            );
            b.zeros(&format!("{bp}intermediate.dense.bias"), inter);
            b.w(&format!("{bp}output.dense.weight"), vec![dim, inter], 70);
            b.zeros(&format!("{bp}output.dense.bias"), dim);
        }
        if s < sw.num_stages() - 1 {
            let dp = format!("{sp}encoder.layers.{s}.downsample.");
            b.ones(&format!("{dp}norm.weight"), 4 * dim);
            b.zeros(&format!("{dp}norm.bias"), 4 * dim);
            b.w(&format!("{dp}reduction.weight"), vec![2 * dim, 4 * dim], 80);
        }
    }
    for &i in &sw.out_indices {
        let dim = sw.stage_dim(i - 1);
        b.ones(&format!("{sp}hidden_states_norms.stage{i}.weight"), dim);
        b.zeros(&format!("{sp}hidden_states_norms.stage{i}.bias"), dim);
    }

    // ---- Neck ----
    let in_dims = sw.out_channels(); // [2, 4]
    for i in 0..cfg.num_feature_levels {
        let lp = format!("model.input_proj_vision.{i}.");
        let in_c = in_dims[i.min(in_dims.len() - 1)];
        b.w(&format!("{lp}0.weight"), vec![d, in_c, 1, 1], 90 + i);
        b.zeros(&format!("{lp}0.bias"), d);
        b.ones(&format!("{lp}1.weight"), d);
        b.zeros(&format!("{lp}1.bias"), d);
    }
    b.w("model.level_embed", vec![cfg.num_feature_levels, d], 95);

    // ---- Text backbone (BERT) ----
    let tp = "model.text_backbone.";
    let tc = &cfg.text_config;
    let th = tc.hidden_size;
    b.w(
        &format!("{tp}embeddings.word_embeddings.weight"),
        vec![tc.vocab_size, th],
        100,
    );
    b.w(
        &format!("{tp}embeddings.position_embeddings.weight"),
        vec![tc.max_position_embeddings, th],
        101,
    );
    b.w(
        &format!("{tp}embeddings.token_type_embeddings.weight"),
        vec![tc.type_vocab_size, th],
        102,
    );
    b.ones(&format!("{tp}embeddings.LayerNorm.weight"), th);
    b.zeros(&format!("{tp}embeddings.LayerNorm.bias"), th);
    for i in 0..tc.num_hidden_layers {
        let lp = format!("{tp}encoder.layer.{i}.");
        for nm in ["query", "key", "value"] {
            b.w(
                &format!("{lp}attention.self.{nm}.weight"),
                vec![th, th],
                110,
            );
            b.zeros(&format!("{lp}attention.self.{nm}.bias"), th);
        }
        b.w(
            &format!("{lp}attention.output.dense.weight"),
            vec![th, th],
            111,
        );
        b.zeros(&format!("{lp}attention.output.dense.bias"), th);
        b.ones(&format!("{lp}attention.output.LayerNorm.weight"), th);
        b.zeros(&format!("{lp}attention.output.LayerNorm.bias"), th);
        b.w(
            &format!("{lp}intermediate.dense.weight"),
            vec![tc.intermediate_size, th],
            112,
        );
        b.zeros(
            &format!("{lp}intermediate.dense.bias"),
            tc.intermediate_size,
        );
        b.w(
            &format!("{lp}output.dense.weight"),
            vec![th, tc.intermediate_size],
            113,
        );
        b.zeros(&format!("{lp}output.dense.bias"), th);
        b.ones(&format!("{lp}output.LayerNorm.weight"), th);
        b.zeros(&format!("{lp}output.LayerNorm.bias"), th);
    }
    b.w("model.text_projection.weight", vec![d, th], 120);
    b.zeros("model.text_projection.bias", d);

    // ---- Enhancer ----
    let nh = cfg.encoder_attention_heads;
    let nl = cfg.num_feature_levels;
    let npts = cfg.encoder_n_points;
    for i in 0..cfg.encoder_layers {
        let fp = format!("model.encoder.layers.{i}.fusion_layer.");
        for nm in [
            "vision_proj",
            "text_proj",
            "values_vision_proj",
            "values_text_proj",
        ] {
            b.w(&format!("{fp}attn.{nm}.weight"), vec![bi, d], 130);
            b.zeros(&format!("{fp}attn.{nm}.bias"), bi);
        }
        for nm in ["out_vision_proj", "out_text_proj"] {
            b.w(&format!("{fp}attn.{nm}.weight"), vec![d, bi], 131);
            b.zeros(&format!("{fp}attn.{nm}.bias"), d);
        }
        b.ones(&format!("{fp}layer_norm_vision.weight"), d);
        b.zeros(&format!("{fp}layer_norm_vision.bias"), d);
        b.ones(&format!("{fp}layer_norm_text.weight"), d);
        b.zeros(&format!("{fp}layer_norm_text.bias"), d);
        b.zeros(&format!("{fp}vision_param"), d); // layerscale 0 → stable
        b.zeros(&format!("{fp}text_param"), d);

        let te = format!("model.encoder.layers.{i}.text_enhancer_layer.");
        for nm in ["query", "key", "value", "out_proj"] {
            b.w(&format!("{te}self_attn.{nm}.weight"), vec![d, d], 140);
            b.zeros(&format!("{te}self_attn.{nm}.bias"), d);
        }
        b.w(&format!("{te}fc1.weight"), vec![bi, d], 141);
        b.zeros(&format!("{te}fc1.bias"), bi);
        b.w(&format!("{te}fc2.weight"), vec![d, bi], 142);
        b.zeros(&format!("{te}fc2.bias"), d);
        b.ones(&format!("{te}layer_norm_before.weight"), d);
        b.zeros(&format!("{te}layer_norm_before.bias"), d);
        b.ones(&format!("{te}layer_norm_after.weight"), d);
        b.zeros(&format!("{te}layer_norm_after.bias"), d);

        let dl = format!("model.encoder.layers.{i}.deformable_layer.");
        b.w(
            &format!("{dl}self_attn.sampling_offsets.weight"),
            vec![nh * nl * npts * 2, d],
            150,
        );
        b.zeros(
            &format!("{dl}self_attn.sampling_offsets.bias"),
            nh * nl * npts * 2,
        );
        b.w(
            &format!("{dl}self_attn.attention_weights.weight"),
            vec![nh * nl * npts, d],
            151,
        );
        b.zeros(
            &format!("{dl}self_attn.attention_weights.bias"),
            nh * nl * npts,
        );
        b.w(&format!("{dl}self_attn.value_proj.weight"), vec![d, d], 152);
        b.zeros(&format!("{dl}self_attn.value_proj.bias"), d);
        b.w(
            &format!("{dl}self_attn.output_proj.weight"),
            vec![d, d],
            153,
        );
        b.zeros(&format!("{dl}self_attn.output_proj.bias"), d);
        b.ones(&format!("{dl}self_attn_layer_norm.weight"), d);
        b.zeros(&format!("{dl}self_attn_layer_norm.bias"), d);
        b.w(
            &format!("{dl}fc1.weight"),
            vec![cfg.encoder_ffn_dim, d],
            154,
        );
        b.zeros(&format!("{dl}fc1.bias"), cfg.encoder_ffn_dim);
        b.w(
            &format!("{dl}fc2.weight"),
            vec![d, cfg.encoder_ffn_dim],
            155,
        );
        b.zeros(&format!("{dl}fc2.bias"), d);
        b.ones(&format!("{dl}final_layer_norm.weight"), d);
        b.zeros(&format!("{dl}final_layer_norm.bias"), d);
    }

    // ---- Query selection ----
    b.w("model.enc_output.weight", vec![d, d], 160);
    b.zeros("model.enc_output.bias", d);
    b.ones("model.enc_output_norm.weight", d);
    b.zeros("model.enc_output_norm.bias", d);
    b.w(
        "model.encoder_output_bbox_embed.layers.0.weight",
        vec![d, d],
        161,
    );
    b.zeros("model.encoder_output_bbox_embed.layers.0.bias", d);
    b.w(
        "model.encoder_output_bbox_embed.layers.1.weight",
        vec![d, d],
        162,
    );
    b.zeros("model.encoder_output_bbox_embed.layers.1.bias", d);
    b.w(
        "model.encoder_output_bbox_embed.layers.2.weight",
        vec![4, d],
        163,
    );
    b.zeros("model.encoder_output_bbox_embed.layers.2.bias", 4);

    // ---- Decoder ----
    for i in 0..cfg.decoder_layers {
        let p = format!("model.decoder.layers.{i}.");
        for nm in ["query", "key", "value", "out_proj"] {
            b.w(&format!("{p}self_attn.{nm}.weight"), vec![d, d], 170);
            b.zeros(&format!("{p}self_attn.{nm}.bias"), d);
            b.w(
                &format!("{p}encoder_attn_text.{nm}.weight"),
                vec![d, d],
                171,
            );
            b.zeros(&format!("{p}encoder_attn_text.{nm}.bias"), d);
        }
        b.ones(&format!("{p}self_attn_layer_norm.weight"), d);
        b.zeros(&format!("{p}self_attn_layer_norm.bias"), d);
        b.ones(&format!("{p}encoder_attn_text_layer_norm.weight"), d);
        b.zeros(&format!("{p}encoder_attn_text_layer_norm.bias"), d);
        b.w(
            &format!("{p}encoder_attn.sampling_offsets.weight"),
            vec![nh * nl * npts * 2, d],
            172,
        );
        b.zeros(
            &format!("{p}encoder_attn.sampling_offsets.bias"),
            nh * nl * npts * 2,
        );
        b.w(
            &format!("{p}encoder_attn.attention_weights.weight"),
            vec![nh * nl * npts, d],
            173,
        );
        b.zeros(
            &format!("{p}encoder_attn.attention_weights.bias"),
            nh * nl * npts,
        );
        b.w(
            &format!("{p}encoder_attn.value_proj.weight"),
            vec![d, d],
            174,
        );
        b.zeros(&format!("{p}encoder_attn.value_proj.bias"), d);
        b.w(
            &format!("{p}encoder_attn.output_proj.weight"),
            vec![d, d],
            175,
        );
        b.zeros(&format!("{p}encoder_attn.output_proj.bias"), d);
        b.ones(&format!("{p}encoder_attn_layer_norm.weight"), d);
        b.zeros(&format!("{p}encoder_attn_layer_norm.bias"), d);
        b.w(&format!("{p}fc1.weight"), vec![cfg.decoder_ffn_dim, d], 176);
        b.zeros(&format!("{p}fc1.bias"), cfg.decoder_ffn_dim);
        b.w(&format!("{p}fc2.weight"), vec![d, cfg.decoder_ffn_dim], 177);
        b.zeros(&format!("{p}fc2.bias"), d);
        b.ones(&format!("{p}final_layer_norm.weight"), d);
        b.zeros(&format!("{p}final_layer_norm.bias"), d);
    }
    let npf = d / 2;
    b.w(
        "model.decoder.reference_points_head.layers.0.weight",
        vec![d, 4 * npf],
        180,
    );
    b.zeros("model.decoder.reference_points_head.layers.0.bias", d);
    b.w(
        "model.decoder.reference_points_head.layers.1.weight",
        vec![d, d],
        181,
    );
    b.zeros("model.decoder.reference_points_head.layers.1.bias", d);
    b.w(
        "model.decoder.bbox_embed.0.layers.0.weight",
        vec![d, d],
        182,
    );
    b.zeros("model.decoder.bbox_embed.0.layers.0.bias", d);
    b.w(
        "model.decoder.bbox_embed.0.layers.1.weight",
        vec![d, d],
        183,
    );
    b.zeros("model.decoder.bbox_embed.0.layers.1.bias", d);
    b.w(
        "model.decoder.bbox_embed.0.layers.2.weight",
        vec![4, d],
        184,
    );
    b.zeros("model.decoder.bbox_embed.0.layers.2.bias", 4);
    b.ones("model.decoder.layer_norm.weight", d);
    b.zeros("model.decoder.layer_norm.bias", d);

    WeightMap::from_tensors(b.t)
}

#[test]
fn full_pipeline_runs_end_to_end() {
    let cfg = tiny_cfg();
    let wm = build_checkpoint(&cfg);
    let model = GroundingDino::from_weights(&wm, cfg).expect("build model");

    // 24x24 image → patch(2) → 12x12 grid → stages → maps.
    let (h, w) = (24, 24);
    let rgb: Vec<u8> = (0..h * w * 3).map(|i| (i % 251) as u8).collect();
    // [CLS] a b . c . [SEP]
    let tokens = text_tokens_from_ids(vec![101, 500, 501, 1012, 502, 1012, 102]);

    // Low thresholds so we get some detections out of the synthetic model.
    let dets = model.detect(&rgb, h, w, &tokens, 0.0, 0.0);

    // num_queries = 5 → at most 5 detections; with threshold 0 all pass.
    assert!(dets.len() <= 5);
    assert!(!dets.is_empty());
    for det in &dets {
        assert!(det.score.is_finite() && (0.0..=1.0).contains(&det.score));
        assert!(det.bbox.iter().all(|v| v.is_finite()));
        // scores sorted descending
    }
    for win in dets.windows(2) {
        assert!(win[0].score >= win[1].score);
    }
}

#[test]
fn on_device_decoder_matches_native() {
    use rlx_grounding_dino::Device;
    use rlx_grounding_dino::preprocess::preprocess_rgb;

    let cfg = tiny_cfg();
    let wm = build_checkpoint(&cfg);
    let model = GroundingDino::from_weights(&wm, cfg).expect("build model");

    let (h, w) = (24, 24);
    let rgb: Vec<u8> = (0..h * w * 3).map(|i| (i % 251) as u8).collect();
    let tokens = text_tokens_from_ids(vec![101, 500, 501, 1012, 502, 1012, 102]);
    let pre = preprocess_rgb(&rgb, h, w);

    // Native vs decoder-on-device (CPU IR) must agree.
    let native = model.detect_preprocessed(
        &pre.pixel_values,
        pre.height,
        pre.width,
        h,
        w,
        &tokens,
        0.0,
        0.0,
    );
    let ondev = model
        .detect_preprocessed_on(
            &pre.pixel_values,
            pre.height,
            pre.width,
            h,
            w,
            &tokens,
            0.0,
            0.0,
            Device::Cpu,
        )
        .expect("on-device decode");

    assert_eq!(native.len(), ondev.len());
    for (n, o) in native.iter().zip(&ondev) {
        assert!(
            (n.score - o.score).abs() < 1e-4,
            "score {} vs {}",
            n.score,
            o.score
        );
        for k in 0..4 {
            assert!(
                (n.bbox[k] - o.bbox[k]).abs() < 1e-3,
                "bbox[{k}] {} vs {}",
                n.bbox[k],
                o.bbox[k]
            );
        }
    }
}
