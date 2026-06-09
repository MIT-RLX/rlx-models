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

//! V-JEPA2 — Meta's self-supervised video ViT (ViT-G and friends).
//!
//! Ships encoder, predictor, attentive pooler, and IR graph paths for all
//! three when `.device(...)` is set on the runner.

pub mod builder;
pub mod cli;
pub mod config;
pub mod encoder;
pub mod flow;
pub mod layers;
pub mod pooler;
pub mod predictor;
pub mod preprocess;
pub mod rope;
pub mod runner;
pub mod weights;

pub use builder::{
    Vjepa2GraphParams, Vjepa2GraphPreprocess, build_vjepa2_encoder_graph_sized,
    build_vjepa2_encoder_hir_sized, build_vjepa2_pooler_graph_sized, build_vjepa2_pooler_hir_sized,
    build_vjepa2_predictor_graph_sized, build_vjepa2_predictor_hir_sized, compile_vjepa2_encoder,
};
pub use config::{IMAGENET_MEAN, IMAGENET_STD, Vjepa2Config, rope_segment_dims};
pub use encoder::{Vjepa2EncoderOutput, encode_video_native, encode_video_native_ext};
pub use flow::{Vjepa2EncoderBuilt, Vjepa2EncoderFlow, Vjepa2PoolerFlow, Vjepa2PredictorFlow};
pub use pooler::{Vjepa2PoolerOutput, pool_native};
pub use predictor::{
    Vjepa2Masks, Vjepa2PredictorLayout, Vjepa2PredictorOutput, predict_native, predictor_mask_rows,
    prepare_predictor_layout,
};
pub use preprocess::{
    Vjepa2PatchEmbedWeights, conv3d_patch_embed, extract_patch_embed_weights, normalize_video_hwc,
};
pub use runner::{
    Vjepa2Output, Vjepa2PoolOutput, Vjepa2PredictOutput, Vjepa2Runner, Vjepa2RunnerBuilder,
};
pub use weights::{
    Vjepa2BlockWeights, Vjepa2EncoderWeights, Vjepa2ModelWeights, Vjepa2PoolerWeights,
    Vjepa2PredictorWeights, extract_encoder_weights, extract_model_weights, extract_pooler_weights,
    extract_predictor_weights,
};

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::weight_map::WeightMap;
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

    fn synthetic_predictor_weights(cfg: &Vjepa2Config) -> WeightMap {
        let enc = cfg.hidden_size;
        let pred = cfg.pred_hidden_size;
        let h = cfg.pred_intermediate_size();
        let mut wm = synthetic_encoder_weights(cfg);
        let mut all: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        for k in wm.keys().map(|s| s.to_string()).collect::<Vec<_>>() {
            all.insert(k.clone(), wm.take(&k).unwrap());
        }
        all.insert(
            "predictor.embeddings.predictor_embeddings.weight".into(),
            (vec![0f32; enc * pred], vec![pred, enc]),
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
        for i in 0..cfg.pred_num_hidden_layers {
            let lp = format!("predictor.layer.{i}");
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
        }
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
            (vec![0f32; pred * enc], vec![enc, pred]),
        );
        all.insert("predictor.proj.bias".into(), (vec![0f32; enc], vec![enc]));
        WeightMap::from_tensors(all)
    }

    #[test]
    fn vit_g_384_config_dims() {
        let cfg = Vjepa2Config::vit_g_384();
        assert_eq!(cfg.hidden_size, 1408);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.num_patches(), 32 * 24 * 24);
    }

    #[test]
    fn encode_synthetic_video_runs() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_encoder_weights(&cfg);
        let enc = extract_encoder_weights(&mut wm, &cfg).unwrap();
        let video = vec![0.0f32; 3 * cfg.frames_per_clip * cfg.crop_size * cfg.crop_size];
        let out = encode_video_native(&enc, &cfg, &video, 1).unwrap();
        assert_eq!(out.tokens.len(), cfg.num_patches() * cfg.hidden_size);
    }

    #[test]
    fn predictor_forward_runs() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_predictor_weights(&cfg);
        let model = extract_model_weights(&mut wm, &cfg).unwrap();
        let pred = model.predictor.as_ref().unwrap();
        let video = vec![0.0f32; 3 * cfg.frames_per_clip * cfg.crop_size * cfg.crop_size];
        let enc = encode_video_native(&model.encoder, &cfg, &video, 1).unwrap();
        let seq = cfg.num_patches();
        let masks = Vjepa2Masks {
            context: (0..seq / 2).collect(),
            target: (seq / 2..seq).collect(),
            mask_index: 0,
        };
        let out = predict_native(&enc.tokens, pred, &cfg, 1, seq, &masks).unwrap();
        assert_eq!(out.tokens.len(), masks.target.len() * cfg.hidden_size);
    }

    #[test]
    fn encoder_graph_builds() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_encoder_weights(&cfg);
        let enc = extract_encoder_weights(&mut wm, &cfg).unwrap();
        let (g, _p, _pre) = build_vjepa2_encoder_graph_sized(&cfg, &enc, 1).unwrap();
        assert_eq!(g.outputs.len(), 1);
    }

    fn synthetic_pooler_weights(cfg: &Vjepa2Config) -> WeightMap {
        let e = cfg.hidden_size;
        let h = cfg.intermediate_size();
        let mut wm = synthetic_encoder_weights(cfg);
        let mut all: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        for k in wm.keys().map(|s| s.to_string()).collect::<Vec<_>>() {
            all.insert(k.clone(), wm.take(&k).unwrap());
        }
        all.insert("pooler.query_tokens".into(), (vec![0f32; e], vec![1, 1, e]));
        for i in 0..cfg.num_pooler_layers {
            let lp = format!("pooler.self_attention_layers.{i}");
            for (suffix, dim) in [("layer_norm1", e), ("layer_norm2", e)] {
                all.insert(
                    format!("{lp}.{suffix}.weight"),
                    (vec![0f32; dim], vec![dim]),
                );
                all.insert(format!("{lp}.{suffix}.bias"), (vec![0f32; dim], vec![dim]));
            }
            for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                all.insert(
                    format!("{lp}.self_attn.{name}.weight"),
                    (vec![0f32; e * e], vec![e, e]),
                );
                all.insert(
                    format!("{lp}.self_attn.{name}.bias"),
                    (vec![0f32; e], vec![e]),
                );
            }
            all.insert(
                format!("{lp}.mlp.fc1.weight"),
                (vec![0f32; h * e], vec![h, e]),
            );
            all.insert(format!("{lp}.mlp.fc1.bias"), (vec![0f32; h], vec![h]));
            all.insert(
                format!("{lp}.mlp.fc2.weight"),
                (vec![0f32; e * h], vec![e, h]),
            );
            all.insert(format!("{lp}.mlp.fc2.bias"), (vec![0f32; e], vec![e]));
        }
        let lp = "pooler.cross_attention_layer";
        for (suffix, dim) in [("layer_norm1", e), ("layer_norm2", e)] {
            all.insert(
                format!("{lp}.{suffix}.weight"),
                (vec![0f32; dim], vec![dim]),
            );
            all.insert(format!("{lp}.{suffix}.bias"), (vec![0f32; dim], vec![dim]));
        }
        for name in ["q_proj", "k_proj", "v_proj"] {
            all.insert(
                format!("{lp}.cross_attn.{name}.weight"),
                (vec![0f32; e * e], vec![e, e]),
            );
            all.insert(
                format!("{lp}.cross_attn.{name}.bias"),
                (vec![0f32; e], vec![e]),
            );
        }
        all.insert(
            format!("{lp}.mlp.fc1.weight"),
            (vec![0f32; h * e], vec![h, e]),
        );
        all.insert(format!("{lp}.mlp.fc1.bias"), (vec![0f32; h], vec![h]));
        all.insert(
            format!("{lp}.mlp.fc2.weight"),
            (vec![0f32; e * h], vec![e, h]),
        );
        all.insert(format!("{lp}.mlp.fc2.bias"), (vec![0f32; e], vec![e]));
        WeightMap::from_tensors(all)
    }

    #[test]
    fn predictor_graph_builds() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_predictor_weights(&cfg);
        let model = extract_model_weights(&mut wm, &cfg).unwrap();
        let pred = model.predictor.as_ref().unwrap();
        let seq = cfg.num_patches();
        let masks = Vjepa2Masks {
            context: (0..seq / 2).collect(),
            target: (seq / 2..seq).collect(),
            mask_index: 0,
        };
        let layout = prepare_predictor_layout(&cfg, &masks, 1).unwrap();
        let mask_rows = predictor_mask_rows(pred, &cfg, &masks, 1);
        let (g, _) =
            build_vjepa2_predictor_graph_sized(&cfg, pred, &layout, &mask_rows, 1).unwrap();
        assert_eq!(g.outputs.len(), 1);
    }

    #[test]
    fn pooler_graph_builds() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_pooler_weights(&cfg);
        let model = extract_model_weights(&mut wm, &cfg).unwrap();
        let pooler = model.pooler.as_ref().unwrap();
        let (g, _) = build_vjepa2_pooler_graph_sized(&cfg, pooler, 1).unwrap();
        assert_eq!(g.outputs.len(), 1);
    }

    #[test]
    fn compiled_cpu_pipeline_matches_native() {
        use rlx_runtime::{Device, Session};

        let cfg = tiny_cfg();
        let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let mut pred_wm = synthetic_predictor_weights(&cfg);
        for k in pred_wm.keys().map(|s| s.to_string()).collect::<Vec<_>>() {
            tensors.insert(k.clone(), pred_wm.take(&k).unwrap());
        }
        let mut pool_wm = synthetic_pooler_weights(&cfg);
        for k in pool_wm.keys().map(|s| s.to_string()).collect::<Vec<_>>() {
            if k.starts_with("pooler.") {
                tensors.insert(k.clone(), pool_wm.take(&k).unwrap());
            }
        }
        let mut wm = WeightMap::from_tensors(tensors);
        let model = extract_model_weights(&mut wm, &cfg).unwrap();
        let video = vec![0.0f32; 3 * cfg.frames_per_clip * cfg.crop_size * cfg.crop_size];
        let seq = cfg.num_patches();
        let masks = Vjepa2Masks {
            context: (0..seq / 2).collect(),
            target: (seq / 2..seq).collect(),
            mask_index: 0,
        };

        let native_enc = encode_video_native(&model.encoder, &cfg, &video, 1).unwrap();
        let native_pred = predict_native(
            &native_enc.tokens,
            model.predictor.as_ref().unwrap(),
            &cfg,
            1,
            seq,
            &masks,
        )
        .unwrap();
        let native_pool = pool_native(
            &native_enc.tokens,
            model.pooler.as_ref().unwrap(),
            &cfg,
            1,
            seq,
        )
        .unwrap();

        let (enc_g, enc_p, _) = build_vjepa2_encoder_graph_sized(&cfg, &model.encoder, 1).unwrap();
        let enc_opts = rlx_core::flow_bridge::compile_options_for_profile(
            &rlx_flow::CompileProfile::encoder(),
            Device::Cpu,
        );
        let mut enc_c = Session::new(Device::Cpu).compile_with(enc_g, &enc_opts);
        Vjepa2GraphParams::from_f32(enc_p).load(&mut enc_c);
        let patch = conv3d_patch_embed(
            &model.encoder.patch,
            &video,
            cfg.frames_per_clip,
            cfg.crop_size,
            cfg.crop_size,
        )
        .unwrap();
        let mut enc_out = enc_c.run(&[("hidden", patch.as_slice())]);
        let compiled_enc = enc_out.remove(0);

        let pred_w = model.predictor.as_ref().unwrap();
        let layout = prepare_predictor_layout(&cfg, &masks, 1).unwrap();
        let mask_rows = predictor_mask_rows(pred_w, &cfg, &masks, 1);
        let (pred_g, pred_params) =
            build_vjepa2_predictor_graph_sized(&cfg, pred_w, &layout, &mask_rows, 1).unwrap();
        let pred_opts = rlx_core::flow_bridge::compile_options_for_profile(
            &rlx_flow::CompileProfile::encoder(),
            Device::Cpu,
        );
        let mut pred_c = Session::new(Device::Cpu).compile_with(pred_g, &pred_opts);
        pred_params.load(&mut pred_c);
        let mut pred_out = pred_c.run(&[("encoder", compiled_enc.as_slice())]);
        let compiled_pred = pred_out.remove(0);

        let (pool_g, pool_params) =
            build_vjepa2_pooler_graph_sized(&cfg, model.pooler.as_ref().unwrap(), 1).unwrap();
        let pool_opts = rlx_core::flow_bridge::compile_options_for_profile(
            &rlx_flow::CompileProfile::encoder(),
            Device::Cpu,
        );
        let mut pool_c = Session::new(Device::Cpu).compile_with(pool_g, &pool_opts);
        pool_params.load(&mut pool_c);
        let mut pool_out = pool_c.run(&[("encoder", compiled_enc.as_slice())]);
        let compiled_pool = pool_out.remove(0);

        assert_eq!(compiled_enc.len(), native_enc.tokens.len());
        assert_eq!(compiled_pred.len(), native_pred.tokens.len());
        assert_eq!(compiled_pool.len(), native_pool.embedding.len());
    }
}
