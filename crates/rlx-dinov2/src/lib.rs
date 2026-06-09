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

//! DINOv2 — Meta's self-supervised ViT (with optional registers).
//!
//! Public entry points:
//!   - [`DinoV2Config`] — model dimensions + variant factories
//!     (`vit_small`, `vit_base`, `vit_large`)
//!   - [`build_dinov2_graph_sized`] — emits the IR graph and the
//!     host-side [`DinoV2PreprocessWeights`].
//!   - [`assemble_hidden`] / [`rgb_u8_to_imagenet_nchw`] — host-side
//!     image → encoder-input plumbing.
//!
//! Weight keys match Meta / candle's safetensors so checkpoints from
//! the HF Hub (e.g. `lmz/candle-dino-v2`) load with no remapping.

pub mod builder;
pub mod cli;
pub mod config;
pub mod flow;
pub mod packed_gguf;
pub mod preprocess;
pub mod runner;

pub use builder::build_dinov2_graph_sized;
pub use config::{DinoV2Config, IMAGENET_MEAN, IMAGENET_STD};
pub use flow::{DinoV2Built, DinoV2Flow, build_dinov2_built, build_dinov2_built_with_packed};
pub use packed_gguf::{gguf_has_packed_linears, load_dinov2_from_gguf};
pub use preprocess::{DinoV2PreprocessWeights, assemble_hidden, rgb_u8_to_imagenet_nchw};
pub use runner::{DinoV2Output, DinoV2Runner, DinoV2RunnerBuilder, DinoV2Variant};

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::weight_map::WeightMap;
    use std::collections::HashMap;

    fn synthetic_weights(cfg: &DinoV2Config) -> WeightMap {
        let h = cfg.hidden_size;
        let int_dim = cfg.intermediate_size();
        let ps = cfg.patch_size;
        let pd = cfg.patch_dim();
        let seq = cfg.seq_len();

        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.0f32; n];

        t.insert(
            "patch_embed.proj.weight".into(),
            (z(h * pd), vec![h, 3, ps, ps]),
        );
        t.insert("patch_embed.proj.bias".into(), (z(h), vec![h]));
        t.insert("cls_token".into(), (z(h), vec![1, 1, h]));
        t.insert("pos_embed".into(), (z(seq * h), vec![1, seq, h]));
        if cfg.num_register_tokens > 0 {
            t.insert(
                "register_tokens".into(),
                (
                    z(cfg.num_register_tokens * h),
                    vec![1, cfg.num_register_tokens, h],
                ),
            );
        }
        for i in 0..cfg.num_hidden_layers {
            let lp = format!("blocks.{i}");
            t.insert(format!("{lp}.norm1.weight"), (z(h), vec![h]));
            t.insert(format!("{lp}.norm1.bias"), (z(h), vec![h]));
            t.insert(
                format!("{lp}.attn.qkv.weight"),
                (z(h * 3 * h), vec![3 * h, h]),
            );
            t.insert(format!("{lp}.attn.qkv.bias"), (z(3 * h), vec![3 * h]));
            t.insert(format!("{lp}.attn.proj.weight"), (z(h * h), vec![h, h]));
            t.insert(format!("{lp}.attn.proj.bias"), (z(h), vec![h]));
            t.insert(format!("{lp}.ls1.gamma"), (z(h), vec![h]));
            t.insert(format!("{lp}.norm2.weight"), (z(h), vec![h]));
            t.insert(format!("{lp}.norm2.bias"), (z(h), vec![h]));
            t.insert(
                format!("{lp}.mlp.fc1.weight"),
                (z(int_dim * h), vec![int_dim, h]),
            );
            t.insert(format!("{lp}.mlp.fc1.bias"), (z(int_dim), vec![int_dim]));
            t.insert(
                format!("{lp}.mlp.fc2.weight"),
                (z(h * int_dim), vec![h, int_dim]),
            );
            t.insert(format!("{lp}.mlp.fc2.bias"), (z(h), vec![h]));
            t.insert(format!("{lp}.ls2.gamma"), (z(h), vec![h]));
        }
        t.insert("norm.weight".into(), (z(h), vec![h]));
        t.insert("norm.bias".into(), (z(h), vec![h]));
        if cfg.num_classes > 0 {
            t.insert(
                "head.weight".into(),
                (z(cfg.num_classes * 2 * h), vec![cfg.num_classes, 2 * h]),
            );
            t.insert(
                "head.bias".into(),
                (z(cfg.num_classes), vec![cfg.num_classes]),
            );
        }
        WeightMap::from_tensors(t)
    }

    #[test]
    fn encoder_only_graph_builds() {
        let mut cfg = DinoV2Config::vit_small(28);
        cfg.num_classes = 0; // encoder-only
        let mut wm = synthetic_weights(&cfg);
        let (g, _params, pre) = build_dinov2_graph_sized(&cfg, &mut wm, 1).unwrap();
        assert_eq!(g.outputs.len(), 1);
        assert_eq!(pre.embed_dim, cfg.hidden_size);
        assert_eq!(wm.len(), 0);
    }

    #[test]
    fn classifier_graph_builds() {
        let cfg = DinoV2Config::vit_small(28); // num_classes defaults to 1000
        let mut wm = synthetic_weights(&cfg);
        let (g, _, _) = build_dinov2_graph_sized(&cfg, &mut wm, 1).unwrap();
        assert_eq!(g.outputs.len(), 1);
        // Final output should be [B, num_classes].
        let out_id = g.outputs[0];
        let s = g.shape(out_id);
        let dims: Vec<usize> = s.dims().iter().map(|d| d.unwrap_static()).collect();
        assert_eq!(dims, vec![1, cfg.num_classes]);
    }

    #[test]
    fn with_register_tokens() {
        let mut cfg = DinoV2Config::vit_small(28);
        cfg.num_register_tokens = 4;
        let mut wm = synthetic_weights(&cfg);
        let (_g, _, pre) = build_dinov2_graph_sized(&cfg, &mut wm, 1).unwrap();
        assert_eq!(pre.register_tokens.len(), 4 * cfg.hidden_size);
        assert_eq!(pre.seq, 1 + 4 + cfg.num_patches());
    }

    /// Build a WeightMap like `synthetic_weights` but with a callback
    /// to override the data for specific keys (preserving shape).
    fn synthetic_weights_with<F: Fn(&str, &mut Vec<f32>)>(
        cfg: &DinoV2Config,
        edit: F,
    ) -> WeightMap {
        let mut wm = synthetic_weights(cfg);
        let keys: Vec<String> = wm.keys().map(|s| s.to_string()).collect();
        let mut all: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        for k in keys {
            let (mut d, s) = wm.take(&k).unwrap();
            edit(&k, &mut d);
            all.insert(k, (d, s));
        }
        WeightMap::from_tensors(all)
    }

    #[test]
    fn assemble_hidden_zero_image_yields_pos_embed_plus_bias_plus_cls() {
        // With zero pixels, the patch projection contributes only its
        // bias; the assembled hidden then equals pos_embed broadcast +
        // [cls; proj_b…; proj_b…] per row.
        let mut cfg = DinoV2Config::vit_small(28);
        cfg.num_classes = 0;
        let h = cfg.hidden_size;
        let seq = cfg.seq_len();
        let np = cfg.num_patches();

        let pos: Vec<f32> = (0..seq * h).map(|i| i as f32 * 1e-3).collect();
        let cls: Vec<f32> = (0..h).map(|i| 100.0 + i as f32).collect();
        let bias: Vec<f32> = (0..h).map(|i| -1.0 - (i as f32) * 0.1).collect();
        let pos_clone = pos.clone();
        let cls_clone = cls.clone();
        let bias_clone = bias.clone();

        let mut wm = synthetic_weights_with(&cfg, |k, d| match k {
            "pos_embed" => d.copy_from_slice(&pos_clone),
            "cls_token" => d.copy_from_slice(&cls_clone),
            "patch_embed.proj.bias" => d.copy_from_slice(&bias_clone),
            _ => {}
        });

        let (_g, _p, pre) = build_dinov2_graph_sized(&cfg, &mut wm, 1).unwrap();
        let image = vec![0f32; 3 * cfg.img_size * cfg.img_size];
        let hidden = assemble_hidden(&pre, &image, 1, cfg.patch_size, cfg.img_size).unwrap();
        assert_eq!(hidden.len(), seq * h);

        // Row 0 = CLS + pos_embed[0]; rows 1..1+np = bias + pos_embed[row]
        for k in 0..h {
            let exp = cls[k] + pos[k];
            assert!(
                (hidden[k] - exp).abs() < 1e-5,
                "cls col {k}: {} vs {}",
                hidden[k],
                exp
            );
        }
        for row in 1..(1 + np) {
            for k in 0..h {
                let exp = bias[k] + pos[row * h + k];
                let got = hidden[row * h + k];
                assert!(
                    (got - exp).abs() < 1e-5,
                    "row {row} col {k}: {got} vs {exp}"
                );
            }
        }
    }
}
