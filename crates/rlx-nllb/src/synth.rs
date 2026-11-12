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

//! Synthetic weights in the real `M2M100ForConditionalGeneration` layout
//! (the tensor names of `facebook/nllb-200-distilled-600M`, at any size), for
//! tests that run without a checkpoint.

use crate::config::NllbConfig;
use crate::weights::lang as lk;
use rlx_core::weight_map::WeightMap;
use std::collections::HashMap;

/// Deterministic weights for `cfg`. LayerNorm gains sit near 1 so
/// activations stay in a realistic range through the stack.
pub fn weights(cfg: &NllbConfig) -> WeightMap {
    let mut t = HashMap::new();
    let mut seed = 1u32;
    let mut put = |key: String, shape: &[usize], base: f32, amp: f32| {
        let n: usize = shape.iter().product();
        let data = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                base + ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * amp
            })
            .collect();
        t.insert(key, (data, shape.to_vec()));
    };
    let d = cfg.d_model;
    put(lk::SHARED.into(), &[cfg.vocab_size, d], 0.0, 0.2);
    let norm = |put: &mut dyn FnMut(String, &[usize], f32, f32), key: String| {
        put(format!("{key}.weight"), &[d], 1.0, 0.2);
        put(format!("{key}.bias"), &[d], 0.0, 0.1);
    };
    let linear =
        |put: &mut dyn FnMut(String, &[usize], f32, f32), key: String, o: usize, i: usize| {
            put(
                format!("{key}.weight"),
                &[o, i],
                0.0,
                2.0 / (i as f32).sqrt(),
            );
            put(format!("{key}.bias"), &[o], 0.0, 0.05);
        };
    for l in 0..cfg.encoder_layers {
        let p = |s: &str| lk::enc_layer(l, s);
        for m in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            linear(&mut put, p(&format!("self_attn.{m}")), d, d);
        }
        norm(&mut put, p("self_attn_layer_norm"));
        linear(&mut put, p("fc1"), cfg.encoder_ffn_dim, d);
        linear(&mut put, p("fc2"), d, cfg.encoder_ffn_dim);
        norm(&mut put, p("final_layer_norm"));
    }
    norm(&mut put, "model.encoder.layer_norm".into());
    for l in 0..cfg.decoder_layers {
        let p = |s: &str| lk::dec_layer(l, s);
        for a in ["self_attn", "encoder_attn"] {
            for m in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                linear(&mut put, p(&format!("{a}.{m}")), d, d);
            }
            norm(&mut put, p(&format!("{a}_layer_norm")));
        }
        linear(&mut put, p("fc1"), cfg.decoder_ffn_dim, d);
        linear(&mut put, p("fc2"), d, cfg.decoder_ffn_dim);
        norm(&mut put, p("final_layer_norm"));
    }
    norm(&mut put, "model.decoder.layer_norm".into());
    WeightMap::from_tensors(t)
}
