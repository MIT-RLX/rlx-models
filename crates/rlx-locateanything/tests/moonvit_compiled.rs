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

//! Compiled MoonViT vs CPU reference on synthetic weights.

use rlx_core::weight_map::WeightMap;
use rlx_locateanything::config::{LocateAnythingConfig, LocateAnythingTextConfig, MoonVitConfig};
use rlx_locateanything::moonvit::{MoonVitCache, encode_image, load_moonvit_weights};
use rlx_locateanything::moonvit_flow::build_moonvit_built;
use rlx_locateanything::preprocess::preprocess_image;
use rlx_runtime::Device;
use std::collections::HashMap;

fn tiny_cfg() -> LocateAnythingConfig {
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
            vocab_size: 128,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
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

fn synth_vision_weights(cfg: &LocateAnythingConfig) -> WeightMap {
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

#[test]
fn moonvit_flow_builds() {
    let cfg = tiny_cfg();
    let mut wm = synth_vision_weights(&cfg);
    let built = build_moonvit_built(&cfg.vision_config, &mut wm, 1, 2, 2, Device::Cpu).unwrap();
    assert_eq!(built.model.primary_shape().rank(), 3);
}

#[test]
fn moonvit_compiled_matches_cpu_reference() {
    let cfg = tiny_cfg();
    let ph = 28usize;
    let pw = 28usize;
    let img = image::RgbImage::new(pw as u32, ph as u32);
    let prep = preprocess_image(&image::DynamicImage::ImageRgb8(img), &cfg).unwrap();

    let mut wm_cpu = synth_vision_weights(&cfg);
    let vit = load_moonvit_weights(&mut wm_cpu, &cfg.vision_config).unwrap();
    let cpu = encode_image(&vit, &prep).unwrap();

    let mut wm_gpu = synth_vision_weights(&cfg);
    let mut cache = MoonVitCache::default();
    let compiled = cache
        .encode(&cfg.vision_config, Some(&mut wm_gpu), &prep, Device::Cpu)
        .unwrap();

    assert_eq!(cpu.len(), compiled.len());
    let n = cpu.len().min(compiled.len());
    let mut max_diff = 0f32;
    for i in 0..n {
        max_diff = max_diff.max((cpu[i] - compiled[i]).abs());
    }
    eprintln!("moonvit cpu vs compiled max_abs_diff={max_diff:.6}");
    assert!(max_diff < 0.05, "max diff {max_diff}");
}

#[cfg(feature = "metal")]
#[test]
fn moonvit_compiled_matches_cpu_on_metal() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip moonvit metal parity: backend not available");
        return;
    }
    let cfg = tiny_cfg();
    let ph = 28usize;
    let img = image::RgbImage::new(ph as u32, ph as u32);
    let prep = preprocess_image(&image::DynamicImage::ImageRgb8(img), &cfg).unwrap();

    let mut wm_cpu = synth_vision_weights(&cfg);
    let vit = load_moonvit_weights(&mut wm_cpu, &cfg.vision_config).unwrap();
    let cpu = encode_image(&vit, &prep).unwrap();

    let mut wm_gpu = synth_vision_weights(&cfg);
    let mut cache = MoonVitCache::default();
    let gpu = cache
        .encode(&cfg.vision_config, Some(&mut wm_gpu), &prep, Device::Metal)
        .unwrap();

    assert_eq!(cpu.len(), gpu.len());
    let mut max_diff = 0f32;
    for (a, b) in cpu.iter().zip(gpu.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    eprintln!("moonvit cpu vs metal max_abs_diff={max_diff:.6}");
    assert!(max_diff < 0.05, "max diff {max_diff}");
}
