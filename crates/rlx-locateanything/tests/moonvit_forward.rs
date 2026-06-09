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

//! MoonViT + projector smoke test on tiny synthetic weights.

use rlx_core::weight_map::WeightMap;
use rlx_locateanything::config::{LocateAnythingConfig, LocateAnythingTextConfig, MoonVitConfig};
use rlx_locateanything::moonvit::{encode_image, load_moonvit_weights};
use rlx_locateanything::preprocess::preprocess_image;
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
            tie_word_embeddings: true,
            block_size: 6,
            causal_attn: false,
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

fn fill_layer(wm: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, i: usize, h: usize, mlp: usize) {
    let z = |n: usize| vec![0.01f32; n];
    let lp = format!("vision_model.encoder.blocks.{i}");
    wm.insert(format!("{lp}.norm0.weight"), (z(h), vec![h]));
    wm.insert(format!("{lp}.norm0.bias"), (z(h), vec![h]));
    wm.insert(format!("{lp}.wqkv.weight"), (z(h * 3 * h), vec![h * 3, h]));
    wm.insert(format!("{lp}.wqkv.bias"), (z(h * 3), vec![h * 3]));
    wm.insert(format!("{lp}.wo.weight"), (z(h * h), vec![h, h]));
    wm.insert(format!("{lp}.wo.bias"), (z(h), vec![h]));
    wm.insert(format!("{lp}.norm1.weight"), (z(h), vec![h]));
    wm.insert(format!("{lp}.norm1.bias"), (z(h), vec![h]));
    wm.insert(format!("{lp}.mlp.fc0.weight"), (z(mlp * h), vec![mlp, h]));
    wm.insert(format!("{lp}.mlp.fc0.bias"), (z(mlp), vec![mlp]));
    wm.insert(format!("{lp}.mlp.fc1.weight"), (z(h * mlp), vec![h, mlp]));
    wm.insert(format!("{lp}.mlp.fc1.bias"), (z(h), vec![h]));
}

#[test]
fn moonvit_encode_smoke() {
    let cfg = tiny_cfg();
    let h = cfg.vision_config.hidden_size;
    let ps = cfg.vision_config.patch_size;
    let ph = 28usize;
    let pw = 28usize;
    let z = |n: usize| vec![0.02f32; n];

    let mut tensors = HashMap::new();
    tensors.insert(
        "vision_model.patch_embed.proj.weight".into(),
        (z(h * 3 * ps * ps), vec![h, 3, ps, ps]),
    );
    tensors.insert("vision_model.patch_embed.proj.bias".into(), (z(h), vec![h]));
    tensors.insert(
        "vision_model.patch_embed.pos_emb.weight".into(),
        (z(4 * 4 * h), vec![4, 4, h]),
    );
    fill_layer(&mut tensors, 0, h, 32);
    tensors.insert(
        "vision_model.encoder.final_layernorm.weight".into(),
        (z(h), vec![h]),
    );
    tensors.insert(
        "vision_model.encoder.final_layernorm.bias".into(),
        (z(h), vec![h]),
    );

    let mut wm = WeightMap::from_tensors(tensors);
    let vit = load_moonvit_weights(&mut wm, &cfg.vision_config).unwrap();

    let img = image::RgbImage::new(pw as u32, ph as u32);
    let prep = preprocess_image(&image::DynamicImage::ImageRgb8(img), &cfg).unwrap();
    let out = encode_image(&vit, &prep).unwrap();
    let grid_h = ph / ps;
    let grid_w = pw / ps;
    let n_out = (grid_h / 2) * (grid_w / 2);
    assert_eq!(out.len(), n_out * h * 4);
}
