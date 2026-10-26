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

//! LM fine-tuning / QAT end to end on a tiny random Qwen3 checkpoint written
//! here: the family is sniffed, the loss falls, and the export reloads under
//! the checkpoint's names and shapes.

#![cfg(all(feature = "train", feature = "qwen3"))]

use std::path::Path;

use rlx_models::lm_train::{LmTrainOptions, qat_lm, trainable_lm};

const VOCAB: usize = 64;
const H: usize = 32;
const LAYERS: usize = 2;
const HEADS: usize = 2;
const KV: usize = 1;
const HD: usize = 16;
const FF: usize = 64;

fn write_checkpoint(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let config = serde_json::json!({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": VOCAB,
        "hidden_size": H,
        "intermediate_size": FF,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV,
        "head_dim": HD,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": true,
        "hidden_act": "silu",
        "torch_dtype": "float32"
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let mut seed = 1u64;
    let mut rand = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
            })
            .collect()
    };
    let mut t: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
        (
            "model.embed_tokens.weight".into(),
            vec![VOCAB, H],
            rand(VOCAB * H, 0.1),
        ),
        ("model.norm.weight".into(), vec![H], vec![1.0; H]),
    ];
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        let s = 1.0 / (H as f32).sqrt();
        t.push((format!("{p}.input_layernorm.weight"), vec![H], vec![1.0; H]));
        t.push((
            format!("{p}.post_attention_layernorm.weight"),
            vec![H],
            vec![1.0; H],
        ));
        t.push((
            format!("{p}.self_attn.q_proj.weight"),
            vec![HEADS * HD, H],
            rand(HEADS * HD * H, s),
        ));
        t.push((
            format!("{p}.self_attn.k_proj.weight"),
            vec![KV * HD, H],
            rand(KV * HD * H, s),
        ));
        t.push((
            format!("{p}.self_attn.v_proj.weight"),
            vec![KV * HD, H],
            rand(KV * HD * H, s),
        ));
        t.push((
            format!("{p}.self_attn.o_proj.weight"),
            vec![H, HEADS * HD],
            rand(H * HEADS * HD, s),
        ));
        t.push((
            format!("{p}.self_attn.q_norm.weight"),
            vec![HD],
            vec![1.0; HD],
        ));
        t.push((
            format!("{p}.self_attn.k_norm.weight"),
            vec![HD],
            vec![1.0; HD],
        ));
        t.push((
            format!("{p}.mlp.gate_proj.weight"),
            vec![FF, H],
            rand(FF * H, s),
        ));
        t.push((
            format!("{p}.mlp.up_proj.weight"),
            vec![FF, H],
            rand(FF * H, s),
        ));
        t.push((
            format!("{p}.mlp.down_proj.weight"),
            vec![H, FF],
            rand(H * FF, 1.0 / (FF as f32).sqrt()),
        ));
    }
    rlx_tune::lm::save_f32_safetensors(&dir.join("model.safetensors"), &t).unwrap();
}

/// A learnable stream: each token is the previous one plus 3, mod the vocab.
fn sequences(n: usize, len: usize) -> Vec<Vec<u32>> {
    (0..n)
        .map(|s| (0..len).map(|i| ((s * 7 + i * 3) % VOCAB) as u32).collect())
        .collect()
}

#[test]
fn fine_tune_and_qat_lower_held_out_loss() {
    let root = std::env::temp_dir().join(format!("rlx-lm-qat-{}", std::process::id()));
    let src = root.join("ckpt");
    write_checkpoint(&src);
    assert_eq!(trainable_lm(&src, 2, 8).unwrap().token_input, "input_ids");

    let (train, holdout) = (sequences(6, 64), sequences(2, 64));
    for format in [None, Some("f8".parse().unwrap())] {
        let out =
            root.join(format!("{format:?}").replace(['(', ')', ' ', '{', '}', ':', ','], "_"));
        let opts = LmTrainOptions {
            format,
            steps: 80,
            lr: 3e-3,
            ..Default::default()
        };
        let r = qat_lm(&src, &train, &holdout, 4, 16, &opts, &out).unwrap();
        eprintln!(
            "{format:?}: held-out loss {:.3} -> {:.3}",
            r.before, r.after
        );
        assert!(
            r.after < 0.6 * r.before,
            "{format:?}: {} -> {}",
            r.before,
            r.after
        );

        // The export has the checkpoint's tensors, names and shapes.
        let a = rlx_tune::lm::safetensors_shapes(&src.join("model.safetensors")).unwrap();
        let b = rlx_tune::lm::safetensors_shapes(&out.join("model.safetensors")).unwrap();
        assert_eq!(a, b);
        assert!(out.join("config.json").is_file());
        // …and holds the trained weights: scoring the reloaded export gives
        // the same held-out loss as the in-memory result.
        let reloaded = trainable_lm(&out, 4, 16).unwrap();
        let map = rlx_tune::lm::map_checkpoint(&reloaded, &out.join("model.safetensors")).unwrap();
        let again = rlx_tune::lm::eval_lm_loss(
            &reloaded,
            &map,
            &Default::default(),
            &holdout,
            format,
            8,
            rlx_runtime::Device::Cpu,
            0,
        )
        .unwrap();
        assert!(
            (again - r.after).abs() < 1e-4,
            "export scores {again}, trained {}",
            r.after
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}
