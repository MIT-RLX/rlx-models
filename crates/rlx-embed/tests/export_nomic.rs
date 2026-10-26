// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! An exported NomicBERT package (rotary attention, SwiGLU), opened by rlx
//! alone, embeds what `RlxNomicModel` and mean pooling do: a two-layer
//! model with random weights and Hugging Face's `nomic_bert` config keys.
#![cfg(feature = "export")]

use std::collections::HashMap;

use rlx_core::export::ExportOptions;
use rlx_core::export::check::write_checkpoint;
use rlx_embed::RlxNomicModel;
use rlx_embed::export::export_rlxp;
use serde_json::json;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.5
        })
        .collect()
}

#[test]
fn an_exported_nomic_bert_embeds_what_the_runner_does() {
    let (vocab, h, inter, layers, heads) = (64, 32, 64, 2, 4);
    let config = json!({
        "model_type": "nomic_bert", "vocab_size": vocab, "n_embd": h, "n_layer": layers,
        "n_head": heads, "n_inner": inter, "n_positions": 512, "type_vocab_size": 2,
        "layer_norm_epsilon": 1e-12, "rotary_emb_base": 1000.0,
    });
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 2u64;
    let mut put = |k: String, shape: Vec<usize>| {
        seed += 17;
        let n = shape.iter().product();
        t.insert(k, (fill(n, seed), shape));
    };
    put("embeddings.word_embeddings.weight".into(), vec![vocab, h]);
    put("embeddings.token_type_embeddings.weight".into(), vec![2, h]);
    put("emb_ln.weight".into(), vec![h]);
    put("emb_ln.bias".into(), vec![h]);
    for i in 0..layers {
        let l = format!("encoder.layers.{i}");
        put(format!("{l}.attn.Wqkv.weight"), vec![3 * h, h]);
        put(format!("{l}.attn.out_proj.weight"), vec![h, h]);
        put(format!("{l}.mlp.fc11.weight"), vec![inter, h]);
        put(format!("{l}.mlp.fc12.weight"), vec![inter, h]);
        put(format!("{l}.mlp.fc2.weight"), vec![h, inter]);
        for n in ["norm1", "norm2"] {
            put(format!("{l}.{n}.weight"), vec![h]);
            put(format!("{l}.{n}.bias"), vec![h]);
        }
    }
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("nomic");
    write_checkpoint(&dir, &config, &t, vocab).unwrap();
    let out = tmp.path().join("nomic.rlxp");
    let opts = ExportOptions {
        seqs: vec![16, 64],
        ..ExportOptions::default()
    };
    let report = export_rlxp(&dir, &out, &opts).unwrap();
    eprintln!("{report}");

    rlx_text::install_model_tokenizer();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    // The runner's config: its own keys.
    let cfg_path = tmp.path().join("nomic_runner_config.json");
    std::fs::write(
        &cfg_path,
        json!({"vocab_size": vocab, "hidden_size": h, "num_hidden_layers": layers,
               "num_attention_heads": heads, "intermediate_size": inter,
               "max_position_embeddings": 512, "head_dim": h / heads})
        .to_string(),
    )
    .unwrap();
    let weights = dir.join("model.safetensors");
    let long: String = (0..40).map(|i| format!("t{} ", 3 + (i * 7) % 60)).collect();
    let texts = ["t5 t9 t13 t17", "t40", long.as_str()];
    let got = m.embed(&texts, &json!({})).unwrap();
    for (text, got) in texts.iter().zip(&got) {
        let ids: Vec<f32> = text
            .split_whitespace()
            .map(|w| w[1..].parse::<f32>().unwrap())
            .collect();
        let n = ids.len();
        let mut r = RlxNomicModel::load_sized(&cfg_path, weights.to_str().unwrap(), 1, n).unwrap();
        let hidden = r.forward(&ids, &vec![1.0; n], &vec![0.0; n]);
        let mut want = vec![0f32; h];
        for row in hidden.chunks_exact(h) {
            want.iter_mut()
                .zip(row)
                .for_each(|(a, b)| *a += b / n as f32);
        }
        let norm = want.iter().map(|x| x * x).sum::<f32>().sqrt();
        want.iter_mut().for_each(|x| *x /= norm);
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(worst < 1e-4, "{n} tokens: off by {worst}");
    }
}
