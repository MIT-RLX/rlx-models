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

//! ClinicalBERT through the `embed` exporter (`rlx_embed::export`): a
//! BERT checkpoint with HF's `bert.` names (`BertForMaskedLM`), random
//! weights, exported and opened by rlx alone, embeds as
//! `ClinicalBertRunner` with mean pooling does.
#![cfg(feature = "export")]

use std::collections::HashMap;

use rlx_clinicalbert::runner::{ClinicalBertRunner, Pooling};
use rlx_core::export::ExportOptions;
use rlx_core::export::check::write_checkpoint;
use rlx_embed::export::export_rlxp;
use rlx_runtime::Device;
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
fn an_exported_clinicalbert_embeds_what_the_runner_does() {
    let (vocab, h, inter, layers, heads, positions) = (64, 32, 64, 2, 4, 64);
    let config = json!({
        "model_type": "bert", "architectures": ["BertForMaskedLM"],
        "vocab_size": vocab, "hidden_size": h, "num_hidden_layers": layers,
        "num_attention_heads": heads, "intermediate_size": inter,
        "max_position_embeddings": positions, "type_vocab_size": 2,
        "layer_norm_eps": 1e-12, "hidden_act": "gelu",
    });
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 5u64;
    let mut put = |k: String, shape: Vec<usize>| {
        seed += 17;
        let n = shape.iter().product();
        t.insert(k, (fill(n, seed), shape));
    };
    let e = "bert.embeddings";
    put(format!("{e}.word_embeddings.weight"), vec![vocab, h]);
    put(
        format!("{e}.position_embeddings.weight"),
        vec![positions, h],
    );
    put(format!("{e}.token_type_embeddings.weight"), vec![2, h]);
    put(format!("{e}.LayerNorm.weight"), vec![h]);
    put(format!("{e}.LayerNorm.bias"), vec![h]);
    for i in 0..layers {
        let l = format!("bert.encoder.layer.{i}");
        for p in ["query", "key", "value"] {
            put(format!("{l}.attention.self.{p}.weight"), vec![h, h]);
            put(format!("{l}.attention.self.{p}.bias"), vec![h]);
        }
        put(format!("{l}.attention.output.dense.weight"), vec![h, h]);
        put(format!("{l}.attention.output.dense.bias"), vec![h]);
        put(format!("{l}.intermediate.dense.weight"), vec![inter, h]);
        put(format!("{l}.intermediate.dense.bias"), vec![inter]);
        put(format!("{l}.output.dense.weight"), vec![h, inter]);
        put(format!("{l}.output.dense.bias"), vec![h]);
        for n in ["attention.output.LayerNorm", "output.LayerNorm"] {
            put(format!("{l}.{n}.weight"), vec![h]);
            put(format!("{l}.{n}.bias"), vec![h]);
        }
    }
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("clinicalbert");
    write_checkpoint(&dir, &config, &t, vocab).unwrap();
    let out = tmp.path().join("clinicalbert.rlxp");
    let opts = ExportOptions {
        seqs: vec![16, 64],
        ..ExportOptions::default()
    };
    eprintln!("{}", export_rlxp(&dir, &out, &opts).unwrap());

    rlx_text::install_model_tokenizer();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let long: String = (0..40).map(|i| format!("t{} ", 3 + (i * 7) % 60)).collect();
    let texts = ["t5 t9 t13 t17", "t40", long.as_str()];
    let got = m.embed(&texts, &json!({})).unwrap();
    for (text, got) in texts.iter().zip(&got) {
        // The test tokenizer's ids are the words' numbers, no specials.
        let ids: Vec<f32> = text
            .split_whitespace()
            .map(|w| w[1..].parse::<f32>().unwrap())
            .collect();
        let n = ids.len();
        let mut r = ClinicalBertRunner::builder()
            .weights(dir.join("model.safetensors"))
            .device(Device::Cpu)
            .max_seq(n)
            .pooling(Pooling::Mean)
            .build()
            .unwrap();
        let positions: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut want = r
            .embed(&ids, &vec![1.0; n], &vec![0.0; n], &positions)
            .unwrap();
        let norm = want.iter().map(|x| x * x).sum::<f32>().sqrt();
        want.iter_mut().for_each(|x| *x /= norm);
        assert_eq!(got.len(), want.len(), "{n} tokens");
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        eprintln!("{n} tokens: off by {worst}");
        assert!(worst < 1e-4, "{n} tokens: off by {worst}");
    }
}
