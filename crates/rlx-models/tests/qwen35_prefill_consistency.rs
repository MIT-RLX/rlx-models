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

// Consistency: one-shot predict vs prefill-cache seed should agree on trunk logits.

mod compile_support;

use rlx_models::Qwen35RunnerBuilder;
use std::path::PathBuf;

fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.into_iter()
        .take(k)
        .map(|i| (i as u32, logits[i]))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for (&x, &y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt()).max(f32::EPSILON)
}

#[test]
fn qwen35_predict_logits_matches_prefill_seed() {
    let weights = match std::env::var("QWEN35_GGUF_PATH") {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ => {
            eprintln!("skip qwen35_prefill_consistency: set QWEN35_GGUF_PATH");
            return;
        }
    };
    if !weights.is_file() {
        panic!("QWEN35_GGUF_PATH={:?} is not a file", weights);
    }

    let prompt = vec![1u32, 2, 3, 4];
    let max_seq = prompt.len().max(8);

    let mut runner = Qwen35RunnerBuilder::default()
        .weights(&weights)
        .max_seq(max_seq)
        .last_logits_only(true)
        .packed_weights(true)
        .build()
        .expect("build runner");

    let predict = runner.predict_logits(&prompt).expect("predict_logits");
    let seed = runner
        .prefill_seed_for_decode(&prompt)
        .expect("prefill_seed_for_decode");

    assert_eq!(
        predict.logits.len(),
        seed.trunk_logits.len(),
        "logit vector lengths differ"
    );

    let cos = cosine(&predict.logits, &seed.trunk_logits);
    eprintln!("qwen35 prefill consistency cosine={cos:.6}");
    assert!(
        cos > 0.999,
        "predict vs prefill-seed cosine {cos} below 0.999"
    );

    let k = 8usize;
    let p_top = top_k(&predict.logits, k);
    let s_top = top_k(&seed.trunk_logits, k);
    assert_eq!(
        p_top.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
        s_top.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
        "top-{k} token ids diverged between predict and prefill-seed"
    );
}
