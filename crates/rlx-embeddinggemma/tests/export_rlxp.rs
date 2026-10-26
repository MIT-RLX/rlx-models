// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! An exported EmbeddingGemma package, opened by rlx alone, embeds what
//! `EmbeddingGemma::embed_text` does — tokenizer, embeddings, both masks,
//! encoder, pooling, dense head and normalization all from the package.
//!
//! ```sh
//! RLX_EMBEDDINGGEMMA_DIR=$PWD/weights/embed/embeddinggemma-300m \
//!     cargo test -p rlx-embeddinggemma --release --features export --test export_rlxp
//! ```
#![cfg(feature = "export")]

use rlx_core::export::ExportOptions;
use rlx_embeddinggemma::EmbeddingGemma;
use rlx_embeddinggemma::export::export_rlxp;
use serde_json::json;

#[test]
fn an_exported_embeddinggemma_embeds_what_the_runner_does() {
    let Ok(dir) = rlx_ir::env::var_result("RLX_EMBEDDINGGEMMA_DIR") else {
        eprintln!("skip: set RLX_EMBEDDINGGEMMA_DIR");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("embeddinggemma.rlxp");
    let opts = ExportOptions {
        seqs: vec![64, 512],
        ..ExportOptions::default()
    };
    let report = export_rlxp(&dir, &out, &opts).unwrap();
    eprintln!(
        "{report}, {} KB",
        std::fs::metadata(&out).unwrap().len() / 1000
    );

    rlx_text::install_model_tokenizer();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut runner = EmbeddingGemma::open(&dir, rlx_runtime::Device::Cpu).unwrap();
    // A text past the sliding window's 257 tokens either way, so the
    // sliding layers' band masks keys.
    let long = "A river runs past the old mill and down to the sea. ".repeat(36);
    let texts = [
        "task: search result | query: what is the capital of France?",
        "Paris is the capital and largest city of France.",
        long.as_str(),
    ];
    let got = m.embed(&texts, &json!({})).unwrap();
    for (text, got) in texts.iter().zip(&got) {
        let want = runner.embed_text("", text).unwrap();
        assert_eq!(got.len(), want.len());
        let cos: f32 = got.iter().zip(&want).map(|(a, b)| a * b).sum();
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        eprintln!("{} chars: cosine {cos}, off by {worst}", text.len());
        assert!(worst < 1e-4, "{text:?}: off by {worst}");
    }
}
