// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! An exported jina-embeddings-v5 package (a task adapter merged), opened
//! by rlx alone, embeds what `JinaV5::embed_text` does.
//!
//! ```sh
//! RLX_JINA_V5_DIR=$PWD/weights/embed/jina-embeddings-v5-text-nano \
//!     cargo test -p rlx-jina-v5 --release --features export --test export_rlxp
//! ```
#![cfg(feature = "export")]

use rlx_core::export::ExportOptions;
use rlx_jina_v5::JinaV5;
use rlx_jina_v5::export::export_rlxp;
use serde_json::json;

#[test]
fn an_exported_jina_v5_embeds_what_the_runner_does() {
    let Ok(dir) = rlx_ir::env::var_result("RLX_JINA_V5_DIR") else {
        eprintln!("skip: set RLX_JINA_V5_DIR");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    rlx_text::install_model_tokenizer();
    let mut first = Vec::new();
    for task in ["retrieval", "text-matching"] {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("jina.rlxp");
        let opts = ExportOptions {
            seqs: vec![64, 256],
            variant: Some(task.into()),
            ..ExportOptions::default()
        };
        let report = export_rlxp(&dir, &out, &opts).unwrap();
        eprintln!("{task}: {report}");
        let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
        let mut runner = JinaV5::open(&dir, task, rlx_runtime::Device::Cpu).unwrap();
        let long = "Embeddings place texts with like meaning close together. ".repeat(12);
        let texts = [
            "Query: what is the capital of France?",
            "Paris.",
            long.as_str(),
        ];
        let got = m.embed(&texts, &json!({})).unwrap();
        first.push(got[0].clone());
        for (text, got) in texts.iter().zip(&got) {
            let want = runner.embed_text("", text).unwrap();
            let worst = got
                .iter()
                .zip(&want)
                .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
            eprintln!("  {} chars: off by {worst}", text.len());
            assert!(worst < 1e-4, "{task} {text:?}: off by {worst}");
        }
    }
    // The task's adapter is in the package: two tasks embed a text apart.
    let apart = first[0]
        .iter()
        .zip(&first[1])
        .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
    assert!(
        apart > 1e-3,
        "the two tasks' packages embed alike ({apart})"
    );
}
