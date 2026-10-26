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

#![cfg(feature = "export")]

//! An exported cross-encoder package scores (query, passage) pairs as
//! `transformers` does. Opt-in: `RLX_TEST_RERANKER=<a
//! BertForSequenceClassification / XLMRobertaForSequenceClassification
//! checkpoint directory>` and `RLX_TEST_RERANKER_REF=<file>`: JSON
//! `{query, passages, scores}`, the scores `model(**tokenizer([query] * n,
//! passages, padding=True)).logits[:, 0]` gives.

use rlx_embed::export::{ExportOptions, export_reranker_rlxp};
use serde_json::{Value, json};

#[test]
fn an_exported_reranker_scores_like_transformers() {
    let (Some(dir), Some(reference)) = (
        rlx_ir::env::var_os("RLX_TEST_RERANKER"),
        rlx_ir::env::var_os("RLX_TEST_RERANKER_REF"),
    ) else {
        eprintln!("skip: set RLX_TEST_RERANKER and RLX_TEST_RERANKER_REF");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let reference: Value = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let query = reference["query"].as_str().unwrap();
    let passages: Vec<&str> = reference["passages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().unwrap())
        .collect();
    let want: Vec<f32> = reference["scores"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_f64().unwrap() as f32)
        .collect();

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("reranker.rlxp");
    let opts = ExportOptions {
        seqs: vec![16, 64],
        texts: 2,
        ..ExportOptions::default()
    };
    let report = export_reranker_rlxp(&dir, &out, &opts).unwrap();
    assert_eq!(
        report.functions,
        ["encode"],
        "{:?}",
        report.symbolic_failure
    );
    assert!(report.referenced > 0);

    let device = rlx_ir::env::var_result("RLX_TEST_DEVICE").unwrap_or_else(|_| "cpu".into());
    rlx_text::install_model_tokenizer();
    let mut m =
        rlx_runtime::model::open_source(out.to_str().unwrap(), json!({"device": device})).unwrap();
    assert_eq!(m.info().task, "text-ranking");
    // Three passages over a batch of two: both functions' batches, one
    // part-filled.
    let got = m.score(query, &passages, &json!({})).unwrap();
    eprintln!("scores {got:?}, transformers {want:?}");
    for (g, w) in got.iter().zip(&want) {
        assert!(
            (g - w).abs() < 1e-3 * w.abs().max(1.0),
            "{got:?} vs {want:?}"
        );
    }
    // The same through `run`.
    let mut inputs = vec![rlx_runtime::model::Tensor::text("query", query)];
    inputs.extend(
        passages
            .iter()
            .map(|p| rlx_runtime::model::Tensor::text("passage", p)),
    );
    let outs = m.run(&inputs, &json!({})).unwrap();
    assert_eq!(outs[0].to_f32().unwrap(), got);
    // An embedding call is refused with what to use.
    let e = m.embed(&["x"], &json!({})).unwrap_err().to_string();
    assert!(e.contains("use score"), "{e}");
}
