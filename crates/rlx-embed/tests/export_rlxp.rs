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

//! An exported encoder package, opened by rlx alone, embeds as rlx-models'
//! own embedder does. Opt-in: `RLX_TEST_EMBED=<a BERT / XLM-R checkpoint
//! directory>`.

use rlx_embed::export::{ExportOptions, export_rlxp};
use rlx_embed::{BertTokenizer, Pooling, RlxBertModel, embed_with_rlx};
use serde_json::json;

#[test]
fn an_exported_package_embeds_like_the_rlx_models_embedder() {
    let Some(dir) = rlx_ir::env::var_os("RLX_TEST_EMBED") else {
        eprintln!("skip: RLX_TEST_EMBED is not set");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("encoder.rlxp");
    let opts = ExportOptions {
        seqs: vec![16, 64],
        texts: 2,
        ..ExportOptions::default()
    };
    let t = std::time::Instant::now();
    let report = export_rlxp(&dir, &out, &opts).unwrap();
    eprintln!(
        "export {:?}: {:?}, {} params referenced, {} held, package {} bytes",
        t.elapsed(),
        report.functions,
        report.referenced,
        report.held,
        std::fs::metadata(&out).unwrap().len()
    );
    eprintln!("symbolic: {:?}", report.symbolic_failure);
    assert_eq!(
        report.functions,
        ["encode"],
        "{:?}",
        report.symbolic_failure
    );
    assert!(report.referenced > 0);

    let texts = [
        "query: how much protein should a female eat",
        "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
        "query: 南瓜的家常做法",
        "passage: A brown fox jumps over the sleeping dog, then runs far away into the forest, past the river.",
    ];

    // As any program would: rlx, the package, a tokenizer.
    // The package on `RLX_TEST_DEVICE` (`cpu` by default); the reference
    // embedder on CPU.
    let device = rlx_ir::env::var_result("RLX_TEST_DEVICE").unwrap_or_else(|_| "cpu".into());
    rlx_text::install_model_tokenizer();
    let t = std::time::Instant::now();
    let mut m =
        rlx_runtime::model::open_source(out.to_str().unwrap(), json!({"device": device})).unwrap();
    let got = m.embed(&texts, &json!({})).unwrap();
    eprintln!("package: open + embed {:?}", t.elapsed());
    let info = m.info();
    assert_eq!(info.task, "embedding");
    assert_eq!(
        info.metadata["model_name"],
        json!(dir.file_name().unwrap().to_str())
    );

    // rlx-models' embedder on the same texts.
    let t = std::time::Instant::now();
    let mut reference = RlxBertModel::load(
        &dir.join("config.json"),
        dir.join("model.safetensors").to_str().unwrap(),
    )
    .unwrap();
    let tokenizer = BertTokenizer::from_dir(&dir, 64).unwrap();
    let want: Vec<Vec<f32>> = texts
        .iter()
        .flat_map(|t| embed_with_rlx(&mut reference, &tokenizer, &[t], Pooling::Mean).unwrap())
        .collect();
    eprintln!("rlx-models embedder: {:?}", t.elapsed());

    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert_eq!(a.len(), b.len());
        let cos: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let max = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        eprintln!("text {i}: cosine {cos:.7}, max |diff| {max:.2e}");
        assert!(cos > 0.99999, "text {i}: cosine {cos}");
    }
}
