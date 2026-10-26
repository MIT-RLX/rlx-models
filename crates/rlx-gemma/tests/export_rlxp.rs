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

//! An exported Gemma package generates what `transformers` does. Opt-in:
//! `RLX_TEST_GEMMA_DIR=<a Gemma 3 text checkpoint directory>` and
//! `RLX_TEST_GEMMA_REF=<file>`: JSON `{prompt, tokens}`, the prompt's ids
//! (with BOS) and the 20 tokens `model.generate(ids, max_new_tokens=20,
//! do_sample=False)` gives in f32. `RLX_TEST_GEMMA_LONG` adds a prompt
//! past the sliding window.

use rlx_gemma::export::export_rlxp;
use rlx_qwen3::export::ExportOptions;
use rlx_runtime::model::{Tensor, ids_of};
use serde_json::{Value, json};

#[test]
fn an_exported_gemma_package_generates_what_transformers_does() {
    let (Some(dir), Some(reference)) = (
        rlx_ir::env::var_os("RLX_TEST_GEMMA_DIR"),
        rlx_ir::env::var_os("RLX_TEST_GEMMA_REF"),
    ) else {
        eprintln!("skip: set RLX_TEST_GEMMA_DIR and RLX_TEST_GEMMA_REF");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let reference: Value = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let list = |k: &str| -> Vec<u32> {
        reference[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect()
    };
    let (prompt, want) = (list("prompt"), list("tokens"));
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("gemma.rlxp");
    // `RLX_TEST_NO_PREFILL=1`: prompts through the step a token at a time.
    let opts = ExportOptions {
        prefill: if std::env::var_os("RLX_TEST_NO_PREFILL").is_some() {
            Vec::new()
        } else {
            ExportOptions::default().prefill
        },
        ..ExportOptions::default()
    };
    let report = export_rlxp(&dir, &out, &opts).unwrap();
    eprintln!("{report:?}");
    assert!(report.stateful_failure.is_none() && report.prefill_failure.is_none());

    let device = rlx_ir::env::var_result("RLX_TEST_DEVICE").unwrap_or_else(|_| "cpu".into());
    rlx_text::install_model_tokenizer();
    let mut m =
        rlx_runtime::model::open_source(out.to_str().unwrap(), json!({"device": device})).unwrap();
    let ids: Vec<i32> = prompt.iter().map(|&i| i as i32).collect();
    let input = Tensor::new(
        "input_ids",
        rlx_ir::DType::I32,
        vec![ids.len()],
        ids.iter().flat_map(|i| i.to_le_bytes()).collect(),
    );
    let opts = json!({"max_tokens": want.len(), "stop_ids": [], "temperature": 0});
    let got = ids_of(&m.run(&[input], &opts).unwrap()[0]).unwrap();
    assert_eq!(got, want, "greedy tokens differ from transformers'");

    // Its chat template, and two prompts decoded together.
    let reply = m
        .chat(
            &[rlx_runtime::model::ChatMessage::new(
                "user",
                "What is the capital of France? One word.",
            )],
            &json!({"max_tokens": 8, "temperature": 0}),
        )
        .unwrap();
    eprintln!("chat: {reply:?}");
    assert!(reply.contains("Paris"), "{reply}");
    let both = m
        .generate_batch(&["The capital of France is", "Water boils at"], &opts)
        .unwrap();
    let alone = m.generate_text("Water boils at", &opts).unwrap();
    assert_eq!(both[1], alone);

    // `RLX_TEST_GEMMA_LONG=<file>`: a prompt past the sliding window
    // (JSON `{prompt, tokens, logits}`: its ids, `transformers`' 20 greedy
    // tokens and last-position logits). The sliding layers see only their
    // window, the full ones everything.
    if let Some(long) = rlx_ir::env::var_os("RLX_TEST_GEMMA_LONG") {
        let long: Value = serde_json::from_slice(&std::fs::read(long).unwrap()).unwrap();
        let ids: Vec<i32> = long["prompt"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let input = || {
            Tensor::new(
                "input_ids",
                rlx_ir::DType::I32,
                vec![ids.len()],
                ids.iter().flat_map(|i| i.to_le_bytes()).collect(),
            )
        };
        let want: Vec<f32> = long["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let got = m.run(&[input()], &json!({})).unwrap()[0].to_f32().unwrap();
        let diff = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "{} tokens: last logits within {diff} of transformers'",
            ids.len()
        );
        assert!(diff < 1e-2, "{diff}");
        let tokens: Vec<u32> = long["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = ids_of(&m.run(&[input()], &opts).unwrap()[0]).unwrap();
        assert_eq!(got, tokens, "greedy tokens past the window differ");
    }
}
