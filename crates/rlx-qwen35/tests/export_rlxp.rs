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

//! An exported Qwen3.5 package — recurrent state and KV caches, packed
//! K-quant weights read from the GGUF — opened by rlx alone, generates the
//! tokens `Qwen35Runner` does. Opt-in: `RLX_TEST_QWEN35_GGUF=<a Qwen3.5
//! GGUF>`; `RLX_TEST_DEVICE` (`cpu` by default).

use rlx_qwen35::Qwen35Runner;
use rlx_qwen35::export::{ExportOptions, export_rlxp};
use rlx_runtime::model::{ChatMessage, Tensor, ids_of};
use serde_json::json;

#[test]
fn an_exported_package_generates_what_qwen35_runner_does() {
    let Some(gguf) = std::env::var_os("RLX_TEST_QWEN35_GGUF") else {
        eprintln!("skip: RLX_TEST_QWEN35_GGUF is not set");
        return;
    };
    let gguf = std::path::PathBuf::from(gguf);
    let device = std::env::var("RLX_TEST_DEVICE").unwrap_or_else(|_| "cpu".into());
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("qwen35.rlxp");
    let opts = ExportOptions {
        uppers: vec![32, 128],
        // `RLX_TEST_NO_SYMBOLIC=1`: one graph per capacity.
        symbolic: std::env::var_os("RLX_TEST_NO_SYMBOLIC").is_none(),
        ..ExportOptions::default()
    };
    let t = std::time::Instant::now();
    let report = export_rlxp(&gguf, &out, &opts).unwrap();
    eprintln!(
        "export {:?}: {report:?}, package {} bytes",
        t.elapsed(),
        std::fs::metadata(&out).unwrap().len()
    );

    rlx_text::install_model_tokenizer();
    let mut m =
        rlx_runtime::model::open_source(out.to_str().unwrap(), json!({"device": device})).unwrap();
    assert_eq!(m.info().task, "text-generation");

    let prompt = "The capital of France is";
    let ids = rlx_qwen35::encode_prompt_from_gguf(&gguf, prompt).unwrap();
    let n = 40;
    let input = Tensor::new(
        "input_ids",
        rlx_ir::DType::I32,
        vec![ids.len()],
        ids.iter().flat_map(|&i| (i as i32).to_le_bytes()).collect(),
    );
    let got = ids_of(&m.run(&[input], &json!({"max_tokens": n})).unwrap()[0]).unwrap();
    let text = m
        .chat(
            &[ChatMessage::user(
                "What is the capital of France? Answer in one word.",
            )],
            &json!({"max_tokens": 64}),
        )
        .unwrap();
    eprintln!("chat: {text:?}");
    drop(m);

    let mut runner = Qwen35Runner::builder()
        .weights(&gguf)
        .device(rlx_runtime::parse_device(&device).unwrap())
        .max_seq(256)
        .build()
        .unwrap();
    let want = runner.generate(&ids, n, |_| true).unwrap();
    eprintln!("package ids: {got:?}\nrunner ids:  {want:?}");
    assert_eq!(got, want[..got.len()], "greedy tokens differ");
    assert!(text.contains("Paris"), "{text}");
}
