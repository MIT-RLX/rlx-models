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

//! The exporter on the rest of the Llama family (Llama, Mistral, Qwen2: no
//! QK-norm): the package's greedy tokens are the Hugging Face model's.
//! Opt-in: `RLX_TEST_LLAMA_DIR=<a Llama-architecture checkpoint directory>`
//! (e.g. SmolLM2-135M) and `RLX_TEST_LLAMA_REF=<file>`, the JSON ids
//! `transformers` greedily generates after "The capital of France is"
//! (`model.generate(ids, max_new_tokens=20, do_sample=False)`, f32).

use rlx_qwen3::export::{ExportOptions, export_rlxp};
use rlx_runtime::model::{Tensor, ids_of};
use serde_json::json;

#[test]
fn an_exported_llama_package_generates_what_transformers_does() {
    let (Some(dir), Some(reference)) = (
        rlx_ir::env::var_os("RLX_TEST_LLAMA_DIR"),
        rlx_ir::env::var_os("RLX_TEST_LLAMA_REF"),
    ) else {
        eprintln!("skip: set RLX_TEST_LLAMA_DIR and RLX_TEST_LLAMA_REF");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let want: Vec<u32> = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("llama.rlxp");
    export_rlxp(&dir, &out, &ExportOptions::default()).unwrap();

    let device = rlx_ir::env::var_result("RLX_TEST_DEVICE").unwrap_or_else(|_| "cpu".into());
    rlx_text::install_model_tokenizer();
    let mut m =
        rlx_runtime::model::open_source(out.to_str().unwrap(), json!({"device": device})).unwrap();
    let info = m.info();
    assert_eq!(info.family, config["model_type"].as_str().unwrap());

    let ids: Vec<u32> = rlx_text::load_tokenizer(&dir.join("tokenizer.json"))
        .unwrap()
        .encode("The capital of France is", false)
        .unwrap();
    let ids: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
    let input = Tensor::new(
        "input_ids",
        rlx_ir::DType::I32,
        vec![ids.len()],
        ids.iter().flat_map(|i| i.to_le_bytes()).collect(),
    );
    let opts = json!({"max_tokens": want.len(), "stop_ids": [], "temperature": 0});
    let got = ids_of(&m.run(&[input], &opts).unwrap()[0]).unwrap();
    assert_eq!(got, want, "greedy tokens differ from transformers'");

    // The batched path: the same prompt alone and beside another.
    let text = m.generate_text("The capital of France is", &opts).unwrap();
    let both = m
        .generate_batch(&["The capital of France is", "Water boils at"], &opts)
        .unwrap();
    assert_eq!(both[0], text);
    eprintln!("{text:?} | {:?}", both[1]);
}
