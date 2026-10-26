// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! A Carbon package completes what `CarbonRunner::complete` does, greedy:
//! an open DNA prompt, a prompt with `<dna>` tags inside text, and plain
//! text — on Carbon-500M where it is on disk (`weights/lm/carbon-500m`, or
//! `RLX_CARBON_DIR`, with Qwen3's `tokenizer.json` beside it).

#![cfg(feature = "export")]

use std::path::{Path, PathBuf};

use rlx_carbon::CarbonRunner;
use rlx_carbon::export::ExportOptions;
use rlx_runtime::Device;
use serde_json::json;

fn dir() -> Option<PathBuf> {
    let p = rlx_ir::env::var("RLX_CARBON_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/lm/carbon-500m")
        });
    (p.join("model.safetensors").exists() && p.join("tokenizer.json").exists()).then_some(p)
}

#[test]
fn a_package_completes_what_carbon_does() {
    let Some(dir) = dir() else {
        eprintln!("no Carbon checkpoint with a tokenizer.json; skipped");
        return;
    };
    let prompts: [(&str, Option<bool>); 3] = [
        ("ATGGCGACCTTTAGCGATCTGCTGAAAGCG", Some(true)),
        ("The coding sequence <dna>ATGAAACGCATTAGC", None),
        ("The human genome contains", None),
    ];
    let n = 24;
    let mut wants = Vec::new();
    {
        let mut r = CarbonRunner::builder()
            .weights(&dir)
            .device(Device::Cpu)
            .build()
            .unwrap();
        for (p, as_dna) in &prompts {
            wants.push(r.complete(p, n, *as_dna).unwrap().text);
        }
    }
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("carbon.rlxp");
    let report = rlx_carbon::export::export_rlxp(&dir, &out, &ExportOptions::default()).unwrap();
    eprintln!("{report}");
    rlx_text::install_model_tokenizer();
    let mut m =
        rlx_runtime::model::open_source(out.to_str().unwrap(), json!({"entry": "complete"}))
            .unwrap();
    for ((p, as_dna), want) in prompts.iter().zip(&wants) {
        let mut opts = json!({"max_tokens": n});
        if let Some(d) = as_dna {
            opts["as_dna"] = json!(d);
        }
        let got = m.generate_text(p, &opts).unwrap();
        eprintln!("{p:?}: {got:?}");
        assert_eq!(&got, want, "{p:?}");
    }
}
