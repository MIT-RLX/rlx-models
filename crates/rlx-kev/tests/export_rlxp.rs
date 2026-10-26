// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! A kev package answers what `DecisionModel::answer` does: the example
//! ticket (a choice with a `null` description, a noul, a score with object
//! instructions, a buried delimiter), one question alone, and a state given
//! as an object — on `jaredpalmer/kev-0.8b` over `Qwen3.5-0.8B-Base` where
//! they are on disk (`KEV_RUN_DIR` and `KEV_BASE_DIR`, or
//! `/Volumes/FOUR/kev`).

#![cfg(feature = "export")]

use std::path::{Path, PathBuf};

use rlx_kev::export::ExportOptions;
use rlx_kev::{ShapePolicy, SystemOneRequest, to_record};
use rlx_runtime::Device;
use serde_json::json;

const TICKET: &str = include_str!("../examples/ticket.json");

fn dir(key: &str, default: &str) -> Option<PathBuf> {
    let p = rlx_ir::env::var(key)
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(default).to_path_buf());
    p.exists().then_some(p)
}

#[test]
fn a_package_answers_what_kev_does() {
    let (Some(run), Some(base)) = (
        dir("KEV_RUN_DIR", "/Volumes/FOUR/kev/kev-0.8b"),
        dir("KEV_BASE_DIR", "/Volumes/FOUR/kev/Qwen3.5-0.8B-Base"),
    ) else {
        eprintln!("no kev checkpoint; skipped");
        return;
    };
    let requests = [
        TICKET.to_string(),
        r#"{"state": "The invoice total is wrong.", "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}}}"#.to_string(),
        r#"{"state": {"customer": "Ann", "order": {"id": 7, "items": ["boots", "socks"]}, "note": "never arrived"}, "questions": {"lost": {"type": "noul", "instructions": "Is the package lost?"}, "item": {"type": "choice", "instructions": "Which item is it about?", "criteria": {"boots": null, "socks": "the socks", "both": null}}}}"#.to_string(),
    ];

    let mut wants = Vec::new();
    {
        let mut m = rlx_kev::load(&run, &base, Device::Cpu, ShapePolicy::default()).unwrap();
        for r in &requests {
            let req = SystemOneRequest::parse(r).unwrap();
            let (rec, _) = to_record(&req).unwrap();
            let probs = m.probs(&rec).unwrap().concat();
            let (answers, _) = m.answer(&req).unwrap();
            wants.push((probs, rlx_kev::api::answers_json(&answers)));
        }
    }

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("kev.rlxp");
    let report =
        rlx_kev::export::export_with_base(&run, &base, &out, &ExportOptions::default()).unwrap();
    eprintln!("{report}");
    rlx_text::install_model_tokenizer();
    let mut pkg = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    for (i, (r, (probs, answers))) in requests.iter().zip(&wants).enumerate() {
        let outs = pkg
            .run(&[], &json!({"request": r}))
            .unwrap_or_else(|e| panic!("request {i}: {e:#}"));
        let got = outs
            .iter()
            .find(|t| t.name == "probabilities")
            .unwrap()
            .to_f32()
            .unwrap();
        assert_eq!(got.len(), probs.len(), "request {i}");
        let worst = got
            .iter()
            .zip(probs)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let text = String::from_utf8(
            outs.iter()
                .find(|t| t.name == "decision")
                .unwrap()
                .data
                .clone(),
        )
        .unwrap();
        let body: serde_json::Value = serde_json::from_str(&text).unwrap();
        let want: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(answers).unwrap()).unwrap();
        eprintln!(
            "request {i}: {} probabilities within {worst:.2e}",
            got.len()
        );
        assert!(worst < 1e-4, "request {i}: off by {worst}");
        assert_eq!(body["answers"], want, "request {i}: the answers");
    }
}
