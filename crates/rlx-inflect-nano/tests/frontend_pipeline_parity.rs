//! Exact token-id parity of the full text frontend (clean → normalize → g2p →
//! ids → blanks) vs the Python reference over the pipeline corpus.

use std::path::PathBuf;

use rlx_inflect_nano::frontend::English;
use serde_json::Value;

fn data_dir() -> Option<PathBuf> {
    let base = std::env::var("RLX_INFLECT_NANO_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/inflect-nano-rlx")
        });
    base.join("config.json").exists().then_some(base)
}

fn ids(case: &Value, key: &str) -> Vec<i64> {
    case[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}

#[test]
fn frontend_pipeline_matches_python() {
    let Some(dir) = data_dir() else {
        eprintln!("skip: bundle not found");
        return;
    };
    let corpus_path = dir.join("fixtures/pipeline_corpus.json");
    if !corpus_path.exists() {
        eprintln!("skip: pipeline_corpus.json not found");
        return;
    }
    let english = English::load(&dir.join("frontend")).expect("load frontend");
    let cases: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(corpus_path).unwrap()).unwrap();

    let mut fails = 0;
    for case in &cases {
        let input = case["input"].as_str().unwrap();
        let (p, t, l) = english.text_to_ids(input, true).expect("text_to_ids");
        let (ep, et, el) = (
            ids(case, "phone_ids"),
            ids(case, "tone_ids"),
            ids(case, "lang_ids"),
        );
        if p != ep || t != et || l != el {
            fails += 1;
            eprintln!("MISMATCH {input:?}");
            if p != ep {
                eprintln!(
                    "  phone exp({}) {:?}\n        got({}) {:?}",
                    ep.len(),
                    ep,
                    p.len(),
                    p
                );
            }
            if t != et {
                eprintln!("  tone  exp {et:?}\n        got {t:?}");
            }
        }
    }
    eprintln!(
        "frontend pipeline parity: {}/{} ok",
        cases.len() - fails,
        cases.len()
    );
    assert_eq!(fails, 0, "{fails} pipeline mismatches");
}
