//! Exact parity of `clean_tinytts_text` + `normalize_text` vs the Python
//! reference over a large numeric corpus (scripts/inflect_nano_frontend_fixtures.py).

use std::path::PathBuf;

use rlx_inflect_nano::frontend::{clean_tinytts_text, normalize_text};
use serde_json::Value;

fn corpus_path() -> Option<PathBuf> {
    let base = std::env::var("RLX_INFLECT_NANO_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/inflect-nano-rlx")
        });
    let p = base.join("fixtures/normalize_corpus.json");
    p.exists().then_some(p)
}

#[test]
fn normalize_matches_python() {
    let Some(path) = corpus_path() else {
        eprintln!("skip: normalize_corpus.json not found");
        return;
    };
    let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut fails = 0;
    let mut shown = 0;
    for case in &cases {
        let input = case["input"].as_str().unwrap();
        let exp_clean = case["cleaned"].as_str().unwrap();
        let exp_norm = case["normalized"].as_str().unwrap();
        let got_clean = clean_tinytts_text(input);
        let got_norm = normalize_text(&got_clean);
        if got_clean != exp_clean || got_norm != exp_norm {
            fails += 1;
            if shown < 25 {
                eprintln!(
                    "MISMATCH input={input:?}\n  clean exp={exp_clean:?} got={got_clean:?}\n  norm  exp={exp_norm:?} got={got_norm:?}"
                );
                shown += 1;
            }
        }
    }
    eprintln!(
        "normalize parity: {}/{} ok",
        cases.len() - fails,
        cases.len()
    );
    assert_eq!(fails, 0, "{fails} normalize mismatches");
}
