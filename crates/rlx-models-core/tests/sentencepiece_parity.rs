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

//! `rlx_core::sentencepiece` vs the C++ SentencePiece: ids and decoded text
//! for every reference `scripts/sentencepiece_reference.py` wrote into
//! `RLX_SPM_REFERENCE_DIR`. Skips without it.

use rlx_models_core::sentencepiece::SentencePiece;

#[test]
fn matches_cpp_sentencepiece() {
    let Some(dir) = rlx_ir::env::var("RLX_SPM_REFERENCE_DIR") else {
        eprintln!("RLX_SPM_REFERENCE_DIR unset: skipping");
        return;
    };
    let mut refs: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    refs.sort();
    assert!(!refs.is_empty(), "no references in {dir}");
    let mut failures = Vec::new();
    let mut total = 0;
    for r in &refs {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(r).unwrap()).unwrap();
        let model = std::path::Path::new(v["model"].as_str().unwrap());
        let sp = SentencePiece::open(model).unwrap();
        let json = Some(model.with_file_name("tokenizer.json"))
            .filter(|p| p.is_file())
            .map(|p| SentencePiece::open(p).unwrap());
        for c in v["cases"].as_array().unwrap() {
            total += 1;
            let text = c["text"].as_str();
            let want: Vec<u32> = c["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|i| i.as_u64().unwrap() as u32)
                .collect();
            let name = r.file_name().unwrap().to_string_lossy();
            if let Some(text) = text {
                let got = sp.encode(text);
                if got != want {
                    failures.push(format!(
                        "{name} encode {text:?}\n  want {want:?}\n  got  {got:?}"
                    ));
                    continue;
                }
                // A `tokenizer.json` beside the model must segment the same.
                if let Some(json) = &json
                    && json.encode(text) != want
                {
                    failures.push(format!(
                        "{name} tokenizer.json encode {text:?}\n  want {want:?}\n  got  {:?}",
                        json.encode(text)
                    ));
                }
            }
            let decoded = sp.decode(&want).unwrap();
            if decoded != c["decoded"].as_str().unwrap() {
                failures.push(format!(
                    "{name} decode {text:?}\n  want {:?}\n  got  {decoded:?}",
                    c["decoded"].as_str().unwrap()
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {total} cases differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
    eprintln!("{total} cases over {} models match", refs.len());
}
