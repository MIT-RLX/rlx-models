// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! The text front end vs `kittenml.kittentts2.text.normalize_text` (with
//! `kitten_text_processing` 0.1.0), both with and without written → spoken
//! normalization, on `tests/fixtures/normalize_cases.json`
//! (`scripts/kittentts2_normalize_fixture.py`). Exact string equality.

use rlx_kittentts2::text::normalize_text;

#[derive(serde::Deserialize)]
struct Case {
    input: String,
    normalized: String,
    raw: String,
}

#[test]
fn front_end_matches_upstream_exactly() {
    let cases: Vec<Case> =
        serde_json::from_str(include_str!("fixtures/normalize_cases.json")).unwrap();
    let mut bad = Vec::new();
    for c in &cases {
        let raw = normalize_text(&c.input, false);
        if raw != c.raw {
            bad.push(format!(
                "raw   {:?}\n   got  {raw:?}\n   want {:?}",
                c.input, c.raw
            ));
        }
        let got = normalize_text(&c.input, true);
        if got != c.normalized {
            bad.push(format!(
                "norm  {:?}\n   got  {got:?}\n   want {:?}",
                c.input, c.normalized
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {} mismatches:\n{}",
        bad.len(),
        2 * cases.len(),
        bad.join("\n")
    );
}
