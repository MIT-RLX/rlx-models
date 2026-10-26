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

//! The rendered prompts against d1's own `prompt.py`, byte for byte, with
//! no weights or tokenizer: `fixtures/d1_reference.json` holds the text the
//! reference built for every case in `fixtures/d1_cases.json`
//! (`scripts/d1_reference.py`).
//!
//! The only tokenizer fact the renderer uses is whether an option code is a
//! single token; on LFM2's tokenizer every `A`..`Z` and `00`..`99` code is,
//! which the stand-in below reproduces.

use anyhow::Result;
use rlx_d1::prompt::{self, PromptStyle, Question, TextTokenizer};
use rlx_d1::{Json, answer};

struct ShortCodesAreOneToken;

impl TextTokenizer for ShortCodesAreOneToken {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let n = text.chars().count();
        Ok(if n <= 2 && !text.starts_with(' ') {
            vec![
                text.chars()
                    .fold(7u32, |h, c| h.wrapping_mul(131).wrapping_add(c as u32)),
            ]
        } else {
            text.chars().map(|c| c as u32).collect()
        })
    }
}

fn load(name: &str) -> Json {
    let text = match name {
        "cases" => include_str!("fixtures/d1_cases.json"),
        _ => include_str!("fixtures/d1_reference.json"),
    };
    Json::parse(text).expect("fixture parses")
}

fn d1_style() -> PromptStyle {
    PromptStyle {
        bos: "<|startoftext|>".into(),
        ..PromptStyle::default()
    }
}

#[test]
fn rendered_prompts_match_the_reference() {
    let cases = load("cases");
    let reference = load("reference");
    let style = d1_style();
    let tok = ShortCodesAreOneToken;
    let mut checked = 0;
    for (case, refc) in cases
        .as_array()
        .unwrap()
        .iter()
        .zip(reference.as_array().unwrap())
    {
        let name = case.get("name").unwrap().as_str().unwrap();
        assert_eq!(refc.get("name").unwrap().as_str(), Some(name));
        let state = case.get("state").unwrap();
        let n_images = case
            .get("images")
            .and_then(Json::as_array)
            .map_or(0, <[Json]>::len);
        let markup = "<image>".repeat(n_images);
        let questions = case.get("questions").unwrap().as_object().unwrap();
        for ((qname, q), row) in questions
            .iter()
            .zip(refc.get("rows").unwrap().as_array().unwrap())
        {
            assert_eq!(row.get("question").unwrap().as_str(), Some(qname.as_str()));
            let q = Question::from_json(q).unwrap();
            let got = prompt::render(&tok, state, &q, &style, &markup).unwrap();
            let want = row.get("text").unwrap().as_str().unwrap();
            assert_eq!(got, want, "{name}/{qname}");
            checked += 1;
        }
    }
    assert_eq!(checked, 15);
}

#[test]
fn answers_have_the_reference_shape() {
    // The reference's own probabilities through `answer` give back its
    // `system_one` answers: field names, order and values.
    let cases = load("cases");
    let reference = load("reference");
    for (case, refc) in cases
        .as_array()
        .unwrap()
        .iter()
        .zip(reference.as_array().unwrap())
    {
        let questions = case.get("questions").unwrap().as_object().unwrap();
        let official = refc.get("system_one").unwrap().get("answers").unwrap();
        for (qname, q) in questions {
            let q = Question::from_json(q).unwrap();
            let want = official.get(qname).unwrap();
            let probs: Vec<f64> = match &q {
                Question::Noul { .. } => {
                    let p = match want.get("noul").unwrap() {
                        Json::Float(p) => *p,
                        other => panic!("noul is {other:?}"),
                    };
                    vec![p, 1.0 - p]
                }
                _ => want
                    .get("probabilities")
                    .unwrap()
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(_, v)| match v {
                        Json::Float(p) => *p,
                        other => panic!("probability is {other:?}"),
                    })
                    .collect(),
            };
            let got = answer(&q, &probs);
            // Same keys in the same order; `score` recomputes a sum, so
            // floats are compared with a tolerance rather than bitwise.
            let (g, w) = (got.as_object().unwrap(), want.as_object().unwrap());
            let keys = |o: &[(String, Json)]| o.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>();
            assert_eq!(keys(g), keys(w), "{qname}");
            for ((k, gv), (_, wv)) in g.iter().zip(w) {
                match (gv, wv) {
                    (Json::Float(a), Json::Float(b)) => {
                        assert!((a - b).abs() < 1e-12, "{qname}.{k}")
                    }
                    _ => assert_eq!(
                        gv.to_python_json(None),
                        wv.to_python_json(None),
                        "{qname}.{k}"
                    ),
                }
            }
        }
    }
}

#[test]
fn option_codes_follow_the_reference_rule() {
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    assert_eq!(
        prompt::option_codes(&s(&["A", "b", " C "])),
        s(&["A", "b", "C"])
    );
    assert_eq!(prompt::option_codes(&s(&["yes", "no"])), s(&["A", "B"]));
    let many: Vec<String> = (0..27).map(|i| format!("o{i}")).collect();
    let codes = prompt::option_codes(&many);
    assert_eq!((codes[0].as_str(), codes[26].as_str()), ("00", "26"));
}
