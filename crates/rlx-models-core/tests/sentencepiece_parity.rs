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

//! `rlx_core::sentencepiece` vs the C++ SentencePiece, for every reference
//! `scripts/sentencepiece_reference.py` wrote into `RLX_SPM_REFERENCE_DIR`:
//! ids, pieces with surfaces and spans, extra options, normalization with
//! offsets, n-best, decoding with spans, and sampling distributions. Skips
//! without the variable.

use std::collections::{BTreeSet, HashMap};

use rlx_models_core::sentencepiece::{EncodeOptions, EncodedPiece, SentencePiece, SplitMix64};
use serde_json::Value;

const SAMPLE_DRAWS: usize = 40_000;

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|i| i.as_u64().unwrap() as u32)
        .collect()
}

fn spans(v: &Value) -> Vec<(String, u32, String, usize, usize)> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| {
            (
                p[0].as_str().unwrap().to_string(),
                p[1].as_u64().unwrap() as u32,
                p[2].as_str().unwrap().to_string(),
                p[3].as_u64().unwrap() as usize,
                p[4].as_u64().unwrap() as usize,
            )
        })
        .collect()
}

fn ours(pieces: &[EncodedPiece]) -> Vec<(String, u32, String, usize, usize)> {
    pieces
        .iter()
        .map(|p| (p.piece.clone(), p.id, p.surface.clone(), p.begin, p.end))
        .collect()
}

/// Total variation between `want` and `got`, outcomes under 1 % of `want`
/// pooled into one bucket so sampling noise stays small.
fn total_variation(want: &HashMap<String, f64>, got: &HashMap<String, f64>) -> f64 {
    let keys: BTreeSet<&String> = want.keys().chain(got.keys()).collect();
    let (mut tv, mut rest_w, mut rest_g) = (0.0, 0.0, 0.0);
    for k in keys {
        let (w, g) = (
            want.get(k).copied().unwrap_or(0.0),
            got.get(k).copied().unwrap_or(0.0),
        );
        if w >= 0.01 {
            tv += (w - g).abs();
        } else {
            rest_w += w;
            rest_g += g;
        }
    }
    0.5 * (tv + (rest_w - rest_g).abs())
}

struct Check {
    failures: Vec<String>,
    total: usize,
}

impl Check {
    fn eq<T: PartialEq + std::fmt::Debug>(
        &mut self,
        what: impl FnOnce() -> String,
        want: T,
        got: T,
    ) {
        self.total += 1;
        if want != got {
            self.failures
                .push(format!("{}\n  want {want:?}\n  got  {got:?}", what()));
        }
    }
}

fn check_model(c: &mut Check, name: &str, v: &Value) {
    let model = std::path::Path::new(v["model"].as_str().unwrap());
    let sp = SentencePiece::open(model).unwrap();
    // A SentencePiece `tokenizer.json` (Unigram) beside the model must
    // segment the same.
    let json = Some(model.with_file_name("tokenizer.json"))
        .filter(|p| {
            std::fs::read_to_string(p)
                .ok()
                .and_then(|t| serde_json::from_str::<Value>(&t).ok())
                .is_some_and(|t| t["model"]["type"] == "Unigram")
        })
        .map(|p| SentencePiece::open(&p).unwrap_or_else(|e| panic!("{}: {e:#}", p.display())));
    // Older references carry only `cases` with ids and decoded text.
    let section = |k: &str| {
        v.get(k)
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    };

    for case in &section("cases") {
        let text = case["text"].as_str().unwrap();
        let e = sp.encode_full(text);
        c.eq(
            || format!("{name} encode {text:?}"),
            ids(&case["ids"]),
            e.ids(),
        );
        if case.get("pieces").is_some() {
            c.eq(
                || format!("{name} pieces {text:?}"),
                spans(&case["pieces"]),
                ours(&e.pieces),
            );
        }
        c.eq(
            || format!("{name} decode {text:?}"),
            case["decoded"].as_str().unwrap().to_string(),
            sp.decode(&ids(&case["ids"])).unwrap(),
        );
        if let Some(json) = &json {
            c.eq(
                || format!("{name} tokenizer.json {text:?}"),
                ids(&case["ids"]),
                json.encode(text),
            );
        }
    }

    for n in &section("normalize") {
        let text = n["text"].as_str().unwrap();
        let (norm, n2o) = sp.normalize_with_offsets(text);
        // The binding reports offsets in characters: one per normalized
        // character plus the end.
        let char_of = |byte: usize| text[..byte].chars().count();
        let mut offs: Vec<usize> = norm.char_indices().map(|(b, _)| char_of(n2o[b])).collect();
        offs.push(char_of(n2o[norm.len()]));
        c.eq(
            || format!("{name} normalize {text:?}"),
            (
                n["normalized"].as_str().unwrap().to_string(),
                ids(&n["offsets"]),
            ),
            (norm, offs.into_iter().map(|o| o as u32).collect()),
        );
    }

    for x in &section("extra") {
        let text = x["text"].as_str().unwrap();
        let o = &x["opts"];
        let flag = |k: &str| o[k].as_bool().unwrap_or(false);
        let opts = EncodeOptions {
            add_bos: flag("add_bos"),
            add_eos: flag("add_eos"),
            reverse: flag("reverse"),
            emit_unk_piece: flag("emit_unk_piece"),
        };
        let got = sp.encode_with(text, &opts);
        if x.get("error").is_some() {
            c.eq(
                || format!("{name} {o} {text:?} is an error"),
                true,
                got.is_err(),
            );
            continue;
        }
        let got = got.unwrap();
        let want_pieces: Vec<String> = x["pieces"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap().to_string())
            .collect();
        c.eq(
            || format!("{name} {o} {text:?}"),
            (ids(&x["ids"]), want_pieces),
            (
                got.ids(),
                got.pieces.iter().map(|p| p.piece.clone()).collect(),
            ),
        );
    }

    for nb in &section("nbest") {
        let text = nb["text"].as_str().unwrap();
        let got = sp
            .nbest_encode(text, nb["n"].as_u64().unwrap() as usize)
            .unwrap();
        let want: Vec<(Vec<u32>, f32)> = nb["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| (ids(&r["ids"]), r["score"].as_f64().unwrap() as f32))
            .collect();
        c.eq(
            || format!("{name} nbest count {text:?}"),
            want.len(),
            got.len(),
        );
        // Equal scores may come out in either order: compare score by rank
        // and paths as sets within a run of equal scores.
        let close = |a: f32, b: f32| (a - b).abs() <= 1e-4 * a.abs().max(1.0);
        let mut i = 0;
        while i < want.len().min(got.len()) {
            let mut j = i + 1;
            while j < want.len() && close(want[j].1, want[i].1) {
                j += 1;
            }
            let j = j.min(got.len());
            for k in i..j {
                c.eq(
                    || format!("{name} nbest score {text:?} #{k}"),
                    true,
                    close(want[k].1, got[k].score),
                );
            }
            // A run cut by the list's end may hold different members.
            if j < want.len() {
                let w: BTreeSet<_> = want[i..j].iter().map(|x| x.0.clone()).collect();
                let g: BTreeSet<_> = got[i..j].iter().map(|x| x.ids()).collect();
                c.eq(|| format!("{name} nbest paths {text:?} #{i}..{j}"), w, g);
            }
            i = j;
        }
    }

    for d in &section("decode") {
        let id_list = ids(&d["ids"]);
        let got = sp.decode_full(&id_list).unwrap();
        c.eq(
            || format!("{name} decode_full {id_list:?}"),
            (d["text"].as_str().unwrap().to_string(), spans(&d["pieces"])),
            (got.text, ours(&got.pieces)),
        );
    }

    let mut rng = SplitMix64(0x5eed);
    for s in &section("sample") {
        let text = s["text"].as_str().unwrap();
        let nbest = s["nbest_size"].as_i64().unwrap() as i32;
        let alpha = s["alpha"].as_f64().unwrap() as f32;
        if s.get("error").is_some() {
            let got = sp.sample_encode(text, nbest, alpha, &mut || rng.uniform());
            c.eq(
                || format!("{name} sample {text:?} is an error"),
                true,
                got.is_err(),
            );
            continue;
        }
        let want: HashMap<String, f64> = s["dist"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, p)| {
                let k: Vec<u32> = serde_json::from_str(k).unwrap();
                (format!("{k:?}"), p.as_f64().unwrap())
            })
            .collect();
        let mut counts: HashMap<String, f64> = HashMap::new();
        for _ in 0..SAMPLE_DRAWS {
            let e = sp
                .sample_encode(text, nbest, alpha, &mut || rng.uniform())
                .unwrap();
            *counts.entry(format!("{:?}", e.ids())).or_default() += 1.0 / SAMPLE_DRAWS as f64;
        }
        let tv = total_variation(&want, &counts);
        // Exact references carry only our sampling noise; empirical ones
        // (BPE-dropout, 20000 C++ draws) carry theirs too.
        let limit = if s["exact"].as_bool().unwrap() {
            0.03
        } else {
            0.05
        };
        c.eq(
            || {
                format!(
                    "{name} sample {text:?} nbest {nbest} alpha {alpha}: total variation {tv:.4}"
                )
            },
            true,
            tv <= limit,
        );
    }
}

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
    let mut c = Check {
        failures: Vec::new(),
        total: 0,
    };
    for r in &refs {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(r).unwrap()).unwrap();
        let name = r.file_stem().unwrap().to_string_lossy().into_owned();
        check_model(&mut c, &name, &v);
    }
    assert!(
        c.failures.is_empty(),
        "{} of {} checks differ:\n{}",
        c.failures.len(),
        c.total,
        c.failures.join("\n")
    );
    eprintln!("{} checks over {} models match", c.total, refs.len());
}
