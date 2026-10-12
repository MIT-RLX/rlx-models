//! The recipe's **data pipeline** must be byte-identical to the reference.
//!
//! `tests/fixtures/recipe_golden.json` is the output of kev's own
//! `kev/data.py` — `augment`, `none_pair`, `materialize` — over the first 40
//! records of `evals/v7/decision-v7`'s training split, at seed 0, for epochs
//! 0 and 1. Comparing against it pins:
//!
//! * the CPython PRNG stream (option order, which none-option is drawn, which
//!   distractor);
//! * the branch thresholds — a tenth of multiple-choice questions have their
//!   *correct* option removed and replaced by "none of the above", so getting
//!   this wrong quietly trains a different task;
//! * `materialize`'s label encoding, which differs per question type (option
//!   name for `choice`, bool for `noul`, level index for `score`);
//! * the rendered option strings, i.e. that training and serving render the
//!   same prompt.
//!
//! Needs the suite, which is too large to commit:
//!
//! ```text
//! KEV_SUITE_DIR=/path/to/kev-suites/v7/decision-v7 \
//!   cargo test -p rlx-kev --features train --test recipe_parity
//! ```

use rlx_kev::pyrandom::PyRandom;
use rlx_kev::recipe::{SuiteLabel, augment, load_split, materialize, none_pair};

fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/recipe_golden.json")).expect("fixture parses")
}

fn suite_dir() -> Option<std::path::PathBuf> {
    let p = std::path::PathBuf::from(std::env::var_os("KEV_SUITE_DIR")?);
    p.join("train.jsonl").exists().then_some(p)
}

/// `criteria` names, in order — the thing the augmentation permutes.
fn criteria_names(q: &rlx_kev::api::Question) -> Option<Vec<String>> {
    match q {
        rlx_kev::api::Question::Choice { criteria, .. } => {
            Some(criteria.keys().map(str::to_string).collect())
        }
        _ => None,
    }
}

#[test]
fn augmentation_matches_the_reference_byte_for_byte() {
    let Some(dir) = suite_dir() else {
        eprintln!("skipping: set KEV_SUITE_DIR to a kev-suites suite directory");
        return;
    };
    let records = load_split(&dir.join("train.jsonl")).expect("suite loads");
    let fx = fixture();
    let cases = fx["cases"].as_array().expect("cases");
    assert!(!cases.is_empty(), "empty fixture");

    let mut checked = 0usize;
    let mut none_swaps = 0usize;
    for case in cases {
        let epoch = case["epoch"].as_u64().expect("epoch");
        let id = case["id"].as_str().expect("id");
        let rec = records
            .iter()
            .find(|r| r.id == id)
            .unwrap_or_else(|| panic!("suite has no record {id}"));

        let mut rng = PyRandom::from_source_seed(0, &format!("{epoch}:{id}"));
        let aug = augment(rec, &mut rng, 0.1, 0.12, 0.15).expect("augment");
        // `none_pair` draws from the SAME generator, after `augment` — so this
        // also checks that `augment` consumed exactly the right number of words.
        let pair = none_pair(rec, &mut rng).expect("none_pair");

        // ── augmented criteria order + label
        let want_aug = case["augmented"].as_object().expect("augmented");
        for q in &aug.questions {
            let w = &want_aug[&q.id];
            match criteria_names(&q.question) {
                Some(got) => {
                    let want: Vec<String> = w["criteria"]
                        .as_array()
                        .expect("criteria")
                        .iter()
                        .map(|x| x.as_str().expect("str").to_string())
                        .collect();
                    assert_eq!(
                        got, want,
                        "record {id} epoch {epoch} question {}: option order/content differs",
                        q.id
                    );
                    let want_label = w["label"].as_str().expect("label");
                    assert_eq!(
                        q.label,
                        SuiteLabel::Name(want_label.to_string()),
                        "record {id} epoch {epoch} question {}: label differs",
                        q.id
                    );
                    if !want.contains(&want_label.to_string()) {
                        panic!("fixture label not among its own options");
                    }
                }
                None => assert!(
                    w["criteria"].is_null(),
                    "non-choice question {} should have no criteria in the fixture",
                    q.id
                ),
            }
        }

        // ── materialized integer labels and rendered options
        let m = materialize(&aug).expect("materialize");
        let want_labels: Vec<usize> = case["labels"]
            .as_array()
            .expect("labels")
            .iter()
            .map(|x| x.as_u64().expect("u64") as usize)
            .collect();
        let got_labels: Vec<usize> = m.record.questions.iter().map(|q| q.label).collect();
        assert_eq!(
            got_labels, want_labels,
            "record {id} epoch {epoch}: materialized labels differ"
        );

        let want_options = case["options"].as_array().expect("options");
        for (qi, q) in m.record.questions.iter().enumerate() {
            let want: Vec<String> = want_options[qi]
                .as_array()
                .expect("options")
                .iter()
                .map(|x| x.as_str().expect("str").to_string())
                .collect();
            assert_eq!(
                q.options, want,
                "record {id} epoch {epoch} question {qi}: rendered options differ"
            );
        }

        // ── the none-of-the-above minimal pair
        let want_pair = case["pair"].as_array().expect("pair");
        assert_eq!(
            pair.len(),
            want_pair.len(),
            "record {id} epoch {epoch}: none_pair produced {} records, want {}",
            pair.len(),
            want_pair.len()
        );
        for (p, wp) in pair.iter().zip(want_pair) {
            let wq = wp.as_object().expect("pair question");
            for q in &p.questions {
                let w = &wq[&q.id];
                let got = criteria_names(&q.question).expect("pair members are choice");
                let want: Vec<String> = w["criteria"]
                    .as_array()
                    .expect("criteria")
                    .iter()
                    .map(|x| x.as_str().expect("str").to_string())
                    .collect();
                assert_eq!(
                    got, want,
                    "record {id} epoch {epoch}: none_pair order differs"
                );
                assert_eq!(
                    q.label,
                    SuiteLabel::Name(w["label"].as_str().expect("label").to_string()),
                    "record {id} epoch {epoch}: none_pair label differs"
                );
            }
        }

        // Count the swap branch so the test cannot pass by never exercising it.
        for (q0, q1) in rec.questions.iter().zip(&aug.questions) {
            if let (SuiteLabel::Name(a), SuiteLabel::Name(b)) = (&q0.label, &q1.label)
                && a != b
            {
                none_swaps += 1;
            }
        }
        checked += 1;
    }
    assert!(checked >= 40, "only {checked} cases checked");
    assert!(
        none_swaps > 0,
        "no record had its correct option swapped for a none-option — the \
         p_none branch was never exercised, so this test proved little"
    );
    eprintln!("{checked} cases matched; {none_swaps} none-option swaps exercised");
}
