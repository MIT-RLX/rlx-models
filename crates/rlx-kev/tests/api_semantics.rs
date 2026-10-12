//! `render`, `option_text`, the confidence formulas and `r2`.
//!
//! Every expected value here was produced by running the corresponding
//! function from `kev/api.py`, not by reading it. `render` is a *text
//! protocol*: whatever it emits becomes the prompt, so a divergence is
//! invisible to every shape check and silently changes what the model is
//! asked.
//!
//! Requests are parsed from JSON **text**, never built through
//! `serde_json::Value` — the latter sorts object keys, which reorders both
//! the questions and the option lists inside them.

use rlx_kev::api::{
    Answer, SystemOneRequest, answers_json, argmax, choice_confidence, option_text, r2,
    score_confidence, to_answers, to_record,
};
use rlx_kev::json::{Json, render};

fn rendered(text: &str) -> String {
    render(&Json::parse(text).expect("parses"), 0)
}

fn value(text: &str) -> Json {
    Json::parse(text).expect("parses")
}

fn request(text: &str) -> SystemOneRequest {
    SystemOneRequest::parse(text).expect("parses")
}

#[test]
fn scalars_print_like_python_str() {
    assert_eq!(rendered(r#""hi""#), "hi");
    assert_eq!(rendered("3"), "3");
    // JSON `3.0` is a float in Python and keeps the point; an integer does
    // not. The parse preserves the same distinction.
    assert_eq!(rendered("3.0"), "3.0");
    assert_eq!(rendered("-2.5"), "-2.5");
    assert_eq!(rendered("true"), "True");
    assert_eq!(rendered("false"), "False");
    assert_eq!(rendered("null"), "");
}

#[test]
fn containers_match_the_reference_layout() {
    assert_eq!(rendered(r#"["a", "b"]"#), "- a\n- b");
    assert_eq!(
        rendered(r#"{"k": "v", "n": {"a": 1, "b": [1, 2]}}"#),
        "k: v\nn:\n  a: 1\n  b:\n    - 1\n    - 2"
    );
    assert_eq!(rendered(r#"[{"a": 1}, "plain"]"#), "- a: 1\n- plain");
}

/// Python dicts render in insertion order, so a state object's field order is
/// part of the prompt. Sorting it would be a silent prompt change.
#[test]
fn object_fields_render_in_document_order_not_sorted() {
    assert_eq!(rendered(r#"{"z": 1, "a": 2}"#), "z: 1\na: 2");
}

#[test]
fn option_text_treats_empty_and_null_descriptions_as_absent() {
    assert_eq!(option_text("returns", None), "returns");
    assert_eq!(option_text("returns", Some(&value("null"))), "returns");
    assert_eq!(option_text("returns", Some(&value(r#""""#))), "returns");
    assert_eq!(
        option_text("returns", Some(&value(r#""Refunds""#))),
        "returns: Refunds"
    );
    assert_eq!(option_text("x", Some(&value(r#"{"a": 1}"#))), "x: a: 1");
}

#[test]
fn confidence_formulas_match_the_reference() {
    let c = choice_confidence(&[0.47, 0.28, 0.25]);
    assert!(
        (c as f64 - 0.20499999999999996).abs() < 1e-6,
        "choice confidence was {c}"
    );
    assert_eq!(choice_confidence(&[1.0]), 1.0);

    let s = score_confidence(&[0.0, 0.56, 0.44]);
    assert!((s as f64 - 0.78).abs() < 1e-6, "score confidence was {s}");
}

/// Python rounds the exact binary double, ties to even. Values arrive as f32
/// widened to f64 (`tensor.tolist()` in the reference), so the expectations
/// below are `round(float(numpy.float32(v)), 2)` — which is *not* the same as
/// rounding the f64 literal: `round(0.005, 2)` is `0.01`, but the nearest f32
/// to 0.005 is below the tie and rounds to `0.0`.
#[test]
fn r2_matches_python_round_on_f32_widened_values() {
    assert_eq!(r2(0.125), 0.12); // exact tie -> even
    assert_eq!(r2(0.135), 0.14); // f32 widening is above the tie
    assert_eq!(r2(0.005), 0.0); // f32 widening is below the tie
    assert_eq!(r2(0.47), 0.47);
    assert_eq!(r2(1.0 / 3.0), 0.33);
    assert_eq!(r2(0.9449999), 0.94);
}

/// Ties go to the lowest index, as Python's `max(range(n), key=…)` does.
/// An MLX `TopK` breaking ties toward the largest index has produced wrong
/// answers in this codebase before, so the direction is pinned.
#[test]
fn argmax_breaks_ties_toward_the_lowest_index() {
    assert_eq!(argmax(&[0.5, 0.5, 0.0]), 0);
    assert_eq!(argmax(&[0.1, 0.9, 0.9]), 1);
    assert_eq!(argmax(&[1.0]), 0);
}

#[test]
fn noul_reports_the_probability_of_yes_with_options_ordered_no_then_yes() {
    let req = request(
        r#"{"state": "I was charged twice.",
            "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}}}"#,
    );
    let (rec, meta) = to_record(&req).expect("record");
    assert_eq!(rec.questions[0].options, vec!["no", "yes"]);
    assert_eq!(meta[0].keys, vec!["false", "true"]);

    let answers = to_answers(&[vec![0.07, 0.93]], &meta).expect("answers");
    assert_eq!(answers[0].0, "billing");
    assert_eq!(answers[0].1, Answer::Noul { noul: 0.93 });
}

/// The regression this whole `Json` type exists for: `billing` is last in the
/// document and must stay last, even though it sorts first.
#[test]
fn choice_keeps_request_order_and_reports_every_option() {
    let req = request(
        r#"{
            "state": "Shoes arrived late and in the wrong size.",
            "questions": {
              "department": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {
                  "returns": "Exchanges, refunds",
                  "shipping": "Delivery status",
                  "billing": null
                }
              }
            }
        }"#,
    );
    let (rec, meta) = to_record(&req).expect("record");
    assert_eq!(
        rec.questions[0].options,
        vec![
            "returns: Exchanges, refunds",
            "shipping: Delivery status",
            "billing"
        ],
        "criteria order is part of the prompt and must survive parsing"
    );
    assert_eq!(meta[0].keys, vec!["returns", "shipping", "billing"]);

    let answers = to_answers(&[vec![0.47, 0.28, 0.25]], &meta).expect("answers");
    match &answers[0].1 {
        Answer::Choice {
            choice,
            confidence,
            probabilities,
        } => {
            assert_eq!(choice, "returns");
            assert_eq!(*confidence, 0.2);
            assert_eq!(
                probabilities,
                &vec![
                    ("returns".to_string(), 0.47),
                    ("shipping".to_string(), 0.28),
                    ("billing".to_string(), 0.25),
                ]
            );
        }
        other => panic!("expected a choice answer, got {other:?}"),
    }
}

/// Questions are answered in request order too.
#[test]
fn questions_keep_request_order() {
    let req = request(
        r#"{"state": "s", "questions": {
             "zebra": {"type": "noul", "instructions": "z"},
             "apple": {"type": "noul", "instructions": "a"}}}"#,
    );
    let (_rec, meta) = to_record(&req).expect("record");
    assert_eq!(
        meta.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
        vec!["zebra", "apple"]
    );
}

#[test]
fn score_reports_the_expected_level_and_a_legend() {
    let req = request(
        r#"{"state": "Where are my shoes?!",
            "questions": {"frustration": {
              "type": "score",
              "instructions": "How frustrated?",
              "criteria": ["Calm", "Frustrated", "Very angry"]}}}"#,
    );
    let (_rec, meta) = to_record(&req).expect("record");
    let answers = to_answers(&[vec![0.0, 0.56, 0.44]], &meta).expect("answers");
    match &answers[0].1 {
        Answer::Score {
            score,
            confidence,
            legend,
            probabilities,
        } => {
            assert_eq!(*score, 1.44);
            assert_eq!(*confidence, 0.78);
            assert_eq!(legend[2], ("2".to_string(), "Very angry".to_string()));
            assert_eq!(probabilities[1], ("1".to_string(), 0.56));
        }
        other => panic!("expected a score answer, got {other:?}"),
    }
}

/// Rounded probabilities must serialize as `0.93`, not as the f32 widening
/// `0.9300000071525574`, and answers must stay in request order.
#[test]
fn answers_serialize_as_short_decimals_in_order() {
    let req = request(
        r#"{"state": "s", "questions": {
             "zebra": {"type": "noul", "instructions": "z"},
             "apple": {"type": "noul", "instructions": "a"}}}"#,
    );
    let (_rec, meta) = to_record(&req).expect("record");
    let answers = to_answers(&[vec![0.07, 0.93], vec![0.5, 0.5]], &meta).expect("answers");
    let text = serde_json::to_string(&answers_json(&answers)).expect("json");
    assert!(text.contains(r#""noul":0.93"#), "got {text}");
    assert!(
        text.find("zebra").expect("present") < text.find("apple").expect("present"),
        "answers must serialize in request order, got {text}"
    );
}

#[test]
fn malformed_requests_are_rejected() {
    let empty_criteria = request(
        r#"{"state": "s", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {}}}}"#,
    );
    assert!(
        to_record(&empty_criteria).is_err(),
        "empty criteria must be rejected"
    );

    let one_level = request(
        r#"{"state": "s", "questions": {"q": {"type": "score", "instructions": "i", "criteria": ["only"]}}}"#,
    );
    assert!(
        to_record(&one_level).is_err(),
        "a 1-level score must be rejected"
    );

    let no_questions = request(r#"{"state": "s", "questions": {}}"#);
    assert!(
        to_record(&no_questions).is_err(),
        "an empty question set must be rejected"
    );

    assert!(
        SystemOneRequest::parse(r#"{"state": "s", "questions": {"q": {"type": "nope"}}}"#).is_err(),
        "an unknown question type must be rejected"
    );
}
