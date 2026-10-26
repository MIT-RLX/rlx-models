//! Parity with Perplexity's own `pplx_decider` package.
//!
//! Fixtures come from `scripts/pplx_decider_reference.py` over
//! `tests/fixtures/requests.json`:
//!
//! * `ref_f32.json` — the release's bf16 safetensors, upcast to f32;
//! * `ref_q8_0.json` — the same code with every backbone tensor and the
//!   readout replaced by the dequantized Q8_0 GGUF. Quantization error is
//!   shared with what this crate runs, so what remains is implementation.
//!
//! The prompt check needs no weights. The rest load the GGUF named by
//! `PPLX_DECIDER_GGUF` (default `weights/lm/pplx-decider-flash-gguf`) on
//! `PPLX_DECIDER_DEVICE` (default `cpu`) and are skipped when it is absent.

use std::path::PathBuf;

use rlx_pplx_decider::protocol::{Escaper, prompt_text};
use rlx_pplx_decider::{Json, PplxDecider, Request};

const REQUESTS: &str = include_str!("fixtures/requests.json");
const DECISION_CONFIG: &str = include_str!("fixtures/decision_config.json");
const REF_F32: &str = include_str!("fixtures/ref_f32.json");
const REF_Q8_0: &str = include_str!("fixtures/ref_q8_0.json");

fn requests() -> Vec<Json> {
    Json::parse(REQUESTS).unwrap().as_array().unwrap().to_vec()
}

fn results(reference: &str) -> Vec<Json> {
    Json::parse(reference)
        .unwrap()
        .get("results")
        .unwrap()
        .as_array()
        .unwrap()
        .to_vec()
}

fn codes() -> Vec<String> {
    Json::parse(DECISION_CONFIG)
        .unwrap()
        .get("codes")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap().to_owned())
        .collect()
}

fn floats(v: &Json) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| match x {
            Json::Float(f) => *f,
            Json::Int(i) => *i as f64,
            other => panic!("not a number: {other:?}"),
        })
        .collect()
}

fn ints(v: &Json) -> Vec<u64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| match x {
            Json::Int(i) => *i as u64,
            Json::UInt(u) => *u,
            other => panic!("not an integer: {other:?}"),
        })
        .collect()
}

#[test]
fn prompt_text_is_byte_identical() {
    let codes = codes();
    let esc = Escaper::default();
    for (i, (r, want)) in requests().iter().zip(results(REF_F32)).enumerate() {
        let req = Request::from_json(r, codes.len()).unwrap();
        let got = prompt_text(&req, &codes, &esc);
        assert_eq!(
            got,
            want.get("text").unwrap().as_str().unwrap(),
            "request {i}: prompt differs"
        );
    }
}

fn open() -> Option<PplxDecider> {
    let path = rlx_ir::env::read("PPLX_DECIDER_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../weights/lm/pplx-decider-flash-gguf")
        });
    if !path.exists() {
        eprintln!(
            "skipping: {} not found (set PPLX_DECIDER_GGUF)",
            path.display()
        );
        return None;
    }
    let device = rlx_ir::env::read("PPLX_DECIDER_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = rlx_cli::parse_device(&device).unwrap();
    Some(PplxDecider::open(&path, device).expect("open decider"))
}

/// Everything that needs the weights, in one load (~5 GB, ~1 min on CPU).
#[test]
fn real_weights_match_the_reference() {
    let Some(mut d) = open() else { return };
    let q8 = results(REF_Q8_0);
    let f32_ref = results(REF_F32);
    let temperature = d.meta().temperature;

    let (mut worst_logit, mut worst_q8_p, mut worst_f32_p) = (0f64, 0f64, 0f64);
    for (i, r) in requests().iter().enumerate() {
        let req = d.request(r).unwrap();

        // Tokenization: the GGUF-embedded tokenizer must reproduce the
        // processor's ids exactly, markers included.
        let enc = d.encode(&req).unwrap();
        let want_ids = ints(q8[i].get("input_ids").unwrap());
        let got_ids: Vec<u64> = enc.input_ids.iter().map(|&t| u64::from(t)).collect();
        assert_eq!(got_ids, want_ids, "request {i}: token ids differ");
        let want_markers = ints(q8[i].get("markers").unwrap());
        let got_markers: Vec<u64> = enc.markers.iter().map(|&m| m as u64).collect();
        assert_eq!(got_markers, want_markers, "request {i}: markers differ");

        let logits = d.logits(&req).unwrap();
        let want_q8 = q8[i].get("logits").unwrap();
        let want_f32 = f32_ref[i].get("logits").unwrap();
        for ql in &logits {
            let a = floats(want_q8.get(&ql.id).unwrap());
            let b = floats(want_f32.get(&ql.id).unwrap());
            assert_eq!(
                a.len(),
                ql.logits.len(),
                "request {i} {}: option count",
                ql.id
            );
            let p = rlx_pplx_decider::protocol::probabilities(&ql.logits, temperature);
            let pa = rlx_pplx_decider::protocol::probabilities(
                &a.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                temperature,
            );
            let pb = rlx_pplx_decider::protocol::probabilities(
                &b.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                temperature,
            );
            for k in 0..a.len() {
                worst_logit = worst_logit.max((f64::from(ql.logits[k]) - a[k]).abs());
                worst_q8_p = worst_q8_p.max((p[k] - pa[k]).abs());
                worst_f32_p = worst_f32_p.max((p[k] - pb[k]).abs());
            }
        }
    }
    eprintln!(
        "max |Δlogit| vs dequantized-Q8_0 reference {worst_logit:.3e}, \
         max |Δp| {worst_q8_p:.3e}; max |Δp| vs bf16 reference {worst_f32_p:.3e}"
    );
    assert!(
        worst_logit < LOGIT_TOL,
        "logits off the Q8_0 reference by {worst_logit}"
    );
    assert!(
        worst_q8_p < 1e-3,
        "probabilities off the Q8_0 reference by {worst_q8_p}"
    );
    // The model card's own Q8_0-vs-f32 figure is 0.016 over its suite.
    assert!(
        worst_f32_p < 0.03,
        "probabilities off the bf16 reference by {worst_f32_p}"
    );
}

/// Between the two measured sides of the gap: CPU lands 4.5e-4 from the
/// dequantized reference (reduction order; logits are ~10-20), while
/// attending the right-padding — a defect that still passes the bf16
/// probability check above — moves a logit by 0.28.
const LOGIT_TOL: f64 = 5e-3;
