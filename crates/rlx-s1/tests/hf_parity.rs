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

//! HuggingFace parity for S1-mini, in three layers:
//!
//!   1. **Prompt string** — `render_prompt` vs `apply_chat_template(...,
//!      enable_thinking=False)`, byte-for-byte.
//!   2. **Prompt ids** — our tokenizer bridge vs HF's, id-for-id.
//!   3. **Greedy output** — our decode vs HF's `generate(do_sample=False)`,
//!      token-for-token.
//!
//! Layer 1 needs nothing but the reference JSON; layers 2 and 3 need the
//! checkpoint. Generate the reference with:
//!
//! ```text
//! python3 scripts/s1_hf_reference.py \
//!     --weights /Volumes/FOUR/weights/lm/s1-mini --out /tmp/s1_reference.json
//! S1_REFERENCE=/tmp/s1_reference.json S1_WEIGHTS=/Volumes/FOUR/weights/lm/s1-mini \
//!     cargo test -p rlx-s1 --test hf_parity -- --nocapture
//! ```
//!
//! Every test no-ops with a printed reason when its inputs are absent, so the
//! suite stays green on a machine without the weights.

use rlx_s1::{Context, Controls, S1Runner, render_prompt};
use serde_json::Value;
use std::path::PathBuf;

fn reference() -> Option<Value> {
    let path = std::env::var("S1_REFERENCE").unwrap_or_else(|_| "/tmp/s1_reference.json".into());
    let raw = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&raw).ok()
}

fn weights() -> Option<PathBuf> {
    let p: PathBuf = std::env::var("S1_WEIGHTS")
        .unwrap_or_else(|_| "/Volumes/FOUR/weights/lm/s1-mini".into())
        .into();
    p.exists().then_some(p)
}

fn controls_of(case: &Value) -> Controls {
    Controls {
        styling: case["styling"].as_str().unwrap().parse().unwrap(),
        structure: case["structure"].as_str().unwrap().parse().unwrap(),
        context: case["context"].as_str().unwrap().parse().unwrap(),
    }
}

fn cases(reference: &Value) -> Vec<Value> {
    reference["cases"].as_array().cloned().unwrap_or_default()
}

#[test]
fn prompt_string_matches_hf_chat_template() {
    let Some(reference) = reference() else {
        eprintln!("skip: no S1_REFERENCE json (run scripts/s1_hf_reference.py)");
        return;
    };
    // The system prompt itself, first — everything else is downstream of it.
    assert_eq!(
        reference["system"].as_str().unwrap(),
        rlx_s1::SYSTEM_PROMPT,
        "system prompt drifted from the model card"
    );
    let cases = cases(&reference);
    assert!(!cases.is_empty(), "reference json has no cases");
    for case in &cases {
        let transcript = case["transcript"].as_str().unwrap();
        let ours = render_prompt(controls_of(case), transcript);
        assert_eq!(
            ours,
            case["prompt"].as_str().unwrap(),
            "prompt mismatch for {transcript:?}"
        );
    }
    eprintln!("[s1-parity] {} prompts byte-identical to HF", cases.len());
}

#[test]
fn prompt_ids_match_hf_tokenizer() {
    let (Some(reference), Some(weights)) = (reference(), weights()) else {
        eprintln!("skip: need S1_REFERENCE json + S1_WEIGHTS checkpoint");
        return;
    };
    // Tokenize only — no graph compile, so this is cheap.
    let runner = S1Runner::builder()
        .weights(&weights)
        .max_seq(64)
        .build()
        .expect("build S1Runner");
    for case in &cases(&reference) {
        let transcript = case["transcript"].as_str().unwrap();
        let ours = runner
            .encode_prompt(transcript, controls_of(case))
            .expect("encode prompt");
        let theirs: Vec<u32> = case["prompt_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(ours, theirs, "prompt ids differ for {transcript:?}");
    }
}

#[test]
fn greedy_output_matches_hf() {
    let (Some(reference), Some(weights)) = (reference(), weights()) else {
        eprintln!("skip: need S1_REFERENCE json + S1_WEIGHTS checkpoint");
        return;
    };
    let mut runner = S1Runner::builder()
        .weights(&weights)
        .build()
        .expect("build S1Runner");

    let mut checked = 0usize;
    for case in &cases(&reference) {
        let Some(expected) = case["output"].as_str() else {
            continue;
        };
        let transcript = case["transcript"].as_str().unwrap();
        let controls = controls_of(case);

        let mut ids: Vec<u32> = Vec::new();
        let got = runner
            .normalize_pass(transcript, controls, |t| ids.push(t))
            .expect("normalize");

        // HF keeps the stop token in `output_ids`; we don't emit it.
        let mut theirs: Vec<u32> = case["output_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        if theirs
            .last()
            .is_some_and(|t| rlx_s1::EOS_TOKENS.contains(t))
        {
            theirs.pop();
        }

        assert_eq!(
            got,
            expected.trim(),
            "text differs for {transcript:?} under {controls}"
        );
        assert_eq!(
            ids, theirs,
            "token ids differ for {transcript:?} under {controls}"
        );
        checked += 1;
        eprintln!("[s1-parity] ok  {controls}  {transcript:?} -> {got:?}");
    }
    assert!(checked > 0, "reference json had no generated outputs");
    eprintln!("[s1-parity] {checked} completions token-identical to HF");
}

/// Filler-only input is documented to normalize to the empty string, and a
/// pipeline is supposed to treat that as success rather than failure.
#[test]
fn filler_only_input_yields_empty_string() {
    let Some(weights) = weights() else {
        eprintln!("skip: no S1_WEIGHTS checkpoint");
        return;
    };
    let mut runner = S1Runner::builder().weights(&weights).build().unwrap();
    assert_eq!(runner.normalize("um").unwrap(), "");
    // Whitespace never reaches the model at all.
    assert_eq!(runner.normalize("   \n ").unwrap(), "");
}

/// One runner is meant to be reused across many calls, so no KV or sampler
/// state may leak between them: the same transcript must decode identically the
/// second time, and an intervening call under different controls must not
/// perturb it.
#[test]
fn repeated_calls_on_one_runner_are_deterministic() {
    let Some(weights) = weights() else {
        eprintln!("skip: no S1_WEIGHTS checkpoint");
        return;
    };
    let transcript = "i think the answer is forty two no sorry forty three";
    let mut runner = S1Runner::builder().weights(&weights).build().unwrap();

    let mut first: Vec<u32> = Vec::new();
    let a = runner
        .normalize_pass(transcript, Controls::new(), |t| first.push(t))
        .unwrap();

    // Different controls, different transcript, in between.
    let _ = runner
        .normalize_with(
            "hey sarah just wanted to follow up",
            Controls::new().context(Context::Email),
        )
        .unwrap();

    let mut second: Vec<u32> = Vec::new();
    let b = runner
        .normalize_pass(transcript, Controls::new(), |t| second.push(t))
        .unwrap();

    assert_eq!(first, second, "decode drifted between calls on one runner");
    assert_eq!(a, b);
    assert_eq!(a, "I think the answer is 43.");
}

/// A transcript past the per-pass budget must come back as a chunked pass that
/// still covers the whole input, not a silently truncated one.
#[test]
fn long_transcript_chunks_instead_of_truncating() {
    let Some(weights) = weights() else {
        eprintln!("skip: no S1_WEIGHTS checkpoint");
        return;
    };
    let runner = S1Runner::builder()
        .weights(&weights)
        .chunk_tokens(24)
        .max_seq(256)
        .build()
        .unwrap();
    let sentence = "so um i need to send the report by friday no wait thursday. ";
    let long = sentence.repeat(12);
    let chunks = runner.chunk(&long).unwrap();
    assert!(chunks.len() > 1, "expected chunking, got {}", chunks.len());
    for c in &chunks {
        assert!(
            runner.count_tokens(c).unwrap() <= 24,
            "chunk over budget: {c:?}"
        );
    }
    // Every word survives the split.
    let joined: String = chunks.join(" ");
    assert_eq!(
        joined.split_whitespace().count(),
        long.split_whitespace().count()
    );
}
