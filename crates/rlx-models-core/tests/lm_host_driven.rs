// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! The host-driven adapter against the loop it stands in for.
//!
//! A family that implements only `predict_logits` cannot be served:
//! `rlx_serve::run_generation` calls `prefill_logits` with `?` and the trait's
//! default is an error. The claim here is that wrapping such a runner makes the
//! server's loop produce exactly what the model's own `generate` produces — the
//! adapter must change the *cost*, never the tokens.

use anyhow::Result;
use rlx_models_core::lm_host_driven::HostDriven;
use rlx_runtime::lm::LmRunner;

const VOCAB: usize = 7;

/// The shape most families in this workspace are in: `predict_logits` only, so
/// `generate` comes from the trait's own (quadratic) default.
struct PredictOnly {
    calls: usize,
}

impl LmRunner for PredictOnly {
    fn family(&self) -> &'static str {
        "predict-only"
    }
    fn vocab_size(&self) -> usize {
        VOCAB
    }
    /// Deterministic and context-sensitive: the argmax depends on every token,
    /// so a wrapper that loses or reorders context gives a different chain.
    fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.calls += 1;
        let mut acc = 0u64;
        for (i, &t) in prompt_ids.iter().enumerate() {
            acc = acc
                .wrapping_mul(1_000_003)
                .wrapping_add((t as u64 + 1) * (i as u64 + 7));
        }
        Ok((0..VOCAB)
            .map(|v| {
                let h = acc.wrapping_mul(v as u64 + 11) >> 13;
                ((h % 1000) as f32) / 1000.0
            })
            .collect())
    }
}

/// A family that answers the pair itself, to check the wrapper steps aside.
struct Native {
    context: Vec<u32>,
    prefills: usize,
}

impl LmRunner for Native {
    fn family(&self) -> &'static str {
        "native"
    }
    fn vocab_size(&self) -> usize {
        VOCAB
    }
    fn predict_logits(&mut self, _p: &[u32]) -> Result<Vec<f32>> {
        Ok(vec![0.0; VOCAB])
    }
    fn prefill_logits(&mut self, p: &[u32]) -> Result<Vec<f32>> {
        self.prefills += 1;
        self.context = p.to_vec();
        Ok(vec![1.0; VOCAB])
    }
    fn decode_logits(&mut self, t: u32) -> Result<Vec<f32>> {
        self.context.push(t);
        Ok(vec![2.0; VOCAB])
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

/// The wrapped host-driven loop produces the model's own greedy chain.
#[test]
fn the_host_driven_loop_matches_generate() {
    let prompt = [3u32, 1, 4, 1, 5];
    let n = 6usize;

    // What the model does on its own.
    let mut plain = PredictOnly { calls: 0 };
    let mut want = Vec::new();
    plain
        .generate(&prompt, n, &mut |t| {
            want.push(t);
            true
        })
        .expect("generate");
    assert_eq!(want.len(), n);

    // What a server does through the adapter: prefill, then one step per token.
    let mut run = HostDriven::new(Box::new(PredictOnly { calls: 0 }));
    assert_eq!(
        run.is_passthrough(),
        None,
        "nothing is known before a prefill"
    );
    let mut logits = run.prefill_logits(&prompt).expect("prefill");
    assert_eq!(
        run.is_passthrough(),
        Some(false),
        "a predict-only runner needs the wrapper"
    );
    let mut got = Vec::new();
    for _ in 0..n {
        let tok = argmax(&logits);
        got.push(tok);
        logits = run.decode_logits(tok).expect("decode");
    }

    assert_eq!(got, want, "the adapter changed the model's output");
    assert_eq!(
        run.context().len(),
        prompt.len() + n,
        "context should hold the prompt plus every decoded token"
    );
    // One per step, which is the quadratic term this makes countable.
    assert_eq!(run.reprefills(), n + 1);
}

/// A family that implements the pair is passed through untouched.
#[test]
fn a_native_runner_is_not_re_prefilled() {
    let mut run = HostDriven::new(Box::new(Native {
        context: Vec::new(),
        prefills: 0,
    }));
    let p = run.prefill_logits(&[1, 2, 3]).expect("prefill");
    assert_eq!(
        run.is_passthrough(),
        Some(true),
        "Native answers the pair itself"
    );
    assert_eq!(
        p,
        vec![1.0; VOCAB],
        "the native prefill should be the one used"
    );
    let d = run.decode_logits(9).expect("decode");
    assert_eq!(
        d,
        vec![2.0; VOCAB],
        "the native decode should be the one used"
    );
    assert_eq!(run.reprefills(), 0, "no re-prefill should have happened");
    // The wrapper kept no context of its own; the family has it.
    assert!(run.context().is_empty());
    assert_eq!(run.inner().context, vec![1, 2, 3, 9]);
}

/// Decoding before prefilling is an error, not a silent empty context.
#[test]
fn decoding_before_prefilling_is_refused() {
    let mut run = HostDriven::new(Box::new(PredictOnly { calls: 0 }));
    let err = run.decode_logits(4).expect_err("no context yet");
    let msg = format!("{err:#}");
    assert!(msg.contains("before prefill_logits"), "{msg}");

    let err = run.prefill_logits(&[]).expect_err("empty prompt");
    assert!(format!("{err:#}").contains("at least one token"), "{err:#}");
}

/// Everything that is not the decode pair delegates.
#[test]
fn the_rest_of_the_trait_delegates() {
    let mut run = HostDriven::new(Box::new(PredictOnly { calls: 0 }));
    assert_eq!(run.family(), "predict-only");
    assert_eq!(run.vocab_size(), VOCAB);
    assert!(!run.supports_multimodal());
    assert!(run.export_session().is_none());
    assert!(run.kv_store_stats().is_none());
    // `predict_logits` reaches the model without touching the wrapper's context.
    run.predict_logits(&[1, 2]).expect("predict");
    assert!(run.context().is_empty());
    assert_eq!(run.reprefills(), 0);
}

/// A boxed trait object — the shape a caller actually holds — works.
///
/// `rlx_serve` takes `Box<dyn LmRunner>`, and `Box<dyn LmRunner>` does not
/// itself implement `LmRunner`, so an adapter generic over a sized `R` would
/// have been unusable by the one caller it exists for.
#[test]
fn a_boxed_trait_object_can_be_wrapped() {
    let boxed: Box<dyn LmRunner> = Box::new(PredictOnly { calls: 0 });
    let mut run: HostDriven<dyn LmRunner> = HostDriven::new(boxed);
    let logits = run.prefill_logits(&[2, 7, 1]).expect("prefill");
    assert_eq!(logits.len(), VOCAB);
    assert_eq!(run.is_passthrough(), Some(false));
    let next = run.decode_logits(3).expect("decode");
    assert_eq!(next.len(), VOCAB);
    assert_eq!(run.context(), &[2, 7, 1, 3]);
    assert_eq!(run.family(), "predict-only");

    // The shape `rlx_serve::host_drivable` returns: wrapped, re-boxed as the
    // trait object its caller holds. Checked here because rlx-serve cannot be
    // built in isolation from the model crates it serves.
    let again: Box<dyn LmRunner> = Box::new(HostDriven::new(
        Box::new(PredictOnly { calls: 0 }) as Box<dyn LmRunner>
    ));
    let mut again = again;
    assert_eq!(again.vocab_size(), VOCAB);
    assert_eq!(
        again.prefill_logits(&[1, 2]).expect("prefill").len(),
        VOCAB,
        "a doubly-boxed wrapper must still serve"
    );
}

/// An error that is *not* "unsupported" is the model's answer, not a cue to
/// fall back.
///
/// Retrying a real failure through a different code path would turn a clear
/// error into either a wrong answer or a confusing one.
#[test]
fn a_real_prefill_error_is_not_treated_as_unsupported() {
    struct Broken;
    impl LmRunner for Broken {
        fn family(&self) -> &'static str {
            "broken"
        }
        fn vocab_size(&self) -> usize {
            VOCAB
        }
        fn predict_logits(&mut self, _p: &[u32]) -> Result<Vec<f32>> {
            panic!("the fallback must not be taken");
        }
        fn prefill_logits(&mut self, _p: &[u32]) -> Result<Vec<f32>> {
            anyhow::bail!("out of memory loading layer 7")
        }
    }
    let mut run = HostDriven::new(Box::new(Broken));
    let err = run.prefill_logits(&[1]).expect_err("the model failed");
    assert!(format!("{err:#}").contains("out of memory"), "{err:#}");
    assert_eq!(
        run.is_passthrough(),
        None,
        "a real failure settles nothing about capability"
    );
}
