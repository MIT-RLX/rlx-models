//! **End-to-end parity against the reference implementation**, on real
//! weights.
//!
//! `tests/fixtures/golden_probs_kev_0_8b.json` holds the probabilities
//! jaredpalmer/kev itself produces for `examples/ticket.json` with
//! `jaredpalmer/kev-0.8b` on `Qwen/Qwen3.5-0.8B-Base`, merged in fp32 on CPU
//! (regenerate with `scripts/kev_reference_probs.py`).
//!
//! This is the test that matters. Two bugs found by exactly this comparison
//! were invisible to everything else, because both leave the model producing
//! a well-formed, plausible-looking distribution:
//!
//! * the MRoPE feed used `key_length / 2` where the flow strides by
//!   `rope_dim_count / 2` (Qwen3.5 is partial-rotary, so 128 vs 32);
//! * HF stores the Gated DeltaNet decay as `A_log` while the builder wants
//!   llama.cpp's `-exp(A_log)`. Position 0 stays *exact* under that bug — the
//!   recurrent state is still zero there — and only later positions drift.
//!
//! Gated on both directories being present:
//!
//! ```text
//! KEV_RUN_DIR=/path/kev-0.8b KEV_BASE_DIR=/path/Qwen3.5-0.8B-Base \
//!   cargo test -p rlx-kev --test real_weights --release
//! ```

use std::path::PathBuf;

use rlx_kev::api::to_record;
use rlx_kev::model::ShapePolicy;
use rlx_runtime::Device;

fn dirs() -> Option<(PathBuf, PathBuf)> {
    let run = PathBuf::from(std::env::var_os("KEV_RUN_DIR")?);
    let base = PathBuf::from(std::env::var_os("KEV_BASE_DIR")?);
    (run.join("head.pt").exists() && base.join("config.json").exists()).then_some((run, base))
}

/// Devices to check. Every backend gets the same assertion — a GPU that
/// silently disagrees with CPU is the failure mode this catches.
///
/// **CoreML / ANE is deliberately absent.** It does not merely fail: on a
/// graph containing `Op::GatedDeltaNet` the emitted MIL passes rlx's own
/// checks but MPSGraph rejects it and the framework calls `abort()`, taking the test
/// process with it — so it cannot be reported as a failure, only as a crash.
/// Reproducible at batch 1 and batch 4:
///
/// ```text
/// v20 = sigmoid(v17)                      declared tensor<fp32, [512, 16]>
/// v24 = reshape(shape=[4,128,16], x=v20)  MPS sees  tensor<4x16x8208xf32>
/// 'mps.reshape' op the result shape is not compatible with the input shape
/// ```
///
/// `examples/coreml_slice_repro.rs` isolates the two obvious suspects — a
/// bare rank-2 `narrow`, and the `concat(weights) -> matmul -> narrow`
/// prologue the builder emits — and both pass on ANE, so the trigger needs
/// the full GDN graph. Upstream `rlx-coreml` defect; re-add ANE once fixed.
#[allow(unused_mut)] // every push below is feature-gated
fn devices() -> Vec<(&'static str, Device)> {
    let mut v = vec![("cpu", Device::Cpu)];
    #[cfg(feature = "metal")]
    v.push(("metal", Device::Metal));
    #[cfg(feature = "mlx")]
    v.push(("mlx", Device::Mlx));
    v
}

struct Fixture {
    ids: Vec<String>,
    probs: Vec<Vec<f32>>,
    temperature: f32,
}

fn fixture() -> Fixture {
    let v: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/golden_probs_kev_0_8b.json"))
            .expect("fixture parses");
    Fixture {
        ids: v["meta"]
            .as_array()
            .expect("meta")
            .iter()
            .map(|m| m["id"].as_str().expect("id").to_string())
            .collect(),
        probs: v["probs"]
            .as_array()
            .expect("probs")
            .iter()
            .map(|p| {
                p.as_array()
                    .expect("vec")
                    .iter()
                    .map(|x| x.as_f64().expect("f") as f32)
                    .collect()
            })
            .collect(),
        temperature: v["temperature"].as_f64().expect("t") as f32,
    }
}

#[test]
fn probabilities_match_the_reference_on_every_backend() {
    let Some((run, base)) = dirs() else {
        eprintln!("skipping: set KEV_RUN_DIR and KEV_BASE_DIR");
        return;
    };
    let fx = fixture();

    let req = rlx_kev::api::SystemOneRequest::parse(include_str!("../examples/ticket.json"))
        .expect("request parses");
    let (rec, meta) = to_record(&req).expect("record");
    assert_eq!(
        meta.iter().map(|m| m.id.clone()).collect::<Vec<_>>(),
        fx.ids,
        "fixture question order must match the request"
    );

    for (name, device) in devices() {
        let mut model = match rlx_kev::load(&run, &base, device, ShapePolicy::default()) {
            Ok(m) => m,
            Err(e) => panic!("{name}: load failed: {e:#}"),
        };
        assert!(
            (model.head().temperature - fx.temperature).abs() < 1e-5,
            "{name}: temperature {} != fixture {}",
            model.head().temperature,
            fx.temperature
        );

        let got = model
            .probs(&rec)
            .unwrap_or_else(|e| panic!("{name}: {e:#}"));
        assert_eq!(got.len(), fx.probs.len(), "{name}: question count");

        let mut worst = 0.0f32;
        for (i, (g, w)) in got.iter().zip(&fx.probs).enumerate() {
            assert_eq!(g.len(), w.len(), "{name}: question {i} option count");
            for (a, b) in g.iter().zip(w) {
                worst = worst.max((a - b).abs());
            }
            let am = |v: &[f32]| rlx_kev::api::argmax(v);
            assert_eq!(
                am(g),
                am(w),
                "{name}: question {} ({}) picked a different option: {g:?} vs {w:?}",
                i,
                fx.ids[i]
            );
        }
        // fp32 CPU reproduces the reference to ~1e-3; GPU backends reorder
        // reductions, so allow a little more before calling it a defect.
        let tol = if name == "cpu" { 2e-3 } else { 1e-2 };
        assert!(
            worst < tol,
            "{name}: max probability difference {worst} exceeds {tol}\n  got {got:?}\n  want {:?}",
            fx.probs
        );
        eprintln!("{name}: max |Δp| = {worst:.6}");
    }
}

/// CoreML must refuse **before** reading the base model, with a reason.
///
/// MPSGraph miscompiles the narrow-then-reshape shape the Gated DeltaNet
/// prologue emits. `rlx-coreml` now rejects that pattern instead of emitting
/// a module that aborts the process, but that rejection lands mid-compile —
/// after the weights are on the heap. This checks the early guard, so a
/// caller learns it cannot use ANE without paying for a 3.5 GB load first.
#[cfg(feature = "coreml")]
#[test]
fn coreml_is_refused_up_front_with_a_named_reason() {
    let Some((run, base)) = dirs() else {
        eprintln!("skipping: set KEV_RUN_DIR and KEV_BASE_DIR");
        return;
    };
    let err = match rlx_kev::load(&run, &base, Device::Ane, ShapePolicy::default()) {
        Ok(_) => panic!("CoreML must not silently accept a Gated DeltaNet trunk"),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        err.contains("CoreML") && err.contains("DeltaNet"),
        "the refusal should name the backend and the cause, got: {err}"
    );
    assert!(
        err.contains("metal") || err.contains("mlx"),
        "the refusal should point at a working device, got: {err}"
    );
}

/// The weight bundle (~3.5 GB in f32) must not stay resident after a build.
///
/// It is only needed while the graph is being built; afterwards the compiled
/// arena owns its own copy. Holding it costs that much memory for the life of
/// the model, so `load` installs a reloader and the bundle is dropped.
#[test]
fn the_weight_bundle_is_released_after_building() {
    let Some((run, base)) = dirs() else {
        eprintln!("skipping: set KEV_RUN_DIR and KEV_BASE_DIR");
        return;
    };
    let req = rlx_kev::api::SystemOneRequest::parse(include_str!("../examples/ticket.json"))
        .expect("request parses");
    let (rec, _meta) = to_record(&req).expect("record");

    let mut model = rlx_kev::load(&run, &base, Device::Cpu, ShapePolicy::default()).expect("load");
    assert!(model.weights_resident(), "resident before the first build");
    let first = model.probs(&rec).expect("forward");
    assert!(
        !model.weights_resident(),
        "the bundle must be released once a graph is built"
    );

    // A shape change must still work — the reloader re-reads from disk.
    let one = rlx_kev::api::SystemOneRequest::parse(
        r#"{"state":"s","questions":{"q":{"type":"noul","instructions":"i"}}}"#,
    )
    .expect("parses");
    let (rec1, _) = to_record(&one).expect("record");
    let reshaped = model.probs(&rec1).expect("forward at a new shape");
    assert_eq!(reshaped.len(), 1);
    assert!((reshaped[0].iter().sum::<f32>() - 1.0).abs() < 1e-5);

    // And the original shape still answers identically after the round trip.
    let again = model.probs(&rec).expect("forward again");
    assert_eq!(
        first, again,
        "re-reading the weights must reproduce the same answers"
    );
}

/// **The prefix cache must not change any answer.**
///
/// It replaces one graph with two — a state pass exporting the trunk's
/// recurrent + KV state, and a per-question continuation resuming from it
/// under a bias mask — so it is a different computation reaching the same
/// place. The uncached path is parity-exact against the reference, so it is
/// the oracle here.
///
/// The bug this caught, and why it needed a test rather than review: the
/// prefill flow's `export_normed_hidden` returns the hidden *after*
/// `output_norm`, while the decode flow's hidden output is the value *before*
/// it. Those differ in only a few large channels — the RMS was 1.763 against
/// an expected 1.759, so it looked like noise, while element 0 was -50
/// instead of -6.3 and every probability was wrong by ~0.27.
#[cfg(feature = "prefix-cache")]
#[test]
fn the_prefix_cache_reproduces_the_uncached_answers() {
    let Some((run, base)) = dirs() else {
        eprintln!("skipping: set KEV_RUN_DIR and KEV_BASE_DIR");
        return;
    };
    // A state long enough to clear the cache's minimum-length floor.
    let para = "Order 88213 was placed on June 2 and the tracking number has not \
                updated since June 9. I called support twice and was told the parcel \
                was still at the depot. Meanwhile my card shows two authorisations \
                for the same amount. ";
    let state: String = para.repeat(12).trim().to_string();
    let body = serde_json::json!({
        "state": state,
        "model": "kev-latest",
        "questions": {
            "department": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"returns": "Exchanges, refunds",
                                        "shipping": "Delivery status, delays",
                                        "billing": serde_json::Value::Null}},
            "escalate":    {"type": "noul",  "instructions": "Does this need urgent attention?"},
            "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                            "criteria": ["Calm", "Frustrated", "Very angry"]}
        }
    });
    let text = serde_json::to_string(&body).expect("request");
    let req = rlx_kev::api::SystemOneRequest::parse(&text).expect("parses");
    let (rec, _meta) = to_record(&req).expect("record");

    let mut plain = rlx_kev::load(&run, &base, Device::Cpu, ShapePolicy::default()).expect("load");
    let want = plain.probs(&rec).expect("uncached forward");
    drop(plain);

    let mut cached = rlx_kev::load(&run, &base, Device::Cpu, ShapePolicy::default())
        .expect("load")
        .with_prefix_cache(rlx_kev::prefix::PrefixCacheConfig {
            capacity: 4,
            min_state_tokens: 384,
        });

    // First call: a cache miss, so the state pass runs.
    let got = cached.probs(&rec).expect("cached forward (miss)");
    let (hits, misses, n) = cached.prefix_stats().expect("stats");
    assert_eq!((hits, misses, n), (0, 1, 1), "first call must be a miss");

    let worst = |a: &[Vec<f32>], b: &[Vec<f32>]| -> f32 {
        a.iter()
            .zip(b)
            .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
            .fold(0.0f32, f32::max)
    };
    let d_miss = worst(&want, &got);
    assert!(
        d_miss < 2e-3,
        "prefix-cached answers differ from uncached by {d_miss}\n  cached   {got:?}\n  uncached {want:?}"
    );

    // Second call: a hit, replaying the branches against the stored state.
    let again = cached.probs(&rec).expect("cached forward (hit)");
    let (hits, _, _) = cached.prefix_stats().expect("stats");
    assert_eq!(hits, 1, "the second call must hit");
    assert_eq!(again, got, "a cache hit must be bit-identical to the miss");

    eprintln!("prefix cache: max |Δp| vs uncached = {d_miss:.6}");
}
