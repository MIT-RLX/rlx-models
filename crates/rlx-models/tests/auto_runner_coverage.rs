// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! What the universal entry point can actually reach.
//!
//! `auto_runner` sniffs a checkpoint and builds the right family's runner. Two
//! things can silently break that: an architecture the registry does not know
//! (so the sniff fails before any dispatch), and an architecture the registry
//! knows but the multiplexer has no arm for (so it bails after). This checks
//! the first half — which is where a newly added crate gets forgotten, because
//! the crate's own tests all pass without it.

use rlx_core::model_registry::{runner_for_gguf_arch, runner_for_hf_model_type};

/// Every registered constructor is reachable by the architecture sniff.
///
/// No hand-maintained list: the constructor registry is enumerable, so this
/// asks the real question — can anything actually select each family? A
/// registered constructor with no `model_registry` entry is dead code, which
/// is how `rlx-qwen4exp` and `rlx-minimax` both shipped complete `LmRunner`
/// implementations that nothing could reach.
#[cfg(feature = "runner")]
#[test]
fn every_registered_constructor_is_reachable_by_a_sniff() {
    rlx_models::run::register_builtin_lm_runners();
    let registered = rlx_core::lm_registry::registered_lm_runners();
    assert!(
        registered.len() >= 11,
        "expected the builtin families, got {registered:?}"
    );
    for name in registered {
        // Test-only registrations from other suites in the same process are
        // not expected to appear in the architecture tables.
        if name.starts_with("test-") {
            continue;
        }
        let reachable = rlx_core::model_registry::registered_gguf_models()
            .iter()
            .any(|r| r.runner == Some(name));
        assert!(
            reachable,
            "`{name}` has a registered constructor but no architecture points at it, \
             so the sniff can never select it"
        );
    }
}

/// An out-of-tree family becomes reachable by registering a constructor.
///
/// This is the point of the registry: a crate outside this workspace adds its
/// own model without patching the facade, and without depending on it.
#[cfg(feature = "runner")]
#[test]
fn an_external_family_can_register_and_be_built() {
    use rlx_core::lm_registry::{
        LmBuildOptions, LmRunnerRegistration, build_lm_runner, register_lm_runner,
    };
    use rlx_runtime::lm::LmRunner;

    struct Outsider;
    impl LmRunner for Outsider {
        fn family(&self) -> &'static str {
            "outsider"
        }
        fn vocab_size(&self) -> usize {
            11
        }
        fn predict_logits(&mut self, _p: &[u32]) -> anyhow::Result<Vec<f32>> {
            Ok(vec![0.0; 11])
        }
    }

    // Before: nothing can build it.
    assert!(
        build_lm_runner(
            "test-outsider",
            std::path::Path::new("/x"),
            &LmBuildOptions::default()
        )
        .is_err()
    );

    // `LmRunnerRegistration` is `#[non_exhaustive]`, so an out-of-tree crate
    // uses the constructor — which is the point: adding a capability field
    // later must not break every registrant.
    register_lm_runner(LmRunnerRegistration::new("test-outsider", |_p, _o| {
        Ok(Box::new(Outsider))
    }));

    // After: the generic path builds it, and `servable` wraps it even though
    // the family implements only `predict_logits`.
    let mut r = build_lm_runner(
        "test-outsider",
        std::path::Path::new("/x"),
        &LmBuildOptions::default().servable(),
    )
    .expect("the registered constructor should run");
    assert_eq!(r.family(), "outsider");
    assert_eq!(
        r.prefill_logits(&[1, 2]).expect("servable").len(),
        11,
        "an external family gets the host-driven pair for free"
    );
}

/// The architectures verified this session resolve to the crates that handle
/// them, rather than to nothing.
#[test]
fn the_recently_verified_architectures_resolve() {
    // Cloudflare/clef — `qwen3_5` is the HF spelling of the qwen35 trunk.
    assert_eq!(runner_for_hf_model_type("qwen3_5"), Some("qwen35"));
    // jialinyyzz/humanizer — `gemma4_unified` is a Gemma 4 flagship.
    assert_eq!(
        runner_for_hf_model_type("gemma4_unified"),
        Some("gemma"),
        "gemma4_unified should reach the gemma runner"
    );
    // microsoft/AesCode-8B and -32B are stock Qwen3-VL fine-tunes, so both
    // spellings have to land on the qwen3-vl runner: `qwen3_vl` is what
    // their HF configs say, `qwen3vl` is what bartowski's GGUF conversions
    // put in `general.architecture`.
    assert_eq!(runner_for_hf_model_type("qwen3_vl"), Some("qwen3-vl"));
    assert_eq!(runner_for_gguf_arch("qwen3vl"), Some("qwen3-vl"));
}

/// An unknown architecture resolves to nothing rather than to a wrong runner.
#[test]
fn an_unknown_architecture_resolves_to_nothing() {
    assert_eq!(runner_for_gguf_arch("definitely-not-a-model"), None);
    assert_eq!(runner_for_hf_model_type("definitely-not-a-model"), None);
    // qwen_image21 is a real architecture this repo does not implement; it
    // must not be silently routed to a text runner.
    assert_eq!(
        runner_for_gguf_arch("qwen_image21"),
        None,
        "a diffusion arch must not resolve to an LM runner"
    );
}

/// A family the registry knows but the multiplexer cannot build now says which
/// crate to use, instead of telling the caller to go find it.
///
/// Gated on `runner`, which is what brings `rlx_models::run` in — the registry
/// half above needs no features, so it keeps running in the default build.
///
/// `phi3` sniffs to the `phi` runner, which has no dispatch arm — so this is
/// the real "known but unreachable" path, driven through a minimal GGUF that
/// carries nothing but the architecture key.
#[cfg(feature = "runner")]
#[test]
fn an_unreachable_family_names_its_crate() {
    use rlx_gguf::GgufWriter;

    let dir = std::env::temp_dir().join(format!("rlx-models-unreachable-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmpdir");
    let path = dir.join("phi3.gguf");
    let mut w = GgufWriter::new();
    w.set_arch("phi3");
    w.write_to_path(&path).expect("write gguf");

    // The sniff resolves it...
    assert_eq!(runner_for_gguf_arch("phi3"), Some("phi"));
    // ...and the multiplexer refuses, naming the crate from the registry hint.
    let err = rlx_models::run::auto_runner(&path)
        .err()
        .expect("phi has no dispatch arm");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("rlx-phi"),
        "the refusal should name the crate that handles it: {msg}"
    );
    assert!(msg.contains("phi"), "and the runner it sniffed: {msg}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `auto_runner_servable` does not swallow a build failure.
///
/// The wrapper is constructed around whatever `auto_runner` returns, so an
/// error has to propagate rather than becoming a runner that fails later.
#[cfg(feature = "runner")]
#[test]
fn the_servable_wrapper_propagates_build_errors() {
    let missing = std::path::Path::new("/definitely/not/a/checkpoint.gguf");
    assert!(rlx_models::run::auto_runner(missing).is_err());
    assert!(
        rlx_models::run::auto_runner_servable(missing).is_err(),
        "wrapping must not turn a failed open into a live runner"
    );
}

/// `AutoOptions` is the one place the knobs live, and its default is explicit.
///
/// Before it existed there was no way to ask the multiplexer for a device, so
/// every runner it built ran on the CPU whatever the machine had — silently,
/// because nothing in the signature mentioned a device.
#[cfg(feature = "runner")]
#[test]
fn auto_options_carry_the_device_and_default_to_cpu() {
    use rlx_models::run::AutoOptions;
    use rlx_runtime::Device;

    let d = AutoOptions::default();
    assert_eq!(d.device, Device::Cpu, "the documented default");
    assert!(d.mmproj.is_none());
    assert!(!d.servable, "wrapping is opt-in");

    let m = AutoOptions::new(Device::Metal)
        .mmproj("mmproj.gguf")
        .servable();
    assert_eq!(m.device, Device::Metal);
    assert_eq!(
        m.mmproj.as_deref(),
        Some(std::path::Path::new("mmproj.gguf"))
    );
    assert!(m.servable);
}

/// The requested device reaches the dispatch, and an impossible one is refused
/// there rather than three layers down.
///
/// This is observable without weights because the device is validated **before**
/// the config read: an exotic device fails naming the device and the sniffed
/// family, while a plausible one gets as far as the missing tensors. If
/// `AutoOptions::device` were being dropped, every arm would report the same
/// metadata error and `Tpu` would be indistinguishable from `Cpu`.
#[cfg(feature = "runner")]
#[test]
fn an_impossible_device_is_refused_before_any_io() {
    use rlx_gguf::GgufWriter;
    use rlx_models::run::{AutoOptions, auto_runner_with};
    use rlx_runtime::Device;

    let dir = std::env::temp_dir().join(format!("rlx-models-device-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmpdir");
    let path = dir.join("qwen3.gguf");
    // Arch only — enough to sniff, not enough to build.
    let mut w = GgufWriter::new();
    w.set_arch("qwen3");
    w.write_to_path(&path).expect("write gguf");

    // Always impossible: no backend exists for these at all.
    for dev in [Device::OpenGl, Device::Hexagon] {
        let err = auto_runner_with(&path, &AutoOptions::new(dev))
            .err()
            .unwrap_or_else(|| panic!("{dev:?} is not an LM device"));
        let msg = format!("{err:#}");
        assert!(
            msg.contains(&format!("{dev:?}")) && msg.contains("not supported"),
            "{dev:?} should be refused by name, got: {msg}"
        );
        assert!(
            msg.contains("qwen3"),
            "and the refusal should name the sniffed family: {msg}"
        );
        assert!(
            !msg.contains("metadata key"),
            "the device must be checked before the config is read: {msg}"
        );
    }

    // A device this build does support gets past validation and fails on the
    // weights instead — which is what makes the above a real distinction.
    for dev in [Device::Cpu, Device::Metal] {
        let err = auto_runner_with(&path, &AutoOptions::new(dev))
            .err()
            .expect("an arch-only GGUF has no tensors");
        assert!(
            !format!("{err:#}").contains("not supported"),
            "{dev:?} is a standard device and must not be refused"
        );
    }

    // `Tpu` is the feature-gated case: a real backend exists upstream, so it is
    // refused for lacking the feature rather than for being impossible, and
    // accepted once the feature is on.
    let err = auto_runner_with(&path, &AutoOptions::new(Device::Tpu))
        .err()
        .expect("an arch-only GGUF has no tensors either way");
    let msg = format!("{err:#}");
    if cfg!(feature = "tpu") {
        assert!(
            msg.contains("metadata key"),
            "with `tpu` on, Tpu must pass validation and fail on the weights: {msg}"
        );
    } else {
        assert!(
            msg.contains("Tpu") && msg.contains("`tpu` feature"),
            "without the feature, the refusal should name it: {msg}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The builtins' declared mmproj capability matches which arms accept one.
///
/// The capability moved from four scattered runtime checks to one declaration
/// per family, so the declaration is now load-bearing: if it says `false` for a
/// VL family, that family's vision path becomes unreachable through the generic
/// entry point, and nothing else would notice.
#[cfg(feature = "runner")]
#[test]
fn the_declared_mmproj_capabilities_are_the_real_ones() {
    use rlx_core::lm_registry::lm_runner_supports_mmproj;
    rlx_models::run::register_builtin_lm_runners();

    // Families whose arms thread an mmproj into the builder.
    for vl in ["qwen35", "gemma", "mistral", "lfm", "qwen3-vl"] {
        assert_eq!(
            lm_runner_supports_mmproj(vl),
            Some(true),
            "`{vl}` accepts an mmproj and must declare it, or its vision path \
             is unreachable from auto_runner"
        );
    }
    // Text-only, or multimodal by a route that is not a separate mmproj file.
    for text in [
        "qwen3",
        "llama32",
        "qwen4exp",
        "minimax-m3",
        "llama4",
        "mllama",
    ] {
        assert_eq!(
            lm_runner_supports_mmproj(text),
            Some(false),
            "`{text}` has no mmproj path and must declare that, so a request \
             is refused rather than ignored"
        );
    }
}

/// A text-only builtin refuses an mmproj, and says what to do instead.
#[cfg(feature = "runner")]
#[test]
fn a_text_only_builtin_refuses_an_mmproj_before_touching_the_weights() {
    use rlx_gguf::GgufWriter;
    use rlx_models::run::{AutoOptions, auto_runner_with};

    let dir = std::env::temp_dir().join(format!("rlx-models-mmproj-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmpdir");
    let path = dir.join("qwen3.gguf");
    let mut w = GgufWriter::new();
    w.set_arch("qwen3");
    w.write_to_path(&path).expect("write gguf");

    let opts = AutoOptions::default().mmproj("some-mmproj.gguf");
    let err = auto_runner_with(&path, &opts)
        .err()
        .expect("qwen3 takes no mmproj");
    let msg = format!("{err:#}");
    assert!(msg.contains("does not accept an mmproj"), "{msg}");
    assert!(msg.contains("qwen3vl"), "and names the alternative: {msg}");
    assert!(
        !msg.contains("metadata key"),
        "refused before the weights were read: {msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
