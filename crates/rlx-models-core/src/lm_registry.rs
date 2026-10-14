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

//! Who can build an [`LmRunner`], and how.
//!
//! [`crate::model_registry`] answers *which crate handles this architecture* —
//! names, arches, `model_type`s, a hint. This answers the next question: given
//! that name, **construct the thing**. They are deliberately separate: the
//! metadata half is a static table with no dependencies, while a constructor
//! has to link the model crate.
//!
//! ## Why a registry instead of a `match`
//!
//! The dispatch used to be a hardcoded `match runner_name { .. }` inside the
//! `rlx-models` facade, which had three consequences:
//!
//! * **An external crate could not add a model.** Its only options were to
//!   patch the facade or to tell users to call its builder directly, giving up
//!   every caller that goes through the generic path.
//! * **Every new family meant editing the facade**, and that edit is easy to
//!   forget: `rlx-qwen4exp` and `rlx-minimax` both shipped complete
//!   `LmRunner` implementations that nothing could reach, because their own
//!   test suites passed without the arm.
//! * **Using the generic path meant depending on ~200 crates**, since the
//!   facade links every family it might dispatch to.
//!
//! Registration inverts all three. A crate — in this workspace or outside it —
//! registers a constructor for its own name, and the generic path finds it.
//!
//! ```no_run
//! use rlx_models_core::lm_registry::{LmBuildOptions, LmRunnerRegistration, register_lm_runner};
//! use rlx_runtime::lm::LmRunner;
//! # struct MyRunner;
//! # impl MyRunner { fn open(_: &std::path::Path, _: rlx_runtime::Device) -> anyhow::Result<Self> { unimplemented!() } }
//! # impl LmRunner for MyRunner {
//! #     fn family(&self) -> &'static str { "mine" }
//! #     fn vocab_size(&self) -> usize { 0 }
//! #     fn predict_logits(&mut self, _: &[u32]) -> anyhow::Result<Vec<f32>> { unimplemented!() }
//! # }
//!
//! register_lm_runner(LmRunnerRegistration::new("mine", |path, opts| {
//!     Ok(Box::new(MyRunner::open(path, opts.device)?))
//! }));
//! ```
//!
//! Register before the first lookup — a `ctor`-style static initializer, or at
//! the top of `main`. Registration is idempotent: the same name twice replaces
//! the earlier entry, which is what lets a host override a builtin.

use anyhow::{Result, bail};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};

use rlx_runtime::Device;
use rlx_runtime::lm::LmRunner;

/// How to build a runner from a checkpoint path.
///
/// One options type rather than a growing family of `build_x_with_y`
/// functions, so adding a knob does not add an entry point.
#[derive(Debug, Clone)]
pub struct LmBuildOptions {
    /// Device handed to the family's builder. `None` = pick one for this
    /// machine; see [`Self::device_or_auto`].
    ///
    /// `Option` because "the caller did not choose" and "the caller chose CPU"
    /// are different instructions, and a bare `Device` cannot tell them apart.
    /// This field used to be a bare `Device` defaulting to `Device::Cpu`, so
    /// every `AutoOptions::default()` caller — `auto_runner`,
    /// `auto_runner_with_mmproj`, `auto_runner_servable*` — silently asked for
    /// CPU on a machine with a GPU, and got it. The name says `auto`; the
    /// default said CPU. `LmBuilderCommon` had already drawn this exact
    /// distinction for the same reason (see its module docs); the registry's
    /// options type had not.
    pub device: Option<Device>,
    /// Vision encoder, for families that accept one. A constructor that has no
    /// multimodal path must **error** on `Some` rather than ignore it.
    pub mmproj: Option<PathBuf>,
    /// Wrap the result in [`crate::lm_host_driven::HostDriven`] so
    /// `prefill_logits` / `decode_logits` work whatever the family implements.
    pub servable: bool,
    /// Context the runner is built for, in tokens. `None` leaves the family
    /// default.
    ///
    /// This is NOT a tuning knob, it is a correctness one. Families size their
    /// bucketed decode compile cache from `max_seq`, and the shared default is
    /// 128 — so an `auto_runner` caller that never set it got a decode ladder
    /// covering ~256 tokens of past. One token beyond that the bucket lookup
    /// misses, and a native-packed (K-quant GGUF) runner has no F32 decode
    /// fallback to drop to, so the chat dies with
    /// `weight not found: model.embed_tokens.weight` — which reads like a
    /// tensor-naming problem and is really the context cap. Every caller that
    /// wanted a real context therefore had to bypass `auto_runner` and reach
    /// for the family builder directly, which is how a routing cascade grows.
    pub max_seq: Option<usize>,
}

impl Default for LmBuildOptions {
    fn default() -> Self {
        LmBuildOptions {
            device: None,
            mmproj: None,
            servable: false,
            max_seq: None,
        }
    }
}

impl LmBuildOptions {
    pub fn new(device: Device) -> Self {
        LmBuildOptions {
            device: Some(device),
            ..Default::default()
        }
    }

    /// The caller's device, or one picked for this machine.
    ///
    /// Resolve this ONCE per build and hand the result to every family builder,
    /// so a single `auto_runner` call cannot put two parts of the same model on
    /// two different devices.
    pub fn device_or_auto(&self) -> Device {
        match self.device {
            Some(d) => d,
            None => {
                let d = crate::device_capabilities::pick_lm_device();
                eprintln!("[auto-runner] auto device → {d:?}");
                d
            }
        }
    }
    pub fn mmproj(mut self, path: impl Into<PathBuf>) -> Self {
        self.mmproj = Some(path.into());
        self
    }
    /// Guarantee the host-driven decode pair.
    pub fn servable(mut self) -> Self {
        self.servable = true;
        self
    }
    /// Build for `tokens` of context. See [`Self::max_seq`].
    pub fn max_seq(mut self, tokens: usize) -> Self {
        self.max_seq = Some(tokens);
        self
    }
    /// The mmproj path, if any.
    pub fn mmproj_path(&self) -> Option<&Path> {
        self.mmproj.as_deref()
    }
}

/// A constructor: checkpoint path + options in, boxed runner out.
pub type LmRunnerCtor = fn(&Path, &LmBuildOptions) -> Result<Box<dyn LmRunner>>;

/// One family's entry in the constructor registry.
///
/// `#[non_exhaustive]`, so construct it with [`Self::new`] and the builder
/// methods rather than a struct literal: this type will grow as families
/// declare more capabilities, and a literal would break every registrant on
/// each addition.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub struct LmRunnerRegistration {
    /// Short runner name, matching [`crate::model_registry`]'s `runner` field —
    /// that is what ties an architecture to this constructor.
    pub runner: &'static str,
    pub build: LmRunnerCtor,
    /// Whether this family's constructor accepts
    /// [`LmBuildOptions::mmproj`].
    ///
    /// Declared rather than discovered: it is a static property of the family,
    /// and it used to be four hand-written `if mmproj.is_some() { bail!(..) }`
    /// checks scattered through the dispatch — which meant a newly registered
    /// family silently *ignored* an mmproj it could not use.
    pub supports_mmproj: bool,
    /// Appended to the refusal when a caller passes an mmproj to a family that
    /// does not take one. Use it to say what to do instead.
    pub mmproj_note: Option<&'static str>,
}

impl LmRunnerRegistration {
    /// A text-only family: no mmproj.
    pub const fn new(runner: &'static str, build: LmRunnerCtor) -> Self {
        LmRunnerRegistration {
            runner,
            build,
            supports_mmproj: false,
            mmproj_note: None,
        }
    }

    /// Declare that the constructor handles [`LmBuildOptions::mmproj`].
    pub const fn with_mmproj(mut self) -> Self {
        self.supports_mmproj = true;
        self
    }

    /// Explain what to do instead, for a family that takes no mmproj.
    pub const fn mmproj_note(mut self, note: &'static str) -> Self {
        self.mmproj_note = Some(note);
        self
    }
}

impl std::fmt::Debug for LmRunnerRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LmRunnerRegistration")
            .field("runner", &self.runner)
            .finish_non_exhaustive()
    }
}

fn registry() -> &'static RwLock<BTreeMap<&'static str, LmRunnerRegistration>> {
    static R: OnceLock<RwLock<BTreeMap<&'static str, LmRunnerRegistration>>> = OnceLock::new();
    R.get_or_init(|| RwLock::new(BTreeMap::new()))
}

/// Register a constructor. Idempotent — the same name replaces the prior entry.
pub fn register_lm_runner(reg: LmRunnerRegistration) {
    registry()
        .write()
        .expect("lm runner registry poisoned")
        .insert(reg.runner, reg);
}

/// Whether a constructor is registered for `runner`.
pub fn lm_runner_is_registered(runner: &str) -> bool {
    registry()
        .read()
        .expect("lm runner registry poisoned")
        .contains_key(runner)
}

/// Every registered runner name, sorted.
///
/// Enumerable on purpose: a test can check that each one is reachable from the
/// architecture tables without anybody maintaining a list by hand.
pub fn registered_lm_runners() -> Vec<&'static str> {
    registry()
        .read()
        .expect("lm runner registry poisoned")
        .keys()
        .copied()
        .collect()
}

/// Whether `runner` accepts an mmproj vision encoder.
///
/// `None` when nothing is registered for the name. Queryable so a caller or a
/// CLI can decide *before* building rather than discovering it from an error.
pub fn lm_runner_supports_mmproj(runner: &str) -> Option<bool> {
    registry()
        .read()
        .expect("lm runner registry poisoned")
        .get(runner)
        .map(|r| r.supports_mmproj)
}

/// Build a runner by name.
///
/// Errors when nothing is registered for `runner`, listing what is — an
/// unregistered name is almost always a missing cargo feature or a crate whose
/// registration never ran.
pub fn build_lm_runner(
    runner: &str,
    path: &Path,
    opts: &LmBuildOptions,
) -> Result<Box<dyn LmRunner>> {
    let reg = {
        let g = registry().read().expect("lm runner registry poisoned");
        g.get(runner).copied()
    };
    let Some(reg) = reg else {
        let known = registered_lm_runners();
        bail!(
            "no LmRunner constructor registered for `{runner}` \
             (registered: {}) — enable the family's cargo feature, or call \
             `register_lm_runner` from the crate that implements it",
            if known.is_empty() {
                "none".to_string()
            } else {
                known.join(", ")
            }
        );
    };
    if opts.mmproj.is_some() && !reg.supports_mmproj {
        // One check instead of one per family, so a family that forgets it
        // cannot silently ignore a vision encoder the caller asked for.
        let note = reg
            .mmproj_note
            .map(|n| format!(" — {n}"))
            .unwrap_or_default();
        bail!("`{runner}` does not accept an mmproj vision encoder{note}");
    }
    let inner = (reg.build)(path, opts)?;
    Ok(if opts.servable {
        Box::new(crate::lm_host_driven::HostDriven::new(inner))
    } else {
        inner
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(&'static str);
    impl LmRunner for Fake {
        fn family(&self) -> &'static str {
            self.0
        }
        fn vocab_size(&self) -> usize {
            7
        }
        fn predict_logits(&mut self, _p: &[u32]) -> Result<Vec<f32>> {
            Ok(vec![0.0; 7])
        }
    }

    /// An out-of-tree crate can register and then be built generically.
    #[test]
    fn a_registered_constructor_is_reachable_by_name() {
        register_lm_runner(LmRunnerRegistration::new("test-external", |_p, _o| {
            Ok(Box::new(Fake("test-external")))
        }));
        assert!(lm_runner_is_registered("test-external"));
        assert!(registered_lm_runners().contains(&"test-external"));

        let r = build_lm_runner(
            "test-external",
            Path::new("/nowhere"),
            &LmBuildOptions::default(),
        )
        .expect("the registered constructor should run");
        assert_eq!(r.family(), "test-external");
    }

    /// `servable` wraps whatever the constructor returned, so a family that
    /// implements only `predict_logits` is still drivable.
    #[test]
    fn servable_wraps_the_constructed_runner() {
        register_lm_runner(LmRunnerRegistration::new("test-predict-only", |_p, _o| {
            Ok(Box::new(Fake("test-predict-only")))
        }));
        // Unwrapped: the host-driven pair is unsupported.
        let mut bare = build_lm_runner(
            "test-predict-only",
            Path::new("/nowhere"),
            &LmBuildOptions::default(),
        )
        .unwrap();
        assert!(
            bare.prefill_logits(&[1]).is_err(),
            "bare runner has no pair"
        );

        let mut wrapped = build_lm_runner(
            "test-predict-only",
            Path::new("/nowhere"),
            &LmBuildOptions::default().servable(),
        )
        .unwrap();
        assert_eq!(
            wrapped.prefill_logits(&[1]).expect("wrapped serves").len(),
            7
        );
    }

    /// An unregistered name says what is registered and what to do.
    #[test]
    fn an_unregistered_name_is_an_informative_error() {
        let err = build_lm_runner("nope", Path::new("/x"), &LmBuildOptions::default())
            .err()
            .expect("nothing registered for `nope`");
        let msg = format!("{err:#}");
        assert!(msg.contains("nope"), "{msg}");
        assert!(msg.contains("register_lm_runner"), "{msg}");
    }

    /// Re-registering replaces, so a host can override a builtin.
    #[test]
    fn registering_the_same_name_twice_replaces() {
        register_lm_runner(LmRunnerRegistration::new("test-dup", |_p, _o| {
            Ok(Box::new(Fake("first")))
        }));
        register_lm_runner(LmRunnerRegistration::new("test-dup", |_p, _o| {
            Ok(Box::new(Fake("second")))
        }));
        let r = build_lm_runner("test-dup", Path::new("/x"), &LmBuildOptions::default()).unwrap();
        assert_eq!(r.family(), "second", "the later registration wins");
    }

    /// `Default` must mean "unspecified", not "CPU".
    ///
    /// The previous version of this test asserted `device == Device::Cpu`, which
    /// locked the bug in rather than catching it: every `auto_runner*`
    /// convenience builds `LmBuildOptions::default()`, so every caller that did
    /// not name a device silently ran on CPU on a machine with a GPU — the
    /// opposite of what the `auto_` prefix promises. A passing test asserting
    /// the wrong default is worse than no test, because it answers the question
    /// before anyone thinks to ask it.
    #[test]
    fn an_unset_device_means_unspecified_not_cpu() {
        let d = LmBuildOptions::default();
        assert!(d.device.is_none(), "the default must not pin a device");
        assert_eq!(
            d.device_or_auto(),
            crate::device_capabilities::pick_lm_device(),
            "an unset device resolves to this machine's pick"
        );
        assert!(d.mmproj_path().is_none());
        assert!(!d.servable);
        assert!(d.max_seq.is_none());

        let o = LmBuildOptions::new(Device::Metal).mmproj("m.gguf").servable();
        assert_eq!(o.device, Some(Device::Metal));
        assert_eq!(o.device_or_auto(), Device::Metal);
        assert_eq!(o.mmproj_path(), Some(Path::new("m.gguf")));
        assert!(o.servable);

        // The distinction that motivates the `Option`: an explicit CPU request
        // must stay CPU and never be re-resolved to a GPU.
        assert_eq!(
            LmBuildOptions::new(Device::Cpu).device_or_auto(),
            Device::Cpu,
            "explicit CPU is a choice, not an absence of one"
        );
    }
}

#[cfg(test)]
mod capability_tests {
    use super::*;

    struct Vl;
    impl LmRunner for Vl {
        fn family(&self) -> &'static str {
            "vl"
        }
        fn vocab_size(&self) -> usize {
            3
        }
        fn predict_logits(&mut self, _p: &[u32]) -> Result<Vec<f32>> {
            Ok(vec![0.0; 3])
        }
        fn supports_multimodal(&self) -> bool {
            true
        }
    }

    /// An mmproj handed to a text-only family is refused centrally, with the
    /// family's own note.
    ///
    /// This used to be four hand-written checks inside the dispatch, which
    /// meant a newly registered family silently *ignored* an mmproj the caller
    /// asked for — the worst of the three outcomes, because the request looks
    /// honoured.
    #[test]
    fn an_mmproj_for_a_text_only_family_is_refused_with_its_note() {
        register_lm_runner(
            LmRunnerRegistration::new("test-textonly", |_p, _o| Ok(Box::new(Vl)))
                .mmproj_note("use the VL variant instead"),
        );
        assert_eq!(lm_runner_supports_mmproj("test-textonly"), Some(false));

        let opts = LmBuildOptions::default().mmproj("m.gguf");
        let err = build_lm_runner("test-textonly", Path::new("/x"), &opts)
            .err()
            .expect("a text-only family must refuse an mmproj");
        let msg = format!("{err:#}");
        assert!(msg.contains("does not accept an mmproj"), "{msg}");
        assert!(msg.contains("use the VL variant instead"), "note: {msg}");

        // Without the mmproj it builds.
        assert!(
            build_lm_runner("test-textonly", Path::new("/x"), &LmBuildOptions::default()).is_ok()
        );
    }

    /// A family that declares `with_mmproj` receives it.
    #[test]
    fn a_vl_family_receives_the_mmproj() {
        register_lm_runner(
            LmRunnerRegistration::new("test-vl", |_p, o| {
                // The constructor sees the path it was promised.
                assert_eq!(o.mmproj_path(), Some(Path::new("m.gguf")));
                Ok(Box::new(Vl))
            })
            .with_mmproj(),
        );
        assert_eq!(lm_runner_supports_mmproj("test-vl"), Some(true));
        let opts = LmBuildOptions::default().mmproj("m.gguf");
        assert!(build_lm_runner("test-vl", Path::new("/x"), &opts).is_ok());
    }

    /// Capability of an unregistered name is unknown, not `false`.
    #[test]
    fn an_unregistered_name_has_no_capability_answer() {
        assert_eq!(lm_runner_supports_mmproj("test-nothing-here"), None);
    }
}
