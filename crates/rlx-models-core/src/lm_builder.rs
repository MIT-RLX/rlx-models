// RLX models — shared LM runner-builder state.
// Copyright (C) 2026 Eugene Hauptmann.
// SPDX-License-Identifier: GPL-3.0-only

//! The part of every family's `*RunnerBuilder` that is the same part.
//!
//! `rlx-gemma`, `rlx-llama32` and `rlx-qwen3` each declared the same nine
//! fields and wrote the same nine setters over them — three times, with the
//! defaults free to drift (and they had: `stream` was `true` in one and
//! `false` in the two that reached it through `#[derive(Default)]`).
//!
//! ## Why not `rlx_runtime::LmRunnerBuilder`
//!
//! Upstream already has a builder with these field *names*, and it has
//! never had a single user in this workspace. Two reasons, and the second
//! one is the one that actually blocks it.
//!
//! It stores **resolved** values — `device: Device`, `max_seq: usize` —
//! where the family builders store `Option<_>`, and for `device` that is
//! the whole point: `None` means "pick one for this machine"
//! ([`pick_lm_device`]), and no value of `Device` means that. Adopting it
//! would quietly pin every family to `Device::Cpu` whenever the caller did
//! not name one. So this type keeps the options, and
//! [`LmBuilderCommon::device_or_auto`] is where "unset" becomes a device —
//! once, with the log line that goes with it.
//!
//! The harder reason: it is typed against a **different `SampleOpts` and
//! `WeightFormat`** than any family uses. This workspace has two of each —
//! `rlx_qwen3::SampleOpts` (`rlx-qwen3/src/sampling.rs`) alongside
//! `rlx_runtime::SampleOpts`, and `rlx_cli::WeightFormat`
//! (`rlx-cli/src/format.rs`) alongside `rlx_runtime::WeightFormat` — same
//! names, distinct types. Every family builder uses the first of each pair;
//! upstream's builder uses the second. Adopting it means converting between
//! two identically-named types at every boundary, which is why it sat
//! unused rather than why nobody got around to it.
//!
//! ## What is deliberately *not* here
//!
//! **`sample` and `format`**, for exactly that reason: their types live in
//! `rlx-qwen3` and `rlx-cli`, both of which depend on this crate, so naming
//! them here would be a dependency cycle. Each family keeps those two
//! fields and setters. Collapsing the two `SampleOpts` (and the two
//! `WeightFormat`) into one type each is the prerequisite for finishing
//! this, and it is a wider change than a builder cleanup.
//!
//! **`packed_weights`**, because the families disagree about what it means:
//! gemma and llama32 auto-enable it for a GGUF over 256 MiB (and gemma
//! forces it off for multimodal), while qwen3 defaults it off with no
//! auto-detection at all. A real difference in behaviour rather than
//! duplicated code.

use crate::device_capabilities::pick_lm_device;
use anyhow::{Result, anyhow, bail};
use rlx_runtime::{ConfigSource, Device};
use std::path::{Path, PathBuf};

/// The fields every LM family's builder has, with "unset" preserved.
#[derive(Debug, Clone)]
pub struct LmBuilderCommon<Cfg> {
    pub weights: Option<PathBuf>,
    pub config: Option<ConfigSource<Cfg>>,
    /// `None` = pick a device for this machine; see [`Self::device_or_auto`].
    pub device: Option<Device>,
    pub max_seq: Option<usize>,
    pub max_memory_gb: Option<f32>,
    /// Accepted and ignored by every family. Kept because the setter is
    /// public API; see the `stream` setters for the history.
    pub stream: bool,
}

// Hand-written rather than derived: `#[derive(Default)]` would demand
// `Cfg: Default`, and no family config is.
impl<Cfg> Default for LmBuilderCommon<Cfg> {
    fn default() -> Self {
        Self {
            weights: None,
            config: None,
            device: None,
            max_seq: None,
            max_memory_gb: None,
            stream: true,
        }
    }
}

impl<Cfg> LmBuilderCommon<Cfg> {
    pub fn new() -> Self {
        Self::default()
    }

    /// The weights path, or the error every family used to spell out itself.
    pub fn weights_path(&self) -> Result<&Path> {
        self.weights
            .as_deref()
            .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))
    }

    /// The caller's device, or one picked for this machine — logged when
    /// picked, because "it ran on the CPU" is otherwise invisible.
    ///
    /// `tag` names the family in that line (`gemma`, `llama32`, …).
    pub fn device_or_auto(&self, tag: &str) -> Device {
        match self.device {
            Some(d) => d,
            None => {
                let d = pick_lm_device();
                eprintln!("[{tag}-runner] auto device → {d:?}");
                d
            }
        }
    }

    /// The caller's device, or `default` with no auto-detection.
    ///
    /// Both this and [`Self::device_or_auto`] exist because the families
    /// genuinely disagree, and the disagreement is worth being able to see:
    /// `gemma` picks the best device for the machine and says so, while
    /// `llama32`, `qwen3` and `qwen35` quietly fall back to `Device::Cpu`.
    /// So `.build()` without `.device(...)` runs one family on the GPU and
    /// three on the CPU — the kind of difference that reads as "gemma is
    /// mysteriously faster". Spelling the policy at each call site keeps it
    /// visible rather than buried in an `unwrap_or`; converging them is a
    /// behaviour change for three crates and belongs in its own commit.
    pub fn device_or(&self, default: Device) -> Device {
        self.device.unwrap_or(default)
    }

    pub fn max_seq_or(&self, default: usize) -> usize {
        self.max_seq.unwrap_or(default)
    }

    /// Refuse a model that would not fit the caller's stated budget.
    ///
    /// `est_bytes` is what the weights would occupy dequantized to F32,
    /// which is the number worth checking: a 4-bit GGUF that fits on disk
    /// is 8x that in an F32 arena.
    pub fn check_memory_cap(&self, est_bytes: u64) -> Result<()> {
        if let Some(cap_gb) = self.max_memory_gb {
            let est_gb = est_bytes as f32 / (1024.0 * 1024.0 * 1024.0);
            if est_gb > cap_gb {
                // qwen3's copy of this check carried the second sentence and
                // the other two did not; the advice is good for all three.
                bail!(
                    "weights would dequant to ~{est_gb:.1} GB at F32, exceeds cap \
                     {cap_gb:.1} GB. Either raise the cap (--max-memory-gb) or pick a \
                     smaller / more aggressively quantized model."
                );
            }
        }
        Ok(())
    }
}

/// The setters every LM family's builder has, over [`LmBuilderCommon`].
///
/// Expand inside the family's own `impl` block, where the builder has a
/// `common: LmBuilderCommon<$cfg>` field:
///
/// ```ignore
/// impl GemmaRunnerBuilder {
///     rlx_core::lm_builder_setters!(GemmaConfig);
///     // family-specific setters follow as usual
/// }
/// ```
///
/// Deliberately not covering `sample`, `format` or `packed_weights` — see
/// the module docs for why each is the family's own.
#[macro_export]
macro_rules! lm_builder_setters {
    ($cfg:ty) => {
        /// Weights file (`.safetensors` / `.gguf`), or a directory to
        /// resolve one from.
        pub fn weights<P: Into<std::path::PathBuf>>(mut self, path: P) -> Self {
            self.common.weights = Some(path.into());
            self
        }

        /// Where to read the config from. Default: the family's embedded
        /// default for a GGUF (its metadata carries one), or the
        /// `config.json` beside the weights for safetensors.
        pub fn config(mut self, src: $crate::reexport::ConfigSource<$cfg>) -> Self {
            self.common.config = Some(src);
            self
        }

        /// Shorthand for [`Self::config`] with an explicit config value.
        pub fn config_value(self, cfg: $cfg) -> Self {
            self.config($crate::reexport::ConfigSource::Explicit(cfg))
        }

        /// Inference device. Unset picks one for this machine.
        pub fn device(mut self, d: $crate::reexport::Device) -> Self {
            self.common.device = Some(d);
            self
        }

        /// Longest sequence the compiled graphs must handle.
        pub fn max_seq(mut self, n: usize) -> Self {
            self.common.max_seq = Some(n);
            self
        }

        /// Refuse to build if the weights would exceed this many GB
        /// dequantized to F32.
        pub fn max_memory_gb(mut self, gb: f32) -> Self {
            self.common.max_memory_gb = Some(gb);
            self
        }

        /// Accepted and ignored.
        ///
        /// Every generate path now invokes the caller's per-token callback
        /// per token — it is the only sink for the ids and it carries the
        /// stop signal — so there is no longer a non-streaming mode to
        /// select. The setter is kept so existing callers compile.
        pub fn stream(mut self, on: bool) -> Self {
            self.common.stream = on;
            self
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct Cfg {
        n: usize,
    }

    #[test]
    fn unset_device_stays_unset_so_auto_pick_is_reachable() {
        // The reason this type keeps `Option<Device>` instead of reusing
        // `rlx_runtime::LmRunnerBuilder`'s resolved `Device`.
        let c = LmBuilderCommon::<Cfg>::new();
        assert!(c.device.is_none(), "unset must be distinguishable");
        assert_eq!(c.device_or_auto("test"), pick_lm_device());

        let c = LmBuilderCommon::<Cfg> {
            device: Some(Device::Cpu),
            ..Default::default()
        };
        assert_eq!(c.device_or_auto("test"), Device::Cpu);
    }

    #[test]
    fn the_two_device_policies_are_both_reachable_and_differ() {
        let unset = LmBuilderCommon::<Cfg>::new();
        assert_eq!(unset.device_or(Device::Cpu), Device::Cpu);
        assert_eq!(unset.device_or_auto("test"), pick_lm_device());

        // Once the caller names one, the policy stops mattering.
        let named = LmBuilderCommon::<Cfg> {
            device: Some(Device::Cpu),
            ..Default::default()
        };
        assert_eq!(named.device_or(Device::Metal), Device::Cpu);
        assert_eq!(named.device_or_auto("test"), Device::Cpu);
    }

    #[test]
    fn defaults_are_one_set_of_defaults() {
        let c = LmBuilderCommon::<Cfg>::default();
        assert!(c.weights.is_none());
        assert!(c.config.is_none());
        assert!(c.max_seq.is_none());
        assert!(c.max_memory_gb.is_none());
        assert_eq!(c.max_seq_or(128), 128);
    }

    #[test]
    fn a_missing_weights_path_names_the_setter_to_call() {
        let e = LmBuilderCommon::<Cfg>::new().weights_path().unwrap_err();
        assert!(e.to_string().contains(".weights("), "got: {e}");
    }

    #[test]
    fn the_memory_cap_compares_f32_bytes_not_file_bytes() {
        let c = LmBuilderCommon::<Cfg> {
            max_memory_gb: Some(1.0),
            ..Default::default()
        };
        // 2 GB dequantized against a 1 GB cap.
        let e = c.check_memory_cap(2 * 1024 * 1024 * 1024).unwrap_err();
        assert!(e.to_string().contains("exceeds cap"), "got: {e}");
        assert!(
            e.to_string().contains("--max-memory-gb"),
            "the message should say what to do about it: {e}"
        );
        assert!(c.check_memory_cap(512 * 1024 * 1024).is_ok());
        // No cap set: anything passes.
        let c = LmBuilderCommon::<Cfg>::new();
        assert!(c.check_memory_cap(u64::MAX).is_ok());
    }
}
