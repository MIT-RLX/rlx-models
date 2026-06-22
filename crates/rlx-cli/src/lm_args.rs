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

//! Shared CLI flags for every per-family LM binary.
//!
//! Each `rlx-<family>/src/cli.rs` today hand-rolls the same
//! `while i < args.len()` loop parsing `--weights / --device / --max-seq
//! / --max-tokens / --prompt / --prompt-ids / --tokenizer / --temperature
//! / --top-p / --format / --packed / --no-stream / --max-memory-gb`.
//!
//! [`LmCliArgs`] is a `clap`-derived struct that captures the shared set
//! and provides [`LmCliArgs::into_builder`] which seeds a generic
//! [`rlx_runtime::LmRunnerBuilder`]. Per-family CLIs can mix in their
//! own structs with `#[command(flatten)]` for arch-specific flags.

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::Result;
use clap::Parser;
use rlx_runtime::{ConfigSource, Device, LmRunnerBuilder, SampleOpts, WeightFormat};

/// Canonical LM CLI flags.
#[derive(Debug, Clone, Parser)]
pub struct LmCliArgs {
    /// Weights file (`.safetensors` / `.gguf`) or directory.
    #[arg(long)]
    pub weights: PathBuf,

    /// Inference device.
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Override the auto-detected weight format.
    #[arg(long, value_parser = parse_format)]
    pub format: Option<WeightFormat>,

    /// Path to a HF `config.json` (default: sibling of `--weights`).
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Prompt text (tokenized via `--tokenizer`).
    #[arg(long)]
    pub prompt: Option<String>,

    /// Pre-tokenized comma-separated u32 ids.
    #[arg(long, value_delimiter = ',')]
    pub prompt_ids: Option<Vec<u32>>,

    /// Tokenizer file (`tokenizer.json`) for `--prompt` / decode.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,

    /// Tokens to generate.
    #[arg(long, default_value_t = 32)]
    pub max_tokens: usize,

    /// Maximum prefill sequence length.
    #[arg(long, default_value_t = 128)]
    pub max_seq: usize,

    /// Refuse to load if F32-dequant estimate exceeds this many GB.
    #[arg(long)]
    pub max_memory_gb: Option<f32>,

    /// Disable streaming (print only the final string).
    #[arg(long)]
    pub no_stream: bool,

    /// Force packed GGUF loading (`Op::DequantMatMul`).
    #[arg(long)]
    pub packed: bool,

    /// Disable packed GGUF loading (overrides auto-detection).
    #[arg(long, conflicts_with = "packed")]
    pub no_packed: bool,

    /// Sampling temperature. `0` = greedy.
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    /// Nucleus sampling top-p.
    #[arg(long, default_value_t = 1.0)]
    pub top_p: f32,

    /// Top-k sampling cutoff.
    #[arg(long)]
    pub top_k: Option<u32>,

    /// Repetition penalty.
    #[arg(long, default_value_t = 1.0)]
    pub repetition_penalty: f32,

    /// GGUF quant preference (e.g. `Q4_K_M`) when `--weights` is a directory.
    #[arg(long, alias = "prefer")]
    pub prefer_gguf: Option<String>,
}

fn parse_format(s: &str) -> Result<WeightFormat, String> {
    WeightFormat::parse(s).map_err(|e| e.to_string())
}

impl LmCliArgs {
    /// Parse a `Device` from the `--device` string using the upstream
    /// `FromStr for Device` impl.
    pub fn device(&self) -> Result<Device> {
        Device::from_str(&self.device).map_err(|e| anyhow::anyhow!("--device {}: {e}", self.device))
    }

    /// Build a sampling option set from the relevant flags. Starts from
    /// `SampleOpts::greedy()` so the new advanced-sampler fields take
    /// their defaults; only the CLI-bound knobs are overwritten.
    pub fn sample_opts(&self) -> SampleOpts {
        let mut s = SampleOpts::greedy();
        s.temperature = self.temperature;
        s.top_p = self.top_p;
        s.top_k = self.top_k;
        s.repetition_penalty = self.repetition_penalty;
        s
    }

    /// Construct an [`LmRunnerBuilder`] pre-populated from the flags.
    /// Per-family runners that wrap [`LmRunnerBuilder`] can call this
    /// once and then layer family-specific options on top.
    pub fn into_builder<Cfg>(self) -> Result<LmRunnerBuilder<Cfg>> {
        let device = self.device()?;
        let packed = if self.packed {
            Some(true)
        } else if self.no_packed {
            Some(false)
        } else {
            None
        };
        let config = self
            .config
            .clone()
            .map(ConfigSource::JsonFile)
            .unwrap_or(ConfigSource::Embedded);

        let mut b = LmRunnerBuilder::<Cfg>::new()
            .weights(self.weights.clone())
            .device(device)
            .max_seq(self.max_seq)
            .stream(!self.no_stream)
            .sample(self.sample_opts())
            .config(config);
        b.format = self.format;
        b.packed_weights = packed;
        b.max_memory_gb = self.max_memory_gb;
        b.prefer_gguf = self.prefer_gguf.clone();
        Ok(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn debug_assert_works() {
        LmCliArgs::command().debug_assert();
    }

    #[test]
    fn defaults() {
        let a = LmCliArgs::try_parse_from(["x", "--weights", "/tmp/m.gguf"]).unwrap();
        assert_eq!(a.device, "cpu");
        assert_eq!(a.max_seq, 128);
        assert_eq!(a.max_tokens, 32);
        assert!(!a.no_stream);
        assert_eq!(a.temperature, 0.0);
    }

    #[test]
    fn packed_conflict() {
        let r =
            LmCliArgs::try_parse_from(["x", "--weights", "/tmp/m.gguf", "--packed", "--no-packed"]);
        assert!(r.is_err());
    }

    #[test]
    fn builder_propagates_packed_override() {
        let a =
            LmCliArgs::try_parse_from(["x", "--weights", "/tmp/m.gguf", "--no-packed"]).unwrap();
        let b: LmRunnerBuilder<()> = a.into_builder().unwrap();
        assert_eq!(b.packed_weights, Some(false));
    }
}
