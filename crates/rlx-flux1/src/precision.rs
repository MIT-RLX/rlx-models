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

//! Number formats for FLUX.1 (`--number-format`).
//!
//! The format is published process-wide (`RLX_NUMBER_FORMAT`) so every graph
//! the pipeline compiles — T5, CLIP, the denoiser chunks, the VAE — rounds
//! weights on upload and every intermediate in-graph (see `rlx_ir::NumberFormat`).
//! On top of the generic rules, each component gets the policy that measured
//! best for it (`examples/format_report.rs`, relative error vs f32, Metal):
//!
//! | component | rule | f16 | bf16 | f8 |
//! |---|---|---|---|---|
//! | T5-XXL | residual stream + matmul outputs in f32 | 97 % → 0.22 % | 2.8e-2 → 1.2e-2 | 29 % → 19 % |
//! | denoiser | plain `f8` = e2m5 weights / e3m4 activations | — | — | 6.7 % → 3.6 % |
//!
//! T5 v1.1's activations exceed f16's range (HF keeps its `wo` in f32 for the
//! same reason); both the residual stream and the matmul outputs overflow, so
//! both stay f32 — f16 operands, f32 accumulation and output, the usual
//! mixed-precision split. In the denoiser, 16-element block scales leave the
//! f8 exponent bits nearly idle, so mantissa bits buy more: e2m5 weights are
//! 3.4× more precise than e4m3 (0.72 % vs 2.5 % weights-only).
//!
//! A user-set `RLX_NUMBER_FORMAT_KEEP` / `_KEEP_PARAMS` / `_ROUND_RESIDUAL`, or
//! an explicit `fNeXmY` spelling other than plain `f8`, is left alone.

use anyhow::Result;
use rlx_runtime::NumberFormat;

/// Setting that overrides FLUX's keep list (comma-separated substrings).
pub const KEEP_PARAMS_ENV: &str = "RLX_NUMBER_FORMAT_KEEP_PARAMS";
const FORMAT_ENV: &str = "RLX_NUMBER_FORMAT";
const KEEP_OPS_ENV: &str = "RLX_NUMBER_FORMAT_KEEP";
const RESIDUAL_ENV: &str = "RLX_NUMBER_FORMAT_ROUND_RESIDUAL";

/// The networks the pipeline compiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    T5,
    Clip,
    Denoiser,
    Vae,
}

/// Narrowest format in use (weights or activations), in bits; `None` for f32/f64.
pub fn narrowest_bits(fmt: NumberFormat) -> Option<u32> {
    [fmt.weights(), fmt.activations()]
        .into_iter()
        .flatten()
        .map(|(f, _)| f.bits())
        .min()
}

/// The format `component` actually compiles under for the requested `fmt`.
pub fn component_format(fmt: NumberFormat, component: Component) -> NumberFormat {
    let plain_f8: NumberFormat = "f8".parse().expect("f8 parses");
    if component == Component::Denoiser && fmt == plain_f8 {
        return "f8e2m5/f8e3m4".parse().expect("tuned f8 parses");
    }
    fmt
}

/// Publish `fmt` for every later compile in this process. `None` restores f32.
pub fn apply(fmt: Option<NumberFormat>) -> Result<()> {
    rlx_core::lm_registry::apply_number_format(fmt)
}

/// The requested format, if any (from `--number-format` / `RLX_NUMBER_FORMAT`).
pub fn active() -> Option<NumberFormat> {
    rlx_ir::env::var(FORMAT_ENV)
        .and_then(|v| v.parse().ok())
        .filter(|f| narrowest_bits(*f).is_some())
}

/// Run `f` (a compile) under `component`'s precision policy, restoring the
/// process settings afterwards.
pub fn with_component<R>(component: Component, f: impl FnOnce() -> R) -> R {
    let Some(fmt) = active().filter(|_| !rlx_ir::env::flag("RLX_FLUX1_NO_POLICY")) else {
        return f();
    };
    let saved: Vec<(&str, Option<String>)> = [FORMAT_ENV, KEEP_OPS_ENV, RESIDUAL_ENV]
        .iter()
        .map(|k| (*k, rlx_ir::env::var(k)))
        .collect();
    let tuned = component_format(fmt, component);
    if tuned != fmt {
        rlx_ir::env::set(FORMAT_ENV, tuned.to_string());
    }
    if matches!(component, Component::T5 | Component::Clip) {
        if saved[1].1.is_none() {
            rlx_ir::env::set(KEEP_OPS_ENV, "MatMul");
        }
        if saved[2].1.is_none() {
            rlx_ir::env::set(RESIDUAL_ENV, "0");
        }
    }
    let out = f();
    for (k, v) in saved {
        match v {
            Some(v) => rlx_ir::env::set(k, v),
            None => rlx_ir::env::unset(k),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_f8_is_tuned_only_for_the_denoiser() {
        let f8: NumberFormat = "f8".parse().unwrap();
        let tuned: NumberFormat = "f8e2m5/f8e3m4".parse().unwrap();
        assert_eq!(component_format(f8, Component::Denoiser), tuned);
        assert_eq!(component_format(f8, Component::T5), f8);
        let explicit: NumberFormat = "f8e5m2".parse().unwrap();
        assert_eq!(component_format(explicit, Component::Denoiser), explicit);
    }
}
