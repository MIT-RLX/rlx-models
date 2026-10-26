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

//! Which devices an upscaler will accept.
//!
//! The standard backend set, plus **`ane`** / **`coreml`** when the crate is
//! built with the `coreml` feature.
//!
//! `rlx_cli::parse_standard_device` rejects `Device::Ane` outright, and
//! `parse_lm_device` accepts it but under language-model semantics these
//! networks do not have. Every architecture here is statically supported on
//! CoreML (`scripts/upscale_backend_matrix.py`), so the right answer is a small
//! parser of this crate's own.

use anyhow::{Result, bail};
use rlx_runtime::Device;

/// Parse a `--device` argument for an upscaler.
pub fn parse_upscale_device(s: &str) -> Result<Device> {
    match s {
        "ane" | "coreml" | "neural-engine" => {
            if cfg!(feature = "coreml") {
                Ok(Device::Ane)
            } else {
                bail!(
                    "device {s:?} needs the `coreml` feature — rebuild with \
                     `--features coreml` (or `macos-gpu`, which includes it)"
                )
            }
        }
        other => rlx_cli::parse_standard_device("upscale", other),
    }
}

/// Accept a [`Device`] that was not obtained from [`parse_upscale_device`].
///
/// The parser is only reached from a CLI; a library caller constructs a
/// `Device` directly. Both paths have to agree, or `--device ane` parses and
/// then fails one layer down with a message naming the *standard* device set —
/// which is what happened before this existed.
pub fn validate_upscale_device(device: Device) -> Result<()> {
    if device == Device::Ane {
        if cfg!(feature = "coreml") {
            return Ok(());
        }
        bail!("upscale: device Ane needs the `coreml` feature");
    }
    rlx_core::validate_standard_device("upscale", device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_standard_devices_still_parse() {
        assert_eq!(parse_upscale_device("cpu").unwrap(), Device::Cpu);
        assert!(parse_upscale_device("metal").is_ok());
        assert!(parse_upscale_device("vulkan").is_ok());
    }

    /// An unbuilt backend must say *which flag to add*, not merely that the
    /// name is unknown — the two failures look identical from the CLI and have
    /// completely different fixes.
    #[test]
    fn ane_is_gated_on_the_feature_and_says_so() {
        let r = parse_upscale_device("ane");
        if cfg!(feature = "coreml") {
            assert_eq!(r.unwrap(), Device::Ane);
        } else {
            let e = format!("{:#}", r.expect_err("ane without the feature must fail"));
            assert!(e.contains("`coreml` feature"), "unhelpful error: {e}");
        }
    }

    /// Parsing and validating must agree: a device the parser accepts has to
    /// survive `Upscaler::from_checkpoint`, which validates independently.
    #[test]
    fn every_parsed_device_also_validates() {
        for name in ["cpu", "metal", "mlx", "vulkan", "gpu", "ane", "coreml"] {
            if let Ok(d) = parse_upscale_device(name) {
                assert!(
                    validate_upscale_device(d).is_ok(),
                    "{name} parses to {d:?} but does not validate"
                );
            }
        }
    }

    #[test]
    fn an_unknown_device_is_rejected() {
        assert!(parse_upscale_device("quantum").is_err());
    }
}
