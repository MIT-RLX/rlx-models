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

//! Export Maya1 as an `.rlxp` package rlx runs alone — text in, 24 kHz
//! speech out, the voice a description (`options.voice`) — through
//! Orpheus's exporter ([`rlx_orpheus::export::export_flavor`]): Maya1's
//! tokens are Orpheus's, its prompt body `<description="…"> text`, its
//! decoding greedy.

use std::path::Path;

use anyhow::Result;
use rlx_orpheus::export::{Flavor, export_flavor};

pub use rlx_orpheus::export::{ExportOptions, ExportReport};

/// The voice when a request describes none (the command's).
pub const DEFAULT_DESCRIPTION: &str = "Realistic male voice in his 30s with an American accent. \
     Normal pitch, warm timbre, conversational pacing.";

/// Export the Maya1 GGUF `gguf` (SNAC's decoder beside it) as the package
/// `out`.
pub fn export_rlxp(gguf: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let flavor = Flavor {
        family: "maya1",
        template: "<description=\"{voice}\"> {text}",
        voice: Some(DEFAULT_DESCRIPTION),
        voices: &[],
        config: crate::maya1_config(),
    };
    export_flavor(gguf, out, opts, &flavor)
}

/// Register this crate's exporter ([`rlx_orpheus::export`]'s registry).
pub fn register() {
    rlx_orpheus::export::register_named("maya1", export_rlxp);
}
