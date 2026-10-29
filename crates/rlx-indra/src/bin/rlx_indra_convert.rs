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

//! `rlx-indra-convert <upstream-dir> <out-dir>`: export the upstream Indra
//! release (`datalab-to/indra-ocr`) to native safetensors (see
//! [`rlx_indra::convert`]).

use anyhow::{Context, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: rlx-indra-convert <upstream-dir> <out-dir>";
    let src = PathBuf::from(args.next().context(usage)?);
    let dst = PathBuf::from(args.next().context(usage)?);
    for f in rlx_indra::convert::convert_dir(&src, &dst)? {
        println!("{f}");
    }
    Ok(())
}
