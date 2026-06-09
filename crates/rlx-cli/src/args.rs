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

use anyhow::{Result, anyhow};

/// Read the next argv token after a flag (e.g. `--weights path`).
pub fn req(args: &[String], i: &mut usize) -> Result<String> {
    let flag = args[*i].clone();
    *i += 1;
    let v = args
        .get(*i)
        .ok_or_else(|| anyhow!("missing value for {flag}"))?
        .clone();
    *i += 1;
    Ok(v)
}
