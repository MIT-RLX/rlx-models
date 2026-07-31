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

//! `rlx-vlash` CLI — run a VLASH π₀ / π₀.₅ policy on an image + state + prompt
//! and print the predicted action chunk.

use anyhow::Result;

fn main() -> Result<()> {
    eprintln!(
        "rlx-vlash: VLASH π₀ / π₀.₅ VLA policy runner.\n\
         Runner CLI wiring is under construction (see src/runner.rs). \
         For now use the crate API + tests/parity.rs."
    );
    Ok(())
}
