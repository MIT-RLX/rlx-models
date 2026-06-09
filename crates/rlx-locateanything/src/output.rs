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

//! Print decoded grounding output (CLI and examples).

use crate::parse::GroundingParse;

/// Print model text and parsed refs / boxes / points to stdout and stderr.
pub fn print_grounding(result: &GroundingParse) {
    println!("{}", result.text);
    if !result.raw.is_empty() && result.raw != result.text {
        eprintln!("[decode-raw] {}", result.raw);
    }
    for label in &result.refs {
        eprintln!("ref: {label}");
    }
    for b in &result.boxes {
        eprintln!("box: ({:.1},{:.1})-({:.1},{:.1})", b.x1, b.y1, b.x2, b.y2);
    }
    for p in &result.points {
        eprintln!("point: ({:.1},{:.1})", p.x, p.y);
    }
}
