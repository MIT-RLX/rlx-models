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

//! Geometry helpers for line polygons and edge extraction.

use rten_imageproc::BoundingRect;
use rten_imageproc::{LineF, RotatedRect};

/// Downward-pointing copy of a line (positive Y).
pub fn downwards_line(line: LineF) -> LineF {
    if line.end.y >= line.start.y {
        line
    } else {
        LineF::from_endpoints(line.end, line.start)
    }
}

/// Left edge of an oriented word rectangle (matches upstream `ocrs` `geom_util`).
pub fn leftmost_edge(r: &RotatedRect) -> LineF {
    let mut corners = r.corners();
    corners.sort_by(|a, b| a.x.total_cmp(&b.x));
    LineF::from_endpoints(corners[0], corners[1])
}

/// Right edge of an oriented word rectangle (matches upstream `ocrs` `geom_util`).
pub fn rightmost_edge(r: &RotatedRect) -> LineF {
    let mut corners = r.corners();
    corners.sort_by(|a, b| a.x.total_cmp(&b.x));
    LineF::from_endpoints(corners[2], corners[3])
}

/// Midpoint line between first and last word in a text line.
pub fn midpoint_line(words: &[RotatedRect]) -> LineF {
    assert!(!words.is_empty());
    LineF::from_endpoints(
        words.first().unwrap().bounding_rect().left_edge().center(),
        words.last().unwrap().bounding_rect().right_edge().center(),
    )
}
