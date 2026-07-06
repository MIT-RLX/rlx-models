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

//! Layout analysis — group detected words into reading-order lines (upstream `ocrs` algorithm).

mod empty_rects;

use empty_rects::{FilterOverlapping, max_empty_rects};
use rten_imageproc::bounding_rect;
use rten_imageproc::{BoundingRect, Line, LineF, Point, Rect, RotatedRect};

use crate::geom::{leftmost_edge, rightmost_edge};

fn rects_separated_by_line(a: &RotatedRect, b: &RotatedRect, l: LineF) -> bool {
    let a_to_b = LineF::from_endpoints(a.center(), b.center());
    a_to_b.intersects(l)
}

/// Group rects into left-to-right lines.
pub fn group_into_lines(rects: &[RotatedRect], separators: &[LineF]) -> Vec<Vec<RotatedRect>> {
    let mut sorted_rects: Vec<_> = rects.to_vec();
    sorted_rects.sort_by_key(|r| r.bounding_rect().left() as i32);

    let mut lines: Vec<Vec<RotatedRect>> = Vec::new();
    let overlap_threshold = 5.;
    let max_h_overlap = 5.;

    while !sorted_rects.is_empty() {
        let mut line = Vec::new();
        line.push(sorted_rects.remove(0));

        loop {
            let last = line.last().unwrap();
            let last_edge = rightmost_edge(last);

            if let Some((i, next_item)) = sorted_rects
                .iter()
                .enumerate()
                .filter(|(_, r)| {
                    let edge = leftmost_edge(r);
                    r.center().x > last.center().x
                        && edge.center().x - last_edge.center().x >= -max_h_overlap
                        && last_edge.vertical_overlap(edge) >= overlap_threshold
                        && !separators
                            .iter()
                            .any(|&s| rects_separated_by_line(last, r, s))
                })
                .min_by_key(|(_, r)| r.center().x as i32)
            {
                line.push(*next_item);
                sorted_rects.remove(i);
            } else {
                break;
            }
        }
        lines.push(line);
    }
    lines
}

fn find_block_separators(words: &[RotatedRect]) -> Vec<Rect> {
    let Some(page_rect) = bounding_rect(words.iter()).map(|br| br.integral_bounding_rect()) else {
        return Vec::new();
    };

    let mut lines = group_into_lines(words, &[]);
    lines.sort_by_key(|l| l.first().unwrap().bounding_rect().top().round() as i32);

    let mut all_word_spacings = Vec::new();
    for line in lines {
        if line.len() > 1 {
            let mut spacings: Vec<_> = line
                .iter()
                .zip(line.iter().skip(1))
                .map(|(cur, next)| {
                    (next.bounding_rect().left() - cur.bounding_rect().right())
                        .max(0.)
                        .round() as i32
                })
                .collect();
            spacings.sort_unstable();
            all_word_spacings.extend_from_slice(&spacings);
        }
    }
    all_word_spacings.sort_unstable();

    let median_word_spacing = all_word_spacings
        .get(all_word_spacings.len() / 2)
        .copied()
        .unwrap_or(10);
    let median_height = words
        .get(words.len() / 2)
        .map_or(10.0, |r| r.height())
        .round() as i32;

    let score = |r: Rect| {
        let aspect_ratio = (r.height() as f32) / (r.width() as f32);
        let aspect_ratio_weight = match aspect_ratio.log2().abs() {
            r if r < 3. => 0.5,
            r if r < 5. => 1.5,
            r => r,
        };
        ((r.area() as f32) * aspect_ratio_weight).sqrt()
    };

    let object_bboxes: Vec<_> = words
        .iter()
        .map(|r| r.bounding_rect().integral_bounding_rect())
        .collect();
    let min_width = median_word_spacing * 3;
    let min_height = (3 * median_height.max(0)) as u32;

    max_empty_rects(
        &object_bboxes,
        page_rect,
        score,
        min_width.try_into().unwrap_or(1),
        min_height,
    )
    .filter_overlapping(0.5)
    .take(80)
    .collect()
}

/// Group words into lines and sort them in reading order (matches upstream `ocrs`).
pub fn find_text_lines(words: &[RotatedRect]) -> Vec<Vec<RotatedRect>> {
    let separators = find_block_separators(words);
    let vertical_separators: Vec<_> = separators
        .iter()
        .map(|r| {
            let center = r.center();
            Line::from_endpoints(
                Point::from_yx(r.top(), center.x).to_f32(),
                Point::from_yx(r.bottom(), center.x).to_f32(),
            )
        })
        .collect();

    let horizontal_separators: Vec<_> = separators
        .iter()
        .map(|r| {
            let center = r.center();
            Line::from_endpoints(
                Point::from_yx(center.y, r.left()).to_f32(),
                Point::from_yx(center.y, r.right()).to_f32(),
            )
        })
        .collect();

    let mut lines = group_into_lines(words, &vertical_separators);

    let midpoint_line = |words: &[RotatedRect]| -> LineF {
        assert!(!words.is_empty());
        Line::from_endpoints(
            words.first().unwrap().bounding_rect().left_edge().center(),
            words.last().unwrap().bounding_rect().right_edge().center(),
        )
    };

    lines.sort_by_key(|words| midpoint_line(words).center().y as i32);

    let is_separated_by = |line_a: LineF, line_b: LineF, separators: &[LineF]| -> bool {
        let a_to_b = Line::from_endpoints(line_a.center(), line_b.center());
        separators.iter().any(|sep| sep.intersects(a_to_b))
    };

    let mut paragraphs: Vec<Vec<Vec<RotatedRect>>> = Vec::new();
    while !lines.is_empty() {
        let seed = lines.remove(0);
        let mut para = vec![seed.clone()];
        let mut prev_line = midpoint_line(&seed);
        let mut index = 0;
        while index < lines.len() {
            let candidate_line = midpoint_line(&lines[index]);
            if prev_line.horizontal_overlap(candidate_line) > 0.
                && !is_separated_by(prev_line, candidate_line, &horizontal_separators)
            {
                para.push(lines.remove(index));
                prev_line = candidate_line;
            } else {
                index += 1;
            }
        }
        paragraphs.push(para);
    }

    let lines = paragraphs
        .into_iter()
        .flat_map(|para| para.into_iter())
        .collect();
    drop_glyph_fragments(lines)
}

/// Discard isolated glyph fragments that belong to another line.
///
/// A wide, thin line can shed its tallest ascenders / lowest descenders as
/// separate one-glyph "lines" directly above or below the body — the detector's
/// connected components don't always bridge the gap. Left alone they recognize
/// as stray single characters bracketing the real text (the "S … T" around a
/// wide line). Such a fragment is already covered by the body line, so we drop
/// a *narrow* line when it sits horizontally inside a much *wider* line's span
/// and vertically within that line's glyph zone. (Merging it into the host
/// instead enlarges the host's crop box vertically and distorts the recognition
/// strip — so we remove it, not fold it in.) A real neighbouring text line
/// spans a similar width, so it never matches the narrowness test; single-line
/// inputs have nothing to drop — well-behaved pages don't regress.
fn drop_glyph_fragments(lines: Vec<Vec<RotatedRect>>) -> Vec<Vec<RotatedRect>> {
    if lines.len() < 2 {
        return lines;
    }
    // (left, right, top, bottom) union of a line's word bounding rects.
    fn bounds(line: &[RotatedRect]) -> Option<(f32, f32, f32, f32)> {
        let mut iter = line.iter();
        let first = iter.next()?.bounding_rect();
        let (mut l, mut r, mut t, mut b) =
            (first.left(), first.right(), first.top(), first.bottom());
        for rect in iter {
            let br = rect.bounding_rect();
            l = l.min(br.left());
            r = r.max(br.right());
            t = t.min(br.top());
            b = b.max(br.bottom());
        }
        Some((l, r, t, b))
    }

    let bnds: Vec<Option<(f32, f32, f32, f32)>> = lines.iter().map(|l| bounds(l)).collect();
    let mut drop = vec![false; lines.len()];
    for small in 0..lines.len() {
        let Some((sl, sr, st, sb)) = bnds[small] else {
            continue;
        };
        let s_cy = 0.5 * (st + sb);
        let s_w = sr - sl;
        for host in 0..lines.len() {
            if host == small || drop[host] {
                continue;
            }
            let Some((hl, hr, ht, hb)) = bnds[host] else {
                continue;
            };
            let h_w = (hr - hl).max(1.0);
            let h_h = (hb - ht).max(1.0);
            // Narrow vs the host (a real neighbouring line spans a similar
            // width, so it never matches).
            if s_w > 0.3 * h_w {
                continue;
            }
            // Horizontally inside the host's span (small slack).
            let margin = 0.02 * h_w;
            if sl < hl - margin || sr > hr + margin {
                continue;
            }
            // Vertically within the host's glyph zone (body ± ~0.6 height).
            if s_cy < ht - 0.6 * h_h || s_cy > hb + 0.6 * h_h {
                continue;
            }
            drop[small] = true;
            break;
        }
    }
    lines
        .into_iter()
        .zip(drop)
        .filter_map(|(line, d)| (!d).then_some(line))
        .collect()
}
