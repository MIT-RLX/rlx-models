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

//! Canvas M-RoPE (MiniCPM-V 4.7): `MiniCPMV4_7Model.get_rope_index` for one
//! unpadded sequence.
//!
//! A picture's thumbnail and its detail slices share one 2-D canvas: slices
//! tile it at their own resolution, the thumbnail is stretched over the whole
//! of it with a rounded `linspace`, and the markers around each crop are
//! pinned to its corners. Text runs 1-D before and after, and the text after
//! an image resumes at `canvas_origin + max(canvas_h, canvas_w) + 1`.
//!
//! The port is literal — including which tokens count as a frame's markers
//! (`\n` is one, so the newline the chat template puts after an image is
//! pulled into the frame span) — because every one of those choices moves a
//! position the model was trained on.

use anyhow::{Result, anyhow, ensure};

use crate::config::MarkerIds;

/// `[t, h, w, 0]` per token, the layout `rlx-qwen35` takes for M-RoPE.
pub type Pos = [usize; 4];

/// Plain 1-D positions (MiniCPM-V 4.6, or any text-only prompt).
pub fn linear_positions(len: usize) -> Vec<Pos> {
    (0..len).map(|p| [p, p, p, 0]).collect()
}

/// One detail crop: `(start, end)` token indices of its image tokens.
type Span = (usize, usize);

struct Frame {
    thumb: Span,
    end: usize,
    slices: Vec<Span>,
}

struct Group {
    start: usize,
    end: usize,
    frames: Vec<Frame>,
}

/// `_group_visual_frames`. `modality[i]` is 0 for text, 1 for an image
/// placeholder, 2 for a video one.
fn group_visual_frames(ids: &[u32], modality: &[u8], m: &MarkerIds) -> Vec<Group> {
    let markers = [
        m.slice_start,
        m.slice_end,
        m.image_start,
        m.image_end,
        m.newline,
    ];
    let is_marker = |t: u32| markers.contains(&t);

    // Maximal runs of one visual modality are the crops.
    let mut crops = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        if modality[i] != 0 {
            let s = i;
            while i < ids.len() && modality[i] == modality[s] {
                i += 1;
            }
            crops.push((s, i));
        } else {
            i += 1;
        }
    }

    // (group_start, frames as (thumb, slices)).
    let mut groups: Vec<(usize, Vec<(Span, Vec<Span>)>)> = Vec::new();
    let mut previous_end = 0usize;
    for crop in crops {
        let opens_slice = crop.0 > 0 && ids[crop.0 - 1] == m.slice_start;
        let has_open_frame = groups.last().is_some_and(|g| !g.1.is_empty());
        if has_open_frame && opens_slice {
            groups
                .last_mut()
                .unwrap()
                .1
                .last_mut()
                .unwrap()
                .1
                .push(crop);
        } else {
            let gap_end = crop.0.saturating_sub(1);
            let adjacent = !groups.is_empty()
                && ids[previous_end.min(gap_end)..gap_end]
                    .iter()
                    .all(|&t| is_marker(t));
            if adjacent {
                groups.last_mut().unwrap().1.push((crop, Vec::new()));
            } else {
                groups.push((crop.0.saturating_sub(1), vec![(crop, Vec::new())]));
            }
        }
        previous_end = crop.1;
    }

    let frame_end = |last_crop_end: usize| {
        let mut e = last_crop_end;
        while e < ids.len() && is_marker(ids[e]) {
            e += 1;
        }
        e
    };
    groups
        .into_iter()
        .map(|(start, frames)| {
            let frames: Vec<Frame> = frames
                .into_iter()
                .map(|(thumb, slices)| Frame {
                    end: frame_end(slices.last().map_or(thumb.1, |s| s.1)),
                    thumb,
                    slices,
                })
                .collect();
            let end = frames.last().map_or(start, |f| f.end);
            Group { start, end, frames }
        })
        .collect()
}

/// `get_vision_position_ids`: `[t, h, w]` for a crop of `grid` patches placed
/// on a `canvas_h × canvas_w` canvas (0 = the crop's own LLM grid) at
/// `(h_off, w_off)`, all offset by `origin`.
fn vision_positions(
    origin: usize,
    grid: (usize, usize),
    canvas_h: usize,
    canvas_w: usize,
    h_off: usize,
    w_off: usize,
    merge: usize,
) -> Vec<Pos> {
    let gh = grid.0 / merge;
    let gw = grid.1 / merge;
    let ch = if canvas_h == 0 { gh } else { canvas_h };
    let cw = if canvas_w == 0 { gw } else { canvas_w };
    let hs = linspace_round(ch, gh);
    let ws = linspace_round(cw, gw);
    let mut out = Vec::with_capacity(gh * gw);
    for &h in &hs {
        for &w in &ws {
            out.push([origin, origin + h + h_off, origin + w + w_off, 0]);
        }
    }
    out
}

/// `torch.linspace(0, extent - 1, n).round().long()`.
///
/// torch builds the f32 ramp from both ends (`start + i·step` for the first
/// half, `end − (n−1−i)·step` for the second) and rounds half to even.
fn linspace_round(extent: usize, n: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }
    let end = (extent as f32) - 1.0;
    let step = end / (n as f32 - 1.0);
    let half = n / 2;
    (0..n)
        .map(|i| {
            let v = if i < half {
                step * i as f32
            } else {
                end - step * (n - 1 - i) as f32
            };
            v.round_ties_even().max(0.0) as usize
        })
        .collect()
}

/// Canvas positions for `ids`. `modality[i]` is 0 for text, 1 for an image
/// placeholder and 2 for a video one; `grids` lists every crop's patch grid
/// in the order the crops appear (each picture's or frame's thumbnail, then
/// its slices). `merge` is the patches-per-
/// token factor along one axis (4 for `16x`, 2 for `4x`).
pub fn canvas_positions(
    ids: &[u32],
    modality: &[u8],
    grids: &[(usize, usize)],
    merge: usize,
    markers: &MarkerIds,
) -> Result<Vec<Pos>> {
    ensure!(
        ids.len() == modality.len(),
        "minicpm-v: ids / modality length"
    );
    let len = ids.len();
    let mut pos: Vec<Pos> = linear_positions(len);
    let mut grid_iter = grids.iter().copied();
    let mut next_grid = || {
        grid_iter
            .next()
            .ok_or_else(|| anyhow!("minicpm-v: more visual crops in the prompt than grids"))
    };
    let set_1d = |pos: &mut Vec<Pos>, a: usize, b: usize, from: usize| {
        for (k, p) in pos[a..b].iter_mut().enumerate() {
            let v = from + k;
            *p = [v, v, v, 0];
        }
    };

    let mut cur_pos = 0usize;
    let mut cur_idx = 0usize;
    for g in group_visual_frames(ids, modality, markers) {
        if g.start > cur_idx {
            set_1d(&mut pos, cur_idx, g.start, cur_pos);
            cur_pos += g.start - cur_idx;
        }
        let mut cursor = g.start;
        for fr in &g.frames {
            let thumb_grid = next_grid()?;
            let frame_start = fr.thumb.0 - 1;
            if frame_start > cursor {
                set_1d(&mut pos, cursor, frame_start, cur_pos);
                cur_pos += frame_start - cursor + 1;
            }
            let frame_end = fr.end.min(g.end);
            let origin = cur_pos;
            let halo = origin.saturating_sub(1);

            let (mut slice_h, mut slice_w, mut cols) = (0usize, 0usize, 0usize);
            let mut canvas_h = thumb_grid.0 / merge;
            let mut canvas_w = thumb_grid.1 / merge;
            let mut first_slice = None;
            if !fr.slices.is_empty() {
                let n = fr.slices.len();
                cols = n;
                for k in 0..n - 1 {
                    if fr.slices[k + 1].0 - fr.slices[k].1 > 2 {
                        cols = k + 1;
                        break;
                    }
                }
                let mut rows = n.checked_div(cols).unwrap_or(1);
                if rows * cols != n {
                    rows = 1;
                    cols = n;
                }
                let fs = next_grid()?;
                first_slice = Some(fs);
                slice_h = fs.0 / merge;
                slice_w = fs.1 / merge;
                canvas_h = rows * slice_h;
                canvas_w = cols * slice_w;
            }

            // Base coat, then `<image>` on the halo outside the canvas.
            for p in &mut pos[frame_start..frame_end] {
                *p = [origin, origin, origin, 0];
            }
            pos[frame_start][1] = halo;
            pos[frame_start][2] = halo;
            // `</image>` on the far corner.
            if fr.thumb.1 < frame_end {
                pos[fr.thumb.1][1] = origin + canvas_h;
                pos[fr.thumb.1][2] = origin + canvas_w;
            }
            let thumb = vision_positions(origin, thumb_grid, canvas_h, canvas_w, 0, 0, merge);
            ensure!(
                thumb.len() == fr.thumb.1 - fr.thumb.0,
                "minicpm-v: thumbnail has {} placeholders, grid {thumb_grid:?} gives {}",
                fr.thumb.1 - fr.thumb.0,
                thumb.len()
            );
            pos[fr.thumb.0..fr.thumb.1].copy_from_slice(&thumb);

            for (k, &(ss, se)) in fr.slices.iter().enumerate() {
                let h_off = (k / cols) * slice_h;
                let w_off = (k % cols) * slice_w;
                let grid = if k == 0 {
                    first_slice.expect("set above")
                } else {
                    next_grid()?
                };
                let (sh, sw) = (grid.0 / merge, grid.1 / merge);
                if ss > frame_start {
                    pos[ss - 1][1] = origin + h_off;
                    pos[ss - 1][2] = origin + w_off;
                }
                if se < frame_end {
                    pos[se][1] = origin + h_off + sh - 1;
                    pos[se][2] = origin + w_off + sw - 1;
                }
                let sp = vision_positions(origin, grid, 0, 0, h_off, w_off, merge);
                ensure!(
                    sp.len() == se - ss,
                    "minicpm-v: slice {k} placeholder count"
                );
                pos[ss..se].copy_from_slice(&sp);
            }

            // A `\n` ending a row of slices sits past that row's right edge.
            for k in 0..fr.slices.len().saturating_sub(1) {
                let (gs, ge) = (fr.slices[k].1, fr.slices[k + 1].0);
                if ge - gs <= 2 {
                    continue;
                }
                let boundary_h = (k / cols + 1) * slice_h - 1;
                let right = cols * slice_w;
                for p in &mut pos[gs + 1..ge - 1] {
                    p[1] = origin + boundary_h;
                    p[2] = origin + right;
                }
            }

            cur_pos = origin + canvas_h.max(canvas_w) + 1;
            cursor = frame_end;
        }
        if cursor < g.end {
            set_1d(&mut pos, cursor, g.end, cur_pos);
            cur_pos += g.end - cursor;
        }
        cur_idx = g.end;
    }
    if cur_idx < len {
        set_1d(&mut pos, cur_idx, len, cur_pos);
    }
    ensure!(
        grid_iter.next().is_none(),
        "minicpm-v: more grids than image crops in the prompt"
    );
    Ok(pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: MarkerIds = MarkerIds {
        image: 9,
        image_start: 1,
        image_end: 2,
        slice_start: 3,
        slice_end: 4,
        newline: 5,
    };

    #[test]
    fn linspace_matches_torch_rounding() {
        // torch.linspace(0, 21, 11).round() → [0,2,4,6,8,10,13,15,17,19,21]
        // (10.5 rounds to 10 under round-half-even; 12.6 → 13).
        assert_eq!(
            linspace_round(22, 11),
            vec![0, 2, 4, 6, 8, 10, 13, 15, 17, 19, 21]
        );
        assert_eq!(linspace_round(4, 4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn thumbnail_only_image() {
        // text text <image> 4 tokens (grid 8x8 / 4 → 2x2) </image> text
        let ids = [7, 7, 1, 9, 9, 9, 9, 2, 7];
        let img: Vec<u8> = ids.iter().map(|&t| u8::from(t == 9)).collect();
        let p = canvas_positions(&ids, &img, &[(8, 8)], 4, &M).unwrap();
        assert_eq!(p[0], [0, 0, 0, 0]);
        assert_eq!(p[2], [2, 1, 1, 0]); // `<image>` on the halo
        assert_eq!(p[3], [2, 2, 2, 0]);
        assert_eq!(p[6], [2, 3, 3, 0]);
        assert_eq!(p[7], [2, 4, 4, 0]); // `</image>` on the far corner
        // Text resumes at origin + max(2, 2) + 1.
        assert_eq!(p[8], [5, 5, 5, 0]);
    }

    #[test]
    fn grid_count_mismatch_is_an_error() {
        let ids = [1, 9, 2];
        let img = [0u8, 1, 0];
        assert!(canvas_positions(&ids, &img, &[], 4, &M).is_err());
    }
}
