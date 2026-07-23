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

//! CRAFT-style post-process: region + link heatmaps → text-line boxes.
//!
//! Threshold `region ∪ link_h`, connected-components (8-connectivity), then merge
//! components with overlapping y-centres into line boxes. Boxes are in detector
//! heatmap space (240×240); the pipeline maps them back to image coordinates.

use std::collections::VecDeque;

#[derive(Clone, Copy, Debug)]
pub struct Box2 {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

/// Group heatmaps into line boxes. `region`/`link_h` are `hw*hw` row-major.
pub fn group_lines(region: &[f32], link_h: &[f32], hw: usize, thresh: f32, min_pixels: usize) -> Vec<Box2> {
    let mask: Vec<bool> = region
        .iter()
        .zip(link_h)
        .map(|(&r, &l)| r >= thresh || l >= thresh)
        .collect();

    // Connected components (BFS, 8-connectivity).
    let mut label = vec![0i32; hw * hw];
    let mut cur = 0i32;
    for start in 0..hw * hw {
        if !mask[start] || label[start] != 0 {
            continue;
        }
        cur += 1;
        let mut q = VecDeque::new();
        q.push_back(start);
        label[start] = cur;
        while let Some(p) = q.pop_front() {
            let (y, x) = ((p / hw) as i32, (p % hw) as i32);
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let (ny, nx) = (y + dy, x + dx);
                    if ny < 0 || ny >= hw as i32 || nx < 0 || nx >= hw as i32 {
                        continue;
                    }
                    let np = ny as usize * hw + nx as usize;
                    if mask[np] && label[np] == 0 {
                        label[np] = cur;
                        q.push_back(np);
                    }
                }
            }
        }
    }

    // Per-component bounding boxes (filter tiny specks).
    let mut counts = vec![0usize; (cur + 1) as usize];
    let mut bb = vec![[hw as i32, hw as i32, 0i32, 0i32]; (cur + 1) as usize]; // x0,y0,x1,y1
    for p in 0..hw * hw {
        let l = label[p];
        if l == 0 {
            continue;
        }
        let (y, x) = ((p / hw) as i32, (p % hw) as i32);
        counts[l as usize] += 1;
        let b = &mut bb[l as usize];
        b[0] = b[0].min(x);
        b[1] = b[1].min(y);
        b[2] = b[2].max(x + 1);
        b[3] = b[3].max(y + 1);
    }
    let mut boxes: Vec<Box2> = (1..=cur as usize)
        .filter(|&i| counts[i] >= min_pixels)
        .map(|i| Box2 {
            x0: bb[i][0] as f32,
            y0: bb[i][1] as f32,
            x1: bb[i][2] as f32,
            y1: bb[i][3] as f32,
        })
        .collect();

    // Merge components whose y-centre falls within an existing line band.
    boxes.sort_by(|a, b| a.y0.partial_cmp(&b.y0).unwrap());
    let mut lines: Vec<Box2> = Vec::new();
    for b in boxes {
        let cy = (b.y0 + b.y1) * 0.5;
        let mut placed = false;
        for l in lines.iter_mut() {
            if cy >= l.y0 - 3.0 && cy <= l.y1 + 3.0 {
                l.x0 = l.x0.min(b.x0);
                l.y0 = l.y0.min(b.y0);
                l.x1 = l.x1.max(b.x1);
                l.y1 = l.y1.max(b.y1);
                placed = true;
                break;
            }
        }
        if !placed {
            lines.push(b);
        }
    }
    lines.sort_by(|a, b| a.y0.partial_cmp(&b.y0).unwrap());
    lines
}
