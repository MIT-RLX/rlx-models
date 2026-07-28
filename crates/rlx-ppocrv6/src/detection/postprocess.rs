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

//! DB text-detection post-process (probability map → polygons).

use crate::config::DetectionParams;
use rten_imageproc::{RetrievalMode, RotatedRect, find_contours, min_area_rect, simplify_polygon};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};

/// One detected text region in original image coordinates.
#[derive(Debug, Clone)]
pub struct DetBox {
    /// Four corner points (x, y) in reading order approx.
    pub points: [[f32; 2]; 4],
    pub score: f32,
}

/// Run DB bitmap → boxes. `prob` is `[H, W]` probability map at network resolution.
/// `sx`/`sy` map network coords → original image.
pub fn db_boxes_from_prob(
    prob: NdTensorView<f32, 2>,
    params: &DetectionParams,
    sx: f32,
    sy: f32,
) -> Vec<DetBox> {
    let [h, w] = prob.shape();
    let binary: Vec<bool> = prob.iter().map(|&p| p > params.thresh).collect();
    let binary_tensor = NdTensor::from_data([h, w], binary);

    let mut out = Vec::new();
    for poly in find_contours(binary_tensor.view(), RetrievalMode::External).iter() {
        if out.len() >= params.max_candidates {
            break;
        }
        if poly.len() < 3 {
            continue;
        }
        let float_points: Vec<_> = poly.iter().map(|p| p.to_f32()).collect();
        let simplified = simplify_polygon(&float_points, 2.);
        let Some(mut rect) = min_area_rect(&simplified) else {
            continue;
        };
        if rect.width().min(rect.height()) < params.min_size {
            continue;
        }
        let score = box_score_fast(prob, &rect);
        if score < params.box_thresh {
            continue;
        }
        // Unclip via area/perimeter expansion on the rotated rect.
        let expand = unclip_expand(rect.width(), rect.height(), params.unclip_ratio);
        rect.resize(rect.width() + 2. * expand, rect.height() + 2. * expand);

        let corners = rect.corners();
        let mut points = [[0f32; 2]; 4];
        for (i, c) in corners.iter().enumerate().take(4) {
            points[i] = [c.x / sx, c.y / sy];
        }
        out.push(DetBox { points, score });
    }
    out
}

fn unclip_expand(width: f32, height: f32, unclip_ratio: f32) -> f32 {
    let area = width * height;
    let peri = 2.0 * (width + height);
    if peri <= 1e-6 {
        return 0.0;
    }
    area * unclip_ratio / peri
}

fn box_score_fast(prob: NdTensorView<f32, 2>, rect: &RotatedRect) -> f32 {
    let [h, w] = prob.shape();
    let corners = rect.corners();
    let xs: Vec<f32> = corners.iter().map(|c| c.x).collect();
    let ys: Vec<f32> = corners.iter().map(|c| c.y).collect();
    let min_x = xs.iter().cloned().fold(f32::INFINITY, f32::min).floor() as isize;
    let max_x = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max).ceil() as isize;
    let min_y = ys.iter().cloned().fold(f32::INFINITY, f32::min).floor() as isize;
    let max_y = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max).ceil() as isize;
    let x0 = min_x.clamp(0, w as isize - 1) as usize;
    let x1 = max_x.clamp(0, w as isize) as usize;
    let y0 = min_y.clamp(0, h as isize - 1) as usize;
    let y1 = max_y.clamp(0, h as isize) as usize;
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let mut sum = 0f32;
    let mut count = 0usize;
    for y in y0..y1 {
        for x in x0..x1 {
            sum += prob[[y, x]];
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}
