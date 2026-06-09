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

//! Post-processing for text detection segmentation masks.

use rten_imageproc::{RetrievalMode, RotatedRect, find_contours, min_area_rect, simplify_polygon};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};

/// Extract oriented word bounding boxes from a per-pixel text probability mask.
pub fn word_rects_from_mask(
    mask: NdTensorView<f32, 2>,
    text_threshold: f32,
    min_area: f32,
) -> Vec<RotatedRect> {
    let binary: Vec<bool> = mask.iter().map(|&prob| prob > text_threshold).collect();
    let [h, w] = mask.shape();
    let expand_dist = 3.;

    let binary_tensor = NdTensor::from_data([h, w], binary);
    find_contours(binary_tensor.view(), RetrievalMode::External)
        .iter()
        .filter_map(|poly| {
            let float_points: Vec<_> = poly.iter().map(|p| p.to_f32()).collect();
            let simplified = simplify_polygon(&float_points, 2.);
            min_area_rect(&simplified).map(|mut rect| {
                rect.resize(
                    rect.width() + 2. * expand_dist,
                    rect.height() + 2. * expand_dist,
                );
                rect
            })
        })
        .filter(|r| r.area() >= min_area)
        .collect()
}
