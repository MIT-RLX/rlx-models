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

//! Shared line cropping, batching, and CTC post-processing for recognition backends.

use crate::ctc::CtcHypothesis;
use crate::geom::{downwards_line, leftmost_edge, rightmost_edge};
use crate::host_resize::resize_bilinear;
use crate::preprocess::BLACK_VALUE;
use crate::text::{TextChar, TextLine};
use rten_imageproc::{BoundingRect, Line, Point, Polygon, Rect, RotatedRect};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, NdTensorViewMut};

#[derive(Clone)]
pub struct TextRecLine {
    pub index: usize,
    pub region: Polygon,
    pub resized_width: u32,
}

pub struct LineRecResult {
    pub line: TextRecLine,
    pub rec_input_len: usize,
    pub ctc_input_len: usize,
    pub ctc_output: CtcHypothesis,
}

pub fn resized_line_width(orig_width: i32, orig_height: i32, height: i32) -> u32 {
    let min_width = 10.;
    let max_width = 2400.;
    let aspect_ratio = orig_width as f32 / orig_height as f32;
    (height as f32 * aspect_ratio).clamp(min_width, max_width) as u32
}

pub fn line_polygon(words: &[RotatedRect]) -> Vec<Point> {
    let mut polygon = Vec::new();
    let floor_point = |p: rten_imageproc::PointF| Point::from_yx(p.y as i32, p.x as i32);

    for word_rect in words.iter() {
        let (left, right) = (
            downwards_line(leftmost_edge(word_rect)),
            downwards_line(rightmost_edge(word_rect)),
        );
        polygon.push(floor_point(left.start));
        polygon.push(floor_point(right.start));
    }
    for word_rect in words.iter().rev() {
        let (left, right) = (
            downwards_line(leftmost_edge(word_rect)),
            downwards_line(rightmost_edge(word_rect)),
        );
        polygon.push(floor_point(right.end));
        polygon.push(floor_point(left.end));
    }
    polygon
}

pub fn prepare_text_line(
    image: NdTensorView<f32, 3>,
    page_rect: Rect,
    line_region: &Polygon,
    resized_width: u32,
    output_height: usize,
) -> NdTensor<f32, 2> {
    let page_index_rect = page_rect.adjust_tlbr(0, 0, -1, -1);
    let grey_chan = image.slice([0]);
    let line_rect = line_region.bounding_rect();
    let mut line_img = NdTensor::full(
        [line_rect.height() as usize, line_rect.width() as usize],
        BLACK_VALUE,
    );

    for in_p in line_region.fill_iter() {
        let out_p = Point::from_yx(in_p.y - line_rect.top(), in_p.x - line_rect.left());
        if !page_index_rect.contains_point(in_p) || !page_index_rect.contains_point(out_p) {
            continue;
        }
        line_img[[out_p.y as usize, out_p.x as usize]] =
            grey_chan[[in_p.y as usize, in_p.x as usize]];
    }

    let reshaped = line_img.reshaped([1, 1, line_img.size(0), line_img.size(1)]);
    let resized_line_img = resize_bilinear(reshaped.view(), output_height, resized_width as usize);
    let out_shape = [resized_line_img.size(2), resized_line_img.size(3)];
    resized_line_img.into_shape(out_shape)
}

pub fn prepare_text_line_batch(
    image: &NdTensorView<f32, 3>,
    lines: &[TextRecLine],
    page_rect: Rect,
    output_height: usize,
    output_width: usize,
) -> NdTensor<f32, 4> {
    let mut output = NdTensor::full([lines.len(), 1, output_height, output_width], BLACK_VALUE);
    for (group_line_index, line) in lines.iter().enumerate() {
        let resized_line_img = prepare_text_line(
            image.view(),
            page_rect,
            &line.region,
            line.resized_width,
            output_height,
        );
        output
            .slice_mut((group_line_index, 0, .., ..(line.resized_width as usize)))
            .copy_from(&resized_line_img);
    }
    output
}

fn polygon_slice_bounding_rect(poly: Polygon, min_x: i32, max_x: i32) -> Option<Rect> {
    poly.edges()
        .filter_map(|e| {
            let e = e.rightwards();
            if (e.start.x < min_x && e.end.x < min_x) || (e.start.x > max_x && e.end.x > max_x) {
                return None;
            }
            let trunc_edge_start = e
                .to_f32()
                .y_for_x(min_x as f32)
                .map_or(e.start, |y| Point::from_yx(y.round() as i32, min_x));
            let trunc_edge_end = e
                .to_f32()
                .y_for_x(max_x as f32)
                .map_or(e.end, |y| Point::from_yx(y.round() as i32, max_x));
            Some(Line::from_endpoints(trunc_edge_start, trunc_edge_end))
        })
        .fold(None, |bounding_rect, e| {
            let edge_br = e.bounding_rect();
            bounding_rect.map(|br| br.union(edge_br)).or(Some(edge_br))
        })
}

pub fn text_lines_from_recognition_results(
    results: &[LineRecResult],
    alphabet: &str,
) -> Vec<Option<TextLine>> {
    results
        .iter()
        .map(|result| {
            let line_rect = result.line.region.bounding_rect();
            let x_scale_factor = (line_rect.width() as f32) / (result.line.resized_width as f32);
            let downsample_factor =
                (result.rec_input_len as f32 / result.ctc_input_len as f32).round() as u32;

            let steps = result.ctc_output.steps();
            let text_line: Vec<TextChar> = steps
                .iter()
                .enumerate()
                .filter_map(|(i, step)| {
                    let start_x = step.pos * downsample_factor;
                    let end_x = if let Some(next_step) = steps.get(i + 1) {
                        next_step.pos * downsample_factor
                    } else {
                        result.line.resized_width
                    };
                    let [start_x, end_x] = [start_x, end_x]
                        .map(|x| line_rect.left() + (x as f32 * x_scale_factor) as i32);
                    if start_x >= line_rect.right() {
                        return None;
                    }
                    let ch = alphabet
                        .chars()
                        .nth((step.label.saturating_sub(1)) as usize)
                        .unwrap_or('?');
                    Some(TextChar {
                        char: ch,
                        rect: polygon_slice_bounding_rect(
                            result.line.region.clone(),
                            start_x,
                            end_x,
                        )
                        .expect("invalid X coords"),
                    })
                })
                .collect();
            if text_line.is_empty() {
                None
            } else {
                Some(TextLine::new(text_line))
            }
        })
        .collect()
}

pub fn filter_excluded_char_labels<'a>(
    excluded_char_labels: Option<&[usize]>,
    input_seq_slice: &'a mut NdTensorViewMut<f32, 2>,
) -> NdTensorView<'a, f32, 2> {
    if let Some(excluded_char_labels) = excluded_char_labels {
        for row in 0..input_seq_slice.size(0) {
            for &excluded_char_label in excluded_char_labels.iter() {
                input_seq_slice[[row, excluded_char_label]] = f32::NEG_INFINITY;
            }
        }
    }
    input_seq_slice.view()
}

pub fn bounding_rect<'a, I: Iterator<Item = &'a RotatedRect>>(
    rects: I,
) -> Option<rten_imageproc::RectF> {
    rects.fold(None, |br, r| match br {
        Some(br) => Some(br.union(r.bounding_rect())),
        None => Some(r.bounding_rect()),
    })
}
