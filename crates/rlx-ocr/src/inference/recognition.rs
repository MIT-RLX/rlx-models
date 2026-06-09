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

//! Text line recognition via ocrs-compatible RTen CRNN models.

use crate::config::{DecodeMethod, RECOGNITION_INPUT_HEIGHT};
use crate::ctc::decode;
use crate::inference::load_rten_model;
use crate::recognition::line_batch::{
    LineRecResult, TextRecLine, bounding_rect, filter_excluded_char_labels, line_polygon,
    prepare_text_line, prepare_text_line_batch, resized_line_width,
    text_lines_from_recognition_results,
};
use crate::text::TextLine;
use anyhow::{Context, Result, anyhow};
use rayon::prelude::*;
use rten::{Dimension, FloatOperators, RunOptions, thread_pool};
use rten_imageproc::{BoundingRect, Polygon, Rect, RotatedRect};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, NdTensorViewMut};
use std::collections::HashMap;
use std::path::Path;

/// Recognition backend for ocrs `.rten` CRNN checkpoints.
pub struct RtenTextRecognizer {
    model: rten::Model,
    input_shape: Vec<Dimension>,
}

impl RtenTextRecognizer {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let model = load_rten_model(path.as_ref())?;
        let input_shape = model_input_shape(&model)?;
        Ok(Self { model, input_shape })
    }

    fn input_height(&self) -> u32 {
        match self.input_shape[2] {
            Dimension::Fixed(size) => size.try_into().unwrap_or(RECOGNITION_INPUT_HEIGHT),
            Dimension::Symbolic(_) => RECOGNITION_INPUT_HEIGHT,
        }
    }

    /// Prepare one text line for the recognition model (debug API, matches upstream `ocrs`).
    pub fn prepare_input(
        &self,
        image: NdTensorView<f32, 3>,
        line: &[RotatedRect],
    ) -> NdTensor<f32, 2> {
        let [_, img_height, img_width] = image.shape();
        let page_rect = Rect::from_hw(img_height as i32, img_width as i32);
        let line_rect = bounding_rect(line.iter())
            .expect("line has no words")
            .integral_bounding_rect();
        let line_poly = Polygon::new(line_polygon(line));
        let rec_img_height = self.input_height();
        let resized_width =
            resized_line_width(line_rect.width(), line_rect.height(), rec_img_height as i32);
        prepare_text_line(
            image,
            page_rect,
            &line_poly,
            resized_width,
            rec_img_height as usize,
        )
    }

    pub fn recognize_text_lines(
        &self,
        image: NdTensorView<f32, 3>,
        lines: &[Vec<RotatedRect>],
        decode_method: DecodeMethod,
        alphabet: &str,
        excluded_char_labels: Option<&[usize]>,
    ) -> Result<Vec<Option<TextLine>>> {
        let [_, img_height, img_width] = image.shape();
        let page_rect = Rect::from_hw(img_height as i32, img_width as i32);
        let rec_img_height = self.input_height();

        let mut line_groups: HashMap<i32, Vec<TextRecLine>> = HashMap::new();
        for (line_index, word_rects) in lines.iter().enumerate() {
            let line_rect = bounding_rect(word_rects.iter())
                .expect("line has no words")
                .integral_bounding_rect();
            let resized_width =
                resized_line_width(line_rect.width(), line_rect.height(), rec_img_height as i32);
            let group_width = resized_width.next_multiple_of(50);
            line_groups
                .entry(group_width as i32)
                .or_default()
                .push(TextRecLine {
                    index: line_index,
                    region: Polygon::new(line_polygon(word_rects)),
                    resized_width,
                });
        }

        let max_lines_per_group = 20;
        let line_groups: Vec<(i32, Vec<TextRecLine>)> = line_groups
            .into_iter()
            .flat_map(|(group_width, lines)| {
                lines
                    .chunks(max_lines_per_group)
                    .map(|chunk| (group_width, chunk.to_vec()))
                    .collect::<Vec<_>>()
            })
            .collect();

        let alphabet_len = alphabet.chars().count();

        let batch_rec_results: Result<Vec<Vec<LineRecResult>>> = thread_pool().run(|| {
            line_groups
                .into_par_iter()
                .map(|(group_width, lines)| {
                    let rec_input = prepare_text_line_batch(
                        &image,
                        &lines,
                        page_rect,
                        rec_img_height as usize,
                        group_width as usize,
                    );
                    let mut rec_output = self.run(rec_input)?;
                    if alphabet_len + 1 != rec_output.size(2) {
                        return Err(anyhow!(
                            "recognition output classes ({}) != alphabet size ({})",
                            rec_output.size(2),
                            alphabet_len + 1
                        ));
                    }
                    let ctc_input_len = rec_output.shape()[1];
                    let group_results = lines
                        .into_iter()
                        .enumerate()
                        .map(|(group_line_index, line)| {
                            let mut input_seq_slice = rec_output.slice_mut([group_line_index]);
                            let input_seq = filter_excluded_char_labels(
                                excluded_char_labels,
                                &mut input_seq_slice,
                            );
                            let ctc_output = decode(input_seq, decode_method);
                            LineRecResult {
                                line,
                                rec_input_len: group_width as usize,
                                ctc_input_len,
                                ctc_output,
                            }
                        })
                        .collect();
                    Ok(group_results)
                })
                .collect()
        });

        let mut line_rec_results: Vec<LineRecResult> =
            batch_rec_results?.into_iter().flatten().collect();
        line_rec_results.sort_by_key(|result| result.line.index);
        Ok(text_lines_from_recognition_results(
            &line_rec_results,
            alphabet,
        ))
    }

    pub fn run(&self, input: NdTensor<f32, 4>) -> Result<NdTensor<f32, 3>> {
        let input: rten_tensor::Tensor = input.into();
        let output = self
            .model
            .run_one(input.view().into(), None::<RunOptions>)
            .context("recognition model run")?;
        let ndim = output.ndim();
        let mut rec_sequence: NdTensor<f32, 3> = output
            .try_into()
            .map_err(|_| anyhow!("expected recognition output to have 3 dims, got {}", ndim))?;
        rec_sequence.permute([1, 0, 2]);
        Ok(rec_sequence)
    }
}

fn model_input_shape(model: &rten::Model) -> Result<Vec<Dimension>> {
    let input_id = model
        .input_ids()
        .first()
        .copied()
        .ok_or_else(|| anyhow!("recognition model has no inputs"))?;
    model
        .node_info(input_id)
        .and_then(|info| info.shape())
        .ok_or_else(|| anyhow!("recognition model does not specify input shape"))
}
