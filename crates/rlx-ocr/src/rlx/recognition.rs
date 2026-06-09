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

use crate::capabilities::validate_device;
use crate::config::{DecodeMethod, RECOGNITION_INPUT_HEIGHT};
use crate::model::{
    NUM_CLASSES, RecognitionGraphConfig, build_recognition_graph, log_softmax_last_axis,
};
use crate::recognition::line_batch::{
    LineRecResult, TextRecLine, bounding_rect, filter_excluded_char_labels, line_polygon,
    prepare_text_line, prepare_text_line_batch, resized_line_width,
    text_lines_from_recognition_results,
};
use crate::text::TextLine;
use crate::weights::{
    HF_RECOGNITION_ST, HF_RECOGNITION_ST_FULL, SafetensorsFile, prefer_safetensors_path,
};
use anyhow::{Result, anyhow};
use rlx_core::flow_bridge::compile_options_for_profile;
use rlx_core::flow_util::attach_built_params;
use rlx_flow::CompileProfile;
use rlx_runtime::{CompiledGraph, Device, Session};
use rten_imageproc::{Polygon, Rect, RotatedRect};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

/// Recognition backend using compiled native RLX CRNN + GRU graphs (per padded width).
pub struct RlxTextRecognizer {
    weights: SafetensorsFile,
    device: Device,
    cache: Mutex<HashMap<usize, CompiledGraph>>,
    input_height: u32,
}

impl RlxTextRecognizer {
    pub fn from_path(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        Self::from_safetensors(path.as_ref(), device)
    }

    pub fn from_safetensors(path: &Path, device: Device) -> Result<Self> {
        validate_device(device)?;
        if !path.is_file() {
            anyhow::bail!("recognition weights not found: {path:?}");
        }
        Ok(Self {
            weights: SafetensorsFile::open(path)?,
            device,
            cache: Mutex::new(HashMap::new()),
            input_height: RECOGNITION_INPUT_HEIGHT,
        })
    }

    pub fn from_model_dir(dir: &Path, device: Device) -> Result<Self> {
        validate_device(device)?;
        let path = prefer_safetensors_path(dir, HF_RECOGNITION_ST, HF_RECOGNITION_ST_FULL);
        Self::from_safetensors(&path, device)
    }

    fn ensure_compiled(&self, compile_width: usize, max_lines_per_group: usize) -> Result<()> {
        let mut cache = self.cache.lock().map_err(|_| anyhow!("lock poisoned"))?;
        if cache.contains_key(&compile_width) {
            return Ok(());
        }
        let mut wm = self.weights.weight_map()?;
        let (graph, params) = build_recognition_graph(
            &mut wm,
            RecognitionGraphConfig {
                batch: max_lines_per_group,
                width: compile_width,
            },
        )?;
        let opts = compile_options_for_profile(&CompileProfile::encoder(), self.device);
        let mut compiled = Session::new(self.device).compile_with(graph, &opts);
        attach_built_params(&mut compiled, params, &[]);
        cache.insert(compile_width, compiled);
        Ok(())
    }

    /// Run recognition on an NCHW batch; returns `[batch, seq, classes]` log-probs.
    pub fn run_batch_logits(&self, input: NdTensor<f32, 4>) -> Result<NdTensor<f32, 3>> {
        let w = input.size(3);
        let batch = input.size(0);
        self.ensure_compiled(w, batch)?;
        let input_vec: Vec<f32> = input.iter().copied().collect();
        let mut cache = self.cache.lock().map_err(|_| anyhow!("lock poisoned"))?;
        let compiled = cache.get_mut(&w).unwrap();
        let mut flat = compiled
            .run(&[("image", input_vec.as_slice())])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("recognition returned no output"))?;
        let seq_len = flat.len() / (batch * NUM_CLASSES);
        log_softmax_last_axis(&mut flat, NUM_CLASSES);
        Ok(NdTensor::from_data([batch, seq_len, NUM_CLASSES], flat))
    }

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
        let resized_width = resized_line_width(
            line_rect.width(),
            line_rect.height(),
            self.input_height as i32,
        );
        prepare_text_line(
            image,
            page_rect,
            &line_poly,
            resized_width,
            self.input_height as usize,
        )
    }

    fn run_group(
        &self,
        image: &NdTensorView<f32, 3>,
        lines: Vec<TextRecLine>,
        page_rect: Rect,
        compile_width: usize,
        rec_img_height: u32,
        decode_method: DecodeMethod,
        alphabet_len: usize,
        excluded_char_labels: Option<&[usize]>,
    ) -> Result<Vec<LineRecResult>> {
        self.ensure_compiled(compile_width, lines.len().max(1))?;
        let rec_input = prepare_text_line_batch(
            image,
            &lines,
            page_rect,
            rec_img_height as usize,
            compile_width,
        );
        let input: Vec<f32> = rec_input.iter().copied().collect();
        let mut cache = self.cache.lock().map_err(|_| anyhow!("lock poisoned"))?;
        let compiled = cache.get_mut(&compile_width).unwrap();
        let outputs = compiled.run(&[("image", input.as_slice())]);
        let flat_out = outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("recognition returned no output"))?;

        let n_classes = alphabet_len + 1;
        let seq_len = flat_out.len() / (lines.len() * n_classes);
        let n = lines.len() * seq_len * n_classes;
        let mut logits = flat_out[..n].to_vec();
        log_softmax_last_axis(&mut logits, n_classes);
        let mut rec_output = NdTensor::from_data([lines.len(), seq_len, n_classes], logits);

        if n_classes != rec_output.size(2) {
            return Err(anyhow!(
                "recognition classes ({}) != alphabet+blank ({n_classes})",
                rec_output.size(2)
            ));
        }
        let ctc_input_len = rec_output.size(1);

        Ok(lines
            .into_iter()
            .enumerate()
            .map(|(group_line_index, line)| {
                let mut input_seq_slice = rec_output.slice_mut([group_line_index]);
                let input_seq =
                    filter_excluded_char_labels(excluded_char_labels, &mut input_seq_slice);
                let ctc_output = crate::ctc::decode(input_seq, decode_method);
                LineRecResult {
                    line,
                    rec_input_len: compile_width,
                    ctc_input_len,
                    ctc_output,
                }
            })
            .collect())
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
        let rec_img_height = self.input_height;

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
        let line_groups: Vec<(usize, Vec<TextRecLine>)> = line_groups
            .into_iter()
            .flat_map(|(group_width, lines)| {
                lines
                    .chunks(max_lines_per_group)
                    .map(|chunk| (group_width as usize, chunk.to_vec()))
                    .collect::<Vec<_>>()
            })
            .collect();

        let alphabet_len = alphabet.chars().count();
        for &(compile_width, ref chunk) in &line_groups {
            self.ensure_compiled(compile_width, chunk.len().max(1))?;
        }

        let group_results: Result<Vec<Vec<LineRecResult>>> = line_groups
            .into_iter()
            .map(|(compile_width, lines)| {
                self.run_group(
                    &image,
                    lines,
                    page_rect,
                    compile_width,
                    rec_img_height,
                    decode_method,
                    alphabet_len,
                    excluded_char_labels,
                )
            })
            .collect();

        let mut line_rec_results: Vec<LineRecResult> =
            group_results?.into_iter().flatten().collect();
        line_rec_results.sort_by_key(|r| r.line.index);
        Ok(text_lines_from_recognition_results(
            &line_rec_results,
            alphabet,
        ))
    }
}
