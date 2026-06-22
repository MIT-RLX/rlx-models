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

//! Task post-processing: turns generated `<loc_N>` token sequences into
//! captions, bounding boxes, quad boxes (OCR), and polygons. Mirrors
//! `Florence2PostProcesser` and `post_process_generation`.
//!
//! Coordinates are decoded structurally from the token ids (each `<loc_N>` is
//! bin `N` of 1000), which is equivalent to HF's text-regex parsing but
//! avoids re-rendering byte-level BPE.

use crate::config::task_post_processing_type;
use crate::tokenizer::{Florence2Tokenizer, NUM_LOC_BINS};
use anyhow::Result;

/// One detected object: a box in pixel coordinates and its label.
#[derive(Debug, Clone, PartialEq)]
pub struct BBoxInstance {
    pub bbox: [f64; 4],
    pub label: String,
}

/// One OCR region: a quadrilateral (8 coords) and its text.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadInstance {
    pub quad_box: [f64; 8],
    pub text: String,
}

/// One polygon group (a label and a list of polygons, each a flat coord list).
#[derive(Debug, Clone, PartialEq)]
pub struct PolygonInstance {
    pub polygons: Vec<Vec<f64>>,
    pub label: String,
}

/// Parsed task output.
#[derive(Debug, Clone, PartialEq)]
pub enum Florence2Result {
    PureText(String),
    BBoxes(Vec<BBoxInstance>),
    Ocr(Vec<QuadInstance>),
    Polygons(Vec<PolygonInstance>),
}

/// Bin `n` (of 1000) → pixel coordinate at the bin centre.
fn dequant(n: usize, size: f64) -> f64 {
    let per_bin = size / NUM_LOC_BINS as f64;
    (n as f64 + 0.5) * per_bin
}

/// A token run split into a decoded text label and the loc bins that follow.
struct Segment {
    label: String,
    locs: Vec<usize>,
}

/// Split the generated ids into `(text-run, following loc-run)` segments.
fn segment(ids: &[u32], tk: &Florence2Tokenizer) -> Result<Vec<Segment>> {
    let mut segs: Vec<Segment> = Vec::new();
    let mut text_buf: Vec<u32> = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        if tk.loc_index(ids[i]).is_some() {
            // Collect the contiguous loc run.
            let mut locs = Vec::new();
            while i < ids.len() {
                if let Some(b) = tk.loc_index(ids[i]) {
                    locs.push(b);
                    i += 1;
                } else {
                    break;
                }
            }
            let label = tk.decode(&text_buf)?.trim().to_string();
            text_buf.clear();
            segs.push(Segment { label, locs });
        } else {
            text_buf.push(ids[i]);
            i += 1;
        }
    }
    Ok(segs)
}

/// Top-level dispatch for a task token (mirrors `post_process_generation`).
pub fn post_process(
    task: &str,
    ids: &[u32],
    tk: &Florence2Tokenizer,
    image_size: (f64, f64),
) -> Result<Florence2Result> {
    match task_post_processing_type(task) {
        "pure_text" => Ok(Florence2Result::PureText(clean_text(ids, tk)?)),
        "description_with_bboxes" | "bboxes" | "phrase_grounding" => {
            Ok(Florence2Result::BBoxes(parse_bboxes(ids, tk, image_size)?))
        }
        "ocr" => Ok(Florence2Result::Ocr(parse_ocr(ids, tk, image_size)?)),
        "polygons" => Ok(Florence2Result::Polygons(parse_polygons(
            ids, tk, image_size,
        )?)),
        "description_with_bboxes_or_polygons" => {
            // `<poly>` selects polygon parsing; otherwise boxes.
            let has_poly = ids.iter().any(|&t| {
                tk.decode_keep_special(&[t])
                    .map(|s| s.contains("<poly>"))
                    .unwrap_or(false)
            });
            if has_poly {
                Ok(Florence2Result::Polygons(parse_polygons(
                    ids, tk, image_size,
                )?))
            } else {
                Ok(Florence2Result::BBoxes(parse_bboxes(ids, tk, image_size)?))
            }
        }
        _ => Ok(Florence2Result::PureText(clean_text(ids, tk)?)),
    }
}

fn clean_text(ids: &[u32], tk: &Florence2Tokenizer) -> Result<String> {
    Ok(tk.decode(ids)?.trim().to_string())
}

/// `([phrase])(<loc>×4)+` → one instance per box, sharing the phrase label.
fn parse_bboxes(
    ids: &[u32],
    tk: &Florence2Tokenizer,
    image_size: (f64, f64),
) -> Result<Vec<BBoxInstance>> {
    let (w, h) = image_size;
    let mut out = Vec::new();
    for seg in segment(ids, tk)? {
        for chunk in seg.locs.chunks_exact(4) {
            out.push(BBoxInstance {
                bbox: [
                    dequant(chunk[0], w),
                    dequant(chunk[1], h),
                    dequant(chunk[2], w),
                    dequant(chunk[3], h),
                ],
                label: seg.label.clone(),
            });
        }
    }
    Ok(out)
}

/// `([text])(<loc>×8)` → quad box (4 points) per OCR line.
fn parse_ocr(
    ids: &[u32],
    tk: &Florence2Tokenizer,
    image_size: (f64, f64),
) -> Result<Vec<QuadInstance>> {
    let (w, h) = image_size;
    let mut out = Vec::new();
    for seg in segment(ids, tk)? {
        for chunk in seg.locs.chunks_exact(8) {
            let mut quad = [0f64; 8];
            for p in 0..4 {
                quad[p * 2] = dequant(chunk[p * 2], w);
                quad[p * 2 + 1] = dequant(chunk[p * 2 + 1], h);
            }
            out.push(QuadInstance {
                quad_box: quad,
                text: seg.label.clone(),
            });
        }
    }
    Ok(out)
}

/// `([phrase])(<loc>+)` → polygon(s); `<sep>` (decoded) separates polygons of
/// the same phrase. Coordinates come in (x,y) pairs.
fn parse_polygons(
    ids: &[u32],
    tk: &Florence2Tokenizer,
    image_size: (f64, f64),
) -> Result<Vec<PolygonInstance>> {
    let (w, h) = image_size;
    let mut out = Vec::new();
    for seg in segment(ids, tk)? {
        if seg.locs.is_empty() {
            continue;
        }
        let mut poly = Vec::new();
        for pair in seg.locs.chunks_exact(2) {
            poly.push(dequant(pair[0], w));
            poly.push(dequant(pair[1], h));
        }
        out.push(PolygonInstance {
            polygons: vec![poly],
            label: seg.label.clone(),
        });
    }
    Ok(out)
}
