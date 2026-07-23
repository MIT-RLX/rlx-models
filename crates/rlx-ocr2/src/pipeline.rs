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

//! End-to-end OCR: image → detector → CRAFT grouping → per-line recognizer → text.

use crate::detection::Detector;
use crate::grouping::group_lines;
use crate::preprocess::{crop_line_luma, detector_input};
use crate::runner::Recognizer;
use anyhow::{Result, anyhow};
use rlx_runtime::Device;
use std::path::Path;

const DET_HW: usize = 240;

#[derive(Clone, Debug)]
pub struct OcrLine {
    pub text: String,
    pub bbox: (u32, u32, u32, u32), // x0,y0,x1,y1 in original-image pixels
}

pub struct Ocr2 {
    detector: Detector,
    recognizer: Recognizer,
    thresh: f32,
    rescorer: Option<crate::rescore::Rescorer>,
    beam: usize,
}

impl Ocr2 {
    pub fn load(
        recipe: &Path,
        det_weights: &Path,
        rec_weights: &Path,
        codemap: &Path,
        device: Device,
    ) -> Result<Self> {
        // Grouping only reads region_score + link_score_horizontal; build a detector that
        // outputs just those two heads (prunes ~28% of ops — the other 5 heads' conv/upsample/
        // softmax tails — with no effect on the retained heatmaps).
        let heads = vec!["region_score".to_string(), "link_score_horizontal".to_string()];
        Ok(Self {
            detector: Detector::load_heads(recipe, det_weights, device, heads)?,
            recognizer: Recognizer::load(rec_weights, codemap, device)?,
            thresh: 0.5,
            rescorer: None,
            beam: 12,
        })
    }

    /// Attach the correction stack (n-gram + lexicon); recognition then uses beam+rescore.
    pub fn with_rescorer(mut self, rescorer: crate::rescore::Rescorer) -> Self {
        self.rescorer = Some(rescorer);
        self
    }

    /// Detect text lines and recognize each, top-to-bottom.
    pub fn recognize_image(&self, path: &Path) -> Result<Vec<OcrLine>> {
        let timed = crate::env::timing();
        let t0 = std::time::Instant::now();
        let (det_in, lb, gray) = detector_input(path)?;
        let t_pre = t0.elapsed();
        let td = std::time::Instant::now();
        let heads = self.detector.forward(&det_in)?;
        let t_det = td.elapsed();
        let head = |name: &str| -> Result<&[f32]> {
            heads
                .iter()
                .find(|(h, _)| h == name)
                .map(|(_, d)| d.as_slice())
                .ok_or_else(|| anyhow!("missing head {name}"))
        };
        let region = head("region_score")?;
        let link_h = head("link_score_horizontal")?;

        let tg = std::time::Instant::now();
        let lines = group_lines(region, link_h, DET_HW, self.thresh, 10);
        let t_grp = tg.elapsed();
        let tr = std::time::Instant::now();
        let n_lines = lines.len();
        let mut out = Vec::new();
        for b in lines {
            // heatmap (240) → detector canvas (×2) → original pixels (inverse letterbox)
            let mx = |v: f32| (v * 2.0 - lb.ox) / lb.scale;
            let my = |v: f32| (v * 2.0 - lb.oy) / lb.scale;
            let (x0, x1, y0, y1) = (mx(b.x0), mx(b.x1), my(b.y0), my(b.y1));
            let (padx, pady) = ((x1 - x0) * 0.02, (y1 - y0) * 0.15);
            let x0 = (x0 - padx).max(0.0) as u32;
            let y0 = (y0 - pady).max(0.0) as u32;
            let x1 = (x1 + padx).min(lb.orig_w as f32) as u32;
            let y1 = (y1 + pady).min(lb.orig_h as f32) as u32;
            if let Some((luma, w)) = crop_line_luma(&gray, x0, y0, x1, y1) {
                let text = match &self.rescorer {
                    Some(r) => self.recognizer.recognize_with_rescorer(&luma, w, r, self.beam)?,
                    None => self.recognizer.recognize(&luma, w)?,
                };
                if !text.trim().is_empty() {
                    out.push(OcrLine { text, bbox: (x0, y0, x1, y1) });
                }
            }
        }
        if timed {
            let t_rec = tr.elapsed();
            eprintln!(
                "[timing] preprocess={:.1}ms detector={:.1}ms grouping={:.2}ms recognize({} lines)={:.1}ms total={:.1}ms",
                t_pre.as_secs_f64() * 1e3,
                t_det.as_secs_f64() * 1e3,
                t_grp.as_secs_f64() * 1e3,
                n_lines,
                t_rec.as_secs_f64() * 1e3,
                t0.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok(out)
    }
}
