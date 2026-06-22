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

//! Detection postprocessing: contrastive class scores, box decoding (cxcywh →
//! xyxy in original-image pixels), and text-token grounding, matching HF
//! `post_process_grounded_object_detection`.

use crate::nn::sigmoid;

/// A single detection.
#[derive(Debug, Clone)]
pub struct Detection {
    /// `[x0, y0, x1, y1]` in original-image pixel coordinates.
    pub bbox: [f32; 4],
    /// Confidence (max sigmoid logit over text tokens).
    pub score: f32,
    /// Text-token positions that ground this detection (`prob > text_threshold`).
    pub token_indices: Vec<usize>,
    /// Decoded phrase (filled only when a tokenizer is available).
    pub label: Option<String>,
}

/// Contrastive class logits `[nq, lt]` = `hidden · text`, `-inf` on padded tokens.
pub fn contrastive_logits(
    hidden: &[f32],
    text: &[f32],
    text_mask: &[u8],
    nq: usize,
    lt: usize,
    d: usize,
) -> Vec<f32> {
    let mut logits = vec![f32::NEG_INFINITY; nq * lt];
    for q in 0..nq {
        for j in 0..lt {
            if text_mask[j] == 0 {
                continue;
            }
            let mut dot = 0f32;
            for c in 0..d {
                dot += hidden[q * d + c] * text[j * d + c];
            }
            logits[q * lt + j] = dot;
        }
    }
    logits
}

/// Decode detections from the decoder outputs.
#[allow(clippy::too_many_arguments)]
pub fn post_process(
    hidden: &[f32],
    boxes: &[f32],
    text: &[f32],
    text_mask: &[u8],
    d: usize,
    orig_h: usize,
    orig_w: usize,
    box_threshold: f32,
    text_threshold: f32,
) -> Vec<Detection> {
    let nq = hidden.len() / d;
    let lt = text.len() / d;
    let logits = contrastive_logits(hidden, text, text_mask, nq, lt, d);

    let (w, h) = (orig_w as f32, orig_h as f32);
    let mut dets = Vec::new();
    for q in 0..nq {
        // prob = sigmoid(logit); score = max over tokens.
        let mut score = 0f32;
        for j in 0..lt {
            let p = sigmoid(logits[q * lt + j]);
            if p > score {
                score = p;
            }
        }
        if score <= box_threshold {
            continue;
        }
        let token_indices: Vec<usize> = (0..lt)
            .filter(|&j| text_mask[j] != 0 && sigmoid(logits[q * lt + j]) > text_threshold)
            .collect();

        let cx = boxes[q * 4];
        let cy = boxes[q * 4 + 1];
        let bw = boxes[q * 4 + 2];
        let bh = boxes[q * 4 + 3];
        let bbox = [
            (cx - bw * 0.5) * w,
            (cy - bh * 0.5) * h,
            (cx + bw * 0.5) * w,
            (cy + bh * 0.5) * h,
        ];
        dets.push(Detection {
            bbox,
            score,
            token_indices,
            label: None,
        });
    }
    // Highest score first.
    dets.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    dets
}

/// Decode the grounded phrase for each detection from the prompt token ids.
#[cfg(feature = "tokenizer")]
pub fn label_detections(
    dets: &mut [Detection],
    input_ids: &[u32],
    tokenizer_json: &std::path::Path,
) -> anyhow::Result<()> {
    use anyhow::anyhow;
    let tk = tokenizers::Tokenizer::from_file(tokenizer_json)
        .map_err(|e| anyhow!("load tokenizer: {e}"))?;
    for det in dets.iter_mut() {
        let ids: Vec<u32> = det
            .token_indices
            .iter()
            .filter_map(|&j| input_ids.get(j).copied())
            .collect();
        let phrase = tk.decode(&ids, true).map_err(|e| anyhow!("decode: {e}"))?;
        det.label = Some(phrase.trim().to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_decode_and_threshold() {
        let d = 2;
        // query0 aligns with token0 strongly; query1 weak.
        let hidden = vec![5.0, -5.0, /*q0*/ -5.0, -5.0 /*q1*/];
        let text = vec![1.0, 0.0, /*t0*/ 0.0, 1.0 /*t1*/];
        let text_mask = vec![1u8, 1];
        // boxes cxcywh normalized.
        let boxes = vec![
            0.5, 0.5, 0.4, 0.2, /*q0*/ 0.5, 0.5, 0.1, 0.1, /*q1*/
        ];
        let dets = post_process(&hidden, &boxes, &text, &text_mask, d, 100, 200, 0.3, 0.25);
        // Only query0 passes the box threshold.
        assert_eq!(dets.len(), 1);
        let det = &dets[0];
        assert!(det.score > 0.9);
        assert_eq!(det.token_indices, vec![0]);
        // bbox: cx 0.5*200=100, w 0.4*200=80 → x0=60, x1=140; cy 0.5*100=50, h 0.2*100=20 → y0=40,y1=60.
        assert!((det.bbox[0] - 60.0).abs() < 1e-3);
        assert!((det.bbox[2] - 140.0).abs() < 1e-3);
        assert!((det.bbox[1] - 40.0).abs() < 1e-3);
        assert!((det.bbox[3] - 60.0).abs() < 1e-3);
    }
}
