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

//! Prefill attention bias for Gemma 4 unified multimodal (`use_bidirectional_attention: "vision"`).
//!
//! Builds a per-head additive mask `[batch, heads, seq, seq]` where allowed
//! positions are `0` and blocked positions are `-1e4`.

use crate::config::GemmaConfig;
use crate::multimodal::GemmaMultimodalConfig;

const BLOCK: f32 = -1e4;

/// Inclusive-exclusive content span inside a media wrapper (between boi/eoi, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaContentSpan {
    pub start: usize,
    pub end: usize,
}

/// Find image/audio/video placeholder token spans (image_token_id rows only).
pub fn media_content_spans(
    token_ids: &[u32],
    mm_cfg: &GemmaMultimodalConfig,
) -> Vec<MediaContentSpan> {
    let mut spans = Vec::new();
    let mut i = 0usize;
    while i < token_ids.len() {
        if Some(token_ids[i]) == mm_cfg.boi_token_id {
            let start = i + 1;
            let mut j = start;
            while j < token_ids.len() && Some(token_ids[j]) != mm_cfg.eoi_token_id {
                j += 1;
            }
            if j > start {
                spans.push(MediaContentSpan { start, end: j });
            }
            i = j.saturating_add(1);
            continue;
        }
        if Some(token_ids[i]) == mm_cfg.boa_token_id {
            let start = i + 1;
            let mut j = start;
            while j < token_ids.len() && Some(token_ids[j]) != mm_cfg.eoa_token_index {
                j += 1;
            }
            if j > start {
                spans.push(MediaContentSpan { start, end: j });
            }
            i = j.saturating_add(1);
            continue;
        }
        i += 1;
    }
    spans
}

fn in_span(spans: &[MediaContentSpan], i: usize, j: usize) -> bool {
    spans
        .iter()
        .any(|s| i >= s.start && i < s.end && j >= s.start && j < s.end)
}

/// Causal mask with full bidirectional attention inside media placeholder spans.
pub fn build_vision_bidirectional_mask_2d(
    token_ids: &[u32],
    mm_cfg: &GemmaMultimodalConfig,
) -> Vec<f32> {
    let seq = token_ids.len();
    let spans = media_content_spans(token_ids, mm_cfg);
    let mut out = vec![0f32; seq * seq];
    for q in 0..seq {
        for k in 0..seq {
            let allow = if in_span(&spans, q, k) { true } else { k <= q };
            if !allow {
                out[q * seq + k] = BLOCK;
            }
        }
    }
    out
}

/// Expand `[seq, seq]` mask to `[batch, heads, seq, seq]` row-major.
pub fn expand_attn_bias(mask_2d: &[f32], batch: usize, num_heads: usize, seq: usize) -> Vec<f32> {
    assert_eq!(mask_2d.len(), seq * seq);
    let mut out = vec![0f32; batch * num_heads * seq * seq];
    for b in 0..batch {
        for h in 0..num_heads {
            let base = (b * num_heads + h) * seq * seq;
            out[base..base + seq * seq].copy_from_slice(mask_2d);
        }
    }
    out
}

pub fn build_multimodal_prefill_attn_bias(
    token_ids: &[u32],
    lm_cfg: &GemmaConfig,
    mm_cfg: &GemmaMultimodalConfig,
    batch: usize,
) -> Option<Vec<f32>> {
    if !lm_cfg.use_bidirectional_vision() {
        return None;
    }
    let seq = token_ids.len();
    let mask = build_vision_bidirectional_mask_2d(token_ids, mm_cfg);
    Some(expand_attn_bias(
        &mask,
        batch,
        lm_cfg.num_attention_heads,
        seq,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_span_bidirectional_not_causal_across_future() {
        let mm = GemmaMultimodalConfig {
            image_token_id: Some(10),
            boi_token_id: Some(1),
            eoi_token_id: Some(2),
            ..Default::default()
        };
        // text, boi, img, img, eoi, text
        let ids = [99, 1, 10, 10, 2, 88];
        let m = build_vision_bidirectional_mask_2d(&ids, &mm);
        let seq = ids.len();
        // img[0] (idx 2) attends to img[1] (idx 3) — future within span
        assert_eq!(m[2 * seq + 3], 0.0);
        // text at 5 cannot attend to future (none), but can attend to img at 3
        assert_eq!(m[5 * seq + 3], 0.0);
        // text at 0 cannot attend to text at 5
        assert_eq!(m[5], BLOCK);
    }
}
