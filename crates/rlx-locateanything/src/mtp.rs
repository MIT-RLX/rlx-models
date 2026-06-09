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

//! Parallel box decoding — MTP sampling (ported from HF `generate_utils`).

use crate::generation::TokenIds;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoxFrameKind {
    Empty,
    Legal,
    Illegal,
}

fn prob_at(probs: &[f32], vocab: usize, pos: usize, token: u32) -> f32 {
    probs
        .get(pos * vocab + token as usize)
        .copied()
        .unwrap_or(0.0)
}

pub fn is_valid_box_frame(probs: &[f32], vocab: usize, ids: &TokenIds) -> BoxFrameKind {
    let p_start = prob_at(probs, vocab, 0, ids.box_start);
    if p_start >= 0.6 {
        let none = prob_at(probs, vocab, 1, ids.none_token);
        let end = prob_at(probs, vocab, 2, ids.box_end);
        let null3 = prob_at(probs, vocab, 3, ids.null_token);
        let null4 = prob_at(probs, vocab, 4, ids.null_token);
        if none > 0.2 && end > 0.2 && null3 > 0.1 && null4 > 0.1 {
            return BoxFrameKind::Empty;
        }
    }
    let end_score = prob_at(probs, vocab, 5, ids.box_end)
        + prob_at(probs, vocab, 5, ids.none_token)
        + prob_at(probs, vocab, 5, ids.im_end);
    if end_score >= 0.2 {
        BoxFrameKind::Legal
    } else {
        BoxFrameKind::Illegal
    }
}

/// Decode 6-position MTP block from per-position logits rows `logits: [6 * vocab]`.
pub fn decode_bbox_block(
    logits: &[f32],
    vocab: usize,
    ids: &TokenIds,
    generation_mode: &str,
) -> Option<Vec<u32>> {
    let block = 6usize;
    if logits.len() < block * vocab {
        return None;
    }
    let mut probs = vec![0f32; block * vocab];
    for t in 0..block {
        let row = &logits[t * vocab..(t + 1) * vocab];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|x| (x - max).exp()).sum();
        for (i, &x) in row.iter().enumerate() {
            probs[t * vocab + i] = (x - max).exp() / sum;
        }
    }
    let frame = is_valid_box_frame(&probs[..vocab], vocab, ids);
    match frame {
        BoxFrameKind::Empty => Some(vec![
            ids.box_start,
            ids.none_token,
            ids.box_end,
            ids.null_token,
            ids.null_token,
            ids.null_token,
        ]),
        BoxFrameKind::Illegal => None,
        BoxFrameKind::Legal => decode_bbox_coords(&probs, vocab, ids, generation_mode),
    }
}

fn decode_bbox_coords(
    probs: &[f32],
    vocab: usize,
    ids: &TokenIds,
    generation_mode: &str,
) -> Option<Vec<u32>> {
    let keep_k = 5usize;
    let mut coords = [0u32; 4];
    for i in 0..4 {
        let row = &probs[(1 + i) * vocab..(2 + i) * vocab];
        let mut top: Vec<(f32, u32)> = row
            .iter()
            .enumerate()
            .map(|(id, &p)| (p, id as u32))
            .collect();
        top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        top.truncate(keep_k);
        let valid: Vec<_> = top
            .iter()
            .filter(|(_, id)| *id >= ids.coord_start && *id <= ids.coord_end)
            .collect();
        if valid.is_empty() {
            return None;
        }
        coords[i] = valid[0].1;
        if generation_mode == "hybrid" && valid.len() > 1 && valid[0].0 < 0.9 {
            let min_id = valid.iter().map(|(_, id)| *id).min().unwrap();
            let max_id = valid.iter().map(|(_, id)| *id).max().unwrap();
            if max_id - min_id > 60 {
                coords[i] = 0;
            }
        }
    }
    Some(vec![
        ids.box_start,
        coords[0],
        coords[1],
        coords[2],
        coords[3],
        ids.box_end,
    ])
}

#[derive(Debug, Clone)]
pub struct PatternOut {
    pub kind: &'static str,
    pub tokens: Vec<u32>,
    pub need_ar: bool,
    pub terminal: bool,
}

pub fn handle_pattern(tokens: &[u32], ids: &TokenIds, generation_mode: &str) -> PatternOut {
    if tokens.is_empty() {
        return PatternOut {
            kind: "im_end",
            tokens: vec![ids.im_end],
            need_ar: false,
            terminal: true,
        };
    }
    if tokens[0] == ids.null_token {
        return PatternOut {
            kind: "im_end",
            tokens: vec![ids.im_end],
            need_ar: false,
            terminal: true,
        };
    }
    if tokens[0] == ids.im_end {
        return PatternOut {
            kind: "im_end",
            tokens: vec![ids.im_end],
            need_ar: false,
            terminal: true,
        };
    }
    if tokens.len() >= 2 && tokens[0] == ids.box_start && tokens[1] == ids.none_token {
        return PatternOut {
            kind: "empty_box",
            tokens: vec![ids.box_start, ids.none_token, ids.box_end],
            need_ar: false,
            terminal: false,
        };
    }
    if tokens[0] == ids.box_start {
        let mut coord_ix = 1usize;
        for &c in &tokens[1..tokens.len().min(5)] {
            if c >= ids.coord_start && c <= ids.coord_end {
                coord_ix += 1;
            } else {
                break;
            }
        }
        if coord_ix == 5 && tokens.get(5) == Some(&ids.box_end) {
            return PatternOut {
                kind: "coord_box",
                tokens: tokens.to_vec(),
                need_ar: false,
                terminal: false,
            };
        }
        if coord_ix == 3 && tokens.get(3) == Some(&ids.box_end) {
            return PatternOut {
                kind: "point_box",
                tokens: tokens[..4].to_vec(),
                need_ar: false,
                terminal: false,
            };
        }
        if generation_mode == "fast" {
            return PatternOut {
                kind: "coord_box",
                tokens: tokens.to_vec(),
                need_ar: false,
                terminal: false,
            };
        }
        return PatternOut {
            kind: "error_box",
            tokens: tokens[..coord_ix].to_vec(),
            need_ar: true,
            terminal: false,
        };
    }
    let mut out: Vec<u32> = tokens.to_vec();
    if let Some(pos) = out.iter().position(|&t| t == ids.null_token) {
        out.truncate(pos);
    }
    if out.len() >= 2 && out[out.len() - 1] == out[out.len() - 2] {
        out.pop();
    }
    PatternOut {
        kind: "ref_object",
        tokens: out,
        need_ar: false,
        terminal: false,
    }
}

/// Greedy argmax per row in a logits slab `[n_pos * vocab]`.
pub fn argmax_rows(logits: &[f32], vocab: usize, n_pos: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(n_pos);
    for t in 0..n_pos {
        let row = &logits[t * vocab..(t + 1) * vocab];
        let (id, _) = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        out.push(id as u32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::TokenIds;

    #[test]
    fn handle_pattern_empty_box() {
        let ids = TokenIds {
            box_start: 10,
            box_end: 11,
            coord_start: 100,
            coord_end: 200,
            ref_start: 0,
            ref_end: 0,
            none_token: 12,
            null_token: 13,
            switch_token: 14,
            text_mask: 15,
            im_end: 99,
        };
        let pat = handle_pattern(&[10, 12, 11, 13, 13, 13], &ids, "hybrid");
        assert_eq!(pat.kind, "empty_box");
        assert_eq!(pat.tokens.len(), 3);
    }
}
