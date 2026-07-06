// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use std::collections::HashSet;

/// Visual KV indices `[start, end)` in the multimodal cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionKeySpan {
    pub start: usize,
    pub end: usize,
}

impl VisionKeySpan {
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains_key(&self, key_idx: usize) -> bool {
        (self.start..self.end).contains(&key_idx)
    }

    pub fn key_indices(&self) -> impl Iterator<Item = usize> + use<> {
        self.start..self.end
    }
}

/// Fig. 2 / Sec. 4.3 — block text→visual edges; vision→vision unchanged.
pub fn decode_mask_row_causal(past_seq: usize, blocked_visual_keys: &[usize]) -> Vec<f32> {
    let blocked: HashSet<usize> = blocked_visual_keys.iter().copied().collect();
    (0..=past_seq)
        .map(|k| if blocked.contains(&k) { 0.0 } else { 1.0 })
        .collect()
}

/// Sec. 4.2–4.3 — mask `ratio` fraction with highest token entropy first.
pub fn block_highest_entropy_keys(
    span: VisionKeySpan,
    token_entropy: &[f32],
    ratio: f32,
) -> Vec<usize> {
    let n = span.len();
    if n == 0 || ratio <= 0.0 {
        return Vec::new();
    }
    assert_eq!(token_entropy.len(), n);
    let block_n = super::dynamics::block_count(n, ratio).min(n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        token_entropy[b]
            .partial_cmp(&token_entropy[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    order
        .into_iter()
        .take(block_n)
        .map(|i| span.start + i)
        .collect()
}

/// Sec. 3.2 ablation — mask lowest μ first (not the proposed AIF rule).
pub fn block_lowest_mu_keys(span: VisionKeySpan, mu: &[f32], ratio: f32) -> Vec<usize> {
    let n = span.len();
    if n == 0 || ratio <= 0.0 {
        return Vec::new();
    }
    assert_eq!(mu.len(), n);
    let block_n = super::dynamics::block_count(n, ratio).min(n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        mu[a]
            .partial_cmp(&mu[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    order
        .into_iter()
        .take(block_n)
        .map(|i| span.start + i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_mask_blocks_visual_only() {
        let m = decode_mask_row_causal(5, &[1, 2]);
        assert_eq!(m, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn highest_entropy_masked_first() {
        let span = VisionKeySpan { start: 0, end: 4 };
        let ent = vec![0.1, 0.9, 0.5, 0.2];
        let blocked = block_highest_entropy_keys(span, &ent, 0.5);
        assert_eq!(blocked.len(), 2);
        assert!(blocked.contains(&1));
    }
}
