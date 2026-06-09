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

// RLX — token sampling (`LLaDA2MoeModelLM._sample_with_temperature_topk_topp`).

/// Greedy or sampled token + probability from logits `[vocab]`.
pub fn sample_logits(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    do_sample: bool,
) -> (u32, f32) {
    let mut work = logits.to_vec();
    if temperature > 0.0 && (temperature - 1.0).abs() > f32::EPSILON {
        for v in &mut work {
            *v /= temperature;
        }
    }
    top_k_logits(&mut work, top_k);
    top_p_logits(&mut work, top_p);
    let probs = softmax(&work);
    let (tok, prob) = if do_sample {
        sample_multinomial(&probs)
    } else {
        argmax(&probs)
    };
    (tok, prob)
}

fn top_k_logits(logits: &mut [f32], k: Option<usize>) {
    let Some(k) = k else { return };
    if k == 0 || k >= logits.len() {
        return;
    }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = logits[order[k - 1]];
    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

fn top_p_logits(logits: &mut [f32], p: Option<f32>) {
    let Some(p) = p else { return };
    if p >= 1.0 {
        return;
    }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let probs = softmax(logits);
    let mut cum = 0f32;
    let mut cut = logits.len();
    for (i, &idx) in order.iter().enumerate() {
        cum += probs[idx];
        if cum > p {
            cut = i + 1;
            break;
        }
    }
    if cut > 0 {
        let first = order[0];
        for (i, &idx) in order.iter().enumerate() {
            if i >= cut {
                logits[idx] = f32::NEG_INFINITY;
            }
        }
        let _ = first;
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let mut exps = vec![0f32; logits.len()];
    let mut sum = 0f32;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_finite() {
            let e = (v - max).exp();
            exps[i] = e;
            sum += e;
        }
    }
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

fn argmax(probs: &[f32]) -> (u32, f32) {
    let mut best_i = 0usize;
    let mut best_v = probs[0];
    for (i, &v) in probs.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    (best_i as u32, best_v)
}

fn sample_multinomial(probs: &[f32]) -> (u32, f32) {
    let r: f32 = rand_deterministic(probs);
    let mut cum = 0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            return (i as u32, p);
        }
    }
    argmax(probs)
}

fn rand_deterministic(probs: &[f32]) -> f32 {
    let mut h = 0u64;
    for (i, &p) in probs.iter().enumerate() {
        h = h
            .wrapping_mul(31)
            .wrapping_add((p.to_bits() as u64) ^ (i as u64));
    }
    ((h % 10_000) as f32) / 10_000.0
}
