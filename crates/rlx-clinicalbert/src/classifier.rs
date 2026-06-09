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

//! Multinomial logistic regression on top of frozen ClinicalBERT pooler
//! features — for sentence-pair classification benchmarks (MedNLI).
//!
//! Pure Rust, dense FP32, mini-batch SGD with momentum + L2 weight decay.
//! Uses `rlx_cpu::blas::sgemm_bias` for the forward matmul so a 14k-example
//! training run finishes in seconds.

use anyhow::{Result, bail};

/// One row of feature `[hidden]` + integer label.
pub struct LabeledFeature<'a> {
    pub features: &'a [f32],
    pub label: usize,
}

/// Trained `[num_classes, hidden]` weight matrix + per-class bias.
pub struct LinearClassifier {
    pub hidden: usize,
    pub num_classes: usize,
    /// `[num_classes, hidden]` row-major. Stored as `[hidden, num_classes]`
    /// transposed (so `features @ weight_t = logits`).
    weight_t: Vec<f32>,
    bias: Vec<f32>,
}

impl LinearClassifier {
    /// Initialize weights with a small Gaussian scale (Xavier-ish).
    pub fn new(hidden: usize, num_classes: usize) -> Self {
        let scale = (2.0_f32 / hidden as f32).sqrt();
        let mut weight_t = vec![0f32; hidden * num_classes];
        // Deterministic pseudo-random — Park-Miller LCG so runs are
        // reproducible without pulling in a `rand` dependency.
        let mut state: u32 = 0x9e37_79b9;
        for w in weight_t.iter_mut() {
            state = state.wrapping_mul(48_271).wrapping_add(0x9e37_79b9);
            let u = (state >> 16) as f32 / 65536.0;
            // Box-Muller-lite: shift so range is roughly [-1, 1).
            *w = (u * 2.0 - 1.0) * scale * 0.5;
        }
        Self {
            hidden,
            num_classes,
            weight_t,
            bias: vec![0f32; num_classes],
        }
    }

    /// Predict the argmax class for one example.
    pub fn predict(&self, features: &[f32]) -> Result<usize> {
        if features.len() != self.hidden {
            bail!(
                "LinearClassifier::predict: expected {} features, got {}",
                self.hidden,
                features.len()
            );
        }
        let mut logits = vec![0f32; self.num_classes];
        // logits = features @ weight_t + bias
        rlx_cpu::blas::sgemm_bias(
            features,
            &self.weight_t,
            &self.bias,
            &mut logits,
            1,
            self.hidden,
            self.num_classes,
        );
        let mut best = 0usize;
        let mut best_val = logits[0];
        for (j, &v) in logits.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best = j;
            }
        }
        Ok(best)
    }

    /// Batched argmax accuracy on a frozen dataset.
    pub fn accuracy(&self, examples: &[LabeledFeature<'_>]) -> Result<f32> {
        if examples.is_empty() {
            return Ok(0.0);
        }
        // Pack all features into a contiguous `[N, hidden]` matrix and do a
        // single GEMM. Cheap on CPU; saves N kernel-launch overhead.
        let n = examples.len();
        let mut feats = vec![0f32; n * self.hidden];
        let mut labels = vec![0usize; n];
        for (i, ex) in examples.iter().enumerate() {
            if ex.features.len() != self.hidden {
                bail!(
                    "LinearClassifier::accuracy: row {i} has {} features, expected {}",
                    ex.features.len(),
                    self.hidden
                );
            }
            feats[i * self.hidden..(i + 1) * self.hidden].copy_from_slice(ex.features);
            labels[i] = ex.label;
        }
        let mut logits = vec![0f32; n * self.num_classes];
        rlx_cpu::blas::sgemm_bias(
            &feats,
            &self.weight_t,
            &self.bias,
            &mut logits,
            n,
            self.hidden,
            self.num_classes,
        );
        let mut correct = 0usize;
        for i in 0..n {
            let row = &logits[i * self.num_classes..(i + 1) * self.num_classes];
            let pred = row
                .iter()
                .enumerate()
                .fold(
                    (0usize, row[0]),
                    |(bi, bv), (j, &v)| {
                        if v > bv { (j, v) } else { (bi, bv) }
                    },
                )
                .0;
            if pred == labels[i] {
                correct += 1;
            }
        }
        Ok(correct as f32 / n as f32)
    }
}

/// Training hyperparameters with sensible defaults for short, dense probes.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Number of full passes over the training set.
    pub epochs: usize,
    /// Mini-batch size.
    pub batch: usize,
    /// SGD step size.
    pub lr: f32,
    /// L2 weight decay coefficient.
    pub l2: f32,
    /// Momentum (0.0 to disable).
    pub momentum: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 20,
            batch: 32,
            lr: 0.1,
            l2: 1e-4,
            momentum: 0.9,
        }
    }
}

/// Train a multinomial logistic regression on frozen features with mini-batch
/// SGD. Each row of `train` is one `(features [H], label)` pair. Reports the
/// final training accuracy and prints a per-epoch summary when `verbose`.
pub fn train_logreg(
    hidden: usize,
    num_classes: usize,
    train: &[LabeledFeature<'_>],
    cfg: &TrainConfig,
    verbose: bool,
) -> Result<LinearClassifier> {
    if train.is_empty() {
        bail!("train_logreg: empty training set");
    }
    let mut clf = LinearClassifier::new(hidden, num_classes);

    // Pre-pack training features into a contiguous `[N, hidden]` so each
    // mini-batch is a simple slice + sgemm_bias.
    let n = train.len();
    let mut feats = vec![0f32; n * hidden];
    let mut labels = vec![0u32; n];
    for (i, ex) in train.iter().enumerate() {
        if ex.features.len() != hidden {
            bail!(
                "train row {i} has {} features, expected {hidden}",
                ex.features.len()
            );
        }
        if ex.label >= num_classes {
            bail!(
                "train row {i} label {} ≥ num_classes {num_classes}",
                ex.label
            );
        }
        feats[i * hidden..(i + 1) * hidden].copy_from_slice(ex.features);
        labels[i] = ex.label as u32;
    }

    // Mini-batch index permutation (Park-Miller LCG, reproducible).
    let mut perm: Vec<usize> = (0..n).collect();
    let mut rng_state: u32 = 0x1234_5678;
    let lcg = |s: &mut u32| -> u32 {
        *s = s.wrapping_mul(48_271).wrapping_add(0x9e37_79b9);
        *s
    };

    // Momentum buffers, same shape as parameters.
    let mut vel_w = vec![0f32; hidden * num_classes];
    let mut vel_b = vec![0f32; num_classes];
    // Scratch for logits and softmax.
    let mut logits = vec![0f32; cfg.batch * num_classes];

    for epoch in 0..cfg.epochs {
        // Shuffle perm.
        for i in (1..n).rev() {
            let j = (lcg(&mut rng_state) as usize) % (i + 1);
            perm.swap(i, j);
        }

        let mut epoch_loss = 0f32;
        let mut epoch_correct = 0usize;
        let mut seen = 0usize;

        for chunk in perm.chunks(cfg.batch) {
            let bs = chunk.len();
            // Pack a mini-batch into contiguous buffers.
            let mut xb = vec![0f32; bs * hidden];
            let mut yb = vec![0u32; bs];
            for (i, &idx) in chunk.iter().enumerate() {
                xb[i * hidden..(i + 1) * hidden]
                    .copy_from_slice(&feats[idx * hidden..(idx + 1) * hidden]);
                yb[i] = labels[idx];
            }

            // Forward: logits = xb @ weight_t + bias  (bs × num_classes)
            if logits.len() < bs * num_classes {
                logits.resize(bs * num_classes, 0.0);
            }
            let logits_slice = &mut logits[..bs * num_classes];
            rlx_cpu::blas::sgemm_bias(
                &xb,
                &clf.weight_t,
                &clf.bias,
                logits_slice,
                bs,
                hidden,
                num_classes,
            );

            // Softmax + cross-entropy + count correct + build gradient
            // (delta = softmax - one_hot).
            let mut delta = vec![0f32; bs * num_classes];
            for i in 0..bs {
                let row = &mut logits_slice[i * num_classes..(i + 1) * num_classes];
                let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for v in row.iter_mut() {
                    *v = (*v - max_logit).exp();
                    sum += *v;
                }
                let inv = 1.0 / sum;
                let mut argmax = 0usize;
                let mut argmax_val = -1f32;
                for (j, v) in row.iter_mut().enumerate() {
                    *v *= inv;
                    if *v > argmax_val {
                        argmax_val = *v;
                        argmax = j;
                    }
                    delta[i * num_classes + j] = *v;
                }
                let y = yb[i] as usize;
                delta[i * num_classes + y] -= 1.0;
                epoch_loss += -row[y].max(1e-12).ln();
                if argmax == y {
                    epoch_correct += 1;
                }
            }

            // Gradient: dW = xb^T @ delta  (hidden × num_classes),
            // averaged over the batch and with L2 decay.
            let inv_bs = 1.0 / bs as f32;
            // grad_w is stored as `[hidden, num_classes]` to match weight_t.
            let mut grad_w = vec![0f32; hidden * num_classes];
            // Manual GEMM for the transpose-A pattern (no sgemm_at helper).
            for h_idx in 0..hidden {
                for c_idx in 0..num_classes {
                    let mut s = 0f32;
                    for i in 0..bs {
                        s += xb[i * hidden + h_idx] * delta[i * num_classes + c_idx];
                    }
                    grad_w[h_idx * num_classes + c_idx] = s * inv_bs;
                }
            }
            let mut grad_b = vec![0f32; num_classes];
            for i in 0..bs {
                for c_idx in 0..num_classes {
                    grad_b[c_idx] += delta[i * num_classes + c_idx];
                }
            }
            for v in grad_b.iter_mut() {
                *v *= inv_bs;
            }

            // SGD with momentum: v ← μ·v + g + λ·w ; w ← w - lr·v
            for j in 0..hidden * num_classes {
                let g = grad_w[j] + cfg.l2 * clf.weight_t[j];
                vel_w[j] = cfg.momentum * vel_w[j] + g;
                clf.weight_t[j] -= cfg.lr * vel_w[j];
            }
            for j in 0..num_classes {
                vel_b[j] = cfg.momentum * vel_b[j] + grad_b[j];
                clf.bias[j] -= cfg.lr * vel_b[j];
            }

            seen += bs;
        }

        if verbose {
            let acc = epoch_correct as f32 / seen as f32;
            let loss = epoch_loss / seen as f32;
            eprintln!(
                "[clf] epoch {:>3}: train_loss={:.4} train_acc={:.4}",
                epoch + 1,
                loss,
                acc
            );
        }
    }

    Ok(clf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny linearly-separable 2D dataset to confirm the optimizer converges
    /// — three clusters at distinct centroids; logreg must reach 100%.
    #[test]
    fn logreg_separates_three_clusters() {
        let hidden = 2;
        let num_classes = 3;
        let centroids = [[0.0_f32, 0.0], [3.0, 0.0], [0.0, 3.0]];
        let mut features: Vec<Vec<f32>> = Vec::new();
        let mut labels: Vec<usize> = Vec::new();
        for (c, ctr) in centroids.iter().enumerate() {
            for k in 0..40 {
                let jitter = (k as f32 * 0.07) - 1.4;
                features.push(vec![ctr[0] + jitter, ctr[1] + jitter * 0.5]);
                labels.push(c);
            }
        }
        let train: Vec<LabeledFeature> = features
            .iter()
            .zip(&labels)
            .map(|(f, l)| LabeledFeature {
                features: f.as_slice(),
                label: *l,
            })
            .collect();
        let cfg = TrainConfig {
            epochs: 100,
            batch: 16,
            lr: 0.2,
            l2: 0.0,
            momentum: 0.9,
        };
        let clf = train_logreg(hidden, num_classes, &train, &cfg, false).unwrap();
        let acc = clf.accuracy(&train).unwrap();
        assert!(acc > 0.98, "got {acc}");
    }
}
