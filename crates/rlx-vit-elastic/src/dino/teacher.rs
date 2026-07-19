// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Host-side EMA teacher + DINO output centering.

use std::collections::HashMap;

/// Exponential moving-average update of the teacher params toward the student:
/// `θ_t ← m·θ_t + (1−m)·θ_s`, for every key present in both maps.
pub fn ema_update(
    teacher: &mut HashMap<String, Vec<f32>>,
    student: &HashMap<String, Vec<f32>>,
    m: f32,
) {
    for (k, tv) in teacher.iter_mut() {
        if let Some(sv) = student.get(k) {
            let n = tv.len().min(sv.len());
            for i in 0..n {
                tv[i] = m * tv[i] + (1.0 - m) * sv[i];
            }
        }
    }
}

/// DINO teacher-output center: an EMA of the batch-mean teacher logits,
/// subtracted before the (sharpened) teacher softmax to avoid collapse.
#[derive(Debug, Clone)]
pub struct Center {
    pub c: Vec<f32>,
    pub momentum: f32,
}

impl Center {
    pub fn new(dim: usize, momentum: f32) -> Self {
        Self {
            c: vec![0.0; dim],
            momentum,
        }
    }

    /// Update from a batch of teacher logits `[rows, dim]` (row-major):
    /// `c ← m·c + (1−m)·mean_rows(logits)`.
    pub fn update(&mut self, logits: &[f32], rows: usize, dim: usize) {
        if rows == 0 {
            return;
        }
        let inv = 1.0 / rows as f32;
        for d in 0..dim {
            let mut s = 0.0f32;
            for r in 0..rows {
                s += logits[r * dim + d];
            }
            let mean = s * inv;
            self.c[d] = self.momentum * self.c[d] + (1.0 - self.momentum) * mean;
        }
    }
}

/// Softmax of `(logits − center) / temp` per row → the teacher target
/// distribution `[rows, dim]` (host-side; fed as a stop-grad graph input).
pub fn teacher_targets(
    logits: &[f32],
    rows: usize,
    dim: usize,
    temp: f32,
    center: &[f32],
) -> Vec<f32> {
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        let row = &logits[r * dim..(r + 1) * dim];
        // Stable softmax of (row - center) / temp.
        let mut mx = f32::NEG_INFINITY;
        for d in 0..dim {
            let z = (row[d] - center[d]) / temp;
            if z > mx {
                mx = z;
            }
        }
        let mut sum = 0.0f32;
        for d in 0..dim {
            let z = ((row[d] - center[d]) / temp - mx).exp();
            out[r * dim + d] = z;
            sum += z;
        }
        let inv = 1.0 / sum.max(1e-20);
        for d in 0..dim {
            out[r * dim + d] *= inv;
        }
    }
    out
}
