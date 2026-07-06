// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Reference ESN loop (Nakajima-style) vs crate predictors.

use rlx_narma10::host::{ORDER, Rng, generate, nrmse};
use rlx_narma10::{EsnRidge, Narma10Predictor, TrainConfig};

fn reference_esn_nrmse(seed: u64, n: usize, washout: usize, train_frac: f64) -> (f64, f64) {
    let series = generate(n, seed);
    let u = &series.inputs;
    let targets = &series.targets;
    let n_units = 300usize;
    let sr = 0.9f64;
    let input_sparsity = 0.5f64;
    let input_scale = 0.1f64;
    let res_sparsity = 0.2f64;

    let mut rng = Rng::new(seed);
    let mut w_in = vec![0f64; n_units];
    for wi in &mut w_in {
        if rng.uniform01() >= input_sparsity {
            *wi = (rng.uniform01() * 2.0 - 1.0) * input_scale;
        }
    }
    let mut w = vec![0f64; n_units * n_units];
    for i in 0..n_units {
        for j in 0..n_units {
            if i != j && rng.uniform01() < res_sparsity {
                w[i * n_units + j] = rng.uniform01() * 2.0 - 1.0;
            }
        }
    }
    let mut v = vec![0.1f64; n_units];
    v[0] = 1.0;
    for _ in 0..64 {
        let mut out = vec![0f64; n_units];
        for i in 0..n_units {
            let row = &w[i * n_units..(i + 1) * n_units];
            out[i] = row.iter().zip(&v).map(|(a, b)| a * b).sum();
        }
        let norm = out.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        v = out.into_iter().map(|x| x / norm).collect();
    }
    let mut out = vec![0f64; n_units];
    for i in 0..n_units {
        let row = &w[i * n_units..(i + 1) * n_units];
        out[i] = row.iter().zip(&v).map(|(a, b)| a * b).sum();
    }
    let radius = out
        .iter()
        .zip(&v)
        .map(|(a, b)| a * b)
        .sum::<f64>()
        .abs()
        .max(1e-12);
    let scale = sr / radius;
    for x in &mut w {
        *x *= scale;
    }

    let train_n = ((targets.len() as f64) * train_frac).floor() as usize;
    let mut x = vec![0f64; n_units];
    let mut design = Vec::new();
    let mut y_train = Vec::new();

    for t in 0..u.len() {
        if t + 1 < u.len() {
            let mut pre = vec![0f64; n_units];
            for i in 0..n_units {
                let mut s = w_in[i] * u[t];
                let row = &w[i * n_units..(i + 1) * n_units];
                s += row.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>();
                pre[i] = s.tanh();
            }
            x = pre;
        }
        let time = t + 1;
        if time >= ORDER {
            let j = time - ORDER;
            if time >= washout && j < train_n {
                design.push(x.clone());
                y_train.push(targets[j]);
            }
        }
    }

    let feat = n_units + 1;
    let mut xtx = vec![0f64; feat * feat];
    let mut xty = vec![0f64; feat];
    for (state, &yt) in design.iter().zip(&y_train) {
        let mut row = vec![1.0];
        row.extend(state);
        for i in 0..feat {
            for j in 0..=i {
                xtx[i * feat + j] += row[i] * row[j];
            }
            xty[i] += row[i] * yt;
        }
    }
    for i in 0..feat {
        xtx[i * feat + i] += 1e-8;
        for j in 0..i {
            xtx[j * feat + i] = xtx[i * feat + j];
        }
    }
    let w_out = solve(feat, &xtx, &xty);
    let train_pred: Vec<f64> = design
        .iter()
        .map(|state| {
            let mut row = vec![1.0];
            row.extend(state);
            row.iter().zip(&w_out).map(|(a, b)| a * b).sum()
        })
        .collect();
    let train_nrmse = nrmse(&train_pred, &y_train);

    x.fill(0.0);
    let mut preds = Vec::new();
    let mut tgs = Vec::new();
    for t in 0..u.len() {
        if t + 1 < u.len() {
            let mut pre = vec![0f64; n_units];
            for i in 0..n_units {
                let mut s = w_in[i] * u[t];
                let row = &w[i * n_units..(i + 1) * n_units];
                s += row.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>();
                pre[i] = s.tanh();
            }
            x = pre;
        }
        let time = t + 1;
        if time >= ORDER {
            let j = time - ORDER;
            if j >= train_n && j < targets.len() {
                let mut row = vec![1.0];
                row.extend(&x);
                preds.push(row.iter().zip(&w_out).map(|(a, b)| a * b).sum());
                tgs.push(targets[j]);
            }
        }
    }
    (train_nrmse, nrmse(&preds, &tgs))
}

fn solve(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut l = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut y = b.to_vec();
    for i in 0..n {
        let mut s = y[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    y
}

#[test]
fn bare_reference_esn_achieves_literature_nrmse() {
    let (train, test) = reference_esn_nrmse(42, 5_000, 100, 0.75);
    eprintln!("bare reference train {train:.4} test {test:.4}");
    assert!(train < 0.35, "bare train {train:.4}");
    assert!(test < 0.35, "bare test {test:.4}");
}

#[test]
fn esn_ridge_near_reference_nrmse() {
    let series = generate(5_000, 42);
    let cfg = TrainConfig {
        washout: 100,
        train_frac: 0.75,
        ridge_lambda: 1e-8,
        seed: 42,
    };

    let (ref_train, ref_test) = reference_esn_nrmse(42, 5_000, 100, 0.75);
    let mut model = EsnRidge::new();
    let report = model.fit(&series, &cfg).unwrap();
    let pred = model.predict_all(&series, &cfg).unwrap();
    assert_eq!(pred.len(), series.targets.len());

    let test_start = report.split_index;
    let test_nrmse = nrmse(&pred[test_start..], &series.targets[test_start..]);

    eprintln!(
        "bare ref train {ref_train:.4} test {ref_test:.4}; esn_ridge train {:.4} test {test_nrmse:.4}",
        report.train_nrmse
    );

    assert!(
        report.train_nrmse < 0.35,
        "train NRMSE {:.4} too high",
        report.train_nrmse
    );
    assert!(test_nrmse < 0.35, "test NRMSE {:.4} too high", test_nrmse);
}
