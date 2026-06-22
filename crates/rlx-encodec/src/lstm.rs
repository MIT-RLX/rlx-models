// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Host LSTM for EnCodec's bottleneck. Runs on the CPU (tiny: latent rate, e.g.
// 75 Hz) — the heavy conv stacks run on the chosen rlx backend. Matches
// PyTorch `nn.LSTM` gate ordering (i, f, g, o). HF `EncodecLSTM` has NO residual
// skip (the layer just learns a near-identity), so the raw output is returned.

use crate::model::LstmW;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `x`: `[dim, T]` channel-first row-major → `[dim, T]` with residual skip.
pub fn forward(w: &LstmW, x: &[f32], dim: usize, t: usize) -> Vec<f32> {
    let h = w.dim;
    debug_assert_eq!(h, dim);
    // per-timestep input [T][dim]
    let mut seq: Vec<Vec<f32>> = (0..t)
        .map(|ti| (0..dim).map(|c| x[c * t + ti]).collect())
        .collect();

    for layer in &w.layers {
        let in_d = layer.w_ih.len() / (4 * h);
        let mut h_prev = vec![0f32; h];
        let mut c_prev = vec![0f32; h];
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(t);
        for xt in &seq {
            let mut g = vec![0f32; 4 * h];
            for (r, gr) in g.iter_mut().enumerate() {
                let mut acc = layer.b_ih[r] + layer.b_hh[r];
                let ih = &layer.w_ih[r * in_d..r * in_d + in_d];
                for (k, &xv) in xt.iter().enumerate() {
                    acc += ih[k] * xv;
                }
                let hh = &layer.w_hh[r * h..r * h + h];
                for (k, &hv) in h_prev.iter().enumerate() {
                    acc += hh[k] * hv;
                }
                *gr = acc;
            }
            let mut h_new = vec![0f32; h];
            let mut c_new = vec![0f32; h];
            for j in 0..h {
                let i = sigmoid(g[j]);
                let f = sigmoid(g[h + j]);
                let gg = g[2 * h + j].tanh();
                let o = sigmoid(g[3 * h + j]);
                let cc = f * c_prev[j] + i * gg;
                c_new[j] = cc;
                h_new[j] = o * cc.tanh();
            }
            h_prev = h_new.clone();
            c_prev = c_new;
            out.push(h_new);
        }
        seq = out;
    }

    // HF EncodecLSTM: out = lstm(x) + x. Back to [dim, T].
    let mut res = vec![0f32; dim * t];
    for ti in 0..t {
        for c in 0..dim {
            res[c * t + ti] = seq[ti][c] + x[c * t + ti];
        }
    }
    res
}
