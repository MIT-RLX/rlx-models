// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Host LSTM (uni + bidirectional) for SpeechTokenizer's SLSTM bottlenecks.
// PyTorch gate order (i,f,g,o); SLSTM adds a skip (`y + x`, with `x` repeated to
// 2×dim in the bidirectional case).

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One LSTM direction-layer: `seq` `[T][in_d]` → `[T][h]`. `reverse` runs the
/// time loop right-to-left (output still in forward time order).
pub struct LayerW<'a> {
    pub w_ih: &'a [f32], // [4h, in_d]
    pub w_hh: &'a [f32], // [4h, h]
    pub b_ih: &'a [f32],
    pub b_hh: &'a [f32],
}

fn run_layer(l: &LayerW, seq: &[Vec<f32>], in_d: usize, h: usize, reverse: bool) -> Vec<Vec<f32>> {
    let t = seq.len();
    let mut h_prev = vec![0f32; h];
    let mut c_prev = vec![0f32; h];
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); t];
    let order: Vec<usize> = if reverse {
        (0..t).rev().collect()
    } else {
        (0..t).collect()
    };
    for &ti in &order {
        let xt = &seq[ti];
        let mut g = vec![0f32; 4 * h];
        for (r, gr) in g.iter_mut().enumerate() {
            let mut acc = l.b_ih[r] + l.b_hh[r];
            let ih = &l.w_ih[r * in_d..r * in_d + in_d];
            for (k, &xv) in xt.iter().enumerate() {
                acc += ih[k] * xv;
            }
            let hh = &l.w_hh[r * h..r * h + h];
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
        out[ti] = h_new;
    }
    out
}

fn to_seq(x: &[f32], dim: usize, t: usize) -> Vec<Vec<f32>> {
    (0..t)
        .map(|ti| (0..dim).map(|c| x[c * t + ti]).collect())
        .collect()
}

/// Unidirectional multi-layer LSTM with skip. `[dim, T]` → `[dim, T]`.
pub fn lstm(fwd: &[LayerW], x: &[f32], dim: usize, t: usize) -> Vec<f32> {
    let mut seq = to_seq(x, dim, t);
    for l in fwd {
        let in_d = l.w_ih.len() / (4 * dim);
        seq = run_layer(l, &seq, in_d, dim, false);
    }
    let mut res = vec![0f32; dim * t];
    for ti in 0..t {
        for c in 0..dim {
            res[c * t + ti] = seq[ti][c] + x[c * t + ti];
        }
    }
    res
}

/// Bidirectional multi-layer LSTM with skip. `[dim, T]` → `[2*dim, T]`.
/// Output `[0:dim]` = forward + x, `[dim:2dim]` = backward + x.
pub fn bilstm(fwd: &[LayerW], rev: &[LayerW], x: &[f32], dim: usize, t: usize) -> Vec<f32> {
    let mut seq = to_seq(x, dim, t); // [T][cur_in]
    let n_layers = fwd.len();
    for li in 0..n_layers {
        let in_d = fwd[li].w_ih.len() / (4 * dim);
        let hf = run_layer(&fwd[li], &seq, in_d, dim, false);
        let hr = run_layer(&rev[li], &seq, in_d, dim, true);
        // concat [hf, hr] → [T][2dim]
        seq = (0..t)
            .map(|ti| {
                let mut v = Vec::with_capacity(2 * dim);
                v.extend_from_slice(&hf[ti]);
                v.extend_from_slice(&hr[ti]);
                v
            })
            .collect();
    }
    // seq is [T][2dim]; skip adds [x, x].
    let mut res = vec![0f32; 2 * dim * t];
    for ti in 0..t {
        for c in 0..dim {
            res[c * t + ti] = seq[ti][c] + x[c * t + ti];
            res[(dim + c) * t + ti] = seq[ti][dim + c] + x[c * t + ti];
        }
    }
    res
}
