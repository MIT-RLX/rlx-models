use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub const RMS_EPS: f32 = 1e-8;

#[derive(Debug, Clone)]
pub struct Embedding {
    pub weight: Array2<f32>,
}

impl Embedding {
    pub fn forward(&self, ids: &[u32]) -> Array2<f32> {
        let t = ids.len();
        let d = self.weight.dim().1;
        let mut out = Array2::<f32>::zeros((t, d));
        for (ti, &id) in ids.iter().enumerate() {
            let row = self.weight.row(id as usize);
            for di in 0..d {
                out[[ti, di]] = row[di];
            }
        }
        out
    }

    pub fn forward_one(&self, id: u32) -> Array1<f32> {
        self.weight.row(id as usize).to_owned()
    }
}

pub fn linear(x: ArrayView2<f32>, w: &Array2<f32>) -> Array2<f32> {
    x.dot(&w.t())
}

pub fn rms_norm(x: ArrayView2<f32>, alpha: &Array1<f32>) -> Array2<f32> {
    let (t, c) = x.dim();
    let mut out = Array2::<f32>::zeros((t, c));
    let inv_c = 1.0 / c as f32;
    for ti in 0..t {
        let row = x.row(ti);
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() * inv_c;
        let scale = 1.0 / (mean_sq + RMS_EPS).sqrt();
        for ci in 0..c {
            out[[ti, ci]] = row[ci] * scale * alpha[ci];
        }
    }
    out
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn swiglu_mlp(
    x: ArrayView2<f32>,
    linear_in: &Array2<f32>,
    linear_out: &Array2<f32>,
) -> Array2<f32> {
    let gate_up = linear(x, linear_in);
    let (t, two_h) = gate_up.dim();
    let h = two_h / 2;
    let mut gated = Array2::<f32>::zeros((t, h));
    for ti in 0..t {
        for hi in 0..h {
            let gate = silu(gate_up[[ti, hi]]);
            let up = gate_up[[ti, h + hi]];
            gated[[ti, hi]] = gate * up;
        }
    }
    linear(gated.view(), linear_out)
}

pub fn softmax_last_dim(logits: &Array2<f32>) -> Array2<f32> {
    let (t, v) = logits.dim();
    let mut out = Array2::<f32>::zeros((t, v));
    for ti in 0..t {
        let row = logits.row(ti);
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for vi in 0..v {
            let e = (row[vi] - max).exp();
            out[[ti, vi]] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for vi in 0..v {
            out[[ti, vi]] *= inv;
        }
    }
    out
}

pub fn rope_tables(
    head_dim: usize,
    max_period: usize,
    positions: &[usize],
) -> (Array2<f32>, Array2<f32>) {
    let half = head_dim / 2;
    let t = positions.len();
    let mut inv_freq = Vec::with_capacity(half);
    for i in 0..half {
        inv_freq.push(1.0 / (max_period as f32).powf(i as f32 / half as f32));
    }
    let mut cos = Array2::<f32>::zeros((t, head_dim));
    let mut sin = Array2::<f32>::zeros((t, head_dim));
    for (ti, &pos) in positions.iter().enumerate() {
        for hi in 0..half {
            let f = pos as f32 * inv_freq[hi];
            let c = f.cos();
            let s = f.sin();
            cos[[ti, hi]] = c;
            cos[[ti, hi + half]] = c;
            sin[[ti, hi]] = s;
            sin[[ti, hi + half]] = s;
        }
    }
    (cos, sin)
}

pub fn apply_rope_vec(q: &mut [f32], k: &mut [f32], cos: ArrayView1<f32>, sin: ArrayView1<f32>) {
    let half = cos.len() / 2;
    for hi in 0..half {
        let qc = q[hi];
        let qs = q[hi + half];
        q[hi] = qc * cos[hi] - qs * sin[hi];
        q[hi + half] = qs * cos[hi + half] + qc * sin[hi + half];
        let kc = k[hi];
        let ks = k[hi + half];
        k[hi] = kc * cos[hi] - ks * sin[hi];
        k[hi + half] = ks * cos[hi + half] + kc * sin[hi + half];
    }
}
