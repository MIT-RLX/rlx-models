//! Monotonic-alignment + latent-sampling glue between the VITS subgraphs.
//!
//! Faithful Rust port of the NumPy stage in `tiny_tts/infer_onnx.py` that sits
//! between the duration predictor and the flow:
//!   1. `w = exp(logw) * x_mask * length_scale`, `w_ceil = ceil(w)`
//!   2. `y_len = max(1, sum(w_ceil))` frames
//!   3. build the hard monotonic alignment `attn[t_y, t_x]` from `w_ceil`
//!   4. expand the prior stats `m_p`, `logs_p` from phones → frames via `attn`
//!   5. sample `z_p = m_p + N(0,1) * exp(logs_p) * noise_scale`
//!
//! Tensors are flat row-major; batch is always 1 so the leading axis is dropped.

/// Small deterministic Gaussian source (xorshift128+ → Box–Muller). The choice of
/// RNG is acoustically irrelevant; a fixed seed makes synthesis reproducible.
pub struct Rng {
    s0: u64,
    s1: u64,
    spare: Option<f32>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // splitmix64 to disperse the seed into two non-zero state words.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            s0: next() | 1,
            s1: next() | 1,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut s1 = self.s0;
        let s0 = self.s1;
        self.s0 = s0;
        s1 ^= s1 << 23;
        self.s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
        self.s1.wrapping_add(s0)
    }

    fn next_unit(&mut self) -> f64 {
        // 53-bit mantissa in [0, 1).
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal sample.
    pub fn randn(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        // Box–Muller; avoid log(0).
        let mut u1 = self.next_unit();
        while u1 <= f64::MIN_POSITIVE {
            u1 = self.next_unit();
        }
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        self.spare = Some((r * theta.sin()) as f32);
        (r * theta.cos()) as f32
    }
}

/// Per-phone frame counts (`w_ceil`) and the total frame length `y_len`.
///
/// `logw`/`x_mask` are length `t_x` (the `[1,1,T]` graph outputs, leading axes
/// dropped). `length_scale > 1` slows speech (more frames), `< 1` speeds it up.
pub fn durations(logw: &[f32], x_mask: &[f32], length_scale: f32) -> (Vec<i64>, usize) {
    let w_ceil: Vec<i64> = logw
        .iter()
        .zip(x_mask)
        .map(|(&lw, &m)| {
            let w = lw.exp() * m * length_scale;
            w.ceil().max(0.0) as i64
        })
        .collect();
    let total: i64 = w_ceil.iter().sum();
    let y_len = total.max(1) as usize;
    (w_ceil, y_len)
}

/// Hard monotonic alignment `attn[t_y, t_x]`: each phone `i` lights up the frame
/// span `[start_i, start_i + w_ceil[i])`. Mirrors `_compute_alignment_path_np`.
pub fn alignment_path(w_ceil: &[i64], t_y: usize) -> Vec<f32> {
    let t_x = w_ceil.len();
    let mut attn = vec![0.0f32; t_y * t_x];
    let mut start: usize = 0;
    for (tx, &dur) in w_ceil.iter().enumerate() {
        let dur = dur.max(0) as usize;
        let end = (start + dur).min(t_y);
        for ty in start..end {
            attn[ty * t_x + tx] = 1.0;
        }
        start = end;
        if start >= t_y {
            break;
        }
    }
    attn
}

/// Expand a prior statistic from phones to frames: `out[c, ty] = Σ_tx attn[ty,tx] * stat[c, tx]`.
/// `stat` is `[c, t_x]` row-major, output is `[c, t_y]` row-major.
pub fn expand_prior(attn: &[f32], stat: &[f32], c: usize, t_x: usize, t_y: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * t_y];
    for ci in 0..c {
        let srow = &stat[ci * t_x..ci * t_x + t_x];
        let orow = &mut out[ci * t_y..ci * t_y + t_y];
        for (ty, o) in orow.iter_mut().enumerate() {
            let arow = &attn[ty * t_x..ty * t_x + t_x];
            let mut acc = 0.0f32;
            for tx in 0..t_x {
                let a = arow[tx];
                if a != 0.0 {
                    acc += a * srow[tx];
                }
            }
            *o = acc;
        }
    }
    out
}

/// Sample the flow input `z_p = m_p + N(0,1) * exp(logs_p) * noise_scale`.
/// All inputs/outputs are `[c, t_y]` row-major.
pub fn sample_z_p(m_exp: &[f32], logs_exp: &[f32], noise_scale: f32, rng: &mut Rng) -> Vec<f32> {
    m_exp
        .iter()
        .zip(logs_exp)
        .map(|(&m, &ls)| m + rng.randn() * ls.exp() * noise_scale)
        .collect()
}
