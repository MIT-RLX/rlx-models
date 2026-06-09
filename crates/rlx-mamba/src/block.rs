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

//! A single Mamba1 block: `in_proj → causal conv1d → SiLU → SSM → SiLU
//! gate → out_proj`. The math mirrors `burn_mamba::mamba1::Mamba1` —
//! using the same sequential selective scan (Algorithm 2 from the
//! paper) — but executes on flat `Vec<f32>` buffers via
//! [`rlx_cpu::blas`].
//!
//! Weight layout note: all linears here store weight as `[in, out]`
//! row-major, since [`rlx_tensor::linear`] is written as `y = x @ W`
//! (no internal transpose). The HF/PyTorch convention is `[out, in]`,
//! so a loader is expected to transpose at load time.

use crate::cache::Mamba1Cache;
use crate::config::Mamba1Config;
use crate::scan::{selective_scan_flow, selective_scan_step_flow};
use anyhow::{Result, ensure};
use rlx_cpu::blas;

/// `out[m, n] = a[m, k] @ b[k, n] + bias[n]` (bias broadcast per row).
/// We avoid `blas::sgemm_bias` directly because its small-shape NEON
/// dispatch (`neon_sgemm_bias_small`) has an `m ≤ 8` accumulator limit
/// that panics on tiny parity-test configs. Plain `sgemm` goes straight
/// to the BLAS, then we add bias here.
fn sgemm_bias_safe(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    blas::sgemm(a, b, out, m, k, n);
    for r in 0..m {
        let row = &mut out[r * n..(r + 1) * n];
        for c in 0..n {
            row[c] += bias[c];
        }
    }
}

/// Mamba1 block weights, all f32 row-major.
#[derive(Debug, Clone)]
pub struct Mamba1Block {
    pub cfg: Mamba1Config,

    /// `[d_model, 2 * d_inner]`.
    pub in_proj_w: Vec<f32>,
    /// `[2 * d_inner]` — zero if `cfg.bias == false`.
    pub in_proj_b: Vec<f32>,

    /// `[d_inner, d_conv]` — depthwise (groups = d_inner), so weight is
    /// one filter per channel.
    pub conv1d_w: Vec<f32>,
    /// `[d_inner]` — zero if `cfg.conv_bias == false`.
    pub conv1d_b: Vec<f32>,

    /// `[d_inner, dt_rank + 2 * d_state]`.
    pub x_proj_w: Vec<f32>,

    /// `[dt_rank, d_inner]`.
    pub dt_proj_w: Vec<f32>,
    /// `[d_inner]`.
    pub dt_proj_b: Vec<f32>,

    /// `[d_inner, d_state]`.
    pub a_log: Vec<f32>,
    /// `[d_inner]`.
    pub d: Vec<f32>,

    /// `[d_inner, d_model]`.
    pub out_proj_w: Vec<f32>,
    /// `[d_model]` — zero if `cfg.bias == false`.
    pub out_proj_b: Vec<f32>,
}

impl Mamba1Block {
    /// Stitch a block together from its raw weight buffers. Shapes must
    /// match the documented contract above.
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        cfg: Mamba1Config,
        in_proj_w: Vec<f32>,
        in_proj_b: Vec<f32>,
        conv1d_w: Vec<f32>,
        conv1d_b: Vec<f32>,
        x_proj_w: Vec<f32>,
        dt_proj_w: Vec<f32>,
        dt_proj_b: Vec<f32>,
        a_log: Vec<f32>,
        d: Vec<f32>,
        out_proj_w: Vec<f32>,
        out_proj_b: Vec<f32>,
    ) -> Result<Self> {
        let h = cfg.d_inner();
        let dr = cfg.dt_rank();
        let n = cfg.d_state;
        let m = cfg.d_model;
        let k = cfg.d_conv;

        ensure!(in_proj_w.len() == m * 2 * h, "in_proj_w shape");
        ensure!(in_proj_b.len() == 2 * h, "in_proj_b shape");
        ensure!(conv1d_w.len() == h * k, "conv1d_w shape");
        ensure!(conv1d_b.len() == h, "conv1d_b shape");
        ensure!(x_proj_w.len() == h * (dr + 2 * n), "x_proj_w shape");
        ensure!(dt_proj_w.len() == dr * h, "dt_proj_w shape");
        ensure!(dt_proj_b.len() == h, "dt_proj_b shape");
        ensure!(a_log.len() == h * n, "a_log shape");
        ensure!(d.len() == h, "d shape");
        ensure!(out_proj_w.len() == h * m, "out_proj_w shape");
        ensure!(out_proj_b.len() == m, "out_proj_b shape");

        Ok(Self {
            cfg,
            in_proj_w,
            in_proj_b,
            conv1d_w,
            conv1d_b,
            x_proj_w,
            dt_proj_w,
            dt_proj_b,
            a_log,
            d,
            out_proj_w,
            out_proj_b,
        })
    }

    /// Toy initializer (`a_log = log(arange(1..=d_state))` per channel,
    /// other weights small uniform pseudo-random). Useful for tests
    /// and benchmarks where the absolute output value doesn't matter —
    /// not a checkpoint loader.
    pub fn random_for_bench(cfg: Mamba1Config, seed: u64) -> Self {
        let m = cfg.d_model;
        let h = cfg.d_inner();
        let dr = cfg.dt_rank();
        let n = cfg.d_state;
        let k = cfg.d_conv;
        let mut rng = SplitMix64(seed);

        let in_proj_w = rng.uniform_vec(m * 2 * h, 1.0 / (m as f32).sqrt());
        let in_proj_b = vec![0.0; 2 * h];
        let conv1d_w = rng.uniform_vec(h * k, 1.0 / (k as f32).sqrt());
        let conv1d_b = rng.uniform_vec(h, 1.0 / (k as f32).sqrt());
        let x_proj_w = rng.uniform_vec(h * (dr + 2 * n), 1.0 / (h as f32).sqrt());
        let dt_proj_w = rng.uniform_vec(dr * h, 1.0 / (dr as f32).sqrt());
        let dt_proj_b = rng.uniform_vec(h, 1.0 / (dr as f32).sqrt());
        // a_log[h, n] = log(arange(1..=n)) broadcast across h, then small noise.
        let mut a_log = vec![0.0; h * n];
        for hi in 0..h {
            for ni in 0..n {
                a_log[hi * n + ni] = ((ni + 1) as f32).ln();
            }
        }
        let d = vec![1.0; h];
        let out_proj_w = rng.uniform_vec(h * m, 1.0 / (h as f32).sqrt());
        let out_proj_b = vec![0.0; m];

        Self {
            cfg,
            in_proj_w,
            in_proj_b,
            conv1d_w,
            conv1d_b,
            x_proj_w,
            dt_proj_w,
            dt_proj_b,
            a_log,
            d,
            out_proj_w,
            out_proj_b,
        }
    }

    /// Parallel/prefill path.
    /// `x` is `[batch, seq, d_model]`, output `[batch, seq, d_model]`.
    pub fn forward(&self, x: &[f32], batch: usize, seq: usize) -> Result<Vec<f32>> {
        let m = self.cfg.d_model;
        let h = self.cfg.d_inner();
        let n = self.cfg.d_state;
        let dr = self.cfg.dt_rank();
        let k = self.cfg.d_conv;
        ensure!(x.len() == batch * seq * m, "forward input shape");

        // in_proj: [batch*seq, d_model] @ [d_model, 2*d_inner] -> [batch*seq, 2*d_inner]
        let bs = batch * seq;
        let mut xz = vec![0.0; bs * 2 * h];
        sgemm_bias_safe(x, &self.in_proj_w, &self.in_proj_b, &mut xz, bs, m, 2 * h);

        // Split into xs and res along last axis. Store as separate buffers in
        // layout [batch, seq, d_inner] (row-major over (batch*seq, d_inner)).
        let mut xs = vec![0.0; bs * h];
        let mut res = vec![0.0; bs * h];
        for r in 0..bs {
            let src = &xz[r * 2 * h..(r + 1) * 2 * h];
            xs[r * h..(r + 1) * h].copy_from_slice(&src[..h]);
            res[r * h..(r + 1) * h].copy_from_slice(&src[h..]);
        }

        // Causal conv1d (depthwise, groups=d_inner, kernel=d_conv) over the
        // sequence axis of `xs`. Reshape mentally: [batch, seq, h] viewed as
        // per-channel sequences, output same shape; output[b, t, c] =
        // sum_{i=0..k} w[c, i] * xs[b, t - (k-1) + i, c] (zeros outside).
        // Result is then SiLU-activated.
        let mut conv_out = vec![0.0; bs * h];
        for b in 0..batch {
            for t in 0..seq {
                for c in 0..h {
                    let mut acc = self.conv1d_b[c];
                    for i in 0..k {
                        let src_t = t as isize - (k as isize - 1) + i as isize;
                        if src_t >= 0 && (src_t as usize) < seq {
                            let v = xs[b * seq * h + (src_t as usize) * h + c];
                            acc += self.conv1d_w[c * k + i] * v;
                        }
                    }
                    conv_out[b * seq * h + t * h + c] = silu(acc);
                }
            }
        }

        // SSM. `x_proj`: project conv_out [bs, h] -> [bs, dr + 2n].
        let dn = dr + 2 * n;
        let mut x_dbl = vec![0.0; bs * dn];
        blas::sgemm(&conv_out, &self.x_proj_w, &mut x_dbl, bs, h, dn);

        // Split into delta_raw [bs, dr], b_mat [bs, n], c_mat [bs, n].
        // Then delta = softplus(delta_raw @ dt_proj_w + dt_proj_b)  [bs, h].
        let mut delta_raw = vec![0.0; bs * dr];
        let mut b_mat = vec![0.0; bs * n];
        let mut c_mat = vec![0.0; bs * n];
        for r in 0..bs {
            let row = &x_dbl[r * dn..(r + 1) * dn];
            delta_raw[r * dr..(r + 1) * dr].copy_from_slice(&row[..dr]);
            b_mat[r * n..(r + 1) * n].copy_from_slice(&row[dr..dr + n]);
            c_mat[r * n..(r + 1) * n].copy_from_slice(&row[dr + n..]);
        }

        let mut dt_pre_softplus = vec![0.0; bs * h];
        sgemm_bias_safe(
            &delta_raw,
            &self.dt_proj_w,
            &self.dt_proj_b,
            &mut dt_pre_softplus,
            bs,
            dr,
            h,
        );

        let mut y = selective_scan_flow(
            batch,
            seq,
            h,
            n,
            &conv_out,
            &dt_pre_softplus,
            &b_mat,
            &c_mat,
            &self.a_log,
            &self.d,
        )?;

        // Gate: y *= silu(res)
        for r in 0..bs {
            for c in 0..h {
                y[r * h + c] *= silu(res[r * h + c]);
            }
        }

        // out_proj: [bs, h] @ [h, d_model] -> [bs, d_model]
        let mut out = vec![0.0; bs * m];
        sgemm_bias_safe(&y, &self.out_proj_w, &self.out_proj_b, &mut out, bs, h, m);
        Ok(out)
    }

    /// Decode-step path. `x` is `[batch, d_model]`. Output: `[batch, d_model]`.
    /// Updates `cache` in place.
    pub fn step(&self, x: &[f32], batch: usize, cache: &mut Mamba1Cache) -> Result<Vec<f32>> {
        let m = self.cfg.d_model;
        let h = self.cfg.d_inner();
        let n = self.cfg.d_state;
        let dr = self.cfg.dt_rank();
        let k = self.cfg.d_conv;
        ensure!(x.len() == batch * m, "step input shape");
        ensure!(cache.batch == batch, "cache batch mismatch");
        ensure!(
            cache.d_inner == h && cache.d_conv == k && cache.d_state == n,
            "cache shape"
        );

        // in_proj
        let mut xz = vec![0.0; batch * 2 * h];
        sgemm_bias_safe(
            x,
            &self.in_proj_w,
            &self.in_proj_b,
            &mut xz,
            batch,
            m,
            2 * h,
        );

        let mut xs = vec![0.0; batch * h];
        let mut res = vec![0.0; batch * h];
        for b in 0..batch {
            let src = &xz[b * 2 * h..(b + 1) * 2 * h];
            xs[b * h..(b + 1) * h].copy_from_slice(&src[..h]);
            res[b * h..(b + 1) * h].copy_from_slice(&src[h..]);
        }

        // Roll conv cache leftward and append xs at position k-1.
        // cache[b, c, :] currently holds [x_{t-k+1}, ..., x_{t-1}, ?]; we
        // shift left to drop the oldest, then write xs at the last slot.
        for b in 0..batch {
            for c in 0..h {
                let base = b * h * k + c * k;
                for i in 0..k - 1 {
                    cache.conv[base + i] = cache.conv[base + i + 1];
                }
                cache.conv[base + k - 1] = xs[b * h + c];
            }
        }

        // Depthwise conv: out[b, c] = sum_i w[c, i] * cache[b, c, i] + bias[c]
        let mut conv_out = vec![0.0; batch * h];
        for b in 0..batch {
            for c in 0..h {
                let base = b * h * k + c * k;
                let mut acc = self.conv1d_b[c];
                for i in 0..k {
                    acc += self.conv1d_w[c * k + i] * cache.conv[base + i];
                }
                conv_out[b * h + c] = silu(acc);
            }
        }

        // x_proj split
        let dn = dr + 2 * n;
        let mut x_dbl = vec![0.0; batch * dn];
        blas::sgemm(&conv_out, &self.x_proj_w, &mut x_dbl, batch, h, dn);

        let mut delta_raw = vec![0.0; batch * dr];
        let mut b_mat = vec![0.0; batch * n];
        let mut c_mat = vec![0.0; batch * n];
        for b in 0..batch {
            let row = &x_dbl[b * dn..(b + 1) * dn];
            delta_raw[b * dr..(b + 1) * dr].copy_from_slice(&row[..dr]);
            b_mat[b * n..(b + 1) * n].copy_from_slice(&row[dr..dr + n]);
            c_mat[b * n..(b + 1) * n].copy_from_slice(&row[dr + n..]);
        }

        let mut dt_pre_softplus = vec![0.0; batch * h];
        sgemm_bias_safe(
            &delta_raw,
            &self.dt_proj_w,
            &self.dt_proj_b,
            &mut dt_pre_softplus,
            batch,
            dr,
            h,
        );

        let mut y = selective_scan_step_flow(
            batch,
            h,
            n,
            &conv_out,
            &dt_pre_softplus,
            &b_mat,
            &c_mat,
            &mut cache.ssm,
            &self.a_log,
            &self.d,
        )?;

        // Gate with silu(res), out_proj.
        for r in 0..batch {
            for c in 0..h {
                y[r * h + c] *= silu(res[r * h + c]);
            }
        }
        let mut out = vec![0.0; batch * m];
        sgemm_bias_safe(
            &y,
            &self.out_proj_w,
            &self.out_proj_b,
            &mut out,
            batch,
            h,
            m,
        );
        Ok(out)
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Tiny LCG-style PRNG so the `random_for_bench` constructor doesn't pull
/// in `rand`. Output is uniform in `[-scale, scale]`.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_f32(&mut self, scale: f32) -> f32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        // Map to [-1, 1).
        let u = ((z >> 40) as f32) / ((1u32 << 24) as f32);
        (u * 2.0 - 1.0) * scale
    }
    fn uniform_vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.next_f32(scale)).collect()
    }
}
