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

//! DIF (decimation-in-frequency) Cooley–Tukey FFT — natural input, bit-reverse output.

use crate::butterfly::{
    ParamSlot, bit_reverse, bit_reverse_permute, num_stages, packed_twiddle_param,
    packed_twiddle_scalar,
};
use crate::config::FftLearnConfig;
use crate::twiddle::{TwiddleSet, exact_twiddles, twiddle_index};
use anyhow::{Result, ensure};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};
use std::collections::HashMap;

#[inline]
fn cmul(re_a: f32, im_a: f32, re_w: f32, im_w: f32) -> (f32, f32) {
    (re_a * re_w - im_a * im_w, re_a * im_w + im_a * re_w)
}

/// One DIF stage: groups of size `m = 2^(stage+1)`, twiddle `W_m^j` on the upper half.
pub(crate) fn apply_dif_stage(
    buf: &[f32],
    next: &mut [f32],
    twiddles: &[f32],
    n_fft: usize,
    stage: usize,
) {
    let m = 2usize << stage;
    let half = n_fft / 2;
    for k in (0..n_fft).step_by(m) {
        for j in 0..(m / 2) {
            let w_base = twiddle_index(stage, j, half, 0);
            let w_re = twiddles[w_base];
            let w_im = twiddles[w_base + 1];
            let i0 = (k + j) * 2;
            let i1 = (k + j + m / 2) * 2;
            let a_re = buf[i0];
            let a_im = buf[i0 + 1];
            let (u_re, u_im) = cmul(buf[i1], buf[i1 + 1], w_re, w_im);
            next[i0] = a_re + u_re;
            next[i0 + 1] = a_im + u_im;
            next[i1] = a_re - u_re;
            next[i1 + 1] = a_im - u_im;
        }
    }
}

/// Forward FFT on interleaved complex `[n_fft, 2]` (bit-reverse input, DIF stages, natural output).
pub fn stockham_forward_eager(input: &[f32], twiddles: &[f32], n_fft: usize) -> Result<Vec<f32>> {
    ensure!(input.len() == n_fft * 2);
    let stages = num_stages(n_fft);
    let half = n_fft / 2;
    ensure!(twiddles.len() >= stages * half * 2);
    let mut buf = input.to_vec();
    bit_reverse_permute(&mut buf, n_fft);
    for s in 0..stages {
        let mut next = vec![0f32; n_fft * 2];
        apply_dif_stage(&buf, &mut next, twiddles, n_fft, s);
        buf = next;
    }
    Ok(buf)
}

pub fn stockham_forward_real_batch(
    signal: &[f32],
    twiddles: &[f32],
    batch: usize,
    n_fft: usize,
) -> Result<Vec<f32>> {
    ensure!(signal.len() == batch * n_fft);
    let mut out = vec![0f32; batch * n_fft * 2];
    for b in 0..batch {
        let mut state = vec![0f32; n_fft * 2];
        for i in 0..n_fft {
            state[i * 2] = signal[b * n_fft + i];
        }
        let spec = stockham_forward_eager(&state, twiddles, n_fft)?;
        out[b * n_fft * 2..(b + 1) * n_fft * 2].copy_from_slice(&spec);
    }
    Ok(out)
}

fn append_dif_butterfly(
    g: &mut Graph,
    cfg: &FftLearnConfig,
    mut state: NodeId,
    tw_set: TwiddleSet,
) -> Result<(Vec<ParamSlot>, NodeId)> {
    let n = cfg.n_fft;
    let half = n / 2;
    let stages = cfg.num_stages();

    let bits = cfg.num_stages();
    let mut reordered = Vec::with_capacity(n);
    for i in 0..n {
        let j = bit_reverse(i, bits);
        reordered.push(g.narrow_(state, 1, j, 1));
    }
    state = g.concat_(reordered, 1);

    // Single packed twiddle param; per-(stage,butterfly) scalars are narrowed out
    // of it (DIF accesses individual twiddles). Unused slices are removed by DCE.
    let (packed, slot) = packed_twiddle_param(g, tw_set, stages, half);
    let params = vec![slot];
    let mut twiddle_nodes: HashMap<(usize, usize), (NodeId, NodeId)> = HashMap::new();
    for s in 0..stages {
        for b in 0..half {
            twiddle_nodes.insert((s, b), packed_twiddle_scalar(g, packed, s, b));
        }
    }

    for s in 0..cfg.num_stages() {
        let m = 2usize << s;
        let mut next_parts = Vec::with_capacity(n);
        for i in 0..n {
            let pos = i % m;
            if pos >= m / 2 {
                continue;
            }
            let partner = i + m / 2;
            let (w_re, w_im) = twiddle_nodes[&(s, pos)];

            let a = g.narrow_(state, 1, i, 1);
            let b_node = g.narrow_(state, 1, partner, 1);
            let a_re = g.narrow_(a, 2, 0, 1);
            let a_im = g.narrow_(a, 2, 1, 1);
            let b_re = g.narrow_(b_node, 2, 0, 1);
            let b_im = g.narrow_(b_node, 2, 1, 1);

            let wb_re_a = g.mul(b_re, w_re);
            let wb_re_b = g.mul(b_im, w_im);
            let wb_re = g.sub(wb_re_a, wb_re_b);
            let wb_im_a = g.mul(b_re, w_im);
            let wb_im_b = g.mul(b_im, w_re);
            let wb_im = g.add(wb_im_a, wb_im_b);

            let top_re = g.add(a_re, wb_re);
            let top_im = g.add(a_im, wb_im);
            let bot_re = g.sub(a_re, wb_re);
            let bot_im = g.sub(a_im, wb_im);

            let top = g.concat_(vec![top_re, top_im], 2);
            let bot = g.concat_(vec![bot_re, bot_im], 2);
            next_parts.push(top);
            next_parts.push(bot);
        }
        state = g.concat_(next_parts, 1);
    }

    Ok((params, state))
}

pub fn build_stockham_forward_graph(cfg: &FftLearnConfig) -> Result<(Graph, Vec<String>)> {
    let n = cfg.n_fft;
    let batch = cfg.batch;
    let f = DType::F32;
    let mut g = Graph::new("fft_stockham_dif");
    let signal_in = g.input("signal", Shape::new(&[batch, n], f));
    let zeros = g.sub(signal_in, signal_in);
    let re = g.reshape_(signal_in, vec![batch as i64, n as i64, 1]);
    let im = g.reshape_(zeros, vec![batch as i64, n as i64, 1]);
    let state = g.concat_(vec![re, im], 2);
    let (params, state) = append_dif_butterfly(&mut g, cfg, state, TwiddleSet::Shared)?;
    g.set_outputs(vec![state]);
    let mut names = Vec::new();
    for p in &params {
        names.push(p.name.clone());
    }
    Ok((g, names))
}

pub fn exact_twiddles_stockham(cfg: &FftLearnConfig) -> Vec<f32> {
    exact_twiddles(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterfly::butterfly_forward_eager;
    use crate::config::FftLearnConfig;
    use crate::reference::{fft_real_batch, max_abs_error};

    #[test]
    fn dif_matches_butterfly_and_ref() {
        for &n in &[4usize, 8, 16, 64, 256] {
            let cfg = FftLearnConfig::new(n, 1).unwrap();
            let tw = exact_twiddles(&cfg);
            let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.17).sin()).collect();
            let mut state = vec![0f32; n * 2];
            for i in 0..n {
                state[i * 2] = signal[i];
            }
            let b = butterfly_forward_eager(&state, &tw, n).unwrap();
            let ref_fft = fft_real_batch(&signal, 1, n).unwrap();
            let s = stockham_forward_eager(&state, &tw, n).unwrap();
            assert!(max_abs_error(&s, &ref_fft) < 1e-3, "n={n} stockham vs ref");
            assert!(max_abs_error(&s, &b) < 1e-3, "n={n} stockham vs butterfly");
        }
    }
}
