//! WaveRNN GRU cell (fused sigmoid / tanh-via-sigmoid / nested fmaf).
//!
//! Product free-run uses mode-0: `sigmoid`/`tanh=2*sigmoid(2x)-1` via `vvexpf`,
//! nested `fmaf` mix. Set `RLX_WR_GRU_PLAIN=1` for plain `vvtanhf` (debug only).

use crate::ops::f16_act;
use rlx_cpu::vmath::{vvexpf, vvrecf, vvtanhf};

const SUB: usize = 224;

/// Plain GRU step (no f16 modes).
pub fn eval(prev: &[f32], rg: &[f32], ig: &[f32], out: &mut [f32]) -> bool {
    eval_mode(prev, rg, ig, out, 0, false, false, false)
}

/// GRU step with optional f16 quantize modes matching `gru_sub_into_mode`.
pub fn eval_mode(
    prev: &[f32],
    rg: &[f32],
    ig: &[f32],
    out: &mut [f32],
    mode: u32,
    n_pre: bool,
    pre: bool,
    prev_f16: bool,
) -> bool {
    if prev.len() != SUB || out.len() != SUB || rg.len() != 3 * SUB || ig.len() != 3 * SUB {
        return false;
    }

    // Product path: fused activations + nested fmaf.
    if std::env::var_os("RLX_WR_GRU_PLAIN").is_none() {
        return eval_with_acts(
            prev,
            rg,
            ig,
            out,
            mode,
            n_pre,
            pre,
            prev_f16,
            sigmoid_vforce,
            tanh_fused,
            true,
        );
    }

    eval_with_acts(
        prev,
        rg,
        ig,
        out,
        mode,
        n_pre,
        pre,
        prev_f16,
        sigmoid_vmath,
        tanh_vmath,
        false,
    )
}

fn sigmoid_vmath(z: &mut [f32]) -> bool {
    let mut tmp = [0.0f32; SUB];
    for j in 0..SUB {
        tmp[j] = -z[j];
    }
    vvexpf(z, &tmp);
    for j in 0..SUB {
        z[j] += 1.0;
    }
    let mut out = [0.0f32; SUB];
    vvrecf(&mut out, z);
    z.copy_from_slice(&out);
    true
}

fn tanh_vmath(n: &mut [f32]) -> bool {
    let mut out = [0.0f32; SUB];
    vvtanhf(&mut out, n);
    n.copy_from_slice(&out);
    true
}

/// vForce-style sigmoid `1/(1+exp(-x))` via `vvexpf`.
fn sigmoid_vforce(z: &mut [f32]) -> bool {
    sigmoid_vmath(z)
}

/// Fused tanh: `2 * sigmoid(2x) - 1`.
fn tanh_fused(n: &mut [f32]) -> bool {
    let mut t = [0.0f32; SUB];
    for j in 0..SUB {
        t[j] = -(n[j] + n[j]);
    }
    let mut e = [0.0f32; SUB];
    vvexpf(&mut e, &t);
    for j in 0..SUB {
        let s = 1.0 / (1.0 + e[j]);
        n[j] = 2.0f32.mul_add(s, -1.0);
    }
    true
}

fn eval_with_acts(
    prev: &[f32],
    rg: &[f32],
    ig: &[f32],
    out: &mut [f32],
    mode: u32,
    n_pre: bool,
    pre: bool,
    prev_f16: bool,
    sigmoid: impl Fn(&mut [f32]) -> bool,
    tanh: impl Fn(&mut [f32]) -> bool,
    nested_fmaf: bool,
) -> bool {
    let mut z = [0.0f32; SUB];
    let mut r = [0.0f32; SUB];
    for j in 0..SUB {
        let mut zi = ig[j] + rg[j];
        let mut ri = ig[SUB + j] + rg[SUB + j];
        if pre {
            zi = f16_act(zi);
            ri = f16_act(ri);
        }
        z[j] = zi;
        r[j] = ri;
    }
    if !sigmoid(&mut z) || !sigmoid(&mut r) {
        return false;
    }
    if mode & 0b0001 != 0 {
        for v in z.iter_mut() {
            *v = f16_act(*v);
        }
    }
    if mode & 0b0010 != 0 {
        for v in r.iter_mut() {
            *v = f16_act(*v);
        }
    }

    let mut n = [0.0f32; SUB];
    for j in 0..SUB {
        let mut v = r[j].mul_add(rg[2 * SUB + j], ig[2 * SUB + j]);
        if pre || n_pre {
            v = f16_act(v);
        }
        n[j] = v;
    }
    if !tanh(&mut n) {
        return false;
    }
    if mode & 0b0100 != 0 {
        for v in n.iter_mut() {
            *v = f16_act(*v);
        }
    }

    for j in 0..SUB {
        let prev_j = if prev_f16 { f16_act(prev[j]) } else { prev[j] };
        let mut y = if nested_fmaf {
            z[j].mul_add(prev_j, (-z[j]).mul_add(n[j], n[j]))
        } else {
            z[j] * prev_j + (1.0 - z[j]) * n[j]
        };
        if mode & 0b1000 != 0 {
            y = f16_act(y);
        }
        out[j] = y;
    }
    true
}
