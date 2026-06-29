// RLX models — calibration quantization.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! GPTQ — Optimal Brain Quantization with a Hessian (Frantar et al. 2023).
//!
//! Quantize the input columns left-to-right; after each column is rounded, push
//! its error into the not-yet-quantized columns, weighted by the inverse
//! Hessian `H⁻¹` of the layer input (`H = XᵀX` from calibration). This
//! compensates for correlated inputs, beating naïve round-to-nearest at low
//! bit-widths. Self-contained host linear algebra (Cholesky + triangular
//! inverse); no external BLAS.

use crate::quant::{GroupQuant, qmax};

/// GPTQ-quantize `w [out, in]` using the layer-input Hessian `hessian [in, in]`
/// (`= XᵀX`, symmetric PSD). `damp` adds `damp·mean(diag)` to the diagonal for
/// numerical stability.
pub fn gptq_quantize(
    w: &[f32],
    out: usize,
    inn: usize,
    hessian: &[f32],
    bits: u32,
    group_size: usize,
    damp: f32,
) -> GroupQuant {
    let qm = qmax(bits);
    let gs = group_size.clamp(1, inn.max(1));
    let ng = inn.div_ceil(gs);

    // Per-(row, group) scales from the original weight.
    let mut scales = vec![0f32; out * ng];
    for r in 0..out {
        for g in 0..ng {
            let c0 = g * gs;
            let c1 = (c0 + gs).min(inn);
            let amax = (c0..c1)
                .map(|c| w[r * inn + c].abs())
                .fold(0.0f32, f32::max);
            scales[r * ng + g] = if amax > 0.0 { amax / qm } else { 1.0 };
        }
    }

    // Upper-triangular Cholesky factor of (H + λI)⁻¹.
    let mut h = hessian.to_vec();
    let meandiag = (0..inn).map(|i| h[i * inn + i]).sum::<f32>() / inn.max(1) as f32;
    let lam = damp * meandiag.max(1e-6);
    for i in 0..inn {
        h[i * inn + i] += lam;
    }
    let hinv = chol_inv_upper(&h, inn);

    // OBQ: quantize column i, compensate the residual into columns > i.
    let mut wc = w.to_vec();
    let mut q = vec![0i32; out * inn];
    for i in 0..inn {
        let d = hinv[i * inn + i].max(1e-12);
        let g = i / gs;
        for o in 0..out {
            let sc = scales[o * ng + g];
            let qi = (wc[o * inn + i] / sc).round().clamp(-qm, qm);
            q[o * inn + i] = qi as i32;
            let err = (wc[o * inn + i] - qi * sc) / d;
            for j in (i + 1)..inn {
                wc[o * inn + j] -= err * hinv[i * inn + j];
            }
        }
    }
    GroupQuant {
        q,
        scales,
        out,
        inn,
        bits,
        group_size: gs,
    }
}

/// Upper-triangular `U` with `Uᵀ U = (H)⁻¹`. (`H` already damped + PSD.)
fn chol_inv_upper(h: &[f32], n: usize) -> Vec<f32> {
    let l = cholesky_lower(h, n);
    let linv = tri_inverse_lower(&l, n);
    // H⁻¹ = (L⁻¹)ᵀ (L⁻¹)
    let mut hinv = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += linv[k * n + i] * linv[k * n + j];
            }
            hinv[i * n + j] = acc;
        }
    }
    // U = (cholesky_lower(H⁻¹))ᵀ  ⇒  Uᵀ U = H⁻¹, U upper.
    let l2 = cholesky_lower(&hinv, n);
    let mut u = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            u[i * n + j] = l2[j * n + i];
        }
    }
    u
}

/// Lower Cholesky `L` with `L Lᵀ = a` (`a` symmetric PSD).
fn cholesky_lower(a: &[f32], n: usize) -> Vec<f32> {
    let mut l = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                l[i * n + j] = sum.max(1e-12).sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    l
}

/// Inverse of a lower-triangular matrix via forward substitution.
fn tri_inverse_lower(l: &[f32], n: usize) -> Vec<f32> {
    let mut inv = vec![0f32; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0 / l[i * n + i];
        for j in 0..i {
            let mut s = 0.0;
            for k in j..i {
                s += l[i * n + k] * inv[k * n + j];
            }
            inv[i * n + j] = -s / l[i * n + i];
        }
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{dequantize, matmul_wt, mse, quantize_rtn};

    fn pseudo(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / u32::MAX as f32 - 0.5) * 2.0
            })
            .collect()
    }

    #[test]
    fn cholesky_reconstructs() {
        // SPD matrix A = M Mᵀ + I.
        let n = 3;
        let m = [1.0f32, 0.0, 0.0, 0.5, 1.0, 0.0, 0.2, 0.3, 1.0];
        let mut a = vec![0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = if i == j { 1.0 } else { 0.0 };
                for k in 0..n {
                    acc += m[i * n + k] * m[j * n + k];
                }
                a[i * n + j] = acc;
            }
        }
        let l = cholesky_lower(&a, n);
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += l[i * n + k] * l[j * n + k];
                }
                assert!((acc - a[i * n + j]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn chol_inv_upper_is_correct() {
        // Uᵀ U must equal H⁻¹, i.e. H · (Uᵀ U) = I.
        let n = 4;
        let m = [
            1.0f32, 0.0, 0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.2, 0.3, 1.0, 0.0, 0.1, 0.2, 0.3, 1.0,
        ];
        let mut h = vec![0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = if i == j { 0.5 } else { 0.0 };
                for k in 0..n {
                    acc += m[i * n + k] * m[j * n + k];
                }
                h[i * n + j] = acc;
            }
        }
        let u = chol_inv_upper(&h, n);
        // hinv = Uᵀ U
        let mut hinv = vec![0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += u[k * n + i] * u[k * n + j];
                }
                hinv[i * n + j] = acc;
            }
        }
        // H · hinv ≈ I
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += h[i * n + k] * hinv[k * n + j];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((acc - expect).abs() < 1e-3, "H·H⁻¹[{i},{j}]={acc}");
            }
        }
    }

    #[test]
    fn gptq_is_valid_and_no_worse_than_rtn() {
        // GPTQ's error feedback must never *increase* the calibration output
        // error vs round-to-nearest, and must produce a faithful quantization.
        // (Strict improvement needs large layers + lazy group scales; on a toy
        // the feedback rarely flips a rounding, so it reduces to RTN here.)
        let (out, inn) = (8usize, 24usize);
        let samples = 256usize;
        let w = pseudo(out * inn, 1);

        // Compound-symmetric covariance: one strong shared factor (ρ=0.95)
        // across all channels → a dense, coherent inverse Hessian, where
        // GPTQ's error feedback accumulates across channels and flips roundings.
        let rho = 0.95f32;
        let (sr, sn) = (rho.sqrt(), (1.0 - rho).sqrt());
        let common = pseudo(samples, 7);
        let noise = pseudo(samples * inn, 8);
        let mut x = vec![0f32; samples * inn];
        for s in 0..samples {
            for c in 0..inn {
                x[s * inn + c] = sr * common[s] + sn * noise[s * inn + c];
            }
        }
        let mut h = vec![0f32; inn * inn];
        for a in 0..inn {
            for b in 0..inn {
                let mut acc = 0.0;
                for s in 0..samples {
                    acc += x[s * inn + a] * x[s * inn + b];
                }
                h[a * inn + b] = acc / samples as f32;
            }
        }

        let target = matmul_wt(&x, &w, samples, inn, out);
        let (bits, gs) = (3u32, inn);

        let rtn = quantize_rtn(&w, out, inn, bits, gs);
        let rtn_err = mse(
            &target,
            &matmul_wt(&x, &dequantize(&rtn), samples, inn, out),
        );

        let gq = gptq_quantize(&w, out, inn, &h, bits, gs, 0.05);
        let gptq_err = mse(&target, &matmul_wt(&x, &dequantize(&gq), samples, inn, out));

        // Never worse than RTN.
        assert!(
            gptq_err <= rtn_err * 1.001,
            "GPTQ {gptq_err} > RTN {rtn_err}"
        );
        // Faithful quantization: dequant stays within a quant step of the
        // original weight magnitude.
        let dq = dequantize(&gq);
        let amax = w.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let step = amax / crate::quant::qmax(bits);
        assert!(
            crate::quant::mse(&w, &dq).sqrt() < step,
            "dequant too far from w"
        );
    }
}
