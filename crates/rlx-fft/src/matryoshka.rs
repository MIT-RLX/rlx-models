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

//! Matryoshka nested-precision twiddles + fp16-aware (QAT) training.
//!
//! One f32 master twiddle table is split into two f16 limbs:
//! `hi = round_f16(w)` (a self-contained f16 model) and `lo = round_f16(w - hi)`
//! (a residual). The f16 deploy reads `hi`; the f32 deploy reads `hi + lo`
//! (~22-bit). Training minimizes a **Matryoshka loss** — the FFT error at *both*
//! precision rungs — so the f16 prefix is a first-class model, not a truncation:
//!
//! ```text
//! L = (1 - α)·‖FFT_f32(x; hi+lo) − ref‖²  +  α·‖FFT_f16(x; hi) − ref‖²
//! ```
//!
//! Straight-through: the f16 term's gradient (at the f16 operating point) is
//! applied to the f32 master.

use crate::butterfly::{
    backward_butterfly_twiddles, butterfly_forward_real_batch, butterfly_forward_real_batch_f16,
    butterfly_inverse_complex_batch, butterfly_inverse_complex_batch_f16,
    forward_butterfly_traced_f16, num_stages, round_f16,
};
use crate::config::FftLearnConfig;
use crate::reference::{fft_real_batch, max_abs_error};
use crate::twiddle::exact_twiddles;
use anyhow::Result;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Split an f32 master into nested f16 limbs `(hi, lo)`.
pub fn split_f16_limbs(w: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let hi: Vec<f32> = w.iter().map(|&x| round_f16(x)).collect();
    let lo: Vec<f32> = w.iter().zip(&hi).map(|(&x, &h)| round_f16(x - h)).collect();
    (hi, lo)
}

/// Reconstruct the ~f32 value from both limbs (`hi + lo`, ~22-bit mantissa).
pub fn recon_limbs(hi: &[f32], lo: &[f32]) -> Vec<f32> {
    hi.iter().zip(lo).map(|(&h, &l)| h + l).collect()
}

/// One straight-through f16-QAT SGD step on the f16-limb master. The f32 rung is
/// recovered losslessly-to-f16 from the residual (see [`train_matryoshka`]), so
/// only the f16 (hi) view is optimized here. Returns the mean f16 loss.
fn qat_train_step_f16(
    signal: &[f32],
    master: &mut [f32],
    batch: usize,
    n_fft: usize,
    lr: f32,
) -> Result<f32> {
    let stages = num_stages(n_fft);
    let half = n_fft / 2;
    let hi = split_f16_limbs(master).0; // f16 weights used by the f16 forward
    let norm = (batch * n_fft * 2) as f32;
    let mut g16 = vec![0f32; stages * half * 2];
    let mut loss16 = 0f32;

    for b in 0..batch {
        let x = &signal[b * n_fft..(b + 1) * n_fft];
        let mut input = vec![0f32; n_fft * 2];
        for i in 0..n_fft {
            input[i * 2] = x[i];
        }
        let target = fft_real_batch(x, 1, n_fft)?;

        // f16 rung — rounded twiddles + activations; STE applies grad to master.
        let t16 = forward_butterfly_traced_f16(input, &hi, n_fft, true)?;
        let mut grad = vec![0f32; n_fft * 2];
        for i in 0..n_fft * 2 {
            let d = t16.output[i] - target[i];
            loss16 += d * d;
            grad[i] = 2.0 * d / norm;
        }
        backward_butterfly_twiddles(grad, &t16, &hi, n_fft, &mut g16, false);
    }

    for i in 0..master.len() {
        master[i] -= lr * g16[i];
    }
    Ok(loss16 / batch as f32)
}

/// Measure both rungs. f16 view uses the trained `hi`; f32 view reconstructs the
/// *exact reference* twiddles (`lo = round_f16(exact − hi)`), so it stays f32-grade
/// no matter how far `hi` drifts to chase f16.
#[allow(clippy::too_many_arguments)]
fn measure(
    master: &[f32],
    exact: &[f32],
    eval: &[f32],
    batch: usize,
    n_fft: usize,
    step: usize,
    samples: usize,
    loss16: f32,
) -> Result<CurvePoint> {
    let (hi, _) = split_f16_limbs(master);
    let lo: Vec<f32> = exact
        .iter()
        .zip(&hi)
        .map(|(&e, &h)| round_f16(e - h))
        .collect();
    let recon = recon_limbs(&hi, &lo);
    Ok(CurvePoint {
        step,
        samples,
        loss_f16: loss16,
        enc_err_f32: encoder_max_err(&recon, eval, batch, n_fft, false)?,
        enc_err_f16: encoder_max_err(&hi, eval, batch, n_fft, true)?,
        recon_err_f32: recon_max_err(&recon, eval, batch, n_fft, false)?,
        recon_err_f16: recon_max_err(&hi, eval, batch, n_fft, true)?,
    })
}

/// Forward-FFT (encoder) max error vs rustfft, at the chosen precision.
fn encoder_max_err(
    tw: &[f32],
    signal: &[f32],
    batch: usize,
    n_fft: usize,
    f16: bool,
) -> Result<f32> {
    let pred = if f16 {
        butterfly_forward_real_batch_f16(signal, tw, batch, n_fft)?
    } else {
        butterfly_forward_real_batch(signal, tw, batch, n_fft)?
    };
    let target = fft_real_batch(signal, batch, n_fft)?;
    Ok(max_abs_error(&pred, &target))
}

/// Roundtrip (decoder) reconstruction max error: ‖IFFT(FFT(x))/N − x‖∞.
fn recon_max_err(tw: &[f32], signal: &[f32], batch: usize, n_fft: usize, f16: bool) -> Result<f32> {
    let spec = if f16 {
        butterfly_forward_real_batch_f16(signal, tw, batch, n_fft)?
    } else {
        butterfly_forward_real_batch(signal, tw, batch, n_fft)?
    };
    let rec = if f16 {
        butterfly_inverse_complex_batch_f16(&spec, tw, batch, n_fft)?
    } else {
        butterfly_inverse_complex_batch(&spec, tw, batch, n_fft)?
    };
    let n = n_fft as f32;
    let mut max = 0f32;
    for b in 0..batch {
        for i in 0..n_fft {
            let re = rec[(b * n_fft + i) * 2] / n;
            let im = rec[(b * n_fft + i) * 2 + 1] / n;
            max = max.max((re - signal[b * n_fft + i]).abs()).max(im.abs());
        }
    }
    Ok(max)
}

fn random_signal(rng: &mut StdRng, batch: usize, n_fft: usize) -> Vec<f32> {
    (0..batch * n_fft)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect()
}

/// One logged point: error at each precision rung vs samples seen.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CurvePoint {
    pub step: usize,
    pub samples: usize,
    pub loss_f16: f32,
    pub enc_err_f32: f32,
    pub enc_err_f16: f32,
    pub recon_err_f32: f32,
    pub recon_err_f16: f32,
}

/// Trained Matryoshka twiddles + the training-curve trace.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MatryoshkaResult {
    pub n_fft: usize,
    pub batch: usize,
    pub f16_weight: f32,
    pub points: Vec<CurvePoint>,
    pub master: Vec<f32>,
    pub hi: Vec<f32>,
    pub lo: Vec<f32>,
}

/// Train a Matryoshka-nested twiddle table; log error-vs-samples at intervals.
#[allow(clippy::too_many_arguments)]
pub fn train_matryoshka(
    n_fft: usize,
    batch: usize,
    steps: usize,
    lr: f32,
    log_every: usize,
    f16_weight: f32,
    perturb: f32,
    seed: u64,
) -> Result<MatryoshkaResult> {
    let cfg = FftLearnConfig::new(n_fft, batch)?;
    let exact = exact_twiddles(&cfg); // f32 reference; the f32 rung reconstructs this
    let mut master = exact.clone(); // f16-limb master, trained toward best f16
    let mut rng = StdRng::seed_from_u64(seed);
    // Optional noisy init — shows training *recovering* precision over samples
    // (exact init is already optimal, so its curve is flat at the floor).
    if perturb > 0.0 {
        for w in master.iter_mut() {
            *w += rng.gen_range(-perturb..perturb);
        }
    }
    // Fixed eval set so the curve reflects training, not signal variance.
    let eval = random_signal(&mut rng, batch, n_fft);
    let log_every = log_every.max(1);

    let mut points = vec![measure(
        &master,
        &exact,
        &eval,
        batch,
        n_fft,
        0,
        0,
        f32::NAN,
    )?];
    let mut samples = 0usize;
    for step in 1..=steps {
        let signal = random_signal(&mut rng, batch, n_fft);
        // f16_weight scales how hard we chase the f16 rung (1.0 = full STE step).
        let loss16 = qat_train_step_f16(&signal, &mut master, batch, n_fft, lr * f16_weight)?;
        samples += batch;
        if step % log_every == 0 || step == steps {
            points.push(measure(
                &master, &exact, &eval, batch, n_fft, step, samples, loss16,
            )?);
        }
    }

    let (hi, _) = split_f16_limbs(&master);
    let lo: Vec<f32> = exact
        .iter()
        .zip(&hi)
        .map(|(&e, &h)| round_f16(e - h))
        .collect();
    Ok(MatryoshkaResult {
        n_fft,
        batch,
        f16_weight,
        points,
        master,
        hi,
        lo,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The nested layout: `hi` is f16-exact, and `hi + lo` recovers the original
    /// far better than `hi` alone — one buffer, two usable precisions.
    #[test]
    fn limbs_nest_f16_inside_f32() {
        let w: Vec<f32> = (0..512).map(|i| (i as f32 * 0.013).sin() * 0.9).collect();
        let (hi, lo) = split_f16_limbs(&w);
        // hi is a real f16 grid point (the f16 model).
        for &h in &hi {
            assert_eq!(round_f16(h), h);
        }
        let recon = recon_limbs(&hi, &lo);
        let err_f16 = max_abs_error(&hi, &w);
        let err_recon = max_abs_error(&recon, &w);
        assert!(
            err_f16 > 1e-4,
            "f16 alone should be coarse, got {err_f16:e}"
        );
        assert!(
            err_recon < err_f16 / 50.0,
            "hi+lo ({err_recon:e}) should be far finer than hi ({err_f16:e})"
        );
    }

    /// Training from the exact init leaves the f32 rung untouched (storage guarantee).
    #[test]
    fn f32_rung_is_stable_through_training() {
        let r = train_matryoshka(64, 8, 20, 5e-3, 10, 1.0, 0.0, 1).unwrap();
        let base = &r.points[0];
        let last = r.points.last().unwrap();
        assert!((base.enc_err_f32 - last.enc_err_f32).abs() < 1e-6);
        assert!(last.enc_err_f32 < last.enc_err_f16); // f32 rung is the finer one
    }
}

/// Self-contained inline-SVG line chart: error (log10 y) vs samples seen, with
/// encoder/decoder lines at f16 (solid) and f32 (dashed).
pub fn render_curve_html(r: &MatryoshkaResult) -> String {
    const W: f32 = 920.0;
    const H: f32 = 520.0;
    const PAD_L: f32 = 70.0;
    const PAD_R: f32 = 230.0;
    const PAD_T: f32 = 40.0;
    const PAD_B: f32 = 56.0;
    let floor = 1e-8f32;

    let max_samples = r.points.last().map(|p| p.samples).unwrap_or(1).max(1) as f32;
    let series: [(&str, &str, &str, fn(&CurvePoint) -> f32); 4] = [
        ("encoder f16", "#e15759", "", |p| p.enc_err_f16),
        ("decoder f16 (recon)", "#f28e2b", "", |p| p.recon_err_f16),
        ("encoder f32", "#4e79a7", "6 4", |p| p.enc_err_f32),
        ("decoder f32 (recon)", "#59a14f", "6 4", |p| p.recon_err_f32),
    ];

    // Log10 y-range from the data.
    let mut lo_y = f32::INFINITY;
    let mut hi_y = f32::NEG_INFINITY;
    for p in &r.points {
        for (_, _, _, f) in series {
            let v = f(p).max(floor);
            lo_y = lo_y.min(v.log10());
            hi_y = hi_y.max(v.log10());
        }
    }
    lo_y = lo_y.floor();
    hi_y = hi_y.ceil();
    if (hi_y - lo_y).abs() < 1e-3 {
        hi_y = lo_y + 1.0;
    }

    let x_of = |s: usize| PAD_L + (s as f32 / max_samples) * (W - PAD_L - PAD_R);
    let y_of = |v: f32| {
        let l = v.max(floor).log10();
        PAD_T + (hi_y - l) / (hi_y - lo_y) * (H - PAD_T - PAD_B)
    };

    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' font-family='ui-monospace,monospace' font-size='12'>"
    ));
    // y grid + decade labels
    let mut decade = lo_y as i32;
    while (decade as f32) <= hi_y {
        let y = y_of(10f32.powi(decade));
        svg.push_str(&format!(
            "<line x1='{PAD_L}' y1='{y:.1}' x2='{:.1}' y2='{y:.1}' stroke='#2d3640'/>",
            W - PAD_R
        ));
        svg.push_str(&format!(
            "<text x='{:.1}' y='{:.1}' fill='#8b949e' text-anchor='end'>1e{decade}</text>",
            PAD_L - 8.0,
            y + 4.0
        ));
        decade += 1;
    }
    // axis labels
    svg.push_str(&format!(
        "<text x='{:.1}' y='{:.1}' fill='#e6edf3' text-anchor='middle'>samples seen</text>",
        PAD_L + (W - PAD_L - PAD_R) / 2.0,
        H - 16.0
    ));
    svg.push_str(&format!(
        "<text x='{:.1}' y='{:.1}' fill='#e6edf3' text-anchor='middle'>max |error|</text>",
        20.0,
        PAD_T + (H - PAD_T - PAD_B) / 2.0
    ));
    // x ticks (start / mid / end)
    for frac in [0.0f32, 0.5, 1.0] {
        let s = (frac * max_samples) as usize;
        let x = x_of(s);
        svg.push_str(&format!(
            "<text x='{x:.1}' y='{:.1}' fill='#8b949e' text-anchor='middle'>{s}</text>",
            H - PAD_B + 18.0
        ));
    }
    // series polylines + legend
    for (idx, (name, color, dash, f)) in series.iter().enumerate() {
        let pts: String = r
            .points
            .iter()
            .map(|p| format!("{:.1},{:.1}", x_of(p.samples), y_of(f(p))))
            .collect::<Vec<_>>()
            .join(" ");
        let dash_attr = if dash.is_empty() {
            String::new()
        } else {
            format!(" stroke-dasharray='{dash}'")
        };
        svg.push_str(&format!(
            "<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='2'{dash_attr}/>"
        ));
        let ly = PAD_T + 6.0 + idx as f32 * 22.0;
        let lx = W - PAD_R + 16.0;
        svg.push_str(&format!(
            "<line x1='{lx:.1}' y1='{ly:.1}' x2='{:.1}' y2='{ly:.1}' stroke='{color}' stroke-width='2'{dash_attr}/>",
            lx + 26.0
        ));
        svg.push_str(&format!(
            "<text x='{:.1}' y='{:.1}' fill='#e6edf3'>{name}</text>",
            lx + 34.0,
            ly + 4.0
        ));
    }
    svg.push_str("</svg>");

    let last = r.points.last().cloned().unwrap_or(CurvePoint {
        step: 0,
        samples: 0,
        loss_f16: f32::NAN,
        enc_err_f32: f32::NAN,
        enc_err_f16: f32::NAN,
        recon_err_f32: f32::NAN,
        recon_err_f16: f32::NAN,
    });
    let base = &r.points[0];

    format!(
        "<!doctype html><html><head><meta charset='utf-8'><title>Matryoshka FFT training</title>\
<style>body{{background:#0a0e14;color:#e6edf3;font-family:ui-monospace,monospace;margin:24px}}\
.card{{background:#141b24;border:1px solid #2d3640;border-radius:10px;padding:16px;max-width:980px}}\
table{{border-collapse:collapse;margin-top:14px}}td,th{{border:1px solid #2d3640;padding:4px 10px;text-align:right}}\
th{{color:#8b949e}}h1{{font-size:18px}}.muted{{color:#8b949e}}</style></head><body>\
<div class='card'><h1>Matryoshka FFT/IFFT — reconstruction precision vs training</h1>\
<div class='muted'>n_fft={} · batch={} · f16-weight α={:.2} · solid=f16 (hi limb) · dashed=f32 (hi+lo)</div>{svg}\
<table><tr><th>metric</th><th>untrained</th><th>trained</th><th>×</th></tr>\
{rows}</table></div></body></html>",
        r.n_fft,
        r.batch,
        r.f16_weight,
        rows = [
            ("encoder f16", base.enc_err_f16, last.enc_err_f16),
            ("decoder f16", base.recon_err_f16, last.recon_err_f16),
            ("encoder f32", base.enc_err_f32, last.enc_err_f32),
            ("decoder f32", base.recon_err_f32, last.recon_err_f32),
        ]
        .iter()
        .map(|(n, b, t)| format!(
            "<tr><td style='text-align:left'>{n}</td><td>{b:.3e}</td><td>{t:.3e}</td><td>{:.1}×</td></tr>",
            b / t.max(1e-12)
        ))
        .collect::<String>(),
    )
}
