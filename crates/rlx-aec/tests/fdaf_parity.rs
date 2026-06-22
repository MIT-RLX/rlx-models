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

//! Compare rlx-aec linear FDAF vs `fdaf-aec` reference crate.

use fdaf_aec::FdafAec;
use rlx_aec::{FdafConfig, FdafNlms, apply_echo, correlation, mse_improvement_db};

fn synth_buffers(n: usize, delay: usize, alpha: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut clean = Vec::with_capacity(n);
    let mut far = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / 16_000.0;
        clean.push(0.35 * (2.0 * std::f32::consts::PI * 300.0 * t).sin());
        far.push(0.45 * (2.0 * std::f32::consts::PI * 500.0 * t).sin());
    }
    let mic = apply_echo(&clean, &far, delay, alpha);
    (clean, far, mic)
}

#[test]
fn rlx_fdaf_vs_fdaf_aec_reference() {
    const FFT: usize = 1024;
    const HOP: usize = FFT / 2;
    let n = 16_000;
    let delay = 180;
    let alpha = 0.6;
    let (clean, far, mic) = synth_buffers(n, delay, alpha);

    let mut aligned = vec![0.0f32; far.len() + delay];
    aligned[delay..delay + far.len()].copy_from_slice(&far);
    let aligned = aligned[..mic.len()].to_vec();

    let cfg = FdafConfig {
        n_fft: FFT,
        frame_samples: HOP,
        step_size: 0.05,
        adapt: true,
        use_residual: false,
    };
    let mut rlx = FdafNlms::new(cfg, None).expect("rlx fdaf");
    let mut rlx_out = vec![0.0f32; mic.len()];
    rlx.process_buffer(&mic, &aligned, &mut rlx_out)
        .expect("rlx");

    let mut ref_aec = FdafAec::new(FFT, 0.05);
    let mut ref_out = Vec::new();
    for pos in (0..mic.len()).step_by(HOP) {
        let end = (pos + HOP).min(mic.len());
        let mut mf = vec![0.0f32; HOP];
        let mut ff = vec![0.0f32; HOP];
        mf[..end - pos].copy_from_slice(&mic[pos..end]);
        ff[..end - pos].copy_from_slice(&aligned[pos..end]);
        let chunk = ref_aec.process(&ff, &mf);
        ref_out.extend_from_slice(&chunk[..end - pos]);
    }

    let rlx_improve = mse_improvement_db(&mic, &rlx_out, &clean);
    let ref_improve = mse_improvement_db(&mic, &ref_out, &clean);
    let rlx_corr = correlation(&clean, &rlx_out);
    let _ref_corr = correlation(&clean, &ref_out);

    assert!(rlx_improve > 3.0, "rlx MSE improve {rlx_improve} dB");
    assert!(rlx_corr > 0.5, "rlx corr {rlx_corr}");
    // Reference may diverge on this synthetic layout; rlx must beat mic baseline.
    assert!(
        rlx_improve >= ref_improve - 5.0,
        "rlx {rlx_improve} dB vs ref {ref_improve} dB"
    );
}
