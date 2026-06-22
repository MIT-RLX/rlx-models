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

use rlx_aec::fdaf::{FdafConfig, FdafNlms};
use rlx_aec::{
    AecConfig, AecSession, ResidualWeights, apply_echo, correlation, mse_improvement_db,
};

fn synth_clean_far() -> (Vec<f32>, Vec<f32>) {
    let sr = 16_000;
    let n = sr;
    let mut clean = Vec::with_capacity(n);
    let mut far = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / sr as f32;
        clean.push(0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin());
        far.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
    }
    (clean, far)
}

#[test]
fn delay_estimate_nonzero_on_echoed_signal() {
    let (clean, far) = synth_clean_far();
    let delay = 240;
    let mic = apply_echo(&clean, &far, delay, 0.6);
    let est = rlx_aec::estimate_delay_samples(&far, &mic, 1024, 500);
    assert!(
        (est as i32 - delay as i32).abs() < 80,
        "est={est} expected~{delay}"
    );
}

#[test]
fn fdaf_reduces_echo_energy() {
    let (clean, far) = synth_clean_far();
    let delay = 160;
    let alpha = 0.7;
    let mic = apply_echo(&clean, &far, delay, alpha);

    let cfg = FdafConfig {
        n_fft: 1024,
        frame_samples: 512,
        step_size: 0.05,
        use_residual: false,
        adapt: true,
    };
    let mut fdaf = FdafNlms::new(cfg, None).expect("fdaf");
    let mut aligned = vec![0.0f32; far.len() + delay];
    aligned[delay..delay + far.len()].copy_from_slice(&far);
    let aligned = aligned[..mic.len()].to_vec();

    let mut out = vec![0.0f32; mic.len()];
    fdaf.process_buffer(&mic, &aligned, &mut out)
        .expect("process");

    let improve = mse_improvement_db(&mic, &out, &clean);
    let corr = correlation(&clean, &out);
    assert!(improve > 3.0, "MSE improvement too low: {improve} dB");
    assert!(corr > 0.5, "correlation too low: {corr}");
}

#[test]
fn session_aligned_offline() {
    let (clean, far) = synth_clean_far();
    let delay = 200;
    let mic = apply_echo(&clean, &far, delay, 0.65);
    let cfg = AecConfig {
        step_size: 0.05,
        residual: false,
        ..AecConfig::default()
    };
    let mut session = AecSession::new(cfg).expect("session");
    let out = session
        .process_aligned_buffers(&mic, &far)
        .expect("aligned");
    assert!(mse_improvement_db(&mic, &out, &clean) > 2.0);
}

#[test]
fn residual_identity_preserves_spectrum() {
    let w = ResidualWeights::identity(1024);
    let mut spec = vec![1.0f32, 2.0, 3.0, 4.0];
    w.apply_spectrum(&mut spec);
    assert_eq!(spec, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn embedded_residual_loads() {
    let w = rlx_aec::embedded_residual_weights().expect("embedded");
    assert_eq!(w.n_fft, 1024);
}

#[test]
fn metrics_helpers() {
    let a = [1.0, 2.0, 3.0];
    let b = [1.0, 2.0, 3.0];
    assert_eq!(rlx_aec::max_abs_error(&a, &b), 0.0);
    assert!(rlx_aec::signal_power(&a) > 0.0);
}
