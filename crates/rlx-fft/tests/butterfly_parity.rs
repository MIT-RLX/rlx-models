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

//! Exact FFT / IFFT parity and training convergence checks.

use rlx_fft::butterfly::{
    build_butterfly_forward_graph, build_butterfly_inverse_graph, butterfly_forward_eager,
    butterfly_forward_real_batch, butterfly_inverse_complex_batch,
};
use rlx_fft::stockham::stockham_forward_eager;
use rlx_fft::{
    EncDecTrainConfig, FftLearnConfig, TrainConfig, TransformDir, exact_twiddles, reference,
    train_butterfly_eager, train_encdec_eager,
};

#[test]
fn exact_twiddles_match_rustfft() {
    for &n in &[64usize, 128, 256] {
        let cfg = FftLearnConfig::new(n, 1).unwrap();
        let tw = exact_twiddles(&cfg);
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin()).collect();
        let pred = butterfly_forward_real_batch(&signal, &tw, 1, n).unwrap();
        let target = reference::fft_real_batch(&signal, 1, n).unwrap();
        let err = reference::max_abs_error(&pred, &target);
        assert!(
            err < 1e-4,
            "n_fft={n} max_abs_error={err} (expected < 1e-4 with exact twiddles)"
        );
    }
}

#[test]
fn exact_itwiddles_match_rustifft() {
    for &n in &[64usize, 128, 256] {
        let cfg = FftLearnConfig::new(n, 1).unwrap();
        let tw = exact_twiddles(&cfg);
        let spectrum: Vec<f32> = (0..n * 2).map(|i| ((i as f32) * 0.05).sin()).collect();
        let pred = butterfly_inverse_complex_batch(&spectrum, &tw, 1, n).unwrap();
        let target = reference::ifft_complex_batch(&spectrum, 1, n).unwrap();
        let err = reference::max_abs_error(&pred, &target);
        assert!(
            err < 1e-4,
            "n_fft={n} ifft max_abs_error={err} (expected < 1e-4 with exact twiddles)"
        );
    }
}

#[test]
fn fft_ifft_roundtrip() {
    let n = 128usize;
    let batch = 2usize;
    let cfg = FftLearnConfig::new(n, batch).unwrap();
    let fft_tw = exact_twiddles(&cfg);
    let ifft_tw = exact_twiddles(&cfg);
    let signal: Vec<f32> = (0..batch * n).map(|i| (i as f32 * 0.03).sin()).collect();
    let spectrum = butterfly_forward_real_batch(&signal, &fft_tw, batch, n).unwrap();
    let recovered = butterfly_inverse_complex_batch(&spectrum, &ifft_tw, batch, n).unwrap();
    let scale = reference::roundtrip_scale(n);
    let mut max_err = 0f32;
    for b in 0..batch {
        for i in 0..n {
            let base = b * n * 2 + i * 2;
            max_err = max_err.max((recovered[base] - signal[b * n + i] * scale).abs());
            max_err = max_err.max(recovered[base + 1].abs());
        }
    }
    assert!(
        max_err < 1e-3,
        "roundtrip max_err={max_err} (expected < 1e-3 for unnormalized fft/ifft pair)"
    );
}

#[test]
fn training_improves_small_fft() {
    let cfg = TrainConfig {
        model: FftLearnConfig::new(64, 4).unwrap(),
        direction: TransformDir::Forward,
        steps: 100,
        lr: 5e-4,
        log_every: 0,
        seed: 7,
        ..TrainConfig::default()
    };
    let report = train_butterfly_eager(&cfg, TransformDir::Forward).unwrap();
    assert!(
        report.final_mse < 1e-3,
        "expected mse < 1e-3 after training, got {}",
        report.final_mse
    );
    assert!(
        report.max_error < 0.05,
        "expected max_err < 0.05, got {}",
        report.max_error
    );
}

#[test]
fn training_improves_small_ifft() {
    let cfg = TrainConfig {
        model: FftLearnConfig::new(64, 4).unwrap(),
        direction: TransformDir::Inverse,
        steps: 100,
        lr: 5e-4,
        log_every: 0,
        seed: 9,
        ..TrainConfig::default()
    };
    let report = train_butterfly_eager(&cfg, TransformDir::Inverse).unwrap();
    assert!(
        report.final_mse < 1e-3,
        "expected ifft mse < 1e-3 after training, got {}",
        report.final_mse
    );
    assert!(
        report.max_error < 0.05,
        "expected ifft max_err < 0.05, got {}",
        report.max_error
    );
}

#[test]
fn training_encdec_improves_roundtrip() {
    let cfg = EncDecTrainConfig {
        model: FftLearnConfig::new(64, 4).unwrap(),
        steps: 150,
        lr: 5e-4,
        spectrum_weight: 1.0,
        seed: 11,
        log_every: 0,
        ..EncDecTrainConfig::default()
    };
    let report = train_encdec_eager(&cfg).unwrap();
    assert!(
        report.reconstruction_mse < 1e-2,
        "expected recon mse < 1e-2, got {}",
        report.reconstruction_mse
    );
    assert!(
        report.spectrum_mse < 1e-3,
        "expected spectrum mse < 1e-3, got {}",
        report.spectrum_mse
    );
    assert!(
        report.roundtrip_max_error < 0.15,
        "expected roundtrip max_err < 0.15, got {}",
        report.roundtrip_max_error
    );
}

#[test]
fn phased_training_improves_each_stage() {
    use rlx_fft::{PhasedTrainConfig, train_phased_encdec};

    let out = std::env::temp_dir().join(format!("rlx_fft_phased_{}", std::process::id()));
    let cfg = PhasedTrainConfig {
        model: FftLearnConfig::new(64, 4).unwrap(),
        encoder_steps: 80,
        decoder_steps: 80,
        joint_steps: 80,
        lr: 5e-4,
        spectrum_weight: 1.0,
        seed: 13,
        log_every: 0,
        out_dir: Some(out.clone()),
    };
    let report = train_phased_encdec(&cfg).unwrap();
    assert_eq!(report.phases.len(), 3);
    let p1 = &report.phases[0];
    let p3 = &report.phases[2];
    assert!(
        p1.encoder_spectrum_max_err < 0.05,
        "phase1 enc max_err={}",
        p1.encoder_spectrum_max_err
    );
    assert!(
        report.phases[1].decoder_time_max_err < 0.05,
        "phase2 dec max_err={}",
        report.phases[1].decoder_time_max_err
    );
    assert!(
        p3.roundtrip_max_err < 0.15,
        "phase3 roundtrip max_err={}",
        p3.roundtrip_max_err
    );
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn stockham_debug_vs_butterfly() {
    let n = 64usize;
    let mut state = vec![0f32; n * 2];
    for i in 0..n {
        state[i * 2] = (i as f32 * 0.1).sin();
    }
    let tw = exact_twiddles(&FftLearnConfig::new(n, 1).unwrap());
    let b = butterfly_forward_eager(&state, &tw, n).unwrap();
    let s = stockham_forward_eager(&state, &tw, n).unwrap();
    let signal: Vec<f32> = state.iter().step_by(2).copied().collect();
    let r = reference::fft_real_batch(&signal, 1, n).unwrap();
    let eb = reference::max_abs_error(&b, &r);
    let es = reference::max_abs_error(&s, &r);
    eprintln!("butterfly vs ref: {eb}, stockham vs ref: {es}");
    assert!(eb < 1e-3);
    assert!(es < 1e-3, "stockham vs ref {es}");
}

#[test]
fn stockham_matches_rustfft() {
    use rlx_fft::stockham::stockham_forward_real_batch;
    for &n in &[64usize, 128, 256] {
        let cfg = FftLearnConfig::new(n, 2).unwrap();
        let tw = exact_twiddles(&cfg);
        let signal: Vec<f32> = (0..n * 2).map(|i| (i as f32 * 0.03).sin()).collect();
        let pred = stockham_forward_real_batch(&signal, &tw, 2, n).unwrap();
        let target = reference::fft_real_batch(&signal, 2, n).unwrap();
        let err = reference::max_abs_error(&pred, &target);
        assert!(err < 1e-3, "n={n} stockham err={err}");
    }
}

#[test]
fn butterfly_graph_builds() {
    let cfg = FftLearnConfig::new(64, 2).unwrap();
    let fwd = build_butterfly_forward_graph(&cfg).unwrap();
    assert_eq!(fwd.params.len(), cfg.twiddle_param_count());
    assert!(!fwd.graph.outputs.is_empty());

    let inv = build_butterfly_inverse_graph(&cfg).unwrap();
    assert_eq!(inv.params.len(), cfg.twiddle_param_count());
    assert!(!inv.graph.outputs.is_empty());
}

#[test]
fn butterfly_compiled_matches_eager() {
    use rlx_fft::compile::try_compile_graph;
    use rlx_fft::twiddle::exact_twiddles;
    use rlx_fft::weights::WeightStore;
    use rlx_runtime::Device;

    let cfg = FftLearnConfig::new(64, 2).unwrap();
    let tw = exact_twiddles(&cfg);
    let built = build_butterfly_forward_graph(&cfg).unwrap();
    let mut compiled = try_compile_graph(Device::Cpu, built.graph).unwrap();
    WeightStore::from_twiddles(&tw, 64).apply_butterfly(&mut compiled, 2, 64);

    let signal: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
    let pred = compiled.run(&[("signal", &signal)]).remove(0);
    let eager = butterfly_forward_real_batch(&signal, &tw, 2, 64).unwrap();
    let err = reference::max_abs_error(&pred, &eager);
    assert!(err < 1e-4, "compiled vs eager err={err}");
}
