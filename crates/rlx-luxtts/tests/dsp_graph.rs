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

//! The log-mel and the ISTFT as rlx graphs (`rlx_core::export::audio`)
//! compute what the runner's `dsp` does, on the fixtures' second of audio.
#![cfg(feature = "export")]

use std::collections::HashMap;
use std::path::PathBuf;

use rlx_core::export::audio::{istft_bases, istft_center, log_mel, stft_bases};
use rlx_luxtts::dsp::{self, HOP, N_FFT, N_FREQ, N_MELS};

fn read(name: &str) -> Vec<f32> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn run(
    g: rlx_ir::Graph,
    params: &HashMap<String, Vec<f32>>,
    inputs: &[(&str, &[f32])],
) -> Vec<f32> {
    let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g);
    for (k, v) in params {
        c.set_param(k, v);
    }
    c.finalize_params();
    c.run(inputs).remove(0)
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn the_log_mel_graph_is_the_runners() {
    let wav = read("golden_wav_in.f32");
    let (want, t) = dsp::VocosFbank::new().log_mel(&wav); // [N_MELS, t]
    let mut signal = dsp::reflect_pad(&wav, N_FFT / 2);
    signal.truncate(HOP * (t + N_FFT / HOP - 1));
    let fb = dsp::mel_filterbank(); // [N_MELS, N_FREQ]
    let mut params = stft_bases(N_FFT, &dsp::hann_periodic(N_FFT));
    params.insert(
        "mel.filters".into(),
        (0..N_FREQ * N_MELS)
            .map(|i| fb[(i % N_MELS) * N_FREQ + i / N_MELS])
            .collect(),
    );
    let got = run(
        log_mel(N_FFT, HOP, N_MELS, t).unwrap(),
        &params,
        &[("signal", &signal)],
    );
    let got: Vec<f32> = (0..N_MELS * t)
        .map(|i| got[(i % t) * N_MELS + i / t])
        .collect();
    let off = worst(&got, &want);
    eprintln!("log-mel [{N_MELS}, {t}]: off by {off}");
    assert!(off < 1e-3, "off by {off}");
}

#[test]
fn the_istft_graph_is_the_runners() {
    let (real, imag) = (read("golden_real.f32"), read("golden_imag.f32"));
    let t = real.len() / N_FREQ;
    let want = dsp::istft(&real, &imag, t);
    let window = dsp::hann_periodic(N_FFT);
    let got = run(
        istft_center(N_FFT, HOP, &window, t).unwrap(),
        &istft_bases(N_FFT, &window),
        &[("real", &real), ("imag", &imag)],
    );
    let peak = want.iter().fold(0f32, |m, x| m.max(x.abs()));
    let off = worst(&got, &want);
    eprintln!("istft {t} frames: off by {off} (peak {peak})");
    assert!(off < 1e-5 * peak.max(1.0), "off by {off}");
}
