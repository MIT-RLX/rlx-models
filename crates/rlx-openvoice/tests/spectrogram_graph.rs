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

//! The spectrogram as an rlx graph (`rlx_core::export::audio::spectrogram`)
//! is the runner's (`dsp::Spectrogram::magnitude`).
#![cfg(feature = "export")]

use rlx_core::export::audio::{spectrogram, stft_bases};
use rlx_openvoice::dsp::{HOP, N_FFT, N_FREQ, Spectrogram, hann_periodic};

#[test]
fn the_spectrogram_graph_is_the_runners() {
    let wav: Vec<f32> = (0..9000)
        .map(|i| (i as f32 * 0.031).sin() * 0.4 + (i as f32 * 0.17).cos() * 0.1)
        .collect();
    let (want, t) = Spectrogram::new().magnitude(&wav); // [N_FREQ, t]
    let pad = (N_FFT - HOP) / 2;
    let n = wav.len();
    let mut padded: Vec<f32> = (0..pad).map(|i| wav[pad - i]).collect();
    padded.extend_from_slice(&wav);
    padded.extend((0..pad).map(|i| wav[n - 2 - i]));
    let signal = &padded[..HOP * (t + N_FFT / HOP - 1)];
    let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu)
        .compile(spectrogram(N_FFT, HOP, 1e-6, t).unwrap());
    for (k, v) in stft_bases(N_FFT, &hann_periodic(N_FFT)) {
        c.set_param(&k, &v);
    }
    c.finalize_params();
    let got = c.run(&[("signal", signal)]).remove(0); // [t, N_FREQ]
    let worst = (0..N_FREQ * t).fold(0f32, |m, i| {
        let (f, k) = (i / t, i % t);
        m.max((got[k * N_FREQ + f] - want[i]).abs())
    });
    eprintln!("spectrogram [{N_FREQ}, {t}]: off by {worst}");
    assert!(worst < 1e-3, "off by {worst}");
}
