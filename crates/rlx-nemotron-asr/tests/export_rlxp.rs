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

//! An exported Nemotron 3.5 ASR (`weights/asr/nemotron-asr`), opened by rlx
//! alone, transcribes as the crate's runner does: the frontend graph's mel
//! is the runner's, and the transcript and its token ids are the runner's.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_core::export::audio::{
    NemoFrontend, NemoPadding, SAMPLES, audio_iota, nemo_features_with, stft_bases,
};
use rlx_nemotron_asr::NemotronAsr;
use rlx_nemotron_asr::export::export_rlxp;
use rlx_nemotron_asr::mel::{Frontend, log_mel};
use rlx_runtime::Device;
use serde_json::json;

fn speech(rate: u32) -> Vec<f32> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/jfk/jfk_rust_speech.wav");
    let bytes = std::fs::read(path).unwrap();
    let w = rlx_nemotron_asr::wav::parse(&bytes).unwrap();
    rlx_nemotron_asr::wav::resample(&w.samples, w.sample_rate, rate)
}

#[test]
fn an_exported_nemotron_asr_transcribes_as_the_runner_does() {
    let nemo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../weights/asr/nemotron-asr/nemotron-3.5-asr-streaming-0.6b.nemo");
    if !nemo.is_file() {
        eprintln!("skip: no Nemotron ASR checkpoint");
        return;
    }
    let mut asr = NemotronAsr::open(&nemo, Device::Cpu).unwrap();
    assert!(asr.set_language("auto"));
    let cfg = asr.config().clone();
    let pcm = speech(cfg.sample_rate as u32);

    // The frontend graph against the runner's mel.
    let model = rlx_nemo::NemoModel::open(&nemo).unwrap();
    let fe = Frontend::from_model(&model, &cfg).unwrap();
    let mut window = vec![0f32; cfg.n_fft];
    let left = (cfg.n_fft - fe.window.len()) / 2;
    window[left..left + fe.window.len()].copy_from_slice(&fe.window);
    let g = nemo_features_with(
        &NemoFrontend {
            n_fft: cfg.n_fft,
            hop: cfg.hop_length,
            n_mels: cfg.n_mels,
            window: window.clone(),
            preemphasis: Some(0.97),
            per_feature: cfg.normalize == "per_feature",
        },
        NemoPadding {
            zeros: true,
            mask_tail: true,
        },
    )
    .unwrap();
    let mut bind = rlx_ir::DimBinding::new();
    bind.set(SAMPLES, pcm.len());
    let g = rlx_ir::dynamic::specialize(&g, &bind);
    let mut c = rlx_runtime::Session::new(Device::Cpu).compile(g);
    let n_freq = cfg.n_fft / 2 + 1;
    let mut filters = vec![0f32; n_freq * cfg.n_mels];
    for m in 0..cfg.n_mels {
        for k in 0..n_freq {
            filters[k * cfg.n_mels + m] = fe.fb[m * n_freq + k];
        }
    }
    for (n, v) in stft_bases(cfg.n_fft, &window) {
        c.set_param(&n, &v);
    }
    c.set_param("mel.filters", &filters);
    c.set_param("audio.iota", &audio_iota());
    c.finalize_params();
    let got = c.run(&[("audio", &pcm)]).remove(0);
    let want = log_mel(&cfg, &pcm, Some(&fe)).data;
    assert_eq!(got.len(), want.len());
    let diff = got
        .iter()
        .zip(&want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("mel: {} values, max |d| {diff}", want.len());
    assert!(diff < 1e-2, "mel max |d| {diff}");

    let want_ids = asr.transcribe_ids(&pcm).unwrap();
    let want_text = asr.decode_text(&want_ids, true);
    drop(asr);

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("nemotron-asr.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&nemo, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let text = m
        .transcribe(&pcm, cfg.sample_rate as u32, &json!({}))
        .unwrap();
    eprintln!("runner:  {want_text}\npackage: {text}");
    assert_eq!(text, want_text);
}
