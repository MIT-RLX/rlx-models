// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Dump the raw 80-bin Kaldi fbank for a wav to a flat f32 file, for parity
//! checking against `torchaudio.compliance.kaldi.fbank`.
//! Usage: `cargo run --example dump_feats -- <a.wav> <out.f32>`

use rlx_funasr::frontend::{FrontendConfig, WavFrontend};
use rlx_funasr::wav;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let wav_path = &args[1];
    let out = &args[2];
    let bytes = std::fs::read(wav_path).unwrap();
    let w = wav::parse(&bytes).unwrap();
    let pcm = wav::resample(&w.samples, w.sample_rate, 16_000);
    let fe = WavFrontend::new(FrontendConfig::default(), None);
    let mode = args.get(3).map(|s| s.as_str()).unwrap_or("raw");
    let fb = if mode == "lfr" {
        fe.extract(&pcm)
    } else {
        fe.fbank(&pcm)
    };
    eprintln!("{mode} {} {}", fb.n_frames, fb.feat_dim);
    let mut bytes = Vec::with_capacity(fb.data.len() * 4);
    for v in &fb.data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(out, bytes).unwrap();
}
