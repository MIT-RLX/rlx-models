// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Dump SenseVoice per-frame CTC logits `[t_total, vocab]` to a flat f32 file.
//! Usage: `cargo run --example dump_logits -- <dir> <wav> <out.f32> [lang]`

use rlx_funasr::audio;
use rlx_funasr::sensevoice::SenseVoice;
use rlx_runtime::Device;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let wav = &args[2];
    let out = &args[3];
    let lang = args.get(4).map(|s| s.as_str()).unwrap_or("auto");
    let pcm = audio::load_mono(std::path::Path::new(wav), 16_000).unwrap();
    let m = SenseVoice::open(std::path::Path::new(dir), Device::Cpu).unwrap();
    let (logits, t) = m.logits(&pcm, lang, false).unwrap();
    let vocab = logits.len() / t;
    eprintln!("logits {t} {vocab}");
    let mut b = Vec::with_capacity(logits.len() * 4);
    for v in &logits {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(out, b).unwrap();
}
