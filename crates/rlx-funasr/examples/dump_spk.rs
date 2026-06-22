// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Dump a CAM++ 192-d speaker embedding to a flat f32 file.
//! Usage: `cargo run --example dump_spk -- <dir> <wav> <out.f32>`

use rlx_funasr::audio;
use rlx_funasr::speaker::CamPlus;
use rlx_runtime::Device;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pcm = audio::load_mono(std::path::Path::new(&args[2]), 16_000).unwrap();
    let m = CamPlus::open(std::path::Path::new(&args[1]), Device::Cpu).unwrap();
    let emb = m.embedding(&pcm).unwrap();
    eprintln!("emb {}", emb.len());
    let mut b = Vec::with_capacity(emb.len() * 4);
    for v in &emb {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&args[3], b).unwrap();
}
