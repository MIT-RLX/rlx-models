// SPDX-License-Identifier: GPL-3.0-only
//! Time two consecutive runs at the same length (2nd hits the graph cache).
use rlx_funasr::{audio, sensevoice::SenseVoice};
use rlx_runtime::Device;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let pcm = audio::load_mono(std::path::Path::new(&a[2]), 16_000).unwrap();
    let m = SenseVoice::open(std::path::Path::new(&a[1]), Device::Cpu).unwrap();
    for i in 0..2 {
        let t = Instant::now();
        let (_l, _n) = m.logits(&pcm, "zh", false).unwrap();
        println!("run {i}: {:.2}s", t.elapsed().as_secs_f32());
    }
}
