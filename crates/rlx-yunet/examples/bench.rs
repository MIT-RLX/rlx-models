// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Detection time per frame: `bench WEIGHTS.safetensors [WIDTH HEIGHT] [cpu|metal]` (default
//! 320×180, the Reachy tracker's size, padded to 320×192).

use std::time::Instant;

use rlx_runtime::Device;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let w: usize = a.get(2).map_or(Ok(320), |s| s.parse())?;
    let h: usize = a.get(3).map_or(Ok(180), |s| s.parse())?;
    let device = if a.get(4).is_some_and(|d| d == "metal") {
        Device::Metal
    } else {
        Device::Cpu
    };
    let mut net = rlx_yunet::YuNet::load(std::path::Path::new(&a[1]), device)?;
    let bgr: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
    let t = Instant::now();
    net.detect_bgr(&bgr, w, h, 3 * w)?;
    println!(
        "first frame (compile + run): {:.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    let n = 50;
    let t = Instant::now();
    for _ in 0..n {
        net.detect_bgr(&bgr, w, h, 3 * w)?;
    }
    println!(
        "{w}x{h} on {device:?}: {:.2} ms per frame ({n} frames)",
        t.elapsed().as_secs_f64() * 1e3 / n as f64
    );
    Ok(())
}
