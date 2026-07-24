//! Cross-backend inventory + RTF matrix for RLX TTS.
//!
//! Product FastSpeech2 + WaveRNN is **host CPU** (BNNS / ndarray). Other RLX
//! devices are probed for availability and reported as unsupported until a
//! Device-backed path lands.
//!
//! ```bash
//! cargo run -p rlx-tts --release --example backend_matrix
//! just tts-backends
//! RLX_ITERS=5 cargo run -p rlx-tts --release --example backend_matrix
//! ```

use std::time::Instant;

use rlx_runtime::{Device, is_available};
use rlx_tts::{RlxTts, VarianceControls, WaveRnnOpts};

const TEXT: &str = "The quick brown fox jumps over the lazy dog near the riverbank at sunrise.";
const LONG: &str = "\
Once upon a time in a quiet valley, a traveler paused by the river and listened \
to the wind in the trees.";

fn iters() -> usize {
    std::env::var("RLX_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
}

fn backends() -> Vec<(Device, &'static str)> {
    let mut v = vec![
        (Device::Cpu, "cpu"),
        (Device::Metal, "metal"),
        (Device::Mlx, "mlx"),
        (Device::Gpu, "gpu"),
        (Device::Cuda, "cuda"),
        (Device::Ane, "ane"),
        (Device::Vulkan, "vulkan"),
    ];
    if let Ok(raw) = std::env::var("RLX_DEVICES") {
        let want: Vec<_> = raw
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .collect();
        v.retain(|(_, name)| want.iter().any(|s| s == name));
    }
    v
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn bench_host(tts: &RlxTts, text: &str, label: &str) -> anyhow::Result<()> {
    let ctrl = VarianceControls::default();
    let vocoder = WaveRnnOpts::product_default();
    let n = iters();
    // Warmup
    let _ = tts.synthesize_text(text, &ctrl, &vocoder)?;
    let mut walls = Vec::with_capacity(n);
    let mut audio = tts.synthesize_text(text, &ctrl, &vocoder)?;
    for _ in 0..n {
        let t0 = Instant::now();
        audio = tts.synthesize_text(text, &ctrl, &vocoder)?;
        walls.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let dur = audio.duration_secs();
    let med = median(walls);
    let rtf = (med / 1000.0) / dur.max(1e-9);
    println!(
        "  {label:8} host-cpu  ok  wall_ms_med={med:.1}  audio_s={dur:.2}  RTF={rtf:.3}  peak={:.3}  iters={n}",
        audio.peak_amplitude()
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let tts = RlxTts::open_default()?;
    println!(
        "rlx-tts backend matrix  bundle={}  rate={}Hz  iters={}",
        tts.bundle_dir().display(),
        tts.sample_rate(),
        iters()
    );
    println!();
    println!(
        "{:<8} {:<8} {:<6} notes",
        "device", "avail", "run",
    );

    let mut any_gpu = false;
    for (dev, name) in backends() {
        let avail = dev == Device::Cpu || is_available(dev);
        if !avail {
            println!("{name:<8} no       skip   not available on this host");
            continue;
        }
        any_gpu |= !matches!(dev, Device::Cpu);
        if matches!(dev, Device::Cpu) {
            println!("{name:<8} yes      run    product FastSpeech2+WaveRNN (host)");
            bench_host(&tts, TEXT, "short")?;
            bench_host(&tts, LONG, "long")?;
        } else {
            println!(
                "{name:<8} yes      skip   product path is host CPU only (no Device kernels yet)"
            );
        }
    }

    println!();
    if any_gpu {
        println!(
            "note: Metal/MLX/GPU/CUDA/ANE are present but unused — rlx-tts compute stays on host BNNS/ndarray."
        );
    }
    println!("done.");
    Ok(())
}
