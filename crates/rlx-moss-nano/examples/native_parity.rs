//! Cross-backend parity: run moss NATIVE on CPU vs another DEVICE, isolating the
//! LM forward (sampled codes) from the codec (audio). Same seed → codes must
//! match; then decode the SAME codes on both devices → audio must match.
//! `RLX_MOSS_DIR=... DEVICE=metal TEXT="Hi." MAXF=6 cargo run -p rlx-moss-nano
//!  --example native_parity --no-default-features --features metal`
use rlx_moss_nano::{MossNative, NativeOpts};
use rlx_runtime::Device;
use std::path::PathBuf;

fn parse_device(s: &str) -> Device {
    match s.to_lowercase().as_str() {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "gpu" | "wgpu" => Device::Gpu,
        "ane" | "coreml" => Device::Ane,
        "cuda" => Device::Cuda,
        _ => Device::Cpu,
    }
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        d += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (d / (na.sqrt() * nb.sqrt())) as f32
}

fn main() -> anyhow::Result<()> {
    let dir =
        PathBuf::from(std::env::var("RLX_MOSS_DIR").unwrap_or("weights/tts/moss-nano".into()));
    let text = std::env::var("TEXT").unwrap_or("Hi.".into());
    let maxf: usize = std::env::var("MAXF")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let seed: u64 = std::env::var("SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let device = parse_device(&std::env::var("DEVICE").unwrap_or("metal".into()));
    let opts = NativeOpts {
        seed,
        max_frames: maxf,
        ..Default::default()
    };

    let cpu = MossNative::load_on(&dir, Device::Cpu)?;
    let gpu = MossNative::load_on(&dir, device)?;
    let voice = cpu.voice_names()[0].clone();
    let vc = cpu.voice_prompt_codes(&voice)?;

    // 0) prefill hidden (before any sampling)
    let h_cpu = cpu.debug_prefill_hidden(&text, &vc, maxf)?;
    let h_gpu = gpu.debug_prefill_hidden(&text, &vc, maxf)?;
    let hc = cos(&h_cpu, &h_gpu);
    let hmax = h_cpu
        .iter()
        .zip(&h_gpu)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "HIDDEN cos={hc:.6} max_abs={hmax:.5} n={} first_cpu={:.4} first_{}={:.4}",
        h_cpu.len(),
        h_cpu.first().copied().unwrap_or(0.0),
        device.name(),
        h_gpu.first().copied().unwrap_or(0.0)
    );

    // 1) LM forward → codes
    let codes_cpu = cpu.generate_codes(&text, &vc, &opts)?;
    let codes_gpu = gpu.generate_codes(&text, &vc, &opts)?;
    let flat = |f: &[Vec<i32>]| -> Vec<i32> { f.iter().flatten().copied().collect() };
    let (fc, fg) = (flat(&codes_cpu), flat(&codes_gpu));
    let matched = fc.len() == fg.len() && fc.iter().zip(&fg).all(|(a, b)| a == b);
    let mism = fc.iter().zip(&fg).filter(|(a, b)| a != b).count();
    eprintln!(
        "CODES  cpu_frames={} {}_frames={} codes_match={} (mismatch {}/{})",
        codes_cpu.len(),
        device.name(),
        codes_gpu.len(),
        matched,
        mism,
        fc.len().min(fg.len())
    );
    if let (Some(f0c), Some(f0g)) = (codes_cpu.first(), codes_gpu.first()) {
        eprintln!("  frame0 cpu  = {:?}", f0c);
        eprintln!("  frame0 {} = {:?}", device.name(), f0g);
    }

    // 2) codec parity on the SAME (cpu) codes
    let a_cpu = cpu.decode_codes(&codes_cpu)?;
    let a_gpu = gpu.decode_codes(&codes_cpu)?;
    let c = cos(&a_cpu, &a_gpu);
    let max_abs = a_cpu
        .iter()
        .zip(&a_gpu)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let pk_cpu = a_cpu.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let pk_gpu = a_gpu.iter().fold(0f32, |m, &x| m.max(x.abs()));
    eprintln!(
        "CODEC  cos={c:.6} max_abs={max_abs:.5} peak_cpu={pk_cpu:.4} peak_{}={pk_gpu:.4} (same codes)",
        device.name()
    );
    Ok(())
}
