//! Cross-backend Conformer-CTC transcription matrix (cold + warm ms).
//!
//! Requires a local `.nemo` and WAV (defaults under `.cache/conformer-ctc/`).
//!
//! ```bash
//! just fetch-conformer-ctc
//! just test-conformer-ctc-backends
//!
//! # Or a subset:
//! RLX_DEVICES=cpu,cuda cargo run -p rlx-conformer-ctc --release \
//!   --example backend_matrix --features nvidia-gpu
//! ```
//!
//! Env overrides: `RLX_CONFORMER_CTC_NEMO`, `RLX_CONFORMER_CTC_WAV`, `RLX_DEVICES`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Result, bail};
use rlx_conformer_ctc::{ConformerCtc, wav};
use rlx_runtime::{Device, is_available};

const REF: &str = "well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait";

fn main() -> Result<()> {
    let nemo = env_path("RLX_CONFORMER_CTC_NEMO")
        .unwrap_or_else(|| PathBuf::from(".cache/conformer-ctc/stt_en_conformer_ctc_small.nemo"));
    let wav_path = env_path("RLX_CONFORMER_CTC_WAV")
        .unwrap_or_else(|| PathBuf::from(".cache/conformer-ctc/sample.wav"));
    if !nemo.is_file() {
        bail!("missing {nemo:?} — run: just fetch-conformer-ctc");
    }
    if !wav_path.is_file() {
        bail!("missing {wav_path:?}");
    }

    let bytes = std::fs::read(&wav_path)?;
    let w = wav::parse(&bytes)?;
    let devices = parse_devices();

    println!("== rlx-conformer-ctc backend matrix ==");
    println!("nemo={}", nemo.display());
    println!(
        "wav={} ({} Hz, {} samples)",
        wav_path.display(),
        w.sample_rate,
        w.samples.len()
    );
    println!(
        "{:<8} {:>8} {:>8} {:>5}  transcript",
        "backend", "cold_ms", "warm_ms", "ok"
    );

    let mut failed = false;
    let mut cpu_text: Option<String> = None;

    for (dev, label) in &devices {
        if *dev != Device::Cpu && !is_available(*dev) {
            println!("{label:<8} {:>8} {:>8} {:>5}  (n/a)", "-", "-", "-");
            continue;
        }
        let (text, cold_ms, warm_ms) = match transcribe_cold_warm(&nemo, *dev, &w) {
            Ok(v) => v,
            Err(e) => {
                println!("{label:<8}  err: {e:#}");
                failed = true;
                continue;
            }
        };
        let n = norm(&text);
        let vs_ref = n == norm(REF);
        let vs_cpu = cpu_text.as_ref().map(|c| n == *c).unwrap_or(true);
        if cpu_text.is_none() {
            cpu_text = Some(n.clone());
        }
        let ok = vs_ref && vs_cpu;
        println!(
            "{label:<8} {cold_ms:>8.1} {warm_ms:>8.1} {:>5}  {text}",
            if ok { "yes" } else { "NO" }
        );
        if !ok {
            failed = true;
            if !vs_ref {
                eprintln!("  != ref: {REF}");
            }
            if !vs_cpu {
                eprintln!("  != cpu: {}", cpu_text.as_deref().unwrap_or(""));
            }
        }
    }

    if failed {
        bail!("conformer-ctc backend matrix failed");
    }
    Ok(())
}

fn transcribe_cold_warm(nemo: &Path, device: Device, w: &wav::Wav) -> Result<(String, f64, f64)> {
    let mut asr = ConformerCtc::open(nemo, device)?;
    let pcm = wav::resample(&w.samples, w.sample_rate, asr.config().sample_rate as u32);
    let t0 = Instant::now();
    let text = asr.transcribe(&pcm)?;
    let cold_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = Instant::now();
    let text2 = asr.transcribe(&pcm)?;
    let warm_ms = t1.elapsed().as_secs_f64() * 1000.0;
    if text2 != text {
        bail!("warm diverged from cold on {device:?}");
    }
    Ok((text, cold_ms, warm_ms))
}

fn norm(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() || c == '\'' {
            out.push(c);
        } else if (c.is_whitespace() || c == ',' || c == '.' || c == ';' || c == ':')
            && !out.ends_with(' ')
        {
            out.push(' ');
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from)
}

fn parse_devices() -> Vec<(Device, &'static str)> {
    if let Ok(raw) = std::env::var("RLX_DEVICES") {
        return raw
            .split(',')
            .filter_map(|s| match s.trim().to_ascii_lowercase().as_str() {
                "cpu" => Some((Device::Cpu, "cpu")),
                "metal" | "mps" => Some((Device::Metal, "metal")),
                "mlx" => Some((Device::Mlx, "mlx")),
                "cuda" => Some((Device::Cuda, "cuda")),
                "rocm" | "hip" => Some((Device::Rocm, "rocm")),
                "gpu" | "wgpu" => Some((Device::Gpu, "wgpu")),
                "vulkan" => Some((Device::Vulkan, "vulkan")),
                _ => None,
            })
            .collect();
    }
    vec![
        (Device::Cpu, "cpu"),
        (Device::Metal, "metal"),
        (Device::Mlx, "mlx"),
        (Device::Cuda, "cuda"),
        (Device::Rocm, "rocm"),
        (Device::Gpu, "wgpu"),
        (Device::Vulkan, "vulkan"),
    ]
}
