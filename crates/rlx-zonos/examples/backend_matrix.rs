//! Zonos cross-backend matrix: sentence text → PCM → Whisper coverage.
//!
//! ```bash
//! cargo run -p rlx-zonos --release --example backend_matrix --features apple-silicon
//! ```
//!
//! Env: `RLX_TEXT`, `RLX_MAX_TOKENS`, `RLX_DEVICES` (comma: cpu,metal,mlx),
//! `RLX_ZONOS_DIR`, `RLX_WHISPER_DIR`.

use std::path::PathBuf;
use std::time::Instant;

use rlx_runtime::{Device, is_available};
use rlx_zonos::{
    DEFAULT_DAC_DIR, DEFAULT_LOCAL_DIR, InferOpts, NativeZonos, SAMPLE_RATE, peak_amplitude,
};

const DEFAULT_TEXT: &str = "The quick brown fox jumps over the lazy dog.";

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::var("RLX_ZONOS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_LOCAL_DIR));
    let dac_dir = PathBuf::from(DEFAULT_DAC_DIR);
    let text = std::env::var("RLX_TEXT").unwrap_or_else(|_| DEFAULT_TEXT.into());
    let max_tokens: usize = std::env::var("RLX_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    anyhow::ensure!(
        model_dir.join("model.safetensors").is_file(),
        "missing Zonos — just fetch-zonos"
    );
    anyhow::ensure!(
        dac_dir.join("model.safetensors").is_file(),
        "missing DAC — just fetch-parler-dac"
    );

    let devices = parse_devices();
    let mut whisper = load_whisper();

    println!("== Zonos compiled cross-backend ==");
    println!("text: {text:?}");
    println!(
        "{:<8} {:>8} {:>7} {:>9} {:>8}",
        "backend", "ms", "peak", "whisper", "cov%"
    );

    for (dev, label) in devices {
        if dev != Device::Cpu && !is_available(dev) {
            println!(
                "{label:<8} {:>8} {:>7} {:>9} {:>8}  (n/a)",
                "-", "-", "-", "-"
            );
            continue;
        }
        let t0 = Instant::now();
        let model = match NativeZonos::open(&model_dir, &dac_dir, dev) {
            Ok(m) => m,
            Err(e) => {
                println!("{label:<8}  load err: {e}");
                continue;
            }
        };
        let opts = InferOpts {
            max_new_tokens: Some(max_tokens),
            greedy: true,
            ..InferOpts::default()
        };
        let pcm = match model.synthesize(&text, &opts) {
            Ok(p) => p,
            Err(e) => {
                println!("{label:<8}  synth err: {e}");
                continue;
            }
        };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let peak = peak_amplitude(&pcm);
        let (cov, hyp) = match whisper.as_mut() {
            Some(w) => coverage(w, &pcm, SAMPLE_RATE, &text),
            None => (0.0, String::from("(no whisper)")),
        };
        println!(
            "{label:<8} {ms:>8.0} {peak:>7.3} {:>9} {cov:>7.0}  {:?}",
            if hyp.is_empty() { "-" } else { "ok" },
            trim(&hyp, 48)
        );
    }
    Ok(())
}

fn trim(s: &str, n: usize) -> &str {
    if s.chars().count() <= n {
        s
    } else {
        let end = s.char_indices().nth(n).map(|(i, _)| i).unwrap_or(s.len());
        &s[..end]
    }
}

fn parse_devices() -> Vec<(Device, &'static str)> {
    let raw = std::env::var("RLX_DEVICES")
        .unwrap_or_else(|_| "cpu,metal,mlx,gpu,cuda,vulkan,coreml".into());
    let mut out = Vec::new();
    for part in raw.split(',') {
        match part.trim().to_ascii_lowercase().as_str() {
            "cpu" => out.push((Device::Cpu, "cpu")),
            "metal" => out.push((Device::Metal, "metal")),
            "mlx" => out.push((Device::Mlx, "mlx")),
            "gpu" | "wgpu" => out.push((Device::Gpu, "gpu")),
            "cuda" => out.push((Device::Cuda, "cuda")),
            "vulkan" => out.push((Device::Vulkan, "vulkan")),
            "coreml" | "ane" => out.push((Device::Ane, "CoreML")),
            other => eprintln!("unknown device {other:?} (skip)"),
        }
    }
    if out.is_empty() {
        out.push((Device::Cpu, "cpu"));
    }
    out
}

fn load_whisper() -> Option<rlx_whisper::WhisperRunner> {
    let dir = std::env::var("RLX_WHISPER_DIR")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            let c = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache");
            ["whisper-base.en", "whisper-tiny.en", "whisper-tiny"]
                .into_iter()
                .map(|n| c.join(n))
                .find(|p| p.join("model.safetensors").is_file())
        })?;
    rlx_whisper::WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .ok()
}

fn coverage(
    w: &mut rlx_whisper::WhisperRunner,
    pcm: &[f32],
    sr: u32,
    expect: &str,
) -> (f64, String) {
    let pcm16 = resample(pcm, sr, rlx_whisper::SAMPLE_RATE as u32);
    let Ok(t) = w.transcribe_greedy(&pcm16) else {
        return (0.0, String::new());
    };
    let want: Vec<_> = expect
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|x| x.len() > 2)
        .map(str::to_string)
        .collect();
    let got = t.to_lowercase();
    let hit = want.iter().filter(|w| got.contains(w.as_str())).count();
    (
        100.0 * hit as f64 / want.len().max(1) as f64,
        t.trim().to_string(),
    )
}

fn resample(pcm: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return pcm.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let n = ((pcm.len() as f64) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let src = i as f64 * ratio;
        let j = src.floor() as usize;
        let frac = (src - j as f64) as f32;
        let a = pcm[j];
        let b = pcm.get(j + 1).copied().unwrap_or(a);
        out.push(a * (1.0 - frac) + b * frac);
    }
    out
}
