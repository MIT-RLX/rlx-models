//! MioTTS cross-backend matrix: LM once (CPU) → MioCodec native per device → Whisper.
//!
//! ```bash
//! cargo run -p rlx-miotts --release --example backend_matrix --features all-backends
//! ```

use std::path::PathBuf;
use std::time::Instant;

use rlx_miotts::{GenerateOpts, MioSession, SAMPLE_RATE, default_codec_dir, default_model_dir};
use rlx_runtime::{Device, is_available};
use rlx_whisper::WhisperRunner;

const DEFAULT_TEXT: &str = "The quick brown fox jumps over the lazy dog.";
const FOX_WORDS: [&str; 6] = ["quick", "brown", "fox", "jumps", "lazy", "dog"];

fn main() -> anyhow::Result<()> {
    let model = std::env::var("RLX_MIOTTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_model_dir());
    let codec = std::env::var("RLX_MIOCODEC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_codec_dir());
    let text = std::env::var("RLX_TEXT").unwrap_or_else(|_| DEFAULT_TEXT.into());
    let seed: u64 = std::env::var("RLX_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let preset = std::env::var("RLX_PRESET").unwrap_or_else(|_| "en_female".into());
    let devices = parse_devices();
    let mut whisper = load_whisper();

    println!("== MioTTS cross-backend (eager LM CPU + MioCodec native) ==");
    println!("text: {text:?}  seed={seed}  preset={preset}");
    println!("model={}  codec={}", model.display(), codec.display());

    let cache = std::env::var("RLX_CODES_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/miotts_fox_codes.json"));

    let t0 = Instant::now();
    let codes: Vec<u32> = if cache.is_file() && std::env::var_os("RLX_FORCE_RESYNTH").is_none() {
        println!("[cache] loading codes from {}", cache.display());
        serde_json::from_str(&std::fs::read_to_string(&cache)?)?
    } else {
        let mut session = MioSession::open(&model, &codec, Device::Cpu)?;
        let opts = GenerateOpts {
            seed,
            max_new_tokens: 400,
            preset: preset.clone(),
        };
        println!("[lm] generating codes on CPU…");
        let codes = session.generate_codes(&text, &opts)?;
        std::fs::write(&cache, serde_json::to_string(&codes)?)?;
        println!(
            "[lm {:?}] codes={}  cache={}",
            t0.elapsed(),
            codes.len(),
            cache.display()
        );
        codes
    };

    println!(
        "{:<8} {:>8} {:>7} {:>8} {:>8}  {}",
        "backend", "ms", "peak", "fox", "cos", "ep"
    );

    let mut cpu_wav: Option<Vec<f32>> = None;
    let mut failed = false;
    let presets = model.join("presets");

    for (dev, label) in &devices {
        if *dev != Device::Cpu && !is_available(*dev) {
            println!(
                "{label:<8} {:>8} {:>7} {:>8} {:>8}  (n/a)",
                "-", "-", "-", "-"
            );
            continue;
        }
        let t1 = Instant::now();
        let pcm = match MioSession::decode_codes_on(&codec, &presets, *dev, &codes, &preset) {
            Ok(p) => p,
            Err(e) => {
                println!("{label:<8}  decode err: {e:#}");
                failed = true;
                continue;
            }
        };
        let ms = t1.elapsed().as_secs_f64() * 1000.0;
        let peak = pcm.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let cos = cpu_wav
            .as_ref()
            .map(|ref_wav| cosine(ref_wav, &pcm))
            .unwrap_or(1.0);
        if cpu_wav.is_none() {
            cpu_wav = Some(pcm.clone());
        }

        let fox = whisper_hits(&mut whisper, &pcm);
        println!("{label:<8} {ms:>8.1} {peak:>7.3} {fox:>3}/6 {cos:>8.5}");
        if fox < 5 || cos < 0.99 {
            failed = true;
        }
    }

    if failed {
        anyhow::bail!("backend matrix failed");
    }
    println!("ok  sample_rate={SAMPLE_RATE}");
    Ok(())
}

fn whisper_hits(whisper: &mut Option<WhisperRunner>, pcm24: &[f32]) -> usize {
    let Some(w) = whisper.as_mut() else {
        return 0;
    };
    let pcm16 = resample_24k_to_16k(pcm24);
    let Ok(t) = w.transcribe_greedy(&pcm16) else {
        return 0;
    };
    let lower = t.to_lowercase();
    FOX_WORDS.iter().filter(|w| lower.contains(*w)).count()
}

fn resample_24k_to_16k(pcm: &[f32]) -> Vec<f32> {
    let out_len = pcm.len() * 2 / 3;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * 3.0 / 2.0;
        let j = src.floor() as usize;
        let f = (src - j as f64) as f32;
        let a = pcm.get(j).copied().unwrap_or(0.0);
        let b = pcm.get(j + 1).copied().unwrap_or(a);
        out.push(a * (1.0 - f) + b * f);
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

fn parse_devices() -> Vec<(Device, &'static str)> {
    let raw = std::env::var("RLX_DEVICES")
        .unwrap_or_else(|_| "cpu,metal,mlx,gpu,vulkan,cuda,rocm".into());
    let mut out = Vec::new();
    for part in raw.split(',') {
        let label = part.trim();
        let dev = match label {
            "cpu" => Device::Cpu,
            "metal" => Device::Metal,
            "mlx" => Device::Mlx,
            "gpu" | "wgpu" => Device::Gpu,
            "vulkan" => Device::Vulkan,
            "cuda" => Device::Cuda,
            "rocm" => Device::Rocm,
            "coreml" | "ane" => Device::Ane,
            _ => continue,
        };
        out.push((
            dev,
            match label {
                "gpu" | "wgpu" => "gpu",
                "coreml" | "ane" => "ane",
                other => {
                    // leak-free: use static for known ones
                    match other {
                        "cpu" => "cpu",
                        "metal" => "metal",
                        "mlx" => "mlx",
                        "vulkan" => "vulkan",
                        "cuda" => "cuda",
                        "rocm" => "rocm",
                        _ => "dev",
                    }
                }
            },
        ));
    }
    out
}

fn load_whisper() -> Option<WhisperRunner> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::var("RLX_WHISPER_DIR")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            ["whisper-tiny", "whisper-tiny.en", "whisper-base.en"]
                .iter()
                .map(|n| root.join(".cache").join(n))
                .find(|p| p.join("model.safetensors").is_file())
        })?;
    WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .language("en")
        .build()
        .ok()
}
