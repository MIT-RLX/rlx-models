//! Sesame cross-backend matrix: LM once (CPU) → Mimi decode per device → Whisper.
//!
//! ```bash
//! cargo run -p rlx-sesame --release --example backend_matrix --features all-backends
//! ```
//!
//! Env: `RLX_TEXT`, `RLX_DEVICES=cpu,metal,mlx,gpu,cuda,vulkan,rocm`,
//! `RLX_WHISPER_DIR`, `RLX_SESAME_DIR`, `RLX_MIMI_DIR`, `RLX_SEED` (default 42),
//! `RLX_CODES_CACHE` (optional JSON reuse of frames).

use std::path::PathBuf;
use std::time::Instant;

use rlx_runtime::{Device, is_available};
use rlx_sesame::{GenerateOpts, SesameSession, default_mimi_dir, default_model_dir};

const DEFAULT_TEXT: &str = "The quick brown fox jumps over the lazy dog.";
const FOX_WORDS: [&str; 6] = ["quick", "brown", "fox", "jumps", "lazy", "dog"];
const LONG_WORDS: [&str; 15] = [
    "quick", "brown", "fox", "jumps", "lazy", "dog", "courage", "kindness", "matter", "people",
    "hard", "times", "help", "each", "other",
];

fn main() -> anyhow::Result<()> {
    let model = std::env::var("RLX_SESAME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_model_dir());
    let mimi_dir = std::env::var("RLX_MIMI_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_mimi_dir());
    let text = std::env::var("RLX_TEXT").unwrap_or_else(|_| DEFAULT_TEXT.into());
    let seed: u64 = std::env::var("RLX_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let devices = parse_devices();
    let mut whisper = load_whisper();

    println!("== Sesame CSM cross-backend (eager LM CPU + Mimi on device) ==");
    println!("text: {text:?}  seed={seed}");
    println!("model={}  mimi={}", model.display(), mimi_dir.display());

    let cache = std::env::var("RLX_CODES_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            if text.contains("Courage") {
                PathBuf::from("/tmp/sesame_long_codes.json")
            } else {
                PathBuf::from("/tmp/sesame_fox_codes.json")
            }
        });

    let want_words: &[&str] = if text.contains("Courage") {
        &LONG_WORDS
    } else {
        &FOX_WORDS
    };
    let min_hits = if text.contains("Courage") { 12 } else { 5 };
    let max_frames: usize = std::env::var("RLX_MAX_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if text.contains("Courage") { 400 } else { 200 });

    let t0 = Instant::now();
    let frames: Vec<Vec<u32>> =
        if cache.is_file() && std::env::var_os("RLX_FORCE_RESYNTH").is_none() {
            println!("[cache] loading frames from {}", cache.display());
            serde_json::from_str(&std::fs::read_to_string(&cache)?)?
        } else {
            let mut session = SesameSession::open_on(&model, &mimi_dir, Device::Cpu)?;
            let opts = GenerateOpts {
                seed,
                max_audio_frames: max_frames,
                ..Default::default()
            };
            println!("[lm] generating frames on CPU…");
            let frames = session.generate_frames(&text, None, &opts)?;
            std::fs::write(&cache, serde_json::to_string(&frames)?)?;
            println!(
                "[lm {:?}] frames={}  cache={}",
                t0.elapsed(),
                frames.len(),
                cache.display()
            );
            frames
        };
    let n_cb = frames.first().map(|r| r.len()).unwrap_or(32);

    println!(
        "{:<8} {:>8} {:>7} {:>8} {:>8}",
        "backend", "ms", "peak", "fox", "cos"
    );

    let mut cpu_wav: Option<Vec<f32>> = None;
    let mut failed = false;

    for (dev, label) in &devices {
        if *dev != Device::Cpu && !is_available(*dev) {
            println!(
                "{label:<8} {:>8} {:>7} {:>8} {:>8}  (n/a)",
                "-", "-", "-", "-"
            );
            continue;
        }
        let t1 = Instant::now();
        let pcm = match SesameSession::decode_frames_on(&mimi_dir, *dev, &frames, n_cb) {
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

        let (hits, hyp) = match whisper.as_mut() {
            Some(w) => word_whisper(w, &pcm, want_words),
            None => (0, String::from("(no whisper)")),
        };
        if hits < min_hits {
            failed = true;
        }
        // Mimi GPU paths can differ slightly; keep a loose waveform gate.
        if *label != "cpu" && cos < 0.95 {
            failed = true;
        }
        println!(
            "{label:<8} {ms:>8.0} {peak:>7.3} {hits:>4}/{:<2} {cos:>8.5}  {:?}",
            want_words.len(),
            trim(&hyp, 56)
        );
    }

    if failed {
        anyhow::bail!(
            "one or more backends failed whisper (≥{min_hits}/{}) or cosine (≥0.95) gate",
            want_words.len()
        );
    }
    Ok(())
}

fn word_whisper(w: &mut rlx_whisper::WhisperRunner, pcm: &[f32], want: &[&str]) -> (usize, String) {
    let pcm16 = resample_24k_to_16k(pcm);
    let hyp = w.transcribe_greedy(&pcm16).unwrap_or_default();
    let lower = hyp.to_ascii_lowercase();
    let hits = want.iter().filter(|w| lower.contains(*w)).count();
    (hits, hyp)
}

fn resample_24k_to_16k(pcm: &[f32]) -> Vec<f32> {
    let ratio = 16_000.0 / 24_000.0;
    let out_len = ((pcm.len() as f64) * ratio).round() as usize;
    let mut out = vec![0.0f32; out_len];
    for (i, o) in out.iter_mut().enumerate() {
        let src = i as f64 / ratio;
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(pcm.len().saturating_sub(1));
        let t = (src - i0 as f64) as f32;
        *o = pcm[i0] * (1.0 - t) + pcm[i1] * t;
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
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
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
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
        .unwrap_or_else(|_| "cpu,metal,mlx,gpu,cuda,vulkan,rocm,coreml".into());
    let mut out = Vec::new();
    for part in raw.split(',') {
        match part.trim().to_ascii_lowercase().as_str() {
            "cpu" => out.push((Device::Cpu, "cpu")),
            "metal" => out.push((Device::Metal, "metal")),
            "mlx" => out.push((Device::Mlx, "mlx")),
            "gpu" | "wgpu" => out.push((Device::Gpu, "gpu")),
            "cuda" => out.push((Device::Cuda, "cuda")),
            "vulkan" => out.push((Device::Vulkan, "vulkan")),
            "rocm" => out.push((Device::Rocm, "rocm")),
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
            let c = PathBuf::from(".cache/whisper-tiny");
            c.join("model.safetensors").is_file().then_some(c)
        })?;
    rlx_whisper::WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .language("en")
        .build()
        .ok()
}
