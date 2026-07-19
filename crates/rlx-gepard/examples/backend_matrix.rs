//! Gepard cross-backend matrix: fox sentence → PCM → Whisper coverage + cos vs CPU.
//!
//! ```bash
//! cargo run -p rlx-gepard --release --example backend_matrix --features all-backends
//! ```
//!
//! Env: `RLX_TEXT`, `RLX_DEVICES=cpu,metal,mlx,gpu`, `RLX_WHISPER_DIR`.

use std::path::PathBuf;
use std::time::Instant;

use rlx_gepard::{GepardSynthesizer, InferOpts, default_seed_for_text};
use rlx_runtime::{Device, is_available};

const DEFAULT_TEXT: &str = "The quick brown fox jumps over the lazy dog.";
const FOX_WORDS: [&str; 6] = ["quick", "brown", "fox", "jumps", "lazy", "dog"];

fn main() -> anyhow::Result<()> {
    let dir = std::env::var("RLX_GEPARD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("weights/tts/gepard"));
    anyhow::ensure!(
        dir.join("model.safetensors").is_file(),
        "missing Gepard weights — just fetch-gepard"
    );
    anyhow::ensure!(
        dir.join("nano_dec_1.89kbps.safetensors").is_file(),
        "missing nano_dec_1.89kbps.safetensors"
    );

    let text = std::env::var("RLX_TEXT").unwrap_or_else(|_| DEFAULT_TEXT.into());
    let devices = parse_devices();
    let mut whisper = load_whisper();

    println!("== Gepard cross-backend (compiled AR + NanoCodec) ==");
    println!("text: {text:?}");
    println!(
        "{:<8} {:>8} {:>7} {:>9} {:>8}",
        "backend", "ms", "peak", "whisper", "cov%"
    );

    let mut cpu_wav: Option<Vec<f32>> = None;
    let mut failed = false;

    for (dev, label) in devices {
        if dev != Device::Cpu && !is_available(dev) {
            println!(
                "{label:<8} {:>8} {:>7} {:>9} {:>8}  (n/a)",
                "-", "-", "-", "-"
            );
            continue;
        }
        let t0 = Instant::now();
        let synth = match GepardSynthesizer::open(&dir, dev) {
            Ok(s) => s.with_opts(InferOpts {
                // Seed 54 is the validated fox sampler (greedy collapses to noise).
                seed: default_seed_for_text(&text),
                ..Default::default()
            }),
            Err(e) => {
                println!("{label:<8}  load err: {e}");
                failed = true;
                continue;
            }
        };
        let pcm = match synth.synthesize(&text, "") {
            Ok(p) => p,
            Err(e) => {
                println!("{label:<8}  synth err: {e}");
                failed = true;
                continue;
            }
        };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let peak = pcm.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let cos = cpu_wav
            .as_ref()
            .map(|ref_wav| cosine(ref_wav, &pcm))
            .unwrap_or(1.0);
        if cpu_wav.is_none() {
            cpu_wav = Some(pcm.clone());
        }
        let (cov, hyp) = match whisper.as_mut() {
            Some(w) => coverage(w, &pcm, &text),
            None => (0.0, String::from("(no whisper)")),
        };
        let fox_hits = fox_hits(&hyp);
        if fox_hits < FOX_WORDS.len() {
            failed = true;
        }
        if label != "cpu" && cos < 0.999 {
            failed = true;
        }
        println!(
            "{label:<8} {ms:>8.0} {peak:>7.3} {:>9} {cov:>7.0}  cos={cos:.5} fox={fox_hits}/6 {:?}",
            if hyp.is_empty() { "-" } else { "ok" },
            trim(&hyp, 40)
        );
    }

    if failed {
        anyhow::bail!("one or more backends failed whisper/cosine gate");
    }
    Ok(())
}

fn fox_hits(hyp: &str) -> usize {
    let lower = hyp.to_ascii_lowercase();
    FOX_WORDS.iter().filter(|w| lower.contains(*w)).count()
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
            let c = PathBuf::from(".cache/whisper-tiny");
            c.join("model.safetensors").is_file().then_some(c)
        })?;
    rlx_whisper::WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .build()
        .ok()
}

fn resample(pcm: &[f32]) -> Vec<f32> {
    let n_out = (pcm.len() as u64 * 16000 / 22050) as usize;
    let scale = 22050.0 / 16000.0;
    (0..n_out)
        .map(|i| {
            let src = i as f64 * scale;
            let i0 = src.floor() as usize;
            let i1 = (i0 + 1).min(pcm.len() - 1);
            let t = (src - i0 as f64) as f32;
            pcm[i0] * (1.0 - t) + pcm[i1] * t
        })
        .collect()
}

fn coverage(w: &mut rlx_whisper::WhisperRunner, pcm: &[f32], reference: &str) -> (f64, String) {
    let hyp = w.transcribe_greedy(&resample(pcm)).unwrap_or_default();
    let ref_words: Vec<String> = reference
        .split_whitespace()
        .map(|s| {
            s.trim_matches(|c: char| !c.is_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|s| !s.is_empty())
        .collect();
    if ref_words.is_empty() {
        return (0.0, hyp);
    }
    let lower = hyp.to_ascii_lowercase();
    let hits = ref_words
        .iter()
        .filter(|w| lower.contains(w.as_str()))
        .count();
    (hits as f64 / ref_words.len() as f64, hyp)
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
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / na.sqrt() / nb.sqrt()
}
