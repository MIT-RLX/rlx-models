// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! MetaVoice: CPU first/second-stage once, EnCodec + Whisper per backend.
//!
//! ```text
//! cargo run -p rlx-metavoice --release --example backend_matrix \
//!   --features "metal,mlx,gpu,coreml"
//! ```
//!
//! Env: `RLX_METAVOICE_DIR`, `RLX_ENCODEC_PATH`, `RLX_WHISPER_DIR`, `RLX_TEXT`,
//! `RLX_MAX_TOKENS`, `RLX_DEVICES=cpu,metal,mlx,wgpu,coreml,cuda`,
//! `RLX_GREEDY=1` (default), `RLX_SAMPLE=1` for top-p, `RLX_SEED`.

use std::path::PathBuf;
use std::time::Instant;

use rlx_metavoice::{
    DEFAULT_ENCODEC_PATH, DEFAULT_LOCAL_DIR, FOX_WORDS, InferOpts, MetaVoice, peak_amplitude,
};
use rlx_runtime::{Device, is_available};

const DEFAULT_TEXT: &str = "The quick brown fox jumps over the lazy dog.";

fn main() -> anyhow::Result<()> {
    let dir = std::env::var("RLX_METAVOICE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_LOCAL_DIR));
    let enc = std::env::var("RLX_ENCODEC_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_ENCODEC_PATH));
    let text = std::env::var("RLX_TEXT").unwrap_or_else(|_| DEFAULT_TEXT.to_string());
    let max_tokens: usize = std::env::var("RLX_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(864);
    let ref_wav = std::env::var("RLX_REFERENCE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dir.join("bria_16k.wav"));

    if !dir.join("first_stage.safetensors").is_file() {
        anyhow::bail!("missing {}", dir.join("first_stage.safetensors").display());
    }
    if !enc.is_file() {
        anyhow::bail!("missing EnCodec {}", enc.display());
    }
    if !ref_wav.is_file() {
        anyhow::bail!(
            "missing speaker reference {} (required for intelligibility)",
            ref_wav.display()
        );
    }

    let devices = parse_devices();
    eprintln!("text: {text:?}");
    eprintln!(
        "devices: {:?}",
        devices.iter().map(|(_, l)| *l).collect::<Vec<_>>()
    );

    let cache = std::env::var("RLX_CODES_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/metavoice_codes.json"));
    let t0 = Instant::now();
    let cpu = MetaVoice::open_with_encodec(&dir, &enc, Device::Cpu)?;
    // Default greedy; set RLX_SAMPLE=1 for top-p (seed 1337).
    let greedy = std::env::var_os("RLX_SAMPLE").is_none();
    let opts = InferOpts {
        max_new_tokens: max_tokens,
        greedy,
        temperature: std::env::var("RLX_TEMP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0),
        top_p: std::env::var("RLX_TOP_P")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.95),
        seed: std::env::var("RLX_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1337),
        ..Default::default()
    };
    eprintln!(
        "[opts] greedy={} max_tokens={} seed={} guidance={}",
        opts.greedy, opts.max_new_tokens, opts.seed, opts.guidance_scale
    );
    let codes = if cache.is_file() && std::env::var_os("RLX_FORCE_RESYNTH").is_none() {
        eprintln!("[cache] loading codes from {}", cache.display());
        let raw = std::fs::read_to_string(&cache)?;
        serde_json::from_str::<Vec<Vec<u32>>>(&raw)?
    } else {
        eprintln!("[spk] {}", ref_wav.display());
        let spk = cpu.embed_reference(&ref_wav)?;
        eprintln!("[load cpu {:?}] generating tokens…", t0.elapsed());
        let t1 = Instant::now();
        let tokens = cpu.generate_tokens(&text, &spk, &opts)?;
        let codes = cpu.tokens_to_codes(&text, &tokens, &spk)?;
        eprintln!(
            "[first+second {:?}] tokens={} frames={} books={}",
            t1.elapsed(),
            tokens.len(),
            codes[0].len(),
            codes.len()
        );
        std::fs::write(&cache, serde_json::to_string(&codes)?)?;
        eprintln!("[cache] wrote {}", cache.display());
        codes
    };

    let mut whisper = load_whisper();
    println!(
        "{:<8} {:>8} {:>7} {:>10} {:>9}",
        "backend", "rtf", "ms", "cos_vs_cpu", "whisper"
    );

    let mut cpu_wav: Option<Vec<f32>> = None;
    let mut min_hits = FOX_WORDS.len();
    let mut last_transcript = String::new();
    for (dev, label) in devices {
        if dev != Device::Cpu && !is_available(dev) {
            println!(
                "{label:<8} {:>8} {:>7} {:>10} {:>9}  (n/a)",
                "-", "-", "-", "-"
            );
            continue;
        }
        let tts = match MetaVoice::open_with_encodec(&dir, &enc, dev) {
            Ok(t) => t,
            Err(e) => {
                println!("{label:<8}  load err: {}", short(&e.to_string()));
                continue;
            }
        };
        let t_run = Instant::now();
        let wav = match tts.decode_codes(&codes) {
            Ok(w) => w,
            Err(e) => {
                println!("{label:<8}  decode err: {}", short(&e.to_string()));
                continue;
            }
        };
        let ms = t_run.elapsed().as_secs_f64() * 1000.0;
        let audio_s = wav.len() as f64 / tts.sample_rate() as f64;
        let rtf = ms / 1000.0 / audio_s.max(1e-6);
        let cos = match &cpu_wav {
            None => {
                cpu_wav = Some(wav.clone());
                1.0
            }
            Some(ref0) => cosine(ref0, &wav),
        };
        let (hits, transcript) = whisper_fox(&mut whisper, &wav, tts.sample_rate());
        min_hits = min_hits.min(hits);
        last_transcript = transcript;
        let wcov = 100.0 * hits as f64 / FOX_WORDS.len() as f64;
        println!(
            "{label:<8} {rtf:>8.3} {ms:>7.1} {cos:>10.4} {hits}/{n} ({wcov:.0}%)  peak={peak:.3}",
            n = FOX_WORDS.len(),
            peak = peak_amplitude(&wav)
        );
        let _ = tts.write_wav(&wav, format!("/tmp/metavoice_{label}.wav"));
    }
    if min_hits < 5 {
        anyhow::bail!(
            "Whisper fox hits {min_hits}/{} < 5 (got {last_transcript:?})",
            FOX_WORDS.len()
        );
    }
    eprintln!("ok: fox Whisper ≥5/6 (min {min_hits}/{})", FOX_WORDS.len());
    Ok(())
}

fn parse_devices() -> Vec<(Device, &'static str)> {
    let raw = std::env::var("RLX_DEVICES")
        .unwrap_or_else(|_| "cpu,metal,mlx,wgpu,coreml,cuda".to_string());
    let mut out = Vec::new();
    for part in raw.split(',') {
        let p = part.trim().to_lowercase();
        let (dev, label) = match p.as_str() {
            "cpu" => (Device::Cpu, "CPU"),
            "metal" | "mps" => (Device::Metal, "Metal"),
            "mlx" | "meta" => (Device::Mlx, "MLX"),
            "wgpu" | "gpu" => (Device::Gpu, "wgpu"),
            "coreml" | "ane" => (Device::Ane, "CoreML"),
            "cuda" => (Device::Cuda, "CUDA"),
            other => {
                eprintln!("skip unknown device {other}");
                continue;
            }
        };
        out.push((dev, label));
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

fn short(s: &str) -> String {
    let s = s.replace('\n', " ");
    if s.len() > 80 {
        format!("{}…", &s[..80])
    } else {
        s
    }
}

fn load_whisper() -> Option<rlx_whisper::WhisperRunner> {
    let dir = if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        PathBuf::from(d)
    } else {
        for name in ["whisper-tiny", "whisper-tiny.en", "whisper-base.en"] {
            let p = PathBuf::from(".cache").join(name);
            if p.join("config.json").exists() {
                return build_whisper(&p);
            }
        }
        return None;
    };
    if dir.join("config.json").exists() {
        build_whisper(&dir)
    } else {
        None
    }
}

fn build_whisper(dir: &std::path::Path) -> Option<rlx_whisper::WhisperRunner> {
    rlx_whisper::WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .ok()
}

fn whisper_fox(
    w: &mut Option<rlx_whisper::WhisperRunner>,
    pcm: &[f32],
    sr: u32,
) -> (usize, String) {
    let Some(runner) = w.as_mut() else {
        return (0, String::new());
    };
    let pcm16 = resample(pcm, sr, rlx_whisper::SAMPLE_RATE as u32);
    let Ok(t) = runner.transcribe_greedy(&pcm16) else {
        return (0, String::new());
    };
    let heard = words(&t);
    let hits = FOX_WORDS
        .iter()
        .filter(|x| heard.iter().any(|h| h == *x || h.contains(*x)))
        .count();
    eprintln!("  [{sr}Hz] whisper: {t}  ({hits}/{})", FOX_WORDS.len());
    (hits, t)
}

fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let n = (x.len() as u64 * to as u64 / from as u64).max(1) as usize;
    (0..n)
        .map(|i| {
            let s = i as f64 * from as f64 / to as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let a = x[idx.min(x.len() - 1)];
            let b = x[(idx + 1).min(x.len() - 1)];
            a + (b - a) * f
        })
        .collect()
}

fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}
