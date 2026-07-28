// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Cross-backend parity + performance matrix for Supertonic-3, plus a
//! rlx-native-vs-onnxruntime ("the original") comparison.
//!
//! (b) For each runnable backend (CPU / Metal / MLX / wgpu / CoreML-ANE) it
//!     synthesizes the same text, measures RTF, computes the cosine similarity
//!     of the waveform vs the CPU reference, and (if a Whisper model is present)
//!     transcribes for a per-backend intelligibility coverage.
//! (c) With `--features onnx` it also times the original ONNX Runtime path
//!     (`synthesize_ort`, CPU EP) and reports the rlx-vs-ort speedup per backend.
//!
//! Run (Apple Silicon):
//!   cargo run --release --example backend_matrix \
//!     --features "metal,mlx,gpu,coreml,onnx" -- weights/tts/supertonic-3
//!
//! CUDA/ROCm are compile-only on macOS — add `cuda`/`rocm` on a Linux+GPU host.

use std::path::PathBuf;
use std::time::Instant;

use rlx_runtime::{Device, is_available};
use rlx_supertonic::{InferOpts, Supertonic, Voice};

const TEXT: &str = "The quick brown fox jumps over the lazy dog near the river bank.";
const WARMUP: usize = 1;
fn iters() -> usize {
    std::env::var("RLX_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
}
const _ITERS_DOC: usize = 3;

fn backends() -> Vec<(Device, &'static str)> {
    let mut v = vec![
        (Device::Cpu, "CPU"),
        (Device::Metal, "Metal"),
        (Device::Mlx, "MLX"),
        (Device::Gpu, "wgpu"),
        (Device::Cuda, "CUDA"),
        (Device::Vulkan, "vulkan"),
        (Device::Ane, "CoreML"),
    ];
    if let Ok(raw) = std::env::var("RLX_DEVICES") {
        let want: Vec<_> = raw
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .collect();
        v.retain(|(d, _)| match d {
            Device::Cpu => want.iter().any(|s| s == "cpu"),
            Device::Metal => want.iter().any(|s| s == "metal"),
            Device::Mlx => want.iter().any(|s| s == "mlx"),
            Device::Gpu => want.iter().any(|s| s == "gpu" || s == "wgpu"),
            Device::Cuda => want.iter().any(|s| s == "cuda"),
            Device::Vulkan => want.iter().any(|s| s == "vulkan"),
            Device::Ane => want.iter().any(|s| s == "ane" || s == "coreml"),
            _ => false,
        });
    }
    v
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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .or_else(|| {
            let p =
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/supertonic-3");
            p.join("onnx/tts.json").is_file().then_some(p)
        })
        .filter(|p| p.join("onnx/tts.json").is_file());
    let Some(dir) = dir else {
        eprintln!("usage: backend_matrix <supertonic-3 dir>  (needs onnx/tts.json)");
        return Ok(());
    };
    let voice = Voice::load(&dir.join("voice_styles/F1.json"))?;
    let opts = InferOpts::default();

    // ── CPU reference (native) ──────────────────────────────────────────────
    let cpu_tts = Supertonic::load_on(&dir, Device::Cpu)?;
    let sr = cpu_tts.sample_rate();
    let cpu_ref = cpu_tts.synthesize(TEXT, "en", &voice, &opts)?;
    let audio_s = cpu_ref.len() as f64 / sr as f64;
    eprintln!(
        "text: {TEXT:?}\naudio: {audio_s:.2}s @ {sr} Hz  ({} samples)\n",
        cpu_ref.len()
    );

    // Optional Whisper for intelligibility coverage.
    let mut whisper = load_whisper();

    println!("== (b) rlx native cross-backend parity + RTF ==");
    println!(
        "{:<8} {:>8} {:>7} {:>10} {:>9}  note",
        "backend", "rtf", "ms", "cos_vs_cpu", "whisper"
    );
    #[cfg(feature = "onnx")]
    let mut native_cpu_ms = f64::NAN;
    for (dev, label) in backends() {
        if dev != Device::Cpu && !is_available(dev) {
            println!(
                "{label:<8} {:>8} {:>7} {:>10} {:>9}  not available/compiled",
                "-", "-", "-", "-"
            );
            continue;
        }
        let tts = match Supertonic::load_on(&dir, dev) {
            Ok(t) => t,
            Err(e) => {
                println!(
                    "{label:<8} {:>8} {:>7} {:>10} {:>9}  load err: {}",
                    "-",
                    "-",
                    "-",
                    "-",
                    short(&e.to_string())
                );
                continue;
            }
        };
        // warmup
        let mut ok = true;
        let mut wav = Vec::new();
        for _ in 0..WARMUP {
            match tts.synthesize(TEXT, "en", &voice, &opts) {
                Ok(w) => wav = w,
                Err(e) => {
                    println!(
                        "{label:<8} {:>8} {:>7} {:>10} {:>9}  run err: {}",
                        "-",
                        "-",
                        "-",
                        "-",
                        short(&e.to_string())
                    );
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let mut times = Vec::new();
        for _ in 0..iters() {
            let t0 = Instant::now();
            wav = tts.synthesize(TEXT, "en", &voice, &opts)?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let ms = median(times);
        #[cfg(feature = "onnx")]
        if dev == Device::Cpu {
            native_cpu_ms = ms;
        }
        let rtf = audio_s / (ms / 1000.0);
        let cos = cosine(&cpu_ref, &wav);
        let whisp = whisper
            .as_mut()
            .map(|w| coverage(w, &wav, sr))
            .map(|c| format!("{c:.2}"))
            .unwrap_or_else(|| "n/a".into());
        println!("{label:<8} {rtf:>7.1}x {ms:>7.0} {cos:>10.5} {whisp:>9}  ");
    }

    // ── (c) rlx native vs onnxruntime (the original) ────────────────────────
    println!("\n== (c) rlx-native vs onnxruntime (original) ==");
    #[cfg(feature = "onnx")]
    {
        let ort_tts = Supertonic::load_on(&dir, Device::Cpu)?;
        let _ = ort_tts.synthesize_ort(TEXT, "en", &voice, &opts)?; // warmup
        let mut times = Vec::new();
        for _ in 0..iters() {
            let t0 = Instant::now();
            let _ = ort_tts.synthesize_ort(TEXT, "en", &voice, &opts)?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let ort_ms = median(times);
        println!(
            "onnxruntime (CPU EP, original): {ort_ms:.0} ms  ({:.1}x RT)",
            audio_s / (ort_ms / 1000.0)
        );
        if native_cpu_ms.is_finite() {
            println!(
                "rlx native CPU:                 {native_cpu_ms:.0} ms  → {:.2}x vs ort",
                ort_ms / native_cpu_ms
            );
        }
        println!(
            "(Metal/MLX/wgpu speedups vs ort = ort_ms / that backend's ms from the table above.)"
        );
    }
    #[cfg(not(feature = "onnx"))]
    {
        println!("build with --features onnx to time the original ONNX Runtime path.");
    }
    Ok(())
}

fn short(s: &str) -> String {
    s.chars().take(60).collect()
}

// ── Whisper intelligibility (optional) ──────────────────────────────────────
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};

fn load_whisper() -> Option<WhisperRunner> {
    let d = std::env::var("RLX_WHISPER_DIR")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            let c = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache");
            ["whisper-base.en", "whisper-tiny.en", "whisper-tiny"]
                .into_iter()
                .map(|n| c.join(n))
                .find(|p| p.join("model.safetensors").is_file())
        })?;
    WhisperRunner::builder()
        .weights(d.join("model.safetensors"))
        .config_path(d.join("config.json"))
        .tokenizer_path(d.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .ok()
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

fn wordset(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(str::to_string)
        .collect()
}

fn coverage(w: &mut WhisperRunner, wav: &[f32], sr: u32) -> f64 {
    let pcm = resample(wav, sr, WHISPER_RATE as u32);
    let Ok(t) = w.transcribe_greedy(&pcm) else {
        return 0.0;
    };
    let refs = wordset(TEXT);
    let heard = wordset(&t);
    let hits = refs
        .iter()
        .filter(|x| heard.iter().any(|h| h == *x || h.contains(x.as_str())))
        .count();
    hits as f64 / refs.len().max(1) as f64
}
