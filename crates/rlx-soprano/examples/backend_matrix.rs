// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Soprano 1.1 cross-backend matrix (native RLX).
//!
//! ```text
//! cargo run -p rlx-soprano --release --example backend_matrix \
//!   --features apple-silicon
//! ```
//!
//! Env: `RLX_SOPRANO_DIR`, `RLX_TEXT`, `RLX_MAX_TOKENS`, `RLX_DEVICES`,
//! `RLX_WHISPER_DIR`, `RLX_GREEDY`.

use std::path::PathBuf;
use std::time::Instant;

use rlx_runtime::{Device, is_available};
use rlx_soprano::{DEFAULT_LOCAL_DIR, InferOpts, NativeSoprano, peak_amplitude};

const DEFAULT_TEXT: &str = "The quick brown fox jumps over the lazy dog.";

fn soprano_loose_or_pack(dir: &std::path::Path) -> bool {
    dir.join("soprano.rlxp").is_file()
        || dir.join("soprano.gguf").is_file()
        || dir
            .join("graphs/soprano_backbone_kv_fp32.rlxp")
            .is_file()
        || dir
            .join("onnx/soprano_backbone_kv_fp32.onnx")
            .is_file()
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::var("RLX_SOPRANO_DIR")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            let p = PathBuf::from(DEFAULT_LOCAL_DIR);
            soprano_loose_or_pack(&p).then_some(p)
        })
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/soprano")
        });
    let text = std::env::var("RLX_TEXT").unwrap_or_else(|_| DEFAULT_TEXT.to_string());
    let max_tokens: usize = std::env::var("RLX_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(96);

    if !soprano_loose_or_pack(&dir) {
        anyhow::bail!("missing soprano.rlxp / nested graphs under {}", dir.display());
    }

    let devices = parse_devices();
    eprintln!("text: {text:?}");
    eprintln!(
        "devices: {:?}",
        devices.iter().map(|(_, l)| *l).collect::<Vec<_>>()
    );

    let opts = InferOpts {
        max_new_tokens: max_tokens,
        // Default greedy for cross-backend parity; set RLX_SAMPLE=1 for sampling.
        greedy: std::env::var_os("RLX_SAMPLE").is_none()
            || std::env::var_os("RLX_GREEDY").is_some(),
        temperature: std::env::var("RLX_TEMP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.3),
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

    let mut whisper = load_whisper();
    let mut cpu_ref: Option<Vec<f32>> = None;

    println!("== Soprano native cross-backend ==");
    println!(
        "{:<8} {:>8} {:>7} {:>10} {:>9} {:>8}",
        "backend", "rtf", "ms", "cos_vs_cpu", "whisper", "peak"
    );

    for (dev, label) in devices {
        if dev != Device::Cpu && !is_available(dev) {
            println!(
                "{label:<8} {:>8} {:>7} {:>10} {:>9} {:>8}  (n/a)",
                "-", "-", "-", "-", "-"
            );
            continue;
        }
        let t0 = Instant::now();
        let model = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            NativeSoprano::open(&dir, dev)
        })) {
            Ok(Ok(m)) => m,
            Ok(Err(e)) => {
                println!("{label:<8}  load err: {}", short(&e.to_string()));
                continue;
            }
            Err(_) => {
                println!("{label:<8}  load panic");
                continue;
            }
        };
        let pcm = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model.synthesize(&text, &opts)
        })) {
            Ok(Ok(p)) => p,
            Ok(Err(e)) => {
                println!("{label:<8}  run err: {}", short(&e.to_string()));
                continue;
            }
            Err(_) => {
                println!("{label:<8}  run panic");
                continue;
            }
        };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let audio_s = pcm.len() as f64 / model.sample_rate() as f64;
        let rtf = if ms > 0.0 {
            audio_s / (ms / 1000.0)
        } else {
            0.0
        };
        let peak = peak_amplitude(&pcm);
        let cos = if let Some(ref cpu) = cpu_ref {
            format!("{:.5}", cosine(cpu, &pcm))
        } else {
            cpu_ref = Some(pcm.clone());
            "ref".into()
        };
        let (whisp, transcript) = match whisper.as_mut() {
            Some(w) => {
                let (cov, t) = coverage(w, &pcm, model.sample_rate(), &text);
                (format!("{cov:.2}"), t)
            }
            None => ("n/a".into(), String::new()),
        };
        println!("{label:<8} {rtf:>7.2}x {ms:>7.0} {cos:>10} {whisp:>9} {peak:>8.3}");
        if !transcript.is_empty() {
            println!("         in : {text}");
            println!("         out: {transcript}");
        }
        let out = PathBuf::from(format!("/tmp/soprano_matrix_{label}.wav"));
        let _ = NativeSoprano::write_wav(&pcm, &out, model.sample_rate());
    }
    Ok(())
}

fn parse_devices() -> Vec<(Device, &'static str)> {
    let all = vec![
        (Device::Cpu, "CPU"),
        (Device::Metal, "Metal"),
        (Device::Mlx, "MLX"),
        (Device::Gpu, "wgpu"),
        (Device::Ane, "CoreML"),
        (Device::Cuda, "CUDA"),
        (Device::Vulkan, "vulkan"),
    ];
    if let Ok(s) = std::env::var("RLX_DEVICES") {
        let want: Vec<&str> = s
            .split(',')
            .map(str::trim)
            .filter(|x| !x.is_empty())
            .collect();
        return all
            .into_iter()
            .filter(|(_, l)| {
                want.iter().any(|w| {
                    w.eq_ignore_ascii_case(l)
                        || (*w == "wgpu" && *l == "wgpu")
                        || (*w == "coreml" && *l == "CoreML")
                })
            })
            .collect();
    }
    all.into_iter()
        .filter(|(d, _)| *d == Device::Cpu || is_available(*d))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

fn short(s: &str) -> String {
    s.chars().take(72).collect()
}

fn load_whisper() -> Option<rlx_whisper::WhisperRunner> {
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
    rlx_whisper::WhisperRunner::builder()
        .weights(d.join("model.safetensors"))
        .config_path(d.join("config.json"))
        .tokenizer_path(d.join("tokenizer.json"))
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
        .filter(|x| x.len() >= 2)
        .map(str::to_string)
        .collect();
    let got = t.to_lowercase();
    let got_toks: Vec<&str> = got
        .split(|c: char| !c.is_alphanumeric())
        .filter(|x| x.len() >= 2)
        .collect();
    if want.is_empty() {
        let ok = !got_toks.is_empty();
        return (if ok { 1.0 } else { 0.0 }, t);
    }
    let hit = want
        .iter()
        .filter(|w| word_near_hit(w, &got, &got_toks))
        .count();
    (hit as f64 / want.len() as f64, t)
}

fn word_near_hit(want: &str, got: &str, got_toks: &[&str]) -> bool {
    if got.contains(want) {
        return true;
    }
    let max_ed = if want.len() >= 7 {
        2
    } else if want.len() >= 5 {
        1
    } else {
        0
    };
    if max_ed == 0 {
        return false;
    }
    got_toks.iter().any(|tok| {
        (tok.len() as isize - want.len() as isize).unsigned_abs() <= max_ed
            && edit_distance(want, tok) <= max_ed
    })
}

fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let (n, m) = (a.len(), b.len());
    let mut prev = (0..=m).collect::<Vec<_>>();
    let mut cur = vec![0; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
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
