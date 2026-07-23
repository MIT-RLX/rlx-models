// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Native KittenTTS RTF matrix: short + long IPA × every available RLX backend.
//!
//! ```text
//! cargo test -p rlx-kittentts --release \
//!   --features native,apple-silicon \
//!   --test native_backend_bench -- --nocapture --test-threads=1
//!
//! # NVIDIA (MSI): --features native,gpu,cuda,vulkan
//! ```
//!
//! Env: `KITTEN_TTS_BENCH_WARM` (default 3), `KITTEN_TTS_BENCH_SKIP_WHISPER=1`.

#![cfg(feature = "native")]

mod support;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rlx_kittentts::{Device, KittenTTS, SAMPLE_RATE as TTS_RATE, assets, infer_opts};
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};
use support::{LONG_IPA, resample_linear, whisper_asr_dir};

const SHORT_IPA: &str = "həˈloʊ";
const SHORT_LABEL: &str = "short";
const LONG_LABEL: &str = "long";

fn warm_runs() -> usize {
    std::env::var("KITTEN_TTS_BENCH_WARM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
        .max(1)
}

fn skip_whisper() -> bool {
    std::env::var("KITTEN_TTS_BENCH_SKIP_WHISPER").is_ok_and(|v| {
        v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
    })
}

/// Optional filter: `KITTEN_TTS_BENCH_DEVICES=cpu,cuda,gpu` (comma-separated).
fn device_allowlist() -> Option<Vec<String>> {
    let raw = std::env::var("KITTEN_TTS_BENCH_DEVICES").ok()?;
    let list: Vec<String> = raw
        .split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    if list.is_empty() {
        None
    } else {
        Some(list)
    }
}

fn device_allowed(dev: Device, allow: &Option<Vec<String>>) -> bool {
    let Some(list) = allow else {
        return true;
    };
    let name = format!("{dev:?}").to_ascii_lowercase();
    list.iter().any(|a| a == &name)
}

fn voices_npz() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("KITTEN_VOICES_NPZ") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    assets::default_model_dir()
        .ok()
        .and_then(|dir| assets::ModelLayout::resolve(&dir).ok())
        .map(|l| l.voices)
        .filter(|p| p.is_file())
}

fn whisper_runner(dir: &Path) -> WhisperRunner {
    WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .expect("whisper runner")
}

fn transcribe(pcm_24k: &[f32], whisper_dir: &Path) -> String {
    let pcm_16k = resample_linear(pcm_24k, TTS_RATE, WHISPER_RATE as u32);
    let mut whisper = whisper_runner(whisper_dir);
    whisper
        .transcribe_greedy(&pcm_16k)
        .map(|t| t.trim().to_string())
        .unwrap_or_else(|e| format!("<whisper error: {e}>"))
}

fn pick_voice(tts: &KittenTTS) -> String {
    tts.voice_names()
        .iter()
        .find(|n| n.eq_ignore_ascii_case("Jasper"))
        .cloned()
        .or_else(|| {
            tts.voice_names()
                .iter()
                .find(|n| n.contains("expr-voice-2-m"))
                .cloned()
        })
        .or_else(|| tts.voice_names().first().cloned())
        .unwrap_or_default()
}

#[derive(Clone)]
struct Row {
    phrase: &'static str,
    backend: String,
    status: String,
    compile_s: f64,
    warm_ms: f64,
    audio_s: f64,
    /// Wall / audio — lower is better; &lt; 1.0 is faster than real-time.
    rtf: f64,
    peak: f32,
    whisper: String,
}

fn run_backend(
    phrase: &'static str,
    ipa: &str,
    dev: Device,
    weights: &Path,
    voices: &Path,
    whisper_dir: Option<&Path>,
) -> Row {
    let token_len = rlx_kittentts::ipa_to_ids(ipa).len().max(1);
    let (seq_len, max_wave) = infer_opts::recommended_native_compile_opts(token_len);
    let label = format!("{dev:?}");

    let load_t0 = Instant::now();
    let tts = match KittenTTS::load_native(
        weights,
        voices,
        Default::default(),
        Default::default(),
        dev,
        seq_len,
        max_wave,
    ) {
        Ok(t) => t,
        Err(e) => {
            return Row {
                phrase,
                backend: label,
                status: format!("load: {e}"),
                compile_s: 0.0,
                warm_ms: 0.0,
                audio_s: 0.0,
                rtf: f64::NAN,
                peak: 0.0,
                whisper: String::new(),
            };
        }
    };
    let voice = pick_voice(&tts);

    if let Err(e) = tts.generate_from_ipa(ipa, &voice, 1.0, 6) {
        return Row {
            phrase,
            backend: label,
            status: format!("warmup: {e:#}"),
            compile_s: load_t0.elapsed().as_secs_f64(),
            warm_ms: 0.0,
            audio_s: 0.0,
            rtf: f64::NAN,
            peak: 0.0,
            whisper: String::new(),
        };
    }
    let compile_s = load_t0.elapsed().as_secs_f64();

    let mut best = f64::MAX;
    let mut audio = Vec::new();
    for _ in 0..warm_runs() {
        let t0 = Instant::now();
        match tts.generate_from_ipa(ipa, &voice, 1.0, 6) {
            Ok(a) => {
                best = best.min(t0.elapsed().as_secs_f64());
                audio = a;
            }
            Err(e) => {
                return Row {
                    phrase,
                    backend: label,
                    status: format!("infer: {e:#}"),
                    compile_s,
                    warm_ms: 0.0,
                    audio_s: 0.0,
                    rtf: f64::NAN,
                    peak: 0.0,
                    whisper: String::new(),
                };
            }
        }
    }
    if audio.is_empty() {
        return Row {
            phrase,
            backend: label,
            status: "empty audio".into(),
            compile_s,
            warm_ms: 0.0,
            audio_s: 0.0,
            rtf: f64::NAN,
            peak: 0.0,
            whisper: String::new(),
        };
    }

    let audio_s = audio.len() as f64 / TTS_RATE as f64;
    let rtf = best / audio_s.max(1e-9);
    let peak = audio.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
    let whisper = match whisper_dir {
        Some(d) if !skip_whisper() => transcribe(&audio, d),
        _ => "—".into(),
    };

    Row {
        phrase,
        backend: label,
        status: "ok".into(),
        compile_s,
        warm_ms: best * 1000.0,
        audio_s,
        rtf,
        peak,
        whisper,
    }
}

fn print_table(rows: &[Row]) {
    eprintln!(
        "{:<6} {:<8} {:>10} {:>9} {:>8} {:>7} {:>7}  {}",
        "phrase", "backend", "compile(s)", "warm(ms)", "audio(s)", "RTF", "peak", "whisper/status"
    );
    for r in rows {
        if r.status == "ok" {
            eprintln!(
                "{:<6} {:<8} {:>10.2} {:>9.1} {:>8.2} {:>7.3} {:>7.3}  {:?}",
                r.phrase, r.backend, r.compile_s, r.warm_ms, r.audio_s, r.rtf, r.peak, r.whisper
            );
        } else {
            eprintln!(
                "{:<6} {:<8} {:>10} {:>9} {:>8} {:>7} {:>7}  {}",
                r.phrase, r.backend, "—", "—", "—", "—", "—", r.status
            );
        }
    }
}

#[test]
fn native_backend_bench() {
    let Some(weights) = assets::default_native_weights_dir() else {
        eprintln!("skip: no native weights (run `just fetch-kittentts` / `just fetch-kitten-rlx-bundle`)");
        return;
    };
    let Some(voices) = voices_npz() else {
        eprintln!("skip: no voices.npz (run `just fetch-kittentts`)");
        return;
    };
    support::setup_native_smoke_env();
    unsafe {
        std::env::remove_var("KITTEN_RLX_DEBUG_DURATION");
        std::env::remove_var("KITTEN_RLX_TIMING");
    }

    let whisper_dir = if skip_whisper() {
        None
    } else {
        whisper_asr_dir()
    };
    if whisper_dir.is_none() && !skip_whisper() {
        eprintln!("note: no whisper weights — RTF only (`just fetch-whisper-base` / tiny)");
    }

    let phrases: &[(&str, &str)] = &[
        (SHORT_LABEL, SHORT_IPA),
        (LONG_LABEL, LONG_IPA),
    ];

    let candidates = [
        Device::Cpu,
        Device::Metal,
        Device::Mlx,
        Device::Gpu,
        Device::Ane,
        Device::Cuda,
        Device::Rocm,
        Device::Vulkan,
    ];

    eprintln!("\n=== KittenTTS native RLX backend RTF matrix ===");
    eprintln!(
        "sample_rate={TTS_RATE} warm_runs={} RTF=wall/audio (<1 faster than real-time)\n",
        warm_runs()
    );
    for (label, ipa) in phrases {
        let n = rlx_kittentts::ipa_to_ids(ipa).len();
        let (seq, wave) = infer_opts::recommended_native_compile_opts(n);
        eprintln!("phrase={label} tokens={n} compile=({seq},{wave}) ipa={ipa:?}");
    }
    eprintln!();

    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let allow = device_allowlist();
    let mut rows = Vec::new();
    for (phrase, ipa) in phrases {
        for &dev in &candidates {
            if !device_allowed(dev, &allow) {
                continue;
            }
            if dev != Device::Cpu && !rlx_runtime::is_available(dev) {
                rows.push(Row {
                    phrase,
                    backend: format!("{dev:?}"),
                    status: "unavailable".into(),
                    compile_s: 0.0,
                    warm_ms: 0.0,
                    audio_s: 0.0,
                    rtf: f64::NAN,
                    peak: 0.0,
                    whisper: String::new(),
                });
                continue;
            }
            // Fresh AOT dir per backend so a crashy device cannot poison the next.
            let aot = std::env::temp_dir().join(format!(
                "kitten_bench_aot_{}_{phrase}_{}",
                std::process::id(),
                format!("{dev:?}").to_ascii_lowercase()
            ));
            let _ = std::fs::create_dir_all(&aot);
            unsafe {
                std::env::set_var("KITTEN_RLX_AOT_CACHE", &aot);
            }
            eprintln!("\n--- {phrase} / {dev:?} ---");
            let w = weights.as_path();
            let v = voices.as_path();
            let wdir = whisper_dir.as_deref();
            let res = catch_unwind(AssertUnwindSafe(|| {
                run_backend(phrase, ipa, dev, w, v, wdir)
            }));
            match res {
                Ok(row) => {
                    // Print incrementally so a later SIGSEGV still leaves a partial table.
                    if row.status == "ok" {
                        eprintln!(
                            "{:<6} {:<8} {:>10.2} {:>9.1} {:>8.2} {:>7.3} {:>7.3}  {:?}",
                            row.phrase,
                            row.backend,
                            row.compile_s,
                            row.warm_ms,
                            row.audio_s,
                            row.rtf,
                            row.peak,
                            row.whisper
                        );
                    } else {
                        eprintln!(
                            "{:<6} {:<8} {:>10} {:>9} {:>8} {:>7} {:>7}  {}",
                            row.phrase, row.backend, "—", "—", "—", "—", "—", row.status
                        );
                    }
                    rows.push(row);
                }
                Err(payload) => {
                    let msg = payload
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "panicked".to_string());
                    let first = msg.lines().next().unwrap_or(&msg).to_string();
                    eprintln!(
                        "{:<6} {:<8} {:>10} {:>9} {:>8} {:>7} {:>7}  PANIC: {first}",
                        phrase,
                        format!("{dev:?}"),
                        "—",
                        "—",
                        "—",
                        "—",
                        "—"
                    );
                    rows.push(Row {
                        phrase,
                        backend: format!("{dev:?}"),
                        status: format!("PANIC: {first}"),
                        compile_s: 0.0,
                        warm_ms: 0.0,
                        audio_s: 0.0,
                        rtf: f64::NAN,
                        peak: 0.0,
                        whisper: String::new(),
                    });
                }
            }
        }
    }

    std::panic::set_hook(default_hook);
    eprintln!("\n=== summary ===");
    print_table(&rows);
    eprintln!();
}
