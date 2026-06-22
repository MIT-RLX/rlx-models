// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Compare production (low-RAM) vs legacy parity/full-graph load + infer timings.
//!
//! ```bash
//! scripts/bench_kitten_native_alloc.sh
//! # or single mode:
//! KITTEN_RLX_INFER=production KITTEN_RLX_TIMING=1 \
//!   cargo run -p rlx-kittentts --features native-fast,metal --release \
//!   --example native_alloc_bench -- --mode production
//! ```

#![cfg(feature = "native-fast")]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_kittentts::phrase_fixtures::LONG_IPA;
use rlx_kittentts::{Device, KittenTTS, assets, infer_opts, ipa_to_ids};
use rlx_runtime::is_available;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BenchMode {
    Production,
    LegacyFull,
}

impl BenchMode {
    fn name(self) -> &'static str {
        match self {
            Self::Production => "production",
            Self::LegacyFull => "legacy_full",
        }
    }

    fn apply_env(self) {
        unsafe {
            match self {
                Self::Production => {
                    std::env::set_var("KITTEN_RLX_INFER", "production");
                    std::env::remove_var("KITTEN_RLX_FULL_GRAPH");
                    std::env::remove_var("KITTEN_RLX_PREWARM");
                    std::env::remove_var("KITTEN_RLX_SKIP_PREWARM");
                    std::env::remove_var("KITTEN_RLX_SEQ_CACHE_CAPACITY");
                }
                Self::LegacyFull => {
                    std::env::set_var("KITTEN_RLX_INFER", "parity");
                    std::env::set_var("KITTEN_RLX_FULL_GRAPH", "1");
                    std::env::set_var("KITTEN_RLX_PREWARM", "1");
                    std::env::set_var("KITTEN_RLX_SEQ_CACHE_CAPACITY", "16");
                    std::env::remove_var("KITTEN_RLX_SKIP_PREWARM");
                }
            }
        }
    }

    fn compile_dims(self, token_len: usize) -> (usize, usize) {
        match self {
            Self::Production => infer_opts::recommended_native_compile_opts(token_len),
            Self::LegacyFull => (128, 367_200),
        }
    }
}

struct Args {
    mode: Option<BenchMode>,
    long: bool,
    device: Device,
}

fn parse_args() -> Result<Args> {
    let mut mode = None;
    let mut long = false;
    let mut device = None;
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--mode" => {
                mode = Some(match raw.get(i + 1).map(String::as_str) {
                    Some("production" | "prod") => BenchMode::Production,
                    Some("legacy" | "legacy_full" | "parity") => BenchMode::LegacyFull,
                    Some(other) => anyhow::bail!("unknown --mode {other}"),
                    None => anyhow::bail!("--mode requires production|legacy_full"),
                });
                i += 2;
            }
            "--long" => {
                long = true;
                i += 1;
            }
            "--device" => {
                let d = raw.get(i + 1).context("--device requires cpu|metal")?;
                device = Some(match d.as_str() {
                    "cpu" => Device::Cpu,
                    "metal" => Device::Metal,
                    other => anyhow::bail!("unknown device {other}"),
                });
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: native_alloc_bench [--mode production|legacy_full] [--long] [--device cpu|metal]"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unexpected arg: {other}"),
        }
    }
    let device = device.unwrap_or_else(|| {
        if is_available(Device::Metal) {
            Device::Metal
        } else {
            Device::Cpu
        }
    });
    Ok(Args { mode, long, device })
}

fn voices_npz() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("KITTEN_VOICES_NPZ") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
    }
    assets::default_model_dir()
        .context("model dir")?
        .join("voices.npz")
        .canonicalize()
        .context("voices.npz")
}

fn fresh_aot_cache(mode: BenchMode) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "kitten_bench_{}_{}",
        mode.name(),
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).ok();
    unsafe {
        std::env::set_var("KITTEN_RLX_AOT_CACHE", &dir);
    }
    dir
}

fn run_mode(mode: BenchMode, ipa: &str, device: Device) -> Result<()> {
    mode.apply_env();
    let _aot = fresh_aot_cache(mode);
    unsafe {
        std::env::set_var("KITTEN_RLX_TIMING", "1");
        // Waveform-only split compile hits a fusion broadcast bug on narrow seq without this.
        if std::env::var("KITTEN_RLX_SKIP_FUSION").is_err() {
            std::env::set_var("KITTEN_RLX_SKIP_FUSION", "1");
        }
    }

    let weights = assets::default_native_weights_dir().context("native weights dir")?;
    let voices = voices_npz()?;
    let ort_model = assets::default_model_dir().ok().and_then(|d| {
        let onnx = d.join("kitten_tts_mini_v0_8.onnx");
        if onnx.is_file() { Some(onnx) } else { None }
    });
    let token_len = ipa_to_ids(ipa).len();
    let (seq_len, max_wave) = mode.compile_dims(token_len);
    let chunks = infer_opts::chunk_plan(ipa_to_ids(ipa).as_slice(), seq_len);

    eprintln!(
        "[bench] mode={} device={device:?} token_len={token_len} compile_seq={seq_len} max_wave={max_wave} chunks={}",
        mode.name(),
        chunks.len()
    );

    let load_t0 = Instant::now();
    let tts = KittenTTS::load_native_with_ort(
        &weights,
        &voices,
        Default::default(),
        Default::default(),
        device,
        seq_len,
        max_wave,
        ort_model.as_deref(),
    )
    .with_context(|| format!("load_native mode={}", mode.name()))?;
    let load_secs = load_t0.elapsed().as_secs_f64();

    let voice = tts.voice_names().first().context("voice")?.clone();

    let cold_t0 = Instant::now();
    let audio1 = tts
        .generate_from_ipa(ipa, &voice, 1.0, 6)
        .with_context(|| format!("cold infer mode={}", mode.name()))?;
    let cold_secs = cold_t0.elapsed().as_secs_f64();

    let warm_t0 = Instant::now();
    let audio2 = tts
        .generate_from_ipa(ipa, &voice, 1.0, 6)
        .with_context(|| format!("warm infer mode={}", mode.name()))?;
    let warm_secs = warm_t0.elapsed().as_secs_f64();

    let peak1 = audio1.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let peak2 = audio2.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(peak1 >= 1e-4 && peak2 >= 1e-4, "inaudible output");

    // Machine-readable summary for scripts (`/usr/bin/time -l` adds peak RSS).
    println!(
        "[bench-result] mode={} phrase={} load_s={load_secs:.3} cold_s={cold_secs:.3} warm_s={warm_secs:.3} samples={} compile_seq={seq_len} max_wave={max_wave}",
        mode.name(),
        if ipa == LONG_IPA { "long" } else { "hello" },
        audio2.len()
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let ipa = if args.long { LONG_IPA } else { "həˈloʊ" };
    let modes: Vec<BenchMode> = match args.mode {
        Some(m) => vec![m],
        None => vec![BenchMode::Production, BenchMode::LegacyFull],
    };
    for mode in modes {
        run_mode(mode, ipa, args.device)?;
    }
    Ok(())
}
