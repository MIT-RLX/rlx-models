// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Pocket TTS bench: measure load time + per-prompt generation across a few
//! prompt lengths. Reports realtime factor.
//!
//! ```bash
//! cargo run -p rlx-pocket-tts --example bench --features hf-download --release
//! ```
//!
//! Env:
//! - `POCKET_TTS_VOICE` — voice name (default `alba`)
//! - `POCKET_TTS_ITERS` — iterations per prompt (default `3`)
//! - `POCKET_TTS_WARMUP` — warmup iterations (default `1`)
//! - `POCKET_TTS_SEED`   — base RNG seed (default `42`)

use std::time::Instant;

use anyhow::Result;
use rlx_pocket_tts::{GenerationOptions, TtsModel};

const PROMPTS: &[(&str, &str)] = &[
    ("short", "Hello world."),
    ("medium", "The quick brown fox jumps over the lazy dog."),
    (
        "long",
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort.",
    ),
];

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<()> {
    let voice_name = std::env::var("POCKET_TTS_VOICE").unwrap_or_else(|_| "alba".to_string());
    let iters = env_usize("POCKET_TTS_ITERS", 3);
    let warmup = env_usize("POCKET_TTS_WARMUP", 1);
    let seed = env_u64("POCKET_TTS_SEED", 42);

    eprintln!("pocket-tts bench  voice={voice_name}  warmup={warmup}  iters={iters}");

    // ── fetch ───────────────────────────────────────────────────────────────
    let t_fetch = Instant::now();
    let assets = rlx_pocket_tts::download::fetch_default_assets()?;
    let voice_path = rlx_pocket_tts::download::fetch_voice(&voice_name)?;
    let dt_fetch = t_fetch.elapsed().as_secs_f32();
    eprintln!("fetch: {dt_fetch:.2}s (mostly cache hit on warm runs)");

    // ── load ────────────────────────────────────────────────────────────────
    let t_load = Instant::now();
    let model = TtsModel::open(&assets.weights, &assets.tokenizer)?;
    let voice = model.load_voice(&voice_path)?;
    let dt_load = t_load.elapsed().as_secs_f32();
    eprintln!(
        "load:  {:.2}s  voice_frames={}  d_model={}",
        dt_load,
        voice.num_frames(),
        voice.embed_dim()
    );

    println!();
    println!(
        "{:<8}  {:>5}  {:>9}  {:>9}  {:>9}  {:>9}  {:>7}",
        "prompt", "iter", "tok", "audio_s", "wall_s", "RT×", "kB/s"
    );
    println!("{:-<70}", "");

    for (label, text) in PROMPTS {
        // Pre-count tokens for context (best-effort).
        let tok = model.tokenizer.encode(text).map(|v| v.len()).unwrap_or(0);

        // Warmup runs (results discarded).
        for w in 0..warmup {
            let mut opts = GenerationOptions::default();
            opts.seed = seed.wrapping_add((w as u64).wrapping_mul(0x9E37_79B9));
            let _ = model.generate(text, &voice, opts)?;
        }

        let mut audio_total = 0.0_f64;
        let mut wall_total = 0.0_f64;
        let mut samples_total: usize = 0;

        for it in 0..iters {
            let mut opts = GenerationOptions::default();
            opts.seed = seed.wrapping_add(((it + 100) as u64).wrapping_mul(0x9E37_79B9));
            let t = Instant::now();
            let audio = model.generate(text, &voice, opts)?;
            let wall = t.elapsed().as_secs_f64();
            let dur = audio.duration_secs() as f64;
            let rt = dur / wall.max(1e-9);
            let throughput_kb_s = (audio.samples.len() as f64 * 4.0 / 1024.0) / wall.max(1e-9);
            println!(
                "{:<8}  {:>5}  {:>9}  {:>9.2}  {:>9.2}  {:>9.2}  {:>7.1}",
                label, it, tok, dur, wall, rt, throughput_kb_s
            );
            audio_total += dur;
            wall_total += wall;
            samples_total += audio.samples.len();
        }
        let rt_mean = audio_total / wall_total.max(1e-9);
        println!(
            "{:<8}  {:>5}  {:>9}  {:>9.2}  {:>9.2}  {:>9.2}  {:>7}",
            label,
            "MEAN",
            tok,
            audio_total / iters as f64,
            wall_total / iters as f64,
            rt_mean,
            "",
        );
        let _ = samples_total;
        println!();
    }

    Ok(())
}
