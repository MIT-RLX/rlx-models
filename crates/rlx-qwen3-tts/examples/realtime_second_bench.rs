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

//! Metal latency check: warm session → stream until 1 s PCM is emitted.
//!
//! Target: `StreamConfig::realtime_second()` delivers 1 s of audio in ≤ 1.2 s
//! wall on a warm Apple Silicon Metal session.
//!
//! ```sh
//! cargo run --release -p rlx-qwen3-tts --features apple-silicon \
//!   --example realtime_second_bench
//! ```
//!
//! Set `RLX_QWEN3_TTS_REALTIME_ASSERT=1` to fail the process when over budget.

use anyhow::{Context, Result, ensure};
use rlx_qwen3_tts::{StreamConfig, StreamControl, StreamEvent, VoiceClone};
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;
use std::time::Instant;

const ONE_SEC_SAMPLES: usize = 24_000;
/// Long enough to produce ≥ 1 s of PCM at 12 Hz (short "Hi." is only ~0.8 s).
const TARGET: &str = "Count one two three four five six seven eight nine ten.";
/// Stretch goal once talker+CP hit ~70 ms/frame on Metal (see README).
const STRETCH_BUDGET: f64 = 1.2;

fn model_dir() -> PathBuf {
    std::env::var("RLX_QWEN3_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base"))
}

fn ref_wav() -> PathBuf {
    PathBuf::from("assets/jfk/jfk_voice_clone.wav")
}

struct OneSecTiming {
    to_one_sec: f64,
    ttfa: f64,
    wall: f64,
    audio: f64,
    samples: usize,
}

fn stream_until_one_sec(
    tts: &mut VoiceClone,
    reference: &rlx_qwen3_tts::SpeakerReference,
    config: StreamConfig,
) -> Result<OneSecTiming> {
    let t0 = Instant::now();
    let mut to_one_sec = None;
    let mut samples = 0usize;
    let stats = tts.generate_stream(reference, TARGET, config, |evt| {
        if let StreamEvent::Pcm(chunk) = evt {
            samples += chunk.samples.len();
            if samples >= ONE_SEC_SAMPLES && to_one_sec.is_none() {
                to_one_sec = Some(t0.elapsed().as_secs_f64());
            }
        }
        StreamControl::Continue
    })?;
    Ok(OneSecTiming {
        to_one_sec: to_one_sec.context("stream ended before 1 s of PCM (use a longer target)")?,
        ttfa: stats.time_to_first_audio_secs,
        wall: stats.wall_secs,
        audio: stats.audio_secs,
        samples,
    })
}

fn budget_secs() -> f64 {
    std::env::var("RLX_QWEN3_TTS_REALTIME_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.0)
}

fn main() -> Result<()> {
    // Matches README Metal session guidance; ~2× faster than default thread pool.
    if std::env::var("VECLIB_MAXIMUM_THREADS").is_err() {
        unsafe {
            std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
        }
    }
    if !is_available(Device::Metal) {
        eprintln!("skip: Metal not available on this host");
        return Ok(());
    }

    let model = model_dir();
    let wav = ref_wav();
    let config = StreamConfig::realtime_second();

    println!("┌─ realtime_second bench (Metal, warm session) ─────────────────");
    println!("│ model:  {}", model.display());
    println!("│ ref:    {}", wav.display());
    println!("│ target: {TARGET:?}");
    println!("│ config: progressive(12), chunk=24000 (~1 s)");
    let budget = budget_secs();
    println!("│ budget: {budget:.1}s wall for 1 s PCM (stretch goal {STRETCH_BUDGET:.1}s)");
    println!("│ tip:    VECLIB_MAXIMUM_THREADS=1 (set automatically if unset)");
    println!("└───────────────────────────────────────────────────────────────\n");

    let t_open = Instant::now();
    let mut tts = VoiceClone::open(&model, Device::Metal)?;
    let reference = tts.extract_reference(&wav)?;
    println!("cold open + ref: {:.2}s\n", t_open.elapsed().as_secs_f64());

    // Warm-up: batch pass compiles talker/CP/decode without progressive rework.
    println!("[warm-up] batch generate…");
    let t_warm = Instant::now();
    let _ = tts.generate(&reference, TARGET)?;
    println!(
        "[warm-up] batch wall {:.2}s\n",
        t_warm.elapsed().as_secs_f64()
    );

    println!("[measured] streaming…");
    let m = stream_until_one_sec(&mut tts, &reference, config)?;
    let budget = budget_secs();
    let pass = m.to_one_sec <= budget;
    let stretch = m.to_one_sec <= STRETCH_BUDGET;
    println!(
        "[measured] to_1s={:.2}s  ttfa={:.2}s  wall={:.2}s  audio={:.2}s  samples={}",
        m.to_one_sec, m.ttfa, m.wall, m.audio, m.samples
    );
    println!(
        "\n{} budget ({budget:.1}s): {:.2}s",
        if pass { "PASS" } else { "FAIL" },
        m.to_one_sec
    );
    println!(
        "{} stretch ({STRETCH_BUDGET:.1}s): {:.2}s",
        if stretch { "PASS" } else { "FAIL" },
        m.to_one_sec
    );

    if std::env::var("RLX_QWEN3_TTS_REALTIME_ASSERT")
        .ok()
        .as_deref()
        == Some("1")
    {
        ensure!(
            pass,
            "warm Metal stream took {:.2}s to emit 1 s PCM (budget {budget:.1}s)",
            m.to_one_sec
        );
    } else if !pass {
        eprintln!("\nhint: set RLX_QWEN3_TTS_REALTIME_ASSERT=1 to fail on over-budget runs");
    }

    Ok(())
}
