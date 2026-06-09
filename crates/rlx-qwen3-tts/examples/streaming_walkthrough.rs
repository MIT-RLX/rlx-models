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

//! Streaming voice clone walkthrough.
//!
//! Demonstrates all three streaming modes and compares them:
//!   - StreamMode::Batched     — chunks emitted after full generation
//!   - StreamMode::PerFrame    — adds AR-level progress callbacks
//!   - StreamMode::Progressive — partial-decode every K codec frames
//!
//! Reports time-to-first-audio, total wall, chunk count, and real-time factor
//! for each mode so you can pick the right one for your use case.
//!
//! Run:
//!   cargo run --release -p rlx-qwen3-tts --example streaming_walkthrough \
//!     --features apple-silicon

use anyhow::Result;
use rlx_qwen3_tts::{StreamConfig, StreamControl, StreamEvent, VoiceClone};
use rlx_runtime::Device;
use std::path::PathBuf;

const TARGET_TEXT: &str = "We choose to go to the moon in this decade, and to do the other things, \
     not because they are easy, but because they are hard.";

fn main() -> Result<()> {
    let model_dir = PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base");
    let ref_wav = PathBuf::from("assets/jfk/jfk_voice_clone.wav");

    println!("┌─ Streaming walkthrough ────────────────────────────────────────");
    println!("│ model:    {}", model_dir.display());
    println!("│ ref WAV:  {}", ref_wav.display());
    println!("│ text:     {TARGET_TEXT:?}");
    println!("└────────────────────────────────────────────────────────────────\n");

    let mut tts = VoiceClone::open(&model_dir, Device::Metal)?;
    let reference = tts.extract_reference(&ref_wav)?;
    println!("✓ model + reference ready ({} dims)\n", reference.dim());

    let configs = [
        (
            "Batched (1 s emit chunks)",
            StreamConfig::batched().with_chunk_samples(24_000),
        ),
        (
            "PerFrame (1 s emit chunks)",
            StreamConfig::per_frame().with_chunk_samples(24_000),
        ),
        (
            "Live: Progressive  4 frames/decode  (~0.33 s audio per partial decode)",
            StreamConfig::progressive(4).with_chunk_samples(8_000),
        ),
        (
            "Live: Progressive  8 frames/decode  (~0.67 s audio per partial decode)",
            StreamConfig::progressive(8).with_chunk_samples(16_000),
        ),
        (
            "Live: Progressive 16 frames/decode  (~1.33 s audio per partial decode)",
            StreamConfig::progressive(16).with_chunk_samples(24_000),
        ),
        (
            "Live: Progressive 32 frames/decode  (~2.67 s audio per partial decode)",
            StreamConfig::progressive(32).with_chunk_samples(24_000),
        ),
        (
            "Live: Progressive 64 frames/decode  (~5.33 s audio per partial decode)",
            StreamConfig::progressive(64).with_chunk_samples(24_000),
        ),
    ];

    println!(
        "{:<55}  {:>9}  {:>9}  {:>7}  {:>6}  {:>6}",
        "Mode", "Frames", "Chunks", "TTFA", "Audio", "RTF"
    );
    println!("{}", "─".repeat(101));

    for (label, config) in configs {
        let mut frame_progress = 0usize;
        let mut frame_max = 0usize;
        let mut chunks_seen = 0usize;
        let mut first_chunk_offset = None::<usize>;

        // Buffer the streamed samples so we can write a WAV and verify precision.
        let mut pcm = Vec::<f32>::new();
        let stats = tts.generate_stream(&reference, TARGET_TEXT, config.clone(), |evt| {
            match evt {
                StreamEvent::FrameProduced {
                    frame_index,
                    max_frames,
                } => {
                    frame_progress = frame_index + 1;
                    frame_max = max_frames;
                }
                StreamEvent::Pcm(chunk) => {
                    if first_chunk_offset.is_none() {
                        first_chunk_offset = Some(chunk.sample_offset);
                    }
                    chunks_seen += 1;
                    pcm.extend_from_slice(&chunk.samples);
                }
            }
            StreamControl::Continue
        })?;
        // Write the accumulated PCM as a WAV so the user can listen / verify.
        let safe_label: String = label
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == ' ')
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join("_");
        let out_wav = std::path::PathBuf::from(format!("/tmp/stream_{}.wav", safe_label));
        rlx_qwen3_tts::runner::write_wav_mono(&out_wav, &pcm, 24_000)?;

        println!(
            "{:<55}  {:>9}  {:>9}  {:>5.2}s  {:>5.2}s  {:>5.2}×",
            label,
            stats.frames_emitted,
            stats.chunks_emitted,
            stats.time_to_first_audio_secs,
            stats.audio_secs,
            stats.realtime_factor(),
        );
        let _ = (frame_progress, frame_max, chunks_seen);
    }

    println!();
    println!("── Notes ──");
    println!("  • Batched: same audio bytes as PerFrame; chunk count varies with chunk_samples.");
    println!(
        "  • PerFrame: identical wall time to Batched; gives you FrameProduced events for progress UIs."
    );
    println!(
        "  • Progressive: each prefix-decode runs on a longer suffix, so total CPU is higher,"
    );
    println!(
        "    but it produces chunks sooner — useful when you want to pipe to a sink (audio device,"
    );
    println!("    network, file) before the full utterance is ready.");
    println!();
    println!("── Async APIs (gated behind features) ──");
    println!("  • cargo build --features async   → futures::Stream<Item = Result<PcmChunk>>");
    println!("  • cargo build --features tokio   → tokio::sync::mpsc::Receiver<Result<PcmChunk>>");

    Ok(())
}
