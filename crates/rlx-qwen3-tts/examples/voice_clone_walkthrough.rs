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

//! Voice clone walkthrough — extract once, generate many.
//!
//! Run:
//!   cargo run --release -p rlx-qwen3-tts --example voice_clone_walkthrough \
//!     --features apple-silicon -- \
//!     --ref-wav assets/jfk/jfk_voice_clone.wav \
//!     --out-dir /tmp/jfk_clones
//!
//! Defaults pick the Qwen3-TTS Base model under `.cache/qwen3-tts/` and
//! the bundled JFK reference clip if no args are given.
//!
//! Two-step workflow:
//!   1. EXTRACT — encode reference WAV into a 1024-d ECAPA x-vector
//!      and save it as a JSON file for reuse.
//!   2. GENERATE — synthesize any text in that voice. Optionally load
//!      the JSON reference instead of re-extracting from WAV.

use anyhow::Result;
use rlx_qwen3_tts::{SpeakerReference, VoiceClone};
use rlx_runtime::Device;
use std::path::PathBuf;
use std::time::Instant;

fn parse_args() -> (PathBuf, PathBuf, PathBuf) {
    let mut model_dir = PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base");
    let mut ref_wav = PathBuf::from("assets/jfk/jfk_voice_clone.wav");
    let mut out_dir = PathBuf::from("/tmp/jfk_clones");
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--model-dir" => {
                model_dir = PathBuf::from(&raw[i + 1]);
                i += 2;
            }
            "--ref-wav" => {
                ref_wav = PathBuf::from(&raw[i + 1]);
                i += 2;
            }
            "--out-dir" => {
                out_dir = PathBuf::from(&raw[i + 1]);
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: voice_clone_walkthrough \
                    [--model-dir DIR] [--ref-wav WAV] [--out-dir DIR]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg {other:?}");
                std::process::exit(2);
            }
        }
    }
    (model_dir, ref_wav, out_dir)
}

fn main() -> Result<()> {
    let (model_dir, ref_wav, out_dir) = parse_args();
    std::fs::create_dir_all(&out_dir)?;

    println!("┌─ Voice clone walkthrough ──────────────────────────────────────");
    println!("│ model:    {}", model_dir.display());
    println!("│ ref WAV:  {}", ref_wav.display());
    println!("│ out dir:  {}", out_dir.display());
    println!("└────────────────────────────────────────────────────────────────\n");

    // ──────────────────────────────────────────────────────────────────────
    //  Open the TTS model.  This is the slow step (~1 second on M3 Pro).
    //  Reuse the returned `VoiceClone` for as many clones as you want.
    // ──────────────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    let mut tts = VoiceClone::open(&model_dir, Device::Metal)?;
    println!(
        "✓ opened model in {:.2}s on {:?}",
        t0.elapsed().as_secs_f64(),
        tts.device()
    );

    // ──────────────────────────────────────────────────────────────────────
    //  STEP 1 — EXTRACT the speaker reference from the WAV.
    //  This is fast (~50 ms).  Save it as JSON so future runs can skip
    //  the WAV entirely.
    // ──────────────────────────────────────────────────────────────────────
    println!("\n── Step 1: extract speaker reference ──────────────────────────");
    let t = Instant::now();
    let reference = tts.extract_reference(&ref_wav)?;
    println!(
        "✓ extracted in {:.2}s ({} dims, norm {:.2})",
        t.elapsed().as_secs_f64(),
        reference.dim(),
        reference.norm()
    );

    let ref_path = out_dir.join("speaker.ref.json");
    reference.save_json(&ref_path)?;
    println!("✓ saved reference to {}", ref_path.display());

    // Demonstrate the round-trip works.
    let loaded = SpeakerReference::load_json(&ref_path)?;
    let self_cosine = reference.cosine(&loaded);
    println!("✓ JSON round-trip cosine = {self_cosine:.6} (should be 1.000000)");

    // ──────────────────────────────────────────────────────────────────────
    //  STEP 2 — GENERATE speech using the reference.  Generate multiple
    //  clones to show that the model is hot and per-clone cost is small.
    // ──────────────────────────────────────────────────────────────────────
    println!("\n── Step 2: generate speech using that reference ───────────────");
    let utterances = [
        ("hello", "Hello, my fellow Americans."),
        (
            "ask_not",
            "Ask not what your country can do for you, ask what you can do for your country.",
        ),
        (
            "rust",
            "I write my software in Rust now, not because it is easy, but because it is fast.",
        ),
    ];

    for (name, text) in &utterances {
        let out = out_dir.join(format!("{name}.wav"));
        let t = Instant::now();
        tts.generate_to_wav(&loaded, text, &out)?;
        let dt = t.elapsed().as_secs_f64();
        // Inspect the produced audio briefly.
        let size_kb = std::fs::metadata(&out)?.len() / 1024;
        println!(
            "  • {:<8} ({:.1}s wall, {}KB)  →  {}",
            name,
            dt,
            size_kb,
            out.display()
        );
    }

    println!("\n┌─ Done ─────────────────────────────────────────────────────────");
    println!(
        "│ The saved reference file at {} can be checked into a repo,",
        ref_path.display()
    );
    println!("│ shipped with your app, or loaded by any other process to clone");
    println!("│ this voice WITHOUT shipping the original WAV.");
    println!("│");
    println!(
        "│ Listen:  afplay {}",
        out_dir.join("ask_not.wav").display()
    );
    println!("└────────────────────────────────────────────────────────────────");

    Ok(())
}
