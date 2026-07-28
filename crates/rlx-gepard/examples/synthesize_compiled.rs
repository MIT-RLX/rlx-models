// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! Example: Generate audio using compiled RLX IR graphs (multi-backend support).
//!
//! This example demonstrates the full synthesis pipeline:
//! 1. Load pretrained Gepard weights
//! 2. Create compiled session for specified device (metal/mlx/cuda/etc)
//! 3. Generate audio via autoregressive decoding
//! 4. Save WAV file

use anyhow::Result;
use rlx_gepard::GepardSynthesizer;
use std::path::Path;

fn main() -> Result<()> {
    // Setup
    let bundle_path = "weights/tts/gepard";
    if !Path::new(bundle_path).exists() {
        eprintln!("Weights not found at {}. Skipping test.", bundle_path);
        return Ok(());
    }

    let text = "Hello world, this is Gepard text to speech.";
    let voice_desc = "warm conversational female voice";

    println!("=== Gepard TTS Synthesis (Compiled Paths) ===");
    println!("Text:  {}", text);
    println!("Voice: {}", voice_desc);
    println!();

    // Test eager path (default)
    println!("Testing eager CPU path...");
    let synth_eager = GepardSynthesizer::new(bundle_path)?;
    match synth_eager.synthesize(text, voice_desc) {
        Ok(audio) => {
            println!("✓ Eager path complete!");
            println!("  Generated {} audio samples", audio.len());
            println!(
                "  Duration: {:.2}s @ 22050 Hz",
                audio.len() as f32 / 22050.0
            );
            save_wav("/tmp/gepard_eager.wav", &audio)?;
        }
        Err(e) => eprintln!("  Error: {}", e),
    }

    println!();

    // Test compiled path on CPU
    println!("Testing compiled CPU path...");
    match GepardSynthesizer::with_device(bundle_path, "cpu") {
        Ok(synth_compiled) => match synth_compiled.synthesize(text, voice_desc) {
            Ok(audio) => {
                println!("✓ Compiled CPU path complete!");
                println!("  Generated {} audio samples", audio.len());
                println!(
                    "  Duration: {:.2}s @ 22050 Hz",
                    audio.len() as f32 / 22050.0
                );
                save_wav("/tmp/gepard_compiled_cpu.wav", &audio)?;
            }
            Err(e) => eprintln!("  Error: {}", e),
        },
        Err(e) => eprintln!("  Failed to create compiled session: {}", e),
    }

    println!();

    // Test compiled path on metal (macOS)
    #[cfg(target_os = "macos")]
    {
        println!("Testing compiled Metal path...");
        match GepardSynthesizer::with_device(bundle_path, "metal") {
            Ok(synth_metal) => match synth_metal.synthesize(text, voice_desc) {
                Ok(audio) => {
                    println!("✓ Compiled Metal path complete!");
                    println!("  Generated {} audio samples", audio.len());
                    println!(
                        "  Duration: {:.2}s @ 22050 Hz",
                        audio.len() as f32 / 22050.0
                    );
                    save_wav("/tmp/gepard_compiled_metal.wav", &audio)?;
                }
                Err(e) => eprintln!("  Error: {}", e),
            },
            Err(e) => eprintln!("  Failed to create Metal session: {}", e),
        }
    }

    println!();
    println!("✓ All tests complete!");
    println!();
    println!("Generated WAV files:");
    println!("  /tmp/gepard_eager.wav");
    println!("  /tmp/gepard_compiled_cpu.wav");
    #[cfg(target_os = "macos")]
    println!("  /tmp/gepard_compiled_metal.wav");

    Ok(())
}

fn save_wav(path: &str, samples: &[f32]) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let channels = 1;
    let sample_rate = 22050u32;
    let byte_rate = sample_rate * channels * 2;
    let block_align = channels * 2;
    let bits_per_sample = 16;

    let file = File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);

    // WAV header
    writer.write_all(b"RIFF")?;
    let file_size = 36 + samples.len() * 2;
    writer.write_all(&(file_size as u32).to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // fmt chunk
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?; // Subchunk1Size
    writer.write_all(&1u16.to_le_bytes())?; // AudioFormat (PCM)
    writer.write_all(&(channels as u16).to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&(block_align as u16).to_le_bytes())?;
    writer.write_all(&(bits_per_sample as u16).to_le_bytes())?;

    // data chunk
    writer.write_all(b"data")?;
    writer.write_all(&(samples.len() as u32 * 2).to_le_bytes())?;

    // Audio samples
    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_all(&pcm.to_le_bytes())?;
    }

    Ok(())
}
