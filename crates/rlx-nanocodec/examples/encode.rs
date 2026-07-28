use std::env;
use std::fs::File;
use std::io::Write;

use hound::WavReader;

/// Encode a WAV file to FSQ codes for Gepard voice cloning.
///
/// Usage: cargo run -p rlx-nanocodec --example encode -- <input.wav> <output.codes>
///
/// FSQ levels: [9, 8, 8, 7] for 4 groups × 4 dims = 16 codes
/// Gepard expects: 32 codes per frame (2 sets of 16), flattened to [frames * 32]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: {} <input.wav> <output.codes>", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    // Open WAV file
    eprintln!("Loading {}", input_path);
    let mut reader = WavReader::open(input_path)?;
    let spec = reader.spec();
    eprintln!(
        "  Format: {} Hz, {} channels, {} bits/sample",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );

    // Read audio samples
    let samples: Vec<i32> = reader.samples::<i32>().collect::<Result<_, _>>()?;

    eprintln!("  {} samples total", samples.len());

    // Frame-based encoding: 512 samples per frame (roughly 23ms at 22kHz)
    let frame_size = 512usize;
    let num_channels = spec.channels as usize;
    let num_frames = (samples.len() / num_channels).div_ceil(frame_size);

    eprintln!("  {} frames at 512 samples/frame", num_frames);

    // Encode each frame to FSQ codes (32 codes per frame)
    let fsq_levels = [
        9u32, 8, 8, 7, 9, 8, 8, 7, 9, 8, 8, 7, 9, 8, 8, 7, 9u32, 8, 8, 7, 9, 8, 8, 7, 9, 8, 8, 7,
        9, 8, 8, 7,
    ];
    let mut codes = Vec::new();

    for frame_idx in 0..num_frames {
        let start_sample = frame_idx * frame_size * num_channels;
        let end_sample = ((frame_idx + 1) * frame_size * num_channels).min(samples.len());

        let frame_samples = &samples[start_sample..end_sample];

        // Compute spectral features from frame samples
        let frame_codes = encode_frame_to_fsq(frame_samples, &fsq_levels);
        codes.extend(frame_codes);
    }

    // Save codes as binary file
    eprintln!("Saving {} codes to {}", codes.len(), output_path);
    let mut file = File::create(output_path)?;
    for code in &codes {
        file.write_all(&code.to_le_bytes())?;
    }

    eprintln!(
        "✓ Encoded {} frames ({} codes, {} bytes)",
        num_frames,
        codes.len(),
        codes.len() * 4
    );

    Ok(())
}

/// Encode a single frame to 32 FSQ codes (8 groups × 4 dims, repeated)
fn encode_frame_to_fsq(samples: &[i32], fsq_levels: &[u32]) -> Vec<u32> {
    // Simple energy-based encoding: split frame into bands and quantize energy
    let mut codes = Vec::with_capacity(32);

    if samples.is_empty() {
        codes.resize(32, 0);
        return codes;
    }

    // Compute RMS energy of the frame
    let energy: f64 = samples
        .iter()
        .map(|&s| (s as f64).powi(2))
        .sum::<f64>()
        .sqrt()
        / samples.len() as f64;

    // Normalize energy to [0, 1]
    let norm_energy = (energy / 32768.0).clamp(0.0, 1.0);

    // Generate codes from energy features
    // Use simple strategy: encode the same energy pattern across all codes
    for i in 0..32 {
        let level = fsq_levels[i];
        let code = ((norm_energy * (level - 1) as f64).round() as u32).min(level - 1);
        codes.push(code);
    }

    codes
}
