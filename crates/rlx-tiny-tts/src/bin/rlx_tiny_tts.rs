//! TinyTTS command-line synthesizer.
//!
//! ```text
//! rlx-tiny-tts --data weights/tiny-tts-rlx --text "Hello world." --out out.wav
//!              [--device cpu|metal|mlx|cuda|rocm|gpu] [--speaker MALE]
//!              [--speed 1.0] [--seed 1234] [--kernel fast|precise|inherit]
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result};
use rlx_tiny_tts::{InferOpts, KernelVariant, TinyTts, audio};

fn opt(flag: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let data = opt("--data").unwrap_or_else(|| "weights/tiny-tts-rlx".to_string());

    // `--pack <dir> --out bundle.rlxpack`: package a bundle directory into a
    // single distributable file (loadable via `--data bundle.rlxpack`).
    if let Some(src_dir) = opt("--pack") {
        let out = opt("--out").unwrap_or_else(|| "tiny-tts.rlxpack".to_string());
        rlx_tiny_tts::asset_source::pack::write_dir(&src_dir, &out)
            .with_context(|| format!("pack {src_dir} → {out}"))?;
        let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
        println!(
            "[tiny-tts] packed {src_dir} → {out} ({:.1} MB)",
            bytes as f64 / 1e6
        );
        return Ok(());
    }

    let text = opt("--text")
        .unwrap_or_else(|| "The weather is nice today, and I feel very relaxed.".to_string());
    let out = opt("--out").unwrap_or_else(|| "out.wav".to_string());
    let speaker = opt("--speaker").unwrap_or_else(|| "MALE".to_string());
    let speed: f32 = opt("--speed").and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let seed: u64 = opt("--seed").and_then(|s| s.parse().ok()).unwrap_or(1234);

    let device = match opt("--device") {
        Some(d) => rlx_runtime::parse_device(&d).map_err(|e| anyhow::anyhow!("{e}"))?,
        None => TinyTts::preferred_device(),
    };

    // Accepts a directory, a packed `.rlxpack` file, or any path `AssetSource`
    // auto-detects — `--data bundle/` or `--data tiny-tts.rlxpack` both work.
    let model = TinyTts::load(PathBuf::from(&data))
        .with_context(|| format!("load TinyTTS bundle from {data}"))?;

    let mut opts = InferOpts::from_config(model.config());
    opts.length_scale = 1.0 / speed.max(1e-3); // speed>1 → faster → fewer frames
    opts.seed = seed;
    // Kernel-variant / precision policy (default fast; also honors RLX_TTS_KERNEL).
    if let Some(k) = opt("--kernel") {
        opts.kernel = match k.to_ascii_lowercase().as_str() {
            "precise" | "exact" | "parity" => KernelVariant::Precise,
            "inherit" | "env" => KernelVariant::Inherit,
            _ => KernelVariant::Fast,
        };
    }
    let _ = model.config().speaker_id(&speaker); // validate speaker name

    let t0 = std::time::Instant::now();
    let wav = model.synthesize_on(&text, device, &opts)?;
    let secs = wav.samples.len() as f32 / wav.sample_rate as f32;
    let elapsed = t0.elapsed().as_secs_f32();
    let peak = audio::peak_amplitude(&wav.samples);

    let normalized = audio::normalize_audio(&wav.samples);
    audio::write_wav(&PathBuf::from(&out), &normalized, wav.sample_rate)?;
    println!(
        "[tiny-tts] device={device:?} \"{text}\" → {out}  ({secs:.2}s audio @ {} Hz, peak={peak:.3}, {elapsed:.2}s synth, {:.1}× RT)",
        wav.sample_rate,
        secs / elapsed.max(1e-6),
    );
    Ok(())
}
