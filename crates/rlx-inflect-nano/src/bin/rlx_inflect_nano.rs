//! Inflect-Nano-v1 text-to-speech CLI.
//!
//! rlx-inflect-nano --data weights/inflect-nano-rlx --text "Hello!" --out out.wav

use std::path::PathBuf;

use anyhow::Result;
use rlx_inflect_nano::{ExecutionMode, InferOpts, InflectNano, audio};

fn opt(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() -> Result<()> {
    let data = opt("--data").unwrap_or_else(|| "weights/inflect-nano-rlx".to_string());
    let Some(text) = opt("--text") else {
        eprintln!("usage: rlx-inflect-nano --data <bundle> --text <text> [--out out.wav]");
        eprintln!("       [--mode latency|precision|memory|hybrid] [--device cpu|metal|mlx|gpu]");
        eprintln!("       [--speed 1.0] [--length-scale 1.0] [--pitch-scale 1.0] [--energy-scale 1.0]");
        std::process::exit(2);
    };
    let out = opt("--out").unwrap_or_else(|| "inflect_nano_out.wav".to_string());
    let opts = InferOpts {
        speed: opt("--speed").and_then(|s| s.parse().ok()).unwrap_or(1.0),
        length_scale: opt("--length-scale")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0),
        pitch_scale: opt("--pitch-scale")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0),
        energy_scale: opt("--energy-scale")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0),
        ..Default::default()
    };

    let model = InflectNano::load_from_dir(&PathBuf::from(&data))?;

    // Streaming mode: emit ~chunk-second chunks; guarantees/reports real-time.
    if std::env::args().any(|a| a == "--stream") {
        let chunk_secs: f32 = opt("--chunk-secs")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        let mut samples = Vec::new();
        let report =
            model.synthesize_stream(&text, &opts, chunk_secs, |c| samples.extend_from_slice(c))?;
        audio::write_wav(&PathBuf::from(&out), &samples, model.cfg.sample_rate)?;
        println!(
            "wrote {out} (streaming, {} chunks of ~{chunk_secs:.2}s, {:.2}s audio in {:.3}s, RTF {:.1}x)",
            report.chunks,
            report.audio_secs,
            report.compute_secs,
            report.rtf()
        );
        println!(
            "real-time: {} (worst chunk {:.1}x — 1s audio in {:.3}s compute)",
            if report.sustains_realtime() {
                "YES"
            } else {
                "no"
            },
            report.worst_chunk_rtf,
            1.0 / report.worst_chunk_rtf.max(1e-6),
        );
        return Ok(());
    }

    let t0 = std::time::Instant::now();
    // `--mode` (execution policy) takes precedence over an explicit `--device`.
    let (wav, how) = if let Some(m) = opt("--mode") {
        let mode = ExecutionMode::parse(&m);
        (
            model.synthesize_mode(&text, &opts, mode)?,
            format!("mode={mode:?}"),
        )
    } else {
        let device = opt("--device").unwrap_or_else(|| "cpu".to_string());
        let wav = synthesize_device(&model, &text, &opts, &device)?;
        (wav, format!("device={device}"))
    };
    let secs = t0.elapsed().as_secs_f32();
    audio::write_wav(&PathBuf::from(&out), &wav.samples, wav.sample_rate)?;
    let audio_secs = wav.samples.len() as f32 / wav.sample_rate as f32;
    println!(
        "wrote {out} ({audio_secs:.2}s @ {} Hz, {how}, {secs:.3}s, RTF {:.1}x)",
        wav.sample_rate,
        audio_secs / secs
    );
    Ok(())
}

/// Synthesize on an explicit device string (`cpu` → host path; others → vocoder graph).
fn synthesize_device(
    model: &InflectNano,
    text: &str,
    opts: &InferOpts,
    device: &str,
) -> Result<rlx_inflect_nano::Wav> {
    #[cfg(feature = "onnx")]
    if device == "coreml" {
        return model.synthesize_coreml(text, opts);
    }
    #[cfg(feature = "rlx-graph")]
    if device != "cpu" {
        return model.synthesize_on(text, opts, rlx_inflect_nano::graph::device_from_str(device));
    }
    let _ = device;
    model.synthesize(text, opts)
}
