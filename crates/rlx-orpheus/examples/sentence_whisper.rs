//! Synthesize one sentence, write WAV, run Whisper validation.
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rlx_orpheus::{BackboneLoadOptions, GenerationConfig, OrpheusTts};

fn write_wav(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create {}", path.display()))?;
    for &s in samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn main() -> Result<()> {
    let text = std::env::var("ORPHEUS_TEXT").unwrap_or_else(|_| "Hello from RLX.".into());
    let voice = std::env::var("ORPHEUS_VOICE").unwrap_or_else(|_| "tara".into());
    let max_tokens: u32 = std::env::var("ORPHEUS_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);
    let out = PathBuf::from(
        std::env::var("ORPHEUS_OUT").unwrap_or_else(|_| "/tmp/orpheus-sentence.wav".into()),
    );
    let gguf = std::env::var("ORPHEUS_GGUF_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf")
        });
    let snac = std::env::var("ORPHEUS_SNAC_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors"));
    if !gguf.is_file() || !snac.is_file() {
        bail!("missing weights — set ORPHEUS_GGUF_PATH and ORPHEUS_SNAC_PATH");
    }

    let runtime = rlx_orpheus::resolve_orpheus_device(
        &std::env::var("ORPHEUS_DEVICE").unwrap_or_else(|_| "auto".into()),
    )?;
    eprintln!(
        "lm={:?} snac={:?} text={text:?} voice={voice:?} max_tokens={max_tokens}",
        runtime.lm, runtime.snac
    );

    let t0 = Instant::now();
    let mut tts = OrpheusTts::load_on_with(
        &gguf,
        &snac,
        runtime,
        BackboneLoadOptions::for_device(runtime.lm),
    )?;
    tts.config = GenerationConfig {
        max_new_tokens: max_tokens,
        top_p: 0.95,
        ..GenerationConfig::default()
    };

    let result = tts.synthesize(&text, Some(&voice))?;
    let peak = result
        .samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "synth {:.1}s: {} codes -> {} samples ({:.2}s) peak={peak:.4}",
        t0.elapsed().as_secs_f64(),
        result.code_count,
        result.samples.len(),
        result.samples.len() as f64 / result.sample_rate as f64,
    );

    write_wav(&out, &result.samples, result.sample_rate)?;
    eprintln!("wrote {}", out.display());

    if let Ok(codes_path) = std::env::var("ORPHEUS_DUMP_CODES") {
        let body = format!(
            "{}\n{}",
            result.codes.len(),
            result
                .codes
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
        std::fs::write(&codes_path, body)?;
        eprintln!("wrote codes {}", codes_path);
    }

    let whisper_dir = std::env::var("RLX_WHISPER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/Shared/rlx-models/.cache/whisper-base.en"));
    let wav16 = out.with_extension("16k.wav");
    let status = Command::new("ffmpeg")
        .args(["-y", "-i"])
        .arg(&out)
        .args(["-ar", "16000", "-ac", "1"])
        .arg(&wav16)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .context("ffmpeg resample")?;
    if !status.success() {
        bail!("ffmpeg failed");
    }

    let whisper_bin =
        std::env::var("RLX_WHISPER_BIN").unwrap_or_else(|_| "target/release/rlx-whisper".into());
    let output = Command::new(&whisper_bin)
        .args([
            "--weights",
            &whisper_dir.join("model.safetensors").to_string_lossy(),
            "--config",
            &whisper_dir.join("config.json").to_string_lossy(),
            "--tokenizer",
            &whisper_dir.join("tokenizer.json").to_string_lossy(),
            "--wav",
            &wav16.to_string_lossy(),
            "--device",
            "cpu",
            "--language",
            "en",
        ])
        .output()
        .with_context(|| format!("run {whisper_bin}"))?;
    if !output.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        bail!("whisper failed");
    }
    let text_out = String::from_utf8_lossy(&output.stdout);
    for line in text_out.lines() {
        if let Some(rest) = line.strip_prefix("[rlx-whisper] transcribed") {
            eprintln!("whisper:{rest}");
        }
    }
    Ok(())
}
