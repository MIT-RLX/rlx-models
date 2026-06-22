//! Regenerate bundled MP4 listening samples under `crates/rlx-orpheus/assets/`.
//!
//! Requires `ffmpeg` on PATH and exported SNAC weights (`ORPHEUS_SNAC_PATH`).
//!
//! ```bash
//! export ORPHEUS_SNAC_PATH=/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors
//! cargo run -p rlx-orpheus --example export_audio_assets --release --features "llama,coreml,metal"
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use rlx_orpheus::{SAMPLE_RATE, SnacBackend, SnacLoadOptions, decode_orpheus_codes};

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

fn write_wav(path: &Path, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
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

fn wav_to_mp4(wav: &Path, mp4: &Path) -> Result<()> {
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            wav.to_str().context("wav path utf8")?,
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            mp4.to_str().context("mp4 path utf8")?,
        ])
        .status()
        .context("spawn ffmpeg")?;
    if !status.success() {
        bail!("ffmpeg failed for {}", mp4.display());
    }
    Ok(())
}

fn golden_codes() -> Result<Vec<i32>> {
    let fixture = include_str!("../tests/fixtures/orpheus_hi_codes.txt");
    let mut lines = fixture.lines().filter(|l| !l.starts_with('#'));
    let _count: usize = lines.next().context("count")?.trim().parse()?;
    lines
        .next()
        .context("codes")?
        .split_whitespace()
        .map(|s| s.parse().context("code"))
        .collect()
}

fn export_snac_mp4(
    snac: &Path,
    coreml: bool,
    codes: &[i32],
    stem: &str,
    out_dir: &Path,
) -> Result<()> {
    let backend = SnacBackend::open(snac, SnacLoadOptions { coreml })?;
    let samples = decode_orpheus_codes(&backend, codes)?;
    let wav = out_dir.join(format!("{stem}.wav"));
    let mp4 = out_dir.join(format!("{stem}.mp4"));
    write_wav(&wav, &samples)?;
    wav_to_mp4(&wav, &mp4)?;
    std::fs::remove_file(&wav).ok();
    eprintln!(
        "wrote {} ({:.2}s, coreml={coreml})",
        mp4.display(),
        samples.len() as f64 / SAMPLE_RATE as f64
    );
    Ok(())
}

fn main() -> Result<()> {
    let snac = std::env::var("ORPHEUS_SNAC_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors"));
    if !snac.is_file() {
        bail!(
            "missing SNAC weights at {} — export with scripts/export_snac_decoder.py",
            snac.display()
        );
    }
    let out_dir = assets_dir();
    std::fs::create_dir_all(&out_dir)?;

    let codes = golden_codes()?;
    export_snac_mp4(&snac, false, &codes, "sample_hi_eager", &out_dir)?;
    export_snac_mp4(&snac, true, &codes, "sample_hi_coreml", &out_dir)?;

    for (env_key, dst) in [
        ("ORPHEUS_SAMPLE_LONG_WAV", "sample_hello_rlx"),
        ("ORPHEUS_SAMPLE_LONGER_WAV", "sample_longer"),
    ] {
        let Some(wav) = std::env::var(env_key)
            .ok()
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
            .filter(|p| p.is_file())
        else {
            eprintln!("skip {dst} (set {env_key} to an existing WAV to refresh)");
            continue;
        };
        let mp4 = out_dir.join(format!("{dst}.mp4"));
        wav_to_mp4(&wav, &mp4)?;
        eprintln!("wrote {}", mp4.display());
    }

    Ok(())
}
