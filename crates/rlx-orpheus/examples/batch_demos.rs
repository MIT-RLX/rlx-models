//! Generate several Orpheus demo WAVs in one process (one backbone load).
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rlx_orpheus::{BackboneLoadOptions, GenerationConfig, OrpheusTts, VoiceCloneReference};
use rlx_runtime::Device;

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

fn synth_named(
    tts: &mut OrpheusTts,
    out: &Path,
    voice: &str,
    text: &str,
    max_tokens: u32,
) -> Result<()> {
    tts.config.max_new_tokens = max_tokens;
    // Default sampling (temperature 0.6, repetition penalty) — greedy argmax
    // often emits out-of-range SNAC codes on Q4_K_M.
    eprintln!(
        "synth {} voice={voice:?} text={text:?} max={max_tokens}",
        out.display()
    );
    let t0 = Instant::now();
    let result = tts.synthesize(text, Some(voice))?;
    write_wav(out, &result.samples, result.sample_rate)?;
    eprintln!(
        "  -> {} codes, {:.2}s audio, {:.1}s wall",
        result.code_count,
        result.samples.len() as f64 / result.sample_rate as f64,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() -> Result<()> {
    let out_dir = PathBuf::from(
        std::env::var("ORPHEUS_DEMO_DIR").unwrap_or_else(|_| "/tmp/orpheus-demos".into()),
    );
    std::fs::create_dir_all(&out_dir)?;

    let gguf = std::env::var("ORPHEUS_GGUF_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf")
        });
    let snac = std::env::var("ORPHEUS_SNAC_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors"));
    if !gguf.is_file() || !snac.is_file() {
        bail!("missing weights — run `just fetch-orpheus fetch-orpheus-snac`");
    }

    // Metal KV + CPU F32 GGUF prefill (reference parity; correct multi-step decode).
    let load_t0 = Instant::now();
    let mut tts = OrpheusTts::load_on_with_device(
        &gguf,
        &snac,
        Device::Metal,
        BackboneLoadOptions::for_device(Device::Metal),
    )?;
    eprintln!("loaded in {:.1}s", load_t0.elapsed().as_secs_f64());

    synth_named(&mut tts, &out_dir.join("short-tara.wav"), "tara", "Hi.", 28)?;
    synth_named(
        &mut tts,
        &out_dir.join("short-leo.wav"),
        "leo",
        "Hello there.",
        28,
    )?;
    let batch = tts.synthesize_batch(&[
        (
            "Hello from RLX Orpheus. This is a longer sample.",
            Some("tara"),
        ),
        ("The quick brown fox jumps over the lazy dog.", Some("mia")),
    ])?;
    for (i, result) in batch.into_iter().enumerate() {
        let name = if i == 0 {
            "long-tara.wav"
        } else {
            "long-mia.wav"
        };
        write_wav(&out_dir.join(name), &result.samples, result.sample_rate)?;
        eprintln!(
            "  -> {} {} codes, {:.2}s audio",
            name,
            result.code_count,
            result.samples.len() as f64 / result.sample_rate as f64
        );
    }

    let ref_json = out_dir.join("jfk_ref.json");
    if ref_json.is_file() {
        let reference = VoiceCloneReference::load_json(&ref_json)?;
        tts.config = GenerationConfig {
            max_new_tokens: 56,
            greedy: true,
            ..GenerationConfig::default()
        };
        let target = "I write my software in Rust because it is fast, safe, and predictable.";
        eprintln!("clone -> {}", out_dir.join("clone-jfk.wav").display());
        let t0 = Instant::now();
        let result =
            tts.synthesize_voice_clone(&reference.transcript, &reference.token_ids, target)?;
        write_wav(
            &out_dir.join("clone-jfk.wav"),
            &result.samples,
            result.sample_rate,
        )?;
        eprintln!(
            "  -> {} codes, {:.1}s wall",
            result.code_count,
            t0.elapsed().as_secs_f64()
        );
    }

    eprintln!("\nWrote demos under {}", out_dir.display());
    Ok(())
}
