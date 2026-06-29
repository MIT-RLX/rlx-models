// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! Orpheus TTS bench across RLX backends with optional Whisper ASR validation.
//!
//! ```bash
//! just fetch-orpheus fetch-orpheus-snac fetch-whisper
//! cargo run -p rlx-orpheus --example tts_bench --release --features apple-silicon -- \
//!   --devices all --whisper
//!
//! # Voice clone (pretrained GGUF + encoded reference JSON):
//! ORPHEUS_PRETRAINED_GGUF=/path/pretrained.gguf \
//! ORPHEUS_CLONE_REF_JSON=/tmp/jfk_orpheus_ref.json \
//! cargo run -p rlx-orpheus --example tts_bench --release --features apple-silicon -- \
//!   --devices metal --voice-clone --whisper
//! ```

use anyhow::{Context, Result, bail};
use rlx_llama32::MetalGgufPrefillMode;
use rlx_orpheus::{
    BackboneLoadOptions, GenerationConfig, OrpheusTts, SAMPLES_PER_FRAME, VoiceCloneReference,
    lm_kv_decode_supported, preferred_synth_device,
};
use rlx_runtime::{Device, is_available};
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};
use std::path::{Path, PathBuf};
use std::time::Instant;

const SAMPLE_RATE: u32 = 24_000;

fn main() -> Result<()> {
    let args = parse_args()?;
    if args.devices.is_empty() {
        bail!("no devices selected");
    }

    let gguf = args
        .weights
        .or_else(default_gguf)
        .filter(|p| p.is_file())
        .ok_or_else(|| anyhow::anyhow!("missing --weights / ORPHEUS_GGUF_PATH"))?;
    let snac = args
        .snac
        .or_else(default_snac)
        .filter(|p| p.is_file())
        .ok_or_else(|| anyhow::anyhow!("missing --snac / ORPHEUS_SNAC_PATH"))?;

    let whisper_dir = if args.whisper {
        Some(resolve_whisper_dir()?)
    } else {
        None
    };

    eprintln!("Orpheus TTS bench");
    eprintln!("  weights: {}", gguf.display());
    eprintln!("  snac:    {}", snac.display());
    eprintln!("  text:    {:?}", args.text);
    eprintln!("  voice:   {:?}", args.voice);
    if args.voice_clone {
        eprintln!("  mode:    voice clone");
    }
    eprintln!();

    print_header();
    for device in &args.devices {
        if !is_available(*device) {
            eprintln!("skip {:?}: unavailable", device);
            continue;
        }
        if args.voice_clone {
            let row = bench_clone(
                &gguf,
                &snac,
                *device,
                args.clone_ref
                    .as_ref()
                    .context("--voice-clone needs ORPHEUS_CLONE_REF_JSON or --clone-ref")?,
                &args.clone_target,
                args.warmup,
                args.iters,
                args.max_tokens,
                args.metal_prefill,
                whisper_dir.as_deref(),
            )?;
            row.print();
            if args.whisper && !row.whisper_ok {
                bail!(
                    "Whisper validation failed on {:?}: got {:?}",
                    device,
                    row.transcript
                );
            }
        } else {
            let row = bench_named(
                &gguf,
                &snac,
                *device,
                &args.text,
                &args.voice,
                args.warmup,
                args.iters,
                args.max_tokens,
                args.metal_prefill,
                whisper_dir.as_deref(),
            )?;
            row.print();
            if args.whisper && !row.whisper_ok {
                bail!(
                    "Whisper validation failed on {:?}: ref={:?} got={:?}",
                    device,
                    args.text,
                    row.transcript
                );
            }
            assert_synthesis_length(&args.text, row.codes, row.samples)?;
        }
    }
    Ok(())
}

struct Args {
    weights: Option<PathBuf>,
    snac: Option<PathBuf>,
    text: String,
    voice: String,
    devices: Vec<Device>,
    whisper: bool,
    voice_clone: bool,
    clone_ref: Option<PathBuf>,
    clone_target: String,
    warmup: u32,
    iters: u32,
    max_tokens: u32,
    metal_prefill: MetalGgufPrefillMode,
}

struct Row {
    label: String,
    wall_ms: f64,
    audio_s: f64,
    rtf: f64,
    codes: usize,
    samples: usize,
    whisper_ok: bool,
    transcript: String,
}

impl Row {
    fn print(&self) {
        eprintln!(
            "{:<12} {:>8.0} {:>8.2} {:>6.2} {:>6} {:>6}  {}",
            self.label,
            self.wall_ms,
            self.audio_s,
            self.rtf,
            self.codes,
            if self.whisper_ok { "ok" } else { "FAIL" },
            self.transcript.trim()
        );
    }
}

fn print_header() {
    eprintln!(
        "{:<12} {:>8} {:>8} {:>6} {:>6} {:>6}  transcript",
        "device", "wall_ms", "audio_s", "rtf", "codes", "whisp"
    );
}

fn parse_args() -> Result<Args> {
    let mut weights = None;
    let mut snac = None;
    let mut text = "Hello from RLX Orpheus.".to_string();
    let mut voice = "tara".to_string();
    let mut devices = vec![preferred_synth_device()];
    let mut whisper = false;
    let mut voice_clone = false;
    let mut clone_ref = std::env::var("ORPHEUS_CLONE_REF_JSON")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.is_file());
    let mut clone_target = "I write my software in Rust because it is fast.".to_string();
    let mut warmup = 1u32;
    let mut iters = 1u32;
    let mut max_tokens = 120u32;
    let mut metal_prefill = rlx_llama32::MetalGgufPrefillMode::CpuF32;

    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        let take = |i: &mut usize| -> Result<String> {
            *i += 1;
            raw.get(*i)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing value for {}", raw[*i - 1]))
        };
        match raw[i].as_str() {
            "--weights" => weights = Some(PathBuf::from(take(&mut i)?)),
            "--snac" => snac = Some(PathBuf::from(take(&mut i)?)),
            "--text" => text = take(&mut i)?,
            "--voice" => voice = take(&mut i)?,
            "--devices" => devices = parse_devices(&take(&mut i)?)?,
            "--whisper" => whisper = true,
            "--voice-clone" => voice_clone = true,
            "--clone-ref" => clone_ref = Some(PathBuf::from(take(&mut i)?)),
            "--clone-target" => clone_target = take(&mut i)?,
            "--warmup" => warmup = take(&mut i)?.parse()?,
            "--iters" => iters = take(&mut i)?.parse()?,
            "--max-tokens" => max_tokens = take(&mut i)?.parse()?,
            "--metal-prefill" => {
                let s = take(&mut i)?;
                metal_prefill = MetalGgufPrefillMode::parse(&s).ok_or_else(|| {
                    anyhow::anyhow!("--metal-prefill: expected auto|cpu|packed|metal")
                })?;
            }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: tts_bench [--weights GGUF] [--snac ST] [--devices all|cpu,metal] \\
  [--text STR] [--voice NAME] [--whisper] [--warmup N] [--iters N] \\
  [--metal-prefill auto|cpu|packed|metal] \\
  [--voice-clone --clone-ref JSON --clone-target STR] \\
  [--max-tokens N]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown arg {other}"),
        }
        i += 1;
    }
    Ok(Args {
        weights,
        snac,
        text,
        voice,
        devices,
        whisper,
        voice_clone,
        clone_ref,
        clone_target,
        warmup,
        iters,
        max_tokens,
        metal_prefill,
    })
}

fn parse_devices(csv: &str) -> Result<Vec<Device>> {
    if csv.eq_ignore_ascii_case("all") {
        let mut out = vec![Device::Cpu];
        for d in [
            Device::Metal,
            Device::Mlx,
            Device::Cuda,
            Device::Rocm,
            Device::Gpu,
            Device::Vulkan,
        ] {
            if is_available(d)
                && !out.contains(&d)
                && (d == Device::Cpu || lm_kv_decode_supported(d))
            {
                out.push(d);
            }
        }
        return Ok(out);
    }
    csv.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(s).context("parse --devices"))
        .collect()
}

fn default_gguf() -> Option<PathBuf> {
    std::env::var("ORPHEUS_PRETRAINED_GGUF")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| {
            std::env::var("ORPHEUS_GGUF_PATH")
                .ok()
                .map(PathBuf::from)
                .filter(|p| p.is_file())
        })
        .or_else(|| {
            let p = PathBuf::from("/tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf");
            p.is_file().then_some(p)
        })
}

fn default_snac() -> Option<PathBuf> {
    std::env::var("ORPHEUS_SNAC_PATH")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| {
            let p = PathBuf::from("/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors");
            p.is_file().then_some(p)
        })
}

fn resolve_whisper_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("RLX_WHISPER_DIR") {
        let p = PathBuf::from(dir);
        if p.join("model.safetensors").is_file() {
            return Ok(p);
        }
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    for name in ["whisper-base.en", "whisper-small.en", "whisper-tiny.en"] {
        let p = root.join(".cache").join(name);
        if p.join("model.safetensors").is_file() {
            return Ok(p);
        }
    }
    bail!("Whisper weights not found — run `just fetch-whisper` or set RLX_WHISPER_DIR")
}

fn bench_named(
    gguf: &Path,
    snac: &Path,
    device: Device,
    text: &str,
    voice: &str,
    warmup: u32,
    iters: u32,
    max_tokens: u32,
    metal_prefill: MetalGgufPrefillMode,
    whisper_dir: Option<&Path>,
) -> Result<Row> {
    let backbone_opts = BackboneLoadOptions::for_tts(device).with_metal_prefill(metal_prefill);
    let mut tts = OrpheusTts::load_on_with_device(gguf, snac, device, backbone_opts)?;
    // Sampling (upstream Orpheus): greedy argmax is degenerate for this TTS LM.
    tts.config = GenerationConfig {
        max_new_tokens: max_tokens,
        ..GenerationConfig::default()
    };
    for _ in 0..warmup {
        let _ = tts.synthesize(text, Some(voice))?;
    }
    let t0 = Instant::now();
    let mut last = None;
    for _ in 0..iters {
        last = Some(tts.synthesize(text, Some(voice))?);
    }
    let wall = t0.elapsed();
    let out = last.expect("iter");
    let transcript = whisper_dir
        .map(|dir| transcribe(&out.samples, dir))
        .unwrap_or_default();
    let whisper_ok = whisper_dir.is_none()
        || (!transcript.trim().is_empty() && covers_reference(text, &transcript, 0.45));
    Ok(row_from(
        device_label(device),
        wall,
        out.code_count,
        out.samples.len(),
        whisper_ok,
        transcript,
    ))
}

fn expected_min_frames(text: &str) -> usize {
    let words = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .count()
        .max(1);
    (words + 1).max(4)
}

fn assert_synthesis_length(text: &str, codes: usize, samples: usize) -> Result<()> {
    let min_frames = expected_min_frames(text);
    let min_codes = min_frames * 7;
    if codes < min_codes {
        bail!("expected >= {min_codes} SNAC codes ({min_frames} frames) for {text:?}, got {codes}");
    }
    let frames = codes / 7;
    let min_samples = min_frames * SAMPLES_PER_FRAME;
    let max_samples = (frames + 1) * SAMPLES_PER_FRAME;
    if samples < min_samples * 3 / 4 {
        bail!(
            "audio too short for {text:?}: {samples} samples ({:.2}s), expected >= {:.2}s",
            samples as f64 / SAMPLE_RATE as f64,
            min_samples as f64 / SAMPLE_RATE as f64
        );
    }
    if samples > max_samples {
        bail!("audio too long for {frames} frames: {samples} samples (max {max_samples})");
    }
    Ok(())
}

fn bench_clone(
    gguf: &Path,
    snac: &Path,
    device: Device,
    ref_json: &Path,
    target: &str,
    warmup: u32,
    iters: u32,
    max_tokens: u32,
    metal_prefill: MetalGgufPrefillMode,
    whisper_dir: Option<&Path>,
) -> Result<Row> {
    let reference = VoiceCloneReference::load_json(ref_json)?;
    let backbone_opts = BackboneLoadOptions::for_tts(device).with_metal_prefill(metal_prefill);
    let mut tts = OrpheusTts::load_on_with_device(gguf, snac, device, backbone_opts)?;
    // Sampling (upstream Orpheus): greedy argmax is degenerate for this TTS LM.
    tts.config = GenerationConfig {
        max_new_tokens: max_tokens,
        ..GenerationConfig::default()
    };
    for _ in 0..warmup {
        let _ = tts.synthesize_voice_clone(&reference.transcript, &reference.token_ids, target)?;
    }
    let t0 = Instant::now();
    let mut last = None;
    for _ in 0..iters {
        last = Some(tts.synthesize_voice_clone(
            &reference.transcript,
            &reference.token_ids,
            target,
        )?);
    }
    let wall = t0.elapsed();
    let out = last.expect("iter");
    let transcript = whisper_dir
        .map(|dir| transcribe(&out.samples, dir))
        .unwrap_or_default();
    let whisper_ok = whisper_dir.is_none()
        || (!transcript.trim().is_empty() && covers_reference(target, &transcript, 0.4));
    Ok(row_from(
        &format!("{}-clone", device_label(device)),
        wall,
        out.code_count,
        out.samples.len(),
        whisper_ok,
        transcript,
    ))
}

fn row_from(
    label: &str,
    wall: std::time::Duration,
    codes: usize,
    samples: usize,
    whisper_ok: bool,
    transcript: String,
) -> Row {
    let audio_s = samples as f64 / SAMPLE_RATE as f64;
    let wall_ms = wall.as_secs_f64() * 1000.0;
    let rtf = if audio_s > 0.0 {
        wall.as_secs_f64() / audio_s
    } else {
        0.0
    };
    Row {
        label: label.to_string(),
        wall_ms,
        audio_s,
        rtf,
        codes,
        samples,
        whisper_ok,
        transcript,
    }
}

fn device_label(device: Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Metal => "metal",
        Device::Mlx => "mlx",
        Device::Cuda => "cuda",
        Device::Rocm => "rocm",
        Device::Gpu => "wgpu",
        Device::Vulkan => "vulkan",
        _ => "other",
    }
}

fn transcribe(pcm_24k: &[f32], whisper_dir: &Path) -> String {
    let mut pcm_16k = resample_linear(pcm_24k, SAMPLE_RATE, WHISPER_RATE as u32);
    if pcm_16k.len() < WHISPER_RATE / 2 {
        pcm_16k.resize(WHISPER_RATE / 2, 0.0);
    }
    WhisperRunner::builder()
        .weights(whisper_dir.join("model.safetensors"))
        .config_path(whisper_dir.join("config.json"))
        .tokenizer_path(whisper_dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .expect("whisper")
        .transcribe_greedy(&pcm_16k)
        .expect("transcribe")
}

fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let out_len = (samples.len() as u64 * to_hz as u64 / from_hz as u64).max(1) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * from_hz as f64 / to_hz as f64;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        let a = samples[idx.min(samples.len() - 1)];
        let b = samples[(idx + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

fn covers_reference(reference: &str, transcript: &str, min_ratio: f32) -> bool {
    let reference_words: Vec<String> = reference
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(str::to_string)
        .collect();
    if reference_words.is_empty() {
        return false;
    }
    let heard: Vec<String> = transcript
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(str::to_string)
        .collect();
    let hits = reference_words
        .iter()
        .filter(|w| heard.iter().any(|h| h == *w || h.contains(w.as_str())))
        .count();
    hits as f32 / reference_words.len() as f32 >= min_ratio
}
