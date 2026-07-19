// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Produce a real WAV file through rlx-kyutai-tts. Layered, with the most
// faithful path first:
//
//   1. Native TTS via `KyutaiTtsSession::generate` — uses the rlx native
//      backbone + DepFormer + Mimi (full text→speech). Currently disabled
//      because the generation loop isn't wired yet.
//
//   2. Mimi codec round-trip on a reference WAV — encodes a real speech WAV
//      through Mimi (the same neural codec the Kyutai TTS LM emits into)
//      and decodes it back. Produces a real audio file using rlx-kyutai-tts's
//      audio output stack (rlx-mimi runtime dep). This is what runs today
//      when the Mimi sidecar weights are present.
//
//   3. Synthetic-codes fallback — generates a deterministic codes pattern
//      and decodes it. Always works (no fetch needed) but produces noise,
//      not speech. Use only as a smoke test.
//
// Run:
//   cargo run --example generate_wav -p rlx-kyutai-tts
//   cargo run --example generate_wav -p rlx-kyutai-tts -- --prompt "Hello." \
//        --out /tmp/hello.wav --reference assets/jfk/jfk_rust_speech.wav

use anyhow::{Context, Result};
use rlx_kyutai_tts::download::{default_kyutai_tts_dir, default_mimi_dir};
use rlx_kyutai_tts::session::{GenerationConfig, KyutaiTtsSession};
use rlx_kyutai_tts::{KyutaiTtsConfig, StreamLayout};
use rlx_mimi::audio::write_wav_mono;
use rlx_mimi::{MimiCodec, MimiCodes, SAMPLE_RATE as MIMI_RATE};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Args {
    prompt: String,
    reference: Option<PathBuf>,
    out_wav: PathBuf,
    mimi_dir: PathBuf,
    tts_dir: PathBuf,
    target_frames: usize,
    num_quantizers: usize,
}

fn parse_args() -> Result<Args> {
    let mut prompt = String::from("Bonjour, comment ça va ?");
    let mut reference: Option<PathBuf> = None;
    let mut out_wav = PathBuf::from("/tmp/rlx-kyutai-tts-out.wav");
    let mut mimi_dir = default_mimi_dir();
    let mut tts_dir = default_kyutai_tts_dir();
    let mut target_frames = 100usize; // ~8 s @ 12.5 Hz
    let mut num_quantizers = 8usize;

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--prompt" => {
                i += 1;
                prompt = argv.get(i).context("--prompt TEXT")?.clone();
            }
            "--reference" => {
                i += 1;
                reference = Some(PathBuf::from(argv.get(i).context("--reference PATH")?));
            }
            "--out" => {
                i += 1;
                out_wav = PathBuf::from(argv.get(i).context("--out PATH")?);
            }
            "--mimi-dir" => {
                i += 1;
                mimi_dir = PathBuf::from(argv.get(i).context("--mimi-dir PATH")?);
            }
            "--tts-dir" => {
                i += 1;
                tts_dir = PathBuf::from(argv.get(i).context("--tts-dir PATH")?);
            }
            "--frames" => {
                i += 1;
                target_frames = argv.get(i).context("--frames N")?.parse()?;
            }
            "--codebooks" => {
                i += 1;
                num_quantizers = argv.get(i).context("--codebooks N")?.parse()?;
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: generate_wav [--prompt TEXT] [--reference WAV] [--out WAV]\n\
                     \n\
                     Layered audio generation through rlx-kyutai-tts.\n\
                     If --reference is set and Mimi weights exist, runs a Mimi\n\
                     round-trip on the reference. Otherwise emits synthetic codes.\n\
                     \n\
                     Options:\n\
                       --prompt TEXT       text prompt (used only by future native TTS)\n\
                       --reference WAV     reference speech WAV for Mimi round-trip\n\
                       --out WAV           output WAV path (default: /tmp/rlx-kyutai-tts-out.wav)\n\
                       --mimi-dir DIR      Mimi codec dir (default: .cache/mimi)\n\
                       --tts-dir DIR       Kyutai TTS dir (default: .cache/kyutai-tts-1.6b-en_fr)\n\
                       --frames N          target Mimi frames for synthetic mode (default: 100)\n\
                       --codebooks N       Mimi codebooks to use (default: 8)\n"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown arg: {other} (try --help)"),
        }
        i += 1;
    }

    Ok(Args {
        prompt,
        reference,
        out_wav,
        mimi_dir,
        tts_dir,
        target_frames,
        num_quantizers,
    })
}

/// Path 1: native TTS through `KyutaiTtsSession::generate`.
fn try_native_tts(args: &Args) -> Result<Option<Vec<f32>>> {
    if !args
        .tts_dir
        .join("dsm_tts_1e68beda@240.safetensors")
        .is_file()
    {
        eprintln!(
            "[path 1] native TTS: skipping — no LM weights at {}",
            args.tts_dir.display()
        );
        return Ok(None);
    }
    let mut session = KyutaiTtsSession::open_with_checkpoint(
        &args.tts_dir,
        &args.mimi_dir,
        Device::Cpu,
        rlx_kyutai_tts::KyutaiTtsCheckpoint::V1_6bEnFr,
    )?;
    let cfg = GenerationConfig {
        max_steps: args.target_frames,
        ..GenerationConfig::default()
    };
    match session.generate(&args.prompt, &cfg) {
        Ok(result) => {
            eprintln!(
                "[path 1] native TTS produced {} samples ({} frames) @ {} Hz",
                result.samples.len(),
                result.audio_frames.len(),
                result.sample_rate
            );
            Ok(Some(result.samples))
        }
        Err(e) => {
            eprintln!("[path 1] native TTS not yet wired: {e:#}");
            Ok(None)
        }
    }
}

/// Path 2: Mimi codec round-trip of a reference WAV.
fn try_mimi_roundtrip(args: &Args, codec: &mut MimiCodec) -> Result<Option<Vec<f32>>> {
    let Some(reference) = args.reference.as_ref() else {
        eprintln!("[path 2] mimi round-trip: skipping — no --reference WAV given");
        return Ok(None);
    };
    if !reference.is_file() {
        anyhow::bail!("--reference {} not found", reference.display());
    }
    eprintln!(
        "[path 2] mimi round-trip: encoding {} → {} codebooks → decode",
        reference.display(),
        args.num_quantizers
    );
    let codes = codec
        .encode_wav(reference, Some(args.num_quantizers))
        .with_context(|| format!("encode {}", reference.display()))?;
    eprintln!(
        "[path 2] encoded {} frames × {} codebooks",
        codes.frames.len(),
        codes.num_quantizers
    );
    let pcm = codec.decode_codes(&codes)?;
    eprintln!("[path 2] decoded {} samples", pcm.len());
    Ok(Some(pcm))
}

/// Path 3: decode a synthetic codes pattern through Mimi.
///
/// Emits a deterministic codebook walk so the output is reproducible.
/// Not speech-like — useful as a smoke test of the audio output pipeline.
fn try_synthetic_codes(args: &Args, codec: &mut MimiCodec) -> Result<Vec<f32>> {
    let tts_cfg = KyutaiTtsConfig::v1_6b_en_fr();
    let layout = StreamLayout::from_config(&tts_cfg);
    let frames = args.target_frames.max(layout.total_steps_for(0));
    let nq = args.num_quantizers.min(layout.num_audio_codebooks());

    eprintln!(
        "[path 3] synthetic codes: {frames} frames × {nq} codebooks (card={})",
        tts_cfg.card
    );

    // Reproducible LCG → codes well within [0, card).
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut codes_frames: Vec<Vec<u32>> = Vec::with_capacity(frames);
    for _ in 0..frames {
        let mut row = Vec::with_capacity(nq);
        for _ in 0..nq {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let code = ((state >> 33) as u32) % (tts_cfg.card as u32);
            row.push(code);
        }
        codes_frames.push(row);
    }

    let codes = MimiCodes {
        frames: codes_frames,
        num_quantizers: nq,
    };
    let pcm = codec.decode_codes(&codes)?;
    eprintln!("[path 3] decoded {} samples", pcm.len());
    Ok(pcm)
}

fn pcm_peak(pcm: &[f32]) -> f32 {
    pcm.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
}

fn ensure_mimi_dir(mimi_dir: &Path) -> Result<()> {
    let cfg = mimi_dir.join("config.json");
    let st = mimi_dir.join("model.safetensors");
    if !cfg.is_file() || !st.is_file() {
        anyhow::bail!(
            "Mimi codec missing at {}. Fetch with:\n  cargo run -p rlx-mimi --features hf-download -- --fetch --model-dir {}",
            mimi_dir.display(),
            mimi_dir.display()
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    eprintln!("rlx-kyutai-tts: generate_wav");
    eprintln!("  tts_dir:  {}", args.tts_dir.display());
    eprintln!("  mimi_dir: {}", args.mimi_dir.display());
    eprintln!("  prompt:   {:?}", args.prompt);
    eprintln!("  out:      {}", args.out_wav.display());

    // Try path 1: native TTS through KyutaiTtsSession.
    if let Some(pcm) = try_native_tts(&args)? {
        write_wav_mono(&args.out_wav, &pcm, MIMI_RATE)?;
        eprintln!(
            "wrote {} ({} samples, peak {:.3})",
            args.out_wav.display(),
            pcm.len(),
            pcm_peak(&pcm)
        );
        return Ok(());
    }

    // Paths 2 and 3 both need the Mimi codec.
    ensure_mimi_dir(&args.mimi_dir)?;
    let mut codec = MimiCodec::open(&args.mimi_dir)
        .with_context(|| format!("open Mimi codec at {}", args.mimi_dir.display()))?;
    eprintln!("loaded Mimi codec from {}", args.mimi_dir.display());

    // Path 2: round-trip a reference WAV.
    if let Some(pcm) = try_mimi_roundtrip(&args, &mut codec)? {
        write_wav_mono(&args.out_wav, &pcm, MIMI_RATE)?;
        eprintln!(
            "wrote {} ({} samples, peak {:.3})",
            args.out_wav.display(),
            pcm.len(),
            pcm_peak(&pcm)
        );
        return Ok(());
    }

    // Path 3: synthetic codes — last-resort smoke test.
    let pcm = try_synthetic_codes(&args, &mut codec)?;
    // Normalise so the smoke-test file isn't deafening.
    let peak = pcm_peak(&pcm).max(1e-6);
    let scale = if peak > 0.95 { 0.5 / peak } else { 1.0 };
    let pcm_norm: Vec<f32> = pcm.iter().map(|v| v * scale).collect();
    write_wav_mono(&args.out_wav, &pcm_norm, MIMI_RATE)?;
    eprintln!(
        "wrote {} ({} samples, raw peak {:.3} → normalised)",
        args.out_wav.display(),
        pcm_norm.len(),
        peak
    );
    eprintln!(
        "(synthetic codes — not speech. For real audio, pass --reference <speech.wav>\n\
         or fetch Kyutai TTS LM weights with `rlx-kyutai-tts --fetch`.)"
    );
    Ok(())
}
