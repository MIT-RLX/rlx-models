// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! `rlx-tts` — native RLX FastSpeech2 + WaveRNN (private local bundle).

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use rlx_tts::{
    BUNDLE_EXTRACT_HINT, DEFAULT_BUNDLE_DIR, DEFAULT_RLXP_NAME, RlxTts, VarianceControls,
    WaveRnnOpts, pack_directory, pack_rlxp, sanitize_manifest, write_wav,
};

const HELP: &str = "\
rlx-tts — RLX FastSpeech2 + WaveRNN

USAGE:
    rlx-tts --text \"<text>\" [--out FILE] [--bundle DIR]
    rlx-tts --phones \"p h o n e s\" [--out FILE] [--bundle DIR]
    rlx-tts --probe-bundle [--bundle DIR]
    rlx-tts --pack-rlxp [--bundle DIR] [--out FILE]
    rlx-tts --pack-gguf [--bundle DIR] [--out FILE]
    rlx-tts --sanitize-manifest PATH

OPTIONS:
    --text <TEXT>       Synthesize from text (Hydra frontend)
    --phones <PHONES>   Space-separated LHP phone symbols or integer ids
    --bundle <PATH>     Bundle dir, rlx-tts.rlxp, or legacy rlx-tts.gguf (default: weights/tts/rlx-tts)
    --out <FILE>        Output WAV path (default: out.wav); pack output when packing
    --seed <U64>        WaveRNN seed (default: 0 → 16807 with NativeBnns on macOS)
    --greedy            Argmax WaveRNN bits (no Gumbel noise)
    --duration-scale <F>  Duration variance scale (default: 1)
    --probe-bundle      Open bundle, print tensor counts, exit
    --pack-rlxp         Pack loose directory (or re-pack GGUF) → rlx-tts.rlxp
    --pack-gguf         Pack loose directory → legacy rlx-tts.gguf (no Python)
    --sanitize-manifest <PATH>  Normalize loose-bundle manifest.json
    -h, --help          Show this help

BUNDLE:
    Prefer weights/tts/rlx-tts/rlx-tts.rlxp (single-file pack).
    Legacy rlx-tts.gguf and loose safetensors + frontend/ still load.
";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut text: Option<String> = None;
    let mut phones: Option<String> = None;
    let mut out = PathBuf::from("out.wav");
    let mut out_set = false;
    let mut bundle: Option<PathBuf> = None;
    let mut seed: u64 = 0;
    let mut greedy = false;
    let mut duration_scale = 1.0f32;
    let mut probe_bundle = false;
    let mut pack_gguf = false;
    let mut pack_rlxp_flag = false;
    let mut sanitize: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--text" => text = Some(next(&mut args, "--text")?),
            "--phones" => phones = Some(next(&mut args, "--phones")?),
            "--out" => {
                out = PathBuf::from(next(&mut args, "--out")?);
                out_set = true;
            }
            "--bundle" => bundle = Some(PathBuf::from(next(&mut args, "--bundle")?)),
            "--seed" => seed = next(&mut args, "--seed")?.parse().context("parse --seed")?,
            "--greedy" => greedy = true,
            "--duration-scale" => {
                duration_scale = next(&mut args, "--duration-scale")?
                    .parse()
                    .context("parse --duration-scale")?
            }
            "--probe-bundle" => probe_bundle = true,
            "--pack-rlxp" => pack_rlxp_flag = true,
            "--pack-gguf" => pack_gguf = true,
            "--sanitize-manifest" => {
                sanitize = Some(PathBuf::from(next(&mut args, "--sanitize-manifest")?))
            }
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(());
            }
            other => bail!("unknown argument {other}\n\n{HELP}"),
        }
    }

    if let Some(path) = sanitize {
        sanitize_manifest(&path)?;
        println!("sanitized {}", path.display());
        return Ok(());
    }

    if pack_rlxp_flag {
        let dir = bundle.unwrap_or_else(|| PathBuf::from(DEFAULT_BUNDLE_DIR));
        let rlxp_out = if out_set {
            out
        } else {
            dir.join(DEFAULT_RLXP_NAME)
        };
        let report = pack_rlxp(&dir, &rlxp_out)?;
        println!(
            "wrote {} ({:.1} MiB; tensors={}, file_kvs={}, blobs={})",
            report.path.display(),
            report.bytes as f64 / (1024.0 * 1024.0),
            report.tensor_count,
            report.file_kv,
            report.blob_count
        );
        return Ok(());
    }

    if pack_gguf {
        let dir = bundle.unwrap_or_else(|| PathBuf::from(DEFAULT_BUNDLE_DIR));
        let gguf_out = if out_set {
            out
        } else {
            dir.join("rlx-tts.gguf")
        };
        let report = pack_directory(&dir, &gguf_out)?;
        println!(
            "wrote {} ({:.1} MiB; tensors={}, file_kvs={}, blobs={})",
            report.path.display(),
            report.bytes as f64 / (1024.0 * 1024.0),
            report.tensor_count,
            report.file_kv,
            report.blob_count
        );
        return Ok(());
    }

    let model = if let Some(dir) = bundle {
        RlxTts::open(dir).with_context(|| BUNDLE_EXTRACT_HINT.to_string())?
    } else {
        RlxTts::open_default().with_context(|| BUNDLE_EXTRACT_HINT.to_string())?
    };

    if probe_bundle {
        let m = model.manifest();
        println!("bundle:  {}", model.bundle_dir().display());
        println!("format:  {}", m.format);
        println!("voice:   {}", m.voice_identifier);
        println!("rate:    {} Hz", m.sample_rate_hz);
        println!(
            "tensors: enc={} dec={} wr={}",
            model.encoder_weights().len(),
            model.decoder_weights().len(),
            model.wavernn_weights().len()
        );
        return Ok(());
    }

    let ctrl = VarianceControls {
        duration_scale,
        ..VarianceControls::default()
    };
    let vocoder = {
        let mut v = WaveRnnOpts::product_default();
        v.seed = Some(if v.rng == rlx_tts::WaveRnnRng::NativeBnns && seed == 0 {
            16_807
        } else {
            seed
        });
        v.greedy = greedy;
        v
    };

    let audio = match (text, phones) {
        (Some(t), None) => model.synthesize_text(&t, &ctrl, &vocoder)?,
        (None, Some(p)) => model.synthesize_phone_string(&p, &ctrl, &vocoder)?,
        (Some(_), Some(_)) => bail!("pass only one of --text or --phones"),
        (None, None) => bail!("pass --text, --phones, --probe-bundle, --pack-rlxp, or --pack-gguf\n\n{HELP}"),
    };

    write_wav(&audio, &out)?;
    println!(
        "wrote {} ({:.2}s, {} Hz, peak={:.3})",
        out.display(),
        audio.duration_secs(),
        audio.sample_rate,
        audio.peak_amplitude()
    );
    Ok(())
}

fn next(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .with_context(|| format!("missing value for {flag}"))
}
