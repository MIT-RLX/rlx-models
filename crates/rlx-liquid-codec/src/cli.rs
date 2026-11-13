// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! `rlx-liquid-codec encode|decode|roundtrip …`.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};

use crate::{LiquidCodec, read_24k, write_mono_16};

pub const USAGE: &str = "\
rlx-liquid-codec — LiquidAI LiquidCodec (24 kHz speech ↔ 12 × 12.5 Hz tokens)

USAGE:
  rlx-liquid-codec encode    --weights DIR --wav IN.wav -o codes.json [--stream]
  rlx-liquid-codec decode    --weights DIR --codes codes.json -o OUT.wav [--stream]
  rlx-liquid-codec roundtrip --weights DIR --wav IN.wav -o OUT.wav [--stream]

OPTIONS:
  --weights DIR|FILE  LiquidAI/Liquid-Codec directory, an LFM3-Audio directory
                      (its audio_tokenizer/), or a model.safetensors
  --device cpu|metal|mlx|cuda|wgpu|vulkan   (default cpu)
  --stream            encode / decode as a stream: 80 ms of samples or one
                      frame at a time (same codes and samples)
  --weight-storage f32|auto  graph weights as f32 (default; or RLX_WEIGHT_STORAGE)
                      or at the checkpoint's bf16 (lossless, half the memory)
  --timing            print graph builds (sizes) and the process footprint
  --max-whole-frames N  clips longer than N frames (default 375 = 30 s) encode
                      in KV-cached 64-frame chunks: memory linear in length

codes.json holds {\"audio_codes\": [[…frames…] × 12]}: codebook-major, the
layout of `LiquidCodecModel.encode(...).audio_codes[0]`.
";

struct Args {
    cmd: String,
    weights: PathBuf,
    device: String,
    wav: Option<PathBuf>,
    codes: Option<PathBuf>,
    out: Option<PathBuf>,
    stream: bool,
    max_whole_frames: Option<usize>,
    timing: bool,
    storage: Option<String>,
}

fn parse(args: &[String]) -> Result<Args> {
    let mut it = args.iter();
    let cmd = it
        .next()
        .ok_or_else(|| anyhow!("missing command\n\n{USAGE}"))?
        .clone();
    if cmd == "-h" || cmd == "--help" {
        print!("{USAGE}");
        std::process::exit(0);
    }
    let mut a = Args {
        cmd,
        weights: PathBuf::from("weights/audio/liquid-codec"),
        device: "cpu".into(),
        wav: None,
        codes: None,
        out: None,
        stream: false,
        max_whole_frames: None,
        timing: false,
        storage: None,
    };
    while let Some(flag) = it.next() {
        let mut val = || {
            it.next()
                .cloned()
                .ok_or_else(|| anyhow!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--weights" => a.weights = val()?.into(),
            "--device" => a.device = val()?,
            "--wav" => a.wav = Some(val()?.into()),
            "--codes" => a.codes = Some(val()?.into()),
            "-o" | "--out" => a.out = Some(val()?.into()),
            "--stream" => a.stream = true,
            "--timing" => a.timing = true,
            "--weight-storage" => a.storage = Some(val()?),
            "--max-whole-frames" => a.max_whole_frames = Some(val()?.parse()?),
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown option {other}\n\n{USAGE}"),
        }
    }
    Ok(a)
}

fn encode(codec: &LiquidCodec, pcm: &[f32], stream: bool) -> Result<Vec<Vec<u32>>> {
    if !stream {
        return codec.encode(pcm);
    }
    let mut s = codec.encode_stream();
    let mut out = Vec::new();
    for piece in pcm.chunks(codec.config().frame_size()) {
        out.extend(s.push(piece)?);
    }
    out.extend(s.finish()?);
    Ok(out)
}

fn decode(codec: &LiquidCodec, codes: &[Vec<u32>], stream: bool) -> Result<Vec<f32>> {
    if !stream {
        return codec.decode(codes);
    }
    let mut s = codec.decode_stream();
    let mut out = Vec::new();
    for f in codes {
        out.extend(s.push(std::slice::from_ref(f))?);
    }
    out.extend(s.finish());
    Ok(out)
}

/// Codebook-major `[n_q][frames]` ↔ frame-major `[frames][n_q]`.
fn transpose(rows: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let n = rows.first().map_or(0, Vec::len);
    (0..n)
        .map(|t| rows.iter().map(|r| r[t]).collect())
        .collect()
}

/// Run the CLI on `args` (without the program name).
pub fn run(args: &[String]) -> Result<()> {
    let a = parse(args)?;
    let device = rlx_cli::parse_device(&a.device)?;
    rlx_lfm3_audio::set_timing(a.timing);
    let need = |p: &Option<PathBuf>, what: &str| {
        p.clone()
            .ok_or_else(|| anyhow!("{} needs {what}\n\n{USAGE}", a.cmd))
    };
    let t0 = Instant::now();
    let mut codec = LiquidCodec::open(&a.weights, device)?;
    if let Some(s) = &a.storage {
        codec.set_weight_storage(s.parse()?);
    }
    if let Some(n) = a.max_whole_frames {
        codec.set_max_whole_frames(n);
    }
    let rate = codec.config().sampling_rate;
    eprintln!("[liquid-codec] loaded in {:.2?}", t0.elapsed());
    match a.cmd.as_str() {
        "encode" => {
            let (wav, out) = (need(&a.wav, "--wav")?, need(&a.out, "-o")?);
            let pcm = read_24k(&wav)?;
            let t = Instant::now();
            let codes = encode(&codec, &pcm, a.stream)?;
            let secs = pcm.len() as f64 / rate as f64;
            eprintln!(
                "[liquid-codec] {} frames from {secs:.2} s in {:.2?} (RTF {:.3})",
                codes.len(),
                t.elapsed(),
                t.elapsed().as_secs_f64() / secs
            );
            let rows: Vec<Vec<u32>> = (0..codec.config().n_q)
                .map(|q| codes.iter().map(|f| f[q]).collect())
                .collect();
            let json = serde_json::json!({ "audio_codes": rows });
            std::fs::write(&out, serde_json::to_string(&json)?)
                .with_context(|| format!("write {}", out.display()))?;
            eprintln!("[liquid-codec] wrote {}", out.display());
        }
        "decode" => {
            let (path, out) = (need(&a.codes, "--codes")?, need(&a.out, "-o")?);
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("read {}", path.display()))?;
            let v: serde_json::Value = serde_json::from_str(&text)?;
            let rows: Vec<Vec<u32>> = serde_json::from_value(
                v.get("audio_codes")
                    .cloned()
                    .ok_or_else(|| anyhow!("{}: no audio_codes", path.display()))?,
            )?;
            let t = Instant::now();
            let pcm = decode(&codec, &transpose(&rows), a.stream)?;
            eprintln!(
                "[liquid-codec] {:.2} s of audio in {:.2?}",
                pcm.len() as f64 / rate as f64,
                t.elapsed()
            );
            write_mono_16(&out, &pcm, rate)?;
            eprintln!("[liquid-codec] wrote {}", out.display());
        }
        "roundtrip" => {
            let (wav, out) = (need(&a.wav, "--wav")?, need(&a.out, "-o")?);
            let pcm = read_24k(&wav)?;
            let t = Instant::now();
            let codes = encode(&codec, &pcm, a.stream)?;
            let te = t.elapsed();
            let t = Instant::now();
            let mut rec = decode(&codec, &codes, a.stream)?;
            let td = t.elapsed();
            rec.truncate(pcm.len());
            let secs = pcm.len() as f64 / rate as f64;
            eprintln!(
                "[liquid-codec] {secs:.2} s, {} frames: encode {te:.2?} (RTF {:.3}), decode {td:.2?} (RTF {:.3})",
                codes.len(),
                te.as_secs_f64() / secs,
                td.as_secs_f64() / secs
            );
            write_mono_16(&out, &rec, rate)?;
            eprintln!("[liquid-codec] wrote {}", out.display());
        }
        other => bail!("unknown command {other}\n\n{USAGE}"),
    }
    Ok(())
}
