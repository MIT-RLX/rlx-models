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

//! Voice clone CLI — clone any speaker, synthesize any text in their voice.
//!
//! Uses the high-level [`rlx_qwen3_tts::VoiceClone`] API for the actual work;
//! everything in this file is argument parsing + I/O.
//!
//! Two modes:
//!
//! Single-target:
//!   jfk_voice_clone --model-dir <Base> --ref-wav <wav>
//!                   --target-text <str> --out-wav <wav>
//!
//! Batch (amortize the ~1 s of model open across many clones):
//!   jfk_voice_clone --model-dir <Base> --ref-wav <wav>
//!                   --targets-file <file> --out-dir <dir>
//!
//! Targets file format (one per line, optional `name|` prefix):
//!   # comments allowed
//!   ask_not|Ask not what your country can do for you, ...
//!   plain target text without a label gets clone_0001.wav

use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::parse_device;
use rlx_qwen3_tts::VoiceClone;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    model_dir: PathBuf,
    ref_wav: PathBuf,
    target_text: Option<String>,
    out_wav: Option<PathBuf>,
    targets_file: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    device: Device,
    max_frames: usize,
}

fn parse_args() -> Result<Args> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut model_dir: Option<PathBuf> = None;
    let mut ref_wav: Option<PathBuf> = None;
    let mut target_text: Option<String> = None;
    let mut out_wav: Option<PathBuf> = None;
    let mut targets_file: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut device = Device::Metal;
    let mut max_frames = 256usize;
    let mut i = 0;
    while i < raw.len() {
        let take = |i: &mut usize| -> Result<String> {
            *i += 1;
            raw.get(*i)
                .cloned()
                .ok_or_else(|| anyhow!("missing value for {}", raw[*i - 1]))
        };
        match raw[i].as_str() {
            "--model-dir" => model_dir = Some(PathBuf::from(take(&mut i)?)),
            "--ref-wav" => ref_wav = Some(PathBuf::from(take(&mut i)?)),
            "--target-text" => target_text = Some(take(&mut i)?),
            "--out-wav" => out_wav = Some(PathBuf::from(take(&mut i)?)),
            "--targets-file" => targets_file = Some(PathBuf::from(take(&mut i)?)),
            "--out-dir" => out_dir = Some(PathBuf::from(take(&mut i)?)),
            "--device" => device = parse_device(&take(&mut i)?)?,
            "--max-frames" => max_frames = take(&mut i)?.parse()?,
            "-h" | "--help" => {
                eprintln!(
                    "Usage: jfk_voice_clone --model-dir <Base> --ref-wav <wav> \
                     (--target-text <str> --out-wav <wav> | --targets-file <file> --out-dir <dir>) \
                     [--device cpu|metal|mlx|cuda] [--max-frames N]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown arg {other:?}"),
        }
        i += 1;
    }
    let single = target_text.is_some() || out_wav.is_some();
    let batch = targets_file.is_some() || out_dir.is_some();
    if single && batch {
        bail!(
            "pass either single (--target-text + --out-wav) or batch (--targets-file + --out-dir), not both"
        );
    }
    if !single && !batch {
        bail!("missing target: pass --target-text + --out-wav, or --targets-file + --out-dir");
    }
    if single && (target_text.is_none() || out_wav.is_none()) {
        bail!("single mode needs both --target-text and --out-wav");
    }
    if batch && (targets_file.is_none() || out_dir.is_none()) {
        bail!("batch mode needs both --targets-file and --out-dir");
    }
    Ok(Args {
        model_dir: model_dir.context("--model-dir required")?,
        ref_wav: ref_wav.context("--ref-wav required")?,
        target_text,
        out_wav,
        targets_file,
        out_dir,
        device,
        max_frames,
    })
}

/// Parse a targets file; returns Vec<(name, text)>. Comments (`#`) and blank
/// lines are skipped. Lines without `name|` get auto-numbered.
fn read_targets(path: &Path) -> Result<Vec<(String, String)>> {
    let body = std::fs::read_to_string(path)
        .with_context(|| format!("read targets file {}", path.display()))?;
    let mut out = Vec::new();
    let mut auto = 1usize;
    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, text) = match line.split_once('|') {
            Some((n, t)) => (n.trim().to_string(), t.trim().to_string()),
            None => (format!("clone_{:04}", auto), line.to_string()),
        };
        if name.is_empty() || text.is_empty() {
            bail!("bad line in {}: {:?}", path.display(), raw);
        }
        auto += 1;
        out.push((name, text));
    }
    if out.is_empty() {
        bail!("targets file {} is empty", path.display());
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    eprintln!(
        "[clone] model={} ref={} device={:?}",
        args.model_dir.display(),
        args.ref_wav.display(),
        args.device
    );

    // Open the model once and reuse across all clones.
    let t = Instant::now();
    let mut tts = VoiceClone::open_with_max_frames(&args.model_dir, args.device, args.max_frames)?;
    eprintln!("[clone] model opened: {:.2}s", t.elapsed().as_secs_f64());

    // Extract the reference once — fast (~50 ms).
    let t = Instant::now();
    let reference = tts.extract_reference(&args.ref_wav)?;
    eprintln!(
        "[clone] reference: {:.2}s ({} dims, norm {:.3})",
        t.elapsed().as_secs_f64(),
        reference.dim(),
        reference.norm()
    );

    if let Some(text) = args.target_text.as_deref() {
        let out = args.out_wav.as_ref().unwrap();
        let t = Instant::now();
        tts.generate_to_wav(&reference, text, out)?;
        eprintln!("[clone] generated: {:.2}s", t.elapsed().as_secs_f64());
        println!("wrote {}", out.display());
    } else {
        let targets = read_targets(args.targets_file.as_ref().unwrap())?;
        let out_dir = args.out_dir.as_ref().unwrap();
        std::fs::create_dir_all(out_dir)?;
        let n = targets.len();
        eprintln!("[clone] batch: {n} targets");
        let t_batch = Instant::now();
        for (idx, (name, text)) in targets.iter().enumerate() {
            let out = out_dir.join(format!("{name}.wav"));
            let t = Instant::now();
            tts.generate_to_wav(&reference, text, &out)?;
            eprintln!(
                "  [{}/{}] {name:<16} {:.2}s  →  {}",
                idx + 1,
                n,
                t.elapsed().as_secs_f64(),
                out.display()
            );
        }
        eprintln!(
            "[clone] batch total: {:.2}s ({} clones)",
            t_batch.elapsed().as_secs_f64(),
            n
        );
    }
    Ok(())
}
