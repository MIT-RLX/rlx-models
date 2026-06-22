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

//! `rlx-funasr` command line.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use rlx_cli::parse_standard_device;
use rlx_runtime::Device;

/// Read the value following a flag at `args[*i]`, advancing `*i` past the value
/// only (the loop's trailing `i += 1` advances past the flag).
fn val(args: &[String], i: &mut usize) -> Result<String> {
    let v = args
        .get(*i + 1)
        .ok_or_else(|| anyhow!("missing value for {}", args[*i]))?
        .clone();
    *i += 1;
    Ok(v)
}

use crate::paraformer::Paraformer;
use crate::pipeline::FunPipeline;
use crate::punc::CtTransformer;
use crate::runner;
use crate::sensevoice::SenseVoice;
use crate::speaker::CamPlus;
use crate::vad::FsmnVad;

/// Dispatch a `rlx-funasr` subcommand.
pub fn run(args: &[String]) -> Result<()> {
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    match cmd {
        "transcribe" => transcribe(&args[1..]),
        "vad" => vad(&args[1..]),
        "punc" => punc(&args[1..]),
        "spk" => spk(&args[1..]),
        "pipeline" => pipeline(&args[1..]),
        "stream" => stream(&args[1..]),
        "dump-keys" => dump_keys(&args[1..]),
        "help" | "-h" | "--help" => {
            print_usage();
            Ok(())
        }
        other => {
            eprintln!("unknown command {other:?}\n");
            print_usage();
            bail!("unknown command {other:?}");
        }
    }
}

fn print_usage() {
    eprintln!(
        "rlx-funasr — FunASR (Paraformer / SenseVoice / FSMN-VAD / CT-Transformer / CAM++) on RLX\n\n\
         USAGE:\n  \
         rlx-funasr transcribe --dir <model_dir> --wav <a.wav> [--device cpu|metal|mlx|cuda|gpu] [--type paraformer|sensevoice] [--lang auto] [--itn]\n  \
         rlx-funasr vad        --dir <vad_dir>   --wav <a.wav> [--device ...]\n  \
         rlx-funasr punc       --dir <punc_dir>  --text \"...\" [--device ...]\n  \
         rlx-funasr spk        --dir <spk_dir>   --wav <a.wav> [--device ...]\n  \
         rlx-funasr pipeline   --vad <d> --asr <d> [--punc <d>] [--spk <d>] --wav <a.wav> [--device ...]\n  \
         rlx-funasr stream     --vad <d> --asr <d> [--punc <d>] --wav <a.wav> [--chunk-ms 500] [--device ...]\n  \
         rlx-funasr dump-keys  --dir <model_dir>\n"
    );
}

fn load_pcm(path: &Path, sr: u32) -> Result<Vec<f32>> {
    crate::audio::load_mono(path, sr)
}

fn transcribe(args: &[String]) -> Result<()> {
    let mut dir = None;
    let mut wav_path = None;
    let mut device = "cpu".to_string();
    let mut ty: Option<String> = None;
    let mut lang = "auto".to_string();
    let mut itn = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => dir = Some(PathBuf::from(val(args, &mut i)?)),
            "--wav" => wav_path = Some(PathBuf::from(val(args, &mut i)?)),
            "--device" => device = val(args, &mut i)?,
            "--type" => ty = Some(val(args, &mut i)?),
            "--lang" => lang = val(args, &mut i)?,
            "--itn" => itn = true,
            other => bail!("unexpected argument {other:?}"),
        }
        i += 1;
    }
    let dir = dir.ok_or_else(|| anyhow!("--dir is required"))?;
    let wav_path = wav_path.ok_or_else(|| anyhow!("--wav is required"))?;
    let device: Device = parse_standard_device("funasr", &device)?;
    let pcm = load_pcm(&wav_path, 16_000)?;

    let kind = ty.or_else(|| runner::detect_kind(&dir).map(|k| format!("{k:?}").to_lowercase()));
    let is_sense = matches!(kind.as_deref(), Some(s) if s.contains("sense"));
    if is_sense {
        let m = SenseVoice::open(&dir, device)?;
        let r = m.transcribe(&pcm, &lang, itn)?;
        if !r.tags.is_empty() {
            eprintln!("[tags] {}", r.tags.join(" "));
        }
        println!("{}", r.text);
    } else {
        let m = Paraformer::open(&dir, device)?;
        println!("{}", m.transcribe(&pcm)?);
    }
    Ok(())
}

fn vad(args: &[String]) -> Result<()> {
    let (dir, wav_path, device) = dir_wav_device(args)?;
    let m = FsmnVad::open(&dir, device)?;
    let pcm = load_pcm(&wav_path, 16_000)?;
    for (s, e) in m.segments(&pcm)? {
        println!("{s:.0}\t{e:.0}");
    }
    Ok(())
}

fn punc(args: &[String]) -> Result<()> {
    let mut dir = None;
    let mut text = None;
    let mut device = "cpu".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => dir = Some(PathBuf::from(val(args, &mut i)?)),
            "--text" => text = Some(val(args, &mut i)?),
            "--device" => device = val(args, &mut i)?,
            other => bail!("unexpected argument {other:?}"),
        }
        i += 1;
    }
    let dir = dir.ok_or_else(|| anyhow!("--dir is required"))?;
    let text = text.ok_or_else(|| anyhow!("--text is required"))?;
    let device: Device = parse_standard_device("funasr", &device)?;
    let m = CtTransformer::open(&dir, device)?;
    println!("{}", m.restore(&text)?);
    Ok(())
}

fn spk(args: &[String]) -> Result<()> {
    let (dir, wav_path, device) = dir_wav_device(args)?;
    let m = CamPlus::open(&dir, device)?;
    let pcm = load_pcm(&wav_path, 16_000)?;
    let emb = m.embedding(&pcm)?;
    eprintln!("[spk] {}-d embedding", emb.len());
    let head: Vec<String> = emb.iter().take(8).map(|v| format!("{v:.4}")).collect();
    println!("{}", head.join(" "));
    Ok(())
}

fn pipeline(args: &[String]) -> Result<()> {
    let mut vad_dir = None;
    let mut asr_dir = None;
    let mut punc_dir = None;
    let mut spk_dir = None;
    let mut wav_path = None;
    let mut device = "cpu".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--vad" => vad_dir = Some(PathBuf::from(val(args, &mut i)?)),
            "--asr" => asr_dir = Some(PathBuf::from(val(args, &mut i)?)),
            "--punc" => punc_dir = Some(PathBuf::from(val(args, &mut i)?)),
            "--spk" => spk_dir = Some(PathBuf::from(val(args, &mut i)?)),
            "--wav" => wav_path = Some(PathBuf::from(val(args, &mut i)?)),
            "--device" => device = val(args, &mut i)?,
            other => bail!("unexpected argument {other:?}"),
        }
        i += 1;
    }
    let asr_dir = asr_dir.ok_or_else(|| anyhow!("--asr is required"))?;
    let wav_path = wav_path.ok_or_else(|| anyhow!("--wav is required"))?;
    let device: Device = parse_standard_device("funasr", &device)?;

    let mut pipe = FunPipeline::new().with_asr(runner::open_asr(&asr_dir, device)?);
    if let Some(d) = vad_dir {
        pipe = pipe.with_vad(FsmnVad::open(&d, device)?);
    }
    if let Some(d) = punc_dir {
        pipe = pipe.with_punc(CtTransformer::open(&d, device)?);
    }
    if let Some(d) = spk_dir {
        pipe = pipe.with_speaker(CamPlus::open(&d, device)?);
    }
    let pcm = load_pcm(&wav_path, 16_000)?;
    let res = pipe.run(&pcm)?;
    for s in &res.segments {
        let spk = s
            .speaker
            .as_ref()
            .map(|e| format!(" spk[{}]", e.len()))
            .unwrap_or_default();
        println!("[{:.0}-{:.0}ms]{spk} {}", s.start_ms, s.end_ms, s.text);
    }
    println!("\n{}", res.text);
    Ok(())
}

fn stream(args: &[String]) -> Result<()> {
    let mut vad_dir = None;
    let mut asr_dir = None;
    let mut punc_dir = None;
    let mut wav_path = None;
    let mut device = "cpu".to_string();
    let mut chunk_ms = 500.0f32;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--vad" => vad_dir = Some(PathBuf::from(val(args, &mut i)?)),
            "--asr" => asr_dir = Some(PathBuf::from(val(args, &mut i)?)),
            "--punc" => punc_dir = Some(PathBuf::from(val(args, &mut i)?)),
            "--wav" => wav_path = Some(PathBuf::from(val(args, &mut i)?)),
            "--chunk-ms" => chunk_ms = val(args, &mut i)?.parse().unwrap_or(500.0),
            "--device" => device = val(args, &mut i)?,
            other => bail!("unexpected argument {other:?}"),
        }
        i += 1;
    }
    let vad_dir = vad_dir.ok_or_else(|| anyhow!("--vad is required"))?;
    let asr_dir = asr_dir.ok_or_else(|| anyhow!("--asr is required"))?;
    let wav_path = wav_path.ok_or_else(|| anyhow!("--wav is required"))?;
    let device: Device = parse_standard_device("funasr", &device)?;

    let mut rec = crate::streaming::StreamingRecognizer::new(
        FsmnVad::open(&vad_dir, device)?,
        runner::open_asr(&asr_dir, device)?,
    );
    if let Some(d) = punc_dir {
        rec = rec.with_punc(CtTransformer::open(&d, device)?);
    }
    let pcm = load_pcm(&wav_path, 16_000)?;
    let chunk = ((chunk_ms / 1000.0) * 16_000.0) as usize;
    for c in pcm.chunks(chunk.max(1)) {
        for s in rec.accept(c)? {
            println!("[{:.0}-{:.0}ms] {}", s.start_ms, s.end_ms, s.text);
        }
    }
    for s in rec.finalize()? {
        println!("[{:.0}-{:.0}ms] {}", s.start_ms, s.end_ms, s.text);
    }
    Ok(())
}

fn dump_keys(args: &[String]) -> Result<()> {
    let mut dir = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => dir = Some(PathBuf::from(val(args, &mut i)?)),
            "--device" => {
                let _ = val(args, &mut i)?;
            }
            other => bail!("unexpected argument {other:?}"),
        }
        i += 1;
    }
    let dir = dir.ok_or_else(|| anyhow!("--dir is required"))?;
    let wm = crate::weights::load_dir(&dir)?;
    let mut keys: Vec<String> = wm.keys().map(|s| s.to_string()).collect();
    keys.sort();
    println!("# {} tensors", keys.len());
    for k in keys {
        let shape = wm.get(&k).map(|(_, s)| s.to_vec()).unwrap_or_default();
        println!("{k}\t{shape:?}");
    }
    Ok(())
}

fn dir_wav_device(args: &[String]) -> Result<(PathBuf, PathBuf, Device)> {
    let mut dir = None;
    let mut wav_path = None;
    let mut device = "cpu".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => dir = Some(PathBuf::from(val(args, &mut i)?)),
            "--wav" => wav_path = Some(PathBuf::from(val(args, &mut i)?)),
            "--device" => device = val(args, &mut i)?,
            other => bail!("unexpected argument {other:?}"),
        }
        i += 1;
    }
    let dir = dir.ok_or_else(|| anyhow!("--dir is required"))?;
    let wav_path = wav_path.ok_or_else(|| anyhow!("--wav is required"))?;
    let device: Device = parse_standard_device("funasr", &device)?;
    Ok((dir, wav_path, device))
}
