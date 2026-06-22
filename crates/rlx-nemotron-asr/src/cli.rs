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

//! `rlx-nemotron-asr` command line:
//!   * `transcribe --nemo <file> --wav <file> [--device cpu]`
//!   * `dump-keys  --nemo <file>` — config summary + state-dict tensor names
//!     (use this to reconcile exact key spellings for a checkpoint).

use std::path::PathBuf;

use anyhow::{Result, bail};
use rlx_cli::{parse_standard_device, req};
use rlx_nemo::NemoModel;
use rlx_runtime::Device;

use crate::config::AsrConfig;
use crate::{NemotronAsr, wav};

pub fn run(args: &[String]) -> Result<()> {
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    match cmd {
        "transcribe" => transcribe(&args[1..]),
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
        "rlx-nemotron-asr — NVIDIA Nemotron 3.5 ASR (FastConformer + RNN-T) on RLX\n\n\
         USAGE:\n  \
         rlx-nemotron-asr transcribe --nemo <file.nemo> --wav <audio.wav> [--lang en-US] [--device cpu|metal|mlx|cuda]\n  \
         rlx-nemotron-asr dump-keys  --nemo <file.nemo>\n"
    );
}

struct Args {
    nemo: PathBuf,
    wav: Option<PathBuf>,
    device: String,
    lang: String,
}

fn parse_args(args: &[String]) -> Result<Args> {
    let mut nemo: Option<PathBuf> = None;
    let mut wav: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut lang = "en-US".to_string();
    // `req` advances `i` past both the flag and its value, so the arms do
    // not increment again.
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--nemo" => nemo = Some(PathBuf::from(req(args, &mut i)?)),
            "--wav" => wav = Some(PathBuf::from(req(args, &mut i)?)),
            "--device" => device = req(args, &mut i)?,
            "--lang" => lang = req(args, &mut i)?,
            other => bail!("unexpected argument {other:?}"),
        }
    }
    let nemo = nemo.ok_or_else(|| anyhow::anyhow!("--nemo <file.nemo> is required"))?;
    Ok(Args {
        nemo,
        wav,
        device,
        lang,
    })
}

fn transcribe(args: &[String]) -> Result<()> {
    let a = parse_args(args)?;
    let wav_path = a
        .wav
        .ok_or_else(|| anyhow::anyhow!("--wav <audio.wav> is required"))?;
    let device: Device = parse_standard_device("nemotron-asr", &a.device)?;

    let mut asr = NemotronAsr::open(&a.nemo, device)?;
    if !asr.set_language(&a.lang) {
        eprintln!(
            "[rlx-nemotron-asr] warning: language {:?} not in prompt_dictionary; using en-US",
            a.lang
        );
    }
    let cfg = asr.config();

    let bytes = std::fs::read(&wav_path)?;
    let w = wav::parse(&bytes)?;
    let pcm = wav::resample(&w.samples, w.sample_rate, cfg.sample_rate as u32);
    eprintln!(
        "[rlx-nemotron-asr] device={device:?} samples={} ({} Hz -> {} Hz)",
        pcm.len(),
        w.sample_rate,
        cfg.sample_rate
    );

    let text = asr.transcribe(&pcm)?;
    println!("{text}");
    Ok(())
}

fn dump_keys(args: &[String]) -> Result<()> {
    let a = parse_args(args)?;
    let model = NemoModel::open(&a.nemo)?;
    let cfg = AsrConfig::from_nemo(model.config())?;
    eprintln!(
        "config: d_model={} layers={} heads={} ff={} subsample={} vocab={} pred_hidden={} joint_hidden={} langs={}",
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.ff_dim(),
        cfg.subsampling_factor,
        cfg.vocab_size,
        cfg.pred_hidden,
        cfg.joint_hidden,
        cfg.num_languages,
    );
    eprintln!("tokenizer artifacts:");
    for t in model.tokenizers() {
        eprintln!("  {} ({} bytes)", t.name, t.bytes.len());
    }
    println!("# {} tensors", model.len());
    for name in model.names() {
        let shape = model.shape_of(&name).unwrap_or(&[]);
        println!("{name}\t{shape:?}");
    }
    Ok(())
}
