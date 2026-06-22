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

use crate::runner::AsrRunner;
use anyhow::{Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut wav: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut max_tokens = 0usize;
    let mut system = String::new();
    let mut dry = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--config" => config = Some(req(args, &mut i)?.into()),
            "--wav" => wav = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--max-tokens" => max_tokens = req(args, &mut i)?.parse()?,
            "--system" | "--context" => system = req(args, &mut i)?,
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "rlx-qwen3-asr — Qwen3-ASR speech recognition\n\
                     Flags: --weights PATH [--config PATH] --wav PATH.wav\n\
                       [--system TEXT] [--device cpu|metal|mlx|cuda|…]\n\
                       [--max-tokens N] [--dry]"
                );
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let device = parse_standard_device("qwen3-asr", &device)?;
    eprintln!("[rlx-qwen3-asr] weights={weights:?} device={device:?} wav={wav:?}");

    let mut builder = AsrRunner::builder().weights(&weights).device(device);
    if let Some(cfg) = config {
        builder = builder.config_path(cfg);
    }
    if max_tokens > 0 {
        builder = builder.max_new_tokens(max_tokens);
    }
    let runner = builder.build()?;

    if dry {
        eprintln!(
            "[rlx-qwen3-asr] dry run ok — audio_token_id={} audio_layers={} text_layers={}",
            runner.config().audio_token_id,
            runner.config().audio.num_hidden_layers,
            runner.config().text.num_hidden_layers
        );
        return Ok(());
    }

    let wav = wav.ok_or_else(|| anyhow!("--wav is required unless --dry"))?;

    #[cfg(feature = "tokenizer")]
    {
        let text = runner.transcribe_wav(&wav, &system)?;
        println!("{text}");
        Ok(())
    }
    #[cfg(not(feature = "tokenizer"))]
    {
        let _ = (wav, system);
        bail!("transcription requires the `tokenizer` feature (vocab.json + merges.txt)");
    }
}
