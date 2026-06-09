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

use crate::embed::decode_token_ids;
use crate::runner::VoxtralRunner;
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut wav: Option<PathBuf> = None;
    let mut prompt_ids: Option<Vec<u32>> = None;
    let mut device = "cpu".to_string();
    let mut max_tokens = 0usize;
    let mut transcribe = false;
    let mut language: Option<String> = None;
    let mut dry = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--config" => config = Some(req(args, &mut i)?.into()),
            "--wav" => wav = Some(req(args, &mut i)?.into()),
            "--prompt-ids" => {
                let s = req(args, &mut i)?;
                prompt_ids = Some(
                    s.split(',')
                        .map(|p| p.trim().parse::<u32>().context("--prompt-ids"))
                        .collect::<Result<Vec<_>>>()?,
                );
            }
            "--device" => device = req(args, &mut i)?,
            "--max-tokens" => {
                max_tokens = req(args, &mut i)?.parse().context("--max-tokens")?;
            }
            "--transcribe" => {
                transcribe = true;
                i += 1;
            }
            "--language" | "--lang" => language = Some(req(args, &mut i)?),
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "rlx-voxtral — Mistral Voxtral speech LM\n\
                     Flags: --weights PATH [--config PATH] [--wav PATH]\n\
                       [--prompt-ids 1,24,24,...] [--transcribe] [--lang en]\n\
                       [--device cpu|metal|cuda|…] [--max-tokens N] [--dry]\n\
                     \n\
                     --transcribe uses HF Whisper mel + Mistral transcription templates.\n\
                     Set RLX_VOXTRAL_PYTHON to a venv python with transformers + mistral-common."
                );
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let device = parse_standard_device("voxtral", &device)?;

    eprintln!(
        "[rlx-voxtral] weights={weights:?} device={device:?} wav={wav:?} transcribe={transcribe}"
    );

    let mut builder = VoxtralRunner::builder().weights(&weights).device(device);
    if let Some(cfg) = config {
        builder = builder.config_path(cfg);
    }
    if max_tokens > 0 {
        builder = builder.max_new_tokens(max_tokens);
    }
    let runner = builder.build()?;

    if dry {
        eprintln!(
            "[rlx-voxtral] dry run ok — audio_token_id={} text_layers={}",
            runner.config().audio_token_id,
            runner.config().text_config.num_hidden_layers
        );
        return Ok(());
    }

    let wav = wav.ok_or_else(|| anyhow!("--wav is required unless --dry"))?;

    let tokens = if transcribe {
        runner.transcribe_wav(&wav, language.as_deref())?
    } else {
        let (mel, _) = crate::audio::pcm_to_mel_and_prompt(
            runner.model_dir(),
            Some(&wav),
            language.as_deref(),
        )?;
        let prompt =
            prompt_ids.ok_or_else(|| anyhow!("--prompt-ids required unless --transcribe"))?;
        runner.generate(&prompt, &mel)?
    };

    let text = decode_token_ids(Some(runner.model_dir()), &tokens)?;
    eprintln!("[rlx-voxtral] token ids: {tokens:?}");
    eprintln!("[rlx-voxtral] text: {text}");
    Ok(())
}
