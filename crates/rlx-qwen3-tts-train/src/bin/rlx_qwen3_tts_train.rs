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

//! Qwen3-TTS training CLI (JFK custom voice).

use anyhow::{Context, Result};
use rlx_qwen3_tts_train::config::JfkLoraConfig;
use rlx_qwen3_tts_train::jfk_lora::train_jfk_lora;
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "help".into());
    match cmd.as_str() {
        "jfk-lora" => {
            let mut cfg = JfkLoraConfig::default();
            cfg.grad_accum = rlx_qwen3_tts_train::config::env_usize(
                "RLX_QWEN3_TTS_TRAIN_GRAD_ACCUM",
                cfg.grad_accum,
            );
            while let Some(a) = args.next() {
                match a.as_str() {
                    "--model-dir" => {
                        cfg.model_dir = PathBuf::from(args.next().context("--model-dir")?)
                    }
                    "--train-jsonl" => {
                        cfg.train_jsonl = PathBuf::from(args.next().context("--train-jsonl")?)
                    }
                    "--out-dir" => cfg.out_dir = PathBuf::from(args.next().context("--out-dir")?),
                    "--device" => cfg.device = Some(args.next().context("--device")?),
                    "--speaker" => cfg.speaker = args.next().context("--speaker")?,
                    "--epochs" => cfg.epochs = args.next().context("--epochs")?.parse()?,
                    "--steps-per-epoch" => {
                        cfg.steps_per_epoch = args.next().context("--steps-per-epoch")?.parse()?
                    }
                    "--rank" => cfg.rank = args.next().context("--rank")?.parse()?,
                    "--max-seq" => cfg.max_seq = args.next().context("--max-seq")?.parse()?,
                    "--n-layers" => cfg.n_layers = args.next().context("--n-layers")?.parse()?,
                    "--grad-accum" => {
                        cfg.grad_accum = args.next().context("--grad-accum")?.parse()?
                    }
                    "--max-clips" => cfg.max_clips = args.next().context("--max-clips")?.parse()?,
                    "--cache" => {
                        cfg.cache_path = Some(PathBuf::from(args.next().context("--cache")?))
                    }
                    "--lr" => cfg.lr = args.next().context("--lr")?.parse()?,
                    "-v" | "--verbose" => cfg.verbose = true,
                    other => anyhow::bail!("unknown flag {other}"),
                }
            }
            train_jfk_lora(&cfg)?;
        }
        "help" | "-h" | "--help" => {
            eprintln!(
                "usage:\n  \
                 rlx-qwen3-tts-train jfk-lora --model-dir DIR --train-jsonl PATH --out-dir DIR \
                 [--device mlx|metal|cpu] [--speaker jfk] [--epochs N] [--steps-per-epoch N] [-v]"
            );
        }
        other => anyhow::bail!("unknown command {other}"),
    }
    Ok(())
}
