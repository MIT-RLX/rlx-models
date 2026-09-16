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

//! `rlx-fireredaudio` CLI — ASR / understand / (generation stub).

use crate::prompt::{DEFAULT_ASR_PROMPT, EditType, FireRedTask};
use crate::runner::FireRedRunner;
use anyhow::{Result, anyhow, bail};
use rlx_runtime::Device;
use std::path::PathBuf;

const HELP: &str = "\
rlx-fireredaudio — FireRedAudio unified audio LM (ASR / understand / TTS scaffold)

USAGE:
    rlx-fireredaudio --weights <DIR> --task <TASK> [OPTIONS]

MODEL:
    --weights <PATH>     FireRedAudio HF dir (…/FireRedAudio or parent with nested folder)
    --device <NAME>      auto|cpu|metal|mlx|cuda|rocm|gpu|vulkan|coreml (default cpu)
    --max-new-tokens <N> generation budget (default 300)
    --max-seq <N>        backbone context ceiling (default 4096)

TASKS:
    --task asr           speech recognition (default prompt)
    --task understand    audio QA (needs --prompt)
    --task tts|edit|voice_design
                         generation API is present; RedAE+DiT graphs not compiled yet

INPUT:
    --audio <WAV>        16 kHz preferred (resampled if needed)
    --prompt <TEXT>      understand question / edit instruction / voice-design text
    --prompt-audio <WAV> TTS reference clip
    --prompt-text <T>    TTS reference transcript
    --target-text <T>    TTS synthesis text
    --language <zh|en>   TTS language (default zh)
    --edit-type <T>      semantic|acoustic (default semantic)
    --enable-thinking    open CoT block (understand only)
    --show-prompt        print ChatML and exit (no weights)
    -h, --help
";

pub fn cli_run(args: &[String]) -> Result<()> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{HELP}");
        return Ok(());
    }

    let mut weights: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut max_new_tokens = 300usize;
    let mut max_seq = 4096usize;
    let mut task = FireRedTask::Asr;
    let mut audio: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut prompt_audio: Option<PathBuf> = None;
    let mut prompt_text: Option<String> = None;
    let mut target_text: Option<String> = None;
    let mut language = "zh".to_string();
    let mut edit_type = EditType::Semantic;
    let mut enable_thinking = false;
    let mut show_prompt = false;

    let mut i = 0usize;
    let need = |args: &[String], i: &mut usize, flag: &str| -> Result<String> {
        *i += 1;
        args.get(*i)
            .cloned()
            .ok_or_else(|| anyhow!("{flag} needs a value"))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(need(args, &mut i, "--weights")?.into()),
            "--device" => device = need(args, &mut i, "--device")?,
            "--max-new-tokens" => {
                max_new_tokens = need(args, &mut i, "--max-new-tokens")?.parse()?;
            }
            "--max-seq" => max_seq = need(args, &mut i, "--max-seq")?.parse()?,
            "--task" => {
                task = match need(args, &mut i, "--task")?.as_str() {
                    "asr" => FireRedTask::Asr,
                    "understand" => FireRedTask::Understand,
                    "tts" => FireRedTask::Tts,
                    "edit" => FireRedTask::Edit,
                    "voice_design" => FireRedTask::VoiceDesign,
                    other => bail!("unknown --task {other}"),
                };
            }
            "--audio" => audio = Some(need(args, &mut i, "--audio")?.into()),
            "--prompt" => prompt = Some(need(args, &mut i, "--prompt")?),
            "--prompt-audio" => prompt_audio = Some(need(args, &mut i, "--prompt-audio")?.into()),
            "--prompt-text" => prompt_text = Some(need(args, &mut i, "--prompt-text")?),
            "--target-text" => target_text = Some(need(args, &mut i, "--target-text")?),
            "--language" => language = need(args, &mut i, "--language")?,
            "--edit-type" => {
                edit_type = match need(args, &mut i, "--edit-type")?.as_str() {
                    "semantic" => EditType::Semantic,
                    "acoustic" => EditType::Acoustic,
                    other => bail!("unknown --edit-type {other}"),
                };
            }
            "--enable-thinking" => enable_thinking = true,
            "--show-prompt" => show_prompt = true,
            other => bail!("unknown arg {other}"),
        }
        i += 1;
    }

    if show_prompt {
        let tok = "<|AUDIO|>";
        let no_lat = "<|AUDIO_NO_LATENT|>";
        let s = match task {
            FireRedTask::Asr => crate::prompt::build_asr_prompt(1, tok)?,
            FireRedTask::Understand => crate::prompt::build_understand_prompt(
                prompt.as_deref().unwrap_or("Describe the audio."),
                1,
                tok,
                enable_thinking,
            )?,
            FireRedTask::Tts => crate::prompt::build_tts_prompt(
                prompt_text.as_deref().unwrap_or("ref"),
                target_text.as_deref().unwrap_or("hello"),
                &language,
                no_lat,
            )?,
            FireRedTask::Edit => crate::prompt::build_edit_prompt(
                prompt.as_deref().unwrap_or("shift the pitch by 3 steps"),
                edit_type,
                no_lat,
            )?,
            FireRedTask::VoiceDesign => crate::prompt::build_voice_design_prompt(
                prompt.as_deref().unwrap_or("bright female voice"),
                target_text.as_deref().unwrap_or("Hello."),
            )?,
        };
        print!("{s}");
        return Ok(());
    }

    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let device = rlx_cli::parse_qwen35_device(&device)?;

    let runner = FireRedRunner::builder()
        .weights(weights)
        .device(device)
        .max_new_tokens(max_new_tokens)
        .max_seq(max_seq)
        .build()?;

    match task {
        FireRedTask::Asr => {
            let audio = audio.ok_or_else(|| anyhow!("--audio is required for asr"))?;
            let out = runner.asr_wav(&audio)?;
            println!("{}", out.answer);
        }
        FireRedTask::Understand => {
            let audio = audio.ok_or_else(|| anyhow!("--audio is required for understand"))?;
            let q = prompt.as_deref().unwrap_or(DEFAULT_ASR_PROMPT).to_string();
            let out = runner.understand_wav(&audio, &q, enable_thinking)?;
            if let Some(r) = out.reasoning {
                println!("CoT:\n{r}\n");
            }
            println!("{}", out.answer);
        }
        FireRedTask::Tts => {
            let pa = prompt_audio.ok_or_else(|| anyhow!("--prompt-audio required"))?;
            let pt = prompt_text.ok_or_else(|| anyhow!("--prompt-text required"))?;
            let tt = target_text.ok_or_else(|| anyhow!("--target-text required"))?;
            let _ = runner.tts(&pt, &pa, &tt, &language)?;
        }
        FireRedTask::Edit | FireRedTask::VoiceDesign => {
            let _ = (edit_type, Device::Cpu);
            bail!(
                "task={} needs RedAE+DiT generation graphs (not compiled yet)",
                task.as_str()
            );
        }
    }
    Ok(())
}
