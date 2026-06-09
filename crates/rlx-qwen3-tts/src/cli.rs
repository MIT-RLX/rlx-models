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

use crate::runner::Qwen3TtsRunner;
use anyhow::{Context, Result};
use rlx_cli::parse_device;
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;

fn log_backend_plan(device: Device) {
    eprintln!(
        "[qwen3-tts] backends: mlx={} metal={} cuda={} rocm={} | selected={device:?}",
        is_available(Device::Mlx),
        is_available(Device::Metal),
        is_available(Device::Cuda),
        is_available(Device::Rocm),
    );
    let cp_eager = !crate::code_predictor::engine::cp_use_compiled_for_device(device);
    let cp_label = if cp_eager {
        "CPU eager".to_string()
    } else {
        format!("compiled ({device:?})")
    };
    let speech_pt = super::speech_tokenizer::speech_pt_backend_label(device) == "compiled";
    let speech_conv = super::speech_tokenizer::speech_conv_backend_label(device);
    let talker = if crate::talker::engine::talker_use_eager_for_device(device) {
        if device == Device::Mlx {
            "CPU eager (MLX — set RLX_QWEN3_TTS_MLX_COMPILED=1 or use --device metal)"
        } else {
            "CPU eager"
        }
    } else if device == Device::Cpu {
        "compiled (CPU)"
    } else {
        "compiled"
    };
    let fusion = if matches!(device, Device::Mlx | Device::Gpu | Device::Vulkan)
        || std::env::var("RLX_QWEN3_TTS_FUSION_SKIP").ok().as_deref() == Some("1")
    {
        "off"
    } else {
        "on (tier-1 Fusable decode)"
    };
    eprintln!(
        "[qwen3-tts] pipeline: talker={talker} ({device:?}), code_predictor={cp_label}, speech_pt={}, speech_conv={speech_conv}, fusion={fusion}",
        if speech_pt {
            format!("compiled ({device:?})")
        } else {
            "CPU eager".into()
        },
    );
    if crate::gpu_pipeline::gpu_session_enabled(device) {
        eprintln!(
            "[qwen3-tts] GPU pipeline (talker+speech on GPU; CP {} on {:?})",
            if cp_eager { "CPU eager" } else { "compiled" },
            device,
        );
        eprintln!(
            "[qwen3-tts] opt out: RLX_QWEN3_TTS_CPU_PIPELINE=1 | (Metal CP via RLX_QWEN3_TTS_CP_METAL=1 currently slower than CPU eager)"
        );
    }
    if crate::synth_opts::megakernel_fast_path() {
        let gpu_kv = match std::env::var("RLX_QWEN3_TTS_GPU_KV").ok().as_deref() {
            Some("0") => "off",
            Some("1") => "on (forced)",
            _ if crate::synth_opts::megakernel_gpu_kv_default(device) => "on (megakernel default)",
            _ => "off",
        };
        eprintln!(
            "[qwen3-tts] megakernel fast path: GPU KV {gpu_kv} (RLX_QWEN3_TTS_GPU_KV=0 to disable)"
        );
    }
    if crate::synth_opts::lazy_talk_buckets() {
        eprintln!(
            "[qwen3-tts] lazy talker buckets (RLX_QWEN3_TTS_PRECOMPILE_BUCKETS=1 for full horizon precompile)"
        );
    }
    if device == Device::Cpu {
        eprintln!(
            "[qwen3-tts] hint: rebuild with `just qwen3-tts` (all-backends) or `--features mlx,metal,cuda` for GPU"
        );
    }
}

fn parse_tts_device(s: &str) -> Result<Device> {
    if s.eq_ignore_ascii_case("auto") || s.eq_ignore_ascii_case("fastest") {
        return Ok(crate::synth_opts::fastest_device());
    }
    parse_device(s)
}

pub fn run(args: &[String]) -> Result<()> {
    let mut model_dir = std::env::var("RLX_QWEN3_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice"));
    let mut device = std::env::var("RLX_QWEN3_TTS_DEVICE")
        .ok()
        .map(|s| parse_tts_device(&s))
        .transpose()?
        .unwrap_or_else(crate::synth_opts::fastest_device);
    let mut eager_talker = false;
    let mut bench_talker = false;
    let mut prefill_seq = 128usize;
    let mut decode_steps = 64usize;
    let mut text = std::env::var("RLX_QWEN3_TTS_TEXT")
        .unwrap_or_else(|_| String::from("Hello from RLX Qwen3-TTS."));
    let mut speaker = String::from("vivian");
    let mut language = String::from("English");
    let mut out_wav = PathBuf::from("qwen3_tts_out.wav");
    let mut max_frames = 0usize;

    let mut it = args.iter().filter(|a| *a != "--");
    while let Some(arg) = it.next() {
        if let Some((flag, value)) = arg.split_once('=') {
            apply_flag(
                flag,
                value,
                &mut model_dir,
                &mut device,
                &mut eager_talker,
                &mut bench_talker,
                &mut prefill_seq,
                &mut decode_steps,
                &mut text,
                &mut speaker,
                &mut language,
                &mut out_wav,
                &mut max_frames,
            )?;
            continue;
        }
        match arg.as_str() {
            "--model-dir" => model_dir = PathBuf::from(it.next().context("--model-dir")?),
            "--device" => device = parse_tts_device(it.next().context("--device")?)?,
            "--max-frames" => max_frames = it.next().context("--max-frames")?.parse()?,
            "--eager-talker" => eager_talker = true,
            "--bench-talker" => bench_talker = true,
            "--prefill-seq" => prefill_seq = it.next().context("--prefill-seq")?.parse()?,
            "--decode-steps" => decode_steps = it.next().context("--decode-steps")?.parse()?,
            "--text" => text = it.next().context("--text")?.clone(),
            "--speaker" => speaker = it.next().context("--speaker")?.clone(),
            "--language" => language = it.next().context("--language")?.clone(),
            "--out-wav" => out_wav = PathBuf::from(it.next().context("--out-wav")?),
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => anyhow::bail!("unknown flag: {other}"),
        }
    }

    log_backend_plan(device);
    let budget = crate::synth_opts::codec_frame_budget(&text, 128, max_frames);
    if budget > 64 && !crate::synth_opts::warmup_all_talk_buckets() {
        eprintln!(
            "[qwen3-tts] note: codec budget {budget} — skipping full Metal bucket dry-run warmup \
             (lazy compile during synthesis; stderr progress). \
             Set RLX_QWEN3_TTS_WARMUP_ALL_BUCKETS=1 to pre-warm all buckets (slow)."
        );
    }
    if max_frames > 0 {
        let auto = crate::synth_opts::codec_frame_budget(&text, 128, 0);
        if max_frames < auto {
            eprintln!(
                "[qwen3-tts] warning: --max-frames {max_frames} is below auto budget {auto} — \
                 audio may truncate before talker EOS (omit flag for auto)"
            );
        }
    }
    let runner = Qwen3TtsRunner::builder()
        .model_dir(&model_dir)
        .device(device)
        .eager_talker(eager_talker)
        .max_frames(max_frames)
        .build()?;

    if bench_talker {
        let report = runner.bench_talker_synthetic(prefill_seq, decode_steps)?;
        report.print_line();
        return Ok(());
    }

    if model_dir.to_string_lossy().contains("jfk-checkpoint") {
        eprintln!(
            "[qwen3-tts] note: finetuned JFK checkpoints may sound wrong in native RLX \
             (codec parity vs HF is in progress). For reference audio use: \
             `just qwen3-tts-jfk-hf-demo`"
        );
    }

    runner.synthesize_custom_voice(&text, &speaker, &language, &out_wav)?;
    println!("wrote {}", out_wav.display());
    Ok(())
}

fn apply_flag(
    flag: &str,
    value: &str,
    model_dir: &mut PathBuf,
    device: &mut Device,
    _eager_talker: &mut bool,
    _bench_talker: &mut bool,
    prefill_seq: &mut usize,
    decode_steps: &mut usize,
    text: &mut String,
    speaker: &mut String,
    language: &mut String,
    out_wav: &mut PathBuf,
    max_frames: &mut usize,
) -> Result<()> {
    match flag {
        "--model-dir" => *model_dir = PathBuf::from(value),
        "--device" => *device = parse_tts_device(value)?,
        "--max-frames" => *max_frames = value.parse()?,
        "--prefill-seq" => *prefill_seq = value.parse()?,
        "--decode-steps" => *decode_steps = value.parse()?,
        "--text" => *text = value.to_string(),
        "--speaker" => *speaker = value.to_string(),
        "--language" => *language = value.to_string(),
        "--out-wav" => *out_wav = PathBuf::from(value),
        other => anyhow::bail!("unknown flag: {other}"),
    }
    Ok(())
}

fn print_help() {
    eprintln!(
        "rlx-qwen3-tts — Qwen3-TTS on RLX\n\
         --model-dir DIR   HF checkpoint (default: RLX_QWEN3_TTS_DIR)\n\
         --device NAME     auto (default) | mlx | metal | cpu | cuda | …\n\
         --max-frames N    Optional hard cap (0 = auto from text length, stop at talker EOS)\n\
         --bench-talker    Synthetic talker prefill+decode benchmark\n\
         --prefill-seq N   Bench prefill length (default 128)\n\
         --decode-steps N  Bench decode steps (default 64)\n\
         --eager-talker    CPU eager talker (not implemented)\n\
         --text STR  (or --text=… for multi-word strings via just)\n\
         --speaker NAME --language LANG --out-wav PATH  Native greedy synthesis"
    );
}
