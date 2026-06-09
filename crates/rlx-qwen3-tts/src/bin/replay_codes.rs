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

//! Decode externally-generated codec frames (e.g. Python ground truth)
//! through the Rust speech decoder. Isolates the speech-decoder path from
//! talker/code-predictor.

use anyhow::{Context, Result};
use rlx_qwen3_tts::speech_tokenizer::decode_codec_frames;
use rlx_runtime::Device;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_dir = PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base");
    let mut codes_json = PathBuf::from("/tmp/py_codes.json");
    let mut out_wav = PathBuf::from("/tmp/replay.wav");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                model_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--codes" => {
                codes_json = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--out-wav" => {
                out_wav = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            o => anyhow::bail!("unknown arg {o:?}"),
        }
    }
    let raw = std::fs::read_to_string(&codes_json).context("read codes")?;
    let frames_i64: Vec<Vec<i64>> = serde_json::from_str(&raw)?;
    let frames: Vec<Vec<u32>> = frames_i64
        .into_iter()
        .map(|row| row.into_iter().map(|v| v as u32).collect())
        .collect();
    eprintln!(
        "decoding {} frames × {} groups",
        frames.len(),
        frames[0].len()
    );
    let pcm = decode_codec_frames(&model_dir, &frames, Device::Cpu)?;
    eprintln!(
        "pcm: {} samples ({:.2}s)",
        pcm.len(),
        pcm.len() as f64 / 24_000.0
    );
    let peak = pcm.iter().map(|s| s.abs()).fold(0f32, f32::max);
    eprintln!("peak={peak:.3}");
    rlx_qwen3_tts::runner::write_wav_mono(&out_wav, &pcm, 24_000)?;
    println!("wrote {}", out_wav.display());
    Ok(())
}
