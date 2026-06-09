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

// Eager vs rlx fused bucketed decode (talker + code predictor).
//
// Fused path: `build_qwen3_*_embeds_built` → `CompileProfile::qwen3_decode()` (Fusable)
// → `BucketedCompileCache` + `set_active_extent` in rlx-runtime.
//
// ```bash
// export RLX_QWEN3_TTS_DIR=.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice
// cargo run -p rlx-models --example qwen3_tts_fusion_ab --release --features metal -- --frames 12
// ```

use anyhow::Context;
use rlx_cli::parse_device;
use rlx_qwen3_tts::fusion_bench::bench_fusion_ab;
use rlx_qwen3_tts::{Qwen3TtsConfig, load::Qwen3TtsWeightStore};
use rlx_runtime::{Device, is_available};
use std::env;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut frames = 12usize;
    let mut warmup = 2usize;
    let mut device = if is_available(Device::Metal) {
        Device::Metal
    } else {
        Device::Cpu
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--frames" => frames = it.next().context("--frames")?.parse()?,
            "--warmup" => warmup = it.next().context("--warmup")?.parse()?,
            "--device" => device = parse_device(it.next().context("--device")?)?,
            "--help" | "-h" => {
                eprintln!("qwen3_tts_fusion_ab [--frames N] [--warmup N] [--device metal|cpu|...]");
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown flag: {other}"),
        }
    }

    let model_dir = env::var("RLX_QWEN3_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice"));
    anyhow::ensure!(model_dir.is_dir(), "model dir: {}", model_dir.display());

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir)?;
    let store = Qwen3TtsWeightStore::open(&model_dir)?;

    eprintln!(
        "[qwen3-tts fusion-bench] session_device={device:?} (Metal talker compiled graphs run on CPU unless RLX_QWEN3_TTS_METAL_COMPILED=1)"
    );

    let summary = bench_fusion_ab(&store, &cfg, device, frames, 16, frames, warmup)?;
    summary.print_summary();

    Ok(())
}
