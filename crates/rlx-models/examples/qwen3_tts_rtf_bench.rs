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

// End-to-end RTF: synthesis seconds / audio seconds (target ≤ 1.0).
//
// 12 codec frames ≈ 1s audio @ 12Hz tokenizer, 24kHz PCM.
//
// ```bash
// export RLX_QWEN3_TTS_DIR=.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice
// RLX_QWEN3_TTS_DEVICE=metal RLX_QWEN3_TTS_TIMING=1 \
//   cargo run -p rlx-models --example qwen3_tts_rtf_bench --release --features apple-silicon
// RLX_QWEN3_TTS_DEVICE=mlx  RLX_QWEN3_TTS_TIMING=1 ...
// RLX_QWEN3_TTS_METAL_MPSGRAPH=1 ...   # Metal + MPSGraph (experimental)
// ```

use rlx_cli::parse_device;
use rlx_models::qwen3_tts::{GenerationConfig, Qwen3TtsSession};
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;

fn pick_device() -> Device {
    if let Ok(s) = std::env::var("RLX_QWEN3_TTS_DEVICE") {
        if s.eq_ignore_ascii_case("auto") || s.eq_ignore_ascii_case("fastest") {
            return rlx_qwen3_tts::synth_opts::fastest_device();
        }
        return parse_device(&s).unwrap_or_else(|e| {
            eprintln!("invalid RLX_QWEN3_TTS_DEVICE={s:?}: {e}; falling back to fastest");
            rlx_qwen3_tts::synth_opts::fastest_device()
        });
    }
    if is_available(Device::Metal) {
        Device::Metal
    } else if is_available(Device::Mlx) {
        Device::Mlx
    } else {
        Device::Cpu
    }
}

fn main() -> anyhow::Result<()> {
    let device = pick_device();
    if !is_available(device) {
        anyhow::bail!("device {device:?} not available on this host");
    }
    let mpsgraph_req = std::env::var("RLX_QWEN3_TTS_METAL_MPSGRAPH")
        .ok()
        .as_deref()
        == Some("1");
    let mpsgraph = rlx_qwen3_tts::compile_opts::metal_mpsgraph_enabled();
    eprintln!(
        "[rtf-bench] device={device:?} metal_mpsgraph={mpsgraph} (req={mpsgraph_req}) talker_eager_default_metal={} metal_compiled={} mlx_compiled={}",
        device == Device::Metal,
        rlx_qwen3_tts::compile_opts::talker_metal_native_compile(device),
        std::env::var("RLX_QWEN3_TTS_MLX_COMPILED").ok().as_deref() == Some("1"),
    );

    let model_dir = std::env::var("RLX_QWEN3_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice"));
    let mut session = Qwen3TtsSession::open(&model_dir, device)?;
    let mut gen_cfg = GenerationConfig::greedy_for_model_dir(&model_dir)?;
    gen_cfg.max_new_tokens = 12;

    eprintln!("=== utterance 1 (warm session) ===");
    let _ =
        session.synthesize_custom_voice("Hello world.", "vivian", "english", &gen_cfg, false)?;

    eprintln!("=== utterance 2 (steady RTF) ===");
    let _ = session.synthesize_custom_voice(
        "One second test.",
        "vivian",
        "english",
        &gen_cfg,
        false,
    )?;

    Ok(())
}
