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

// Two back-to-back utterances: second run skips megakernel warmup.
//
// ```bash
// export RLX_QWEN3_TTS_DIR=.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice
// RLX_QWEN3_TTS_TIMING=1 cargo run -p rlx-models --example qwen3_tts_session_bench --release --features metal
// ```

use rlx_models::qwen3_tts::{GenerationConfig, Qwen3TtsSession};
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let device = if is_available(Device::Metal) {
        Device::Metal
    } else {
        Device::Cpu
    };
    let model_dir = std::env::var("RLX_QWEN3_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice"));
    let mut session = Qwen3TtsSession::open(&model_dir, device)?;
    let mut gen_cfg = GenerationConfig::greedy_for_model_dir(&model_dir)?;
    gen_cfg.max_new_tokens = 22;

    eprintln!("=== utterance 1 (warm with real prompt) ===");
    let _ = session.synthesize_custom_voice("Hi.", "vivian", "english", &gen_cfg, true)?;

    eprintln!("=== utterance 2 (reuse session, new prompt) ===");
    let _ = session.synthesize_custom_voice("Hello.", "vivian", "english", &gen_cfg, true)?;

    Ok(())
}
