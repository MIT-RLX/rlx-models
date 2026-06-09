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

//! Download [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice).
//!
//! ```bash
//! just fetch-qwen3-tts
//! export RLX_QWEN3_TTS_DIR=.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice
//! ```

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let dest = std::env::var("RLX_QWEN3_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice"));

    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(rlx_qwen3_tts::HF_MODEL_ID_06B_CUSTOM.to_string());
    std::fs::create_dir_all(&dest)?;

    for name in [
        "config.json",
        "model.safetensors",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
    ] {
        let path = repo.get(name)?;
        let out = dest.join(name);
        if out.exists() {
            continue;
        }
        std::fs::copy(&path, &out)?;
        println!("fetched {name}");
    }

    for name in [
        "speech_tokenizer/config.json",
        "speech_tokenizer/configuration.json",
        "speech_tokenizer/model.safetensors",
        "speech_tokenizer/preprocessor_config.json",
    ] {
        let path = repo.get(name)?;
        let out = dest.join(name);
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if out.exists() {
            continue;
        }
        std::fs::copy(&path, &out)?;
        println!("fetched {name}");
    }

    if !dest.join("tokenizer.json").is_file() {
        eprintln!(
            "warning: tokenizer.json missing — re-run fetch or copy from the HuggingFace repo"
        );
    }

    println!("Qwen3-TTS ready under:\n  {}", dest.display());
    println!("\nexport RLX_QWEN3_TTS_DIR={}", dest.display());
    Ok(())
}
