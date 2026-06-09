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

//! Download [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) for JFK fine-tune.
//!
//! ```bash
//! just fetch-qwen3-tts-base
//! export RLX_QWEN3_TTS_BASE_DIR=.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base
//! ```

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let dest = std::env::var("RLX_QWEN3_TTS_BASE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base"));

    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(rlx_qwen3_tts::HF_MODEL_ID_06B_BASE.to_string());
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
        fetch_one(&repo, &dest, name);
    }

    for name in [
        "speech_tokenizer/config.json",
        "speech_tokenizer/configuration.json",
        "speech_tokenizer/model.safetensors",
        "speech_tokenizer/preprocessor_config.json",
    ] {
        fetch_one(&repo, &dest, name);
    }

    // Base repo may omit standalone tokenizer files; reuse CustomVoice bundle if present.
    let custom = PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice");
    for name in [
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
    ] {
        let out = dest.join(name);
        if !out.is_file() {
            let src = custom.join(name);
            if src.is_file() {
                std::fs::copy(&src, &out)?;
                println!("linked {name} from CustomVoice cache");
            }
        }
    }

    println!("Qwen3-TTS Base ready under:\n  {}", dest.display());
    println!("\nexport RLX_QWEN3_TTS_BASE_DIR={}", dest.display());
    Ok(())
}

fn fetch_one(repo: &hf_hub::api::sync::ApiRepo, dest: &std::path::Path, name: &str) {
    let out = dest.join(name);
    if out.exists() {
        return;
    }
    match repo.get(name) {
        Ok(path) => {
            if let Some(parent) = out.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if std::fs::copy(&path, &out).is_ok() {
                println!("fetched {name}");
            }
        }
        Err(e) => eprintln!("skip {name}: {e}"),
    }
}
