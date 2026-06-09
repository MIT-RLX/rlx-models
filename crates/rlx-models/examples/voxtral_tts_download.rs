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

//! Download [mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603).
//!
//! ```bash
//! just fetch-voxtral-tts
//! export RLX_VOXTRAL_TTS_DIR=.cache/voxtral/Voxtral-4B-TTS-2603
//! ```

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let dest = std::env::var("RLX_VOXTRAL_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/voxtral/Voxtral-4B-TTS-2603"));

    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model("mistralai/Voxtral-4B-TTS-2603".to_string());
    std::fs::create_dir_all(&dest)?;

    for name in [
        "params.json",
        "tekken.json",
        "consolidated.safetensors",
        "README.md",
    ] {
        let path = repo.get(name)?;
        let out = dest.join(name);
        if out.exists() {
            continue;
        }
        std::fs::copy(&path, &out)?;
        println!("fetched {name}");
    }

    let voice_dir = dest.join("voice_embedding");
    std::fs::create_dir_all(&voice_dir)?;
    for voice in rlx_voxtral_tts::PRESET_VOICES {
        let name = format!("{voice}.pt");
        let path = repo.get(&format!("voice_embedding/{name}"))?;
        let out = voice_dir.join(&name);
        if out.exists() {
            continue;
        }
        std::fs::copy(&path, &out)?;
        println!("fetched voice_embedding/{name}");
    }

    println!("Voxtral TTS ready under:\n  {}", dest.display());
    println!("\nexport RLX_VOXTRAL_TTS_DIR={}", dest.display());
    Ok(())
}
