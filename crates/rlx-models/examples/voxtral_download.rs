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

//! Download [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507).
//!
//! ```bash
//! just fetch-voxtral
//! export RLX_VOXTRAL_DIR=.cache/voxtral/Voxtral-Mini-3B-2507
//! ```

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let dest = std::env::var("RLX_VOXTRAL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/voxtral/Voxtral-Mini-3B-2507"));

    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model("mistralai/Voxtral-Mini-3B-2507".to_string());
    std::fs::create_dir_all(&dest)?;
    for name in [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tekken.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ] {
        let path = repo.get(name)?;
        let out = dest.join(name);
        if out.exists() {
            continue;
        }
        std::fs::copy(&path, &out)?;
        println!("fetched {name}");
    }

    println!("Voxtral ready under:\n  {}", dest.display());
    println!("\nexport RLX_VOXTRAL_DIR={}", dest.display());
    println!(
        "export RLX_VOXTRAL_WEIGHTS={}",
        dest.join("model-00001-of-00002.safetensors").display()
    );
    Ok(())
}
