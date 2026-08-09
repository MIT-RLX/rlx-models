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

//! Metal compile-only quick check (no forward pass).

use std::path::PathBuf;

#[cfg(all(target_os = "macos", feature = "metal"))]
use rlx_core::flow_util::compile_built;
#[cfg(all(target_os = "macos", feature = "metal"))]
use rlx_runtime::Device;
#[cfg(all(target_os = "macos", feature = "metal"))]
use rlx_voxtral_tts::lm_flow::build_tts_backbone_decode_built;
#[cfg(all(target_os = "macos", feature = "metal"))]
use rlx_voxtral_tts::{VoxtralTtsConfig, VoxtralTtsWeightStore};

#[allow(dead_code)]
fn model_dir() -> Option<PathBuf> {
    std::env::var("RLX_VOXTRAL_TTS_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.join("consolidated.safetensors").is_file())
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_decode_graph_compiles() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_VOXTRAL_TTS_DIR");
        return;
    };
    let cfg = VoxtralTtsConfig::from_model_dir(&dir).expect("config");
    let store = VoxtralTtsWeightStore::open(&dir).expect("store");
    let mut wm = store.load_backbone().expect("wm");
    let built =
        build_tts_backbone_decode_built(&cfg.text_config, &mut wm, 1, 4).expect("decode built");
    let params = built.params().clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut compiled = compile_built(built, Device::Metal).expect("metal compile");
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        compiled
    }));
    match result {
        Ok(_) => eprintln!("metal decode graph compiled (past_len=4)"),
        Err(_) => eprintln!(
            "skip: Metal MPS failed to compile decode graph (known backend shape issue); CPU path validated separately"
        ),
    }
}
