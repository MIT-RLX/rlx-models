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

//! KittenTTS — lightweight ONNX text-to-speech for RLX.
//!
//! ## Backends
//!
//! | Feature | RLX runtime | ONNX Runtime EP        |
//! |---------|-------------|------------------------|
//! | `onnx`  | —           | CPU (default)          |
//! | `native`| RLX runtime | Decomposed `kitten_tts_mini_rlx` graph (no ORT) |
//! | `rlx`   | RLX runtime | ORT inference + RLX path deps for future parity |
//! | `metal` | Metal       | CoreML (macOS / iOS)   |
//! | `mlx`   | MLX         | CoreML (Apple GPU)     |
//! | `cuda`  | CUDA        | CUDA                   |
//! | `rocm`  | ROCm        | ROCm                   |
//! | `gpu`   | wgpu        | DirectML / CUDA / CoreML |
//! | `full`  | all above   | all ORT EPs + RLX path deps |

pub mod assets;
#[cfg(feature = "onnx")]
pub mod backend;
pub mod backend_kind;
pub mod cli;
pub mod config;
pub mod download;
pub mod features;
pub mod infer_opts;
pub mod model;
#[cfg(feature = "native")]
pub mod native;
pub mod npz;
pub mod phonemize;
#[cfg(feature = "espeak")]
pub mod preprocess;
pub mod tokenize;

pub use assets::{
    DEFAULT_LOCAL_DIR, ModelLayout, default_model_dir, default_native_weights_dir,
    find_native_weights, find_rlx_bundle,
};
#[cfg(feature = "onnx")]
pub use backend::{OrtSession, build_onnx_session, execution_providers_for, validate_device};
pub use config::{DEFAULT_HF_REPO, ModelConfig};
pub use download::{fetch_default, fetch_to_local_dir};
pub use features::{
    cuda_feature_enabled, enabled_backend_labels, espeak_feature_enabled, gpu_feature_enabled,
    metal_feature_enabled, mlx_feature_enabled, native_feature_enabled, onnx_feature_enabled,
    rlx_feature_enabled, rocm_feature_enabled,
};
pub use infer_opts::{SAMPLES_PER_DURATION_UNIT, recommended_native_compile_opts};
pub use model::{KittenTTS, MIN_AUDIBLE_PEAK, SAMPLE_RATE, peak_amplitude};
pub use npz::{NpyArray, load_npz, parse_npy};
pub use phonemize::{DEFAULT_LANG, is_espeak_available, phonemize, phonemize_lang, set_data_path};
pub use rlx_runtime::{Device, fastest_device, is_available, parse_device, parse_device_list};
pub use tokenize::{
    ipa_content_len, ipa_style_index, ipa_text_style_index, ipa_to_ids, warn_unknown_ipa_chars,
};
