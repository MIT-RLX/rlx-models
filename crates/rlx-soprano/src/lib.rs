// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Soprano 1.1 — ultra-light ~80M Qwen3 AR TTS @ 32 kHz (Apache-2.0).
//!
//! Prefer packed [`soprano.rlxp`](https://huggingface.co/eugenehp/soprano) with
//! nested `graphs/*.rlxp`. Legacy `soprano.gguf` still loads. Runtime materializes
//! nested packs + tokenizer and lowers via `rlx-onnx-import` (no ORT; Hub has no
//! `.onnx`). Pack-time source:
//! [`KevinAHM/soprano-1.1-onnx`](https://huggingface.co/KevinAHM/soprano-1.1-onnx).
//!
//! ```bash
//! just fetch-soprano
//! just soprano-demo
//! ```

pub mod gguf_bundle;
pub mod native;
pub mod native_qwen3;

pub use gguf_bundle::{
    DEFAULT_GGUF_NAME, DEFAULT_RLXP_NAME, FORMAT as GGUF_FORMAT, HF_REPO, PackReport, open_gguf,
    open_rlxp, pack_directory, pack_rlxp, resolve_gguf_path, resolve_rlxp_path,
};
pub use native::{
    DEFAULT_LOCAL_DIR, HIDDEN, InferOpts, NativeSoprano, SAMPLE_RATE, format_prompt, peak_amplitude,
};
pub use rlx_runtime::{Device, parse_device};
