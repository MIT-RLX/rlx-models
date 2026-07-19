// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Soprano 1.1 — ultra-light ~80M Qwen3 AR TTS @ 32 kHz (Apache-2.0).
//!
//! Native (ort-free) path imports
//! [`KevinAHM/soprano-1.1-onnx`](https://huggingface.co/KevinAHM/soprano-1.1-onnx)
//! into RLX (`rlx-onnx-import`): KV-cache backbone + vocoder decoder.
//!
//! ```bash
//! just fetch-soprano
//! just soprano-demo
//! ```

pub mod native;

pub use native::{
    DEFAULT_LOCAL_DIR, HIDDEN, InferOpts, NativeSoprano, SAMPLE_RATE, format_prompt, peak_amplitude,
};
pub use rlx_runtime::{Device, parse_device};
