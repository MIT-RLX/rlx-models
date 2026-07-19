// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Parler-TTS Mini unit tests (no heavy ONNX compile unless weights + env are set).

use rlx_parlertts::{DEFAULT_DAC_DIR, DEFAULT_LOCAL_DIR, InferOpts, ParlerTTSConfig};

#[test]
fn config_defaults_point_at_centralized_tts_weights() {
    let cfg = ParlerTTSConfig::default();
    assert_eq!(cfg.model_dir, DEFAULT_LOCAL_DIR);
    assert_eq!(cfg.dac_dir, DEFAULT_DAC_DIR);
    assert_eq!(cfg.language, "en");
}

#[test]
fn infer_opts_defaults_are_sane() {
    let o = InferOpts::default();
    assert!(o.max_steps >= 64);
    assert!(o.top_k >= 1);
    assert!(o.temperature > 0.0);
}
