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

//! Compiled Ministral LM validation on real Voxtral-4B-TTS weights.
//!
//! ```bash
//! export RLX_VOXTRAL_TTS_DIR=/path/to/Voxtral-4B-TTS-2603
//! just test-voxtral-tts-compiled-lm
//! ```

mod common;

use common::{PREFILL_SEQ, compiled_trace, eager_trace, l2, load_backbone, model_dir};
use rlx_runtime::Device;
use rlx_voxtral_tts::lm_flow::{build_tts_backbone_decode_built, build_tts_backbone_prefill_built};
use std::time::Instant;

#[test]
fn lm_flow_graphs_build_on_real_weights() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_VOXTRAL_TTS_DIR with consolidated.safetensors");
        return;
    };
    let (cfg, store, _) = load_backbone(&dir);
    let mut wm = store.load_backbone().expect("backbone wm");
    build_tts_backbone_prefill_built(&cfg.text_config, &mut wm, 1, PREFILL_SEQ, true)
        .expect("prefill built");
    let mut wm2 = store.load_backbone().expect("backbone wm decode");
    build_tts_backbone_decode_built(&cfg.text_config, &mut wm2, 1, PREFILL_SEQ)
        .expect("decode built");
    eprintln!("prefill/decode HIR built for seq={PREFILL_SEQ}");
}

#[test]
fn eager_lm_prefill_and_kv_decode_smoke() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_VOXTRAL_TTS_DIR with consolidated.safetensors");
        return;
    };
    let (cfg, _, tensors) = load_backbone(&dir);
    let trace = eager_trace(&tensors, &cfg.text_config, cfg.text_config.hidden_size);
    assert_eq!(trace.len(), 1 + common::DECODE_STEPS);
    for (i, h) in trace.iter().enumerate() {
        assert!(h.iter().all(|v| v.is_finite()), "eager step {i} non-finite");
    }
    eprintln!(
        "eager LM: {} steps, last norm {:.4}",
        trace.len(),
        l2(trace.last().unwrap())
    );
}

#[allow(dead_code)]
fn gpu_compiled_forward_smoke(device: Device, label: &str) {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_VOXTRAL_TTS_DIR with consolidated.safetensors");
        return;
    };
    if !rlx_runtime::is_available(device) {
        eprintln!("skip: {label} ({device:?}) not available");
        return;
    }
    let (cfg, store, _) = load_backbone(&dir);
    let t0 = Instant::now();
    let trace = compiled_trace(
        &store,
        &cfg.text_config,
        device,
        cfg.text_config.hidden_size,
    );
    let elapsed = t0.elapsed();
    assert_eq!(trace.len(), 1 + common::DECODE_STEPS);
    for (i, h) in trace.iter().enumerate() {
        assert!(
            h.iter().all(|v| v.is_finite()),
            "{label} step {i} non-finite"
        );
    }
    eprintln!(
        "{label} compiled LM: {} steps, last norm {:.4}, {:.1}s",
        trace.len(),
        l2(trace.last().unwrap()),
        elapsed.as_secs_f64()
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_compiled_lm_matches_eager() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_VOXTRAL_TTS_DIR with consolidated.safetensors");
        return;
    };
    if std::env::var("RLX_VOXTRAL_TTS_METAL_LM").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_VOXTRAL_TTS_METAL_LM=1 for full Metal forward parity");
        return;
    }
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal not available");
        return;
    }
    let (cfg, store, tensors) = load_backbone(&dir);
    let hidden = cfg.text_config.hidden_size;
    let eager = eager_trace(&tensors, &cfg.text_config, hidden);
    let metal = compiled_trace(&store, &cfg.text_config, Device::Metal, hidden);
    assert_eq!(eager.len(), metal.len());
    for (i, (e, m)) in eager.iter().zip(metal.iter()).enumerate() {
        let cos = cosine(e.as_slice().unwrap(), m.as_slice().unwrap());
        let mad = max_abs_diff(e.as_slice().unwrap(), m.as_slice().unwrap());
        eprintln!("metal vs eager step={i} cos={cos:.6} max_abs={mad:.6}");
        assert!(cos >= 0.999, "step {i} cosine {cos}");
        assert!(mad <= 0.05, "step {i} max_abs {mad}");
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_compiled_lm_forward_smoke() {
    gpu_compiled_forward_smoke(Device::Metal, "metal");
}

#[test]
#[cfg(feature = "gpu")]
fn wgpu_compiled_lm_forward_smoke() {
    if std::env::var("RLX_VOXTRAL_TTS_WGPU_LM").ok().as_deref() != Some("1") {
        eprintln!(
            "skip: set RLX_VOXTRAL_TTS_WGPU_LM=1 for full wgpu forward (layer-sharded; slow first run)"
        );
        return;
    }
    gpu_compiled_forward_smoke(Device::Gpu, "wgpu");
}

#[test]
#[cfg(feature = "gpu")]
fn wgpu_compiled_lm_matches_eager() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_VOXTRAL_TTS_DIR with consolidated.safetensors");
        return;
    };
    if std::env::var("RLX_VOXTRAL_TTS_WGPU_LM").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_VOXTRAL_TTS_WGPU_LM=1 for full wgpu forward parity");
        return;
    };
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("skip: wgpu (Device::Gpu) not available");
        return;
    }
    let (cfg, store, tensors) = load_backbone(&dir);
    let hidden = cfg.text_config.hidden_size;
    let eager = eager_trace(&tensors, &cfg.text_config, hidden);
    let gpu = compiled_trace(&store, &cfg.text_config, Device::Gpu, hidden);
    assert_eq!(eager.len(), gpu.len());
    for (i, (e, g)) in eager.iter().zip(gpu.iter()).enumerate() {
        let cos = cosine(e.as_slice().unwrap(), g.as_slice().unwrap());
        let mad = max_abs_diff(e.as_slice().unwrap(), g.as_slice().unwrap());
        eprintln!("wgpu vs eager step={i} cos={cos:.6} max_abs={mad:.6}");
        assert!(cos >= 0.99, "step {i} cosine {cos}");
        assert!(mad <= 0.1, "step {i} max_abs {mad}");
    }
}
