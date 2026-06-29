// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Native AIF probe prefill + decode-step on each RLX backend (skips when unavailable).

use rlx_qwen25_vl::{
    AifDynamicsMode, MEDIA_MARKER, MultimodalPrefill, MultimodalPrompt, Qwen25VlRunner,
    Qwen25VlRunnerBuilder, synth,
    vision::{MmProjWeights, Qwen25VlVisionEncoder},
};
use rlx_runtime::{Device, is_available};

fn fake_tokenizer(text: &str) -> anyhow::Result<Vec<u32>> {
    Ok(text.bytes().map(|b| (b as u32 % 31 + 1).max(1)).collect())
}

struct SynthFixture {
    runner: Qwen25VlRunner,
    n_vision: usize,
    n_layers: usize,
    prefill: MultimodalPrefill,
}

fn synth_fixture(device: Device, img: usize, prompt: &str, mode: AifDynamicsMode) -> SynthFixture {
    let mmcfg = synth::tiny_mmproj_cfg();
    let mmweights = MmProjWeights::synthetic(&mmcfg);
    let lmcfg = synth::tiny_lm_cfg();
    let lmweights = synth::synth_lm_weight_map(&lmcfg);

    let runner = Qwen25VlRunnerBuilder::default()
        .lm_config(lmcfg.clone())
        .inline_lm_weights(lmweights.clone())
        .inline_mmproj(mmcfg.clone(), mmweights.clone())
        .device(device)
        .max_seq(64)
        .aif_dynamics_mode(mode)
        .build()
        .expect("runner");

    let rgb: Vec<u8> = (0..(img * img * 3)).map(|i| (i % 251) as u8).collect();
    let mut enc = Qwen25VlVisionEncoder::from_parts(mmcfg, mmweights, img, img).expect("vision");
    let vision = enc.encode_rgb(&rgb, img, img).expect("encode");
    let mm = MultimodalPrompt {
        prompt,
        vision: &vision,
    };
    let embed = lmweights
        .get("model.embed_tokens.weight")
        .map(|(d, _)| d.as_slice())
        .expect("embed");
    let prefill = mm
        .assemble(fake_tokenizer, embed, lmcfg.lm.hidden_size, 0)
        .expect("assemble");

    SynthFixture {
        n_vision: vision.n_tokens,
        n_layers: lmcfg.lm.num_hidden_layers,
        runner,
        prefill,
    }
}

fn run_native_probe_on(device: Device) {
    let prompt = format!("q{MEDIA_MARKER}a");
    let mut fx = synth_fixture(device, 4, &prompt, AifDynamicsMode::PrefillV2t);
    fx.runner
        .prefill_from_assembled_probe(fx.prefill)
        .expect("probe prefill");
    let probe = fx.runner.probe_aif_native().expect("native probe");
    assert_eq!(probe.dynamics.len(), fx.n_vision);
    assert_eq!(probe.dynamics[0].len(), fx.n_layers);
    assert!(probe.mu.iter().all(|v| v.is_finite()));
}

fn run_decode_step_probe_on(device: Device) {
    let prompt = format!("x{MEDIA_MARKER}y");
    let mut fx = synth_fixture(device, 8, &prompt, AifDynamicsMode::DecodeStep);
    fx.runner
        .prefill_from_assembled(fx.prefill)
        .expect("prefill");
    let probe = fx.runner.probe_aif_native().expect("decode-step probe");
    assert_eq!(probe.dynamics.len(), fx.n_vision);
    assert_eq!(probe.dynamics[0].len(), fx.n_layers);
    assert!(probe.mu.iter().all(|v| v.is_finite()));
    assert!(
        probe.dynamics.iter().flatten().any(|&v| v > 0.0),
        "decode-step dynamics should be non-zero on {device:?}"
    );
}

#[allow(dead_code)]
fn run_if_available(device: Device, f: fn(Device)) {
    if !is_available(device) {
        eprintln!("skip qwen25-vl native probe {device:?}: backend unavailable");
        return;
    }
    f(device);
}

#[test]
fn native_probe_prefill_cpu() {
    run_native_probe_on(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn native_probe_prefill_metal() {
    run_if_available(Device::Metal, run_native_probe_on);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn native_probe_prefill_mlx() {
    run_if_available(Device::Mlx, run_native_probe_on);
}

#[cfg(feature = "cuda")]
#[test]
fn native_probe_prefill_cuda() {
    run_if_available(Device::Cuda, run_native_probe_on);
}

#[cfg(feature = "rocm")]
#[test]
fn native_probe_prefill_rocm() {
    run_if_available(Device::Rocm, run_native_probe_on);
}

#[cfg(feature = "gpu")]
#[test]
fn native_probe_prefill_wgpu() {
    run_if_available(Device::Gpu, run_native_probe_on);
}

#[cfg(feature = "vulkan")]
#[test]
fn native_probe_prefill_vulkan() {
    run_if_available(Device::Vulkan, run_native_probe_on);
}

#[test]
fn native_probe_decode_step_cpu() {
    run_decode_step_probe_on(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn native_probe_decode_step_metal() {
    run_if_available(Device::Metal, run_decode_step_probe_on);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn native_probe_decode_step_mlx() {
    run_if_available(Device::Mlx, run_decode_step_probe_on);
}

#[cfg(feature = "cuda")]
#[test]
fn native_probe_decode_step_cuda() {
    run_if_available(Device::Cuda, run_decode_step_probe_on);
}

#[cfg(feature = "rocm")]
#[test]
fn native_probe_decode_step_rocm() {
    run_if_available(Device::Rocm, run_decode_step_probe_on);
}

#[cfg(feature = "gpu")]
#[test]
fn native_probe_decode_step_wgpu() {
    run_if_available(Device::Gpu, run_decode_step_probe_on);
}

#[cfg(feature = "vulkan")]
#[test]
fn native_probe_decode_step_vulkan() {
    run_if_available(Device::Vulkan, run_decode_step_probe_on);
}

fn run_aif_masked_decode_on(device: Device, mode: AifDynamicsMode, use_probe_prefill: bool) {
    let prompt = format!("x{MEDIA_MARKER}y");
    let mut fx = synth_fixture(device, 8, &prompt, mode);
    if use_probe_prefill {
        fx.runner
            .prefill_from_assembled_probe(fx.prefill.clone())
            .expect("prefill");
    } else {
        fx.runner
            .prefill_from_assembled(fx.prefill.clone())
            .expect("prefill");
    }
    let probe = fx.runner.probe_aif_native().expect("probe");
    let span = fx.runner.vision_key_span().expect("vision span");
    let blocked = probe.blocked_keys(span);
    assert!(
        !blocked.is_empty(),
        "native probe should block at least one visual key (blocked={blocked:?})"
    );

    fx.runner.clear_aif_decode();
    fx.runner
        .prefill_from_assembled(fx.prefill.clone())
        .expect("prefill");
    let baseline = fx.runner.decode_step(7).expect("baseline");

    fx.runner
        .prefill_from_assembled(fx.prefill)
        .expect("prefill");
    fx.runner.apply_aif_probe(&probe).expect("apply");
    let masked = fx.runner.decode_step(7).expect("decode");
    assert!(masked.iter().all(|v| v.is_finite()));
    assert_ne!(
        baseline, masked,
        "AIF mask ({mode:?}) should change {device:?} decode logits"
    );
}

fn run_aif_masked_decode_prefill_on(device: Device) {
    run_aif_masked_decode_on(device, AifDynamicsMode::PrefillV2t, true);
}

fn run_aif_masked_decode_decode_step_on(device: Device) {
    run_aif_masked_decode_on(device, AifDynamicsMode::DecodeStep, false);
}

#[allow(dead_code)]
fn run_masked_decode_if_available(device: Device, f: fn(Device)) {
    if !is_available(device) {
        eprintln!("skip qwen25-vl aif masked decode {device:?}: backend unavailable");
        return;
    }
    f(device);
}

#[test]
fn aif_masked_decode_cpu() {
    run_aif_masked_decode_prefill_on(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn aif_masked_decode_mlx() {
    run_masked_decode_if_available(Device::Mlx, run_aif_masked_decode_prefill_on);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn aif_masked_decode_metal() {
    run_masked_decode_if_available(Device::Metal, run_aif_masked_decode_prefill_on);
}

#[cfg(feature = "cuda")]
#[test]
fn aif_masked_decode_cuda() {
    run_masked_decode_if_available(Device::Cuda, run_aif_masked_decode_prefill_on);
}

#[cfg(feature = "rocm")]
#[test]
fn aif_masked_decode_rocm() {
    run_masked_decode_if_available(Device::Rocm, run_aif_masked_decode_prefill_on);
}

#[cfg(feature = "gpu")]
#[test]
fn aif_masked_decode_wgpu() {
    run_masked_decode_if_available(Device::Gpu, run_aif_masked_decode_prefill_on);
}

#[cfg(feature = "vulkan")]
#[test]
fn aif_masked_decode_vulkan() {
    run_masked_decode_if_available(Device::Vulkan, run_aif_masked_decode_prefill_on);
}

#[test]
fn aif_masked_decode_decode_step_cpu() {
    run_aif_masked_decode_decode_step_on(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn aif_masked_decode_decode_step_metal() {
    run_masked_decode_if_available(Device::Metal, run_aif_masked_decode_decode_step_on);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn aif_masked_decode_decode_step_mlx() {
    run_masked_decode_if_available(Device::Mlx, run_aif_masked_decode_decode_step_on);
}

#[cfg(feature = "cuda")]
#[test]
fn aif_masked_decode_decode_step_cuda() {
    run_masked_decode_if_available(Device::Cuda, run_aif_masked_decode_decode_step_on);
}

#[cfg(feature = "rocm")]
#[test]
fn aif_masked_decode_decode_step_rocm() {
    run_masked_decode_if_available(Device::Rocm, run_aif_masked_decode_decode_step_on);
}

#[cfg(feature = "gpu")]
#[test]
fn aif_masked_decode_decode_step_wgpu() {
    run_masked_decode_if_available(Device::Gpu, run_aif_masked_decode_decode_step_on);
}

#[cfg(feature = "vulkan")]
#[test]
fn aif_masked_decode_decode_step_vulkan() {
    run_masked_decode_if_available(Device::Vulkan, run_aif_masked_decode_decode_step_on);
}
