//! Mimi GPU codec quick check (env `RLX_MIMI_GPU_SMOKE=1`).

use rlx_mimi::{
    MimiCodec, default_mimi_dir, device_ready, parse_mimi_device, resolve_codec_device,
};
use rlx_runtime::Device;
use std::path::PathBuf;

#[test]
fn gpu_codec_roundtrip_smoke() {
    if std::env::var("RLX_MIMI_GPU_SMOKE").ok().as_deref() != Some("1") {
        eprintln!("skip gpu_codec_roundtrip_smoke (set RLX_MIMI_GPU_SMOKE=1)");
        return;
    }
    let device_name = std::env::var("RLX_MIMI_GPU_DEVICE").unwrap_or_else(|_| "metal".into());
    let requested = parse_mimi_device(&device_name).expect("device");
    if !device_ready(requested) {
        eprintln!("skip: {requested:?} not ready");
        return;
    }
    let device = resolve_codec_device(requested);
    if device == Device::Cpu {
        eprintln!("skip: resolved to CPU");
        return;
    }
    let mimi_dir = default_mimi_dir();
    rlx_mimi::ensure_weights(&mimi_dir).expect("mimi weights");
    let moshi_dir = std::env::var("RLX_MOSHI_DIR").ok().map(PathBuf::from);
    if resolve_codec_device(requested) != Device::Cpu
        && rlx_mimi::resolve_candle_weights(&mimi_dir, moshi_dir.as_deref()).is_none()
    {
        eprintln!(
            "skip: no {} for GPU mimi (set RLX_MOSHI_DIR to moshiko bundle)",
            rlx_mimi::MIMI_CANDLE_SIDECAR
        );
        return;
    }
    let codec = MimiCodec::open_on_with_moshi(&mimi_dir, moshi_dir.as_deref(), requested, Some(8))
        .expect("open");
    assert_ne!(codec.device(), Device::Cpu);
    let pcm: Vec<f32> = (0..1920).map(|i| (i as f32 / 1920.0).sin()).collect();
    let codes = codec.encode_pcm(&pcm, Some(8)).expect("encode");
    let recon = codec.decode_codes(&codes).expect("decode");
    assert!(!recon.is_empty());
}
