//! End-to-end encode/decode with the native codec (any RLX backend tag; CPU SIMD by default).
//!
//! ```bash
//! RLX_TSAC_E2E=1 cargo test -p rlx-tsac --test roundtrip --release -- --nocapture
//! ```

use rlx_runtime::Device;
use rlx_tsac::{TsacBackendKind, TsacCodec, TsacOptions, default_tsac_dir, ensure_tsac};
use std::path::PathBuf;

fn e2e_enabled() -> bool {
    std::env::var("RLX_TSAC_E2E").ok().as_deref() == Some("1")
}

fn fixture_wav() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../rlx-qwen3-tts/examples/audio/ask_not.wav")
}

#[test]
fn roundtrip_native_cpu() {
    if !e2e_enabled() {
        eprintln!("skip roundtrip_native_cpu (set RLX_TSAC_E2E=1)");
        return;
    }
    let dir = default_tsac_dir();
    ensure_tsac(&dir).expect("fetch TSAC with `just fetch-tsac`");
    let wav = fixture_wav();
    assert!(wav.is_file(), "missing fixture {}", wav.display());

    let out = std::env::temp_dir().join(format!("rlx-tsac-roundtrip-{}.wav", std::process::id()));
    let codec = TsacCodec::open_with_options(
        &dir,
        TsacOptions {
            device: Device::Cpu,
            backend: TsacBackendKind::Native,
            quality: Some(6),
            verbose: true,
            ..Default::default()
        },
    )
    .unwrap();
    let stats = codec.roundtrip(&wav, &out).expect("roundtrip");
    assert!(out.is_file());
    assert!(stats.tsac_bytes > 0);
    let _ = std::fs::remove_file(&out);
}

#[test]
fn roundtrip_metal_tag_resolves() {
    if !e2e_enabled() {
        return;
    }
    let dir = default_tsac_dir();
    ensure_tsac(&dir).unwrap();
    let wav = fixture_wav();
    let out = std::env::temp_dir().join(format!("rlx-tsac-metal-{}.wav", std::process::id()));
    let codec = TsacCodec::open_with_options(
        &dir,
        TsacOptions {
            device: Device::Metal,
            backend: TsacBackendKind::Auto,
            quality: Some(6),
            ..Default::default()
        },
    )
    .unwrap();
    let _stats = codec.roundtrip(&wav, &out).expect("metal-tag roundtrip");
    assert!(out.is_file());
    let _ = std::fs::remove_file(&out);
}
