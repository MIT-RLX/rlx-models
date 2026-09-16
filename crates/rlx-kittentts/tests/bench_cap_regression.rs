//! The TTS bench's load parameters must produce real audio on every backend.
//!
//! `rlx-tts-bench`'s kittentts adapter loads with `(256 tokens, 200_000 samples)`. 200 000 is
//! not a whole number of vocoder frames, so the NSF sine chain was built at
//! `ceil(200_000/300)*300 = 200_100` while the wave axis it feeds stayed at 200 000. MLX
//! rejected the resulting `Reshape` outright; CPU and Metal accepted it and read a
//! 100-sample-misaligned harmonic source. `bundle_patches::align_waveform_cap` now holds the
//! invariant — this test pins the end-to-end behaviour at exactly those parameters.

#![cfg(feature = "native")]

mod support;

use std::sync::Mutex;

use rlx_kittentts::{Device, KittenTTS, assets, phrase_fixtures::LONG_IPA};

/// See `native_smoke.rs`: the native engine's compile caps are process-global.
static NATIVE_COMPILE_LOCK: Mutex<()> = Mutex::new(());

/// Exactly what `rlx_tts_bench::adapters::kittentts::make` passes.
const BENCH_TOKENS: usize = 256;
const BENCH_WAVE_SAMPLES: usize = 200_000;

fn synth_at_bench_caps(device: Device) -> Option<Vec<f32>> {
    let dir = assets::default_model_dir().ok()?;
    if assets::default_native_weights_dir().is_none() {
        eprintln!("skip: no decomposed native weights");
        return None;
    }
    support::setup_native_smoke_env();
    let _guard = NATIVE_COMPILE_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let tts = KittenTTS::load_native_from_dir(&dir, device, BENCH_TOKENS, BENCH_WAVE_SAMPLES)
        .expect("load kittentts at bench caps");
    let voice = tts.voice_names().first().expect("voice").clone();
    Some(
        tts.generate_from_ipa(LONG_IPA, &voice, 1.0, 6)
            .expect("generate at bench caps"),
    )
}

#[test]
fn bench_caps_synthesize_on_cpu() {
    let Some(audio) = synth_at_bench_caps(Device::Cpu) else {
        return;
    };
    // The long fixture is ~5 s; a dead NSF source or a truncated wave shows up as both a
    // short buffer and a collapsed peak, so assert on both.
    support::assert_audible(&audio, 80_000);
}

#[cfg(feature = "mlx")]
#[test]
fn bench_caps_synthesize_on_mlx() {
    if !rlx_runtime::is_available(Device::Mlx) {
        eprintln!("skip: mlx unavailable");
        return;
    }
    let Some(audio) = synth_at_bench_caps(Device::Mlx) else {
        return;
    };
    support::assert_audible(&audio, 80_000);
}
