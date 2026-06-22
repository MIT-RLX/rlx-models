// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Per-backend Orpheus TTS bench + Whisper ASR loop (named voice + optional clone).
//
// Run:
//   just fetch-orpheus fetch-orpheus-snac fetch-whisper
//   just test-orpheus-backends-whisper
//
// Voice clone (pretrained GGUF + encoded reference JSON):
//   ORPHEUS_PRETRAINED_GGUF=… ORPHEUS_CLONE_REF_JSON=… \
//     cargo test -p rlx-orpheus --test backends_whisper --features all-backends --release voice_clone_whisper -- --nocapture

mod support;

use rlx_orpheus::{VoiceCloneReference, lm_kv_decode_supported};
use rlx_runtime::{Device, is_available};
use support::{
    BenchRow, bench_named_voice, bench_text, bench_voice, bench_voice_clone, device_label,
    orpheus_gguf_path, orpheus_pretrained_gguf_path, require_weights, synth_bench_to_row,
    voice_clone_ref_path, whisper_asr_dir,
};

fn clone_target_text() -> String {
    std::env::var("ORPHEUS_CLONE_TARGET")
        .unwrap_or_else(|_| "I write my software in Rust because it is fast.".into())
}

fn run_named_voice_bench(device: Device) -> anyhow::Result<BenchRow> {
    if !is_available(device) {
        anyhow::bail!("device {:?} unavailable", device);
    }
    let Some((gguf, snac)) = require_weights() else {
        anyhow::bail!("missing Orpheus weights — run `just fetch-orpheus fetch-orpheus-snac`");
    };
    let whisper = whisper_asr_dir();
    if whisper.is_none() {
        eprintln!("warning: Whisper not found — run `just fetch-whisper`");
    }
    let text = bench_text();
    let voice = bench_voice();
    eprintln!(
        "bench named voice on {:?}: voice={voice:?} text={text:?}",
        device
    );
    let result = bench_named_voice(&gguf, &snac, device, &text, &voice, whisper.as_deref())?;
    Ok(synth_bench_to_row(
        device_label(device),
        device,
        &result,
        &text,
    ))
}

fn run_voice_clone_bench(device: Device) -> anyhow::Result<BenchRow> {
    if !is_available(device) {
        anyhow::bail!("device {:?} unavailable", device);
    }
    let Some(gguf) = orpheus_pretrained_gguf_path().or_else(orpheus_gguf_path) else {
        anyhow::bail!("missing GGUF — set ORPHEUS_PRETRAINED_GGUF or run `just fetch-orpheus`");
    };
    let Some(snac) = support::snac_decoder_path() else {
        anyhow::bail!("missing SNAC — run `just fetch-orpheus-snac`");
    };
    let Some(ref_path) = voice_clone_ref_path() else {
        anyhow::bail!(
            "missing clone reference — run `just orpheus-encode-ref WAV TRANSCRIPT` and set ORPHEUS_CLONE_REF_JSON"
        );
    };
    let whisper = whisper_asr_dir();
    let reference = VoiceCloneReference::load_json(&ref_path)?;
    let target = clone_target_text();
    eprintln!(
        "bench voice clone on {:?}: ref={} target={target:?}",
        device,
        ref_path.display()
    );
    let result = bench_voice_clone(
        &gguf,
        &snac,
        device,
        &reference,
        &target,
        whisper.as_deref(),
    )?;
    Ok(synth_bench_to_row(
        &format!("{}-clone", device_label(device)),
        device,
        &result,
        &target,
    ))
}

macro_rules! backend_named_voice_test {
    ($name:ident, $dev:expr, $feat:meta) => {
        #[cfg($feat)]
        #[test]
        fn $name() -> anyhow::Result<()> {
            if !is_available($dev) {
                eprintln!("skip: {:?} unavailable", $dev);
                return Ok(());
            }
            if !lm_kv_decode_supported($dev) {
                eprintln!("skip: {:?} LM KV decode not enabled", $dev);
                return Ok(());
            }
            if require_weights().is_none() {
                eprintln!("skip: missing Orpheus weights");
                return Ok(());
            }
            let row = run_named_voice_bench($dev)?;
            row.print();
            assert!(
                row.codes >= 28,
                "expected >= 28 SNAC codes on {:?}, got {}",
                $dev,
                row.codes
            );
            if whisper_asr_dir().is_some() {
                assert!(
                    row.whisper_ok,
                    "Whisper failed on {:?}\nref: {}\ngot: {}",
                    $dev,
                    bench_text(),
                    row.transcript
                );
            }
            Ok(())
        }
    };
}

macro_rules! backend_clone_test {
    ($name:ident, $dev:expr, $feat:meta) => {
        #[cfg($feat)]
        #[test]
        #[ignore = "opt-in voice clone — set ORPHEUS_CLONE_BENCH=1"]
        fn $name() -> anyhow::Result<()> {
            if std::env::var("ORPHEUS_CLONE_BENCH").ok().as_deref() != Some("1") {
                eprintln!("skip clone bench: set ORPHEUS_CLONE_BENCH=1");
                return Ok(());
            }
            if !is_available($dev) {
                eprintln!("skip: {:?} unavailable", $dev);
                return Ok(());
            }
            if !lm_kv_decode_supported($dev) {
                eprintln!("skip: {:?} LM KV decode not enabled", $dev);
                return Ok(());
            }
            let row = run_voice_clone_bench($dev)?;
            row.print();
            assert!(row.codes >= 28, "clone produced too few codes");
            if whisper_asr_dir().is_some() {
                assert!(
                    row.whisper_ok,
                    "Whisper failed clone on {:?}\nref: {}\ngot: {}",
                    $dev,
                    clone_target_text(),
                    row.transcript
                );
            }
            Ok(())
        }
    };
}

#[test]
fn named_voice_whisper_cpu() -> anyhow::Result<()> {
    if std::env::var("ORPHEUS_CPU_BENCH").ok().as_deref() != Some("1") {
        eprintln!("skip CPU bench: packed 3B is slow — set ORPHEUS_CPU_BENCH=1");
        return Ok(());
    }
    if require_weights().is_none() {
        eprintln!("skip: missing Orpheus weights");
        return Ok(());
    }
    let row = run_named_voice_bench(Device::Cpu)?;
    row.print();
    if whisper_asr_dir().is_some() {
        assert!(row.whisper_ok, "Whisper failed on CPU");
    }
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "slow + ~12GB RAM — set ORPHEUS_METAL_BENCH=1"]
fn named_voice_whisper_metal_full() -> anyhow::Result<()> {
    if std::env::var("ORPHEUS_METAL_BENCH").ok().as_deref() != Some("1") {
        eprintln!("skip full Metal bench: set ORPHEUS_METAL_BENCH=1");
        return Ok(());
    }
    if require_weights().is_none() {
        eprintln!("skip: missing Orpheus weights");
        return Ok(());
    }
    if !is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return Ok(());
    }
    let row = run_named_voice_bench(Device::Metal)?;
    row.print();
    assert!(row.codes >= 28, "expected >= 28 codes, got {}", row.codes);
    if whisper_asr_dir().is_some() {
        assert!(row.whisper_ok, "Whisper failed on Metal");
    }
    Ok(())
}
backend_named_voice_test!(
    named_voice_whisper_mlx,
    Device::Mlx,
    all(target_os = "macos", feature = "mlx")
);
backend_named_voice_test!(named_voice_whisper_cuda, Device::Cuda, feature = "cuda");
backend_named_voice_test!(named_voice_whisper_rocm, Device::Rocm, feature = "rocm");
backend_named_voice_test!(named_voice_whisper_wgpu, Device::Gpu, feature = "gpu");
backend_named_voice_test!(
    named_voice_whisper_vulkan,
    Device::Vulkan,
    feature = "vulkan"
);

backend_clone_test!(voice_clone_whisper_cpu, Device::Cpu, feature = "llama");
backend_clone_test!(
    voice_clone_whisper_metal,
    Device::Metal,
    all(target_os = "macos", feature = "metal")
);
backend_clone_test!(
    voice_clone_whisper_mlx,
    Device::Mlx,
    all(target_os = "macos", feature = "mlx")
);
