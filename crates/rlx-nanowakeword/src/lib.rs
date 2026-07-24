// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

pub mod cli;
pub mod session;

#[cfg(feature = "onnx")]
pub mod onnx;

pub use session::{NanoWakeWordEngine, NanoWakeWordWeights};
pub use rlx_wake::{
    SAMPLE_RATE_16K, WakeCnnConfig, WakeConfig, WakeEngine, WakeStep,
    assert_100_percent_parity, available_devices, bench_device_label, bench_engine,
    bind_streaming_device, load_wav_mono_f32, parse_device_list, peak_score, print_bench_table,
    resolve_device, resample_linear, run_backend_parity, score_wav, streaming_execution_device,
};
