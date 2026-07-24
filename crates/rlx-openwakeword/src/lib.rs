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

//! Native openWakeWord wake-word detection on RLX.

pub mod cli;
pub mod embedding;
pub mod phrase;
pub mod session;
pub mod train;

#[cfg(feature = "onnx")]
pub mod onnx;

pub use session::{OpenWakeWordEngine, OpenWakeWordWeights};
pub use train::train_phrase_head;
pub use rlx_wake::{
    SAMPLE_RATE_16K, WakeConfig, WakeEngine, WakeStep, assert_100_percent_parity,
    available_devices, bench_device_label, bench_engine, best_f1_threshold, bind_streaming_device,
    detection_stats, float_precision, load_wav_mono_f32, parse_device_list, peak_of, peak_score,
    print_bench_table, print_detection_stats, resolve_device, resample_linear, run_backend_parity,
    score_wav, streaming_execution_device,
};
