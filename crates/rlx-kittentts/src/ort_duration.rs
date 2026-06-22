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

//! ONNX Runtime duration oracle for native infer (exact length alignment).

#![cfg(all(feature = "native", feature = "onnx"))]

use std::sync::Mutex;

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;

/// When `true`, seed native duration carry from ORT when an ONNX session is attached.
///
/// Narrow compile slots still need ORT duration carry: the native fixed-point loop
/// diverges from ORT (inflated per-token sums) and the vocoder stays near-silent
/// without the correct alignment buffer.
///
/// Set `KITTEN_RLX_NATIVE_DURATION_LOOP=1` to force the native duration loop instead.
pub fn ort_duration_carry_seed_enabled(_compile_seq: usize) -> bool {
    !std::env::var("KITTEN_RLX_NATIVE_DURATION_LOOP")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
}

/// When `true` (default), native infer uses ORT duration for trimming when a session is attached.
pub fn ort_duration_oracle_enabled() -> bool {
    !std::env::var("KITTEN_RLX_NO_ORT_DURATION")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
}

/// Run ORT once and return the waveform output (output index 0).
pub fn fetch_ort_waveform(
    session: &Mutex<Session>,
    ids: &[i64],
    style: &[f32],
    speed: f32,
) -> Result<Vec<f32>> {
    let seq_len = ids.len();
    let style_dim = style.len();
    let t_input_ids =
        Tensor::<i64>::from_array(([1usize, seq_len], ids.to_vec())).context("input_ids")?;
    let t_style =
        Tensor::<f32>::from_array(([1usize, style_dim], style.to_vec())).context("style")?;
    let t_speed = Tensor::<f32>::from_array(([1usize], vec![speed])).context("speed")?;

    let mut session = session.lock().expect("ORT session mutex");
    let outputs = session
        .run(ort::inputs![t_input_ids, t_style, t_speed])
        .context("ORT waveform infer")?;

    let (_shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("extract ORT waveform tensor")?;
    Ok(data.to_vec())
}

/// When `true` (default), fall back to ORT waveform if chunked native underruns ORT duration length.
pub fn ort_waveform_fallback_enabled() -> bool {
    !std::env::var("KITTEN_RLX_NO_ORT_WAVEFORM_FALLBACK")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
}

/// Run ORT once and return the per-token `duration` output (output index 1).
pub fn fetch_ort_duration(
    session: &Mutex<Session>,
    ids: &[i64],
    style: &[f32],
    speed: f32,
) -> Result<Vec<i64>> {
    Ok(fetch_ort_outputs(session, ids, style, speed)?.0)
}

/// Run ORT once and return waveform (0) + duration (1) — avoids a second session.run on fallback.
pub fn fetch_ort_outputs(
    session: &Mutex<Session>,
    ids: &[i64],
    style: &[f32],
    speed: f32,
) -> Result<(Vec<i64>, Vec<f32>)> {
    let seq_len = ids.len();
    let style_dim = style.len();
    let t_input_ids =
        Tensor::<i64>::from_array(([1usize, seq_len], ids.to_vec())).context("input_ids")?;
    let t_style =
        Tensor::<f32>::from_array(([1usize, style_dim], style.to_vec())).context("style")?;
    let t_speed = Tensor::<f32>::from_array(([1usize], vec![speed])).context("speed")?;

    let mut session = session.lock().expect("ORT duration session mutex");
    let outputs = session
        .run(ort::inputs![t_input_ids, t_style, t_speed])
        .context("ORT duration oracle infer")?;

    let (_wave_shape, waveform) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("extract ORT waveform tensor")?;
    let (_dur_shape, duration) = outputs[1]
        .try_extract_tensor::<i64>()
        .context("extract ORT duration tensor")?;
    Ok((duration.to_vec(), waveform.to_vec()))
}
