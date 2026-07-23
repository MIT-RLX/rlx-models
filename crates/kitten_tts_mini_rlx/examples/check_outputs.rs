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

//! Dump native graph output sizes and duration scalar.

use rlx_runtime::Device;

fn main() -> anyhow::Result<()> {
    let weights = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights");
    kitten_tts_mini_rlx::set_env_var(
        kitten_tts_mini_rlx::opts::ONNX_BUNDLE_ENV,
        weights.join("rlx_bundle"),
    );

    let seq = kitten_tts_mini_rlx::opts::compile_sequence_length_from_env().unwrap_or(128);
    let opts = kitten_tts_mini_rlx::GraphOptions {
        sequence_length: seq,
        max_waveform_samples: seq.saturating_mul(600).saturating_mul(5).max(24_000),
    };
    let mut graph = kitten_tts_mini_rlx::bundle_compile::compile_from_bundle_fresh(
        Device::Cpu,
        &weights.join("rlx_bundle"),
        &opts,
    )?;

    // "həˈloʊ" — matches rlx-kittentts `ipa_to_ids`
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 10, 0];
    let ids_use: Vec<i64> = if ids.len() <= seq {
        ids.clone()
    } else {
        ids[..seq].to_vec()
    };
    let style = vec![0.0f32; 256];
    let speed = 1.0f32;

    let ids_bytes: Vec<u8> = ids_use.iter().flat_map(|v| v.to_le_bytes()).collect();
    let style_bytes: Vec<u8> = style.iter().flat_map(|v| v.to_le_bytes()).collect();
    let speed_bytes: Vec<u8> = speed.to_le_bytes().to_vec();

    kitten_tts_mini_rlx::opts::set_compile_sequence_length(ids_use.len());
    let shape_bytes: Vec<u8> = [1i64, ids_use.len() as i64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    graph.set_param_typed(
        kitten_tts_mini_rlx::opts::RUNTIME_INPUT_IDS_SHAPE,
        &shape_bytes,
        rlx_ir::DType::I64,
    );

    let outs = graph.run_typed(&[
        ("input_ids", ids_bytes.as_slice(), rlx_ir::DType::I64),
        ("style", style_bytes.as_slice(), rlx_ir::DType::F32),
        ("speed", speed_bytes.as_slice(), rlx_ir::DType::F32),
    ]);

    for (i, (bytes, dt)) in outs.iter().enumerate() {
        eprintln!("out[{i}]: dtype={dt:?} bytes={}", bytes.len());
        if *dt == rlx_ir::DType::I64 && bytes.len() >= 8 {
            let n = bytes.len() / 8;
            let sum: i64 = bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .sum();
            eprintln!(
                "  i64 elems={n} sum={sum} first={}",
                i64::from_le_bytes(bytes[0..8].try_into().unwrap())
            );
            let preview: Vec<i64> = bytes
                .chunks_exact(8)
                .take(16)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            eprintln!("  i64[0..16]={preview:?}");
        }
        if *dt == rlx_ir::DType::F32 && !bytes.is_empty() {
            let n = bytes.len() / 4;
            let first: f32 = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
            let max_abs = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).abs())
                .fold(0.0f32, f32::max);
            eprintln!("  f32 elems={n} first={first} max_abs={max_abs}");
        }
    }
    Ok(())
}
