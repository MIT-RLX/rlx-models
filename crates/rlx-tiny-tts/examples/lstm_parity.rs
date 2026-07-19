//! ONNX float-LSTM import parity harness. Imports `<dir>/lstm.onnx` (generated
//! by scratchpad/gen_lstm.py), runs it on the RLX CPU backend with `<dir>/x.f32`,
//! and writes `<dir>/y_rlx.f32`. A companion Python step compares against the
//! onnxruntime reference `y_ref.f32`.
//!
//! Run: cargo run -p rlx-tiny-tts --example lstm_parity -- <dir> [seq]

use std::path::PathBuf;

use rlx_runtime::{DType, Device};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("usage: lstm_parity <dir> [seq]"));
    let seq: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);

    let x_bytes = std::fs::read(dir.join("x.f32"))?;

    let mut g = rlx_tiny_tts::model::compile_graph(&dir, "lstm", Device::Cpu, seq)?;
    let out = g.run_typed(&[("X", &x_bytes, DType::F32)]);
    let (y_bytes, dt) = out.into_iter().next().expect("no LSTM output");
    anyhow::ensure!(dt == DType::F32, "unexpected output dtype {dt:?}");

    std::fs::write(dir.join("y_rlx.f32"), &y_bytes)?;
    let y: Vec<f32> = y_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    println!(
        "y_rlx: {} values, first 8: {:?}",
        y.len(),
        &y[..y.len().min(8)]
    );
    Ok(())
}
