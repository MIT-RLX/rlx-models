//! Keystone check: import + compile + run the TinyTTS `decoder` ONNX graph on
//! CPU (it has no Split — the simplest of the four). Proves the rlx-onnx-import
//! → rlx-ir → run_typed path works end-to-end before wiring the full pipeline.
//!
//! Run: `cargo run -p rlx-tiny-tts --example keystone -- weights/tiny-tts-rlx`

use std::path::PathBuf;

use rlx_runtime::{DType, Device};

fn main() -> anyhow::Result<()> {
    let bundle = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "weights/tiny-tts-rlx".to_string());
    let onnx_dir = PathBuf::from(bundle).join("onnx");

    let y_len = 64usize; // frames
    let c = 80usize; // latent channels
    println!("compiling decoder for y_len={y_len} on CPU…");
    let mut dec = rlx_tiny_tts::model::compile_graph(&onnx_dir, "decoder", Device::Cpu, y_len)?;

    let z = vec![0.05f32; c * y_len];
    let g = vec![0.0f32; c];
    let z_b: Vec<u8> = z.iter().flat_map(|x| x.to_le_bytes()).collect();
    let g_b: Vec<u8> = g.iter().flat_map(|x| x.to_le_bytes()).collect();

    println!("running decoder…");
    let out = dec.run_typed(&[("z", &z_b, DType::F32), ("g", &g_b, DType::F32)]);
    let (bytes, dt) = out.first().expect("decoder output");
    let n = bytes.len() / dt.size_bytes().max(1);
    println!(
        "decoder output: dtype={dt:?} samples={n} (expected ≈ {})",
        y_len * 512
    );
    anyhow::ensure!(n > 0, "decoder produced no samples");
    Ok(())
}
