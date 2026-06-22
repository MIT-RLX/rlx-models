//! Try to import+compile each TinyTTS subgraph on CPU via the crate's own
//! import path, isolating failures.
//! Run: `cargo run -p rlx-tiny-tts --example compile_all -- weights/tiny-tts-rlx 24`

use std::path::PathBuf;

use rlx_runtime::Device;

fn main() {
    let mut args = std::env::args().skip(1);
    let bundle = args.next().unwrap_or_else(|| "weights/tiny-tts-rlx".into());
    let seq: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(24);
    let onnx_dir = PathBuf::from(bundle).join("onnx");

    for comp in ["duration_predictor", "flow", "text_encoder", "decoder"] {
        let dir = onnx_dir.clone();
        let r = std::panic::catch_unwind(|| {
            rlx_tiny_tts::model::compile_graph(&dir, comp, Device::Cpu, seq)
        });
        match r {
            Ok(Ok(_)) => println!("OK    {comp} (seq={seq})"),
            Ok(Err(e)) => println!("ERR   {comp}: {e:#}"),
            Err(_) => println!("PANIC {comp} (seq={seq})"),
        }
    }
}
