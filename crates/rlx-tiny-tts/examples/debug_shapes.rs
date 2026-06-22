//! Dump propagated node output_meta for one TinyTTS subgraph (pre-lowering).
//! Run: `cargo run -p rlx-tiny-tts --example debug_shapes -- weights/tiny-tts-rlx text_encoder 23 bert`

use std::path::PathBuf;

use rlx_onnx_import::{ImportOptions, prepare_onnx_file, shape_propagate::propagate_shapes};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let bundle = args.next().unwrap_or_else(|| "weights/tiny-tts-rlx".into());
    let comp = args.next().unwrap_or_else(|| "text_encoder".into());
    let seq: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(23);
    let filter = args.next().unwrap_or_else(|| "bert".into());
    let path = PathBuf::from(bundle)
        .join("onnx")
        .join(format!("{comp}.onnx"));

    let opts = ImportOptions {
        sequence_length: seq,
        max_waveform_samples: (seq * 1024).max(48_000),
        use_quantized_kernels: false,
        strict: false,
        ..ImportOptions::default()
    };
    let (manifest, mut nodes, _p, _i, init_shapes) = prepare_onnx_file(&path)?;
    for node in &mut nodes {
        for meta in &mut node.output_meta {
            *meta = serde_json::json!({});
        }
    }
    propagate_shapes(&mut nodes, &manifest, &init_shapes, &opts);
    for n in &nodes {
        if n.name.contains(&filter) || n.op == "Conv" {
            let shapes: Vec<_> = n
                .output_meta
                .iter()
                .map(|m| m.get("shape").cloned().unwrap_or(serde_json::json!("?")))
                .collect();
            println!(
                "{:<40} {:<8} in={:?} out={:?}",
                n.name, n.op, n.inputs, shapes
            );
        }
    }
    Ok(())
}
