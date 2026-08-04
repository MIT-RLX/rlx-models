// Dump the compiled-graph shapes around decoder_body's layers.0 attention, using
// the EXACT import path the codec runs (rlx-tiny-tts import_graph_named), to pin
// where the [B,T,H,D]=[1,200,8,32] RoPE layout gets mis-reshaped.
//   cargo run -p rlx-miotts --example attn_shape_probe
use rlx_tiny_tts::model::import_graph_named;

fn main() -> anyhow::Result<()> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let onnx = root.join("weights/tts/miocodec/decoder_body.onnx");
    let (hir, _params, report) = import_graph_named(&onnx, "decoder_body", 100, false, &[])?;
    eprintln!(
        "[probe] lowered={} stubbed={} unsupported={:?}",
        report.lowered, report.stubbed, report.unsupported
    );
    let graph = rlx_ir::hir_to_graph(hir)?;
    let dims = |id: rlx_ir::NodeId| format!("{:?}", graph.node(id).shape.dims());

    // Print every node whose name mentions layers.0/attention, with I/O shapes.
    let want = std::env::var("WANT").unwrap_or_else(|_| "wave_decoder/layers.0/attention".into());
    let mut count = 0;
    for n in graph.nodes() {
        let nm = n.name.clone().unwrap_or_default();
        if !nm.contains(want.as_str()) {
            continue;
        }
        let ins: Vec<String> = n.inputs.iter().map(|&i| dims(i)).collect();
        println!(
            "{:16} out={} name={}\n     in={:?}",
            format!("{:?}", n.op.kind()),
            dims(n.id),
            nm.rsplit('/').next().unwrap_or(""),
            ins
        );
        count += 1;
        if count > 60 {
            break;
        }
    }
    Ok(())
}
