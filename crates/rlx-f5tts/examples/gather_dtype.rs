use rlx_runtime::Op;
use rlx_tiny_tts::model::import_graph_named;
use std::path::PathBuf;
fn main() -> anyhow::Result<()> {
    let path = PathBuf::from("weights/tts/f5tts/F5_Preprocess.onnx");
    let named = [
        ("audio_len", 87546usize),
        ("text_ids_len", 107),
        ("max_duration", 580),
        ("text_embed_len", 612),
    ];
    let (hir, _, _) = import_graph_named(&path, "F5_Preprocess", 580, true, &named)?;
    let g = hir.lower_to_mir()?.into_graph();
    for n in g.nodes() {
        if let Op::Gather { axis } = n.op {
            let tab = &g.node(n.inputs[0]).shape;
            let idx = &g.node(n.inputs[1]).shape;
            println!(
                "Gather axis={axis} table={:?} idx={:?} out={:?}",
                tab, idx, n.shape
            );
        }
    }
    Ok(())
}
