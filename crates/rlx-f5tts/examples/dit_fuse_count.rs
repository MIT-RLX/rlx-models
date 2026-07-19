//! Count `AdaLayerNorm` / `GatedResidual` after F5 Transformer import + fuse.
//!
//! ```bash
//! cargo run -p rlx-f5tts --example dit_fuse_count --release
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use rlx_f5tts::config::DEFAULT_LOCAL_DIR;
use rlx_ir::Op;
use rlx_opt::{FuseAdaLayerNorm, FuseGatedResidual, Pass, specialize_params};
use rlx_tiny_tts::model::import_graph_named;

fn count(g: &rlx_ir::Graph) -> (usize, usize, usize, usize) {
    let mut ada = 0usize;
    let mut gate = 0usize;
    let mut ln = 0usize;
    let mut nodes = 0usize;
    for n in g.nodes() {
        nodes += 1;
        match &n.op {
            Op::AdaLayerNorm { .. } => ada += 1,
            Op::GatedResidual => gate += 1,
            Op::LayerNorm { .. } | Op::RmsNorm { .. } => ln += 1,
            _ => {}
        }
    }
    (nodes, ada, gate, ln)
}

fn uniform_fills(params: &HashMap<String, Vec<f32>>) -> HashMap<String, Vec<f32>> {
    params
        .iter()
        .filter_map(|(k, v)| {
            let first = *v.first()?;
            if v.iter().all(|&x| x == first) {
                Some((k.clone(), v.clone()))
            } else {
                None
            }
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let model = std::env::var("RLX_F5TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_LOCAL_DIR));
    let d: usize = std::env::var("MD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let path = model.join("F5_Transformer.onnx");
    anyhow::ensure!(path.is_file(), "missing {}", path.display());
    let named = &[("max_duration", d), ("text_embed_len", 612usize)];
    let (hir, params, report) = import_graph_named(&path, "F5_Transformer", d, false, named)?;
    eprintln!(
        "import stubbed={} unsupported={:?} params={} uniform={}",
        report.stubbed,
        report.unsupported,
        params.len(),
        uniform_fills(&params).len()
    );
    let hir_g = rlx_ir::hir_to_graph(hir).map_err(|e| anyhow::anyhow!("hir_to_graph: {e}"))?;
    let (n0, a0, g0, l0) = count(&hir_g);
    eprintln!("pre-fuse: nodes={n0} AdaLN={a0} Gate={g0} LN/RMS={l0}");

    let specialized = specialize_params(&hir_g, &uniform_fills(&params));
    let fused = FuseGatedResidual.run(FuseAdaLayerNorm.run(specialized));
    let (n1, a1, g1, l1) = count(&fused);
    eprintln!("post-fuse: nodes={n1} AdaLayerNorm={a1} GatedResidual={g1} LN/RMS left={l1}");

    // Session pipeline (what tiny-tts runs with param_bindings).
    let (hir2, params2, _) = import_graph_named(&path, "F5_Transformer", d, false, named)?;
    let mut opts = rlx_runtime::CompileOptions::default();
    opts = opts.param_bindings(uniform_fills(&params2));
    let result = rlx_runtime::compile_hir_stages(rlx_runtime::Device::Cpu, hir2, &opts)
        .map_err(|e| anyhow::anyhow!("compile_hir_stages: {e}"))?;
    let g = rlx_runtime::graph_from_lir(result.lir);
    let (n2, a2, g2, l2) = count(&g);
    eprintln!("session-pipe CPU: nodes={n2} AdaLayerNorm={a2} GatedResidual={g2} LN/RMS left={l2}");
    anyhow::ensure!(
        a1 > 0 || a2 > 0,
        "F5 Transformer still has 0 AdaLayerNorm after specialize+fuse (Gate={g1}/{g2})"
    );
    Ok(())
}
