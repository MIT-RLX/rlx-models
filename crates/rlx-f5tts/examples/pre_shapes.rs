//! Debug: import F5_Preprocess and print each output's resolved HIR shape.
//! `RLX_F5TTS_DIR=... cargo run -p rlx-f5tts --example pre_shapes --release`
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(std::env::var("RLX_F5TTS_DIR").unwrap_or("weights/tts/f5tts".into()));
    let comp = std::env::var("COMP").unwrap_or("F5_Preprocess".into());
    let path = dir.join(format!("{comp}.onnx"));
    // Same named dims as the native path for the smoke inputs (n=12000, t=9, md=141).
    let env = |k: &str, def: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(def)
    };
    let (n, t, d, tel) = (env("N", 12000), env("T", 9), env("D", 141), 612usize);
    // Generic override: NAMED="batch=1,prefill_seq=10" for non-f5 graphs.
    let named_owned: Vec<(String, usize)> = std::env::var("NAMED")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|kv| kv.split_once('='))
                .map(|(k, v)| (k.trim().to_string(), v.trim().parse().unwrap_or(1)))
                .collect()
        })
        .unwrap_or_default();
    let f5_named = [
        ("audio_len", n),
        ("text_ids_len", t),
        ("max_duration", d),
        ("text_embed_len", tel),
    ];
    let named: Vec<(&str, usize)> = if named_owned.is_empty() {
        f5_named.to_vec()
    } else {
        named_owned.iter().map(|(k, v)| (k.as_str(), *v)).collect()
    };
    let named = named.as_slice();
    let comp_static: &'static str = Box::leak(comp.clone().into_boxed_str());
    let (hir, _params, report) =
        rlx_tiny_tts::model::import_graph_named(&path, comp_static, d, false, named)?;
    println!(
        "report: lowered={} skipped={} stubbed={} unsupported={:?}",
        report.lowered, report.skipped, report.stubbed, report.unsupported
    );
    if !report.stubbed_nodes.is_empty() {
        println!("stubbed_nodes: {:?}", report.stubbed_nodes);
    }
    for (i, oid) in hir.outputs.iter().enumerate() {
        let node = hir.node(*oid);
        println!(
            "out[{i}] name={:?} shape={:?} dtype={:?}",
            node.name,
            node.shape.dims(),
            node.shape.dtype()
        );
    }
    // Top-N largest node buffers (by element count) — find arena bloat.
    if std::env::var("TOPN").is_ok() {
        let mut sizes: Vec<(usize, String, Vec<i64>)> = hir
            .nodes()
            .iter()
            .map(|node| {
                let dims: Vec<i64> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i64)
                    .collect();
                let n: usize = dims.iter().map(|&d| d.max(1) as usize).product();
                (n, node.name.clone().unwrap_or_default(), dims)
            })
            .collect();
        sizes.sort_by(|a, b| b.0.cmp(&a.0));
        let total: usize = sizes.iter().map(|s| s.0).sum();
        println!(
            "---- top-20 largest nodes (total elems={total}, ~{:.1} GB f32) ----",
            total as f64 * 4.0 / 1e9
        );
        for (n, name, dims) in sizes.iter().take(20) {
            println!("  {:>12} elems  {name:40} {dims:?}", n);
        }
    }
    // Dump the mel-chain nodes (Conv / MatMul / Log / Clip / Abs / Transpose / Concat).
    let want = std::env::var("DUMP").unwrap_or_default();
    if !want.is_empty() {
        println!("---- mel-chain node shapes ----");
        for node in hir.nodes() {
            if let Some(nm) = &node.name {
                if want.split(',').any(|w| nm.contains(w)) {
                    println!("{nm:40} {:?}", node.shape.dims());
                }
            }
        }
    }
    Ok(())
}
