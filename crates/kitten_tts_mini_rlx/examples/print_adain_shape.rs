use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::import_from_bundle_cached;
use std::collections::BTreeMap;

fn main() {
    let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    let opts = GraphOptions {
        sequence_length: 16,
        max_waveform_samples: 55_200,
    };
    let import = import_from_bundle_cached(&bundle, &opts).expect("import");
    let mut by_name: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for n in import.hir.nodes() {
        let Some(name) = n.name.as_deref() else { continue };
        if !(name.contains("/generator/") && (name.contains("InstanceNormalization") || name.contains("noise_convs")))
        {
            continue;
        }
        let dims = format!("{:?}", n.shape.dims());
        by_name
            .entry(name.to_string())
            .or_default()
            .push(format!("{dims} op={:?}", n.op));
    }
    for (name, shapes) in &by_name {
        println!("{name} ({} nodes)", shapes.len());
        for s in shapes.iter().take(4) {
            println!("  {s}");
        }
    }
    println!("total named groups: {}", by_name.len());
}
