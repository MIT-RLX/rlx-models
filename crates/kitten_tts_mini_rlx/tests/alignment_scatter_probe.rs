// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    compile_probe_graph, import_from_bundle_cached, probe_output_f32_at, run_parity_inputs,
};
use rlx_runtime::Device;

fn find_node(
    import: &kitten_tts_mini_rlx::bundle_compile::BundleImport,
    name: &str,
) -> rlx_ir::hir::HirNodeId {
    import
        .hir
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("missing {name}"))
        .id
}

#[test]
fn scatter_nd_matches_ort_layout() {
    kitten_tts_mini_rlx::bundle_compile::ensure_kernels_registered();
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    let seq = 8usize;
    let opts = GraphOptions {
        sequence_length: seq,
        max_waveform_samples: seq.saturating_mul(600).saturating_add(12_000),
    };
    let import = import_from_bundle_cached(&bundle_dir, &opts).expect("import");
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 0];
    let style = vec![0.0f32; 256];

    for (name, expect_nz) in [("/ScatterND", 34usize), ("/Unsqueeze_11", 34usize)] {
        let id = find_node(&import, name);
        let mut graph = compile_probe_graph(Device::Cpu, &bundle_dir, &opts, &import, id, name)
            .expect("compile");
        let outs = run_parity_inputs(&mut graph, seq, &ids, &style);
        let vals = probe_output_f32_at(&outs, 0).expect("probe");
        let nz: usize = vals.iter().filter(|&&v| v != 0.0).count();
        eprintln!(
            "{name} shape={:?} len={} nz={} t0={:?}",
            import.hir.node(id).shape.dims(),
            vals.len(),
            nz,
            &vals[..8.min(vals.len())]
        );
        assert_eq!(nz, expect_nz, "{name} nonzero count");
        if name == "/ScatterND" {
            // ORT row-major [8,34]: row0 cols 0..18 are 1.
            for j in 0..19 {
                assert_eq!(vals[j], 1.0, "scatter row0 col {j}");
            }
        }
    }
}
