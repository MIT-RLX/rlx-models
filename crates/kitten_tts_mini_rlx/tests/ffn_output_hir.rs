// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Inspect ffn_output Add wiring in imported HIR.

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::import_from_bundle_cached;

#[test]
fn ffn_output_add_hir_inputs() {
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    if !bundle_dir.join("manifest.json").exists() {
        return;
    }
    let opts = GraphOptions {
        sequence_length: 8,
        max_waveform_samples: 24_000,
    };
    let import = import_from_bundle_cached(&bundle_dir, &opts).expect("import");
    let add_name = "/bert/encoder/albert_layer_groups.0/albert_layers.0/ffn_output/Add";
    let mm_name = "/bert/encoder/albert_layer_groups.0/albert_layers.0/ffn_output/MatMul_quant_f32";
    let add = import
        .hir
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some(add_name))
        .expect("Add");
    let mm = import
        .hir
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some(mm_name))
        .expect("QMatMul");
    eprintln!("QMatMul op={:?}", mm.op);
    for (i, &inp) in import.hir.node(add.id).inputs.iter().enumerate() {
        let n = &import.hir.node(inp);
        eprintln!("Add input[{i}] name={:?} op={:?}", n.name, n.op);
    }
}
