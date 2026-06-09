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

//! ffn_output/Add should differ from QMatMul by bias (single probe each).

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    compile_probe_graph, import_from_bundle_cached, probe_output_f32_at, run_parity_inputs,
};
use rlx_runtime::Device;
use std::process::Command;

fn load_style_row() -> Vec<f32> {
    let voices =
        std::path::Path::new("/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/voices.npz");
    if !voices.is_file() {
        return vec![0.0; 256];
    }
    let out = Command::new("python3")
        .args([
            "-c",
            "import numpy as np,sys; z=np.load(sys.argv[1]); sys.stdout.buffer.write(z['expr-voice-2-m'][6].astype('float32').tobytes())",
            voices.to_str().unwrap(),
        ])
        .output()
        .expect("style");
    out.stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn ffn_output_add_single_probe_includes_bias() {
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    if !bundle_dir.join("manifest.json").exists() {
        return;
    }
    let opts = GraphOptions {
        sequence_length: 8,
        max_waveform_samples: 24_000,
    };
    let import = import_from_bundle_cached(&bundle_dir, &opts).expect("import");
    let find = |name: &str| {
        import
            .hir
            .nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some(name))
            .map(|n| n.id)
            .unwrap_or_else(|| panic!("missing {name}"))
    };
    kitten_tts_mini_rlx::set_env_var("KITTEN_RLX_SKIP_FUSION", "1");
    kitten_tts_mini_rlx::opts::set_compile_sequence_length(8);
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 0];
    let style = load_style_row();
    let mm_name = "/bert/encoder/albert_layer_groups.0/albert_layers.0/ffn_output/MatMul_quant_f32";
    let add_name = "/bert/encoder/albert_layer_groups.0/albert_layers.0/ffn_output/Add";
    let mut mm_graph = compile_probe_graph(
        Device::Cpu,
        &bundle_dir,
        &opts,
        &import,
        find(mm_name),
        mm_name,
    )
    .expect("mm");
    let mut add_graph = compile_probe_graph(
        Device::Cpu,
        &bundle_dir,
        &opts,
        &import,
        find(add_name),
        add_name,
    )
    .expect("add");
    let mm =
        probe_output_f32_at(&run_parity_inputs(&mut mm_graph, 8, &ids, &style), 0).expect("mm");
    let add =
        probe_output_f32_at(&run_parity_inputs(&mut add_graph, 8, &ids, &style), 0).expect("add");
    let idx = 1272usize;
    let delta = (add[idx] - mm[idx]).abs();
    eprintln!("idx {idx}: mm={} add={} delta={delta}", mm[idx], add[idx]);
    assert!(
        delta > 1e-3,
        "single-probe Add should include bias; mm and add both {}",
        mm[idx]
    );
}
