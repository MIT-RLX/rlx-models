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

//! Print LIR shapes on the duration LSTM path after full compile.

use rlx_onnx_import::{ImportOptions, build_hir_from_bundle, load_bundle};
use rlx_runtime::{CompileOptions, Device, stages};

fn main() -> anyhow::Result<()> {
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    let seq = kitten_tts_mini_rlx::opts::compile_sequence_length_from_env().unwrap_or(8);
    kitten_tts_mini_rlx::opts::set_compile_sequence_length(seq);
    let opts = ImportOptions {
        sequence_length: seq,
        max_waveform_samples: seq.saturating_mul(600).saturating_add(12_000),
        ..ImportOptions::default()
    };
    kitten_tts_mini_rlx::kernels::register_native_kernels();
    let bundle = load_bundle(&bundle_dir)?;
    let (hir, _, _, _) = build_hir_from_bundle(&bundle, opts)?;
    let mut compile_opts = CompileOptions::default();
    compile_opts.fusion_opts.skip_fusion = true;
    let result = stages::compile_hir_stages(Device::Cpu, hir, &compile_opts)?;
    let g = stages::graph_from_lir(result.lir);
    for node in g.nodes() {
        let name = node.name.as_deref().unwrap_or("");
        let op = format!("{:?}", node.op);
        let hit = name.contains("text_encoder_1/Where_4")
            || name.contains("lstm/Transpose")
            || op.contains("DynamicQuantizeLSTM");
        if !hit {
            continue;
        }
        let in0 = node
            .inputs
            .first()
            .map(|&id| g.node(id).shape.dims().to_vec());
        eprintln!(
            "LIR {name:?} op={op} in0={in0:?} out={:?}",
            node.shape.dims()
        );
    }
    eprintln!("LIR node count={}", g.nodes().len());
    Ok(())
}
