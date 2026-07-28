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

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{import_from_bundle_cached, prepare_hir_for_compile};

fn main() -> anyhow::Result<()> {
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    let seq = kitten_tts_mini_rlx::opts::compile_sequence_length_from_env().unwrap_or(8);
    let opts = GraphOptions {
        sequence_length: seq,
        max_waveform_samples: seq.saturating_mul(600).saturating_add(12_000),
    };
    let import = import_from_bundle_cached(&bundle_dir, &opts)?;
    let (hir, _) = prepare_hir_for_compile(
        import.hir,
        &import.params,
        &import.typed,
        opts.sequence_length,
        opts.max_waveform_samples,
    );
    let g = hir.lower_to_mir()?.into_graph();
    for node in g.nodes() {
        let name = node.name.as_deref().unwrap_or("");
        let op = format!("{:?}", node.op);
        let hit = name == "/MatMul"
            || name == "/MatMul_1"
            || name.contains("text_encoder_1/Where_4")
            || name.contains("lstm/Transpose")
            || op.contains("DynamicQuantizeLSTM");
        if !hit {
            continue;
        }
        let inputs: Vec<_> = node
            .inputs
            .iter()
            .map(|&id| {
                let input = g.node(id);
                (
                    input.name.as_deref().unwrap_or(""),
                    input.shape.dims().to_vec(),
                )
            })
            .collect();
        eprintln!(
            "LIR {name:?} op={op} inputs={inputs:?} out={:?}",
            node.shape.dims()
        );
    }
    eprintln!("LIR node count={}", g.nodes().len());
    Ok(())
}
