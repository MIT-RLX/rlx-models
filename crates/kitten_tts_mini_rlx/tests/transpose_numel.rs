//! Find HIR Expand nodes in l_sin_gen that would OOB at runtime.

use kitten_tts_mini_rlx::bundle_compile::import_from_bundle_cached;
use kitten_tts_mini_rlx::compile_profile::compile_slot_length;
use kitten_tts_mini_rlx::opts::GraphOptions;
use rlx_ir::Op;
use rlx_ir::hir::HirOp;

fn expand_broadcast_ok(in_shape: &rlx_ir::Shape, out_shape: &rlx_ir::Shape) -> bool {
    let in_rank = in_shape.rank();
    let out_rank = out_shape.rank();
    if in_rank > out_rank {
        return false;
    }
    let pad = out_rank.saturating_sub(in_rank);
    for i in 0..out_rank {
        let in_d = if i < pad {
            1
        } else {
            in_shape.dim(i - pad).unwrap_static()
        };
        let out_d = out_shape.dim(i).unwrap_static();
        if in_d != out_d && in_d != 1 {
            return false;
        }
    }
    true
}

#[test]
fn l_sin_gen_expand_inputs_are_broadcastable() {
    kitten_tts_mini_rlx::bundle_compile::ensure_kernels_registered();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    for (runtime_tokens, max_wave) in [(8usize, 8 * 600 + 12_000), (25, 25 * 600 + 12_000)] {
        let seq = compile_slot_length(runtime_tokens);
        let opts = GraphOptions {
            sequence_length: seq,
            max_waveform_samples: max_wave,
        };
        let import = import_from_bundle_cached(&dir, &opts).expect("import");
        let mut bad = Vec::new();
        for n in import.hir.nodes() {
            let Some(name) = n.name.as_deref() else {
                continue;
            };
            if !name.contains("/decoder/generator/m_source/l_sin_gen/Expand") {
                continue;
            }
            let HirOp::Mir(Op::Expand { .. }) = &n.op else {
                continue;
            };
            let Some(&inp) = n.inputs.first() else {
                continue;
            };
            if !expand_broadcast_ok(&import.hir.node(inp).shape, &n.shape) {
                bad.push((
                    name.to_string(),
                    format!("in_name={:?}", import.hir.node(inp).name),
                ));
            }
        }
        for (name, msg) in &bad {
            eprintln!("seq={seq} expand {name}: {msg}");
        }
        assert!(
            bad.is_empty(),
            "seq={seq} bad l_sin_gen expands: {}",
            bad.len()
        );
    }
}

#[test]
fn transpose_input_output_numel_match_on_matmul_path() {
    kitten_tts_mini_rlx::bundle_compile::ensure_kernels_registered();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    for (runtime_tokens, max_wave) in [(8usize, 8 * 600 + 12_000), (25, 25 * 600 + 12_000)] {
        let seq = compile_slot_length(runtime_tokens);
        let opts = GraphOptions {
            sequence_length: seq,
            max_waveform_samples: max_wave,
        };
        let import = import_from_bundle_cached(&dir, &opts).expect("import");
        let mut bad = Vec::new();
        for n in import.hir.nodes() {
            let HirOp::Mir(Op::Transpose { .. }) = &n.op else {
                continue;
            };
            let Some(&inp) = n.inputs.first() else {
                continue;
            };
            let in_n = import.hir.node(inp).shape.num_elements().unwrap_or(0);
            let out_n = n.shape.num_elements().unwrap_or(0);
            if in_n != out_n && in_n > 0 && out_n > 0 {
                bad.push((n.name.clone(), in_n, out_n));
            }
        }
        assert!(
            bad.is_empty(),
            "seq={seq} transpose numel mismatches: {}",
            bad.len()
        );
    }
}
