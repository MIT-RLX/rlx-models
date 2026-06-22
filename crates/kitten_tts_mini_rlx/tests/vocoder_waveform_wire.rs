//! Native vocoder waveform: `/decoder/generator/Slice_3` must be a real narrow, not a zero stub.

#![cfg(feature = "native")]

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::native::flow::build_native_hir;
use rlx_ir::Op;
use rlx_ir::hir::{HirNodeId, HirOp};

const SLICE_3_STUB: &str = "__stub__//decoder/generator/Slice_3_output_0";

#[test]
fn native_vocoder_slice3_is_real_narrow() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights");
    if !dir.join("model.safetensors").is_file() {
        eprintln!("skip: native weights missing");
        return;
    }
    let opts = GraphOptions {
        sequence_length: 8,
        max_waveform_samples: 8 * 600 + 12_000,
    };
    let weights = kitten_tts_mini_rlx::load_weights(&dir).expect("weights");
    let (hir, params, _) = build_native_hir(&weights, &opts).expect("hir");

    assert!(
        !params.contains_key(SLICE_3_STUB),
        "Slice_3 must not be a zero param stub"
    );

    let has_narrow = (0..hir.len()).any(|idx| {
        let id = HirNodeId(idx as u32);
        matches!(&hir.node(id).op, HirOp::Mir(Op::Narrow { .. }))
    });
    assert!(has_narrow, "expected Op::Narrow for vocoder Slice_3");
}
