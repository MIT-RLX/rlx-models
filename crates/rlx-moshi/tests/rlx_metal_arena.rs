// Regression test for the Metal "arena > maxBufferLength → nil device" crash:
// compile the full-7B bucketed temporal decode graph (no weights — cheap) on
// Metal. With output-ancestor pinning disabled for the in-order MPS executor the
// arena stays ~27 GB (reused) instead of ~45 GB (pinned); if the fix regressed,
// the arena would exceed maxBufferLength and the MPS assertion would abort this
// test process. (Set RLX_METAL_DEBUG=1 for the arena breakdown.)

use rlx_moshi::config::LmConfig;
use rlx_moshi::rlx_lm::{HeliumDims, build_temporal_decode_graph_bucketed};
use rlx_runtime::{Device, Session, is_available};

#[test]
fn metal_7b_graph_compiles() {
    if !is_available(Device::Metal) {
        eprintln!("skip metal_7b_graph_compiles: no Metal");
        return;
    }
    let cfg = LmConfig::v0_1();
    let dims = HeliumDims::from_cfg(&cfg.transformer, cfg.text_out_vocab_size);
    let graph = build_temporal_decode_graph_bucketed(&dims, 3);
    // Reaching the assert means the single-buffer arena allocated successfully
    // (would abort otherwise) — i.e. it fit under the device's maxBufferLength.
    let _compiled = Session::new(Device::Metal).compile(graph);
    eprintln!("7B temporal decode graph compiled on Metal (arena fit under maxBufferLength)");
}
