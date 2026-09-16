//! `last_logits_only` must return the logits of the token `last_token_idx`
//! names — not position 0.
//!
//! Two things make that worth pinning. The prefill flow declares
//! `last_token_idx` at flow level as **F32** (`flow.rs`), while the
//! `GatherLastToken` block declares its own input of the same name as **I32**
//! (`rlx-flow/src/blocks/gather_last_token.rs`). So the graph carries two
//! `Op::Input` nodes sharing one name and disagreeing on dtype, and binding is
//! by name and reaches a single node — `rlx_ir::verify_unique_leaf_names`
//! reports it.
//!
//! Both failure modes land in the same place: an unbound index reads zeros, and
//! an F32 index reinterpreted as raw integer bytes becomes a huge out-of-range
//! value that clamps. Either way the gather returns **position 0**. The flow's
//! own comment records having been bitten by the dtype half of this before.
//!
//! The check compares `last_logits_only = true` against the full-sequence
//! logits: the gathered row must equal the row at `last_token_idx`, and — since
//! position 0 is the specific wrong answer both modes produce — must *not*
//! equal row 0.

use rlx_qwen35::synth::{synth_weights, tiny_cfg};
use rlx_runtime::{Device, Session};

const BATCH: usize = 1;
const SEQ: usize = 4;

/// Prompt whose last token differs from its first, so "gathered row 0" and
/// "gathered the last row" are distinguishable.
const INPUT_IDS: [f32; SEQ] = [5.0, 11.0, 2.0, 19.0];

/// Deterministic pseudo-random scalar from a seed and index.
fn hashed(seed: u64, i: usize) -> f32 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x >> 40) as f32) / 8_388_608.0 - 1.0
}

/// Replace every parameter with full-rank pseudo-random values.
///
/// `synth_weights` builds its projections and embedding table from `ramp()` — a
/// strictly linear sequence — and the resulting model is so nearly
/// position-invariant that the logits at position 0 and position 3 differ by
/// ~1e-4. That is far too little to tell a correct gather from one that clamped
/// to position 0, which is exactly what this test has to distinguish. Keyed by
/// name so both runs below get identical weights.
fn randomize(params: &mut std::collections::HashMap<String, Vec<f32>>) {
    for (name, data) in params.iter_mut() {
        let seed = name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x1000_0000_01b3)
        });
        for (i, v) in data.iter_mut().enumerate() {
            *v = 0.25 * hashed(seed, i);
        }
    }
}

fn logits(last_logits_only: bool) -> (Vec<f32>, usize) {
    let cfg = tiny_cfg();
    let weights = synth_weights(&cfg);
    let (hir, mut params, packed) = rlx_qwen35::build_qwen35_prefill_flow(
        &cfg,
        &weights,
        BATCH,
        SEQ,
        true,
        last_logits_only,
        false,
    )
    .expect("build prefill flow");
    assert!(packed.is_empty(), "synthetic weights should not be packed");
    randomize(&mut params);

    let mut compiled = Session::new(Device::Cpu)
        .compile_hir(hir)
        .expect("compile prefill");
    for (name, data) in &params {
        compiled.set_param(name, data);
    }

    let last = (SEQ - 1) as f32;
    let feeds: Vec<(&str, &[f32])> = if last_logits_only {
        vec![
            ("input_ids", &INPUT_IDS[..]),
            ("last_token_idx", std::slice::from_ref(&last)),
        ]
    } else {
        vec![("input_ids", &INPUT_IDS[..])]
    };
    let out = compiled.run(&feeds);
    let vocab = out[0].len() / if last_logits_only { 1 } else { SEQ };
    (out[0].clone(), vocab)
}

#[test]
fn last_logits_only_gathers_the_last_token_not_position_zero() {
    let (full, vocab) = logits(false);
    assert_eq!(
        full.len(),
        SEQ * vocab,
        "full run should emit every position"
    );

    let row = |p: usize| &full[p * vocab..(p + 1) * vocab];
    // The rows must actually differ, or the test cannot tell them apart.
    let spread = row(0)
        .iter()
        .zip(row(SEQ - 1))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        spread > 1e-3,
        "position 0 and position {} produce the same logits (max diff {spread}); \
         this prompt cannot distinguish a correct gather from a clamp-to-zero",
        SEQ - 1
    );

    let (gathered, gathered_vocab) = logits(true);
    assert_eq!(gathered_vocab, vocab, "vocab size changed between runs");
    assert_eq!(
        gathered.len(),
        vocab,
        "last_logits_only should emit one row"
    );

    let diff_to = |p: usize| {
        gathered
            .iter()
            .zip(row(p))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    let to_last = diff_to(SEQ - 1);
    let to_first = diff_to(0);

    assert!(
        to_last < 1e-4,
        "last_logits_only returned something other than position {}: \
         max diff to the last row {to_last}, to row 0 {to_first}{}",
        SEQ - 1,
        if to_first < 1e-4 {
            " — it returned position 0, the value an unbound or \
             byte-reinterpreted last_token_idx produces"
        } else {
            ""
        }
    );
}

/// KNOWN, LATENT — the prefill graph carries two `last_token_idx` inputs.
///
/// `rlx-qwen35`'s prefill flow declares `last_token_idx` at flow level as
/// **F32**; the `GatherLastToken` block declares its own input of the same name
/// as **I32**. Both nodes are live, and binding reaches only one of them.
///
/// Measured on `tiny_cfg`, batch 1, seq 4 (node ids from the lowered MIR):
///
/// ```text
/// %211 Input I32  ← bound        %213 Gather{axis:1}(hidden[1,4,16]) → [1,1,1,16]
/// %1   Input F32  ← unbound      %220 Gather{axis:1}(logits[1,1,1,32]) → [1,1,1,1,32]
/// ```
///
/// `%213` is the real gather and takes the bound index, so the logits are
/// right — `last_logits_only_gathers_the_last_token_not_position_zero` above
/// confirms it. `%220` is a *second, redundant* gather over the already-gathered
/// logits, along an axis the first gather collapsed to extent 1, where index 0
/// is the only legal index. The zeros the unbound node reads are therefore
/// harmless, and the only visible trace is a spurious rank-5 output shape.
///
/// It is latent rather than benign. Which of two same-named nodes gets bound is
/// not a property anything guarantees; if it went the other way, `%213` would
/// gather index 0 and every `last_logits_only` prefill would silently return
/// the *first* token's logits. The flow's own comment records being bitten by
/// the dtype half of this already.
///
/// Left failing-but-ignored rather than fixed: removing the redundant gather
/// changes the logits output rank, which the runner, speculative-decode and
/// serving paths all consume, so it wants its own change with its own tests.
#[test]
#[ignore = "known latent: two last_token_idx inputs (F32 flow-level + I32 block); see the doc comment"]
fn prefill_graph_has_unique_leaf_names() {
    let cfg = tiny_cfg();
    let weights = synth_weights(&cfg);
    let (hir, _params, _packed) =
        rlx_qwen35::build_qwen35_prefill_flow(&cfg, &weights, BATCH, SEQ, true, true, false)
            .expect("build prefill flow");
    let graph = hir.lower_to_mir().expect("lower to MIR").into_graph();

    let errors = rlx_ir::verify_unique_leaf_names(&graph);
    assert!(
        errors.is_empty(),
        "prefill graph has duplicate leaf names: {}",
        errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    );
}
