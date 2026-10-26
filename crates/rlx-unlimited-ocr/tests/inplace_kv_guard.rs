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

//! Single-step decode must append K/V in place; chunk decode must not.
//!
//! `Op::KvAppend` writes one row into a cache declared with a spare row, which
//! is O(1) instead of `concat`'s O(context) recopy. Two ways to lose it, both
//! silent, so both are pinned here:
//!
//! * the fast path quietly stops being emitted — `concat` is still CORRECT, so
//!   nothing fails and decode just gets slower with context. That is exactly
//!   how it became dead code across the workspace once before;
//! * it gets emitted where it must not. `seq > 1` (chunked / speculative
//!   decode) appends `seq` rows, and a one-row write would drop the rest. This
//!   crate's `chunk_decode_matches_stepwise_decode` caught that for real, every
//!   position off by ~1.0, when the flow declared the spare row for all `seq`.

use rlx_ir::{HirOp, Op};
use rlx_unlimited_ocr::lm_graph::build_unlimited_ocr_decode_built_ext;

mod tiny;
use tiny::{cfg, synthetic_weights};

fn kv_appends(batch: usize, past_seq: usize, seq: usize) -> usize {
    let c = cfg();
    let mut wm = synthetic_weights(&c);
    let built = build_unlimited_ocr_decode_built_ext(&c, &mut wm, batch, past_seq, seq > 1)
        .expect("decode flow should build");
    let hir = built.into_hir().expect("hir");
    hir.nodes()
        .iter()
        .filter(|n| matches!(n.op, HirOp::Mir(Op::KvAppend { .. })))
        .count()
}

#[test]
fn single_step_decode_appends_kv_in_place() {
    let c = cfg();
    assert_eq!(
        kv_appends(1, 4, 1),
        2 * c.num_hidden_layers,
        "decode fell back to concat: expected one KvAppend per K and per V. \
         Note this crate's decode layer is CRATE-LOCAL (lm_graph.rs), so it has \
         to route its cache growth through rlx_flow::blocks::append_kv_row."
    );
}

#[test]
fn batched_decode_keeps_concat() {
    // `batch > 1` strides the axis-1 prefix of a `[batch, seq, kv_dim]` cache,
    // so no single offset describes the output and it cannot alias.
    assert_eq!(
        kv_appends(2, 4, 1),
        0,
        "batched decode must not emit KvAppend: the prefix is strided"
    );
}
