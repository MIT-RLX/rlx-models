// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Guards on the rlx-ir → RTL lowering.
//!
//! The authoritative check is the SystemVerilog testbench, which requires
//! `iverilog` and 43 M simulated cycles. These are the invariants that can be
//! asserted cheaply, so a lowering regression fails here rather than in a
//! simulator someone has to remember to run:
//!
//! ```text
//! cargo run -p rlx-ten-vad --example export_rtl
//! iverilog -g2012 -I rtl -o sim rtl/tv_lut.sv rtl/tv_core.sv tb/tv_tb.sv
//! (cd rtl && vvp ../sim +frames=250)
//! ```

use rlx_fpga::seq::{SeqConfig, SeqOp, lower_graph};
use rlx_ten_vad::model::{Shape, build_graph};
use rlx_ten_vad::weights::TenVadWeights;

fn lowered() -> rlx_fpga::seq::SeqModel {
    let w = TenVadWeights::embedded();
    let (graph, params) = build_graph(Shape::Streaming, w).expect("build graph");
    let cfg = SeqConfig::default().with_carry([("h1", 1), ("c1", 2), ("h2", 3), ("c2", 4)]);
    lower_graph(&graph, &params, &cfg).expect("lower")
}

#[test]
fn lowers_every_stage_and_nothing_else() {
    let m = lowered();
    // conv0 dw/pw, pool, two separable pairs, two LSTM (matvec + gate), two
    // dense, done.
    assert_eq!(m.descriptors.len(), 14, "stage count");
    assert_eq!(
        m.descriptors
            .iter()
            .filter(|d| d.op == SeqOp::Gate as u32)
            .count(),
        2,
        "both LSTM cells must collapse into Gate descriptors"
    );
    assert_eq!(m.descriptors.last().unwrap().op, SeqOp::Done as u32);
}

#[test]
fn work_matches_the_scalar_net() {
    // 79,295 multiply-accumulates per frame, counted independently from the
    // layer shapes. A lowering that drops or duplicates a stage moves this.
    assert_eq!(lowered().macs(), 79_295);
}

#[test]
fn weights_are_the_whole_network_and_nothing_more() {
    let m = lowered();
    assert_eq!(m.weights.len(), 74_986, "int16 weight count");
}

#[test]
fn scratch_buffers_are_reused() {
    // Conv intermediates are written by one stage and dead after the next, so
    // they should share space. Without reuse this graph needs 3,827 activation
    // words; with it, under two thousand — which drops the emitted RAM from
    // 4096 words to 2048, and the ECP5 BRAM count from 74 blocks to 70.
    //
    // Note what is *not* tested here: that no live buffer is clobbered. That
    // cannot be checked from addresses alone, because reuse means the same
    // address deliberately holds different values at different times — the
    // check would need the allocator's own liveness, and would then only be
    // testing its bookkeeping against itself. The end-to-end proof is the
    // SystemVerilog testbench, which is what caught this going wrong.
    let m = lowered();
    let naive: u32 = m
        .descriptors
        .iter()
        .filter(|d| d.op != SeqOp::Done as u32)
        .map(|d| {
            if d.op == SeqOp::Gate as u32 {
                2 * d.n_i
            } else {
                d.n_o.saturating_sub(1) * d.sd_o + d.n_i.saturating_sub(1) * d.sd_i + 1
            }
        })
        .sum();
    assert!(
        (m.aram_used as u32) < naive,
        "activation RAM {} words is no smaller than the {naive} written without reuse",
        m.aram_used
    );
    assert!(
        m.aram_used <= 2048,
        "activation RAM grew to {} words — reuse may have regressed",
        m.aram_used
    );
}

#[test]
fn carried_state_is_written_only_by_its_own_gate() {
    // Cell state is read once per frame by the gate that updates it, so a
    // live-range analysis scoped to a single inference sees it die immediately.
    // If it were handed out as scratch the model would still score frame 0
    // correctly and drift from frame 1 on — the expensive kind of wrong.
    let m = lowered();
    let gates: Vec<_> = m
        .descriptors
        .iter()
        .filter(|x| x.op == SeqOp::Gate as u32)
        .map(|x| (x.base_c, x.base_c + x.n_i, x.base_d, x.base_d + x.n_i))
        .collect();
    assert_eq!(gates.len(), 2);

    for (i, x) in m.descriptors.iter().enumerate() {
        if x.op == SeqOp::Gate as u32 || x.op == SeqOp::Done as u32 {
            continue;
        }
        let hi = x.base_d + x.n_o.saturating_sub(1) * x.sd_o + x.n_i.saturating_sub(1) * x.sd_i + 1;
        for &(c0, c1, h0, h1) in &gates {
            assert!(
                x.base_d >= c1 || c0 >= hi,
                "stage {i} writes {}..{hi} over cell state {c0}..{c1}",
                x.base_d
            );
            assert!(
                x.base_d >= h1 || h0 >= hi,
                "stage {i} writes {}..{hi} over hidden state {h0}..{h1}",
                x.base_d
            );
        }
    }
}

#[test]
fn fits_the_configured_activation_ram() {
    let m = lowered();
    assert!(
        m.prob_addr < m.aram_words,
        "output at {} exceeds {} words",
        m.prob_addr,
        m.aram_words
    );
    assert_eq!(m.feat_len, 3 * 41, "feature window");
}

#[test]
fn lstm_state_is_updated_in_place() {
    // Carry means the hidden state a gate writes is the same storage the next
    // frame's projection reads. If aliasing broke, the two gates would write
    // somewhere disjoint from any stage's inputs and the model would be
    // stateless — which the parity suite would only catch after many frames.
    let m = lowered();
    let gates: Vec<_> = m
        .descriptors
        .iter()
        .filter(|d| d.op == SeqOp::Gate as u32)
        .collect();
    assert_eq!(gates.len(), 2);
    let reads_h: Vec<_> = m
        .descriptors
        .iter()
        .filter(|d| d.op == SeqOp::MatVec as u32)
        .map(|d| d.base_a..d.base_a + d.n_tap)
        .collect();
    for g in gates {
        let h = g.base_d;
        assert!(
            reads_h.iter().any(|r| r.contains(&h)),
            "hidden state at {h} is never read back by a later projection"
        );
    }
}
