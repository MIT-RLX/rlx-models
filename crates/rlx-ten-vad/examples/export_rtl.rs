// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Export the FPGA datapath straight from the rlx-ir graph.
//!
//! Nothing here is TEN-VAD-specific: `rlx-fpga`'s sequential target lowers the
//! same graph the runtime executes, so a retrained network re-exports with no
//! hand-edited Verilog.
//!
//! ```text
//! cargo run -p rlx-ten-vad --example export_rtl
//! ```

use rlx_fpga::seq::{SeqConfig, SeqOp, export_graph_seq};
use rlx_ten_vad::model::{Shape, build_graph};
use rlx_ten_vad::weights::TenVadWeights;

fn main() -> anyhow::Result<()> {
    let w = TenVadWeights::embedded();
    let (graph, params) = build_graph(Shape::Streaming, w)?;

    // The streaming graph exposes LSTM state as ordinary inputs and outputs;
    // this is the pairing that lets the engine update them in place.
    let cfg = SeqConfig::default().with_carry([("h1", 1), ("c1", 2), ("h2", 3), ("c2", 4)]);

    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../rlx-ten-vad-fpga/rtl");
    let model = export_graph_seq(&graph, &params, &cfg, &out).map_err(anyhow::Error::msg)?;

    println!("exported to {}", out.display());
    println!("  descriptors {}", model.descriptors.len());
    println!(
        "  weights     {} int16 ({:.1} kB)",
        model.weights.len(),
        model.weights.len() as f64 * 2.0 / 1024.0
    );
    println!(
        "  activation  {} words used of {} budget -> RTL declares {}",
        model.aram_used,
        model.aram_words,
        model.aram_used.max(1).next_power_of_two()
    );
    println!("  MACs/frame  {}", model.macs());
    println!(
        "  features    [{}..{})",
        model.feat_base,
        model.feat_base + model.feat_len
    );
    println!("  probability @ {}", model.prob_addr);
    println!();
    for (i, (d, label)) in model.descriptors.iter().zip(&model.labels).enumerate() {
        let op = match d.op {
            x if x == SeqOp::MatVec as u32 => "matvec",
            x if x == SeqOp::Pool as u32 => "pool",
            x if x == SeqOp::Gate as u32 => "gate",
            _ => "done",
        };
        println!(
            "  {i:2} {op:7} {label:12} n_o={:3} n_i={:4} taps={:4} -> @{}",
            d.n_o, d.n_i, d.n_tap, d.base_d
        );
    }
    Ok(())
}
