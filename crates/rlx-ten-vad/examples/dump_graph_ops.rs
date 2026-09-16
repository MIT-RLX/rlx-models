// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Print the rlx-ir op sequence for each graph shape — the input the FPGA
//! lowering has to handle.

use rlx_ten_vad::model::{Shape, build_graph};
use rlx_ten_vad::weights::TenVadWeights;

fn main() -> anyhow::Result<()> {
    let w = TenVadWeights::embedded();
    for shape in [Shape::Streaming, Shape::Chunk(1)] {
        let (g, params) = build_graph(shape, w)?;
        println!(
            "=== {shape:?} — {} nodes, {} params ===",
            g.nodes().len(),
            params.len()
        );
        for (i, n) in g.nodes().iter().enumerate() {
            let op = format!("{:?}", n.op);
            let op = op
                .split_once(&['{', '('][..])
                .map_or(op.clone(), |(h, _)| h.trim().to_string());
            println!("  {i:3} {op:22} in={:?} shape={:?}", n.inputs, n.shape);
        }
        println!();
    }
    Ok(())
}
