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

//! The published checkpoint against `scripts/flint_reference.py`.
//!
//! Needs `RLX_FLINT_DIR` (the HF checkpoint) and `RLX_FLINT_REF` (a reference
//! dump over it); skipped otherwise. `just test-flint-parity` produces both.
//! Optional `RLX_FLINT_DEVICE` (default `cpu`).
//!
//! Every position of the prompt is compared, token by token through the decode
//! graphs and again through chunked prefill — the paths a generation takes.

use rlx_flint::{FlintRunner, RunnerOptions};

fn reference(path: &str) -> (Vec<u32>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let ids = st
        .tensor("ids")
        .unwrap()
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c) as u32)
        .collect();
    let logits = st
        .tensor("logits")
        .unwrap()
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    (ids, logits)
}

#[test]
fn every_position_matches_the_reference() {
    let (Some(dir), Some(refp)) = (
        rlx_ir::env::var("RLX_FLINT_DIR"),
        rlx_ir::env::var("RLX_FLINT_REF"),
    ) else {
        eprintln!("RLX_FLINT_DIR / RLX_FLINT_REF unset — skipping");
        return;
    };
    let device =
        rlx_cli::parse_device(&rlx_ir::env::var("RLX_FLINT_DEVICE").unwrap_or("cpu".into()))
            .unwrap();
    let (ids, want) = reference(&refp);
    // One runner per iteration, dropped before the next opens, so only one
    // copy of the weights is resident at a time.
    for chunk in [1, 8] {
        let opts = RunnerOptions::new(device)
            .max_context(ids.len().next_power_of_two().max(64))
            .prefill_chunk(chunk);
        let mut r = FlintRunner::open(&dir, opts).unwrap();
        let got = r.all_logits(&ids).unwrap();
        let v = r.config().vocab_size;
        let rows = want.len() / v;
        let off = (ids.len() - rows) * v;
        for row in 0..rows {
            let a = &got[off + row * v..off + (row + 1) * v];
            let b = &want[row * v..(row + 1) * v];
            let d = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max);
            assert!(d < 2e-3, "chunk {chunk}, row {row}: max |Δ| {d}");
            assert_eq!(
                rlx_runtime::search::argmax_u32(a),
                rlx_runtime::search::argmax_u32(b),
                "chunk {chunk}, row {row}: top-1 differs"
            );
        }
    }
}
