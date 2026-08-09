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

//! The runner caches the compiled decode graph, so a second `generate` on the
//! same runner skips graph build + weight re-attach (+ the process-global CPU
//! dequant warmup) and is much faster — with identical output. Gated on
//! `RLX_LFM_WEIGHTS`.

use rlx_lfm::{Lfm2GgufRunner, resolve_gguf};
use rlx_runtime::Device;
use std::path::PathBuf;

#[test]
fn warm_generate_reuses_compiled_graph() {
    let Some(w) = std::env::var("RLX_LFM_WEIGHTS").ok().map(PathBuf::from) else {
        eprintln!("skip: set RLX_LFM_WEIGHTS=<LFM2 .gguf>");
        return;
    };
    let gguf = resolve_gguf(&w).expect("resolve gguf");
    let runner = Lfm2GgufRunner::open(&gguf, Device::Cpu).expect("open runner");
    let prompt: Vec<u32> = vec![100, 2000, 500, 42];
    let n = 16;

    let t0 = std::time::Instant::now();
    let cold = runner.generate(&prompt, n, |_| true).expect("cold");
    let cold_dt = t0.elapsed();

    let t1 = std::time::Instant::now();
    let warm = runner.generate(&prompt, n, |_| true).expect("warm");
    let warm_dt = t1.elapsed();

    assert_eq!(cold, warm, "cached graph changed the output");
    eprintln!(
        "cold {:.2?} → warm {:.2?}  ({:.1}× faster; graph + weights reused)",
        cold_dt,
        warm_dt,
        cold_dt.as_secs_f64() / warm_dt.as_secs_f64().max(1e-9)
    );
    assert!(
        warm_dt < cold_dt,
        "warm run ({warm_dt:?}) should beat cold ({cold_dt:?})"
    );
}
