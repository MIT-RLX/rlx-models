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

#![cfg(feature = "export")]

//! Where the prompt runs: an exported Qwen3 package on `RLX_TEST_DEVICE`
//! with its prefill there, and with `prefill_device: "cpu"` (the K/V rows
//! then copied to the device's caches). Opt-in: `RLX_BENCH_PREFILL=1` and
//! `RLX_TEST_QWEN3_DIR`.
//!
//! Each prompt length runs once to compile, then best of five: the first
//! token alone (the prompt), and two (the prompt, the caches handed to the
//! step, one step). Both placements must pick the same tokens.

use rlx_qwen3::export::{ExportOptions, export_rlxp};
use rlx_runtime::model::{Tensor, ids_of};
use serde_json::json;

#[test]
fn prefill_on_the_device_or_the_cpu() {
    let (Some(dir), Some(_)) = (
        std::env::var_os("RLX_TEST_QWEN3_DIR"),
        std::env::var_os("RLX_BENCH_PREFILL"),
    ) else {
        eprintln!("skip: set RLX_BENCH_PREFILL=1 and RLX_TEST_QWEN3_DIR");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let device = std::env::var("RLX_TEST_DEVICE").unwrap_or_else(|_| "metal".into());
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("qwen3.rlxp");
    let opts = ExportOptions {
        uppers: vec![512],
        ..ExportOptions::default()
    };
    export_rlxp(&dir, &out, &opts).unwrap();
    rlx_text::install_model_tokenizer();

    let text = "The history of the city goes back many centuries, and its streets \
                still carry the names of the merchants, scholars and builders who \
                shaped it. "
        .repeat(20);
    let all: Vec<u32> = rlx_text::load_tokenizer(&dir.join("tokenizer.json"))
        .unwrap()
        .encode(&text, false)
        .unwrap();
    let lengths = [16usize, 64, 250];
    assert!(all.len() >= 262, "prompt text too short: {}", all.len());

    let mut rows = Vec::new();
    let mut tokens: Vec<Vec<Vec<u32>>> = Vec::new();
    for prefill in [None, Some("cpu")] {
        let mut options = json!({"device": device});
        if let Some(p) = prefill {
            options["prefill_device"] = json!(p);
        }
        let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), options).unwrap();
        let mut picked = Vec::new();
        for &n in &lengths {
            // A different prompt each run (the text from a later token):
            // the same prompt again is a continuation of the cache, and
            // its first token comes back without running anything.
            let mut run = |k: usize, start: usize| {
                let prompt: Vec<i32> = all[start..start + n].iter().map(|&i| i as i32).collect();
                let input = Tensor::new(
                    "input_ids",
                    rlx_ir::DType::I32,
                    vec![prompt.len()],
                    prompt.iter().flat_map(|i| i.to_le_bytes()).collect(),
                );
                let t = std::time::Instant::now();
                let ids = ids_of(&m.run(&[input], &json!({"max_tokens": k})).unwrap()[0]).unwrap();
                (t.elapsed().as_secs_f64() * 1e3, ids)
            };
            run(2, 0); // compile the prompt's size and the step
            picked.push(run(2, 1).1);
            let t1 = (2..7).map(|s| run(1, s).0).fold(f64::MAX, f64::min);
            let t2 = (7..12).map(|s| run(2, s).0).fold(f64::MAX, f64::min);
            rows.push((prefill.unwrap_or(device.as_str()).to_string(), n, t1, t2));
        }
        tokens.push(picked);
    }
    eprintln!("device {device}: prompt run on | tokens | first token ms | two tokens ms");
    for (on, n, t1, t2) in &rows {
        eprintln!("  {on:>6} | {n:>4} | {t1:>8.1} | {t2:>8.1}");
    }
    assert_eq!(
        tokens[0], tokens[1],
        "the prompt's placement changed the tokens"
    );
}
