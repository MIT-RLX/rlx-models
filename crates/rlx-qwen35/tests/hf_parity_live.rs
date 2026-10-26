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

//! Parity vs HF transformers for an HF safetensors Qwen3.5 directory
//! (`Qwen/Qwen3.5-0.8B`), loaded with no config source — the builder detects
//! the directory, takes the trunk depth from the tensor names and converts
//! `A_log`.
//!
//! ```sh
//! python3 scripts/lm_hf_reference.py weights/lm/qwen3.5-0.8b /tmp/qwen35-08b.ref.json
//! RLX_QWEN35_HF_DIR=$PWD/weights/lm/qwen3.5-0.8b RLX_QWEN35_HF_REF=/tmp/qwen35-08b.ref.json \
//!     cargo test -p rlx-qwen35 --release --test hf_parity_live -- --nocapture
//! ```

use rlx_qwen35::Qwen35Runner;
use rlx_runtime::Device;

/// `RLX_TEST_DEVICE=metal` (with the matching feature) runs on a GPU.
fn test_device() -> Device {
    rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .map(|d| rlx_cli::parse_device(&d).expect("RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

#[test]
fn hf_dir_matches_transformers() {
    let (Ok(dir), Ok(path)) = (
        rlx_ir::env::var_result("RLX_QWEN35_HF_DIR"),
        rlx_ir::env::var_result("RLX_QWEN35_HF_REF"),
    ) else {
        eprintln!("skip: set RLX_QWEN35_HF_DIR and RLX_QWEN35_HF_REF (scripts/lm_hf_reference.py)");
        return;
    };
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read reference")).unwrap();
    let ids = |k: &str| -> Vec<u32> {
        v[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };
    let floats = |k: &str| -> Vec<f32> {
        v[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    };
    let (prompt, greedy, margins, hf_logits) = (
        ids("prompt_ids"),
        ids("greedy"),
        floats("margins"),
        floats("last_logits"),
    );

    let mut runner = Qwen35Runner::builder()
        .weights(&dir)
        .device(test_device())
        .max_seq(prompt.len() + greedy.len())
        .skip_auto_mmproj(true)
        .build()
        .expect("build from HF dir");

    let out = runner.predict_logits(&prompt).expect("logits");
    let vocab = out.vocab_size;
    let last = &out.logits[out.logits.len() - vocab..];
    assert_eq!(vocab, hf_logits.len(), "vocab size");
    let max_abs = last
        .iter()
        .zip(&hf_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("last-position logits: max |rlx - hf| = {max_abs:.3e}");
    // Measured 3.4e-4 (f32, different GDN/attention reduction order).
    assert!(max_abs < 1e-3, "logits mismatch: {max_abs}");

    let got = runner
        .generate(&prompt, greedy.len(), |_| true)
        .expect("generate");
    eprintln!("rlx greedy: {got:?}\nhf  greedy: {greedy:?}");
    // Agree up to the first divergence, and diverge (if at all) only where the
    // reference itself is a near-tie.
    if let Some(i) = got.iter().zip(&greedy).position(|(a, b)| a != b) {
        eprintln!(
            "first divergence at step {i}: reference top-2 margin {:.4}",
            margins[i]
        );
        assert!(
            margins[i] < 0.05,
            "greedy diverged at step {i} where the reference margin is {}",
            margins[i]
        );
    }
}
