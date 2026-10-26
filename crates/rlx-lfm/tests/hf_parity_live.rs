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

//! Parity vs HF transformers for an HF safetensors LFM2.5 directory
//! (`LiquidAI/LFM2.5-230M`, `LiquidAI/LFM2.5-350M`).
//!
//! ```sh
//! python3 scripts/lm_hf_reference.py weights/lm/lfm2.5-230m /tmp/lfm25-230m.ref.json
//! RLX_LFM_HF_DIR=weights/lm/lfm2.5-230m RLX_LFM_HF_REF=/tmp/lfm25-230m.ref.json \
//!     cargo test -p rlx-lfm --release --test hf_parity_live -- --nocapture
//! ```

use rlx_lfm::Lfm2GgufRunner;
use rlx_runtime::Device;

/// `RLX_TEST_DEVICE=metal` (with the matching feature) runs on a GPU.
fn test_device() -> Device {
    rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .map(|d| rlx_cli::parse_device(&d).expect("RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

struct Reference {
    prompt: Vec<u32>,
    greedy: Vec<u32>,
    margins: Vec<f32>,
    last_logits: Vec<f32>,
}

fn reference() -> Option<(String, Reference)> {
    let dir = rlx_ir::env::var_result("RLX_LFM_HF_DIR").ok()?;
    let path = rlx_ir::env::var_result("RLX_LFM_HF_REF").ok()?;
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
    Some((
        dir,
        Reference {
            prompt: ids("prompt_ids"),
            greedy: ids("greedy"),
            margins: floats("margins"),
            last_logits: floats("last_logits"),
        },
    ))
}

#[test]
fn hf_dir_matches_transformers() {
    let Some((dir, r)) = reference() else {
        eprintln!("skip: set RLX_LFM_HF_DIR and RLX_LFM_HF_REF (scripts/lm_hf_reference.py)");
        return;
    };
    let runner = Lfm2GgufRunner::open(&dir, test_device()).expect("open HF dir");

    let logits = runner.predict_logits(&r.prompt).expect("logits");
    assert_eq!(logits.len(), r.last_logits.len(), "vocab size");
    let max_abs = logits
        .iter()
        .zip(&r.last_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = r.last_logits.iter().map(|x| x.abs()).fold(0f32, f32::max);
    eprintln!("last-position logits: max |rlx - hf| = {max_abs:.3e} (max |hf| = {scale:.2})");
    // Measured: legitimate f32 reduction-order noise 4e-5 (230M) / 6e-5
    // (350M); a RoPE θ off by 1% gives 1.8e-3. The bound sits between.
    assert!(max_abs < 5e-4, "logits mismatch: {max_abs}");

    // Incremental decode, compared against the transformers greedy run.
    let got = runner
        .generate(&r.prompt, r.greedy.len(), |_| true)
        .expect("generate");
    eprintln!("rlx greedy: {got:?}\nhf  greedy: {:?}", r.greedy);
    // Agreement up to the first divergence, and that divergence (if any) only
    // at a reference near-tie: transformers' own cached and uncached paths
    // disagree at a 0.009 margin on LFM2.5-230M.
    let first_diff = got.iter().zip(&r.greedy).position(|(a, b)| a != b);
    match first_diff {
        None => assert_eq!(got.len(), r.greedy.len(), "greedy length"),
        Some(i) => {
            let margin = r.margins[i];
            eprintln!("first divergence at step {i}: reference top-2 margin {margin:.4}");
            assert!(
                margin < 0.05,
                "greedy diverged at step {i} where the reference margin is {margin}"
            );
        }
    }
}
