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

//! Cross-backend check of the **real** backbone, against CPU.
//!
//! The component suites pin every hand-written graph against upstream torch,
//! and `e2e_parity` pins the whole loop — but both run on fixtures a few
//! megabytes wide. Some backend defects only appear at checkpoint scale: wgpu
//! parks parameters in a second buffer once the arena crosses its 4 GiB
//! storage-binding cap, and that path is not exercised by anything small.
//!
//! So this runs one prefill of the actual 1 B backbone on CPU and on
//! `RLX_TADA_TEST_DEVICE`, and compares hidden states. It needs the real
//! checkpoint; set `RLX_TADA_WEIGHTS` to the model root, otherwise it skips.
//!
//! The extra assertion is the one that matters: a backend that silently reads
//! zero weights produces a hidden state that is *constant across positions*,
//! which looks finite and plausible downstream and only shows up as flat
//! prosody several stages later. Checking for that directly names the failure.

mod common;
use common::device;

use rlx_llama32::Llama32Config;
use rlx_runtime::Device;
use rlx_tada::backbone::Backbone;
use rlx_tada::config::TadaConfig;
use rlx_tada::model::find_backbone;
use rlx_tada::weights::TensorStore;
use std::sync::Arc;

/// Deterministic pseudo-embeddings — the point is cross-backend agreement on a
/// fixed input, not that the input means anything.
fn synthetic_embeds(batch: usize, seq: usize, hidden: usize) -> Vec<f32> {
    let mut state = 0x243f_6a88_85a3_08d3u64;
    (0..batch * seq * hidden)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

#[test]
fn backbone_prefill_matches_cpu() {
    let Some(root) = std::env::var_os("RLX_TADA_WEIGHTS") else {
        eprintln!("RLX_TADA_WEIGHTS not set — skipping real-weight backend check");
        return;
    };
    let dev = device();
    let root = std::path::PathBuf::from(root);
    let dir = find_backbone(&root).expect("locate backbone");
    let cfg = TadaConfig::from_file(&dir.join("config.json")).expect("TadaConfig");
    let llama: Llama32Config =
        Llama32Config::from_file(&dir.join("config.json")).expect("Llama32Config");
    let store =
        Arc::new(TensorStore::open(&dir.join("model.safetensors")).expect("open checkpoint"));

    let (batch, seq, hidden) = (2usize, 32usize, cfg.hidden_size);
    let embeds = synthetic_embeds(batch, seq, hidden);

    let run = |d: Device| {
        let mut bb =
            Backbone::new(store.clone(), llama.clone(), d, batch, "matrix_test").expect("backbone");
        let out = bb.prefill(&embeds, seq).expect("prefill");
        let stride = out.row_stride();
        // Last real position of batch row 0.
        (out.hidden()[..stride * hidden].to_vec(), stride)
    };

    let (cpu, stride) = run(Device::Cpu);
    if dev == Device::Cpu {
        // Still worth asserting the baseline is not itself degenerate.
        assert!(
            position_spread(&cpu, stride, hidden) > 1e-3,
            "CPU hidden is constant across positions"
        );
        return;
    }
    let (got, got_stride) = run(dev);
    assert_eq!(stride, got_stride, "row stride differs");

    // A backend reading zero weights gives the same vector at every position.
    let spread = position_spread(&got, stride, hidden);
    assert!(
        spread > 1e-3,
        "{dev:?}: hidden state is constant across positions (spread {spread:.2e}) — \
         the backbone is ignoring its input, which is what silently-zero weights \
         look like"
    );

    let mut worst = 0f32;
    let mut at = 0usize;
    for (i, (a, b)) in got.iter().zip(&cpu).enumerate() {
        let d = (a - b).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst < 5e-3,
        "{dev:?}: prefill hidden deviates from CPU by {worst} at index {at} \
         (position {}, channel {})",
        at / hidden,
        at % hidden
    );
}

/// Largest per-channel spread across positions — near zero means the graph
/// produced the same vector everywhere.
fn position_spread(h: &[f32], positions: usize, hidden: usize) -> f32 {
    let mut worst = 0f32;
    for c in 0..hidden.min(64) {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for p in 0..positions {
            let v = h[p * hidden + c];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        worst = worst.max(hi - lo);
    }
    worst
}
