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

//! Env-gated: drain vision / speech GGUF checkpoints to [`rlx_core::WeightMap`].
//!
//! ```text
//! SAM3_GGUF_PATH=/path/sam3.gguf \
//! DINOV2_GGUF_PATH=/path/dinov2.gguf \
//! cargo test -p rlx-models --test vision_gguf_load --release -- --nocapture
//! ```
//!
//! Compile + forward: see [`vision_gguf_compile`].

#[path = "vision_gguf/support.rs"]
mod support;

use rlx_core::{
    DINOV2_GGUF_ARCHES, FLUX_GGUF_ARCHES, SAM3_GGUF_ARCHES, W2V_BERT_GGUF_ARCHES, load_weight_map,
};

use support::env_gguf_path;

fn drain_quick_check(var: &str, arches: &[&str]) {
    let Some(path) = env_gguf_path(var) else {
        eprintln!("skip: set {var} to a .gguf file");
        return;
    };
    let arch = rlx_core::gguf_architecture_from_path(&path).expect("read gguf arch");
    let wm = load_weight_map(&path, arches)
        .unwrap_or_else(|e| panic!("{var} {path:?} (arch={arch}): load_weight_map failed: {e:#}"));
    assert!(
        wm.len() > 8,
        "{var}: expected many tensors after drain, got {}",
        wm.len()
    );
    let sample: Vec<_> = wm.keys().take(4).collect();
    eprintln!(
        "{var}: arch={arch} tensors={} sample_keys={sample:?}",
        wm.len()
    );
}

#[test]
fn sam3_gguf_drain() {
    drain_quick_check("SAM3_GGUF_PATH", SAM3_GGUF_ARCHES);
}

#[test]
fn dinov2_gguf_drain() {
    drain_quick_check("DINOV2_GGUF_PATH", DINOV2_GGUF_ARCHES);
}

#[test]
fn flux_gguf_drain() {
    drain_quick_check("FLUX_GGUF_PATH", FLUX_GGUF_ARCHES);
}

#[test]
fn w2v_bert_gguf_drain() {
    drain_quick_check("W2V_BERT_GGUF_PATH", W2V_BERT_GGUF_ARCHES);
}
