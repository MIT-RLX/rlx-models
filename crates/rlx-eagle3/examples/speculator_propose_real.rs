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

//! Drive `Eagle3Speculator::propose` end-to-end through the HIR
//! runner against real RedHatAI/Gemma 4 31B draft weights. Compares
//! scalar fallback vs HIR (Metal preferred, then MLX, then CPU).
//!
//! Run:
//! ```bash
//! cargo run -p rlx-eagle3 --release --features "metal mlx" \
//!     --example speculator_propose_real -- \
//!     /Users/Shared/rlx-models/.eagle3-bench/weights/draft
//! ```

use anyhow::{Context, Result};
use rlx_eagle3::config::Eagle3Config;
use rlx_eagle3::draft::DraftGeom;
use rlx_eagle3::speculator::{Eagle3Speculator, VerifierHiddenSource};
use rlx_eagle3::weights::Eagle3DraftWeights;
use rlx_runtime::Device;
use rlx_runtime::spec_decode::Speculator;
use std::path::PathBuf;
use std::time::Instant;

/// Deterministic verifier hidden source — same shape and pattern
/// as `crate::examples::load_real_draft`.
struct SynthHidden {
    target_hidden: usize,
    layers: usize,
}
impl VerifierHiddenSource for SynthHidden {
    fn aux_hidden_states(&self) -> Result<Vec<Vec<f32>>> {
        Ok((0..self.layers)
            .map(|l| {
                (0..self.target_hidden)
                    .map(|d| ((d as f32) * 0.001 - (l as f32) * 0.0007).sin())
                    .collect()
            })
            .collect())
    }
    fn target_hidden_size(&self) -> usize {
        self.target_hidden
    }
    fn num_aux_layers(&self) -> usize {
        self.layers
    }
}

fn pick_device() -> Device {
    use rlx_runtime::is_available;
    if is_available(Device::Metal) {
        Device::Metal
    } else if is_available(Device::Mlx) {
        Device::Mlx
    } else {
        Device::Cpu
    }
}

fn main() -> Result<()> {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: speculator_propose_real <draft-dir>")?;

    let cfg = Eagle3Config::from_file(dir.join("config.json"))?;
    let geom = DraftGeom::from_cfg(&cfg);
    let weights_for_scalar = Eagle3DraftWeights::open(dir.join("model.safetensors"))?;
    let weights_for_hir = Eagle3DraftWeights::open(dir.join("model.safetensors"))?;
    let n = cfg.speculative_tokens;

    // Scalar path.
    println!("→ Scalar `propose(n={n})` on real RedHatAI weights");
    let h_scalar = SynthHidden {
        target_hidden: geom.h_target,
        layers: cfg
            .eagle_aux_hidden_state_layer_ids
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(3),
    };
    let mut scalar = Eagle3Speculator::new(cfg.clone(), weights_for_scalar, h_scalar)?;
    let ctx: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let t0 = Instant::now();
    let scalar_prop = scalar.propose(&ctx, n);
    println!(
        "   {:?} in {:.2}s ({:.2} tok/s)",
        scalar_prop.tokens,
        t0.elapsed().as_secs_f32(),
        n as f32 / t0.elapsed().as_secs_f32().max(1e-6),
    );

    // HIR path on the best available backend.
    let dev = pick_device();
    println!("\n→ HIR `propose(n={n})` on real weights · device = {dev:?}");
    let h_hir = SynthHidden {
        target_hidden: geom.h_target,
        layers: cfg
            .eagle_aux_hidden_state_layer_ids
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(3),
    };
    let mut hir = Eagle3Speculator::new(cfg, weights_for_hir, h_hir)?.with_hir_runner(dev, n)?;
    assert!(hir.uses_hir());

    // Warm a couple of rounds so compile + first-call cost don't dominate.
    let _ = hir.propose(&ctx, n);
    let _ = hir.propose(&ctx, n);

    let t0 = Instant::now();
    let iters = 10;
    let mut last = Vec::new();
    for _ in 0..iters {
        last = hir.propose(&ctx, n).tokens;
    }
    let secs = t0.elapsed().as_secs_f32();
    println!(
        "   {last:?} in {:.2}s avg ({:.2} tok/s · {iters} rounds)",
        secs / iters as f32,
        (iters * n) as f32 / secs.max(1e-6),
    );

    println!(
        "\n✓ Eagle3Speculator::propose now routes through the HIR draft graph\n  \
         when `with_hir_runner(device, n_max)` is attached."
    );
    Ok(())
}
