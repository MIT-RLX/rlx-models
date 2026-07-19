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

//! Native HIR builders for Kitten TTS mini (weights-only path).

pub mod modules;
mod rng;
pub mod vocoder;

pub use modules::MODULE_INDEX;

use std::collections::HashMap;

use anyhow::Result;
use rlx_ir::hir::HirModule;

use crate::opts::GraphOptions;
use crate::weights::LoadedWeights;

/// Weights-only native build (no `rlx_bundle/graph.json`) is no longer supported:
/// the transpiled `graph.rs` HIR builder was removed in favour of the data-driven
/// bundle path. `compile()` prefers the bundle whenever it is present; this stub
/// only fires in the (unshipped) bundle-absent fallback.
pub fn build_native_hir(
    _weights: &LoadedWeights,
    _opts: &GraphOptions,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, Vec<u8>)> {
    anyhow::bail!(
        "no rlx_bundle/graph.json found: the weights-only transpiled path was removed; \
         ship the rlx_bundle (graph.json + weights) for the native data-driven path"
    )
}

/// Post-import HIR fixes for bundle compile (RNG + F0 bypass; duration carry is already
/// wired via [`crate::bundle_patches::patch_bundle_nodes`] string rewrites).
pub fn finish_bundle_hir_for_compile(
    hir: &mut HirModule,
    params: &mut HashMap<String, Vec<f32>>,
    sequence_length: usize,
    max_waveform_samples: usize,
) {
    let _ = crate::bundle_patches::inject_vocoder_dynamic_alignment(
        hir,
        sequence_length,
        max_waveform_samples,
    );
    rng::inject_vocoder_rng(hir);
    params.remove(rng::VOCODER_RNG_STUB);
}
