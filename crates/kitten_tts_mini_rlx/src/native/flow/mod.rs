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

mod duration;
pub mod modules;
mod rng;
pub mod vocoder;

pub use modules::MODULE_INDEX;

use std::collections::HashMap;

use anyhow::Result;
use rlx_ir::hir::HirModule;

use crate::opts::GraphOptions;
use crate::weights::LoadedWeights;

/// Build the full native Kitten graph from decomposed weights (no `graph.json`).
pub fn build_native_hir(
    weights: &LoadedWeights,
    opts: &GraphOptions,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, Vec<u8>)> {
    let graph_opts = crate::graph::GraphOptions {
        sequence_length: opts.sequence_length,
        max_waveform_samples: opts.max_waveform_samples,
    };
    let (mut hir, mut params) = crate::graph::build_hir(weights, &graph_opts)?;
    crate::bundle_patches::apply_hir_patches(
        &mut hir,
        opts.sequence_length,
        opts.max_waveform_samples,
    );
    crate::bundle_patches::inject_vocoder_dynamic_alignment(
        &mut hir,
        opts.sequence_length,
        opts.max_waveform_samples,
    );
    let carry_bytes = duration::inject_duration_carry(&mut hir, opts.sequence_length);
    rng::inject_vocoder_rng(&mut hir);
    params.remove(rng::VOCODER_RNG_STUB);
    Ok((hir, params, carry_bytes))
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
