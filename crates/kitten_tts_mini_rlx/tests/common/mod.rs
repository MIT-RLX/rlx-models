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

//! Shared helpers for Kitten bundle HIR integration tests.

#![allow(dead_code)]

use rlx_ir::hir::HirModule;
use rlx_onnx_import::tensor_data::TypedParams;
use rlx_onnx_import::{ImportOptions, ImportReport, RlxBundle, load_bundle as load_rlx_bundle};

type HirBuildResult = (
    HirModule,
    std::collections::HashMap<String, Vec<f32>>,
    TypedParams,
    ImportReport,
);

pub fn bundle_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle")
}

pub fn load_bundle() -> Option<RlxBundle> {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        eprintln!("skip: {}", dir.display());
        return None;
    }
    Some(load_rlx_bundle(&dir).expect("load bundle"))
}

pub fn opts_seq8() -> ImportOptions {
    ImportOptions {
        sequence_length: 8,
        max_waveform_samples: 24_000,
        ..ImportOptions::quant_bundle()
    }
}

/// Lower bundle → HIR with model-specific graph rewrites (duration carry).
pub fn build_hir(bundle: &RlxBundle, mut opts: ImportOptions) -> anyhow::Result<HirBuildResult> {
    kitten_tts_mini_rlx::bundle_patches::set_import_sequence_length(opts.sequence_length);
    opts.output_shape_fix = Some(kitten_tts_mini_rlx::bundle_patches::import_output_shape_fix);
    kitten_tts_mini_rlx::bundle_compile::build_hir_from_bundle_with_rewrites(bundle, opts)
}
