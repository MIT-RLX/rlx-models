// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Export a Parakeet-TDT `.nemo` as an `.rlxp` package rlx runs alone:
//! `Model.transcribe(audio)`, through `rlx-nemotron-asr`'s FastConformer
//! transducer exporter ([`rlx_nemotron_asr::export::transducer_rlxp`]) with
//! this model's search: the joint's duration head moves the frame by
//! `durations[argmax]`, and the start-of-sequence step's state is kept, as
//! [`crate::Parakeet`] decodes.

use std::path::Path;

use anyhow::{Context, Result};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_nemo::NemoModel;
use rlx_nemotron_asr::export::{Transducer, transducer_rlxp};

/// Export the Parakeet-TDT `.nemo` at `path` (beside it, the Hugging Face
/// `model.safetensors` of the same weights, which the package references)
/// as the package `out`.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let model = NemoModel::open(path).with_context(|| format!("open {}", path.display()))?;
    let durations = crate::runner::tdt_durations(&model);
    drop(model);
    transducer_rlxp(
        path,
        out,
        opts,
        &Transducer {
            name: "parakeet-tdt",
            exported_by: "rlx-parakeet",
            license: "cc-by-4.0",
            durations,
            sos_carry: true,
        },
    )
}

/// Register this crate's exporter, taken by name (`--family parakeet-tdt`):
/// a `.nemo` names no `model_type`.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new(
        "parakeet-tdt",
        "automatic-speech-recognition",
        export_rlxp,
    ));
}
