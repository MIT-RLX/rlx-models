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

//! Export MioTTS as an `.rlxp` package rlx runs alone — text in, 24 kHz
//! speech out (`Model::synthesize`, a preset voice `options.voice`) —
//! through the one exporter ([`rlx_core::export`]).
//!
//! The language model is Qwen3's package ([`rlx_qwen3::export::export_rlxp_with`]:
//! its `generate` entry); MioCodec's `decoder_body` (the checkpoint's ONNX,
//! imported once, here, at the codec's fixed length) and its ISTFT
//! ([`istft_same`]) are graphs beside it; the program
//! (`program/`) runs the language model (`entry.run`), takes the speech
//! codes out of its ids and the codec over them.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rlx_core::export::audio::{istft_bases, istft_same};
use rlx_core::export::{ExportOptions, ExportReport, program};
use serde_json::json;

use crate::codec::{GLOBAL_DIM, HOP, N_FFT, SAMPLE_RATE};
use crate::tokens::{EOS, SPEECH_BASE, SPEECH_CODEBOOK, SPEECH_LEN};

/// MioCodec beside the model in `dir`: its `miocodec/`, else a sibling's.
fn codec_dir(dir: &Path) -> Result<PathBuf> {
    let beside = dir.parent().map(|p| p.join("miocodec"));
    for p in std::iter::once(dir.join("miocodec")).chain(beside) {
        if p.join("decoder_body.onnx").is_file() {
            return Ok(p);
        }
    }
    bail!(
        "MioTTS decodes with MioCodec: none at {}/miocodec or beside it",
        dir.display()
    )
}

/// Export the MioTTS checkpoint in `dir` (Qwen3's: `config.json`,
/// `model.safetensors`, `tokenizer.json`; `presets/*.f32`; MioCodec beside
/// it) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let codec = codec_dir(dir)?;
    let presets = dir.join("presets");
    let defaults = crate::session::GenerateOpts::default();
    let window: Vec<f32> = (0..N_FFT)
        .map(|k| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * k as f32 / N_FFT as f32).cos()))
        .collect();
    rlx_qwen3::export::export_rlxp_with(dir, out, opts, &mut |pkg| {
        pkg.model["family"] = json!("miotts");
        pkg.model["task"] = json!("text-to-speech");
        let (body, mut params) = rlx_tiny_tts::model::import_rlx_graph(
            &codec.join("decoder_body.onnx"),
            "decoder_body",
            SPEECH_LEN,
            &[],
        )?;
        let frames = body
            .node(body.outputs[0])
            .shape
            .dims()
            .last()
            .map(|d| d.unwrap_static())
            .context("decoder_body's frames")?;
        pkg.function(body);
        pkg.function(istft_same(N_FFT, HOP, &window, frames)?);
        params.extend(istft_bases(N_FFT, &window));
        let dims = pkg.param_dims();
        let mut names: Vec<&String> = params.keys().collect();
        names.sort();
        for n in names {
            let d = dims
                .get(n)
                .cloned()
                .unwrap_or_else(|| vec![params[n].len()]);
            pkg.hold_f32(n.clone(), d, &params[n]);
        }
        let mut voices = Vec::new();
        for e in std::fs::read_dir(&presets)? {
            let p = e?.path();
            if p.extension().and_then(|x| x.to_str()) != Some("f32") {
                continue;
            }
            let name = p
                .file_stem()
                .and_then(|s| s.to_str())
                .context("preset name")?;
            let bytes = std::fs::read(&p)?;
            if bytes.len() != 4 * GLOBAL_DIM {
                continue;
            }
            pkg.sidecar(
                &format!("presets/{name}.f32"),
                "application/octet-stream",
                bytes,
            );
            voices.push(name.to_string());
        }
        voices.sort();
        let module =
            program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))?;
        program::entry(
            pkg,
            "speak",
            module,
            &["synthesize"],
            "text-to-speech",
            json!({
                "speech_base": SPEECH_BASE,
                "codebook": SPEECH_CODEBOOK,
                "eos": EOS,
                "speech_len": SPEECH_LEN,
                "max_new_tokens": defaults.max_new_tokens,
                "temperature": 0.8,
                "seed": defaults.seed,
                "preset": defaults.preset,
                "sample_rate": SAMPLE_RATE,
            }),
        );
        pkg.metadata("sample_rate", json!(SAMPLE_RATE));
        pkg.metadata("voices", json!(voices));
        Ok(())
    })
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("miotts", "text-to-speech", export_rlxp).claims(&["miotts"]));
}
