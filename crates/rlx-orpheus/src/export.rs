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

//! Export Orpheus — or a model of its family (Maya1: the same tokens, its
//! own prompt body) — as an `.rlxp` package rlx runs alone: text in, 24 kHz
//! speech out (`Model::synthesize`, the voice `options.voice`), through the
//! one exporter ([`rlx_core::export`]).
//!
//! The language model is the GGUF Llama's package
//! ([`rlx_llama32::export::export_gguf_with`]: its `generate` entry); SNAC's
//! lookup and decoder ([`rlx_snac::export::snac_functions`]) are graphs
//! beside it; the program (`program/`) builds the prompt, runs the language
//! model a token at a time (its logits penalized and sampled as the runner
//! does), takes the seven codes of each frame out of its ids and SNAC over
//! them.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use rlx_core::export::program;
use serde_json::{Value, json};

pub use rlx_core::export::{ExportOptions, ExportReport};

use crate::GenerationConfig;
use crate::download::SNAC_DECODER_SAFETENSORS;
use crate::tokens::{
    BOS_TOKEN_ID, CUSTOM_TOKEN_BASE, PROMPT_END_IDS, PROMPT_START_ID, SNAC_TOKEN_OFFSET,
    STOP_TOKEN_ID, VOICES,
};

/// SNAC's length symbol in its graphs.
const SNAC_FRAMES: u32 = 0;
/// SNAC 24 kHz's sample rate.
const SAMPLE_RATE: u32 = 24_000;

/// What sets one model of the family apart: its name, its prompt body, its
/// sampling.
pub struct Flavor {
    pub family: &'static str,
    /// The prompt body with a voice (`{voice}`, `{text}`).
    pub template: &'static str,
    /// The voice when a request names none; `None`: the text alone.
    pub voice: Option<&'static str>,
    pub voices: &'static [&'static str],
    pub config: GenerationConfig,
}

/// Orpheus: `{voice}: {text}`, one of its eight voices.
pub fn orpheus() -> Flavor {
    Flavor {
        family: "orpheus",
        template: "{voice}: {text}",
        voice: None,
        voices: VOICES,
        config: GenerationConfig::default(),
    }
}

/// SNAC's decoder: `ORPHEUS_SNAC_PATH`, else `snac/` (or `snac-24khz/`)
/// in the GGUF's directory or beside it.
fn snac_path(gguf: &Path) -> Result<PathBuf> {
    if let Some(p) = crate::decoder::decoder_weights_path_if_available() {
        return Ok(p);
    }
    let dir = gguf.parent().unwrap_or(Path::new("."));
    let roots = std::iter::once(dir).chain(dir.parent());
    for root in roots {
        for sub in ["snac", "snac-24khz"] {
            let p = root.join(sub).join(SNAC_DECODER_SAFETENSORS);
            if p.is_file() {
                return Ok(p);
            }
        }
    }
    bail!(
        "Orpheus decodes with SNAC: no {SNAC_DECODER_SAFETENSORS} in snac/ beside {} (or set ORPHEUS_SNAC_PATH)",
        gguf.display()
    )
}

/// Export the Orpheus GGUF `gguf` (SNAC's decoder beside it) as the
/// package `out`.
pub fn export_rlxp(gguf: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    export_flavor(gguf, out, opts, &orpheus())
}

/// [`export_rlxp`] for the family model `flavor` describes.
pub fn export_flavor(
    gguf: &Path,
    out: &Path,
    opts: &ExportOptions,
    flavor: &Flavor,
) -> Result<ExportReport> {
    let snac = snac_path(gguf)?;
    let c = &flavor.config;
    rlx_llama32::export::export_gguf_with(gguf, out, opts, &mut |pkg| {
        pkg.model["family"] = json!(flavor.family);
        pkg.model["task"] = json!("text-to-speech");
        let (graphs, params, _) = rlx_snac::export::snac_functions(&snac, SNAC_FRAMES)?;
        for g in graphs {
            pkg.function(g);
        }
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
        let module =
            program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))?;
        program::entry(
            pkg,
            "speak",
            module,
            &["synthesize", "run"],
            "text-to-speech",
            json!({
                "template": flavor.template,
                "voice": flavor.voice.map_or(Value::Null, |v| json!(v)),
                "start": PROMPT_START_ID,
                "bos": BOS_TOKEN_ID,
                "end": PROMPT_END_IDS,
                "stop": STOP_TOKEN_ID,
                "custom_base": CUSTOM_TOKEN_BASE,
                "snac_offset": SNAC_TOKEN_OFFSET,
                "max_new_tokens": c.max_new_tokens,
                "temperature": c.temperature,
                "top_p": c.top_p,
                "top_k": c.top_k,
                "repetition_penalty": c.repetition_penalty,
                "seed": c.seed,
                "greedy": c.greedy,
                "sample_rate": SAMPLE_RATE,
            }),
        );
        pkg.metadata("sample_rate", json!(SAMPLE_RATE));
        pkg.metadata("voices", json!(flavor.voices));
        Ok(())
    })
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("orpheus", "text-to-speech", export_rlxp));
}

/// Register `export` as the text-to-speech exporter `name`: a model of
/// the family in a crate of its own (Maya1).
pub fn register_named(
    name: &'static str,
    export: fn(&Path, &Path, &ExportOptions) -> Result<ExportReport>,
) {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new(name, "text-to-speech", export));
}
