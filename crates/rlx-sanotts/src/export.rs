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

//! Export a sanoTTS voice package (`manifest.json`, `weights.fp16.bin`,
//! `piper-phoneme-config.json`) as an `.rlxp` package rlx runs alone: text
//! in, the voice's speech out.
//!
//! Its graphs are the crate's ([`crate::graph`]): `duration` (the duration
//! student's log durations) and `tokens` (the acoustic student's token
//! stack), symbolic in the tokens; `frames` (the acoustic frame stack and
//! the whole `piperlite` decoder), symbolic in the frames. The program
//! (`program/`, wasm32-wasip1) phonemizes with espeak-ng's English data
//! (`text_to_phonemes_phonemizer`, as [`crate::frontend`]), maps codepoints
//! through the voice's table with sanoTTS's framing, rounds the durations
//! half to even, expands the tokens to frames with their positional rows
//! ([`crate::model::acoustic_frame_input`]) and clamps the waveform.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_core::export::graph::symbolic_in_lifted_many;
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_core::flow_util::{built_from_hir, graph_from_built};
use rlx_ir::Graph;
use serde_json::json;

use crate::config::{AcousticConfig, DecoderConfig, DurationConfig};
use crate::graph::{TokenStage, frame_graph, token_graph};
use crate::voicepack::VoicePack;

/// The symbols: the tokens, the frames.
const TOKENS: u32 = 0;
const FRAMES: u32 = 1;

type Built = (Graph, HashMap<String, Vec<f32>>);

fn lowered(hir: (rlx_ir::HirModule, HashMap<String, Vec<f32>>), name: &str) -> Result<Built> {
    let (mut g, params) = graph_from_built(built_from_hir(hir.0, hir.1)?)?;
    g.name = name.into();
    Ok((g, params))
}

fn files(dir: &Path, base: &Path, out: &mut Vec<(String, PathBuf)>) -> Result<()> {
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir() {
            files(&p, base, out)?;
        } else {
            let rel = p.strip_prefix(base)?.to_string_lossy().replace('\\', "/");
            out.push((rel, p));
        }
    }
    Ok(())
}

/// Export the sanoTTS voice package in `dir` as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let pack = VoicePack::load_from_dir(dir)?;
    let du_cfg = pack.component_config::<DurationConfig>("duration")?;
    let ac_cfg = pack.component_config::<AcousticConfig>("acoustic")?;
    let de_cfg = pack.component_config::<DecoderConfig>("decoder")?;
    let du = pack.component_tensors("duration")?;
    let ac = pack.component_tensors("acoustic")?;
    let de = pack.component_tensors("decoder")?;
    let table: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("piper-phoneme-config.json"))
            .context("piper-phoneme-config.json")?,
    )?;
    let espeak_voice = pack
        .phoneme_table()
        .map(|t| t.espeak_voice.clone())
        .unwrap_or_else(|_| "en-us".into());
    let name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("sanotts");
    let mut pkg = Package::new(
        &format!("sanotts-{name}"),
        &opts.version,
        "sanotts",
        "text-to-speech",
        "rlx-sanotts",
    );
    pkg.model["license"] = json!("apache-2.0");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    for (stage, gname) in [
        (TokenStage::Duration, "duration"),
        (TokenStage::Tokens, "tokens"),
    ] {
        let build = |n: usize| {
            lowered(
                token_graph(&du, &du_cfg, &ac, &ac_cfg, &de, &de_cfg, stage, n)?,
                gname,
            )
        };
        let (g, _) =
            symbolic_in_lifted_many(&[(TOKENS, vec![37, 53, 20])], &[0], |n| Ok(build(n[0])?.0))?;
        pkg.function(g);
        params.extend(build(20)?.1);
    }
    let build = |f: usize| lowered(frame_graph(&ac, &ac_cfg, &de, &de_cfg, f)?, "frames");
    let (g, _) =
        symbolic_in_lifted_many(&[(FRAMES, vec![37, 53, 20])], &[0], |n| Ok(build(n[0])?.0))?;
    pkg.function(g);
    params.extend(build(20)?.1);
    pkg.dims = vec![("tokens".into(), TOKENS), ("frames".into(), FRAMES)];
    let dims = pkg.param_dims();
    let mut names: Vec<&String> = params.keys().collect();
    names.sort();
    for n in names {
        let d = dims
            .get(n)
            .cloned()
            .with_context(|| format!("no graph takes {n}"))?;
        pkg.hold_f32(n.clone(), d, &params[n]);
    }

    let espeak = tempfile::tempdir()?;
    espeak_ng::install_bundled_language(espeak.path(), "en")
        .map_err(|e| anyhow::anyhow!("espeak-ng's English data: {e}"))?;
    let mut data = Vec::new();
    files(espeak.path(), espeak.path(), &mut data)?;
    for (rel, p) in data {
        pkg.sidecar(
            &format!("espeak/{rel}"),
            "application/octet-stream",
            std::fs::read(p)?,
        );
    }
    let module = program::build_for(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"),
        "wasm32-wasip1",
    )
    .context("build sanoTTS's program")?;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize"],
        "text-to-speech",
        json!({
            "phoneme_id_map": table["phoneme_id_map"],
            "espeak_voice": espeak_voice,
            "espeak": "/package/espeak",
            "duration_vocab": du_cfg.vocab_size,
            "acoustic_vocab": ac_cfg.vocab_size,
            "fallback": crate::model::SCHWA_FALLBACK_ID,
            "max_tokens": du_cfg.max_tokens,
            "max_duration": du_cfg.max_duration,
            "hidden": ac_cfg.hidden,
            "length_scale": pack.duration_length_scale(),
            "hop": crate::HOP,
            "sample_rate": pack.sample_rate(),
        }),
    );
    pkg.metadata("sample_rate", json!(pack.sample_rate()));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family sanotts`).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("sanotts", "text-to-speech", export_rlxp));
}
