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

//! Export LuxTTS as an `.rlxp` package rlx runs alone — text and a voice
//! to clone in, 24 kHz speech out (`Model::synthesize`, the voice
//! `options.prompt`) — through the one exporter ([`rlx_core::export`]).
//!
//! Its graphs are rlx graphs symbolic in their lengths: the encoder body,
//! the flow decoder and the vocoder's spectral head the checkpoint's ONNX
//! imported once, here ([`symbolic_in_lifted`]); the prompt's log-mel and
//! the vocoder's ISTFT, which the runner does in Rust ([`crate::dsp`]),
//! [`log_mel`] and [`istft_center`]. The rest of what
//! [`crate::LuxTts::synthesize`] does — espeak-ng, the token table, the
//! prompt's padding, the length regulator, the noise, the flow-matching
//! update — is the package's program (`program/`, wasm32-wasip1).

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_core::export::audio::{istft_bases, istft_center, log_mel, stft_bases};
use rlx_core::export::graph::{symbolic_in, symbolic_in_lifted};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport, Package, program};
use rlx_ir::Graph;
use serde_json::json;

use crate::config::Layout;
use crate::dsp::{self, HOP, N_FFT, N_FREQ, N_MELS};
use crate::model::InferOpts;

/// The symbols: the ids, the flow's frames, the vocoder's and the prompt's.
const IDS: u32 = 0;
const FRAMES: u32 = 1;
const VOCODER: u32 = 2;
const PROMPT: u32 = 3;
/// The flow decoder's frames modulo this pick its graph.
const DOWNSAMPLE: usize = 4;

/// The ONNX graph `file` imported at `len` (its length named `dim`) as the
/// rlx graph `name`, its params kept.
fn import(
    file: &Path,
    name: &str,
    dim: &str,
    len: usize,
    params: &RefCell<HashMap<String, Vec<f32>>>,
) -> Result<Graph> {
    let (g, weights) = rlx_tiny_tts::model::import_rlx_graph(file, name, len, &[(dim, len)])?;
    let mut held = params.borrow_mut();
    for (k, v) in weights {
        held.entry(k).or_insert(v);
    }
    Ok(g)
}

/// Every file under `dir`, by its path relative to `base`.
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

/// Export the LuxTTS checkpoint in `dir` (`tokens.txt`, `encoder_body.onnx`,
/// `fm_decoder.onnx`, `onnx/vocoder_spec.onnx`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    export_family(
        dir,
        out,
        opts,
        "luxtts",
        "rlx-luxtts",
        &InferOpts::default(),
    )
}

/// [`export_rlxp`] for a model of LuxTTS' architecture (ZipVoice, which
/// LuxTTS fine-tunes): its `family`, the crate exporting it, and its
/// sampler's defaults. A directory with no `config.json` names the package.
pub fn export_family(
    dir: &Path,
    out: &Path,
    opts: &ExportOptions,
    family: &str,
    exported_by: &str,
    defaults: &InferOpts,
) -> Result<ExportReport> {
    let layout = Layout::resolve(dir)?;
    let encoder_body = layout
        .encoder_body
        .clone()
        .context("no encoder_body.onnx (scripts/export_encoder_body.py)")?;
    let mut pkg = if dir.join("config.json").is_file() {
        let ck = Checkpoint::open(dir)?;
        Package::for_checkpoint(&ck, opts, family, "text-to-speech", exported_by)
    } else {
        let name = layout
            .dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(family)
            .to_string();
        let mut pkg = Package::new(&name, &opts.version, family, "text-to-speech", exported_by);
        if let Ok(card) = std::fs::read_to_string(dir.join("README.md")) {
            rlx_core::export::add_huggingface_card(&mut pkg.model, &card);
        }
        pkg
    };

    let held = RefCell::new(HashMap::new());
    let mut params = HashMap::new();
    for (file, name, dim, sym, sizes) in [
        (&encoder_body, "encoder_body", "S", IDS, (37, 45, 53)),
        (
            &layout.vocoder_spec,
            "vocoder_spec",
            "l",
            VOCODER,
            (37, 45, 53),
        ),
    ] {
        let (g, tables) =
            symbolic_in_lifted(sym, sizes, 8192, |n| import(file, name, dim, n, &held))
                .with_context(|| name.to_string())?;
        params.extend(tables);
        pkg.function(g);
    }
    // The flow decoder (a Zipformer) downsamples its frames: a dim is
    // `ceil(T / d)`, so one graph per `T mod DOWNSAMPLE`, each symbolic in
    // `K` (`T = DOWNSAMPLE · K + r`), its weights the same.
    for r in 0..DOWNSAMPLE {
        let name = format!("fm_decoder_{r}");
        let (mut g, tables) = symbolic_in_lifted(FRAMES, (37, 45, 53), 8192, |k| {
            import(
                &layout.fm_decoder,
                "fm_decoder",
                "T",
                DOWNSAMPLE * k + r,
                &held,
            )
        })
        .with_context(|| name.clone())?;
        g.name = name;
        params.extend(tables);
        pkg.function(g);
    }
    params.extend(held.take());

    let window = dsp::hann_periodic(N_FFT);
    let mel = symbolic_in(PROMPT, (37, 45, 53), |t| log_mel(N_FFT, HOP, N_MELS, t))?;
    let istft = symbolic_in(VOCODER, (37, 45, 53), |t| {
        istft_center(N_FFT, HOP, &window, t)
    })?;
    pkg.function(mel);
    pkg.function(istft);
    params.extend(stft_bases(N_FFT, &window));
    params.extend(istft_bases(N_FFT, &window));
    let fb = dsp::mel_filterbank();
    params.insert(
        "mel.filters".into(),
        (0..N_FREQ * N_MELS)
            .map(|i| fb[(i % N_MELS) * N_FREQ + i / N_MELS])
            .collect(),
    );

    pkg.dims = vec![
        ("ids".into(), IDS),
        ("frame_blocks".into(), FRAMES),
        ("vocoder_frames".into(), VOCODER),
        ("prompt_frames".into(), PROMPT),
    ];
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

    pkg.sidecar(
        "tokens.txt",
        "text/plain",
        std::fs::read(dir.join("tokens.txt"))?,
    );
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
    )?;
    let d = defaults;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize"],
        "text-to-speech",
        json!({
            "espeak": "/package/espeak",
            "fm_decoders": DOWNSAMPLE,
            "target_rms": crate::model::TARGET_RMS,
            "feat_scale": crate::model::FEAT_SCALE,
            "defaults": {
                "num_step": d.num_step,
                "guidance_scale": d.guidance_scale,
                "speed": d.speed,
                "speed_mult": d.speed_mult,
                "t_shift": d.t_shift,
            },
        }),
    );
    pkg.metadata("sample_rate", json!(dsp::SR));
    pkg.metadata(
        "prompt",
        json!("options.prompt: {audio: [24 kHz mono samples], sample_rate: 24000, text: its transcript}"),
    );
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("luxtts", "text-to-speech", export_rlxp).claims(&["luxtts"]));
}
