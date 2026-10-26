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

//! Export MiraTTS as an `.rlxp` package rlx runs alone: text and a voice
//! (a reference clip, or its 32 global tokens) in, 16 kHz speech out,
//! through the one exporter ([`rlx_core::export`]).
//!
//! The Qwen2 LM is its checkpoint's language-model package
//! ([`rlx_qwen3::export::export_rlxp_with`]: its `generate` entry). BiCodec's
//! ONNX graphs (`decoders/`, the export-time reference) are rlx IR, at the
//! fixed sizes the runner runs them: `detokenizer` (249 acoustic codes, 32
//! global tokens → audio) and `s_encoder` (the reference's mel → its 32
//! global tokens); the mel itself (`s_mel`: 6 s of 16 kHz, Hann-windowed
//! 640-sample frames every 320, their 1024-point magnitude spectra through
//! the Slaney filterbank) is a graph too. The program (`program/`) builds the
//! prompt, samples the LM a token at a time (its repetition penalty, then the
//! host's `sample`: temperature 0.8, top-k 50, top-p 0.95, min-p 0.05, each
//! token from the stream `seed + step`) and decodes the acoustic codes.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::{Package, program};
use rlx_ir::{DType, Graph, GraphExt, Shape};
use serde_json::json;

pub use rlx_core::export::{ExportOptions, ExportReport};

use crate::codec::{CONTEXT_LEN, SAMPLE_RATE, SPEECH_LEN};
use crate::encoder::REF_SAMPLES;
use crate::tokens;

/// The mel's framing (FastBiCodec's).
const N_FFT: usize = 1024;
const WIN: usize = 640;
const HOP: usize = 320;
const N_MELS: usize = 128;
const F_MIN: f64 = 10.0;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn hz_to_mel(hz: f64) -> f64 {
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / (200.0 / 3.0);
    let logstep = 6.4f64.ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        hz / (200.0 / 3.0)
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / (200.0 / 3.0);
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        mel * (200.0 / 3.0)
    }
}

/// The Slaney filterbank, `[bins, mels]` (the runner's, transposed).
fn filterbank() -> Vec<f32> {
    let n_bins = N_FFT / 2 + 1;
    let sr = SAMPLE_RATE as f64;
    let freqs: Vec<f64> = (0..n_bins).map(|k| k as f64 * sr / N_FFT as f64).collect();
    let n_pts = N_MELS + 2;
    let pts: Vec<f64> = (0..n_pts)
        .map(|i| {
            mel_to_hz(
                hz_to_mel(F_MIN)
                    + (hz_to_mel(sr * 0.5) - hz_to_mel(F_MIN)) * i as f64 / (n_pts - 1) as f64,
            )
        })
        .collect();
    let mut fb = vec![0f32; n_bins * N_MELS];
    for m in 0..N_MELS {
        let (left, center, right) = (pts[m], pts[m + 1], pts[m + 2]);
        for (k, &f) in freqs.iter().enumerate() {
            fb[k * N_MELS + m] = if f < left || f > right {
                0.0
            } else if f <= center {
                ((f - left) / (center - left).max(1e-8)) as f32
            } else {
                ((right - f) / (right - center).max(1e-8)) as f32
            };
        }
    }
    fb
}

/// `pcm` `[96000]` → `mel` `[1, frames, 128]`: each 640-sample frame (two
/// adjacent 320-sample blocks) Hann-windowed, its 1024-point DFT (the frame
/// zero-padded) as `cos` / `sin` matmuls, the magnitude through the
/// filterbank. The DFT bases and the filterbank are params.
fn mel_graph() -> (Graph, HashMap<String, Vec<f32>>) {
    let frames = 1 + (REF_SAMPLES - WIN) / HOP;
    let blocks = REF_SAMPLES / HOP;
    let n_bins = N_FFT / 2 + 1;
    let mut g = Graph::new("s_mel");
    let pcm = g.input("pcm", f32s(&[REF_SAMPLES]));
    let b = g.reshape_(pcm, vec![blocks as i64, HOP as i64]);
    let first = g.narrow_(b, 0, 0, frames);
    let second = g.narrow_(b, 0, 1, frames);
    let framed = g.concat_(vec![first, second], 1); // [frames, 640]
    let window = g.param("s_mel.window", f32s(&[1, WIN]));
    let framed = g.mul(framed, window);
    let cos = g.param("s_mel.cos", f32s(&[WIN, n_bins]));
    let sin = g.param("s_mel.sin", f32s(&[WIN, n_bins]));
    let re = g.mm(framed, cos);
    let im = g.mm(framed, sin);
    let re2 = g.mul(re, re);
    let im2 = g.mul(im, im);
    let power = g.add(re2, im2);
    let mag = g.sqrt(power);
    let fb = g.param("s_mel.filterbank", f32s(&[n_bins, N_MELS]));
    let mel = g.mm(mag, fb);
    let mel = g.reshape_(mel, vec![1, frames as i64, N_MELS as i64]);
    g.node_mut(mel).name = Some("mel".into());
    g.set_outputs(vec![mel]);

    let mut params = HashMap::new();
    params.insert(
        "s_mel.window".into(),
        (0..WIN)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / WIN as f32).cos())
            .collect(),
    );
    let (mut c, mut s) = (vec![0f32; WIN * n_bins], vec![0f32; WIN * n_bins]);
    for n in 0..WIN {
        for k in 0..n_bins {
            let a = -2.0 * std::f64::consts::PI * (k * n) as f64 / N_FFT as f64;
            c[n * n_bins + k] = a.cos() as f32;
            s[n * n_bins + k] = a.sin() as f32;
        }
    }
    params.insert("s_mel.cos".into(), c);
    params.insert("s_mel.sin".into(), s);
    params.insert("s_mel.filterbank".into(), filterbank());
    (g, params)
}

/// The mel frames of the reference: what `s_encoder` is built at.
pub fn mel_frames() -> usize {
    1 + (REF_SAMPLES - WIN) / HOP
}

/// BiCodec's graphs at the runner's sizes, and their params.
fn codec_functions(decoders: &Path) -> Result<Vec<(Graph, HashMap<String, Vec<f32>>)>> {
    let (detok, p1) = rlx_tiny_tts::model::import_rlx_graph(
        &decoders.join("detokenizer.onnx"),
        "detokenizer",
        SPEECH_LEN,
        &[],
    )?;
    let t = mel_frames();
    let (senc, p2) = rlx_tiny_tts::model::import_rlx_graph(
        &decoders.join("s_encoder.onnx"),
        "s_encoder",
        t,
        &[("mel_time_steps", t), ("batch_size", 1)],
    )?;
    Ok(vec![(detok, p1), (senc, p2), mel_graph()])
}

/// Export the MiraTTS directory `dir` (its Qwen2 checkpoint and
/// `tokenizer.json`, BiCodec's ONNX in `decoders/`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let decoders = dir.join("decoders");
    rlx_qwen3::export::export_rlxp_with(dir, out, opts, &mut |pkg: &mut Package| {
        pkg.model["family"] = json!("miratts");
        pkg.model["task"] = json!("text-to-speech");
        let mut held = HashMap::new();
        for (g, p) in codec_functions(&decoders)? {
            pkg.function(g);
            held.extend(p);
        }
        let dims = pkg.param_dims();
        let mut names: Vec<&String> = held.keys().collect();
        names.sort();
        for n in names {
            let d = dims.get(n).cloned().unwrap_or_else(|| vec![held[n].len()]);
            pkg.hold_f32(n.clone(), d, &held[n]);
        }
        let module =
            program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))
                .context("build MiraTTS' program")?;
        program::entry(
            pkg,
            "speak",
            module,
            &["synthesize", "run"],
            "text-to-speech",
            json!({
                "task_tts": tokens::TASK_TTS,
                "start_text": tokens::START_TEXT,
                "end_text": tokens::END_TEXT,
                "context_audio_start": tokens::CONTEXT_AUDIO_START,
                "context_audio_end": tokens::CONTEXT_AUDIO_END,
                "prompt_speech_start": tokens::PROMPT_SPEECH_START,
                "context_base": tokens::CONTEXT_BASE,
                "speech_base": tokens::SPEECH_BASE,
                "speech_codebook": tokens::SPEECH_CODEBOOK,
                "stops": [tokens::EOS, tokens::END_ACOUSTIC_TOKEN],
                "speech_len": SPEECH_LEN,
                "context_len": CONTEXT_LEN,
                "ref_samples": REF_SAMPLES,
                "max_new_tokens": SPEECH_LEN,
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.95,
                "min_p": 0.05,
                "repetition_penalty": 1.2,
                "sample_rate": SAMPLE_RATE,
            }),
        );
        pkg.metadata("sample_rate", json!(SAMPLE_RATE));
        Ok(())
    })
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("miratts", "text-to-speech", export_rlxp));
}
