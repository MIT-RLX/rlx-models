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

//! Export NeuTTS — its GGUF backbone and NeuCodec's decoder — as an `.rlxp`
//! package rlx runs alone: text in, 24 kHz speech in a reference voice out
//! (`Model::synthesize`), through the one exporter ([`rlx_core::export`]).
//!
//! The backbone is the GGUF's language model package (a Qwen2 GGUF, Air,
//! through [`rlx_qwen3::export::export_gguf_with`]; a Llama one, Nano,
//! through [`rlx_llama32::export::export_gguf_with`]: its `generate`
//! entry). NeuCodec's decoder is three graphs beside it, symbolic in the
//! frames: `neucodec_embed` (the FSQ codebook's rows of the codes, through
//! `project_out` and `fc_post_a`), `neucodec_decode` (XCodec2's backbone
//! and head, [`rlx_xcodec::export::decoder_functions`]) and
//! `neucodec_istft` (the head's inverse STFT, "same" padding) — as
//! transformers' `NeuCodecModel.decode` computes them. The program
//! (`program/`) phonemizes with espeak-ng (its English data in the
//! package), builds the prompt around the voice's reference codes, samples
//! the backbone a token at a time (top-k 50, top-p 0.9, each token from the
//! stream `seed + step`) and decodes the speech tokens it says.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rlx_core::export::graph::symbolic_in;
use rlx_core::export::{Package, program};
use rlx_ir::{DType, Graph, GraphExt, Shape};
use serde_json::{Value, json};

pub use rlx_core::export::{ExportOptions, ExportReport};

use crate::tokens::STOP_TOKEN;

/// The frames' symbol in NeuCodec's graphs.
const FRAMES: u32 = 0;
/// NeuCodec's output rate.
const SAMPLE_RATE: u32 = 24_000;
/// FSQ: eight dimensions of four levels, 65 536 codes.
const FSQ_LEVELS: usize = 4;
const FSQ_DIMS: usize = 8;
const CODES: usize = 65_536;
/// The sampling of NeuTTS' backbone.
const TOP_K: u64 = 50;
const TOP_P: f64 = 0.9;
const TEMPERATURE: f64 = 1.0;
const MAX_NEW_TOKENS: u32 = 2048;
/// The voice a request names none of, when the package has it.
const DEFAULT_VOICE: &str = "jo";

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// NeuCodec's decoder: `NEUTTS_DECODER_PATH`, else
/// `neucodec_decoder.safetensors` beside the GGUF.
fn decoder_path(gguf: &Path) -> Result<PathBuf> {
    if let Some(p) = crate::decoder::decoder_weights_path_if_available() {
        return Ok(p);
    }
    let p = gguf
        .parent()
        .unwrap_or(Path::new("."))
        .join("neucodec_decoder.safetensors");
    if !p.is_file() {
        bail!(
            "NeuTTS decodes with NeuCodec: no {} (or set NEUTTS_DECODER_PATH)",
            p.display()
        );
    }
    Ok(p)
}

/// The FSQ codebook, `[65 536, 8]`: each code's base-4 digits, scaled to
/// `(d − 2) / 2` — transformers' `NeuCodecFiniteScalarQuantization`.
fn fsq_codebook() -> Vec<f32> {
    let half = (FSQ_LEVELS / 2) as f32;
    let mut book = Vec::with_capacity(CODES * FSQ_DIMS);
    for code in 0..CODES {
        let mut c = code;
        for _ in 0..FSQ_DIMS {
            book.push(((c % FSQ_LEVELS) as f32 - half) / half);
            c /= FSQ_LEVELS;
        }
    }
    book
}

/// `codes` `[t]` i32 → `emb` `[t, 1024]`: the codebook's rows through
/// `project_out` (8 → 2048) and `fc_post_a` (2048 → 1024).
fn embed_graph(t: usize, quant: usize, hidden: usize) -> Graph {
    let mut g = Graph::new("neucodec_embed");
    let codes = g.input("codes", Shape::new(&[t], DType::I32));
    let book = g.param("fsq.codebook", f32s(&[CODES, FSQ_DIMS]));
    let pw = g.param("fsq.project_out.weight", f32s(&[quant, FSQ_DIMS]));
    let pb = g.param("fsq.project_out.bias", f32s(&[quant]));
    let fw = g.param("fc_post_a.weight", f32s(&[hidden, quant]));
    let fb = g.param("fc_post_a.bias", f32s(&[hidden]));
    let rows = g.gather(book, codes, 0, f32s(&[t, FSQ_DIMS]));
    let pwt = g.transpose_(pw, vec![1, 0]);
    let x = g.mm(rows, pwt);
    let x = g.add(x, pb);
    let fwt = g.transpose_(fw, vec![1, 0]);
    let x = g.mm(x, fwt);
    let emb = g.add(x, fb);
    g.node_mut(emb).name = Some("emb".into());
    g.set_outputs(vec![emb]);
    g
}

/// The safetensors tensor `name` as f32.
fn tensor(st: &safetensors::SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name).with_context(|| format!("missing {name}"))?;
    if t.dtype() != safetensors::Dtype::F32 {
        bail!("{name}: {:?}, not f32", t.dtype());
    }
    Ok(rlx_ir::bytes::decode_le_vec::<f32>(t.data()))
}

/// NeuCodec's decoder in `file` as the package's three functions and their
/// params.
fn neucodec_functions(file: &Path) -> Result<(Vec<Graph>, HashMap<String, Vec<f32>>)> {
    let bytes = std::fs::read(file)?;
    let w = rlx_xcodec::model::XcodecWeights::from_safetensors_prefixed(&bytes, "generator.")?;
    let (mut decode, mut istft, mut params) = rlx_xcodec::export::decoder_functions(&w, FRAMES)?;
    decode.name = "neucodec_decode".into();
    istft.name = "neucodec_istft".into();
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let pw = tensor(&st, "generator.quantizer.project_out.weight")?;
    let fw = tensor(&st, "fc_post_a.weight")?;
    let quant = pw.len() / FSQ_DIMS;
    let hidden = fw.len() / quant;
    let embed = symbolic_in(FRAMES, (5, 7, 3), |t| Ok(embed_graph(t, quant, hidden)))?;
    params.insert("fsq.codebook".into(), fsq_codebook());
    params.insert("fsq.project_out.weight".into(), pw);
    params.insert(
        "fsq.project_out.bias".into(),
        tensor(&st, "generator.quantizer.project_out.bias")?,
    );
    params.insert("fc_post_a.weight".into(), fw);
    params.insert("fc_post_a.bias".into(), tensor(&st, "fc_post_a.bias")?);
    Ok((vec![embed, decode, istft], params))
}

/// A 1-D `.npy` of little-endian `i4` / `i8` (a voice's reference codes).
pub fn read_npy_codes(path: &Path) -> Result<Vec<i32>> {
    let b = std::fs::read(path)?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        bail!("{}: not an .npy", path.display());
    }
    let (len, start) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&b[start..start + len])?;
    let data = &b[start + len..];
    if header.contains("'fortran_order': True") {
        bail!("{}: fortran order", path.display());
    }
    Ok(if header.contains("'<i4'") {
        rlx_ir::bytes::decode_le_vec::<i32>(data)
    } else if header.contains("'<i8'") {
        rlx_ir::bytes::decode_le_vec::<i64>(data)
            .into_iter()
            .map(|c| c as i32)
            .collect()
    } else {
        bail!("{}: codes neither <i4 nor <i8 ({header})", path.display())
    })
}

/// The reference voices: each `<name>.npy` (its codes) with `<name>.txt`
/// (what it says) in `NEUTTS_SAMPLES`, else `samples/` beside the GGUF.
fn voices(gguf: &Path) -> Result<serde_json::Map<String, Value>> {
    let dir = match std::env::var("NEUTTS_SAMPLES") {
        Ok(d) if !d.is_empty() => PathBuf::from(d),
        _ => gguf.parent().unwrap_or(Path::new(".")).join("samples"),
    };
    let mut out = serde_json::Map::new();
    if !dir.is_dir() {
        return Ok(out);
    }
    let mut names: Vec<PathBuf> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "npy"))
        .collect();
    names.sort();
    for npy in names {
        let txt = npy.with_extension("txt");
        if !txt.is_file() {
            continue;
        }
        let name = npy
            .file_stem()
            .and_then(|s| s.to_str())
            .context("voice name")?;
        let codes = read_npy_codes(&npy)?;
        if codes.iter().any(|c| !(0..CODES as i32).contains(c)) {
            bail!("{}: a code outside NeuCodec's 65 536", npy.display());
        }
        let text = std::fs::read_to_string(&txt)?.trim().to_string();
        out.insert(name.into(), json!({"codes": codes, "text": text}));
    }
    Ok(out)
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

/// The id the GGUF's tokenizer gives `token` (one of its added tokens).
fn token_id(gguf: &Path, token: &str) -> Result<u32> {
    match rlx_qwen35::encode_prompt_from_gguf(gguf, token)?.as_slice() {
        [id] => Ok(*id),
        ids => bail!("{token}: {} ids, not one ({ids:?})", ids.len()),
    }
}

/// Export the NeuTTS GGUF `gguf` (NeuCodec's decoder beside it, its
/// reference voices in `samples/`) as the package `out`.
pub fn export_rlxp(gguf: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let decoder = decoder_path(gguf)?;
    let voices = voices(gguf)?;
    let speech_base = token_id(gguf, "<|speech_0|>")?;
    if token_id(gguf, &format!("<|speech_{}|>", CODES - 1))? != speech_base + CODES as u32 - 1 {
        bail!("{}: its speech tokens are not in order", gguf.display());
    }
    let stop = token_id(gguf, STOP_TOKEN)?;
    let default_voice = if voices.contains_key(DEFAULT_VOICE) {
        Value::from(DEFAULT_VOICE)
    } else {
        voices.keys().next().map_or(Value::Null, |v| json!(v))
    };
    let names: Vec<String> = voices.keys().cloned().collect();
    let mut extend = |pkg: &mut Package| -> Result<()> {
        pkg.model["family"] = json!("neutts");
        pkg.model["task"] = json!("text-to-speech");
        let (graphs, params) = neucodec_functions(&decoder)?;
        for g in graphs {
            pkg.function(g);
        }
        let dims = pkg.param_dims();
        let mut held: Vec<&String> = params.keys().collect();
        held.sort();
        for n in held {
            let d = dims
                .get(n)
                .cloned()
                .unwrap_or_else(|| vec![params[n].len()]);
            pkg.hold_f32(n.clone(), d, &params[n]);
        }
        // espeak-ng's English data, for the program's phonemizer.
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
        program::entry(
            pkg,
            "speak",
            module,
            &["synthesize", "run"],
            "text-to-speech",
            json!({
                "language": "en-us",
                "espeak": "/package/espeak",
                "voices": voices,
                "default_voice": default_voice,
                "speech_base": speech_base,
                "speech_count": CODES,
                "stop": stop,
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "top_p": TOP_P,
                "seed": 0,
                "sample_rate": SAMPLE_RATE,
            }),
        );
        pkg.metadata("sample_rate", json!(SAMPLE_RATE));
        pkg.metadata("voices", json!(names));
        Ok(())
    };
    let raw = rlx_gguf::GgufFile::from_path(gguf)
        .map_err(|e| anyhow::anyhow!("open {}: {e}", gguf.display()))?;
    let arch = raw
        .metadata
        .get("general.architecture")
        .and_then(rlx_gguf::MetaValue::as_str)
        .unwrap_or("")
        .to_string();
    drop(raw);
    match arch.as_str() {
        "qwen2" => rlx_qwen3::export::export_gguf_with(
            gguf,
            out,
            opts,
            Some(rlx_qwen35::gguf_tokenizer_json),
            &mut extend,
        ),
        "llama" => rlx_llama32::export::export_gguf_with(gguf, out, opts, &mut extend),
        a => bail!(
            "{}: a NeuTTS backbone is Qwen2 or Llama, not {a:?}",
            gguf.display()
        ),
    }
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("neutts", "text-to-speech", export_rlxp));
}
