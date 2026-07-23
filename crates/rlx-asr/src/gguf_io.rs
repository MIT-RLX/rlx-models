// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Pack / load ASR weights as a single GGUF (`weights/asr/model.gguf`).
//!
//! Tensor naming (`general.architecture = "rlx-asr"`):
//! - `encoder.<npz_key>` — folded whole-model float tensors
//! - `decoder.<name>` — AED maps (`embed`, `effective_Ah_tok`, `W_out`, …)
//! - `codebook.layer{N}.linear_k.{int8,scale}` — TEXT K int8 + row scale
//! - `ls.layer{N}.linear_v.{weight_128,weight_pad,int8,scale}` — LS V
//! - `tp.<stem>` — FST byte blobs (I8)
//! - `silence_fbank` — mel silence template
//!
//! Metadata: `rlx-asr.units` (string array), optional `rlx-asr.etiquette_json`.

use crate::npy_io::{load_npy, load_npz, read_f32_txt, NpyArray, NpyDType};
use crate::spec::{DECODER_DIM, VOCAB};
use crate::weights::read_f32_bin;
use anyhow::{bail, Context, Result};
use rlx_gguf::{GgmlType, GgufFile, GgufWriter, MetaValue};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

pub const ASR_GGUF_ARCH: &str = "rlx-asr";
pub const ASR_GGUF_FORMAT: &str = "rlx-asr-gguf-v1";
pub const DEFAULT_GGUF_NAME: &str = "model.gguf";

#[derive(Debug, Default)]
pub struct PackReport {
    pub path: PathBuf,
    pub n_tensors: usize,
    pub bytes: u64,
    pub skipped: Vec<String>,
}

/// Pack ASR tensors into one GGUF.
///
/// `root` is the published tree (sidecars + output). Tensor pack sources may
/// live under `root` or a cache via [`pack_source_roots`].
pub fn pack_asr_gguf(root: &Path, out: &Path) -> Result<PackReport> {
    let sources = pack_source_roots(root);
    let mut w = GgufWriter::new();
    w.set_arch(ASR_GGUF_ARCH);
    w.set_meta("general.name", MetaValue::String("rlx-asr".into()));
    w.set_meta(
        "general.description",
        MetaValue::String("Native RLX streaming Conformer ASR weights".into()),
    );
    w.set_meta(
        "rlx-asr.format",
        MetaValue::String(ASR_GGUF_FORMAT.into()),
    );
    w.set_meta("rlx-asr.vocab_size", MetaValue::U32(VOCAB as u32));
    w.set_meta("rlx-asr.decoder_dim", MetaValue::U32(DECODER_DIM as u32));

    let mut skipped = Vec::new();
    let mut names: BTreeSet<String> = BTreeSet::new();

    // units.txt → metadata string array
    let units_path = first_existing(
        &sources
            .iter()
            .flat_map(|s| [s.join("units.txt"), s.join("misc/units.txt")])
            .collect::<Vec<_>>(),
    )
    .unwrap_or_else(|| root.join("units.txt"));
    if units_path.is_file() {
        let text = fs::read_to_string(&units_path)?;
        let pieces: Vec<MetaValue> = text
            .lines()
            .map(|line| {
                let piece = line.split_whitespace().next().unwrap_or("").to_string();
                MetaValue::String(piece)
            })
            .collect();
        w.set_meta("rlx-asr.vocab_size_units", MetaValue::U32(pieces.len() as u32));
        if !pieces.is_empty() {
            w.set_meta("rlx-asr.units", MetaValue::Array(pieces));
        }
    } else {
        skipped.push("units.txt".into());
    }

    if let Some(eti_path) = first_existing(
        &sources
            .iter()
            .map(|s| s.join("etiquette.json"))
            .collect::<Vec<_>>(),
    ) {
        if let Ok(eti) = fs::read_to_string(eti_path) {
            w.set_meta("rlx-asr.etiquette_json", MetaValue::String(eti));
        }
    }

    if let Some(silence) = first_existing(
        &sources
            .iter()
            .flat_map(|s| [s.join("silence-fbank.txt"), s.join("misc/silence-fbank.txt")])
            .collect::<Vec<_>>(),
    ) {
        let v = read_f32_txt(&silence)?;
        if !v.is_empty() {
            add_f32(&mut w, &mut names, "silence_fbank", vec![v.len()], &v)?;
        }
    }

    // Encoder folded NPZ
    if let Some(enc_npz) = resolve_encoder_npz(root) {
        for (key, arr) in load_npz(&enc_npz)? {
            let name = format!("encoder.{key}");
            add_npy(&mut w, &mut names, &name, &arr)?;
        }
    } else {
        skipped.push("encoder_whole_model.npz".into());
    }

    for cand in resolve_body_residual_candidates(root) {
        if !cand.is_file() {
            continue;
        }
        for (key, arr) in load_npz(&cand)? {
            if key != "R" {
                continue;
            }
            let name = "encoder.frontend.body_residual_ls.R".to_string();
            if names.contains(&name) {
                continue;
            }
            add_npy(&mut w, &mut names, &name, &arr)?;
        }
        break;
    }

    // Decoder AED bins
    let decoder_bins = [
        ("embed", "embed.bin"),
        ("effective_Ah_tok", "effective_Ah_tok.bin"),
        ("effective_Ah", "effective_Ah.bin"),
        ("effective_Ak", "effective_Ak.bin"),
        ("effective_Av", "effective_Av.bin"),
        ("effective_We", "effective_We.bin"),
        ("effective_Wc", "effective_Wc.bin"),
        ("effective_bh", "effective_bh.bin"),
        ("W_out", "W_out.bin"),
        ("b_out", "b_out.bin"),
    ];
    let dec_dirs = resolve_decoder_dirs(root);
    for (logical, file) in decoder_bins {
        let Some(p) = dec_dirs.iter().map(|d| d.join(file)).find(|p| p.is_file()) else {
            continue;
        };
        if logical == "effective_Ah" && names.contains("decoder.effective_Ah_tok") {
            continue;
        }
        let v = read_f32_bin(&p)?;
        add_f32(
            &mut w,
            &mut names,
            &format!("decoder.{logical}"),
            vec![v.len()],
            &v,
        )?;
    }

    // Codebook
    if let Some(cb_root) = resolve_codebook_by_layer(root) {
        for layer_dir in sorted_dirs(&cb_root)? {
            let layer =
                parse_layer_name(layer_dir.file_name().and_then(|s| s.to_str()).unwrap_or(""))
                    .unwrap_or(0);
            let prefix = format!("codebook.layer{layer}.linear_k");
            let int8 = layer_dir.join("linear_k_int8.bin");
            let scale = layer_dir.join("linear_k_scale_fp32.bin");
            if int8.is_file() {
                let bytes = fs::read(&int8)?;
                add_i8(
                    &mut w,
                    &mut names,
                    &format!("{prefix}.int8"),
                    vec![DECODER_DIM, DECODER_DIM],
                    &bytes,
                )?;
            } else if let Ok(arr) = load_npy(&layer_dir.join("linear_k_int8.npy")) {
                add_npy(&mut w, &mut names, &format!("{prefix}.int8"), &arr)?;
            }
            if scale.is_file() {
                let v = read_f32_bin(&scale)?;
                add_f32(
                    &mut w,
                    &mut names,
                    &format!("{prefix}.scale"),
                    vec![v.len()],
                    &v,
                )?;
            } else if let Ok(arr) = load_npy(&layer_dir.join("linear_k_scale_fp32.npy")) {
                add_npy(&mut w, &mut names, &format!("{prefix}.scale"), &arr)?;
            }
        }
    }

    // LS V projections
    if let Some(ls_by) = resolve_ls_by_layer(root) {
        for layer_dir in sorted_dirs(&ls_by)? {
            let layer =
                parse_layer_name(layer_dir.file_name().and_then(|s| s.to_str()).unwrap_or(""))
                    .unwrap_or(0);
            let prefix = format!("ls.layer{layer}");
            for (file, suffix, shape) in [
                (
                    "linear_v_ls_128x512.bin",
                    "linear_v.weight_128",
                    Some(vec![128usize, 512]),
                ),
                (
                    "linear_v_ls_512x512_headpad.bin",
                    "linear_v.weight_pad",
                    Some(vec![512, 512]),
                ),
                (
                    "linear_v_ls_int8.bin",
                    "linear_v.int8",
                    Some(vec![128, 512]),
                ),
                ("linear_v_ls_scale_fp32.bin", "linear_v.scale", None),
            ] {
                let p = layer_dir.join(file);
                if !p.is_file() {
                    continue;
                }
                let name = format!("{prefix}.{suffix}");
                if file.contains("int8") {
                    let bytes = fs::read(&p)?;
                    let shape = shape.unwrap_or_else(|| vec![bytes.len()]);
                    add_i8(&mut w, &mut names, &name, shape, &bytes)?;
                } else {
                    let v = read_f32_bin(&p)?;
                    let shape = shape.unwrap_or_else(|| vec![v.len()]);
                    let expect: usize = shape.iter().product();
                    if expect != 0 && expect != v.len() {
                        bail!("{}: got {} f32, shape {shape:?}", p.display(), v.len());
                    }
                    add_f32(&mut w, &mut names, &name, shape, &v)?;
                }
            }
        }
    }

    // TP FSTs
    for tp in resolve_tp_dirs(root) {
        for ent in fs::read_dir(&tp)? {
            let ent = ent?;
            let p = ent.path();
            if p.extension().and_then(|e| e.to_str()) != Some("fst") {
                continue;
            }
            let stem = p.file_stem().unwrap().to_string_lossy().to_string();
            let name = format!("tp.{stem}");
            if names.contains(&name) {
                continue;
            }
            let bytes = fs::read(&p)?;
            add_i8(&mut w, &mut names, &name, vec![bytes.len()], &bytes)?;
        }
    }

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    w.write_to_path(out)
        .with_context(|| format!("write {}", out.display()))?;
    let bytes = fs::metadata(out)?.len();
    Ok(PackReport {
        path: out.to_path_buf(),
        n_tensors: names.len(),
        bytes,
        skipped,
    })
}

/// Roots searched for pack sources (published tree + optional cache).
fn pack_source_roots(root: &Path) -> Vec<PathBuf> {
    let mut out = vec![root.to_path_buf()];
    if let Some(p) = std::env::var_os("RLX_ASR_PACK_SRC") {
        out.push(PathBuf::from(p));
    }
    if let Some(repo) = root.parent().and_then(|p| p.parent()) {
        out.push(repo.join(".cache/asr"));
    }
    out.push(PathBuf::from(".cache/asr"));
    out
}

fn first_existing(cands: &[PathBuf]) -> Option<PathBuf> {
    cands.iter().find(|p| p.is_file()).cloned()
}

fn resolve_encoder_npz(root: &Path) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("RLX_ASR_ENCODER_NPZ") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let mut cands = Vec::new();
    for s in pack_source_roots(root) {
        cands.push(s.join("encoder/encoder_whole_model.npz"));
        cands.push(s.join("asr_weights/encoder_native/encoder_whole_model.npz"));
        cands.push(s.join("encoder_native/encoder_whole_model.npz"));
    }
    cands.into_iter().find(|p| p.is_file())
}

fn resolve_body_residual_candidates(root: &Path) -> Vec<PathBuf> {
    let mut cands = Vec::new();
    for s in pack_source_roots(root) {
        cands.push(s.join("encoder/body_residual_ls.npz"));
        cands.push(s.join("ls/body_residual_ls.npz"));
        cands.push(s.join("asr_weights/encoder_ls_projections/body_residual_ls.npz"));
    }
    cands
}

fn resolve_decoder_dirs(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for s in pack_source_roots(root) {
        for p in [
            s.join("decoder"),
            s.join("asr_weights"),
            s.clone(),
        ] {
            if p.is_dir() {
                out.push(p);
            }
        }
    }
    out
}

fn resolve_codebook_by_layer(root: &Path) -> Option<PathBuf> {
    for s in pack_source_roots(root) {
        for p in [
            s.join("codebook/by_layer"),
            s.join("asr_weights/encoder_text_codebook/k_codebook/by_layer"),
        ] {
            if p.is_dir() {
                return Some(p);
            }
        }
    }
    None
}

fn resolve_ls_by_layer(root: &Path) -> Option<PathBuf> {
    for s in pack_source_roots(root) {
        for p in [
            s.join("ls/by_layer"),
            s.join("asr_weights/encoder_ls_projections/by_layer"),
        ] {
            if p.is_dir() {
                return Some(p);
            }
        }
    }
    None
}

fn resolve_tp_dirs(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for s in pack_source_roots(root) {
        let p = s.join("TP");
        if p.is_dir() {
            out.push(p);
        }
    }
    out
}

fn sorted_dirs(root: &Path) -> Result<Vec<PathBuf>> {
    let mut dirs: Vec<PathBuf> = fs::read_dir(root)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    Ok(dirs)
}

fn parse_layer_name(name: &str) -> Option<usize> {
    let s = name.strip_prefix('L').or_else(|| name.strip_prefix('l'))?;
    s.parse().ok()
}

fn add_f32(
    w: &mut GgufWriter,
    names: &mut BTreeSet<String>,
    name: &str,
    shape: Vec<usize>,
    data: &[f32],
) -> Result<()> {
    if !names.insert(name.to_string()) {
        return Ok(());
    }
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    w.add_tensor_bytes(name, shape, GgmlType::F32, bytes)?;
    Ok(())
}

fn add_i8(
    w: &mut GgufWriter,
    names: &mut BTreeSet<String>,
    name: &str,
    shape: Vec<usize>,
    data: &[u8],
) -> Result<()> {
    if !names.insert(name.to_string()) {
        return Ok(());
    }
    let expect: usize = shape.iter().product();
    if expect != 0 && expect != data.len() {
        bail!("{name}: i8 len {} != shape product {expect}", data.len());
    }
    w.add_tensor_bytes(name, shape, GgmlType::I8, data.to_vec())?;
    Ok(())
}

fn add_npy(
    w: &mut GgufWriter,
    names: &mut BTreeSet<String>,
    name: &str,
    arr: &NpyArray,
) -> Result<()> {
    match arr.dtype {
        NpyDType::F32 => {
            let mut v = Vec::with_capacity(arr.n_elements());
            for c in arr.data.chunks_exact(4) {
                v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            add_f32(w, names, name, arr.shape.clone(), &v)
        }
        NpyDType::F64 => {
            // Downcast debug/fit dumps to f32 for the runtime pack.
            let mut v = Vec::with_capacity(arr.n_elements());
            for c in arr.data.chunks_exact(8) {
                let x = f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
                v.push(x as f32);
            }
            add_f32(w, names, name, arr.shape.clone(), &v)
        }
        NpyDType::I8 | NpyDType::U8 => add_i8(w, names, name, arr.shape.clone(), &arr.data),
        NpyDType::I64 => {
            // Token id dumps — store as F32 ids for simplicity.
            let mut v = Vec::with_capacity(arr.n_elements());
            for c in arr.data.chunks_exact(8) {
                let x = i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
                v.push(x as f32);
            }
            add_f32(w, names, name, arr.shape.clone(), &v)
        }
        other => {
            eprintln!("[rlx-asr] skip {name}: unsupported npy dtype {other:?}");
            Ok(())
        }
    }
}

/// Opened ASR GGUF with helpers for runtime tensors.
pub struct AsrGguf {
    pub path: PathBuf,
    file: GgufFile,
}

impl AsrGguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = GgufFile::from_path(&path).with_context(|| format!("open {}", path.display()))?;
        Ok(Self { path, file })
    }

    pub fn file(&self) -> &GgufFile {
        &self.file
    }

    pub fn has(&self, name: &str) -> bool {
        self.file.get(name).is_some()
    }

    pub fn f32_tensor(&self, name: &str) -> Result<Vec<f32>> {
        let (data, _shape) = self.file.dequant_f32(name)?;
        Ok(data)
    }

    pub fn i8_tensor(&self, name: &str) -> Result<Vec<i8>> {
        let t = self
            .file
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))?;
        let bytes = self.file.tensor_bytes(t)?;
        Ok(bytes.iter().map(|&b| b as i8).collect())
    }

    /// Raw bytes for an I8 blob tensor (e.g. `tp.us2common`).
    pub fn blob(&self, name: &str) -> Result<Vec<u8>> {
        let t = self
            .file
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))?;
        Ok(self.file.tensor_bytes(t)?.to_vec())
    }

    /// Locale Hammer FSTs embedded as `tp.<stem>` blobs.
    pub fn load_hammer(&self, locale: &str) -> Result<crate::textproc::Hammer> {
        crate::textproc::Hammer::load_from_blobs(|stem| {
            let key = format!("tp.{stem}");
            self.blob(&key).ok()
        }, locale)
    }

    pub fn units(&self) -> Option<Vec<String>> {
        let mv = self.file.metadata.get("rlx-asr.units")?;
        let MetaValue::Array(items) = mv else {
            return None;
        };
        Some(
            items
                .iter()
                .filter_map(|v| match v {
                    MetaValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
        )
    }

    pub fn etiquette_json(&self) -> Option<&str> {
        match self.file.metadata.get("rlx-asr.etiquette_json") {
            Some(MetaValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn silence_fbank(&self) -> Result<Vec<f32>> {
        self.f32_tensor("silence_fbank")
    }

    /// Load AED maps from GGUF (`decoder.*` tensors).
    pub fn load_effective_step1(&self) -> Result<crate::effective_decoder::EffectiveStep1> {
        let embed = self.f32_tensor("decoder.embed")?;
        let ah = self
            .f32_tensor("decoder.effective_Ah_tok")
            .or_else(|_| self.f32_tensor("decoder.effective_Ah"))?;
        let w_out = self.f32_tensor("decoder.W_out")?;
        let b_out = self.f32_tensor("decoder.b_out")?;
        if embed.len() != VOCAB * DECODER_DIM {
            bail!("embed len {}", embed.len());
        }
        if ah.len() != DECODER_DIM * DECODER_DIM {
            bail!("Ah len {}", ah.len());
        }
        if w_out.len() != VOCAB * DECODER_DIM {
            bail!("W_out len {}", w_out.len());
        }
        if b_out.len() != VOCAB {
            bail!("b_out len {}", b_out.len());
        }
        Ok(crate::effective_decoder::EffectiveStep1 {
            embed,
            ah_tok: ah,
            w_out,
            b_out,
            ak: self.f32_tensor("decoder.effective_Ak").ok(),
            av: self.f32_tensor("decoder.effective_Av").ok(),
            we: self.f32_tensor("decoder.effective_We").ok(),
            wc: self.f32_tensor("decoder.effective_Wc").ok(),
            b_h: self.f32_tensor("decoder.effective_bh").ok(),
        })
    }
}

/// Resolve `model.gguf` under an ASR root (or `RLX_ASR_GGUF`).
pub fn resolve_gguf_path(root: &Path) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("RLX_ASR_GGUF") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    for name in [DEFAULT_GGUF_NAME, "asr.gguf", "rlx-asr.gguf"] {
        let p = root.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_gguf_has_units_and_aed() {
        let root = crate::asr_dir();
        let Some(path) = resolve_gguf_path(&root) else {
            return;
        };
        let g = AsrGguf::open(&path).expect("open model.gguf");
        let units = g.units().expect("rlx-asr.units");
        assert!(units.len() >= 6000, "units={}", units.len());
        assert!(g.has("silence_fbank"));
        assert!(g.has("decoder.embed"));
        let _ = g.load_effective_step1().expect("AED tensors");
        let _ = g.silence_fbank().expect("silence_fbank");
    }
}
