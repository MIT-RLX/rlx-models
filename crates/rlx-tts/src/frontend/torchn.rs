//! TorchN BPE G2P — vocab/BPE load + optional weight sidecar.
//!
//! Full quantized Transformer decode is implemented when a local
//! `g2p_seq2seq.safetensors` sidecar exists (shipped inside `rlx-tts.gguf`).
//! Without weights, this module still exposes BPE tokenization and symbol
//! tables for frontend plumbing / tests.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

use crate::ops::{layer_norm, linear_in_out, relu_inplace};

#[derive(Debug, Clone)]
pub struct SymbolTable {
    pub name: String,
    pub id_to_sym: Vec<String>,
    pub sym_to_id: HashMap<String, u32>,
}

impl SymbolTable {
    pub fn get(&self, sym: &str) -> Option<u32> {
        self.sym_to_id.get(sym).copied()
    }

    pub fn symbol(&self, id: u32) -> Option<&str> {
        self.id_to_sym.get(id as usize).map(|s| s.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct TorchnG2p {
    pub model_type: String,
    pub version: String,
    pub num_bpe: usize,
    pub input_symbols: SymbolTable,
    pub output_symbols: SymbolTable,
    pub bpe_merges: Vec<(String, String)>,
    pub weights_path: Option<PathBuf>,
    /// Number of `qat.{i}.weight` tensors in the safetensors sidecar (0 if absent).
    pub qat_count: usize,
}

impl TorchnG2p {
    /// Prefer the compact `g2p_bpe.json` sidecar (no 29 MB binary in RAM).
    /// Falls back to parsing `g2p_seq2seq.bin` only when
    /// `RLX_TTS_LOAD_TORCHN_BIN=1` is set.
    /// Callers should gate open with `RLX_TTS_LOAD_TORCHN=1` — BPE JSON alone
    /// is ~1.5 MB parsed into merge tables.
    pub fn load_prefer_sidecar(frontend_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = frontend_dir.as_ref();
        let json = dir.join("g2p_bpe.json");
        if json.is_file() {
            return Self::load_bpe_json(&json);
        }
        let bin = dir.join("g2p_seq2seq.bin");
        if bin.is_file() && std::env::var_os("RLX_TTS_LOAD_TORCHN_BIN").is_some() {
            return Self::load_bin(&bin);
        }
        bail!(
            "TorchN G2P sidecar missing ({}). Expected g2p_bpe.json from the rlx-tts GGUF/frontend bundle.",
            json.display()
        )
    }

    pub fn load_bpe_json(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        let num_bpe = v
            .get("num_bpe")
            .and_then(|x| x.as_u64())
            .unwrap_or(0) as usize;
        let merges = v
            .get("merges")
            .and_then(|x| x.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|pair| {
                        let a = pair.get(0)?.as_str()?.to_string();
                        let b = pair.get(1)?.as_str()?.to_string();
                        Some((a, b))
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let input_symbols = symbol_table_from_json(v.get("input_symbols"), "isyms")?;
        let output_symbols = symbol_table_from_json(v.get("output_symbols"), "osyms")?;
        ensure!(!merges.is_empty(), "g2p_bpe.json has no merges");
        let weights_path = {
            let side = path.with_file_name("g2p_seq2seq.safetensors");
            side.is_file().then_some(side)
        };
        let qat_count = weights_path
            .as_ref()
            .and_then(|p| count_qat_tensors(p).ok())
            .unwrap_or(0);
        Ok(Self {
            model_type: "TorchN_BPE_PyTorch_sidecar".into(),
            version: "0.2".into(),
            num_bpe: if num_bpe == 0 { merges.len() } else { num_bpe },
            input_symbols,
            output_symbols,
            bpe_merges: merges,
            weights_path,
            qat_count,
        })
    }

    pub fn load(bin_path: impl AsRef<Path>) -> Result<Self> {
        Self::load_bin(bin_path)
    }

    pub fn load_bin(bin_path: impl AsRef<Path>) -> Result<Self> {
        let bin_path = bin_path.as_ref();
        let bytes =
            std::fs::read(bin_path).with_context(|| format!("read {}", bin_path.display()))?;
        let model_type = extract_tag_string(&bytes, b"<ModelType>")
            .unwrap_or_else(|| "TorchN_BPE_PyTorch".into());
        ensure!(
            model_type.contains("TorchN") || model_type.contains("BPE"),
            "unexpected G2P model type: {model_type}"
        );
        let version = extract_after_tag(&bytes, b"<Version>").unwrap_or_else(|| "0.2".into());
        let num_bpe = parse_num_bpe(&bytes).context("NumBpe")?;
        let input_symbols = parse_symbol_table(&bytes, b"<HasInputSymbolTable>", b"isyms")
            .context("input symbol table")?;
        let output_symbols = parse_symbol_table(&bytes, b"<HasOutputSymbolTable>", b"osyms")
            .context("output symbol table")?;
        let bpe_merges = parse_bpe_merges(&bytes, num_bpe)?;
        let weights_path = {
            let side = bin_path.with_extension("safetensors");
            if side.is_file() {
                Some(side)
            } else {
                let alt = bin_path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join("g2p_seq2seq.safetensors");
                alt.is_file().then_some(alt)
            }
        };
        let qat_count = weights_path
            .as_ref()
            .and_then(|p| count_qat_tensors(p).ok())
            .unwrap_or(0);
        Ok(Self {
            model_type,
            version,
            num_bpe,
            input_symbols,
            output_symbols,
            bpe_merges,
            weights_path,
            qat_count,
        })
    }

    pub fn has_weights(&self) -> bool {
        self.qat_count > 0 && self.weights_path.is_some()
    }

    /// Encode a word to input embedding ids.
    /// Known lexicon words map 1→1; otherwise greedy BPE over `merges`.
    /// Appends `</s>` (ModelType `AddSrcEos`) unless `RLX_TTS_TORCHN_NO_EOS` is set.
    pub fn encode_word(&self, word: &str) -> Vec<u32> {
        let w = word.trim();
        if w.is_empty() {
            return Vec::new();
        }
        // Whole-word lookup (primary path — isyms is a 19k wordpiece/lexicon table).
        let mut ids = None;
        for cand in [w.to_string(), w.to_ascii_lowercase(), w.to_ascii_uppercase()] {
            if let Some(id) = self.input_symbols.get(&cand) {
                ids = Some(vec![id]);
                break;
            }
        }
        let mut ids = ids.unwrap_or_else(|| self.encode_chars_raw(w));
        self.append_src_eos(&mut ids);
        ids
    }

    /// Character-level encode (one id per Unicode scalar) + optional `</s>`.
    pub fn encode_chars(&self, word: &str) -> Vec<u32> {
        let mut ids = self.encode_chars_raw(word);
        self.append_src_eos(&mut ids);
        ids
    }

    fn encode_chars_raw(&self, word: &str) -> Vec<u32> {
        let unk = self.input_symbols.get("<unk>").unwrap_or(3);
        word.chars()
            .map(|c| {
                let s = c.to_string();
                self.input_symbols
                    .get(&s)
                    .or_else(|| self.input_symbols.get(&s.to_ascii_lowercase()))
                    .unwrap_or(unk)
            })
            .collect()
    }

    fn append_src_eos(&self, ids: &mut Vec<u32>) {
        if std::env::var_os("RLX_TTS_TORCHN_NO_EOS").is_some() || ids.is_empty() {
            return;
        }
        let eos = self
            .input_symbols
            .get("</s>")
            .or_else(|| self.input_symbols.get("<s>"))
            .unwrap_or(2);
        if ids.last().copied() != Some(eos) {
            ids.push(eos);
        }
    }

    /// Subword BPE fallback over `merges` (no ``</w>`` marker in this bundle).
    pub fn encode_bpe(&self, word: &str) -> Vec<u32> {
        let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        let merge_rank: HashMap<(String, String), usize> = self
            .bpe_merges
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i))
            .collect();
        loop {
            if chars.len() < 2 {
                break;
            }
            let mut best: Option<(usize, usize)> = None;
            for i in 0..chars.len() - 1 {
                let key = (chars[i].clone(), chars[i + 1].clone());
                if let Some(&rank) = merge_rank.get(&key) {
                    if best.map(|(_, r)| rank < r).unwrap_or(true) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", chars[i], chars[i + 1]);
            chars[i] = merged;
            chars.remove(i + 1);
        }
        let unk = self.input_symbols.get("<unk>").unwrap_or(3);
        chars
            .iter()
            .map(|s| self.input_symbols.get(s).unwrap_or(unk))
            .collect()
    }

    /// Decode output symbol ids to a compact LHP-ish string (no packing).
    pub fn decode_outputs(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if let Some(sym) = self.output_symbols.symbol(id) {
                if sym == "<s>" || sym == "</s>" || sym == "<eps>" || sym == "<unk>" {
                    continue;
                }
                out.push_str(sym);
            }
        }
        out
    }

    /// Run neural G2P when exported Softmax weights are present.
    ///
    /// returns `None` unless the greedy decode looks like a plausible LHP
    /// compound so HydraLite can fall back to the Nashville lexicon.
    pub fn pronounce(&self, word: &str) -> Result<Option<String>> {
        if !self.has_weights() {
            return Ok(None);
        }
        // Opt out of Softmax for A/B: `RLX_TTS_TORCHN_DECODE=0`.
        if std::env::var("RLX_TTS_TORCHN_DECODE").ok().as_deref() == Some("0") {
            return Ok(None);
        }
        let Some(path) = &self.weights_path else {
            return Ok(None);
        };
        let phones = probe_greedy_phones(path, self, word, 32)?;
        if phones.is_empty() {
            return Ok(None);
        }
        // Reject single-token garbage / digit-heavy Softmax misses.
        let compact = phones.join("-");
        let plausible = phones.iter().all(|p| {
            !p.is_empty()
                && p.chars()
                    .all(|c| c.is_ascii_alphabetic() || c == ':' || c == '$' || c == '^')
        }) && (phones.len() >= 2 || compact.contains(':'));
        if !plausible {
            return Ok(None);
        }
        // Prefer complete compounds that exist in the output symbol table.
        if phones.len() > 1 {
            let joined = phones.join("-");
            if self.output_symbols.get(&joined).is_none()
                && self.output_symbols.get(&format!("{joined}:")).is_none()
            {
                // Still accept if every atom is a known phone-ish token.
                let atoms_ok = phones.iter().all(|p| {
                    self.output_symbols.get(p).is_some()
                        || self.output_symbols.get(&format!("{p}-")).is_some()
                });
                if !atoms_ok {
                    return Ok(None);
                }
            }
        }
        Ok(Some(phones.join("")))
    }
}

/// One QuantizedAffineTransform layer (dequantized on the fly).
#[derive(Debug, Clone)]
pub struct QatLinear {
    pub out_features: usize,
    pub in_features: usize,
    pub scale: f32,
    pub zero_point: i32,
    /// Row-major int8 `[out, in]`.
    pub weight: Vec<i8>,
    pub bias: Vec<f32>,
}

impl QatLinear {
    /// `y = ((W_int8 - zp) / scale) @ x`-style: x `[T, in]` → y `[T, out]`.
    /// Negative `zero_point` (bundle sentinel `-1`) is treated as 0.
    pub fn forward(&self, x: &ndarray::Array2<f32>) -> Result<ndarray::Array2<f32>> {
        ensure!(
            x.ncols() == self.in_features,
            "QAT in_features {} vs x ncols {}",
            self.in_features,
            x.ncols()
        );
        ensure!(
            self.weight.len() == self.out_features * self.in_features,
            "QAT weight len"
        );
        ensure!(self.bias.len() == self.out_features, "QAT bias len");
        let scale = if self.scale.abs() < 1e-12 {
            1.0
        } else {
            self.scale
        };
        let inv = 1.0 / scale;
        // Bundle stores `zp = -1` on all 256×256 QATs (94) and Softmax.
        // Wide FFN QATs (42) store `zp = -4`. Default: any negative zp → 0
        // (sentinel). `RLX_TTS_TORCHN_ZP_LITERAL=1` keeps the stored value;
        // `RLX_TTS_TORCHN_ZP_FFN_LITERAL=1` keeps only zp≤-4 (FFN) literal.
        let zp = if std::env::var_os("RLX_TTS_TORCHN_ZP_LITERAL").is_some() {
            self.zero_point as f32
        } else if std::env::var_os("RLX_TTS_TORCHN_ZP_FFN_LITERAL").is_some()
            && self.zero_point <= -4
        {
            self.zero_point as f32
        } else if self.zero_point < 0 {
            0.0
        } else {
            self.zero_point as f32
        };
        // Bundle stores row-major `[out, in]`. `RLX_TTS_TORCHN_QAT_TRANSPOSE=1`
        // treats storage as `[in, out]` instead (A/B).
        let transpose = std::env::var_os("RLX_TTS_TORCHN_QAT_TRANSPOSE").is_some();
        let mut w = vec![0.0f32; self.in_features * self.out_features];
        if transpose {
            for i in 0..self.in_features {
                for o in 0..self.out_features {
                    let q = self.weight[i * self.out_features + o] as f32;
                    w[i * self.out_features + o] = (q - zp) * inv;
                }
            }
        } else {
            for o in 0..self.out_features {
                for i in 0..self.in_features {
                    let q = self.weight[o * self.in_features + i] as f32;
                    w[i * self.out_features + o] = (q - zp) * inv;
                }
            }
        }
        Ok(linear_in_out(
            x,
            &w,
            self.in_features,
            self.out_features,
            Some(&self.bias),
        ))
    }
}

#[derive(Debug, Clone)]
pub struct TorchnLayerNorm {
    pub eps: f32,
    /// Kaldi `UnbiasedVar T` — this TorchN bundle uses unbiased variance.
    pub unbiased: bool,
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
}

impl TorchnLayerNorm {
    pub fn forward(&self, x: &ndarray::Array2<f32>) -> Result<ndarray::Array2<f32>> {
        ensure!(x.ncols() == self.gamma.len(), "LN width");
        Ok(layer_norm(
            x,
            &self.gamma,
            &self.beta,
            self.eps,
            self.unbiased,
        ))
    }
}

/// Load a single QAT + optional LN from the safetensors sidecar (streaming header).
pub fn load_qat(path: &Path, index: usize) -> Result<QatLinear> {
    let tensors = read_named_tensors(
        path,
        &[
            format!("qat.{index}.weight"),
            format!("qat.{index}.bias"),
            format!("qat.{index}.scale"),
            format!("qat.{index}.zero_point"),
        ],
    )?;
    let (w_raw, w_shape) = tensors
        .get(&format!("qat.{index}.weight"))
        .context("weight")?;
    let (b_raw, _) = tensors.get(&format!("qat.{index}.bias")).context("bias")?;
    let (s_raw, _) = tensors.get(&format!("qat.{index}.scale")).context("scale")?;
    let (z_raw, _) = tensors
        .get(&format!("qat.{index}.zero_point"))
        .context("zp")?;
    ensure!(w_shape.len() == 2, "QAT weight rank");
    let out_f = w_shape[0];
    let in_f = w_shape[1];
    let weight = decode_i8(w_raw)?;
    let bias = decode_f32_bytes(b_raw)?;
    let scale = decode_f32_bytes(s_raw)?[0];
    let zero_point = decode_i32(z_raw)?[0];
    Ok(QatLinear {
        out_features: out_f,
        in_features: in_f,
        scale,
        zero_point,
        weight,
        bias,
    })
}

pub fn load_ln(path: &Path, index: usize) -> Result<TorchnLayerNorm> {
    let tensors = read_named_tensors(
        path,
        &[
            format!("ln.{index}.gamma"),
            format!("ln.{index}.beta"),
            format!("ln.{index}.eps"),
        ],
    )?;
    let gamma = decode_f32_bytes(
        &tensors
            .get(&format!("ln.{index}.gamma"))
            .context("gamma")?
            .0,
    )?;
    let beta = decode_f32_bytes(&tensors.get(&format!("ln.{index}.beta")).context("beta")?.0)?;
    let eps = decode_f32_bytes(&tensors.get(&format!("ln.{index}.eps")).context("eps")?.0)?[0];
    // Bundle LayerNorms all use `<UnbiasedVar> T` (see export meta).
    Ok(TorchnLayerNorm {
        eps,
        unbiased: true,
        gamma,
        beta,
    })
}

#[cfg(test)]
/// FFN block smoke: LN → up(1024) → ReLU → down(256), matching exported shapes.
pub fn probe_ffn_block(path: &Path, up_qat: usize, down_qat: usize, ln: usize) -> Result<f32> {
    let up = load_qat(path, up_qat)?;
    let down = load_qat(path, down_qat)?;
    let ln = load_ln(path, ln)?;
    ensure!(up.in_features == 256 && up.out_features == 1024, "up shape");
    ensure!(down.in_features == 1024 && down.out_features == 256, "down shape");
    let x = ndarray::Array2::<f32>::ones((2, 256));
    let h = ln.forward(&x)?;
    let mut h = up.forward(&h)?;
    relu_inplace(&mut h);
    let y = down.forward(&h)?;
    ensure!(y.dim() == (2, 256), "ffn out");
    let mean = y.iter().sum::<f32>() / y.len() as f32;
    ensure!(mean.is_finite(), "ffn non-finite");
    Ok(mean)
}

/// One SelfAttention block (Key/Value/Query/Output QAT + 2-head MHA).
#[derive(Debug, Clone)]
pub struct TorchnSelfAttention {
    pub key: QatLinear,
    pub value: QatLinear,
    pub query: QatLinear,
    pub output: QatLinear,
    pub num_heads: usize,
    pub scale: f32,
    pub add_query: bool,
}

impl TorchnSelfAttention {
    /// Load first encoder block: qat 0=K, 1=V, 2=Q, 3=O.
    #[allow(dead_code)]
    pub fn load_block0(path: &Path) -> Result<Self> {
        Ok(Self {
            key: load_qat(path, 0)?,
            value: load_qat(path, 1)?,
            query: load_qat(path, 2)?,
            output: load_qat(path, 3)?,
            num_heads: 2,
            scale: 1.0 / (128.0f32).sqrt(),
            add_query: true,
        })
    }

    /// `x: [T, 256]` → `[T, 256]`.
    ///
    /// `AddQuery=T` default: `O(attn) + Q` (encode Softmax gold ranks beat
    /// `O(attn+Q)` by ~60× on `hello`). `RLX_TTS_TORCHN_ADDQ_BEFORE_O=1`
    /// restores `O(attn+Q)`. `RLX_TTS_TORCHN_ADDQ_INPUT=1`: `O(attn) + x`.
    pub fn forward(&self, x: &ndarray::Array2<f32>) -> Result<ndarray::Array2<f32>> {
        let (t, d) = x.dim();
        ensure!(d == 256, "expected d_model 256");
        ensure!(self.num_heads > 0 && d % self.num_heads == 0, "heads");
        let hdim = d / self.num_heads;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;
        let q = self.query.forward(x)?;
        let mut ctx = ndarray::Array2::<f32>::zeros((t, d));
        for head in 0..self.num_heads {
            let hs = head * hdim;
            let mut scores = ndarray::Array2::<f32>::zeros((t, t));
            for i in 0..t {
                for j in 0..t {
                    let mut dot = 0.0f32;
                    for c in 0..hdim {
                        dot += q[[i, hs + c]] * k[[j, hs + c]];
                    }
                    scores[[i, j]] = dot * self.scale;
                }
            }
            for i in 0..t {
                let mut row = scores.row_mut(i);
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                let inv = 1.0 / sum.max(1e-12);
                for v in row.iter_mut() {
                    *v *= inv;
                }
            }
            for i in 0..t {
                for c in 0..hdim {
                    let mut acc = 0.0f32;
                    for j in 0..t {
                        acc += scores[[i, j]] * v[[j, hs + c]];
                    }
                    ctx[[i, hs + c]] = acc;
                }
            }
        }
        let no_addq = std::env::var_os("RLX_TTS_TORCHN_NO_ADDQ").is_some();
        let addq_before_o = std::env::var_os("RLX_TTS_TORCHN_ADDQ_BEFORE_O").is_some();
        let addq_input = std::env::var_os("RLX_TTS_TORCHN_ADDQ_INPUT").is_some();
        let addq_both = std::env::var_os("RLX_TTS_TORCHN_ADDQ_BOTH").is_some();
        if self.add_query && !no_addq && addq_before_o {
            ctx = ctx + &q;
        }
        let mut y = self.output.forward(&ctx)?;
        if self.add_query && !no_addq {
            if addq_both {
                // `O(attn[+Q]) + Q + x`
                y = y + &q + x;
            } else if addq_input {
                y = y + x;
            } else if !addq_before_o {
                // Default: `O(attn) + Q`
                y = y + &q;
            }
            // `ADDQ_BEFORE_O` alone: already folded into `ctx` before `O`.
        }
        Ok(y)
    }
}

#[cfg(test)]
/// Encoder block 0 quick check (file-order path via `TorchnEncoderBlock`).
pub fn probe_encoder_block0(path: &Path) -> Result<f32> {
    let block = TorchnEncoderBlock::load(path, 0)?;
    let x = ndarray::Array2::<f32>::from_elem((4, 256), 0.1);
    let y = block.forward(&x)?;
    let mean = y.iter().sum::<f32>() / y.len() as f32;
    ensure!(mean.is_finite(), "encoder block non-finite mean={mean}");
    let peak = y.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()));
    ensure!(peak < 1e6, "encoder block peak={peak}");
    Ok(mean)
}

/// CompressedWordVec embedding: uint8 table with affine dequant
/// `lo + u8 * (hi-lo)/255` (Kaldi kOneByte-style; not signed int8+128).
#[derive(Debug, Clone)]
pub struct TorchnEmbedding {
    pub vocab: usize,
    pub dim: usize,
    pub lo: f32,
    pub hi: f32,
    /// Raw bytes (safetensors may label `I8`/`U8`; always treated as u8).
    pub weight: Vec<u8>,
}

/// EncPos/DecPos codebook. Prefer decompressed `F32` rows from export.
#[derive(Debug, Clone)]
pub struct TorchnCodebook {
    pub rows: usize,
    pub dim: usize,
    pub floats: Option<Vec<f32>>,
    pub lo: f32,
    pub hi: f32,
    pub weight: Vec<u8>,
}

impl TorchnCodebook {
    pub fn load(path: &Path) -> Result<Option<Self>> {
        let f32_names = ["embed.codebook.weight".to_string()];
        if let Ok(tensors) = read_named_tensors(path, &f32_names) {
            let (w_raw, shape) = tensors.get("embed.codebook.weight").context("cb weight")?;
            if shape.len() == 2 && w_raw.len() == shape[0] * shape[1] * 4 {
                return Ok(Some(Self {
                    rows: shape[0],
                    dim: shape[1],
                    floats: Some(decode_f32_bytes(w_raw)?),
                    lo: -1.0,
                    hi: 1.0,
                    weight: Vec::new(),
                }));
            }
        }
        let names = [
            "embed.codebook.weight".to_string(),
            "embed.codebook.scale_mid".to_string(),
            "embed.codebook.scale_hi".to_string(),
        ];
        let Ok(tensors) = read_named_tensors(path, &names) else {
            return Ok(None);
        };
        let (w_raw, shape) = tensors.get("embed.codebook.weight").context("cb weight")?;
        ensure!(shape.len() == 2, "codebook rank");
        let mid = decode_f32_bytes(&tensors.get("embed.codebook.scale_mid").context("mid")?.0)?[0];
        let hi = decode_f32_bytes(&tensors.get("embed.codebook.scale_hi").context("hi")?.0)?[0];
        Ok(Some(Self {
            rows: shape[0],
            dim: shape[1],
            floats: None,
            lo: mid,
            hi,
            weight: decode_u8(w_raw)?,
        }))
    }

    pub fn lookup_row(&self, pos: usize) -> Result<ndarray::Array1<f32>> {
        let idx = pos.min(self.rows.saturating_sub(1));
        if let Some(ref floats) = self.floats {
            let row = &floats[idx * self.dim..(idx + 1) * self.dim];
            return Ok(ndarray::Array1::from_vec(row.to_vec()));
        }
        let scale = (self.hi - self.lo) / 255.0;
        let row = &self.weight[idx * self.dim..(idx + 1) * self.dim];
        let mut out = ndarray::Array1::<f32>::zeros(self.dim);
        for (c, &q) in row.iter().enumerate() {
            out[c] = self.lo + q as f32 * scale;
        }
        Ok(out)
    }

    pub fn add_positions(&self, x: &mut ndarray::Array2<f32>) -> Result<()> {
        ensure!(x.ncols() == self.dim, "codebook dim");
        for t in 0..x.nrows() {
            let pe = self.lookup_row(t)?;
            for c in 0..self.dim {
                x[[t, c]] += pe[c];
            }
        }
        Ok(())
    }
}

impl TorchnEmbedding {
    pub fn load_input(path: &Path) -> Result<Self> {
        let tensors = read_named_tensors(
            path,
            &[
                "embed.input.weight".into(),
                "embed.input.scale_lo".into(),
                "embed.input.scale_hi".into(),
            ],
        )?;
        let (w_raw, shape) = tensors.get("embed.input.weight").context("input weight")?;
        ensure!(shape.len() == 2, "embed rank");
        let vocab = shape[0];
        let dim = shape[1];
        let lo = decode_f32_bytes(&tensors.get("embed.input.scale_lo").context("lo")?.0)?[0];
        let hi = decode_f32_bytes(&tensors.get("embed.input.scale_hi").context("hi")?.0)?[0];
        Ok(Self {
            vocab,
            dim,
            lo,
            hi,
            weight: decode_u8(w_raw)?,
        })
    }

    pub fn lookup(&self, ids: &[u32]) -> Result<ndarray::Array2<f32>> {
        let scale = (self.hi - self.lo) / 255.0;
        let mut out = ndarray::Array2::<f32>::zeros((ids.len(), self.dim));
        for (r, &id) in ids.iter().enumerate() {
            let idx = (id as usize).min(self.vocab.saturating_sub(1));
            let row = &self.weight[idx * self.dim..(idx + 1) * self.dim];
            for (c, &q) in row.iter().enumerate() {
                out[[r, c]] = self.lo + q as f32 * scale;
            }
        }
        Ok(out)
    }
}

/// One encoder block (SelfAttn + FFN) with residual + LayerNorm.
#[derive(Debug, Clone)]
pub struct TorchnEncoderBlock {
    pub attn: TorchnSelfAttention,
    pub ln_attn: TorchnLayerNorm,
    pub up: QatLinear,
    pub down: QatLinear,
    pub ln_ffn: TorchnLayerNorm,
}

impl TorchnEncoderBlock {
    pub fn load(path: &Path, block: usize) -> Result<Self> {
        let base = block * 6;
        let ln_base = block * 2;
        Ok(Self {
            attn: TorchnSelfAttention {
                key: load_qat(path, base)?,
                value: load_qat(path, base + 1)?,
                query: load_qat(path, base + 2)?,
                output: load_qat(path, base + 3)?,
                num_heads: 2,
                scale: 1.0 / (128.0f32).sqrt(),
                add_query: true,
            },
            ln_attn: load_ln(path, ln_base)?,
            up: load_qat(path, base + 4)?,
            down: load_qat(path, base + 5)?,
            ln_ffn: load_ln(path, ln_base + 1)?,
        })
    }

    pub fn forward(&self, x: &ndarray::Array2<f32>) -> Result<ndarray::Array2<f32>> {
        // Product default: classic pre-norm residuals (best Softmax ranks so far).
        // Opt out with `RLX_TTS_TORCHN_ENC_POSTNORM=1` for file-order A/B.
        // `RLX_TTS_TORCHN_ENC_RES_SCALE=s`: multiply attn/FFN branches by s before add.
        let res_scale: f32 = std::env::var("RLX_TTS_TORCHN_ENC_RES_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        // Hybrid: pre-norm attention residual + file-order Parallel FFN post-norm.
        // `RLX_TTS_TORCHN_ENC_HYBRID=1`
        if std::env::var_os("RLX_TTS_TORCHN_ENC_HYBRID").is_some() {
            let n = self.ln_attn.forward(x)?;
            let mut a = self.attn.forward(&n)?;
            if (res_scale - 1.0).abs() > 1e-8 {
                a.mapv_inplace(|v| v * res_scale);
            }
            let h = x + a;
            let mut ff = self.up.forward(&h)?;
            relu_inplace(&mut ff);
            let mut ff = self.down.forward(&ff)?;
            if (res_scale - 1.0).abs() > 1e-8 {
                ff.mapv_inplace(|v| v * res_scale);
            }
            return self.ln_ffn.forward(&(h + ff));
        }
        // Product default: classic pre-norm. Opt out: `RLX_TTS_TORCHN_ENC_POSTNORM=1`.
        if std::env::var_os("RLX_TTS_TORCHN_ENC_POSTNORM").is_none() {
            let n = self.ln_attn.forward(x)?;
            let mut a = self.attn.forward(&n)?;
            if (res_scale - 1.0).abs() > 1e-8 {
                a.mapv_inplace(|v| v * res_scale);
            }
            let h = x + a;
            let n = self.ln_ffn.forward(&h)?;
            let mut ff = self.up.forward(&n)?;
            relu_inplace(&mut ff);
            let mut ff = self.down.forward(&ff)?;
            if (res_scale - 1.0).abs() > 1e-8 {
                ff.mapv_inplace(|v| v * res_scale);
            }
            let y = h + ff;
            return Ok(y);
        }
        let mut h = self.attn.forward(x)?;
        if std::env::var_os("RLX_TTS_TORCHN_ENC_ATTN_RES").is_some() {
            if (res_scale - 1.0).abs() > 1e-8 {
                h.mapv_inplace(|v| v * res_scale);
            }
            h = h + x;
        }
        let h = self.ln_attn.forward(&h)?;
        let mut ff = self.up.forward(&h)?;
        relu_inplace(&mut ff);
        let mut ff = self.down.forward(&ff)?;
        if (res_scale - 1.0).abs() > 1e-8 {
            ff.mapv_inplace(|v| v * res_scale);
        }
        let y = &h + &ff;
        self.ln_ffn.forward(&y)
    }
}

/// Full 16-block encoder.
pub struct TorchnEncoder {
    pub blocks: Vec<TorchnEncoderBlock>,
}

impl TorchnEncoder {
    pub fn load(path: &Path) -> Result<Self> {
        let mut blocks = Vec::with_capacity(16);
        for i in 0..16 {
            blocks.push(TorchnEncoderBlock::load(path, i)?);
        }
        Ok(Self { blocks })
    }

    pub fn forward(&self, mut x: ndarray::Array2<f32>) -> Result<ndarray::Array2<f32>> {
        // Product default: pre-norm residuals (ENC_PRENORM). Opt out with
        // `RLX_TTS_TORCHN_ENC_POSTNORM=1` for file-order A/B.
        let n = std::env::var("RLX_TTS_TORCHN_ENC_LAYERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(8) // Softmax ranks peak around L=8; full 16 collapses.
            .min(self.blocks.len());
        for b in self.blocks.iter().take(n) {
            x = b.forward(&x)?;
        }
        // Optional encoder-output renormalization for prenorm magnitude mismatch.
        // `RLX_TTS_TORCHN_ENC_OUT_SCALE=s` multiplies; `ENC_OUT_UNIT=1` L2-normalizes
        // each row to unit length then scales by `ENC_OUT_SCALE` (default 13).
        if std::env::var_os("RLX_TTS_TORCHN_ENC_OUT_UNIT").is_some() {
            let target: f32 = std::env::var("RLX_TTS_TORCHN_ENC_OUT_SCALE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(13.0);
            for mut row in x.rows_mut() {
                let n = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                let s = target / n;
                row.mapv_inplace(|v| v * s);
            }
        } else if let Ok(s) = std::env::var("RLX_TTS_TORCHN_ENC_OUT_SCALE") {
            if let Ok(scale) = s.parse::<f32>() {
                x.mapv_inplace(|v| v * scale);
            }
        }
        Ok(x)
    }
}

#[cfg(test)]
/// Encode BPE ids through input embed + 16 encoder blocks; returns hidden mean.
pub fn probe_encode_ids(path: &Path, ids: &[u32]) -> Result<(f32, usize)> {
    let emb = TorchnEmbedding::load_input(path)?;
    let enc = TorchnEncoder::load(path)?;
    let x = emb.lookup(ids)?;
    let y = enc.forward(x)?;
    let mean = y.iter().sum::<f32>() / y.len() as f32;
    ensure!(mean.is_finite(), "full encoder non-finite");
    Ok((mean, y.nrows()))
}

impl TorchnEmbedding {
    pub fn load_output(path: &Path) -> Result<Self> {
        let tensors = read_named_tensors(
            path,
            &[
                "embed.output.weight".into(),
                "embed.output.scale_lo".into(),
                "embed.output.scale_hi".into(),
            ],
        )?;
        let (w_raw, shape) = tensors.get("embed.output.weight").context("output weight")?;
        ensure!(shape.len() == 2, "embed rank");
        Ok(Self {
            vocab: shape[0],
            dim: shape[1],
            lo: decode_f32_bytes(&tensors.get("embed.output.scale_lo").context("lo")?.0)?[0],
            hi: decode_f32_bytes(&tensors.get("embed.output.scale_hi").context("hi")?.0)?[0],
            weight: decode_u8(w_raw)?,
        })
    }

    /// Project hidden `[T, dim]` to vocab logits via dequantized embedding rows.
    pub fn project(&self, h: &ndarray::Array2<f32>) -> Result<ndarray::Array2<f32>> {
        ensure!(h.ncols() == self.dim, "proj dim");
        let scale = (self.hi - self.lo) / 255.0;
        let (t, _) = h.dim();
        let mut logits = ndarray::Array2::<f32>::zeros((t, self.vocab));
        for v in 0..self.vocab {
            let row = &self.weight[v * self.dim..(v + 1) * self.dim];
            for i in 0..t {
                let mut dot = 0.0f32;
                for c in 0..self.dim {
                    let e = self.lo + row[c] as f32 * scale;
                    dot += h[[i, c]] * e;
                }
                logits[[i, v]] = dot;
            }
        }
        Ok(logits)
    }
}

/// Decoder block: AverageFfn → LN → cross-MHA → LN → Parallel FFN → LN,
/// wrapped as `Parallel(Identity, …)`; `forward` uses pre-norm residuals.
#[derive(Debug, Clone)]
pub struct TorchnDecoderBlock {
    pub avg_up: QatLinear,
    pub avg_down: QatLinear,
    pub ln_avg: TorchnLayerNorm,
    pub query: QatLinear,
    pub key: QatLinear,
    pub value: QatLinear,
    pub output: QatLinear,
    pub ln_attn: TorchnLayerNorm,
    pub up: QatLinear,
    pub down: QatLinear,
    pub ln_ffn: TorchnLayerNorm,
    pub num_heads: usize,
    /// Decoder block 1 is `SupervisedMultiHeadAttention` with `SupervisedHeads=1`.
    pub supervised_heads: usize,
    pub scale: f32,
    pub add_query: bool,
}

impl TorchnDecoderBlock {
    /// Decoder block `i` starts at QAT `96 + i*8`, LN `32 + i*3`.
    /// AverageAttention: two 256×256 QATs + ReLU (not the 1024-wide FFN).
    pub fn load(path: &Path, block: usize) -> Result<Self> {
        let base = 96 + block * 8;
        let ln = 32 + block * 3;
        Ok(Self {
            avg_up: load_qat(path, base)?,
            avg_down: load_qat(path, base + 1)?,
            ln_avg: load_ln(path, ln)?,
            // After MHA tag: Query → Key → Value → Output (stack trails).
            query: load_qat(path, base + 2)?,
            key: load_qat(path, base + 3)?,
            value: load_qat(path, base + 4)?,
            output: load_qat(path, base + 5)?,
            ln_attn: load_ln(path, ln + 1)?,
            up: load_qat(path, base + 6)?,
            down: load_qat(path, base + 7)?,
            ln_ffn: load_ln(path, ln + 2)?,
            num_heads: 2,
            // Block 1: `<SupervisedMultiHeadAttention><SupervisedHeads> 1`.
            supervised_heads: if block == 1 { 1 } else { 0 },
            scale: 1.0 / (128.0f32).sqrt(),
            add_query: true,
        })
    }

    pub fn forward(
        &self,
        x: &ndarray::Array2<f32>,
        memory: &ndarray::Array2<f32>,
    ) -> Result<ndarray::Array2<f32>> {
        // File order (confirmed via `</AverageAttention>` / `</ParallelComponent>`):
        //   Parallel(Identity, AverageAttention=AvgFfn only) → LN
        //   → cross-MHA (replaces; no residual) → LN
        //   → Parallel(Identity, wide FFN) → LN
        // `RLX_TTS_TORCHN_DEC_PRENORM=1` keeps the older pre-norm+cross-residual path.
        if std::env::var_os("RLX_TTS_TORCHN_DEC_PRENORM").is_some() {
            let n = self.ln_avg.forward(x)?;
            let mut h = self.avg_up.forward(&n)?;
            relu_inplace(&mut h);
            let h = self.avg_down.forward(&h)?;
            let h = x + h;

            let n = self.ln_attn.forward(&h)?;
            let ctx = self.cross_attn(&n, memory)?;
            let h = h + ctx;

            let n = self.ln_ffn.forward(&h)?;
            let mut ff = self.up.forward(&n)?;
            relu_inplace(&mut ff);
            let ff = self.down.forward(&ff)?;
            return Ok(h + ff);
        }

        let mut af = self.avg_up.forward(x)?;
        relu_inplace(&mut af);
        let af = self.avg_down.forward(&af)?;
        let h = self.ln_avg.forward(&(x + af))?;

        // Cross-MHA replaces the stream (not Parallel-wrapped in the bin).
        let h = self.cross_attn(&h, memory)?;
        let h = self.ln_attn.forward(&h)?;

        let mut ff = self.up.forward(&h)?;
        relu_inplace(&mut ff);
        let ff = self.down.forward(&ff)?;
        self.ln_ffn.forward(&(h + ff))
    }

    fn cross_attn(
        &self,
        h: &ndarray::Array2<f32>,
        memory: &ndarray::Array2<f32>,
    ) -> Result<ndarray::Array2<f32>> {
        let q = self.query.forward(h)?;
        let k = self.key.forward(memory)?;
        let v = self.value.forward(memory)?;
        let (t, d) = h.dim();
        let (tm, _) = memory.dim();
        let hdim = d / self.num_heads;
        let mut ctx = ndarray::Array2::<f32>::zeros((t, d));
        // AlignModel / ShiftedAlignments: supervised heads hard-attend to a
        // source position. Default: clamp decoder step → source index (shifted
        // by 0). `RLX_TTS_TORCHN_ALIGN_SHIFT=1` uses i→min(i+1,tm-1).
        // `RLX_TTS_TORCHN_ALIGN_WORD=1` always attends to source row 0.
        let align_shift: usize = std::env::var("RLX_TTS_TORCHN_ALIGN_SHIFT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let align_word = std::env::var_os("RLX_TTS_TORCHN_ALIGN_WORD").is_some();
        for head in 0..self.num_heads {
            let hs = head * hdim;
            let supervised = head < self.supervised_heads
                && std::env::var_os("RLX_TTS_TORCHN_NO_SUPERVISED").is_none();
            if supervised {
                for i in 0..t {
                    let src = if align_word {
                        0usize
                    } else {
                        (i + align_shift).min(tm.saturating_sub(1))
                    };
                    for c in 0..hdim {
                        ctx[[i, hs + c]] = v[[src, hs + c]];
                    }
                }
                continue;
            }
            let mut scores = ndarray::Array2::<f32>::zeros((t, tm));
            for i in 0..t {
                for j in 0..tm {
                    let mut dot = 0.0f32;
                    for c in 0..hdim {
                        dot += q[[i, hs + c]] * k[[j, hs + c]];
                    }
                    scores[[i, j]] = dot * self.scale;
                }
            }
            for i in 0..t {
                let mut row = scores.row_mut(i);
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in row.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum.max(1e-12);
                for s in row.iter_mut() {
                    *s *= inv;
                }
            }
            for i in 0..t {
                for c in 0..hdim {
                    let mut acc = 0.0f32;
                    for j in 0..tm {
                        acc += scores[[i, j]] * v[[j, hs + c]];
                    }
                    ctx[[i, hs + c]] = acc;
                }
            }
        }
        let no_addq = std::env::var_os("RLX_TTS_TORCHN_NO_ADDQ").is_some();
        let addq_before_o = std::env::var_os("RLX_TTS_TORCHN_ADDQ_BEFORE_O").is_some();
        let addq_input = std::env::var_os("RLX_TTS_TORCHN_ADDQ_INPUT").is_some();
        let addq_both = std::env::var_os("RLX_TTS_TORCHN_ADDQ_BOTH").is_some();
        if self.add_query && !no_addq && addq_before_o {
            ctx = ctx + &q;
        }
        let mut y = self.output.forward(&ctx)?;
        if self.add_query && !no_addq {
            if addq_both {
                y = y + &q + h;
            } else if addq_input {
                y = y + h;
            } else if !addq_before_o {
                y = y + &q;
            }
        }
        Ok(y)
    }
}

#[cfg(test)]
pub fn probe_greedy_debug(
    path: &Path,
    g2p: &TorchnG2p,
    word: &str,
    max_phones: usize,
) -> Result<(Vec<(f32, u32, String)>, Vec<String>)> {
    let ids = if std::env::var_os("RLX_TTS_TORCHN_CHARS").is_some() {
        g2p.encode_chars(word)
    } else {
        g2p.encode_word(word)
    };
    ensure!(!ids.is_empty(), "empty encode");
    let in_emb = TorchnEmbedding::load_input(path)?;
    let out_emb = TorchnEmbedding::load_output(path)?;
    let proj = TorchnSoftmaxProj::load(path)?;
    let mut x = in_emb.lookup(&ids)?;
    if std::env::var_os("RLX_TTS_TORCHN_NO_ENCPOS").is_none() {
        if let Some(cb) = TorchnCodebook::load(path)? {
            cb.add_positions(&mut x)?;
        }
    }
    let bos = g2p.output_symbols.get("<s>").unwrap_or(2);
    let (h, tag) = if std::env::var_os("RLX_TTS_TORCHN_ENCODE_ONLY").is_some() {
        let x = if std::env::var_os("RLX_TTS_TORCHN_EMBED_ONLY").is_some() {
            x
        } else {
            TorchnEncoder::load(path)?.forward(x)?
        };
        let (t, d) = x.dim();
        let mut mean = ndarray::Array2::<f32>::zeros((1, d));
        for c in 0..d {
            let mut s = 0.0f32;
            for r in 0..t {
                s += x[[r, c]];
            }
            mean[[0, c]] = s / t.max(1) as f32;
        }
        (mean, format!("encode_only_t={t}"))
    } else {
        let enc = TorchnEncoder::load(path)?;
        let memory = enc.forward(x)?;
        let mem_mean = memory.iter().sum::<f32>() / memory.len() as f32;
        let mut dec_blocks = Vec::with_capacity(5);
        for i in 0..5 {
            dec_blocks.push(TorchnDecoderBlock::load(path, i)?);
        }
        let mut h = out_emb.lookup(&[bos])?;
        if std::env::var_os("RLX_TTS_TORCHN_NO_ENCPOS").is_none() {
            if let Some(cb) = TorchnCodebook::load(path)? {
                cb.add_positions(&mut h)?;
            }
        }
        for b in &dec_blocks {
            h = b.forward(&h, &memory)?;
        }
        (h, format!("mem_mean={mem_mean:.4} t={}", memory.nrows()))
    };
    let logits = if let Some(ref proj) = proj {
        proj.project(&h)?
    } else {
        out_emb.project(&h)?
    };
    let row = logits.row(0);
    let mut scored: Vec<(f32, usize)> = row.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut top = Vec::new();
    for (v, i) in scored.into_iter().take(5) {
        let sym = g2p
            .output_symbols
            .symbol(i as u32)
            .unwrap_or("?")
            .to_string();
        top.push((v, i as u32, sym));
    }
    top.push((0.0, 0, tag));
    let phones = probe_greedy_phones(path, g2p, word, max_phones)?;
    Ok((top, phones))
}

/// Untied Softmax projection (`Quantized8BitLinearTransform`).
#[derive(Debug, Clone)]
pub struct TorchnSoftmaxProj {
    pub out_features: usize,
    pub in_features: usize,
    pub scale: f32,
    pub zero_point: i32,
    pub weight: Vec<i8>,
}

impl TorchnSoftmaxProj {
    pub fn load(path: &Path) -> Result<Option<Self>> {
        let names = [
            "proj.weight".to_string(),
            "proj.scale".to_string(),
            "proj.zero_point".to_string(),
        ];
        let Ok(tensors) = read_named_tensors(path, &names) else {
            return Ok(None);
        };
        let (w_raw, shape) = tensors.get("proj.weight").context("proj.weight")?;
        ensure!(shape.len() == 2, "proj rank");
        Ok(Some(Self {
            out_features: shape[0],
            in_features: shape[1],
            scale: decode_f32_bytes(&tensors.get("proj.scale").context("scale")?.0)?[0],
            zero_point: decode_i32(&tensors.get("proj.zero_point").context("zp")?.0)?[0],
            weight: decode_i8(w_raw)?,
        }))
    }

    pub fn project(&self, h: &ndarray::Array2<f32>) -> Result<ndarray::Array2<f32>> {
        ensure!(h.ncols() == self.in_features, "proj in dim");
        ensure!(
            self.weight.len() == self.out_features * self.in_features,
            "proj weight len"
        );
        let scale = if self.scale.abs() < 1e-12 {
            1.0
        } else {
            self.scale
        };
        let inv = 1.0 / scale;
        let zp = if self.zero_point < 0 {
            0.0
        } else {
            self.zero_point as f32
        };
        let mut w = vec![0.0f32; self.in_features * self.out_features];
        for o in 0..self.out_features {
            for i in 0..self.in_features {
                let q = self.weight[o * self.in_features + i] as f32;
                w[i * self.out_features + o] = (q - zp) * inv;
            }
        }
        Ok(linear_in_out(
            h,
            &w,
            self.in_features,
            self.out_features,
            None,
        ))
    }
}

///
/// `RLX_TTS_TORCHN_ENCODE_ONLY=1` skips the decoder and projects encoder
/// memory (mean over tokens) through Softmax.
/// `RLX_TTS_TORCHN_WORD_MEM=1` keeps only the first encoder row as memory
/// (drop `</s>`), matching AlignModel probes where eos-attention hurts ranks.
/// Softmax argmax skips `@@` BPE continuations.
pub fn probe_greedy_phones(
    path: &Path,
    g2p: &TorchnG2p,
    word: &str,
    max_phones: usize,
) -> Result<Vec<String>> {
    let ids = if std::env::var_os("RLX_TTS_TORCHN_CHARS").is_some() {
        g2p.encode_chars(word)
    } else {
        g2p.encode_word(word)
    };
    ensure!(!ids.is_empty(), "empty encode");
    let in_emb = TorchnEmbedding::load_input(path)?;
    let out_emb = TorchnEmbedding::load_output(path)?;
    let proj = TorchnSoftmaxProj::load(path)?;
    let mut x = in_emb.lookup(&ids)?;
    // EncPos is always part of input Parallel in the bin; opt out with NO_ENCPOS.
    if std::env::var_os("RLX_TTS_TORCHN_NO_ENCPOS").is_none() {
        if let Some(cb) = TorchnCodebook::load(path)? {
            cb.add_positions(&mut x)?;
        }
    }

    let bos = g2p.output_symbols.get("<s>").unwrap_or(2);
    let eos = g2p.output_symbols.get("</s>").unwrap_or(bos);

    if std::env::var_os("RLX_TTS_TORCHN_ENCODE_ONLY").is_some() {
        let x = if std::env::var_os("RLX_TTS_TORCHN_EMBED_ONLY").is_some() {
            x
        } else {
            TorchnEncoder::load(path)?.forward(x)?
        };
        let (t, d) = x.dim();
        let mut mean = ndarray::Array2::<f32>::zeros((1, d));
        for c in 0..d {
            let mut s = 0.0f32;
            for r in 0..t {
                s += x[[r, c]];
            }
            mean[[0, c]] = s / t.max(1) as f32;
        }
        let logits = if let Some(ref proj) = proj {
            proj.project(&mean)?
        } else {
            out_emb.project(&mean)?
        };
        let row = logits.row(0);
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if is_skip_out_token(g2p, i as u32, bos) {
                continue;
            }
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        return Ok(phones_from_out_ids(g2p, &[best as u32], bos));
    }

    let enc = TorchnEncoder::load(path)?;
    let mut memory = enc.forward(x)?;
    // Product default: word-only memory (drop `</s>`). Opt out: `RLX_TTS_TORCHN_FULL_MEM=1`.
    if std::env::var_os("RLX_TTS_TORCHN_FULL_MEM").is_none() && memory.nrows() > 1 {
        memory = memory.slice(ndarray::s![0..1, ..]).to_owned();
    }
    let n_dec = std::env::var("RLX_TTS_TORCHN_DEC_LAYERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5)
        .min(5);
    let mut dec_blocks = Vec::with_capacity(n_dec);
    for i in 0..n_dec {
        dec_blocks.push(TorchnDecoderBlock::load(path, i)?);
    }
    let mut h = out_emb.lookup(&[bos])?;
    if std::env::var_os("RLX_TTS_TORCHN_NO_ENCPOS").is_none() {
        if let Some(cb) = TorchnCodebook::load(path)? {
            cb.add_positions(&mut h)?;
        }
    }
    let mut out_ids: Vec<u32> = Vec::new();
    for _ in 0..max_phones {
        for b in &dec_blocks {
            h = b.forward(&h, &memory)?;
        }
        let logits = if let Some(ref proj) = proj {
            proj.project(&h)?
        } else {
            out_emb.project(&h)?
        };
        let row = logits.row(0);
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if is_skip_out_token(g2p, i as u32, bos) {
                continue;
            }
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        let id = best as u32;
        if id == eos && !out_ids.is_empty() {
            break;
        }
        out_ids.push(id);
        // Word→pronunciation often emits one multi-phone token then stops.
        if let Some(sym) = g2p.output_symbols.symbol(id) {
            if sym.contains('-') || (sym.len() > 1 && sym != "<unk>") {
                break;
            }
        }
        h = out_emb.lookup(&[id])?;
    }
    Ok(phones_from_out_ids(g2p, &out_ids, bos))
}

fn phones_from_out_ids(g2p: &TorchnG2p, out_ids: &[u32], bos: u32) -> Vec<String> {
    let mut phones = Vec::new();
    for &id in out_ids {
        if is_skip_out_token(g2p, id, bos) {
            continue;
        }
        if let Some(sym) = g2p.output_symbols.symbol(id) {
            let sym = sym.trim_end_matches("@@");
            if sym.contains('-') {
                for part in sym.split('-') {
                    if !part.is_empty() {
                        phones.push(part.to_string());
                    }
                }
            } else if !sym.is_empty() {
                phones.push(sym.to_string());
            }
        }
    }
    phones
}

fn is_skip_out_token(g2p: &TorchnG2p, id: u32, bos: u32) -> bool {
    if id == 0 || id == bos {
        return true;
    }
    match g2p.output_symbols.symbol(id) {
        Some("<eps>" | "<unk>" | "<s>" | "</s>" | "." | "," | "?" | "!") => true,
        // BPE continuation markers (`h-E-@@`, `s-@@`, …) — prefer complete compounds.
        Some(sym) if sym.ends_with("@@") => true,
        _ => false,
    }
}

fn read_named_tensors(
    path: &Path,
    names: &[String],
) -> Result<HashMap<String, (Vec<u8>, Vec<usize>)>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hdr_len_buf = [0u8; 8];
    file.read_exact(&mut hdr_len_buf)?;
    let hdr_len = u64::from_le_bytes(hdr_len_buf) as usize;
    ensure!(hdr_len > 0 && hdr_len < 64 * 1024 * 1024, "bad hdr");
    let mut hdr_bytes = vec![0u8; hdr_len];
    file.read_exact(&mut hdr_bytes)?;
    let header: serde_json::Value = serde_json::from_slice(&hdr_bytes)?;
    let obj = header.as_object().context("header object")?;
    let data_base = 8u64 + hdr_len as u64;
    let want: HashMap<&str, ()> = names.iter().map(|s| (s.as_str(), ())).collect();
    let mut out = HashMap::new();
    for (name, info) in obj {
        if name == "__metadata__" || !want.contains_key(name.as_str()) {
            continue;
        }
        let dtype = info.get("dtype").and_then(|v| v.as_str()).context("dtype")?;
        let shape: Vec<usize> = info
            .get("shape")
            .and_then(|v| v.as_array())
            .context("shape")?
            .iter()
            .map(|x| x.as_u64().map(|n| n as usize).context("dim"))
            .collect::<Result<_>>()?;
        let offsets = info
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .context("offsets")?;
        let start = offsets[0].as_u64().context("off0")? + data_base;
        let end = offsets[1].as_u64().context("off1")? + data_base;
        let mut raw = vec![0u8; (end - start) as usize];
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(&mut raw)?;
        // Keep dtype in a side channel via empty check — callers know expected type.
        let _ = dtype;
        out.insert(name.clone(), (raw, shape));
    }
    ensure!(out.len() == names.len(), "missing tensors in {}", path.display());
    Ok(out)
}

fn decode_f32_bytes(raw: &[u8]) -> Result<Vec<f32>> {
    ensure!(raw.len() % 4 == 0, "F32 len");
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn decode_i8(raw: &[u8]) -> Result<Vec<i8>> {
    Ok(raw.iter().map(|&b| b as i8).collect())
}

fn decode_u8(raw: &[u8]) -> Result<Vec<u8>> {
    Ok(raw.to_vec())
}

fn decode_i32(raw: &[u8]) -> Result<Vec<i32>> {
    ensure!(raw.len() % 4 == 0, "I32 len");
    Ok(raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}


/// Count `qat.{i}.weight` entries in a TorchN safetensors header (no full load).
fn count_qat_tensors(path: &Path) -> Result<usize> {
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hdr_len_buf = [0u8; 8];
    use std::io::Read;
    file.read_exact(&mut hdr_len_buf)?;
    let hdr_len = u64::from_le_bytes(hdr_len_buf) as usize;
    ensure!(hdr_len > 0 && hdr_len < 64 * 1024 * 1024, "bad header len");
    let mut hdr_bytes = vec![0u8; hdr_len];
    file.read_exact(&mut hdr_bytes)?;
    let header: serde_json::Value = serde_json::from_slice(&hdr_bytes)?;
    let obj = header
        .as_object()
        .with_context(|| format!("safetensors header not object {}", path.display()))?;
    Ok(obj
        .keys()
        .filter(|k| k.starts_with("qat.") && k.ends_with(".weight"))
        .count())
}

fn symbol_table_from_json(v: Option<&serde_json::Value>, name: &str) -> Result<SymbolTable> {
    let mut sym_to_id = HashMap::new();
    let mut id_to_sym = Vec::new();
    if let Some(obj) = v.and_then(|x| x.as_object()) {
        for (sym, idv) in obj {
            // Drop binary junk keys / absurd ids from imperfect bin harvest.
            if !sym.is_ascii() || sym.chars().any(|c| c.is_control()) {
                continue;
            }
            let id = idv.as_u64().or_else(|| idv.as_i64().map(|x| x as u64)).unwrap_or(0) as u32;
            if id > 100_000 {
                continue;
            }
            if id as usize >= id_to_sym.len() {
                id_to_sym.resize(id as usize + 1, String::new());
            }
            id_to_sym[id as usize] = sym.clone();
            sym_to_id.insert(sym.clone(), id);
        }
    }
    ensure!(!sym_to_id.is_empty(), "empty symbol table {name} in g2p_bpe.json");
    Ok(SymbolTable {
        name: name.into(),
        id_to_sym,
        sym_to_id,
    })
}

fn extract_tag_string(bytes: &[u8], tag: &[u8]) -> Option<String> {
    let i = find_subslice(bytes, tag)?;
    let start = i + tag.len();
    let mut end = start;
    while end < bytes.len() && bytes[end] != b'<' && bytes[end] != 0 {
        end += 1;
    }
    let s = String::from_utf8_lossy(&bytes[start..end])
        .trim()
        .to_string();
    if s.is_empty() { None } else { Some(s) }
}

fn extract_after_tag(bytes: &[u8], tag: &[u8]) -> Option<String> {
    let i = find_subslice(bytes, tag)?;
    let mut start = i + tag.len();
    while start < bytes.len() && bytes[start].is_ascii_whitespace() {
        start += 1;
    }
    // Skip kaldi-style length prefix bytes if present.
    if start + 4 < bytes.len() && bytes[start] == 0x04 {
        start += 1;
        let n = bytes[start] as usize;
        start += 1;
        if start + n <= bytes.len() {
            return Some(String::from_utf8_lossy(&bytes[start..start + n]).into_owned());
        }
    }
    let mut end = start;
    while end < bytes.len() && (bytes[end].is_ascii_graphic() || bytes[end] == b'.') {
        end += 1;
    }
    Some(String::from_utf8_lossy(&bytes[start..end]).into_owned())
}

fn parse_num_bpe(bytes: &[u8]) -> Result<usize> {
    let i = find_subslice(bytes, b"<NumBpe>")
        .context("missing <NumBpe>")?;
    let mut start = i + b"<NumBpe>".len();
    while start < bytes.len() && bytes[start].is_ascii_whitespace() {
        start += 1;
    }
    // Observed: `<NumBpe> \x04\xa8a\x00\x00` → little-endian u32 after 0x04 tag.
    if start < bytes.len() && bytes[start] == 0x04 && start + 5 <= bytes.len() {
        let n = u32::from_le_bytes(bytes[start + 1..start + 5].try_into().unwrap());
        return Ok(n as usize);
    }
    let mut end = start;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    std::str::from_utf8(&bytes[start..end])?
        .parse()
        .context("parse NumBpe decimal")
}

fn parse_symbol_table(bytes: &[u8], has_tag: &[u8], name_hint: &[u8]) -> Result<SymbolTable> {
    let i = find_subslice(bytes, has_tag).with_context(|| {
        format!(
            "missing {}",
            String::from_utf8_lossy(has_tag)
        )
    })?;
    let end = (i + 2_000_000).min(bytes.len());
    let region = &bytes[i..end];
    let name_at = find_subslice(region, name_hint).context("symbol table name")?;
    let name = String::from_utf8_lossy(name_hint).into_owned();
    let count_off = name_at + name_hint.len();
    ensure!(count_off + 16 <= region.len(), "symbol count truncated");
    let count = u64::from_le_bytes(region[count_off..count_off + 8].try_into().unwrap()) as usize;
    ensure!(count > 0 && count < 500_000, "bad symbol count {count}");
    let mut off = count_off + 16;
    let mut sym_to_id = HashMap::new();
    let mut id_to_sym = Vec::new();
    for _ in 0..count {
        ensure!(off + 4 <= region.len(), "symbol len truncated");
        let n = u32::from_le_bytes(region[off..off + 4].try_into().unwrap()) as usize;
        ensure!(
            n > 0 && n <= 128 && off + 4 + n + 8 <= region.len(),
            "bad symbol entry"
        );
        let sym = String::from_utf8_lossy(&region[off + 4..off + 4 + n]).into_owned();
        let id =
            u64::from_le_bytes(region[off + 4 + n..off + 4 + n + 8].try_into().unwrap()) as u32;
        off = off + 4 + n + 8;
        if id as usize >= id_to_sym.len() {
            id_to_sym.resize(id as usize + 1, String::new());
        }
        if id_to_sym[id as usize].is_empty() {
            id_to_sym[id as usize] = sym.clone();
        }
        sym_to_id.entry(sym).or_insert(id);
    }
    ensure!(!sym_to_id.is_empty(), "empty symbol table {name}");
    Ok(SymbolTable {
        name,
        id_to_sym,
        sym_to_id,
    })
}

fn parse_bpe_merges(bytes: &[u8], num_bpe: usize) -> Result<Vec<(String, String)>> {
    let i = find_subslice(bytes, b"<NumBpe>").context("NumBpe for merges")?;
    // Merges are ASCII lines `a b` / `th e</w>` after the NumBpe field.
    let region = &bytes[i..];
    let text = String::from_utf8_lossy(region);
    let mut merges = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('<') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(a) = parts.next() else { continue };
        let Some(b) = parts.next() else { continue };
        if parts.next().is_some() {
            continue;
        }
        // Filter obvious non-merge junk.
        if a.len() > 32 || b.len() > 32 {
            continue;
        }
        merges.push((a.to_string(), b.to_string()));
        if merges.len() >= num_bpe {
            break;
        }
    }
    ensure!(
        !merges.is_empty(),
        "no BPE merges parsed near NumBpe (want {num_bpe})"
    );
    Ok(merges)
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_frontend(name: &str) -> PathBuf {
        if let Some(root) = crate::gguf_bundle::default_extract_dir() {
            let p = root.join("frontend").join(name);
            if p.is_file() || name.is_empty() {
                return p;
            }
        }
        let roots = [
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/rlx-tts/frontend"),
            PathBuf::from("weights/tts/rlx-tts/frontend"),
            PathBuf::from("../weights/tts/rlx-tts/frontend"),
        ];
        for root in roots {
            let p = root.join(name);
            if p.is_file() {
                return p;
            }
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../weights/tts/rlx-tts/frontend")
            .join(name)
    }

    #[test]
    fn load_torchn_header_if_present() {
        let side = cache_frontend("g2p_bpe.json");
        let bin = cache_frontend("g2p_seq2seq.bin");
        let g2p = if side.is_file() {
            TorchnG2p::load_bpe_json(&side).expect("load sidecar")
        } else if bin.is_file() {
            TorchnG2p::load_bin(&bin).expect("load torchn bin")
        } else {
            return;
        };
        assert!(g2p.num_bpe >= 1000, "num_bpe={}", g2p.num_bpe);
        assert!(
            g2p.input_symbols.sym_to_id.len() >= 10_000,
            "isyms={}",
            g2p.input_symbols.sym_to_id.len()
        );
        assert_eq!(g2p.encode_word("hello"), g2p.encode_word("HELLO"));
        assert_ne!(g2p.encode_word("hi"), g2p.encode_word("hello"));
        assert!(!g2p.output_symbols.sym_to_id.is_empty());
        assert!(!g2p.bpe_merges.is_empty());
        let ids = g2p.encode_word("hello");
        assert!(!ids.is_empty(), "encode hello");
        let st = cache_frontend("g2p_seq2seq.safetensors");
        if st.is_file() {
            assert!(
                g2p.qat_count >= 100,
                "expected exported QAT weights, qat_count={}",
                g2p.qat_count
            );
            assert!(g2p.has_weights());
            let mean = probe_ffn_block(&st, 4, 5, 0).expect("ffn quick check");
            assert!(mean.is_finite(), "ffn mean={mean}");
            let enc = probe_encoder_block0(&st).expect("encoder block0");
            assert!(enc.is_finite(), "encoder mean={enc}");
            let ids = g2p.encode_word("hello");
            let (mean, t) = probe_encode_ids(&st, &ids).expect("full encoder");
            assert!(mean.is_finite() && t == ids.len(), "full enc mean={mean} t={t}");
            let phones = probe_greedy_phones(&st, &g2p, "hi", 8).expect("greedy");
            assert!(phones.len() <= 8);
        }
    }

    #[test]
    fn greedy_vs_nashville_sample() {
        let side = cache_frontend("g2p_bpe.json");
        let st = cache_frontend("g2p_seq2seq.safetensors");
        if !side.is_file() || !st.is_file() {
            return;
        }
        let g2p = TorchnG2p::load_bpe_json(&side).expect("bpe");
        if !g2p.has_weights() {
            return;
        }
        let cases = ["hi", "hello", "street"];
        let mut report = String::new();
        for word in cases {
            let ids = g2p.encode_word(word);
            let (top, phones) = probe_greedy_debug(&st, &g2p, word, 8).expect("greedy");
            report.push_str(&format!(
                "{word} ids={ids:?} top={top:?} phones={phones:?}\n"
            ));
        }
        assert_ne!(g2p.encode_word("hi"), g2p.encode_word("hello"));
        assert!(!report.is_empty());
        eprintln!("{report}");
    }

    /// --lib enc_identity_softmax_probe -- --nocapture`
    #[test]
    fn enc_identity_softmax_probe() {
        if std::env::var_os("RLX_TTS_TORCHN_ENC_PROBE").is_none() {
            return;
        }
        let side = cache_frontend("g2p_bpe.json");
        let st = cache_frontend("g2p_seq2seq.safetensors");
        if !side.is_file() || !st.is_file() {
            return;
        }
        let g2p = TorchnG2p::load_bpe_json(&side).expect("bpe");
        let proj = TorchnSoftmaxProj::load(&st).expect("proj").expect("untied");
        let words = ["hello", "hi", "street"];
        let golds = ["h-E-l-O:", "h-Y:", "s-t-r-i:-t"];
        let in_emb = TorchnEmbedding::load_input(&st).expect("in");
        let enc = TorchnEncoder::load(&st).expect("enc");
        let mut vecs: Vec<Vec<f32>> = Vec::new();
        for (w, g) in words.iter().zip(golds) {
            let ids = g2p.encode_word(w);
            let mut x = in_emb.lookup(&ids).expect("lookup");
            if std::env::var_os("RLX_TTS_TORCHN_NO_ENCPOS").is_none() {
                if let Some(cb) = TorchnCodebook::load(&st).expect("cb") {
                    cb.add_positions(&mut x).expect("pos");
                }
            }
            let y = enc.forward(x).expect("fwd");
            let peak = y.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()));
            let (t, d) = y.dim();
            let mut mean = ndarray::Array2::<f32>::zeros((1, d));
            for c in 0..d {
                let mut s = 0.0f32;
                for r in 0..t {
                    s += y[[r, c]];
                }
                mean[[0, c]] = s / t.max(1) as f32;
            }
            let logits = proj.project(&mean).expect("logits");
            let gid = g2p.output_symbols.get(g).expect("gold id") as usize;
            let gscore = logits[[0, gid]];
            let mut rank = 1usize;
            let mut top_i = 0usize;
            let mut top_v = f32::NEG_INFINITY;
            for (i, &v) in logits.row(0).iter().enumerate() {
                if v > gscore {
                    rank += 1;
                }
                if v > top_v {
                    top_v = v;
                    top_i = i;
                }
            }
            let top = g2p
                .output_symbols
                .symbol(top_i as u32)
                .unwrap_or("?")
                .to_string();
            let m = mean.row(0).to_vec();
            let norm = m.iter().map(|v| v * v).sum::<f32>().sqrt();
            eprintln!("{w}: peak={peak:.3} ||m||={norm:.3} gold_rank={rank} g={gscore:.3} top1={top}");
            vecs.push(m);
        }
        let cos = |a: &[f32], b: &[f32]| {
            let mut dot = 0.0f32;
            let mut na = 0.0f32;
            let mut nb = 0.0f32;
            for i in 0..a.len() {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            dot / (na.sqrt() * nb.sqrt()).max(1e-12)
        };
        eprintln!(
            "cos hello/hi={:.4} hello/street={:.4} hi/street={:.4}",
            cos(&vecs[0], &vecs[1]),
            cos(&vecs[0], &vecs[2]),
            cos(&vecs[1], &vecs[2])
        );
    }

    /// Per-layer cos(hello,hi) after each encoder block.
    /// `RLX_TTS_TORCHN_LAYER_PROBE=1 cargo test -p rlx-tts --release --lib
    /// enc_layerwise_identity_probe -- --nocapture`
    #[test]
    fn enc_layerwise_identity_probe() {
        if std::env::var_os("RLX_TTS_TORCHN_LAYER_PROBE").is_none() {
            return;
        }
        let side = cache_frontend("g2p_bpe.json");
        let st = cache_frontend("g2p_seq2seq.safetensors");
        if !side.is_file() || !st.is_file() {
            return;
        }
        let g2p = TorchnG2p::load_bpe_json(&side).expect("bpe");
        let in_emb = TorchnEmbedding::load_input(&st).expect("in");
        let enc = TorchnEncoder::load(&st).expect("enc");
        let words = ["hello", "hi", "street"];
        let mut xs = Vec::new();
        for w in words {
            let ids = g2p.encode_word(w);
            let mut x = in_emb.lookup(&ids).expect("lu");
            if std::env::var_os("RLX_TTS_TORCHN_NO_ENCPOS").is_none() {
                if let Some(cb) = TorchnCodebook::load(&st).expect("cb") {
                    cb.add_positions(&mut x).expect("pos");
                }
            }
            xs.push(x);
        }
        let mean = |y: &ndarray::Array2<f32>| -> Vec<f32> {
            let (t, d) = y.dim();
            let mut m = vec![0.0f32; d];
            for c in 0..d {
                let mut s = 0.0f32;
                for r in 0..t {
                    s += y[[r, c]];
                }
                m[c] = s / t.max(1) as f32;
            }
            m
        };
        let cos = |a: &[f32], b: &[f32]| {
            let mut dot = 0.0f32;
            let mut na = 0.0f32;
            let mut nb = 0.0f32;
            for i in 0..a.len() {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            dot / (na.sqrt() * nb.sqrt()).max(1e-12)
        };
        let peak = |y: &ndarray::Array2<f32>| {
            y.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()))
        };
        let m0: Vec<_> = xs.iter().map(|x| mean(x)).collect();
        eprintln!(
            "L=0 (embed): cos h/hi={:.4} h/st={:.4} peak={:.3}",
            cos(&m0[0], &m0[1]),
            cos(&m0[0], &m0[2]),
            peak(&xs[0])
        );
        for (li, block) in enc.blocks.iter().enumerate() {
            for x in xs.iter_mut() {
                *x = block.forward(x).expect("fwd");
            }
            let ms: Vec<_> = xs.iter().map(|x| mean(x)).collect();
            eprintln!(
                "L={}: cos h/hi={:.4} h/st={:.4} peak_h={:.3} ||h||={:.3}",
                li + 1,
                cos(&ms[0], &ms[1]),
                cos(&ms[0], &ms[2]),
                peak(&xs[0]),
                ms[0].iter().map(|v| v * v).sum::<f32>().sqrt()
            );
        }
    }

    /// `RLX_TTS_TORCHN_DEC_PROBE=1 cargo test -p rlx-tts --release --lib
    /// dec_bos_softmax_probe -- --nocapture`
    #[test]
    fn dec_bos_softmax_probe() {
        if std::env::var_os("RLX_TTS_TORCHN_DEC_PROBE").is_none() {
            return;
        }
        let side = cache_frontend("g2p_bpe.json");
        let st = cache_frontend("g2p_seq2seq.safetensors");
        if !side.is_file() || !st.is_file() {
            return;
        }
        let g2p = TorchnG2p::load_bpe_json(&side).expect("bpe");
        let words = ["hello", "hi", "street"];
        let golds = ["h-E-l-O:", "h-Y:", "s-t-r-i:-t"];
        for (w, g) in words.iter().zip(golds) {
            let (top, phones) = probe_greedy_debug(&st, &g2p, w, 8).expect("dbg");
            let gid = g2p.output_symbols.get(g).expect("gold") as usize;
            let top1 = top
                .iter()
                .find(|(_, id, sym)| *id != 0 || sym.as_str() != top.last().unwrap().2)
                .map(|t| format!("{}:{:.2}", t.2, t.0))
                .unwrap_or_default();
            let ids = g2p.encode_word(w);
            let in_emb = TorchnEmbedding::load_input(&st).expect("in");
            let out_emb = TorchnEmbedding::load_output(&st).expect("out");
            let proj = TorchnSoftmaxProj::load(&st).expect("p").expect("untied");
            let mut x = in_emb.lookup(&ids).expect("x");
            if let Some(cb) = TorchnCodebook::load(&st).expect("cb") {
                cb.add_positions(&mut x).ok();
            }
            let mut memory = TorchnEncoder::load(&st).expect("e").forward(x).expect("m");
            if std::env::var_os("RLX_TTS_TORCHN_WORD_MEM").is_some() && memory.nrows() > 1 {
                memory = memory.slice(ndarray::s![0..1, ..]).to_owned();
            }
            let bos = g2p.output_symbols.get("<s>").unwrap_or(2);
            let mut h = out_emb.lookup(&[bos]).expect("bos");
            if let Some(cb) = TorchnCodebook::load(&st).expect("cb2") {
                cb.add_positions(&mut h).ok();
            }
            let n_dec = std::env::var("RLX_TTS_TORCHN_DEC_LAYERS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(5)
                .min(5);
            for i in 0..n_dec {
                let b = TorchnDecoderBlock::load(&st, i).expect("db");
                h = b.forward(&h, &memory).expect("df");
            }
            let logits = proj.project(&h).expect("logits");
            let gscore = logits[[0, gid]];
            let mut rank = 1usize;
            let mut best_i = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in logits.row(0).iter().enumerate() {
                if v > gscore {
                    rank += 1;
                }
                if v > best_v {
                    best_v = v;
                    best_i = i;
                }
            }
            let top_sym = g2p
                .output_symbols
                .symbol(best_i as u32)
                .unwrap_or("?")
                .to_string();
            eprintln!(
                "{w}: gold_rank={rank} g={gscore:.3} top1={top_sym} phones={phones:?} dbg_top={top1}"
            );
            if std::env::var_os("RLX_TTS_TORCHN_TOP20").is_some() {
                let bos = g2p.output_symbols.get("<s>").unwrap_or(2);
                let mut scored: Vec<(f32, usize)> = logits
                    .row(0)
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, v)| (v, i))
                    .collect();
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                let mut n = 0usize;
                for (v, i) in scored {
                    if is_skip_out_token(&g2p, i as u32, bos) {
                        continue;
                    }
                    let sym = g2p.output_symbols.symbol(i as u32).unwrap_or("?");
                    eprintln!("  top{n}: {v:.3} {sym}");
                    n += 1;
                    if n >= 20 {
                        break;
                    }
                }
            }
        }
    }
}
