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

//! The checkpoint's tensors under the names the graph builders read.
//!
//! Both releases carry the same weights under two spellings: the
//! `LiquidAI/Liquid-Codec` repo (`_frontend.`, `quantizer.fsq.`,
//! `self_attn.gate_proj`) and LFM3-Audio's `audio_tokenizer/` (`frontend.`,
//! `quantizer.`). Both become one set of names here, with the LFM3 layers
//! renamed to the names of [`rlx_lfm3_audio::trunk`] (llama.cpp's `clip`
//! spelling, [`MMPROJ_NAMES`](rlx_lfm3_audio::trunk::MMPROJ_NAMES)) so that
//! builder runs the two stacks unchanged:
//!
//! | checkpoint | here |
//! |---|---|
//! | `encoder.lfm.layers.{i}.…` / `decoder.lfm.layers.{i}.…` | `enc.blk.{i}.…` / `dec.blk.{i}.…` |
//! | `{encoder,decoder}.lfm.embedding_norm` (the final norm) | `enc.output_norm` / `dec.output_norm` |
//! | `_frontend.` / `frontend.` | `frontend.` |
//! | `quantizer.fsq.` / `quantizer.` | `quantizer.` |

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use rlx_core::weight_loader::WeightLoader;
use rlx_ir::DType;
use safetensors::{Dtype, SafeTensors};

/// LFM3 layer tensors: checkpoint suffix → trunk suffix.
const LAYER_NAMES: &[(&str, &str)] = &[
    ("operator_norm", "ln1"),
    ("ffn_norm", "ln2"),
    ("post_operator_norm", "attn_post_norm"),
    ("post_ffn_norm", "ffn_post_norm"),
    ("conv.conv", "conv_dw"),
    ("conv.in_proj", "conv_pw1"),
    ("conv.out_proj", "conv_pw2"),
    ("self_attn.q_proj", "attn_q"),
    ("self_attn.k_proj", "attn_k"),
    ("self_attn.v_proj", "attn_v"),
    ("self_attn.o_proj", "attn_out"),
    ("self_attn.q_norm", "attn_q_norm"),
    ("self_attn.k_norm", "attn_k_norm"),
    ("self_attn.gate_proj", "attn_gate"),
    ("self_attn.g_proj", "attn_gate"),
    ("feed_forward.w1", "ffn_gate"),
    ("feed_forward.w3", "ffn_up"),
    ("feed_forward.w2", "ffn_down"),
];

/// The name a checkpoint tensor is read under.
pub fn canonical(name: &str) -> String {
    for (stack, short) in [("encoder.lfm.", "enc."), ("decoder.lfm.", "dec.")] {
        let Some(rest) = name.strip_prefix(stack) else {
            continue;
        };
        if let Some(rest) = rest.strip_prefix("layers.")
            && let Some((layer, tail)) = rest.split_once('.')
        {
            let (base, kind) = tail.rsplit_once('.').unwrap_or((tail, ""));
            let base = LAYER_NAMES
                .iter()
                .find(|(from, _)| *from == base)
                .map_or(base, |(_, to)| to);
            return format!("{short}blk.{layer}.{base}.{kind}");
        }
        if let Some(kind) = rest.strip_prefix("embedding_norm.") {
            return format!("{short}output_norm.{kind}");
        }
    }
    if let Some(rest) = name.strip_prefix("_frontend.") {
        return format!("frontend.{rest}");
    }
    if let Some(rest) = name.strip_prefix("quantizer.fsq.") {
        return format!("quantizer.{rest}");
    }
    name.to_string()
}

/// `model.safetensors` of `path`: the file itself, the codec directory's,
/// or the `audio_tokenizer/` of an LFM3-Audio directory. A directory counts
/// as the codec's when its `config.json` says `model_type: liquid_codec` (or
/// it has no `config.json`) — an LFM3-Audio directory also holds a
/// `model.safetensors`: the 4 GB LM.
pub fn resolve(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    let is_codec_dir = |dir: &Path| -> bool {
        if !dir.join("model.safetensors").is_file() {
            return false;
        }
        let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
            return true;
        };
        serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|v| v.get("model_type")?.as_str().map(|t| t == "liquid_codec"))
            .unwrap_or(false)
    };
    for dir in [path.to_path_buf(), path.join("audio_tokenizer")] {
        if is_codec_dir(&dir) {
            return Ok(dir.join("model.safetensors"));
        }
    }
    bail!(
        "no LiquidCodec model.safetensors in {} (nor in its audio_tokenizer/)",
        path.display()
    )
}

/// The checkpoint as stored (bf16, 147 MB), indexed under [`canonical`]
/// names; a tensor is widened to f32 only when a graph build reads it.
pub struct CodecWeights {
    bytes: Vec<u8>,
    /// Canonical name → (dtype, shape, byte range in `bytes`).
    index: HashMap<String, (Dtype, Vec<usize>, Range<usize>)>,
    path: PathBuf,
}

impl CodecWeights {
    /// Load `model.safetensors` (see [`resolve`]).
    pub fn open(path: &Path) -> Result<Self> {
        let path = resolve(path)?;
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let (header, meta) =
            SafeTensors::read_metadata(&bytes).map_err(|e| anyhow!("{}: {e}", path.display()))?;
        let data = 8 + header;
        let mut index = HashMap::new();
        for (name, info) in meta.tensors() {
            let (a, b) = info.data_offsets;
            if !matches!(info.dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
                bail!("{}: {name} is {:?}", path.display(), info.dtype);
            }
            index.insert(
                canonical(&name),
                (info.dtype, info.shape.clone(), data + a..data + b),
            );
        }
        Ok(Self { bytes, index, path })
    }

    /// The safetensors file the weights came from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn has(&self, key: &str) -> bool {
        self.index.contains_key(key)
    }

    /// Shape of `key`.
    pub fn shape(&self, key: &str) -> Result<&[usize]> {
        self.index
            .get(key)
            .map(|(_, s, _)| s.as_slice())
            .ok_or_else(|| anyhow!("liquid-codec: tensor {key} missing"))
    }

    /// `key` widened to f32, row-major, and its shape.
    pub fn read(&self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (dtype, shape, range) = self
            .index
            .get(key)
            .ok_or_else(|| anyhow!("weight not found: {key}"))?;
        let raw = self.bytes[range.clone()].as_chunks::<2>().0;
        let data = match dtype {
            Dtype::BF16 => raw
                .iter()
                .map(|c| half::bf16::from_le_bytes(*c).to_f32())
                .collect(),
            Dtype::F16 => raw
                .iter()
                .map(|c| half::f16::from_le_bytes(*c).to_f32())
                .collect(),
            _ => self.bytes[range.clone()]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
        };
        Ok((data, shape.clone()))
    }

    fn stored_dtype(&self, key: &str) -> Option<DType> {
        self.index.get(key).map(|(d, _, _)| match d {
            Dtype::BF16 => DType::BF16,
            Dtype::F16 => DType::F16,
            _ => DType::F32,
        })
    }

    /// A loader over these weights for one graph build; `take` widens a
    /// copy, so every build reads the same weights.
    pub fn loader(&self) -> Shared<'_> {
        Shared(self)
    }
}

/// A non-consuming [`WeightLoader`] over [`CodecWeights`].
pub struct Shared<'a>(&'a CodecWeights);

impl WeightLoader for Shared<'_> {
    fn format_id(&self) -> &'static str {
        "safetensors"
    }

    fn len(&self) -> usize {
        self.0.index.len()
    }

    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.0.read(key)
    }

    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (d, s) = self.take(key)?;
        let [rows, cols] = s[..] else {
            bail!("{key}: transpose requires 2D, got {s:?}");
        };
        let mut t = vec![0f32; d.len()];
        for r in 0..rows {
            for c in 0..cols {
                t[c * rows + r] = d[r * cols + c];
            }
        }
        Ok((t, vec![cols, rows]))
    }

    fn stored_dtype(&self, key: &str) -> Option<DType> {
        self.0.stored_dtype(key)
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.0.index.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::canonical;

    #[test]
    fn both_spellings_meet() {
        for (from, to) in [
            (
                "decoder.lfm.layers.2.self_attn.gate_proj.weight",
                "dec.blk.2.attn_gate.weight",
            ),
            (
                "encoder.lfm.layers.0.conv.in_proj.weight",
                "enc.blk.0.conv_pw1.weight",
            ),
            (
                "encoder.lfm.layers.7.feed_forward.w3.weight",
                "enc.blk.7.ffn_up.weight",
            ),
            (
                "encoder.lfm.embedding_norm.weight",
                "enc.output_norm.weight",
            ),
            ("_frontend.in_proj.bias", "frontend.in_proj.bias"),
            ("frontend.in_proj.bias", "frontend.in_proj.bias"),
            (
                "quantizer.fsq.project_in.weight",
                "quantizer.project_in.weight",
            ),
            ("quantizer.project_in.weight", "quantizer.project_in.weight"),
            ("decoder.lin.weight", "decoder.lin.weight"),
        ] {
            assert_eq!(canonical(from), to, "{from}");
        }
    }
}
