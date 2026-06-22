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

//! Reads a single-file EAGLE3 draft `model.safetensors` checkpoint
//! (the layout RedHatAI publishes) and surfaces the tensors the draft
//! graph needs.
//!
//! Tensor names expected (per the speculators reference):
//!
//! | Field | Tensor name | Shape |
//! |---|---|---|
//! | `fc.weight` | `fc.weight` | `[H_draft, 3 * H_target]` |
//! | `embed_tokens.weight` | `embed_tokens.weight` | `[V_target, H_draft]` (shared with verifier vocab) |
//! | `lm_head.weight` | `lm_head.weight` | `[V_draft, H_draft]` |
//! | `hidden_norm.weight` | `hidden_norm.weight` | `[H_target]` |
//! | `input_norm.weight` (if `norm_before_fc=true`) | `input_norm.weight` | `[3 * H_target]` |
//! | `verifier_norm.weight` | `verifier_norm.weight` | `[H_draft]` |
//! | `norm.weight` | `norm.weight` | `[H_draft]` |
//! | Decoder layer-0 tensors | `midlayer.{q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,input_layernorm,post_attention_layernorm}.weight` | (Llama-style) |
//! | `d2t` | `d2t` | `[V_draft]` (u32 / i64) |
//! | `t2d` (optional) | `t2d` | `[V_target]` (bool / u8) |
//!
//! Some checkpoints publish `layers.0.<name>` instead of `midlayer.<name>`.
//! [`Eagle3DraftWeights::open`] accepts either.

use anyhow::{Context, Result, bail};
use safetensors::SafeTensors;
use safetensors::tensor::Dtype as StDtype;
use std::path::Path;

/// A single tensor loaded from the draft checkpoint, materialized as
/// `f32` rows so downstream HIR constants can ingest it directly.
#[derive(Debug, Clone)]
pub struct DraftTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// A loaded EAGLE3 draft checkpoint. Holds raw f32 tensors for the
/// pieces the [`crate::draft`] graph + [`crate::d2t`] map both need.
#[derive(Debug, Clone)]
pub struct Eagle3DraftWeights {
    /// All draft tensors, keyed by their canonical name *after*
    /// stripping any `model.` / `midlayer.` / `layers.0.` prefix to
    /// the form used in the spec table above.
    tensors: std::collections::HashMap<String, DraftTensor>,
    /// Raw `d2t` table (target ids indexed by draft id).
    d2t: Vec<u32>,
}

impl Eagle3DraftWeights {
    /// Open a single-file `model.safetensors` (the RedHatAI layout).
    pub fn open(file: impl AsRef<Path>) -> Result<Self> {
        let path = file.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("read eagle3 draft safetensors {path:?}"))?;
        Self::from_bytes(&bytes)
    }

    /// Parse from in-memory safetensors bytes (used by tests).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let st = SafeTensors::deserialize(bytes).context("parse eagle3 draft safetensors")?;
        let mut tensors = std::collections::HashMap::new();
        let mut d2t_opt: Option<Vec<u32>> = None;
        let names = st.names();
        for raw_name in names {
            let view = st
                .tensor(raw_name)
                .with_context(|| format!("look up {raw_name}"))?;
            let canon = canonicalize_name(raw_name);
            // Special-case d2t/t2d — they're integer LUTs, not float
            // weights.
            if canon == "d2t" {
                d2t_opt = Some(read_integer_lut(view.dtype(), view.data())?);
                continue;
            }
            if canon == "t2d" {
                // Validation buffer; we don't use it at inference.
                continue;
            }

            let dtype = view.dtype();
            let shape: Vec<usize> = view.shape().to_vec();
            let data = read_float_tensor_as_f32(dtype, view.data(), &shape)
                .with_context(|| format!("decode {raw_name} as f32"))?;
            tensors.insert(
                canon.clone(),
                DraftTensor {
                    name: canon,
                    shape,
                    data,
                },
            );
        }

        let d2t = d2t_opt.context("eagle3 draft safetensors missing required `d2t` tensor")?;

        Ok(Self { tensors, d2t })
    }

    /// Borrow a tensor by canonical name. See module docs for the
    /// table of names.
    pub fn get(&self, canonical: &str) -> Option<&DraftTensor> {
        self.tensors.get(canonical)
    }

    /// Iterate over all canonical tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Borrow the raw d2t table.
    pub fn d2t(&self) -> &[u32] {
        &self.d2t
    }

    /// Number of float tensors loaded (excludes d2t/t2d).
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Strip optional prefixes (`model.`, `midlayer.`, `layers.0.`) so
/// downstream code can use a single canonical name regardless of
/// which speculators version produced the checkpoint.
fn canonicalize_name(raw: &str) -> String {
    let stripped = raw.strip_prefix("model.").unwrap_or(raw);
    // `midlayer.<x>` and `layers.0.<x>` both refer to the single
    // draft transformer block — flatten to `decoder.<x>` so the
    // build code is independent of the on-disk naming.
    if let Some(rest) = stripped.strip_prefix("midlayer.") {
        return format!("decoder.{rest}");
    }
    if let Some(rest) = stripped.strip_prefix("layers.0.") {
        return format!("decoder.{rest}");
    }
    stripped.to_string()
}

fn read_float_tensor_as_f32(dtype: StDtype, raw: &[u8], shape: &[usize]) -> Result<Vec<f32>> {
    let n: usize = shape.iter().product();
    let out = match dtype {
        StDtype::F32 => {
            if raw.len() != n * 4 {
                bail!(
                    "f32 tensor size mismatch: {} bytes for {} elems",
                    raw.len(),
                    n
                );
            }
            bytemuck::cast_slice::<u8, f32>(raw).to_vec()
        }
        StDtype::BF16 => {
            if raw.len() != n * 2 {
                bail!(
                    "bf16 tensor size mismatch: {} bytes for {} elems",
                    raw.len(),
                    n
                );
            }
            let bf: &[half::bf16] = bytemuck::cast_slice(raw);
            bf.iter().map(|x| x.to_f32()).collect()
        }
        StDtype::F16 => {
            if raw.len() != n * 2 {
                bail!(
                    "f16 tensor size mismatch: {} bytes for {} elems",
                    raw.len(),
                    n
                );
            }
            let f: &[half::f16] = bytemuck::cast_slice(raw);
            f.iter().map(|x| x.to_f32()).collect()
        }
        other => bail!("unsupported eagle3 weight dtype: {other:?}"),
    };
    Ok(out)
}

fn read_integer_lut(dtype: StDtype, raw: &[u8]) -> Result<Vec<u32>> {
    match dtype {
        StDtype::U32 => Ok(bytemuck::cast_slice::<u8, u32>(raw).to_vec()),
        StDtype::I32 => {
            let s: &[i32] = bytemuck::cast_slice(raw);
            // Reject negatives — d2t/t2d are vocab indices.
            for (i, &v) in s.iter().enumerate() {
                if v < 0 {
                    bail!("d2t: negative value at index {i}: {v}");
                }
            }
            Ok(s.iter().map(|v| *v as u32).collect())
        }
        StDtype::I64 => {
            let s: &[i64] = bytemuck::cast_slice(raw);
            for (i, &v) in s.iter().enumerate() {
                if v < 0 || v > u32::MAX as i64 {
                    bail!("d2t: out-of-range value at index {i}: {v}");
                }
            }
            Ok(s.iter().map(|v| *v as u32).collect())
        }
        StDtype::U16 => {
            let s: &[u16] = bytemuck::cast_slice(raw);
            Ok(s.iter().map(|v| *v as u32).collect())
        }
        other => bail!("unsupported d2t/t2d dtype: {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::serialize;
    use safetensors::tensor::{Dtype as StDtype, TensorView};
    use std::collections::HashMap;

    /// Build a minimal in-memory safetensors blob with a few f32 weights plus a
    /// u32 d2t LUT. In-memory `serialize` keeps the tests parallel-safe (no
    /// shared temp files).
    fn synth_safetensors() -> Vec<u8> {
        let fc_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let lm_head_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let d2t_data: Vec<u32> = vec![3, 7, 11];

        let fc_bytes: Vec<u8> = bytemuck::cast_slice(&fc_data).to_vec();
        let lm_head_bytes: Vec<u8> = bytemuck::cast_slice(&lm_head_data).to_vec();
        let d2t_bytes: Vec<u8> = bytemuck::cast_slice(&d2t_data).to_vec();

        let fc_view = TensorView::new(StDtype::F32, vec![2, 2], &fc_bytes).unwrap();
        let lm_head_view = TensorView::new(StDtype::F32, vec![3, 2], &lm_head_bytes).unwrap();
        let d2t_view = TensorView::new(StDtype::U32, vec![3], &d2t_bytes).unwrap();

        // Test prefix canonicalization: emit `model.fc.weight` and
        // `midlayer.q_proj.weight` and verify they get rewritten.
        let q_proj_data: Vec<f32> = vec![0.5; 4];
        let q_proj_bytes: Vec<u8> = bytemuck::cast_slice(&q_proj_data).to_vec();
        let q_proj_view = TensorView::new(StDtype::F32, vec![2, 2], &q_proj_bytes).unwrap();

        let mut map: HashMap<&str, TensorView<'_>> = HashMap::new();
        map.insert("model.fc.weight", fc_view);
        map.insert("lm_head.weight", lm_head_view);
        map.insert("d2t", d2t_view);
        map.insert("midlayer.q_proj.weight", q_proj_view);

        serialize(&map, None).unwrap()
    }

    #[test]
    fn loads_f32_weights_and_d2t() {
        let bytes = synth_safetensors();
        let w = Eagle3DraftWeights::from_bytes(&bytes).unwrap();

        // `model.` prefix stripped:
        let fc = w.get("fc.weight").unwrap();
        assert_eq!(fc.shape, vec![2, 2]);
        assert_eq!(fc.data, vec![0.1, 0.2, 0.3, 0.4]);

        // `midlayer.` rewritten to `decoder.`:
        let q = w.get("decoder.q_proj.weight").unwrap();
        assert_eq!(q.shape, vec![2, 2]);

        let head = w.get("lm_head.weight").unwrap();
        assert_eq!(head.shape, vec![3, 2]);
        assert_eq!(head.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert_eq!(w.d2t(), &[3, 7, 11]);
        assert_eq!(w.len(), 3); // fc, lm_head, decoder.q_proj
    }

    #[test]
    fn fails_when_d2t_missing() {
        let fc_data: Vec<f32> = vec![0.0; 4];
        let fc_bytes: Vec<u8> = bytemuck::cast_slice(&fc_data).to_vec();
        let fc_view = TensorView::new(StDtype::F32, vec![2, 2], &fc_bytes).unwrap();
        let mut map: HashMap<&str, TensorView<'_>> = HashMap::new();
        map.insert("fc.weight", fc_view);
        let bytes = serialize(&map, None).unwrap();

        let err = Eagle3DraftWeights::from_bytes(&bytes).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("d2t"), "expected d2t error, got: {msg}");
    }

    #[test]
    fn d2t_accepts_i64_dtype() {
        let d2t: Vec<i64> = vec![5, 6, 7];
        let bytes_d2t: Vec<u8> = bytemuck::cast_slice(&d2t).to_vec();
        let d2t_view = TensorView::new(StDtype::I64, vec![3], &bytes_d2t).unwrap();
        let mut map: HashMap<&str, TensorView<'_>> = HashMap::new();
        map.insert("d2t", d2t_view);
        let bytes = serialize(&map, None).unwrap();

        let w = Eagle3DraftWeights::from_bytes(&bytes).unwrap();
        assert_eq!(w.d2t(), &[5, 6, 7]);
    }
}
