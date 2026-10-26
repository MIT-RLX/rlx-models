// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! The distilled student flow (`decoders/student_w4.pt`): the meanflow S3Gen
//! flow at about half width, one Euler step, BitNet layers.
//!
//! Every `Linear` / `Conv1d` except `final_proj`, `input_embedding` and
//! `spk_embed_affine_layer` is a `BitLinear` / `BitConv1d`: weights stored as
//! per-output-channel 4-bit codes (`w = code / scale`), inputs fake-quantized
//! to int8 per token (`norm_input` is off in the released file). The
//! checkpoint is a `torch.save` dict — `arch`, `mode`, `bits`, `norm_input`,
//! `quantised: {name: {packed|codes, scale, shape, numel}}`, `dense` — read
//! with `rlx-torch-ckpt`. HiFT is still the teacher's.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use rlx_torch_ckpt::PtModel;
use rlx_torch_ckpt::pickle::Value;

use super::nn::{BitCfg, S3GenWeights};

/// Prompt the student was distilled against: ≤ 4 s of reference, ≤ 112 tokens.
pub const STUDENT_REF_SECONDS: f32 = 4.0;
pub const STUDENT_MAX_PROMPT_TOKENS: usize = 112;

fn dict_get<'a>(d: &'a [(Value, Value)], key: &str) -> Option<&'a Value> {
    d.iter()
        .find(|(k, _)| matches!(k, Value::Str(s) if &**s == key))
        .map(|(_, v)| v)
}

fn as_dict(v: &Value) -> Result<Vec<(Value, Value)>> {
    match v {
        Value::Dict(d) => Ok(d.borrow().clone()),
        other => bail!("expected a dict, got {other:?}"),
    }
}

fn as_int(v: Option<&Value>) -> Result<i64> {
    match v {
        Some(Value::Int(i)) => Ok(*i),
        Some(Value::Bool(b)) => Ok(*b as i64),
        other => bail!("expected an int, got {other:?}"),
    }
}

fn as_shape(v: Option<&Value>) -> Result<Vec<usize>> {
    let items = match v {
        Some(Value::List(l)) => l.borrow().clone(),
        Some(Value::Tuple(t)) => t.as_slice().to_vec(),
        other => bail!("expected a shape list, got {other:?}"),
    };
    items
        .iter()
        .map(|x| Ok(as_int(Some(x))? as usize))
        .collect()
}

/// `_unpack4`: two codes per byte, high nibble first, each biased by 8.
pub fn unpack4(packed: &[f32], numel: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(numel);
    for &b in packed {
        let b = b as u8;
        out.push(((b >> 4) as i32 - 8) as f32);
        out.push(((b & 0x0F) as i32 - 8) as f32);
    }
    out.truncate(numel);
    out
}

/// Student flow weights (as `flow.*`) merged with the teacher's HiFT
/// (`mel2wav.*` from `s3gen_meanflow.safetensors`).
pub fn open(student: &Path, teacher: &Path) -> Result<S3GenWeights> {
    let root = rlx_torch_ckpt::read_torch_pickle_root(student)?;
    let top = as_dict(&root)?;
    let mode = match dict_get(&top, "mode") {
        Some(Value::Str(s)) => s.to_string(),
        other => bail!("student checkpoint: bad mode {other:?}"),
    };
    let bits = as_int(dict_get(&top, "bits"))?;
    let norm_input = as_int(dict_get(&top, "norm_input"))? != 0;
    // `w<bits>[_pt][g<group>][a16]`: `a16` keeps activations in full precision.
    ensure!(
        mode.starts_with('w') && !mode.contains("_pt") && !mode.contains('g'),
        "student mode {mode:?} not supported (per-channel `w<bits>` only)"
    );
    let bit = BitCfg {
        norm_input,
        // `RLX_KITTENTTS2_STUDENT_NO_ACT_QUANT=1` skips the int8 fake-quant,
        // to compare the network against a reference run the same way.
        act_quant: !mode.ends_with("a16")
            && !rlx_ir::env::flag("RLX_KITTENTTS2_STUDENT_NO_ACT_QUANT"),
    };

    let model = PtModel::open(student).with_context(|| format!("open {}", student.display()))?;
    let quantised = as_dict(dict_get(&top, "quantised").context("no `quantised`")?)?;
    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut bit_keys = Vec::new();
    for (k, v) in &quantised {
        let Value::Str(name) = k else { continue };
        let entry = as_dict(v)?;
        let shape = as_shape(dict_get(&entry, "shape"))?;
        let numel: usize = shape.iter().product();
        let codes = match bits {
            4 => unpack4(
                &model.tensor(&format!("quantised.{name}.packed"))?.data,
                numel,
            ),
            8 => model.tensor(&format!("quantised.{name}.codes"))?.data,
            b => bail!("{b}-bit student weights not supported"),
        };
        ensure!(
            codes.len() == numel,
            "{name}: {} codes for shape {shape:?}",
            codes.len()
        );
        let scale = model.tensor(&format!("quantised.{name}.scale"))?.data;
        let rows = shape[0];
        ensure!(
            scale.len() == rows,
            "{name}: {} scales for {rows} rows",
            scale.len()
        );
        let per = numel / rows;
        let w: Vec<f32> = codes
            .chunks_exact(per)
            .zip(&scale)
            .flat_map(|(row, &s)| row.iter().map(move |&c| c / s))
            .collect();
        let key = format!("flow.{name}");
        bit_keys.push(key.clone());
        tensors.insert(key, (w, shape));
    }
    let dense = as_dict(dict_get(&top, "dense").context("no `dense`")?)?;
    for (k, _) in &dense {
        let Value::Str(name) = k else { continue };
        let t = model.tensor(&format!("dense.{name}"))?;
        tensors.insert(format!("flow.{name}"), (t.data, t.shape));
    }
    let hift = S3GenWeights::open_prefixes(teacher, &["mel2wav."])?;
    for key in hift.keys_with_prefix("mel2wav.") {
        tensors.insert(
            key.clone(),
            (hift.get(&key)?.to_vec(), hift.shape(&key)?.to_vec()),
        );
    }
    Ok(S3GenWeights::from_tensors(tensors).with_bit_layers(bit_keys, bit))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nibbles_unpack_high_first_with_bias() {
        assert_eq!(unpack4(&[0x8F as f32, 0x07 as f32], 3), [0.0, 7.0, -8.0]);
    }
}
