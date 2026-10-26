// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! `lm/model-ternary.safetensors` → a Qwen3 GGUF whose linears are `Q2_0`.
//!
//! KittenTTS 2 ships its language model as `kittenml-ternary-1.58`: every
//! linear weight is ternary within 128-wide groups (`{-s, 0, +s}`), stored as
//! base-3 digits five to a byte over the flattened tensor plus one bf16 scale
//! per group. That is exactly what `Q2_0` (ggml type 42, group 128) encodes —
//! a 2-bit code `q` meaning `(q - 1) · d` with one f16 `d` per group — so the
//! transcode is lossless whenever each bf16 scale is representable in f16,
//! which [`transcode`] checks rather than assumes.
//!
//! The output is a standard llama.cpp-named Qwen3 GGUF, so the whole packed
//! `rlx-qwen3` path (fused `DequantMatMul` projections on every backend that
//! has a `Q2_0` kernel, host-gathered embeddings, bucketed resident decode)
//! runs it unchanged. Non-ternary tensors keep their values: the token
//! embedding stays BF16 (the tied LM head can then run it packed), norms go
//! to F32. The checkpoint's `spk_proj.*` head is
//! not part of a Qwen3 and is read separately (see [`crate::lm`]).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use half::{bf16, f16};
use rlx_gguf::{GgmlType, GgufWriter, MetaValue};
use rlx_qwen3::Qwen3Config;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

/// Group width the checkpoint was quantized with (and `Q2_0`'s block size).
pub const GROUP: usize = 128;
const TRITS_PER_BYTE: usize = 5;
const Q2_0_BLOCK: usize = 2 + GROUP / 4;

/// One `name::codes` / `name::scale` pair as declared in the file metadata.
#[derive(Debug, Clone, Deserialize)]
struct PackedSpec {
    shape: Vec<usize>,
}

/// Facts about a finished transcode, for logging and tests.
#[derive(Debug, Clone, Default)]
pub struct TranscodeReport {
    pub ternary_tensors: usize,
    pub dense_tensors: usize,
    pub bytes: u64,
}

/// Decode `numel` base-3 digits (trits `0/1/2` for `-s/0/+s`).
pub fn unpack_trits(packed: &[u8], numel: usize) -> Result<Vec<u8>> {
    ensure!(
        packed.len() * TRITS_PER_BYTE >= numel,
        "{} packed bytes hold fewer than {numel} trits",
        packed.len()
    );
    let mut out = Vec::with_capacity(numel);
    'outer: for &b in packed {
        let mut v = b;
        ensure!(v < 243, "byte {b} is not five base-3 digits");
        for _ in 0..TRITS_PER_BYTE {
            if out.len() == numel {
                break 'outer;
            }
            out.push(v % 3);
            v /= 3;
        }
    }
    Ok(out)
}

/// `[rows, cols]` trits + `[rows, cols / 128]` bf16 scales → `Q2_0` bytes.
///
/// Fails if a scale is not exactly representable in f16, since the result
/// would then silently differ from the checkpoint.
pub fn trits_to_q2_0(trits: &[u8], scales: &[bf16], rows: usize, cols: usize) -> Result<Vec<u8>> {
    ensure!(
        cols.is_multiple_of(GROUP),
        "row width {cols} is not a multiple of {GROUP}"
    );
    let groups = cols / GROUP;
    ensure!(
        trits.len() == rows * cols,
        "trit count {} != {rows}x{cols}",
        trits.len()
    );
    ensure!(
        scales.len() == rows * groups,
        "scale count {} != {rows}x{groups}",
        scales.len()
    );
    let mut out = vec![0u8; rows * groups * Q2_0_BLOCK];
    for (blk, (chunk, &s)) in trits.as_chunks::<GROUP>().0.iter().zip(scales).enumerate() {
        let wide = s.to_f32();
        let d = f16::from_f32(wide);
        if d.to_f32() != wide {
            bail!("group scale {wide:e} (block {blk}) has no exact f16 representation");
        }
        let o = &mut out[blk * Q2_0_BLOCK..(blk + 1) * Q2_0_BLOCK];
        o[..2].copy_from_slice(&d.to_le_bytes());
        for (j, &t) in chunk.iter().enumerate() {
            o[2 + j / 4] |= t << ((j % 4) * 2);
        }
    }
    Ok(out)
}

/// HF Qwen3 tensor name → llama.cpp GGUF name.
fn gguf_name(hf: &str) -> Option<String> {
    match hf {
        "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        "model.norm.weight" => return Some("output_norm.weight".into()),
        _ => {}
    }
    let rest = hf.strip_prefix("model.layers.")?;
    let (layer, leaf) = rest.split_once('.')?;
    let leaf = match leaf {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        _ => return None,
    };
    Some(format!("blk.{layer}.{leaf}"))
}

fn bf16_values(view: &safetensors::tensor::TensorView<'_>, name: &str) -> Result<Vec<bf16>> {
    ensure!(
        view.dtype() == Dtype::BF16,
        "{name}: expected BF16, found {:?}",
        view.dtype()
    );
    Ok(view
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_le_bytes([b[0], b[1]]))
        .collect())
}

/// Write the Qwen3 metadata the GGUF readers (and llama.cpp) expect.
fn set_qwen3_meta(w: &mut GgufWriter, cfg: &Qwen3Config) {
    let u = |v: usize| MetaValue::U32(v as u32);
    w.set_arch("qwen3");
    w.set_meta(
        "general.name",
        MetaValue::String("KittenTTS 2 speech LM".into()),
    );
    w.set_meta(
        "general.source",
        MetaValue::String("KittenML/kitten-tts-2 lm/model-ternary.safetensors".into()),
    );
    w.set_meta("qwen3.block_count", u(cfg.num_hidden_layers));
    w.set_meta("qwen3.context_length", u(cfg.max_position_embeddings));
    w.set_meta("qwen3.embedding_length", u(cfg.hidden_size));
    w.set_meta("qwen3.feed_forward_length", u(cfg.intermediate_size));
    w.set_meta("qwen3.attention.head_count", u(cfg.num_attention_heads));
    w.set_meta("qwen3.attention.head_count_kv", u(cfg.num_key_value_heads));
    w.set_meta("qwen3.attention.key_length", u(cfg.head_dim));
    w.set_meta("qwen3.attention.value_length", u(cfg.head_dim));
    w.set_meta(
        "qwen3.rope.freq_base",
        MetaValue::F32(cfg.rope_theta as f32),
    );
    w.set_meta(
        "qwen3.attention.layer_norm_rms_epsilon",
        MetaValue::F32(cfg.rms_norm_eps as f32),
    );
    w.set_meta("qwen3.vocab_size", u(cfg.vocab_size));
}

/// How one checkpoint tensor lands in the GGUF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Placement {
    /// Ternary `[rows, cols]` → `Q2_0` (`name::codes` + `name::scale`).
    Ternary { rows: usize, cols: usize },
    /// The token embedding, kept BF16.
    Embedding,
    /// Norm weights, bf16 → F32.
    Dense,
}

/// One planned output tensor.
#[derive(Debug, Clone)]
pub struct PlannedTensor {
    pub hf: String,
    pub gguf: String,
    pub placement: Placement,
}

/// Decide every output tensor from the checkpoint's names and packed index
/// alone (no tensor data), checking the layout `Q2_0` and the Qwen3 loader
/// need. `names` are the safetensors keys (`name`, `name::codes`, …).
pub fn plan<'a>(
    names: impl IntoIterator<Item = &'a str>,
    packed: &HashMap<String, Vec<usize>>,
    cfg: &Qwen3Config,
) -> Result<Vec<PlannedTensor>> {
    let mut bases: Vec<&str> = names
        .into_iter()
        .map(|n| n.split("::").next().unwrap_or(n))
        .filter(|n| !n.starts_with("spk_proj."))
        .collect();
    bases.sort_unstable();
    bases.dedup();
    let mut out = Vec::with_capacity(bases.len());
    for hf in bases {
        let gguf = gguf_name(hf).with_context(|| format!("no GGUF name for {hf}"))?;
        let placement = match packed.get(hf) {
            Some(shape) => {
                let [rows, cols] = shape[..] else {
                    bail!("{hf}: packed tensor of rank {}", shape.len());
                };
                ensure!(
                    cols.is_multiple_of(GROUP),
                    "{hf}: width {cols} is not a multiple of {GROUP}"
                );
                Placement::Ternary { rows, cols }
            }
            None if hf == "model.embed_tokens.weight" => Placement::Embedding,
            None => Placement::Dense,
        };
        out.push(PlannedTensor {
            hf: hf.to_string(),
            gguf,
            placement,
        });
    }
    let ternary = out
        .iter()
        .filter(|t| matches!(t.placement, Placement::Ternary { .. }))
        .count();
    let expected = cfg.num_hidden_layers * 7;
    ensure!(
        ternary == expected,
        "{ternary} ternary tensors, a {}-layer Qwen3 has {expected}",
        cfg.num_hidden_layers
    );
    let expected_total = cfg.num_hidden_layers * 11 + 2;
    ensure!(
        out.len() == expected_total,
        "{} tensors besides spk_proj, a {}-layer Qwen3 has {expected_total}",
        out.len(),
        cfg.num_hidden_layers
    );
    Ok(out)
}

/// Parse the `packed` index of a `kittenml-ternary-1.58` header.
fn packed_index(md: &HashMap<String, String>) -> Result<HashMap<String, Vec<usize>>> {
    ensure!(
        md.get("format").map(String::as_str) == Some("kittenml-ternary-1.58"),
        "not a kittenml-ternary-1.58 checkpoint (format {:?})",
        md.get("format")
    );
    let group: usize = md
        .get("group_size")
        .context("missing group_size")?
        .parse()
        .context("group_size")?;
    ensure!(group == GROUP, "group size {group}, Q2_0 needs {GROUP}");
    let packed: HashMap<String, PackedSpec> =
        serde_json::from_str(md.get("packed").context("missing packed index")?)
            .context("packed index")?;
    Ok(packed.into_iter().map(|(k, v)| (k, v.shape)).collect())
}

/// Transcode `ternary` (the packed safetensors) to a `Q2_0` Qwen3 GGUF at `out`.
pub fn transcode(ternary: &Path, cfg: &Qwen3Config, out: &Path) -> Result<TranscodeReport> {
    let bytes = std::fs::read(ternary).with_context(|| format!("read {}", ternary.display()))?;
    let (_, meta) = SafeTensors::read_metadata(&bytes).context("safetensors header")?;
    let md = meta.metadata().clone().unwrap_or_default();
    let packed = packed_index(&md).with_context(|| ternary.display().to_string())?;
    let st = SafeTensors::deserialize(&bytes).context("safetensors body")?;
    let planned = plan(st.names(), &packed, cfg)?;

    let mut w = GgufWriter::new();
    set_qwen3_meta(&mut w, cfg);
    let mut report = TranscodeReport::default();
    for t in planned {
        let name = t.hf.as_str();
        match t.placement {
            Placement::Ternary { rows, cols } => {
                let codes = st.tensor(&format!("{name}::codes"))?;
                let scale = st.tensor(&format!("{name}::scale"))?;
                let trits = unpack_trits(codes.data(), rows * cols)?;
                let scales = bf16_values(&scale, name)?;
                let q = trits_to_q2_0(&trits, &scales, rows, cols)
                    .with_context(|| format!("transcode {name}"))?;
                report.bytes += q.len() as u64;
                w.add_tensor_bytes(t.gguf, vec![cols, rows], GgmlType::Q2_0, q)?;
                report.ternary_tensors += 1;
            }
            Placement::Embedding | Placement::Dense => {
                let view = st.tensor(name)?;
                let vals = bf16_values(&view, name)?;
                let shape: Vec<usize> = view.shape().iter().rev().copied().collect();
                let (dtype, data) = if t.placement == Placement::Embedding {
                    (GgmlType::BF16, view.data().to_vec())
                } else {
                    (
                        GgmlType::F32,
                        vals.iter().flat_map(|v| v.to_f32().to_le_bytes()).collect(),
                    )
                };
                report.bytes += data.len() as u64;
                w.add_tensor_bytes(t.gguf, shape, dtype, data)?;
                report.dense_tensors += 1;
            }
        }
    }
    let tmp = out.with_extension("gguf.partial");
    w.write_to_path(&tmp)?;
    std::fs::rename(&tmp, out).with_context(|| format!("rename to {}", out.display()))?;
    Ok(report)
}

/// Read the speaker projection head (`spk_proj.0` Linear 512→hidden,
/// `spk_proj.1` LayerNorm) from the packed checkpoint as f32.
pub fn read_spk_proj(ternary: &Path) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let bytes = std::fs::read(ternary).with_context(|| format!("read {}", ternary.display()))?;
    let st = SafeTensors::deserialize(&bytes).context("safetensors body")?;
    let mut out = HashMap::new();
    for name in st.names() {
        if let Some(leaf) = name.strip_prefix("spk_proj.") {
            let view = st.tensor(name)?;
            let vals = bf16_values(&view, name)?;
            out.insert(
                leaf.to_string(),
                (
                    vals.iter().map(|v| v.to_f32()).collect(),
                    view.shape().to_vec(),
                ),
            );
        }
    }
    ensure!(
        !out.is_empty(),
        "no spk_proj.* in {} — this checkpoint cannot be speaker-conditioned",
        ternary.display()
    );
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    fn pack(trits: &[u8]) -> Vec<u8> {
        trits
            .chunks(TRITS_PER_BYTE)
            .map(|c| c.iter().rev().fold(0u8, |acc, &t| acc * 3 + t))
            .collect()
    }

    #[test]
    fn trits_round_trip_through_base3_bytes() {
        let trits: Vec<u8> = (0..1003).map(|i| ((i * 7 + i / 3) % 3) as u8).collect();
        assert_eq!(unpack_trits(&pack(&trits), trits.len()).unwrap(), trits);
    }

    /// The transcoded blocks must dequantize to exactly `sign · scale` — the
    /// value kittenml's `unpack_state_dict` reconstructs.
    #[test]
    fn q2_0_blocks_dequantize_to_the_ternary_values() {
        let (rows, cols) = (3, 256);
        let trits: Vec<u8> = (0..rows * cols)
            .map(|i| ((i * 5 + i / 7) % 3) as u8)
            .collect();
        let scales: Vec<bf16> = (0..rows * cols / GROUP)
            .map(|i| bf16::from_f32(0.0123 * (i + 1) as f32))
            .collect();
        let q = trits_to_q2_0(&trits, &scales, rows, cols).unwrap();
        let deq = rlx_gguf::q2_dequant::dequant_q2_0(&q, rows * cols).unwrap();
        for (i, (&t, &d)) in trits.iter().zip(&deq).enumerate() {
            let want = (t as f32 - 1.0) * scales[i / GROUP].to_f32();
            assert_eq!(d, want, "element {i}");
        }
    }

    #[test]
    fn rejects_a_scale_f16_cannot_hold() {
        let trits = vec![1u8; GROUP];
        // bf16 keeps f32's exponent range; f16 overflows past 65504.
        let scales = [bf16::from_f32(1.0e6)];
        assert!(trits_to_q2_0(&trits, &scales, 1, GROUP).is_err());
    }

    /// The published checkpoint's header (names, shapes, packed index),
    /// planned with the real 1.7B config: every tensor is placed, the seven
    /// linears per layer are ternary, and each `::codes` / `::scale` pair is
    /// sized for its declared shape.
    #[test]
    fn published_manifest_plans_completely() {
        #[derive(Deserialize)]
        struct Entry {
            dtype: String,
            shape: Vec<usize>,
        }
        #[derive(Deserialize)]
        struct Manifest {
            metadata: serde_json::Value,
            tensors: std::collections::BTreeMap<String, Entry>,
        }
        let m: Manifest =
            serde_json::from_str(include_str!("../tests/fixtures/lm_ternary_manifest.json"))
                .unwrap();
        let md: HashMap<String, String> = [
            ("format", m.metadata["format"].as_str().unwrap().to_string()),
            (
                "group_size",
                m.metadata["group_size"].as_str().unwrap().to_string(),
            ),
            ("packed", m.metadata["packed"].to_string()),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let packed = packed_index(&md).unwrap();
        let cfg =
            Qwen3Config::from_json_str(include_str!("../tests/fixtures/lm_config.json")).unwrap();
        let planned = plan(m.tensors.keys().map(String::as_str), &packed, &cfg).unwrap();
        for t in &planned {
            if let Placement::Ternary { rows, cols } = t.placement {
                let codes = &m.tensors[&format!("{}::codes", t.hf)];
                let scale = &m.tensors[&format!("{}::scale", t.hf)];
                assert_eq!(codes.dtype, "U8");
                assert_eq!(
                    codes.shape,
                    [(rows * cols).div_ceil(TRITS_PER_BYTE)],
                    "{}",
                    t.hf
                );
                assert_eq!(scale.shape, [rows, cols / GROUP], "{}", t.hf);
            } else {
                assert_eq!(m.tensors[&t.hf].dtype, "BF16", "{}", t.hf);
            }
        }
        let mut gguf: Vec<&str> = planned.iter().map(|t| t.gguf.as_str()).collect();
        gguf.sort_unstable();
        gguf.dedup();
        assert_eq!(
            gguf.len(),
            planned.len(),
            "two tensors map to one GGUF name"
        );
        assert_eq!(
            m.tensors["model.embed_tokens.weight"].shape,
            [cfg.vocab_size, cfg.hidden_size]
        );
        assert_eq!(m.tensors["spk_proj.0.weight"].shape, [cfg.hidden_size, 512]);
    }

    #[test]
    fn every_qwen3_weight_has_a_gguf_name() {
        for leaf in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            assert!(
                gguf_name(&format!("model.layers.27.{leaf}")).is_some(),
                "{leaf}"
            );
        }
        assert_eq!(
            gguf_name("model.norm.weight").unwrap(),
            "output_norm.weight"
        );
        assert!(gguf_name("spk_proj.0.weight").is_none());
    }
}
