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

//! Packed / mmap Laguna weights — **no quant→F32 expand**.
//!
//! Matmul / expert packs stay as [`MatWeight::Packed`] metadata (bytes remain
//! in the retained [`rlx_core::GgufLoader`] mmap). Norms, router bias, and
//! other already-F32/F16/BF16 side tensors copy via
//! [`rlx_core::GgufLoader::take_native_f32`].

use crate::config::LagunaConfig;
use anyhow::{Context, Result, bail};
use rlx_core::GgufLoader;
use rlx_ir::quant::QuantScheme;
use std::collections::HashMap;

/// Graph / eager name → (GGUF key, scheme, logical shape).
pub type PackedParams = HashMap<String, (String, QuantScheme, Vec<usize>)>;

#[derive(Debug, Clone)]
pub enum MatWeight {
    /// K-quant / Q4 bytes stay in mmap; shape is safetensors-order.
    Packed {
        key: String,
        scheme: QuantScheme,
        shape: Vec<usize>,
    },
    /// Native F32/F16/BF16 host copy (norms, biases, small gates).
    F32(Vec<f32>),
}

impl MatWeight {
    pub fn is_packed(&self) -> bool {
        matches!(self, Self::Packed { .. })
    }

    pub fn packed_meta(&self) -> Option<(&str, QuantScheme, &[usize])> {
        match self {
            Self::Packed { key, scheme, shape } => Some((key.as_str(), *scheme, shape.as_slice())),
            Self::F32(_) => None,
        }
    }

    pub fn f32_bytes(&self) -> usize {
        match self {
            Self::F32(v) => v.len() * 4,
            Self::Packed { .. } => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub enum LagunaPackedFfn {
    Dense {
        gate: MatWeight,
        up: MatWeight,
        down: MatWeight,
    },
    Moe {
        router: MatWeight,
        gate_bias: Option<Vec<f32>>,
        gate_exps: MatWeight,
        up_exps: MatWeight,
        down_exps: MatWeight,
        shared_gate: MatWeight,
        shared_up: MatWeight,
        shared_down: MatWeight,
    },
}

#[derive(Debug, Clone)]
pub struct LagunaPackedLayer {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub wq: MatWeight,
    pub wk: MatWeight,
    pub wv: MatWeight,
    pub wo: MatWeight,
    /// Softplus attention output gate (per-head or per-element).
    pub wg: Option<MatWeight>,
    pub ffn: LagunaPackedFfn,
}

#[derive(Debug, Clone)]
pub struct LagunaPackedWeights {
    pub token_embd: MatWeight,
    pub output_norm: Vec<f32>,
    pub output: Option<MatWeight>,
    pub layers: Vec<LagunaPackedLayer>,
    /// Logical name → packed metadata for compile/upload later.
    pub packed_params: PackedParams,
    pub packed_tensor_count: usize,
    pub native_f32_bytes: usize,
}

impl LagunaPackedWeights {
    /// Resident host float footprint (norms / biases only) — not mmap size.
    pub fn estimate_resident_bytes(&self) -> usize {
        self.native_f32_bytes
    }

    pub fn from_loader(loader: &mut GgufLoader, cfg: &LagunaConfig) -> Result<Self> {
        if loader.architecture() != "laguna" {
            bail!(
                "LagunaPackedWeights: expected arch=laguna, got {}",
                loader.architecture()
            );
        }
        let mut packed_params = PackedParams::new();
        let mut packed_tensor_count = 0usize;
        let mut native_f32_bytes = 0usize;

        let token_embd = take_mat_required(
            loader,
            "token_embd.weight",
            /*expert*/ false,
            &mut packed_params,
            "embed",
            &mut packed_tensor_count,
            &mut native_f32_bytes,
        )?;
        if matches!(token_embd, MatWeight::F32(_)) {
            bail!(
                "LagunaPackedWeights: token_embd must stay packed (refusing full vocab F32 drain)"
            );
        }

        let output_norm = take_native(loader, "output_norm.weight", &mut native_f32_bytes)?;
        let output = if loader.file().tensors.contains_key("output.weight") {
            Some(take_mat_required(
                loader,
                "output.weight",
                false,
                &mut packed_params,
                "unembed",
                &mut packed_tensor_count,
                &mut native_f32_bytes,
            )?)
        } else {
            None
        };
        if let Some(MatWeight::F32(_)) = output.as_ref() {
            bail!("LagunaPackedWeights: output.weight must stay packed when present");
        }

        let n_layer = cfg.num_hidden_layers;
        let mut layers = Vec::with_capacity(n_layer);
        for il in 0..n_layer {
            layers.push(load_layer(
                loader,
                cfg,
                il,
                &mut packed_params,
                &mut packed_tensor_count,
                &mut native_f32_bytes,
            )?);
        }

        Ok(Self {
            token_embd,
            output_norm,
            output,
            layers,
            packed_params,
            packed_tensor_count,
            native_f32_bytes,
        })
    }
}

fn take_native(loader: &mut GgufLoader, key: &str, native_f32_bytes: &mut usize) -> Result<Vec<f32>> {
    let (data, _shape) = loader
        .take_native_f32(key)
        .with_context(|| format!("native F32 side tensor `{key}`"))?;
    *native_f32_bytes += data.len() * 4;
    Ok(data)
}

fn take_native_opt(
    loader: &mut GgufLoader,
    key: &str,
    native_f32_bytes: &mut usize,
) -> Result<Option<Vec<f32>>> {
    if !loader.file().tensors.contains_key(key) {
        return Ok(None);
    }
    Ok(Some(take_native(loader, key, native_f32_bytes)?))
}

fn take_mat_required(
    loader: &mut GgufLoader,
    key: &str,
    expert: bool,
    packed_params: &mut PackedParams,
    logical: &str,
    packed_tensor_count: &mut usize,
    native_f32_bytes: &mut usize,
) -> Result<MatWeight> {
    match loader.take_packed_metadata(key)? {
        Some((scheme, shape)) => {
            let shape = if expert {
                reshape_expert_packed(shape)?
            } else {
                shape
            };
            packed_params.insert(logical.to_string(), (key.to_string(), scheme, shape.clone()));
            *packed_tensor_count += 1;
            Ok(MatWeight::Packed {
                key: key.to_string(),
                scheme,
                shape,
            })
        }
        None => {
            // Uncompressed float — allowed only for small side tensors, not embd/experts.
            if expert {
                bail!(
                    "Laguna packed: expert tensor `{key}` is not a supported packed quant \
                     (use packed quants or RLX_LAGUNA_ALLOW_F32_EXPAND=1)"
                );
            }
            let (data, _shape) = loader.take_native_f32(key).with_context(|| {
                format!(
                    "Laguna packed: `{key}` is not packed-quant and not native float \
                     (opt in with RLX_LAGUNA_ALLOW_F32_EXPAND=1 for quant→F32)"
                )
            })?;
            *native_f32_bytes += data.len() * 4;
            Ok(MatWeight::F32(data))
        }
    }
}

fn reshape_expert_packed(shape: Vec<usize>) -> Result<Vec<usize>> {
    if shape.len() != 3 {
        bail!("MoE expert pack: expected rank-3 shape after reverse, got {shape:?}");
    }
    // ggml [ne0, ne1, ne2] = [k, n, E] (Laguna / llama.cpp) → reverse → [E, n, k].
    // Each expert slab is a 2-D `[n, k]` matrix for `gguf_matmul_bt`.
    Ok(shape)
}

fn load_layer(
    loader: &mut GgufLoader,
    cfg: &LagunaConfig,
    il: usize,
    packed_params: &mut PackedParams,
    packed_tensor_count: &mut usize,
    native_f32_bytes: &mut usize,
) -> Result<LagunaPackedLayer> {
    let attn_norm = take_native(
        loader,
        &format!("blk.{il}.attn_norm.weight"),
        native_f32_bytes,
    )?;
    let ffn_norm = take_native(
        loader,
        &format!("blk.{il}.ffn_norm.weight"),
        native_f32_bytes,
    )?;
    let q_norm = take_native_opt(
        loader,
        &format!("blk.{il}.attn_q_norm.weight"),
        native_f32_bytes,
    )?;
    let k_norm = take_native_opt(
        loader,
        &format!("blk.{il}.attn_k_norm.weight"),
        native_f32_bytes,
    )?;

    let wq = take_mat_required(
        loader,
        &format!("blk.{il}.attn_q.weight"),
        false,
        packed_params,
        &format!("layers.{il}.wq"),
        packed_tensor_count,
        native_f32_bytes,
    )?;
    let wk = take_mat_required(
        loader,
        &format!("blk.{il}.attn_k.weight"),
        false,
        packed_params,
        &format!("layers.{il}.wk"),
        packed_tensor_count,
        native_f32_bytes,
    )?;
    let wv = take_mat_required(
        loader,
        &format!("blk.{il}.attn_v.weight"),
        false,
        packed_params,
        &format!("layers.{il}.wv"),
        packed_tensor_count,
        native_f32_bytes,
    )?;
    let wo = take_mat_required(
        loader,
        &format!("blk.{il}.attn_output.weight"),
        false,
        packed_params,
        &format!("layers.{il}.wo"),
        packed_tensor_count,
        native_f32_bytes,
    )?;

    let gate_key = format!("blk.{il}.attn_gate.weight");
    let wg = if loader.file().tensors.contains_key(&gate_key) {
        Some(take_mat_required(
            loader,
            &gate_key,
            false,
            packed_params,
            &format!("layers.{il}.wg"),
            packed_tensor_count,
            native_f32_bytes,
        )?)
    } else {
        None
    };

    let ffn = if cfg.is_dense_mlp(il) {
        LagunaPackedFfn::Dense {
            gate: take_mat_required(
                loader,
                &format!("blk.{il}.ffn_gate.weight"),
                false,
                packed_params,
                &format!("layers.{il}.gate"),
                packed_tensor_count,
                native_f32_bytes,
            )?,
            up: take_mat_required(
                loader,
                &format!("blk.{il}.ffn_up.weight"),
                false,
                packed_params,
                &format!("layers.{il}.up"),
                packed_tensor_count,
                native_f32_bytes,
            )?,
            down: take_mat_required(
                loader,
                &format!("blk.{il}.ffn_down.weight"),
                false,
                packed_params,
                &format!("layers.{il}.down"),
                packed_tensor_count,
                native_f32_bytes,
            )?,
        }
    } else {
        let gate_bias = take_native_opt(
            loader,
            &format!("blk.{il}.exp_probs_b.bias"),
            native_f32_bytes,
        )?
        .or(take_native_opt(
            loader,
            &format!("blk.{il}.ffn_exp_probs_b.bias"),
            native_f32_bytes,
        )?);
        LagunaPackedFfn::Moe {
            router: take_mat_required(
                loader,
                &format!("blk.{il}.ffn_gate_inp.weight"),
                false,
                packed_params,
                &format!("layers.{il}.gate_weight"),
                packed_tensor_count,
                native_f32_bytes,
            )?,
            gate_bias,
            gate_exps: take_mat_required(
                loader,
                &format!("blk.{il}.ffn_gate_exps.weight"),
                true,
                packed_params,
                &format!("layers.{il}.gate_exps"),
                packed_tensor_count,
                native_f32_bytes,
            )?,
            up_exps: take_mat_required(
                loader,
                &format!("blk.{il}.ffn_up_exps.weight"),
                true,
                packed_params,
                &format!("layers.{il}.up_exps"),
                packed_tensor_count,
                native_f32_bytes,
            )?,
            down_exps: take_mat_required(
                loader,
                &format!("blk.{il}.ffn_down_exps.weight"),
                true,
                packed_params,
                &format!("layers.{il}.down_exps"),
                packed_tensor_count,
                native_f32_bytes,
            )?,
            shared_gate: take_mat_required(
                loader,
                &format!("blk.{il}.ffn_gate_shexp.weight"),
                false,
                packed_params,
                &format!("layers.{il}.shared_gate"),
                packed_tensor_count,
                native_f32_bytes,
            )?,
            shared_up: take_mat_required(
                loader,
                &format!("blk.{il}.ffn_up_shexp.weight"),
                false,
                packed_params,
                &format!("layers.{il}.shared_up"),
                packed_tensor_count,
                native_f32_bytes,
            )?,
            shared_down: take_mat_required(
                loader,
                &format!("blk.{il}.ffn_down_shexp.weight"),
                false,
                packed_params,
                &format!("layers.{il}.shared_down"),
                packed_tensor_count,
                native_f32_bytes,
            )?,
        }
    };

    Ok(LagunaPackedLayer {
        attn_norm,
        ffn_norm,
        q_norm,
        k_norm,
        wq,
        wk,
        wv,
        wo,
        wg,
        ffn,
    })
}

/// Process RSS in bytes (macOS `ps` reports KB). Best-effort for CLI checks.
pub fn process_rss_bytes() -> Option<u64> {
    let pid = std::process::id();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let kb: u64 = s.trim().parse().ok()?;
    Some(kb.saturating_mul(1024))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_gguf::{GgmlType, GgufWriter, MetaValue, bytes_for_public};

    fn write_tiny_packed_fixture(path: &std::path::Path) {
        // Dims chosen so every Q4_K mat has n_elements % 256 == 0 and
        // row lengths match `gguf_matmul_bt` / embed gather.
        const H: u32 = 256;
        const INTER: u32 = 256;
        const HEADS: u32 = 4;
        const KV: u32 = 2;
        const HD: u32 = 64; // H / HEADS
        const VOCAB: u32 = 256;

        let mut w = GgufWriter::new();
        w.set_arch("laguna");
        w.set_meta("laguna.block_count", MetaValue::U32(1));
        w.set_meta("laguna.embedding_length", MetaValue::U32(H));
        w.set_meta("laguna.feed_forward_length", MetaValue::U32(INTER));
        w.set_meta("laguna.attention.head_count", MetaValue::U32(HEADS));
        w.set_meta("laguna.attention.head_count_kv", MetaValue::U32(KV));
        w.set_meta("laguna.attention.key_length", MetaValue::U32(HD));
        w.set_meta("laguna.attention.value_length", MetaValue::U32(HD));
        w.set_meta("laguna.expert_count", MetaValue::U32(0));
        w.set_meta("laguna.expert_used_count", MetaValue::U32(0));
        w.set_meta("laguna.expert_feed_forward_length", MetaValue::U32(8));
        w.set_meta("laguna.vocab_size", MetaValue::U32(VOCAB));
        w.set_meta("laguna.context_length", MetaValue::U32(128));
        w.set_meta("laguna.leading_dense_block_count", MetaValue::U32(1));

        let q4 = |n: usize| {
            let nbytes = bytes_for_public(GgmlType::Q4K, n).unwrap();
            vec![0u8; nbytes]
        };
        let f32_bytes = |n: usize| -> Vec<u8> {
            let v = vec![1.0f32; n];
            v.into_iter().flat_map(|x| x.to_le_bytes()).collect()
        };
        let h = H as usize;
        let inter = INTER as usize;
        let q_dim = (HEADS * HD) as usize;
        let kv_dim = (KV * HD) as usize;
        let vocab = VOCAB as usize;

        w.add_tensor_bytes(
            "token_embd.weight",
            vec![h, vocab],
            GgmlType::Q4K,
            q4(h * vocab),
        )
        .unwrap();
        w.add_tensor_bytes("output_norm.weight", vec![h], GgmlType::F32, f32_bytes(h))
            .unwrap();
        for name in [
            "attn_norm.weight",
            "ffn_norm.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
        ] {
            w.add_tensor_bytes(
                &format!("blk.0.{name}"),
                vec![h],
                GgmlType::F32,
                f32_bytes(h),
            )
            .unwrap();
        }
        // ggml [in, out] pairs for HF [out, in] after reverse
        for (name, shape, n) in [
            ("attn_q.weight", vec![h, q_dim], h * q_dim),
            ("attn_k.weight", vec![h, kv_dim], h * kv_dim),
            ("attn_v.weight", vec![h, kv_dim], h * kv_dim),
            ("attn_output.weight", vec![q_dim, h], q_dim * h),
            ("ffn_gate.weight", vec![h, inter], h * inter),
            ("ffn_up.weight", vec![h, inter], h * inter),
            ("ffn_down.weight", vec![inter, h], inter * h),
        ] {
            w.add_tensor_bytes(&format!("blk.0.{name}"), shape, GgmlType::Q4K, q4(n))
                .unwrap();
        }
        w.write_to_path(path).unwrap();
    }

    #[test]
    fn packed_load_tiny_dense_fixture() {
        let path = std::env::temp_dir().join("rlx_laguna_packed_tiny.gguf");
        write_tiny_packed_fixture(&path);
        let mut loader = GgufLoader::from_file(path.to_str().unwrap()).unwrap();
        let mut cfg = LagunaConfig::from_gguf(loader.file()).unwrap();
        cfg.mlp_layer_types = vec![crate::config::MlpLayerType::Dense];
        cfg.layer_types = vec![crate::config::AttnLayerType::Full];
        cfg.num_hidden_layers = 1;
        cfg.hidden_size = 256;
        cfg.intermediate_size = 256;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 2;
        cfg.head_dim = 64;
        cfg.vocab_size = 256;
        cfg.num_attention_heads_per_layer = vec![4];
        cfg.gating = crate::config::AttnGating::Off;

        let w = LagunaPackedWeights::from_loader(&mut loader, &cfg).unwrap();
        assert!(w.token_embd.is_packed());
        assert!(w.packed_tensor_count >= 7);
        assert!(w.estimate_resident_bytes() > 0);
        assert!(w.estimate_resident_bytes() < 10_000_000);
        assert!(matches!(w.layers[0].ffn, LagunaPackedFfn::Dense { .. }));
        let mut loader2 = GgufLoader::from_file(path.to_str().unwrap()).unwrap();
        let err = {
            use rlx_core::WeightLoader;
            loader2.take("blk.0.ffn_down.weight").unwrap_err()
        };
        assert!(
            format!("{err:#}").contains("disabled") || format!("{err:#}").contains("FORBIDDEN"),
            "expected F32 expand blocked, got: {err:#}"
        );

        let next =
            crate::packed_forward::greedy_next(&cfg, &w, &loader, &[1, 2, 3], None).unwrap();
        assert!((next as usize) < cfg.vocab_size);
        let out =
            crate::packed_forward::generate(&cfg, &w, &loader, &[1, 2], 2, |_| {}, None).unwrap();
        assert_eq!(out.len(), 4);
        // Cached generate must match full-recompute greedy_next loop.
        let mut recomputed = vec![1u32, 2];
        for _ in 0..2 {
            let next = crate::packed_forward::greedy_next(
                &cfg,
                &w,
                &loader,
                &recomputed,
                None,
            )
            .unwrap();
            recomputed.push(next);
        }
        assert_eq!(out, recomputed);
        let _ = std::fs::remove_file(&path);
    }
}
