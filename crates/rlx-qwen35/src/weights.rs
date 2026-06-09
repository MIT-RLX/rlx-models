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

//! Qwen3.5 / Qwen3.6 weight loader.
//!
//! Resolves every per-layer tensor named by `llama.cpp`'s
//! `src/models/qwen35.cpp` (commit referenced in
//! [`super::SOURCE_REF`]) from a `WeightLoader` (today only
//! `GgufLoader` — the unsloth/froggeric files don't ship
//! safetensors). Each tensor is dequantized to `Vec<f32>` and
//! shape-checked against the config.
//!
//! The resulting [`Qwen35Weights`] holds three groups:
//!
//!   - **Embed/output**: `token_embd`, `output_norm`, optional `output`
//!     (tied to embed if missing).
//!   - **Trunk layers** `0..num_main_layers`: each is one of
//!     - [`Qwen35TrunkLayer::Linear`] (gated DeltaNet block) — for
//!       layers where `(i + 1) % full_attention_interval != 0`.
//!     - [`Qwen35TrunkLayer::FullAttn`] (standard attention) — for
//!       layers where `(i + 1) % full_attention_interval == 0`.
//!   - **MTP layers** `num_main_layers..num_hidden_layers`: full
//!     attention plus the NextN-specific `eh_proj` / `enorm` / `hnorm`
//!     / optional `embed_tokens` / `shared_head_*` tensors.
//!
//! This struct is the input the future forward-graph builder consumes;
//! it intentionally doesn't depend on `Graph` / `Session` so it can be
//! unit-tested against a tiny synthesized GGUF.

use crate::config::Qwen35Config;
use anyhow::{Context, Result, anyhow};
use rlx_core::weight_loader::{GgufLoader, WeightLoader};
use rlx_ir::quant::QuantScheme;

/// Storage variant for matmul weight tensors. The big projections
/// (qkv / gate / ffn / lm_head) dominate the load footprint; the
/// `Packed` variant keeps GGUF K-quant bytes in-place so the graph
/// can emit `Op::DequantMatMul` instead of a full F32 dequant.
///
/// Norm weights, conv kernels, scalar params etc. stay as
/// [`Vec<f32>`] in the layer structs (their footprint is negligible
/// and the `RmsNorm` / `Conv` ops don't have a packed variant).
#[derive(Debug, Clone)]
pub enum MatWeight {
    /// Already dequantized to f32, row-major `[out, in]`. The
    /// builder transposes to `[in, out]` before issuing `MatMul`.
    F32(Vec<f32>),
    /// GGUF-packed K-quant metadata only. The actual bytes are
    /// looked up in the loader at upload time via
    /// [`rlx_core::weight_loader::GgufLoader::tensor_bytes_borrowed`]
    /// — eliminates the per-tensor `Vec<u8>` allocation that
    /// otherwise costs ~16 GB of memcpy on Qwen3.6-27B Q4_K_M.
    ///
    /// `key` is the loader-resolvable name (post-HF↔GGUF mapping);
    /// `shape` is `[out, in]` after the safetensors-style dim
    /// reversal.
    Packed {
        key: String,
        scheme: QuantScheme,
        shape: Vec<usize>,
    },
}

impl MatWeight {
    pub fn len(&self) -> usize {
        match self {
            MatWeight::F32(v) => v.len(),
            MatWeight::Packed { shape, .. } => shape.iter().product(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// `[out, in]` on-disk shape. For the F32 variant the caller is
    /// expected to track this externally (we return an empty Vec).
    pub fn shape(&self) -> &[usize] {
        match self {
            MatWeight::F32(_) => &[],
            MatWeight::Packed { shape, .. } => shape,
        }
    }
    pub fn is_packed(&self) -> bool {
        matches!(self, MatWeight::Packed { .. })
    }
    /// Loader-resolvable key for the packed variant. `None` for F32.
    pub fn packed_key(&self) -> Option<&str> {
        match self {
            MatWeight::F32(_) => None,
            MatWeight::Packed { key, .. } => Some(key.as_str()),
        }
    }
}

/// Per-layer feed-forward: dense SwiGLU or MoE (routed + gated shared expert).
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Qwen35LayerFfn {
    Dense {
        gate: MatWeight,
        up: MatWeight,
        down: MatWeight,
    },
    Moe(Qwen35MoeFfn),
}

/// MoE FFN tensors for one decoder layer (trunk or MTP).
#[derive(Debug, Clone)]
pub struct Qwen35MoeFfn {
    /// Router logits: `[n_embd, n_expert]`.
    pub router: MatWeight,
    /// Expert gate projections: GroupedMatMul layout `[n_expert, n_embd, n_ff_exp]`.
    pub gate_exps: MatWeight,
    pub up_exps: MatWeight,
    /// Expert down projections: `[n_expert, n_ff_exp, n_embd]`.
    pub down_exps: MatWeight,
    /// Shared-expert router weight `[n_embd]` (`ffn_gate_inp_shexp`).
    pub shared_router: Vec<f32>,
    pub shared_gate: MatWeight,
    pub shared_up: MatWeight,
    pub shared_down: MatWeight,
}

/// One trunk-layer tensor bundle. Either a gated-DeltaNet "linear
/// attention" block or a standard full-attention block.
#[derive(Debug, Clone)]
pub enum Qwen35TrunkLayer {
    Linear(Qwen35LinearLayer),
    FullAttn(Qwen35FullAttnLayer),
}

/// Gated DeltaNet ("linear attention") trunk layer. Mirrors
/// `qwen35.cpp::load_block_trunk` for the `is_recurrent(il)` branch.
#[derive(Debug, Clone)]
pub struct Qwen35LinearLayer {
    /// `[n_embd]`
    pub attn_norm: Vec<f32>,
    /// `[n_embd]`
    pub attn_post_norm: Vec<f32>,
    /// Fused `[gate, x, k, B, C]`-style projection:
    /// `[n_embd, 2*key_dim + value_dim]` with `key_dim =
    /// ssm_state*group_count`, `value_dim = ssm_state*dt_rank`.
    pub attn_qkv: MatWeight,
    /// `[n_embd, value_dim]` — z gating projection.
    pub attn_gate: MatWeight,
    /// Depthwise 1-D conv weights over the fused channels:
    /// `[ssm_conv_kernel, key_dim*2 + value_dim]`. Kept dense —
    /// `Op::Conv` has no packed variant and the conv kernel is
    /// tiny vs the projections.
    pub ssm_conv1d: Vec<f32>,
    /// `[dt_rank]` — delta-t bias.
    pub ssm_dt_bias: Vec<f32>,
    /// `[dt_rank]` — A (no-scan; used directly as scalar gate
    /// multiplier per head).
    pub ssm_a: Vec<f32>,
    /// `[n_embd, dt_rank]` — β projection.
    pub ssm_beta: MatWeight,
    /// `[n_embd, dt_rank]` — α projection.
    pub ssm_alpha: MatWeight,
    /// `[ssm_state]` — per-state-row RMS norm gate.
    pub ssm_norm: Vec<f32>,
    /// `[value_dim, n_embd]` — output projection.
    pub ssm_out: MatWeight,
    pub ffn: Qwen35LayerFfn,
}

/// Standard full-attention trunk layer (interspersed every
/// `full_attention_interval` blocks). Per `qwen35.cpp::load_block_trunk`
/// non-recurrent branch.
#[derive(Debug, Clone)]
pub struct Qwen35FullAttnLayer {
    pub attn_norm: Vec<f32>,
    pub attn_post_norm: Vec<f32>,
    /// `[n_embd, n_embd_head_k * n_head * 2]` — joint Q + gate
    /// projection (Qwen3-Next style).
    pub attn_q_gate: MatWeight,
    pub attn_k: MatWeight,
    pub attn_v: MatWeight,
    pub attn_output: MatWeight,
    pub attn_q_norm: Vec<f32>,
    pub attn_k_norm: Vec<f32>,
    pub ffn: Qwen35LayerFfn,
}

/// One MTP (NextN) layer. Per `qwen35.cpp::load_block_mtp`.
#[derive(Debug, Clone)]
pub struct Qwen35MtpLayer {
    /// Base full-attention sub-block (shares the same shapes as
    /// [`Qwen35FullAttnLayer`]).
    pub base: Qwen35FullAttnLayer,
    /// `[2*n_embd, n_embd]` — concatenated [e, h] → hidden projection.
    pub eh_proj: MatWeight,
    /// `[n_embd]`
    pub enorm: Vec<f32>,
    /// `[n_embd]`
    pub hnorm: Vec<f32>,
    /// `[n_embd, n_vocab]` — optional; if absent the MTP head reuses
    /// the trunk's `token_embd`.
    pub embed_tokens: Option<MatWeight>,
    /// `[n_embd, n_vocab]` — optional; if absent the MTP head reuses
    /// the trunk's `output` (or tied `token_embd`).
    pub shared_head_head: Option<MatWeight>,
    /// `[n_embd]` — optional; if absent the MTP head reuses
    /// `output_norm`.
    pub shared_head_norm: Option<Vec<f32>>,
}

/// Top-level Qwen3.5 / Qwen3.6 weight bundle.
#[derive(Debug, Clone)]
pub struct Qwen35Weights {
    /// `[n_vocab, n_embd]`. Kept as `Vec<f32>` because the embed
    /// table is always materialized for the `Op::Gather` lookup
    /// (no packed-gather kernel today).
    pub token_embd: Vec<f32>,
    /// `[n_embd]`
    pub output_norm: Vec<f32>,
    /// `[n_vocab, n_embd]` — optional; tied to `token_embd` if absent.
    /// May be packed when loaded via `from_loader_packed`.
    pub output: Option<MatWeight>,
    /// Packed K-quant bytes for tied LM head (`token_embd.weight`) when
    /// the GGUF table is quantized. Gather still uses [`Self::token_embd`]
    /// (eager F32); this is only for `DequantMatMul` on the logits path.
    pub token_embd_lm: Option<MatWeight>,
    pub trunk_layers: Vec<Qwen35TrunkLayer>,
    pub mtp_layers: Vec<Qwen35MtpLayer>,
}

impl Qwen35Weights {
    /// LM head width: tied embeddings use the full embedding table
    /// (often wider than `cfg.vocab_size` on Qwen3.5 checkpoints).
    pub fn lm_vocab_size(&self, cfg: &Qwen35Config) -> usize {
        if self.token_embd.is_empty() || cfg.hidden_size == 0 {
            return cfg.vocab_size;
        }
        self.token_embd.len() / cfg.hidden_size
    }
}

impl Qwen35Weights {
    /// Resolve every named tensor for a Qwen3.5 file. Drains the
    /// loader's `take()` cache as it goes — the caller should not
    /// expect to read these tensors back out afterwards. Errors on
    /// the first missing required tensor with a precise key + reason.
    ///
    /// All matmul weights are loaded as `MatWeight::F32` (eager
    /// dequant). For ≥14 B GGUFs use [`Self::from_loader_packed`]
    /// to keep K-quant bytes packed in the arena.
    pub fn from_loader(loader: &mut dyn WeightLoader, cfg: &Qwen35Config) -> Result<Self> {
        Self::from_loader_inner(loader, cfg, /*pack*/ None)
    }

    /// Variant of [`Self::from_loader`] that keeps every K-quant
    /// matmul weight packed (Q4_K / Q5_K / Q6_K / Q8_K) so the
    /// builder can emit `Op::DequantMatMul`. Non-K-quant tensors
    /// (F32, F16, BF16, legacy Q4_0/Q5_0/Q8_0) still fall through
    /// to the dequant-to-F32 path.
    ///
    /// Memory savings on Qwen3.6-27B-Q4_K_M: ~65 GB → ~16 GB.
    pub fn from_loader_packed(loader: &mut GgufLoader, cfg: &Qwen35Config) -> Result<Self> {
        // Capture the raw pointer first so the &mut borrow that
        // follows doesn't alias it (Rust's borrow checker rejects
        // `&mut loader` and `loader as *mut` in the same call).
        let pack_via = loader as *mut GgufLoader;
        Self::from_loader_inner(loader, cfg, Some(pack_via))
    }

    fn from_loader_inner(
        loader: &mut dyn WeightLoader,
        cfg: &Qwen35Config,
        pack_via: Option<*mut GgufLoader>,
    ) -> Result<Self> {
        let n_layer = cfg.num_hidden_layers;
        let nextn = cfg.nextn_predict_layers;
        if nextn >= n_layer {
            return Err(anyhow!(
                "qwen35: nextn_predict_layers={nextn} must be < num_hidden_layers={n_layer}",
            ));
        }
        let n_main = n_layer - nextn;
        let interval = cfg.full_attention_interval.max(1);

        let token_embd_lm = pack_via.and_then(|p| peek_gguf_packed_mat(p, "token_embd.weight"));
        let token_embd = take_f32(loader, "token_embd.weight")?;
        let output_norm = take_f32(loader, "output_norm.weight")?;
        let output = take_mat(loader, "output.weight", pack_via).ok();

        let mut trunk_layers = Vec::with_capacity(n_main);
        for il in 0..n_main {
            let is_full_attn = ((il + 1) % interval) == 0;
            if is_full_attn {
                trunk_layers.push(Qwen35TrunkLayer::FullAttn(load_full_attn_layer(
                    loader, il, cfg, pack_via,
                )?));
            } else {
                trunk_layers.push(Qwen35TrunkLayer::Linear(load_linear_layer(
                    loader, il, cfg, pack_via,
                )?));
            }
        }

        let mut mtp_layers = Vec::with_capacity(nextn);
        for il in n_main..n_layer {
            mtp_layers.push(load_mtp_layer(loader, il, cfg, pack_via)?);
        }

        Ok(Self {
            token_embd,
            output_norm,
            output,
            token_embd_lm,
            trunk_layers,
            mtp_layers,
        })
    }
}

fn peek_gguf_packed_mat(loader: *mut GgufLoader, key: &str) -> Option<MatWeight> {
    use rlx_gguf::GgmlType;
    use rlx_ir::quant::QuantScheme;
    let g = unsafe { &*loader };
    let t = g.file().get(key)?;
    let scheme = match t.dtype {
        GgmlType::Q4K => QuantScheme::GgufQ4K,
        GgmlType::Q5K => QuantScheme::GgufQ5K,
        GgmlType::Q6K => QuantScheme::GgufQ6K,
        GgmlType::Q8K => QuantScheme::GgufQ8K,
        _ => return None,
    };
    let mut shape = t.shape.clone();
    shape.reverse();
    Some(MatWeight::Packed {
        key: key.to_string(),
        scheme,
        shape,
    })
}

fn take_f32(loader: &mut dyn WeightLoader, key: &str) -> Result<Vec<f32>> {
    let (data, _shape) = loader
        .take(key)
        .with_context(|| format!("missing tensor: {key}"))?;
    Ok(data)
}

/// Take a matmul tensor: if `pack_via` is provided, try the packed
/// loader first and only fall back to F32 dequant when the source
/// tensor isn't a K-quant. SAFETY: `pack_via` must point at the
/// same `GgufLoader` instance backing `loader`; the wrapper exists
/// purely to thread the concrete-type method through the dyn-trait
/// API. Constructed by [`Qwen35Weights::from_loader_packed`].
fn take_mat(
    loader: &mut dyn WeightLoader,
    key: &str,
    pack_via: Option<*mut GgufLoader>,
) -> Result<MatWeight> {
    if let Some(p) = pack_via {
        // SAFETY: `p` was derived from the same `&mut GgufLoader`
        // the caller already has exclusive access to via `loader`;
        // we use it only to call `take_packed_metadata`, which
        // doesn't alias with anything else inside this function.
        let g: &mut GgufLoader = unsafe { &mut *p };
        match g.take_packed_metadata(key) {
            Ok(Some((scheme, shape))) => {
                return Ok(MatWeight::Packed {
                    key: key.to_string(),
                    scheme,
                    shape,
                });
            }
            Ok(None) => { /* not a K-quant; fall through to F32 */ }
            Err(_e) => { /* missing or already-taken; F32 will error */ }
        }
    }
    let (data, _shape) = loader
        .take(key)
        .with_context(|| format!("missing tensor: {key}"))?;
    Ok(MatWeight::F32(data))
}

/// Expert 3-D tensors: try packed K-quant first (native GGML layout,
/// expert dimension outermost). F32 fallback permutes to `[E, K, N]`.
fn take_expert_mat(
    loader: &mut dyn WeightLoader,
    key: &str,
    pack_via: Option<*mut GgufLoader>,
) -> Result<MatWeight> {
    if let Some(p) = pack_via {
        let g: &mut GgufLoader = unsafe { &mut *p };
        if let Ok(Some((scheme, shape))) = g.take_packed_metadata(key) {
            if shape.len() == 3 {
                let n_expert = shape[2];
                return Ok(MatWeight::Packed {
                    key: key.to_string(),
                    scheme,
                    shape: vec![n_expert, shape[0], shape[1]],
                });
            }
        }
    }
    let (data, shape) = loader
        .take(key)
        .with_context(|| format!("missing MoE tensor: {key}"))?;
    if shape.len() != 3 {
        return Err(anyhow!(
            "MoE tensor {key}: expected rank-3 GGML shape, got {shape:?}"
        ));
    }
    let n_expert = shape[2];
    let permuted = permute_ggml_expert_to_grouped(&data, shape[0], shape[1], n_expert);
    Ok(MatWeight::F32(permuted))
}

fn permute_ggml_expert_to_grouped(data: &[f32], d0: usize, d1: usize, n_expert: usize) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    for e in 0..n_expert {
        for i0 in 0..d0 {
            for i1 in 0..d1 {
                let src = i0 + d0 * i1 + d0 * d1 * e;
                let dst = e * (d0 * d1) + i0 * d1 + i1;
                out[dst] = data[src];
            }
        }
    }
    out
}

fn load_layer_ffn(
    loader: &mut dyn WeightLoader,
    il: usize,
    cfg: &Qwen35Config,
    pack_via: Option<*mut GgufLoader>,
) -> Result<Qwen35LayerFfn> {
    let p = |suffix: &str| format!("blk.{il}.{suffix}");
    if cfg.is_moe() {
        Ok(Qwen35LayerFfn::Moe(load_moe_ffn(
            loader, il, cfg, pack_via,
        )?))
    } else {
        Ok(Qwen35LayerFfn::Dense {
            gate: take_mat(loader, &p("ffn_gate.weight"), pack_via)?,
            up: take_mat(loader, &p("ffn_up.weight"), pack_via)?,
            down: take_mat(loader, &p("ffn_down.weight"), pack_via)?,
        })
    }
}

fn load_moe_ffn(
    loader: &mut dyn WeightLoader,
    il: usize,
    cfg: &Qwen35Config,
    pack_via: Option<*mut GgufLoader>,
) -> Result<Qwen35MoeFfn> {
    let p = |suffix: &str| format!("blk.{il}.{suffix}");
    let router = take_mat(loader, &p("ffn_gate_inp.weight"), pack_via)?;
    let down_exps = take_expert_mat(loader, &p("ffn_down_exps.weight"), pack_via)?;
    let (gate_exps, up_exps) = match (
        take_expert_mat(loader, &p("ffn_gate_exps.weight"), pack_via),
        take_expert_mat(loader, &p("ffn_up_exps.weight"), pack_via),
    ) {
        (Ok(g), Ok(u)) => (g, u),
        _ => {
            let fused = take_expert_mat(loader, &p("ffn_gate_up_exps.weight"), pack_via)?;
            split_fused_gate_up_exps(fused, cfg)?
        }
    };
    Ok(Qwen35MoeFfn {
        router,
        gate_exps,
        up_exps,
        down_exps,
        shared_router: take_f32(loader, &p("ffn_gate_inp_shexp.weight"))?,
        shared_gate: take_mat(loader, &p("ffn_gate_shexp.weight"), pack_via)?,
        shared_up: take_mat(loader, &p("ffn_up_shexp.weight"), pack_via)?,
        shared_down: take_mat(loader, &p("ffn_down_shexp.weight"), pack_via)?,
    })
}

/// Split fused `ffn_gate_up_exps` after permute: `[n_expert, 2*n_ff, n_embd]`.
fn split_fused_gate_up_exps(
    fused: MatWeight,
    cfg: &Qwen35Config,
) -> Result<(MatWeight, MatWeight)> {
    let MatWeight::F32(data) = fused else {
        return Err(anyhow!(
            "fused gate_up_exps must be F32 after take_expert_mat"
        ));
    };
    let n_ff = cfg.expert_ffn_dim();
    let n_embd = cfg.hidden_size;
    let n_expert = cfg.num_experts;
    let expected = 2 * n_ff * n_embd * n_expert;
    if data.len() != expected {
        return Err(anyhow!(
            "fused gate_up_exps: len {} != 2*{n_ff}*{n_embd}*{n_expert}",
            data.len()
        ));
    }
    let expert_slab = 2 * n_ff * n_embd;
    let half = n_ff * n_embd;
    let mut gate = Vec::with_capacity(n_expert * half);
    let mut up = Vec::with_capacity(n_expert * half);
    for e in 0..n_expert {
        let base = e * expert_slab;
        gate.extend_from_slice(&data[base..base + half]);
        up.extend_from_slice(&data[base + half..base + expert_slab]);
    }
    Ok((MatWeight::F32(gate), MatWeight::F32(up)))
}

fn load_linear_layer(
    loader: &mut dyn WeightLoader,
    il: usize,
    cfg: &Qwen35Config,
    pack_via: Option<*mut GgufLoader>,
) -> Result<Qwen35LinearLayer> {
    let p = |suffix: &str| format!("blk.{il}.{suffix}");
    Ok(Qwen35LinearLayer {
        attn_norm: take_f32(loader, &p("attn_norm.weight"))?,
        attn_post_norm: take_f32(loader, &p("post_attention_norm.weight"))?,
        attn_qkv: take_mat(loader, &p("attn_qkv.weight"), pack_via)?,
        attn_gate: take_mat(loader, &p("attn_gate.weight"), pack_via)?,
        ssm_conv1d: take_f32(loader, &p("ssm_conv1d.weight"))?,
        ssm_dt_bias: take_f32(loader, &p("ssm_dt.bias"))?,
        ssm_a: take_f32(loader, &p("ssm_a"))?,
        ssm_beta: take_mat(loader, &p("ssm_beta.weight"), pack_via)?,
        ssm_alpha: take_mat(loader, &p("ssm_alpha.weight"), pack_via)?,
        ssm_norm: take_f32(loader, &p("ssm_norm.weight"))?,
        ssm_out: take_mat(loader, &p("ssm_out.weight"), pack_via)?,
        ffn: load_layer_ffn(loader, il, cfg, pack_via)?,
    })
}

fn load_full_attn_layer(
    loader: &mut dyn WeightLoader,
    il: usize,
    cfg: &Qwen35Config,
    pack_via: Option<*mut GgufLoader>,
) -> Result<Qwen35FullAttnLayer> {
    let p = |suffix: &str| format!("blk.{il}.{suffix}");
    Ok(Qwen35FullAttnLayer {
        attn_norm: take_f32(loader, &p("attn_norm.weight"))?,
        attn_post_norm: take_f32(loader, &p("post_attention_norm.weight"))?,
        attn_q_gate: take_mat(loader, &p("attn_q.weight"), pack_via)?,
        attn_k: take_mat(loader, &p("attn_k.weight"), pack_via)?,
        attn_v: take_mat(loader, &p("attn_v.weight"), pack_via)?,
        attn_output: take_mat(loader, &p("attn_output.weight"), pack_via)?,
        attn_q_norm: take_f32(loader, &p("attn_q_norm.weight"))?,
        attn_k_norm: take_f32(loader, &p("attn_k_norm.weight"))?,
        ffn: load_layer_ffn(loader, il, cfg, pack_via)?,
    })
}

fn load_mtp_layer(
    loader: &mut dyn WeightLoader,
    il: usize,
    cfg: &Qwen35Config,
    pack_via: Option<*mut GgufLoader>,
) -> Result<Qwen35MtpLayer> {
    let base = load_full_attn_layer(loader, il, cfg, pack_via)?;
    let p = |suffix: &str| format!("blk.{il}.nextn.{suffix}");
    let eh_proj = take_mat(loader, &p("eh_proj.weight"), pack_via)?;
    let enorm = take_f32(loader, &p("enorm.weight"))?;
    let hnorm = take_f32(loader, &p("hnorm.weight"))?;
    let embed_tokens = take_mat(loader, &p("embed_tokens.weight"), pack_via).ok();
    let shared_head_head = take_mat(loader, &p("shared_head_head.weight"), pack_via).ok();
    let shared_head_norm = take_f32(loader, &p("shared_head_norm.weight")).ok();
    Ok(Qwen35MtpLayer {
        base,
        eh_proj,
        enorm,
        hnorm,
        embed_tokens,
        shared_head_head,
        shared_head_norm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Tiny in-memory `WeightLoader` that hands back a unique
    /// constant-valued vector for each requested key. The shape we
    /// return doesn't matter for this basic test — we only verify
    /// that the right key set was requested and that the resulting
    /// `Qwen35Weights` slots them into the right struct fields.
    struct MockLoader {
        store: HashMap<String, (Vec<f32>, Vec<usize>)>,
    }

    impl WeightLoader for MockLoader {
        fn len(&self) -> usize {
            self.store.len()
        }
        fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
            self.store
                .remove(key)
                .ok_or_else(|| anyhow!("mock: missing key {key}"))
        }
        fn take_transposed(&mut self, _key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
            unimplemented!("mock loader: not used by qwen35 loader")
        }
        fn remaining_keys(&self) -> Vec<String> {
            self.store.keys().cloned().collect()
        }
    }

    fn populate(store: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str, marker: f32) {
        store.insert(key.to_string(), (vec![marker], vec![1]));
    }

    fn build_synth_store(cfg: &Qwen35Config) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
        let mut store = HashMap::new();
        populate(&mut store, "token_embd.weight", 1.0);
        populate(&mut store, "output_norm.weight", 2.0);
        // `output.weight` intentionally omitted to exercise the
        // tied-embeddings path.

        let n_main = cfg.num_hidden_layers - cfg.nextn_predict_layers;
        let interval = cfg.full_attention_interval.max(1);

        for il in 0..n_main {
            let is_full_attn = ((il + 1) % interval) == 0;
            let p = |suf: &str| format!("blk.{il}.{suf}");
            if is_full_attn {
                for k in [
                    "attn_norm.weight",
                    "post_attention_norm.weight",
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                    "attn_q_norm.weight",
                    "attn_k_norm.weight",
                    "ffn_gate.weight",
                    "ffn_down.weight",
                    "ffn_up.weight",
                ] {
                    populate(&mut store, &p(k), 10.0 + il as f32);
                }
            } else {
                for k in [
                    "attn_norm.weight",
                    "post_attention_norm.weight",
                    "attn_qkv.weight",
                    "attn_gate.weight",
                    "ssm_conv1d.weight",
                    "ssm_dt.bias",
                    "ssm_a",
                    "ssm_beta.weight",
                    "ssm_alpha.weight",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                    "ffn_gate.weight",
                    "ffn_down.weight",
                    "ffn_up.weight",
                ] {
                    populate(&mut store, &p(k), 100.0 + il as f32);
                }
            }
        }

        for il in n_main..cfg.num_hidden_layers {
            let p = |suf: &str| format!("blk.{il}.{suf}");
            for k in [
                "attn_norm.weight",
                "post_attention_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "ffn_gate.weight",
                "ffn_down.weight",
                "ffn_up.weight",
                "nextn.eh_proj.weight",
                "nextn.enorm.weight",
                "nextn.hnorm.weight",
            ] {
                populate(&mut store, &p(k), 1000.0 + il as f32);
            }
        }
        store
    }

    fn dummy_cfg() -> Qwen35Config {
        // Mirrors Qwen3.5-0.8B: 25 layers, 1 MTP, full_attn every 4.
        // The synthetic store ignores hidden_size etc., so the
        // loader's shape checks fall back to whatever the GGUF
        // reports (here single-element [1]).
        Qwen35Config {
            vocab_size: 0,
            hidden_size: 1024,
            intermediate_size: 3584,
            num_hidden_layers: 6,
            nextn_predict_layers: 1,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            key_length: 128,
            value_length: 128,
            max_position_embeddings: 40_960,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            rope_dim_count: 64,
            rope_dim_sections: vec![],
            full_attention_interval: 4,
            ssm_conv_kernel: 4,
            ssm_group_count: 16,
            ssm_inner_size: 2048,
            ssm_state_size: 128,
            ssm_time_step_rank: 16,
            tie_word_embeddings: true,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        }
    }

    /// 6-layer trunk (interval=4 → layer 3 is full-attn, others linear) +
    /// 1 MTP layer. Verify each layer is classified correctly and the
    /// MTP block exists with the NextN tensors loaded.
    #[test]
    fn qwen35_weights_loader_classifies_layers_and_loads_mtp() {
        let cfg = dummy_cfg();
        let mut loader = MockLoader {
            store: build_synth_store(&cfg),
        };
        let w = Qwen35Weights::from_loader(&mut loader, &cfg).expect("load qwen35 weights");

        // 5 linear + 1 full-attn trunk (6 main layers, interval=4)
        // = layer 3 (zero-indexed: il=3 → (3+1)%4==0) full-attn,
        // others linear.
        assert_eq!(w.trunk_layers.len(), 5); // num_hidden_layers - nextn = 6 - 1 = 5
        for (i, layer) in w.trunk_layers.iter().enumerate() {
            let want_full = ((i + 1) % 4) == 0;
            match (want_full, layer) {
                (true, Qwen35TrunkLayer::FullAttn(_)) => {}
                (false, Qwen35TrunkLayer::Linear(_)) => {}
                _ => panic!(
                    "layer {i}: want_full={want_full}, got {:?}",
                    std::mem::discriminant(layer)
                ),
            }
        }

        // 1 MTP layer with required tensors loaded; optional
        // shared-head tensors omitted in the synth store → None.
        assert_eq!(w.mtp_layers.len(), 1);
        let mtp = &w.mtp_layers[0];
        // Mock loader returns F32 only (no packed bytes); verify
        // the synth eh_proj came through as MatWeight::F32.
        assert_eq!(mtp.eh_proj.len(), 1);
        assert!(matches!(mtp.eh_proj, MatWeight::F32(_)));
        assert_eq!(mtp.enorm.len(), 1);
        assert_eq!(mtp.hnorm.len(), 1);
        assert!(mtp.embed_tokens.is_none());
        assert!(mtp.shared_head_head.is_none());
        assert!(mtp.shared_head_norm.is_none());

        // Tied LM head: `output.weight` was intentionally omitted
        // from the synth store, so `output` should be None and the
        // caller is expected to fall back to `token_embd`.
        assert!(w.output.is_none());
        assert_eq!(w.token_embd.len(), 1);
        assert_eq!(w.output_norm.len(), 1);
    }

    /// Missing required tensor: error mentions the exact key.
    #[test]
    fn qwen35_weights_loader_reports_missing_tensor_key() {
        let cfg = dummy_cfg();
        let mut store = build_synth_store(&cfg);
        store.remove("blk.2.ssm_conv1d.weight");
        let mut loader = MockLoader { store };
        let err = Qwen35Weights::from_loader(&mut loader, &cfg).expect_err("must error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("blk.2.ssm_conv1d.weight"),
            "error message must point at the missing key: {msg}"
        );
    }
}
