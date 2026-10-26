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

//! LFM2 / LFM2.5 **GGUF** runner (`general.architecture = lfm2`).
//!
//! LFM2.5 is a hybrid **ShortConv + GQA-attention** decoder (LiquidAI). The
//! transformer graph is the validated
//! [`rlx_core::standard_decoder::build_lfm2_prefill`] (ShortConv mixer for most
//! layers, GQA attention on the `full_attn_layers`, SwiGLU FFN, tied LM head).
//! That builder speaks the HuggingFace/mlx tensor-name convention, so this
//! module bridges the GGUF world with:
//!
//! 1. [`lfm2_spec_from_gguf`] — read the `lfm2.*` metadata (per-layer attn
//!    routing comes from the `lfm2.attention.head_count_kv` array: `0` =
//!    ShortConv, non-zero = attention).
//! 2. [`GgufNameShim`] — a [`WeightLoader`] that renames GGUF tensors
//!    (`blk.N.shortconv.*`, `blk.N.attn_*`, `token_embd*`, …) to the HF names
//!    the builder loads (`model.layers.N.conv.*`, `…self_attn.*`,
//!    `model.embed_tokens`, …), dequantizing K-quant tensors to F32 on `take`.
//!
//! The F32 graph runs natively on every backend.

use anyhow::{Result, anyhow};
use rlx_core::standard_decoder::{
    DenseStore, Lfm2Spec, TypedParams, build_lfm2_decode_stored, build_lfm2_prefill_stored,
};
use rlx_core::weight_loader::{PrefixRenameLoader, WeightLoader};
use rlx_core::weight_storage::{HostTable, WeightStorage};
use rlx_core::{AssetSource, SafetensorsSource};
use rlx_flow::CompileProfile;
use rlx_gguf::{GgufFile, MetaValue};
use rlx_ir::quant::QuantScheme;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Map an HF/mlx LFM2 tensor name (what `build_lfm2_prefill` requests) to the
/// GGUF (`llama.cpp` `lfm2`) tensor name. Unrecognized names pass through.
fn hf_to_gguf_name(name: &str) -> String {
    match name {
        "model.embed_tokens.weight" => return "token_embd.weight".into(),
        // LFM2's final RMSNorm is stored as `token_embd_norm` in GGUF but the
        // builder loads it as `model.embedding_norm.weight`.
        "model.embedding_norm.weight" => return "token_embd_norm.weight".into(),
        "output.weight" => return "output.weight".into(),
        _ => {}
    }
    if let Some(rest) = name.strip_prefix("model.layers.")
        && let Some((idx, suffix)) = rest.split_once('.')
    {
        let mapped = match suffix {
            "operator_norm.weight" => Some("attn_norm.weight"),
            "ffn_norm.weight" => Some("ffn_norm.weight"),
            "conv.in_proj.weight" => Some("shortconv.in_proj.weight"),
            "conv.conv.weight" => Some("shortconv.conv.weight"),
            "conv.out_proj.weight" => Some("shortconv.out_proj.weight"),
            "self_attn.q_proj.weight" => Some("attn_q.weight"),
            "self_attn.k_proj.weight" => Some("attn_k.weight"),
            "self_attn.v_proj.weight" => Some("attn_v.weight"),
            "self_attn.out_proj.weight" => Some("attn_output.weight"),
            "self_attn.q_layernorm.weight" => Some("attn_q_norm.weight"),
            "self_attn.k_layernorm.weight" => Some("attn_k_norm.weight"),
            "feed_forward.w1.weight" => Some("ffn_gate.weight"),
            "feed_forward.w3.weight" => Some("ffn_up.weight"),
            "feed_forward.w2.weight" => Some("ffn_down.weight"),
            _ => None,
        };
        if let Some(g) = mapped {
            return format!("blk.{idx}.{g}");
        }
    }
    name.to_string()
}

/// A [`WeightLoader`] that renames HF/mlx LFM2 keys to GGUF keys and delegates
/// to the wrapped GGUF loader. Both the dense accessors (`take` /
/// `take_transposed`, used for the embed / norms / conv weight) and the
/// **packed** K-quant accessors (`packed_meta` / `take_packed` /
/// `tensor_bytes_borrowed`, used for the linear projections) are remapped, so
/// projections stay packed (`Op::DequantMatMul`) instead of dequantizing to
/// F32. That keeps the arena small enough for wgpu's ~4 GiB buffer limit.
pub struct GgufNameShim {
    inner: Box<dyn WeightLoader>,
}

impl GgufNameShim {
    pub fn new(inner: Box<dyn WeightLoader>) -> Self {
        Self { inner }
    }
}

impl WeightLoader for GgufNameShim {
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.inner.take(&hf_to_gguf_name(key))
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.inner.take_transposed(&hf_to_gguf_name(key))
    }
    fn take_packed(
        &mut self,
        key: &str,
    ) -> Result<Option<rlx_core::weight_map::PackedWeightTensor>> {
        self.inner.take_packed(&hf_to_gguf_name(key))
    }
    fn packed_meta(&self, key: &str) -> Option<(QuantScheme, Vec<usize>)> {
        self.inner.packed_meta(&hf_to_gguf_name(key))
    }
    fn tensor_bytes_borrowed(&self, key: &str) -> Option<&[u8]> {
        self.inner.tensor_bytes_borrowed(&hf_to_gguf_name(key))
    }
    /// Forwarded (through the same name remap) so the packed upload can
    /// `pread` instead of borrowing the mmap. Without this the trait default
    /// returns `false` and the streaming path silently falls back.
    fn read_tensor_bytes_into(&self, key: &str, buf: &mut Vec<u8>) -> Result<bool> {
        self.inner
            .read_tensor_bytes_into(&hf_to_gguf_name(key), buf)
    }
    fn stored_dtype(&self, key: &str) -> Option<rlx_ir::DType> {
        self.inner.stored_dtype(&hf_to_gguf_name(key))
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.inner.remaining_keys()
    }
    fn arch_hint(&self) -> Option<&str> {
        self.inner.arch_hint()
    }
}

/// Compile options for the packed LFM2 graphs.
///
/// Fusion is **off** by default (the K-quant-safe setting: blanket
/// RMSNorm→DequantMatMul fusion can skew K-quant numerics). It gave **no
/// measurable tps gain** for this graph — decode is DequantMatMul
/// weight-bandwidth-bound, and the fusion pass doesn't fire on `DequantMatMul`
/// nodes (clean back-to-back Metal A/B: fusion on ≈ off within noise). The
/// pipeline's CSE (common-subexpression elimination) already runs by default
/// (`RLX_DISABLE_CSE=1` opts out) and likewise contributes ~0 on this decode
/// graph. `RLX_LFM_FUSION=1` re-enables fusion for experimentation (validated
/// bit-identical to the unfused reference on CPU/Metal — safe, just not faster).
fn packed_compile_opts(exec_device: Device) -> rlx_runtime::CompileOptions {
    if rlx_ir::env::flag("RLX_LFM_FUSION") {
        let mut p = CompileProfile::qwen3_prefill();
        p.fusion.skip = false;
        rlx_core::flow_bridge::compile_options_for_profile(&p, exec_device)
    } else {
        rlx_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
            &CompileProfile::qwen3_prefill(),
            exec_device,
        )
    }
}

fn meta_u32(raw: &GgufFile, key: &str) -> Option<u32> {
    raw.metadata.get(key).and_then(MetaValue::as_u32)
}

fn meta_f32(raw: &GgufFile, key: &str) -> Option<f32> {
    raw.metadata.get(key).and_then(|v| match v {
        MetaValue::F32(x) => Some(*x),
        MetaValue::F64(x) => Some(*x as f32),
        _ => None,
    })
}

/// Build an [`Lfm2Spec`] from `lfm2.*` GGUF metadata.
pub fn lfm2_spec_from_gguf(raw: &GgufFile) -> Result<Lfm2Spec> {
    let arch = raw
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        .unwrap_or("");
    if !matches!(arch, "lfm2" | "lfm2_5" | "lfm25") {
        return Err(anyhow!(
            "lfm2_spec_from_gguf: expected general.architecture=lfm2, got {arch:?}"
        ));
    }
    let hidden = meta_u32(raw, "lfm2.embedding_length")
        .ok_or_else(|| anyhow!("missing lfm2.embedding_length"))? as usize;
    let heads = meta_u32(raw, "lfm2.attention.head_count")
        .ok_or_else(|| anyhow!("missing lfm2.attention.head_count"))? as usize;
    let layers = meta_u32(raw, "lfm2.block_count")
        .ok_or_else(|| anyhow!("missing lfm2.block_count"))? as usize;
    let inter = meta_u32(raw, "lfm2.feed_forward_length")
        .ok_or_else(|| anyhow!("missing lfm2.feed_forward_length"))? as usize;
    let vocab = meta_u32(raw, "lfm2.vocab_size")
        .map(|v| v as usize)
        .or_else(|| {
            raw.tensors
                .get("token_embd.weight")
                .map(|t| t.shape[t.shape.len() - 1])
        })
        .ok_or_else(|| anyhow!("cannot determine vocab_size"))?;
    let conv_kernel = meta_u32(raw, "lfm2.shortconv.l_cache").unwrap_or(3) as usize;
    let rope_theta = meta_f32(raw, "lfm2.rope.freq_base").unwrap_or(1_000_000.0) as f64;
    let eps = meta_f32(raw, "lfm2.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
    let head_dim = hidden.checked_div(heads).unwrap_or(hidden);

    // Per-layer attention routing: `head_count_kv` is an array (0 = ShortConv,
    // non-zero = GQA attention). Some writers emit a scalar (all-attention).
    let kv_meta = raw
        .metadata
        .get("lfm2.attention.head_count_kv")
        .ok_or_else(|| anyhow!("missing lfm2.attention.head_count_kv"))?;
    let (num_kv, full_attn_layers) = match kv_meta.as_array() {
        Some(items) => {
            let vals: Vec<u32> = items.iter().map(|v| v.as_u32().unwrap_or(0)).collect();
            let nkv = vals.iter().copied().find(|&x| x != 0).unwrap_or(0) as usize;
            let attn: Vec<usize> = vals
                .iter()
                .enumerate()
                .filter_map(|(i, &x)| (x != 0).then_some(i))
                .collect();
            (nkv, attn)
        }
        None => {
            let nkv = kv_meta.as_u32().unwrap_or(heads as u32) as usize;
            (nkv, (0..layers).collect())
        }
    };
    if num_kv == 0 {
        return Err(anyhow!(
            "lfm2: no attention layers found (head_count_kv all zero)"
        ));
    }

    Ok(Lfm2Spec {
        vocab_size: vocab,
        hidden_size: hidden,
        intermediate_size: inter,
        num_hidden_layers: layers,
        num_attention_heads: heads,
        num_key_value_heads: num_kv,
        head_dim,
        conv_dim: hidden,
        conv_kernel,
        full_attn_layers,
        rope_theta,
        rms_norm_eps: eps,
    })
}

/// The HF directory `path` names, if it is one: a directory with a
/// `config.json` and no `.gguf`, or a `.safetensors` file inside such a
/// directory.
fn hf_dir(path: &Path) -> Option<PathBuf> {
    let dir = if path.is_dir() {
        path.to_path_buf()
    } else if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
        path.parent()?.to_path_buf()
    } else {
        return None;
    };
    if !dir.join("config.json").is_file() || resolve_gguf(&dir).is_ok() {
        return None;
    }
    Some(dir)
}

/// True when `path` is an HF LFM2 directory whose config describes the hybrid
/// ShortConv + attention schedule (`layer_types`), as opposed to the legacy
/// SSM variant [`LfmRunner`](crate::LfmRunner) handles. An LFM2-VL directory
/// counts too: its text model is read from `text_config`.
pub fn is_lfm2_hf_dir(path: &Path) -> bool {
    hf_dir(path)
        .and_then(|d| std::fs::read_to_string(d.join("config.json")).ok())
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .is_some_and(|v| {
            lfm2_text_config(&v)
                .get("layer_types")
                .is_some_and(|l| l.is_array())
        })
}

/// The text-model block of an HF config: `text_config` for LFM2-VL
/// (`Lfm2VlForConditionalGeneration`), the config itself for plain LFM2.
pub fn lfm2_text_config(cfg: &serde_json::Value) -> &serde_json::Value {
    cfg.get("text_config").unwrap_or(cfg)
}

/// Where an LFM2-VL checkpoint keeps its text model's tensors. The builders
/// ask for `model.layers.N.*` / `model.embed_tokens.weight`; a VL file stores
/// them one level down.
const VL_TEXT_PREFIX: &str = "model.language_model.";

/// Wrap `inner` so text-model keys resolve inside an LFM2-VL checkpoint.
fn vl_text_loader(inner: Box<dyn WeightLoader>) -> Box<dyn WeightLoader> {
    Box::new(PrefixRenameLoader::new(inner, "model.", VL_TEXT_PREFIX))
}

/// Build an [`Lfm2Spec`] from an HF `config.json` value. `intermediate_size`
/// is passed in rather than read: LFM2.5-350M's config says 6656 while its FFN
/// tensors are 4608 wide (`block_auto_adjust_ff_dim`), so the caller takes it
/// from the checkpoint.
pub fn lfm2_spec_from_hf_config(cfg: &serde_json::Value, intermediate: usize) -> Result<Lfm2Spec> {
    let u = |k: &str| -> Result<usize> {
        cfg.get(k)
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize)
            .ok_or_else(|| anyhow!("lfm2 config.json: missing {k}"))
    };
    let hidden = u("hidden_size")?;
    let heads = u("num_attention_heads")?;
    let layer_types = cfg
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow!("lfm2 config.json: missing layer_types"))?;
    let full_attn_layers: Vec<usize> = layer_types
        .iter()
        .enumerate()
        .filter_map(|(i, t)| (t.as_str() == Some("full_attention")).then_some(i))
        .collect();
    let layers = u("num_hidden_layers")?;
    if layer_types.len() != layers {
        return Err(anyhow!(
            "lfm2 config.json: {} layer_types for {layers} layers",
            layer_types.len()
        ));
    }
    let rope_theta = cfg
        .get("rope_parameters")
        .and_then(|r| r.get("rope_theta"))
        .or_else(|| cfg.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(1_000_000.0);
    let eps = cfg
        .get("norm_eps")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(1e-5) as f32;
    Ok(Lfm2Spec {
        vocab_size: u("vocab_size")?,
        hidden_size: hidden,
        intermediate_size: intermediate,
        num_hidden_layers: layers,
        num_attention_heads: heads,
        num_key_value_heads: u("num_key_value_heads")?,
        head_dim: hidden / heads,
        conv_dim: u("conv_dim").unwrap_or(hidden),
        conv_kernel: u("conv_L_cache").unwrap_or(3),
        full_attn_layers,
        rope_theta,
        rms_norm_eps: eps,
    })
}

/// Spec + eos id for an HF LFM2 directory; the FFN width comes from the
/// layer-0 gate projection. `vl` = the text model sits inside an LFM2-VL
/// checkpoint (config under `text_config`, tensors under
/// `model.language_model.`).
fn lfm2_spec_from_hf(
    raw_config: &str,
    weights: &SafetensorsSource,
    vl: bool,
) -> Result<(Lfm2Spec, u32)> {
    let cfg: serde_json::Value =
        serde_json::from_str(raw_config).map_err(|e| anyhow!("lfm2: parse config.json: {e}"))?;
    let mut loader = weights.loader()?;
    if vl {
        loader = vl_text_loader(loader);
    }
    let (_, w1) = loader.take("model.layers.0.feed_forward.w1.weight")?;
    let intermediate = *w1
        .first()
        .ok_or_else(|| anyhow!("lfm2: feed_forward.w1 has no shape"))?;
    let spec = lfm2_spec_from_hf_config(lfm2_text_config(&cfg), intermediate)?;
    let eos = cfg
        .get("eos_token_id")
        .and_then(serde_json::Value::as_u64)
        .map_or(u32::MAX, |v| v as u32);
    Ok((spec, eos))
}

/// The text model of an HF LFM2 / LFM2-VL directory: its spec, and a loader
/// whose keys are the text model's (`model.layers.N.*`,
/// `model.embed_tokens.weight`) whichever layout the checkpoint has.
pub fn hf_text_model(dir: &Path) -> Result<(Lfm2Spec, Box<dyn WeightLoader>)> {
    let raw = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| anyhow!("lfm2: read {}/config.json: {e}", dir.display()))?;
    let vl = serde_json::from_str::<serde_json::Value>(&raw)
        .map_err(|e| anyhow!("lfm2: parse config.json: {e}"))?
        .get("text_config")
        .is_some();
    let src = SafetensorsSource::File(dir.to_path_buf());
    let (spec, _) = lfm2_spec_from_hf(&raw, &src, vl)?;
    let loader = src.loader()?;
    Ok((spec, if vl { vl_text_loader(loader) } else { loader }))
}

/// A compiled + weight-attached decode graph, cached and reused across
/// [`Lfm2GgufRunner::generate`] calls so repeated generations skip the
/// per-call graph build and the ~1.6 GB weight re-attach.
struct DecodeSession {
    compiled: rlx_runtime::CompiledGraph,
    names: Vec<String>,
    max_kv: usize,
    /// `[vocab, hidden]` embedding table when the graph leaves the vocabulary
    /// to the host (WebGL — see [`build_lfm2_decode_ext`]): token rows are
    /// gathered from it and the tied LM head applied with it.
    vocab: Option<Vec<f32>>,
}

/// A compiled prefill graph; `vocab` is the tied embedding table when the
/// device keeps the vocabulary on the host (WebGL).
struct PrefillGraph {
    compiled: rlx_runtime::CompiledGraph,
    vocab: Option<Vec<f32>>,
}

/// The cached graph behind [`Lfm2GgufRunner::hidden_from_embeds`]: input
/// embeddings in, final-normed hidden out, compiled for `seq` positions.
struct EmbedsPrefill {
    seq: usize,
    compiled: rlx_runtime::CompiledGraph,
}

/// One decode step's graph inputs, owned.
struct StepFeed {
    /// `token_id` (as f32) — or, with a host vocabulary, its embedding row.
    token: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    key_mask: Vec<f32>,
    write_oh: Vec<f32>,
    write_inv: Vec<f32>,
}

impl StepFeed {
    fn inputs(&self, host_vocab: bool) -> [(&str, &[f32]); 6] {
        [
            (
                if host_vocab {
                    "inputs_embeds"
                } else {
                    "token_id"
                },
                &self.token,
            ),
            ("rope_cos", &self.cos),
            ("rope_sin", &self.sin),
            ("key_mask", &self.key_mask),
            ("kv_write_oh", &self.write_oh),
            ("kv_write_inv", &self.write_inv),
        ]
    }
}

/// LFM2 / LFM2.5 GGUF text runner (hybrid ShortConv + GQA attention).
///
/// [`generate`](Self::generate) uses **incremental O(n) decode**: one fixed-max
/// [`build_lfm2_decode`](rlx_core::standard_decoder::build_lfm2_decode) graph stepped one token at a time, with a per-layer
/// conv-state cache (ShortConv) and KV cache (attention). Weights stay packed
/// (`DequantMatMul`). The compiled graph is cached and reused across calls (the
/// CPU dequant cache is process-global, so warmup is amortized too).
/// [`generate_prefill`](Self::generate_prefill) keeps the O(n²) re-prefill path
/// as a correctness reference.
pub struct Lfm2GgufRunner {
    weights: PathBuf,
    /// Set for an HF safetensors checkpoint (`weights` is then its
    /// directory, or empty when it came from memory) rather than a `.gguf`.
    hf: Option<SafetensorsSource>,
    /// The HF checkpoint is an LFM2-VL one: text tensors live under
    /// `model.language_model.`.
    vl: bool,
    device: Device,
    spec: Lfm2Spec,
    eos_id: u32,
    decode: std::sync::Mutex<Option<DecodeSession>>,
    embeds_prefill: std::sync::Mutex<Option<EmbedsPrefill>>,
    embed_table: std::sync::OnceLock<HostTable>,
    /// How dense (non-packed) weight matrices are stored; see
    /// [`Self::set_weight_storage`].
    weight_storage: WeightStorage,
}

impl Lfm2GgufRunner {
    /// Open an LFM2/LFM2.5 checkpoint and parse its config: a `.gguf` file, or
    /// an HF safetensors directory (`config.json` + `model.safetensors`) such
    /// as `LiquidAI/LFM2.5-230M`. An LFM2-VL directory (`LiquidAI/LFM2.5-VL-3B`,
    /// `LiquidAI/d1-3B`) opens as its text model; the vision tower is ignored.
    pub fn open(weights: impl Into<PathBuf>, device: Device) -> Result<Self> {
        let weights = weights.into();
        rlx_core::validate_browser_capable_device("lfm", device)?;
        if let Some(dir) = hf_dir(&weights) {
            return Self::open_source(&AssetSource::dir(dir), device);
        }
        let raw = GgufFile::from_path(&weights)
            .map_err(|e| anyhow!("lfm2: open GGUF {}: {e}", weights.display()))?;
        let spec = lfm2_spec_from_gguf(&raw)?;
        let eos_id = meta_u32(&raw, "tokenizer.ggml.eos_token_id").unwrap_or(u32::MAX);
        Ok(Self {
            weights,
            hf: None,
            vl: false,
            device,
            spec,
            eos_id,
            decode: std::sync::Mutex::new(None),
            embeds_prefill: std::sync::Mutex::new(None),
            embed_table: std::sync::OnceLock::new(),
            weight_storage: WeightStorage::from_env().unwrap_or(WeightStorage::Auto),
        })
    }

    /// The checkpoint path: the `.gguf` file, or the HF directory (empty for
    /// an in-memory checkpoint).
    pub fn weights_path(&self) -> &Path {
        &self.weights
    }

    /// A fresh loader over the checkpoint, presenting the HF tensor names the
    /// builders request. HF checkpoints already use them; GGUF goes through
    /// [`GgufNameShim`].
    fn open_loader(&self) -> Result<Box<dyn WeightLoader>> {
        if let Some(src) = &self.hf {
            let loader = src.loader()?;
            return Ok(if self.vl {
                vl_text_loader(loader)
            } else {
                loader
            });
        }
        let inner = rlx_core::weight_registry::open_weight_loader(&self.weights)
            .map_err(|e| anyhow!("lfm2: open weights {}: {e}", self.weights.display()))?;
        Ok(Box::new(GgufNameShim::new(inner)))
    }

    /// Open an HF-layout checkpoint (`config.json` + safetensors) from any
    /// [`AssetSource`] — a directory, a `.rlxpack`, or in-memory files (the
    /// browser path).
    pub fn open_source(src: &AssetSource, device: Device) -> Result<Self> {
        rlx_core::validate_browser_capable_device("lfm", device)?;
        // A directory is opened as a whole so sharded checkpoints load too.
        let (weights, source) = match src.dir_path() {
            Some(dir) => (dir.clone(), SafetensorsSource::File(dir)),
            None => (
                PathBuf::new(),
                SafetensorsSource::from_asset(src, "model.safetensors")?,
            ),
        };
        let raw_config = src.read_to_string("config.json")?;
        let vl = serde_json::from_str::<serde_json::Value>(&raw_config)
            .map_err(|e| anyhow!("lfm2: parse config.json: {e}"))?
            .get("text_config")
            .is_some();
        let (spec, eos_id) = lfm2_spec_from_hf(&raw_config, &source, vl)?;
        Ok(Self {
            weights,
            hf: Some(source),
            vl,
            device,
            spec,
            eos_id,
            decode: std::sync::Mutex::new(None),
            embeds_prefill: std::sync::Mutex::new(None),
            embed_table: std::sync::OnceLock::new(),
            weight_storage: WeightStorage::from_env().unwrap_or(WeightStorage::Auto),
        })
    }

    pub fn config(&self) -> &Lfm2Spec {
        &self.spec
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    /// Build + compile the prefill graph for a fixed sequence length `seq`.
    ///
    /// Linear projections stay **packed** (`Op::DequantMatMul` over the GGUF
    /// K-quant blobs) so the arena holds activations + the F32 embed only —
    /// small enough for wgpu. Weight bytes are borrowed straight from the GGUF
    /// mmap at attach (zero-copy). Metal/MLX/wgpu/Vulkan/CoreML dequant on the
    /// selected device; wgpu keeps the packed blobs in a separate buffer.
    fn compile_for(&self, seq: usize) -> Result<PrefillGraph> {
        let seq = seq.max(1);
        let mut shim = self.open_loader()?;
        let exec_device = self.pick_exec_device();
        // WebGL: embedding gather and tied LM head on the host, as in decode.
        let host_vocab = exec_device == Device::OpenGl;
        let mut packed: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
        let mut store = DenseStore::new(self.weight_storage);
        let (graph, params) = build_lfm2_prefill_stored(
            &self.spec,
            shim.as_mut(),
            seq,
            &mut packed,
            host_vocab,
            &mut store,
        )?;
        let opts = packed_compile_opts(exec_device);
        let mut compiled = rlx_core::flow_bridge::packed_gguf_compile_guard(exec_device, || {
            Session::new(exec_device).compile_with(graph, &opts)
        });
        self.attach(&mut compiled, &params, &store.typed, &packed, shim.as_ref())?;
        let vocab = if host_vocab {
            Some(self.open_loader()?.take("model.embed_tokens.weight")?.0)
        } else {
            None
        };
        Ok(PrefillGraph { compiled, vocab })
    }

    /// Logits `[seq, vocab]` for `ids` (padded to the compiled length).
    fn run_prefill(&self, graph: &mut PrefillGraph, ids: &[u32]) -> Result<Vec<f32>> {
        let h = self.spec.hidden_size;
        let Some(table) = graph.vocab.as_deref() else {
            let ids_f32: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
            let out = graph.compiled.run(&[("input_ids", ids_f32.as_slice())]);
            return out
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("lfm2: no logits"));
        };
        let mut embeds = Vec::with_capacity(ids.len() * h);
        for &t in ids {
            let row = t as usize * h;
            embeds.extend_from_slice(
                table
                    .get(row..row + h)
                    .ok_or_else(|| anyhow!("lfm2: token {t} past the vocabulary"))?,
            );
        }
        let out = graph.compiled.run(&[("inputs_embeds", embeds.as_slice())]);
        let hidden = out
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("lfm2: no hidden"))?;
        Ok(hidden
            .chunks_exact(h)
            .flat_map(|row| self.logits_from(Some(table), row.to_vec()))
            .collect())
    }

    /// Attach dense params + packed (mmap-borrowed) weights to a compiled graph.
    fn attach(
        &self,
        compiled: &mut rlx_runtime::CompiledGraph,
        params: &HashMap<String, Vec<f32>>,
        typed: &TypedParams,
        packed: &HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
        shim: &dyn WeightLoader,
    ) -> Result<()> {
        for (n, d) in params {
            compiled.set_param(n, d);
        }
        for (n, (bytes, dtype)) in typed {
            compiled.set_param_typed(n, bytes, *dtype);
        }
        // Empty bytes = GGUF zero-copy marker; otherwise the bytes are owned.
        //
        // For the GGUF case `pread` into one reused scratch rather than
        // borrowing the mmap: `set_param_typed` copies through the slice, so a
        // borrow faults in every page of the checkpoint and leaves a second
        // full-size copy resident (unreclaimable on macOS short of `munmap`).
        // Falls back to the borrow when the loader has no streaming backing.
        // Revert: RLX_LFM_MMAP_UPLOAD=1.
        let stream = !rlx_ir::env::flag("RLX_LFM_MMAP_UPLOAD");
        let mut scratch: Vec<u8> = Vec::new();
        for (n, (bytes, _scheme, _shape)) in packed {
            if bytes.is_empty() {
                if stream && shim.read_tensor_bytes_into(n, &mut scratch)? {
                    compiled.set_param_typed(n, &scratch, rlx_ir::DType::U8);
                    continue;
                }
                let slice = shim.tensor_bytes_borrowed(n).ok_or_else(|| {
                    anyhow!("lfm2: packed weight {n} bytes unavailable at attach")
                })?;
                compiled.set_param_typed(n, slice, rlx_ir::DType::U8);
            } else {
                compiled.set_param_typed(n, bytes.as_slice(), rlx_ir::DType::U8);
            }
        }
        Ok(())
    }

    /// Choose the device to actually compile/run the graph on.
    ///
    /// wgpu and Vulkan used to be rerouted to the host CPU here because the
    /// ShortConv graph came out wrong on them. It no longer does: LFM2.5-230M
    /// (HF) matches transformers on wgpu (logits within 4e-5, same greedy
    /// tokens) and the LFM2-350M Q4_K_M GGUF generates the CPU's text
    /// token-for-token on wgpu and Vulkan. The generic
    /// `RLX_PACKED_GGUF_{WGPU,VULKAN}_HOST` switches still force the host.
    fn pick_exec_device(&self) -> Device {
        rlx_core::flow_bridge::packed_gguf_execution_device(self.device)
    }

    /// How dense (non-packed) weight matrices — and the host copy of the tied
    /// embedding — are stored. [`WeightStorage::Auto`] (the default) keeps a
    /// bf16 / f16 checkpoint at its own width, which is lossless: the products
    /// are the same as with f32 weights at half the memory.
    /// [`WeightStorage::F32`] restores the old f32 expansion; `F16` / `Bf16`
    /// convert. GGUF K-quant projections stay packed regardless. Takes effect
    /// for graphs built after the call (cached graphs are dropped).
    pub fn set_weight_storage(&mut self, storage: WeightStorage) {
        if storage != self.weight_storage {
            self.weight_storage = storage;
            self.embed_table = std::sync::OnceLock::new();
            *self.decode.get_mut().unwrap_or_else(|e| e.into_inner()) = None;
            *self
                .embeds_prefill
                .get_mut()
                .unwrap_or_else(|e| e.into_inner()) = None;
        }
    }

    /// [`Self::set_weight_storage`], builder style.
    pub fn with_weight_storage(mut self, storage: WeightStorage) -> Self {
        self.set_weight_storage(storage);
        self
    }

    pub fn weight_storage(&self) -> WeightStorage {
        self.weight_storage
    }

    /// The tied `[vocab, hidden]` embedding table (also the LM head), read
    /// once and kept on the host at the [weight storage](Self::set_weight_storage)
    /// width (a bf16 checkpoint's 1 GB f32 table is 0.5 GB).
    pub fn embed_table(&self) -> Result<&HostTable> {
        if let Some(t) = self.embed_table.get() {
            return Ok(t);
        }
        // A quantized GGUF table stays packed under `Auto` (rows decode on
        // demand — lossless, ~4.5x smaller than widening it).
        // Through the runner's own loader (mmap'd, bytes streamed with pread):
        // a second `GgufFile::from_path` read the whole checkpoint into memory.
        if self.hf.is_none() && self.weight_storage == WeightStorage::Auto {
            let loader = self.open_loader()?;
            let dtype = loader
                .gguf_file()
                .and_then(|f| f.tensors.get("token_embd.weight"))
                .map(|t| t.dtype);
            let mut bytes = Vec::new();
            if let Some(dtype) = dtype
                && !matches!(
                    dtype,
                    rlx_gguf::GgmlType::F32 | rlx_gguf::GgmlType::F16 | rlx_gguf::GgmlType::BF16
                )
                && (loader.read_tensor_bytes_into("model.embed_tokens.weight", &mut bytes)?
                    || loader
                        .tensor_bytes_borrowed("model.embed_tokens.weight")
                        .map(|b| bytes = b.to_vec())
                        .is_some())
                && let Some(table) = HostTable::packed(bytes, dtype, self.spec.hidden_size)
                && table.len() == self.spec.vocab_size * self.spec.hidden_size
            {
                return Ok(self.embed_table.get_or_init(|| table));
            }
        }
        let mut loader = self.open_loader()?;
        let key = "model.embed_tokens.weight";
        let dtype = self.weight_storage.resolve(loader.stored_dtype(key));
        let (table, shape) = loader.take(key)?;
        let want = [self.spec.vocab_size, self.spec.hidden_size];
        if shape != want {
            return Err(anyhow!(
                "lfm2: embed_tokens is {shape:?}, config says {want:?}"
            ));
        }
        Ok(self
            .embed_table
            .get_or_init(|| HostTable::from_f32(table, dtype)))
    }

    /// Input embeddings `[ids.len(), hidden]` for token ids.
    pub fn embed_ids(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let table = self.embed_table()?;
        let h = self.spec.hidden_size;
        let mut out = vec![0f32; ids.len() * h];
        for (row, &t) in out.chunks_exact_mut(h).zip(ids) {
            if t as usize >= self.spec.vocab_size {
                return Err(anyhow!("lfm2: token {t} past the vocabulary"));
            }
            table.read_into(t as usize * h, row);
        }
        Ok(out)
    }

    /// One logit of the tied LM head: `embed[id] · hidden` (in f64), for
    /// callers that score a handful of tokens instead of the vocabulary.
    pub fn lm_head_logit(&self, id: u32, hidden: &[f32]) -> Result<f64> {
        let h = self.spec.hidden_size;
        if hidden.len() != h || id as usize >= self.spec.vocab_size {
            return Err(anyhow!(
                "lfm2: lm_head_logit({id}) on {} values",
                hidden.len()
            ));
        }
        Ok(self.embed_table()?.dot(id as usize * h, hidden))
    }

    /// Final-normed hidden states `[n, hidden]` of one causal pass over the
    /// input embeddings `embeds` (`[n, hidden]`, row-major).
    ///
    /// For callers that build the input sequence themselves — image features
    /// scattered over placeholder tokens, a decision model reading one answer
    /// slot — and score hidden states against [`embed_table`](Self::embed_table)
    /// (the tied LM head) only where they need to. The graph is compiled for a
    /// power-of-two length and kept: inputs are right-padded with zeros, which
    /// a causal stack never lets reach a real position. It is rebuilt when a
    /// longer input arrives, or when inputs get 4x shorter than it.
    pub fn hidden_from_embeds(&self, embeds: &[f32]) -> Result<Vec<f32>> {
        let h = self.spec.hidden_size;
        if embeds.is_empty() || !embeds.len().is_multiple_of(h) {
            return Err(anyhow!(
                "lfm2: {} embedding values is not a whole number of {h}-wide rows",
                embeds.len()
            ));
        }
        let n = embeds.len() / h;
        let bucket = n.max(64).next_power_of_two();
        let mut guard = self
            .embeds_prefill
            .lock()
            .map_err(|_| anyhow!("lfm2: prefill cache poisoned"))?;
        if guard
            .as_ref()
            .is_none_or(|g| g.seq < n || g.seq >= 4 * bucket)
        {
            // Drop the old graph (and its weight arena) before building the
            // next one, so the two never coexist.
            *guard = None;
            *guard = Some(self.compile_embeds_prefill(bucket)?);
        }
        let graph = guard.as_mut().expect("just built");
        let mut padded = vec![0f32; graph.seq * h];
        padded[..embeds.len()].copy_from_slice(embeds);
        let out = graph
            .compiled
            .run(&[("inputs_embeds", padded.as_slice())])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("lfm2: no hidden"))?;
        Ok(out[..n * h].to_vec())
    }

    fn compile_embeds_prefill(&self, seq: usize) -> Result<EmbedsPrefill> {
        let mut shim = self.open_loader()?;
        let exec_device = self.pick_exec_device();
        let mut packed: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
        let mut store = DenseStore::new(self.weight_storage);
        let (graph, params) = build_lfm2_prefill_stored(
            &self.spec,
            shim.as_mut(),
            seq,
            &mut packed,
            true,
            &mut store,
        )?;
        let opts = packed_compile_opts(exec_device);
        let mut compiled = rlx_core::flow_bridge::packed_gguf_compile_guard(exec_device, || {
            Session::new(exec_device).compile_with(graph, &opts)
        });
        self.attach(&mut compiled, &params, &store.typed, &packed, shim.as_ref())?;
        Ok(EmbedsPrefill { seq, compiled })
    }

    /// Last-position logits `[vocab]` after a single prefill of `prompt`.
    pub fn predict_logits(&self, prompt: &[u32]) -> Result<Vec<f32>> {
        if prompt.is_empty() {
            return Err(anyhow!("lfm2: empty prompt"));
        }
        let seq = prompt.len();
        let mut graph = self.compile_for(seq)?;
        let logits = self.run_prefill(&mut graph, prompt)?;
        let vocab = self.spec.vocab_size;
        Ok(logits[(seq - 1) * vocab..seq * vocab].to_vec())
    }

    /// **O(n²)** greedy generation (reference): compiles one fixed-length prefill
    /// graph and re-runs the whole padded sequence per step. Kept for validating
    /// the incremental [`generate`](Self::generate) path.
    pub fn generate_prefill(
        &self,
        prompt: &[u32],
        n_new: usize,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(anyhow!("lfm2: empty prompt"));
        }
        let vocab = self.spec.vocab_size;
        let total = prompt.len() + n_new;
        let mut graph = self.compile_for(total)?;
        let mut buf = vec![0u32; total];
        buf[..prompt.len()].copy_from_slice(prompt);
        let mut cur = prompt.len();
        let mut out = Vec::with_capacity(n_new);
        while cur < total {
            let logits = self.run_prefill(&mut graph, &buf)?;
            let row = &logits[(cur - 1) * vocab..cur * vocab];
            let tok = argmax(row);
            out.push(tok);
            let keep = on_token(tok);
            if tok == self.eos_id || !keep {
                break;
            }
            buf[cur] = tok;
            cur += 1;
        }
        Ok(out)
    }

    /// Build + compile + attach a fixed-`max_kv` decode graph (one-time cost,
    /// cached in the runner by [`generate`](Self::generate)).
    fn build_decode_session(&self, max_kv: usize) -> Result<DecodeSession> {
        let mut shim = self.open_loader()?;
        let exec_device = self.pick_exec_device();
        // NOTE: this used to force `RLX_Q4K_FUSED_MIN_N=2048` on CPU, citing a
        // ~2× decode win. Removed: the fused kernel quantizes the activation to
        // Q8_K, so once the threshold pulls in the FFN matmuls the error
        // compounds through every remaining layer — which is what broke
        // `decode_parity_live` (incremental decode vs the f32 prefill
        // reference). Upstream rlx defaults the path off for exactly this
        // reason; forcing it on here silently overrode that policy process-wide
        // via an `env::set` from inside a builder.
        //
        // Agreement with the f32 prefill reference, 8 random-id prompts × 8
        // tokens (deterministic):
        //
        // | `RLX_Q4K_FUSED_MIN_N` | 350M | 2.6B |
        // |---|---|---|
        // | unset (off)     | 100%  | 100%  |
        // | 65536 (LM head) | 100%  | 100%  |
        // | 4096            | 75.9% | 90.6% |
        // | 2048 (was)      | 72.2% | 92.2% |
        //
        // Speed, arms alternated *inside one process* and min-of-5 (sequential
        // per-config runs were useless here — two arms executing identical code
        // differed by 28% on machine drift alone):
        //
        // | arm | 350M | 2.6B |
        // |---|---|---|
        // | off             | 33.7 tok/s | 8.4 tok/s |
        // | 65536 (LM head) | +1.4%      | −0.4%     |
        // | 2048 (was)      | −14.1%     | +11.1%    |
        //
        // It also decides peak RSS, because the exact path dequantizes Q4_K to
        // an f32 cache while the fused kernel reads the packed bytes in place —
        // on 2.6B, 14.45 GB exact vs 5.17 GB fused (−9.3 GB, an 8.7× vs 3.1×
        // amplification of the 1.67 GB checkpoint), stable to ±0.02 GB over
        // three interleaved rounds.
        //
        // So the old default was strictly worse on 350M (slower *and* lossy),
        // while on 2.6B it bought ~11% decode and ~8.8 GB for non-bit-exact
        // output — a real tradeoff, but nothing like the 2× claimed, and not
        // one to make silently on a crate whose own parity test asserts
        // exactness. Callers who want it — especially memory-constrained ones,
        // where the RSS saving dwarfs the throughput delta — opt in per-process:
        //
        //     RLX_Q4K_FUSED_MIN_N=2048
        //
        // The LM-head-only width (`n == vocab_size`) is parity-exact in these
        // samples but measures as a wash on speed, so it is not worth
        // recommending either.
        //
        // Cache-thrash protection is unaffected: that lives in rlx-cpu's
        // `prefer_cached_blas`, which routes to the fused kernel on its own when
        // the f32 dequant cache would thrash, independent of this threshold.
        let mut packed: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
        // WebGL keeps every tensor in one 2-D texture, which a [vocab, hidden]
        // table outgrows; there the vocabulary stays on the host.
        let host_vocab = exec_device == Device::OpenGl;
        let mut store = DenseStore::new(self.weight_storage);
        let (graph, params, names) = build_lfm2_decode_stored(
            &self.spec,
            shim.as_mut(),
            max_kv,
            &mut packed,
            host_vocab,
            &mut store,
        )?;
        let vocab = if host_vocab {
            Some(self.open_loader()?.take("model.embed_tokens.weight")?.0)
        } else {
            None
        };
        let opts = packed_compile_opts(exec_device);
        let mut compiled = rlx_core::flow_bridge::packed_gguf_compile_guard(exec_device, || {
            Session::new(exec_device).compile_with(graph, &opts)
        });
        self.attach(&mut compiled, &params, &store.typed, &packed, shim.as_ref())?;
        Ok(DecodeSession {
            compiled,
            names,
            max_kv,
            vocab,
        })
    }

    /// **O(n)** greedy generation via incremental decode. Steps one token at a
    /// time over a cached fixed-`max_kv` [`build_lfm2_decode`](rlx_core::standard_decoder::build_lfm2_decode) graph: each
    /// ShortConv layer keeps a small conv-state, each attention layer keeps a KV
    /// cache, so every step is O(1) FLOP-heavy work. The compiled graph +
    /// attached weights are cached and reused across calls (rebuilt only when a
    /// larger `max_kv` is needed), so repeated generations skip the graph build
    /// and the ~1.6 GB weight re-attach. The prompt is streamed through the same
    /// graph to warm the caches. `on_token` returning `false` / eos halts.
    pub fn generate(
        &self,
        prompt: &[u32],
        n_new: usize,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(anyhow!("lfm2: empty prompt"));
        }
        let p = prompt.len();
        let need = p + n_new;

        // Reuse the cached decode graph when it's large enough; otherwise
        // (re)build at a rounded-up capacity so a range of lengths can share it.
        let mut guard = self.decode.lock().unwrap();
        if guard.as_ref().map(|s| s.max_kv < need).unwrap_or(true) {
            let cap = need.div_ceil(128) * 128;
            *guard = Some(self.build_decode_session(cap)?);
        }
        let session = guard.as_mut().unwrap();
        let DecodeSession {
            compiled,
            names,
            max_kv,
            vocab,
        } = session;
        let max_kv = *max_kv;

        // Cache buffers keyed by graph *input* name (zeroed initial state).
        let mut state = self.initial_state(max_kv);

        // On GPU backends with device-resident handle support (Metal/MLX/CUDA/
        // ROCm) keep the per-layer KV + conv state ON DEVICE across decode steps:
        // bind each `_in_` buffer once as a resident GPU handle fed in-place from
        // its matching `_out_` output slot, then read back ONLY the logits each
        // step. This removes the per-step full-state host round-trip (~2 transfers
        // per layer, H2D + D2H) that otherwise dominates discrete-GPU decode. The
        // graph math is unchanged, so output is token-identical to the host path.
        // wgpu/Vulkan compile to host CPU here (compiled.device() == Cpu) and take
        // the host path below, as does a real CPU device. `RLX_DISABLE_GPU_KV`
        // forces the host path. Re-binding to the zeroed buffers each call clears
        // any residue left resident from a prior `generate` on the cached graph.
        let mut use_gpu_kv = rlx_core::device_supports_gpu_kv(compiled.device());
        if use_gpu_kv {
            for (j, out_name) in names.iter().enumerate() {
                let in_name = out_name.replace("_out_", "_in_");
                if let Some(buf) = state.get(&in_name) {
                    if !compiled.bind_gpu_handle(&in_name, buf) {
                        use_gpu_kv = false;
                        break;
                    }
                    compiled.set_gpu_handle_feed(&in_name, j + 1);
                }
            }
        }

        // Optional per-step profiling (`RLX_LFM_PROFILE=1`): confirms whether the
        // resident-KV path is active and splits device `run()` from host work.
        let profile = rlx_ir::env::flag("RLX_LFM_PROFILE");
        let mut t_run = std::time::Duration::ZERO;
        let mut t_host = std::time::Duration::ZERO;
        let mut steps = 0usize;

        let mut out = Vec::with_capacity(n_new);
        for pos in 0..max_kv {
            let tok = if pos < p {
                prompt[pos]
            } else {
                match out.last() {
                    Some(&t) => t,
                    None => break,
                }
            };
            // Clock reads only when profiling: `Instant::now` panics on wasm32.
            let s_iter = profile.then(std::time::Instant::now);
            // Per-step inputs (kept alive across the borrow of `state`).
            let feed = self.step_feed(vocab.as_deref(), tok, pos, max_kv);
            let fixed = feed.inputs(vocab.is_some());

            let s_run = profile.then(std::time::Instant::now);
            let logits: Vec<f32> = if use_gpu_kv {
                // Resident state stays on device; only logits (slot 0) come back.
                let mut outs = compiled.run_read_outputs(&fixed, Some(&[0]));
                if outs.is_empty() {
                    return Err(anyhow!("lfm2: no logits"));
                }
                outs.swap_remove(0)
            } else {
                // Host path: pass the full state in, read the merged state back.
                let outs = {
                    let mut inputs: Vec<(&str, &[f32])> = fixed.to_vec();
                    for (k, v) in &state {
                        inputs.push((k.as_str(), v.as_slice()));
                    }
                    compiled.run(&inputs)
                };
                let logits = outs
                    .first()
                    .ok_or_else(|| anyhow!("lfm2: no logits"))?
                    .clone();
                for (i, name) in names.iter().enumerate() {
                    let in_name = name.replace("_out_", "_in_");
                    if let Some(buf) = state.get_mut(&in_name) {
                        buf.copy_from_slice(&outs[i + 1]);
                    }
                }
                logits
            };
            let logits = self.logits_from(vocab.as_deref(), logits);
            if profile {
                if let (Some(s_iter), Some(s_run)) = (s_iter, s_run) {
                    t_run += s_run.elapsed();
                    t_host += s_run.saturating_duration_since(s_iter);
                }
                steps += 1;
            }

            if pos + 1 >= p {
                let nt = argmax(&logits);
                out.push(nt);
                let keep = on_token(nt);
                if nt == self.eos_id || !keep || out.len() >= n_new {
                    break;
                }
            }
        }
        if profile {
            let n = steps.max(1) as f64;
            eprintln!(
                "[lfm-profile] gpu_kv={use_gpu_kv} steps={steps} \
                 run={:.2}ms/step host_pre={:.2}ms/step",
                t_run.as_secs_f64() * 1e3 / n,
                t_host.as_secs_f64() * 1e3 / n,
            );
        }
        Ok(out)
    }
}

impl Lfm2GgufRunner {
    /// Zeroed per-layer state: KV caches for attention layers, the conv
    /// window for ShortConv layers — keyed by graph input name.
    fn initial_state(&self, max_kv: usize) -> HashMap<String, Vec<f32>> {
        let spec = &self.spec;
        let kv_dim = spec.num_key_value_heads * spec.head_dim;
        let mut state = HashMap::new();
        for il in 0..spec.num_hidden_layers {
            if spec.full_attn_layers.contains(&il) {
                state.insert(format!("k_in_{il}"), vec![0f32; max_kv * kv_dim]);
                state.insert(format!("v_in_{il}"), vec![0f32; max_kv * kv_dim]);
            } else {
                state.insert(
                    format!("conv_state_in_{il}"),
                    vec![0f32; (spec.conv_kernel - 1) * spec.conv_dim],
                );
            }
        }
        state
    }

    /// Inputs for decoding `tok` at position `pos`.
    fn step_feed(&self, vocab: Option<&[f32]>, tok: u32, pos: usize, max_kv: usize) -> StepFeed {
        let dh = self.spec.head_dim;
        let half = dh / 2;
        let mut cos = vec![0f32; half];
        let mut sin = vec![0f32; half];
        for i in 0..half {
            let fr = self.spec.rope_theta.powf(-(2.0 * i as f64) / dh as f64);
            let (s, c) = (pos as f64 * fr).sin_cos();
            cos[i] = c as f32;
            sin[i] = s as f32;
        }
        let mut key_mask = vec![0f32; max_kv];
        key_mask[..=pos].fill(1.0);
        let mut write_oh = vec![0f32; max_kv];
        write_oh[pos] = 1.0;
        let write_inv = write_oh.iter().map(|&x| 1.0 - x).collect();
        let token = match vocab {
            Some(table) => {
                let h = self.spec.hidden_size;
                table[tok as usize * h..(tok as usize + 1) * h].to_vec()
            }
            None => vec![tok as f32],
        };
        StepFeed {
            token,
            cos,
            sin,
            key_mask,
            write_oh,
            write_inv,
        }
    }

    /// Output 0 is the logits — or, with a host vocabulary, the final hidden
    /// state, scored here against the tied embedding.
    fn logits_from(&self, vocab: Option<&[f32]>, out0: Vec<f32>) -> Vec<f32> {
        let Some(table) = vocab else {
            return out0;
        };
        table
            .chunks_exact(self.spec.hidden_size)
            .map(|row| row.iter().zip(&out0).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// [`generate`](Self::generate) on wasm WebGPU, where reading results back
    /// from the GPU cannot block. Every step passes the full state through the
    /// graph (no resident-handle path in the browser).
    #[cfg(target_arch = "wasm32")]
    pub async fn generate_async(
        &self,
        prompt: &[u32],
        n_new: usize,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(anyhow!("lfm2: empty prompt"));
        }
        let need = prompt.len() + n_new;
        // Take the cached session out for the run instead of holding the
        // mutex across awaits; it goes back when the run ends.
        let cached = self.decode.lock().unwrap().take();
        let mut session = match cached {
            Some(s) if s.max_kv >= need => s,
            _ => self.build_decode_session(need.div_ceil(128) * 128)?,
        };
        let result = self
            .decode_async(&mut session, prompt, n_new, &mut on_token)
            .await;
        *self.decode.lock().unwrap() = Some(session);
        result
    }

    #[cfg(target_arch = "wasm32")]
    async fn decode_async(
        &self,
        session: &mut DecodeSession,
        prompt: &[u32],
        n_new: usize,
        on_token: &mut impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let max_kv = session.max_kv;
        let mut state = self.initial_state(max_kv);
        // Keep the per-layer KV + conv state on the GPU, as `generate` does
        // natively: bind each `_in_` once, feed it from its `_out_` on the
        // device, and read back only output 0 — the full state used to make a
        // host round trip every token.
        let compiled = &mut session.compiled;
        let mut resident = rlx_core::device_supports_gpu_kv(compiled.device());
        if resident {
            for (j, out_name) in session.names.iter().enumerate() {
                let in_name = out_name.replace("_out_", "_in_");
                if let Some(buf) = state.get(&in_name) {
                    if !compiled.bind_gpu_handle(&in_name, buf) {
                        resident = false;
                        break;
                    }
                    compiled.set_gpu_handle_feed(&in_name, j + 1);
                }
            }
        }
        let mut out = Vec::with_capacity(n_new);
        for pos in 0..max_kv {
            let tok = match prompt.get(pos) {
                Some(&t) => t,
                None => match out.last() {
                    Some(&t) => t,
                    None => break,
                },
            };
            let feed = self.step_feed(session.vocab.as_deref(), tok, pos, max_kv);
            let fixed = feed.inputs(session.vocab.is_some());
            let wants_logits = pos + 1 >= prompt.len();
            let first = if resident {
                // Prompt tokens read nothing back: no wait between them.
                let read: &[usize] = if wants_logits { &[0] } else { &[] };
                let mut outs = compiled
                    .run_read_outputs_async(&fixed, Some(read))
                    .await
                    .map_err(|e| anyhow!("lfm2: {e}"))?;
                if !wants_logits {
                    continue;
                }
                if outs.is_empty() {
                    return Err(anyhow!("lfm2: no logits"));
                }
                outs.swap_remove(0)
            } else {
                let mut inputs: Vec<(&str, &[f32])> = fixed.to_vec();
                for (k, v) in &state {
                    inputs.push((k.as_str(), v.as_slice()));
                }
                let outs = compiled
                    .run_async(&inputs)
                    .await
                    .map_err(|e| anyhow!("lfm2: {e}"))?;
                let mut outs = outs.into_iter();
                let first = outs.next().ok_or_else(|| anyhow!("lfm2: no logits"))?;
                for (name, value) in session.names.iter().zip(outs) {
                    if let Some(buf) = state.get_mut(&name.replace("_out_", "_in_")) {
                        buf.copy_from_slice(&value);
                    }
                }
                first
            };
            if wants_logits {
                let nt = argmax(&self.logits_from(session.vocab.as_deref(), first));
                out.push(nt);
                if nt == self.eos_id || !on_token(nt) || out.len() >= n_new {
                    break;
                }
            }
        }
        Ok(out)
    }
}

fn argmax(row: &[f32]) -> u32 {
    rlx_runtime::search::argmax_u32(row)
}

impl rlx_cli::LmRunner for Lfm2GgufRunner {
    fn family(&self) -> &'static str {
        "lfm"
    }
    fn vocab_size(&self) -> usize {
        self.spec.vocab_size
    }
    fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        Lfm2GgufRunner::predict_logits(self, prompt_ids)
    }
    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        Lfm2GgufRunner::generate(self, prompt_ids, n_new, on_token)
    }
}

/// Convenience: resolve the GGUF path (dir with one `.gguf`, or the file).
pub fn resolve_gguf(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    if path.is_dir() {
        let mut ggufs: Vec<PathBuf> = std::fs::read_dir(path)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
            .collect();
        ggufs.sort();
        return ggufs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no .gguf in {}", path.display()));
    }
    Err(anyhow!("weights path not found: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_remap_covers_all_lfm2_tensors() {
        assert_eq!(
            hf_to_gguf_name("model.embed_tokens.weight"),
            "token_embd.weight"
        );
        assert_eq!(
            hf_to_gguf_name("model.embedding_norm.weight"),
            "token_embd_norm.weight"
        );
        assert_eq!(
            hf_to_gguf_name("model.layers.0.operator_norm.weight"),
            "blk.0.attn_norm.weight"
        );
        assert_eq!(
            hf_to_gguf_name("model.layers.3.conv.in_proj.weight"),
            "blk.3.shortconv.in_proj.weight"
        );
        assert_eq!(
            hf_to_gguf_name("model.layers.2.self_attn.q_proj.weight"),
            "blk.2.attn_q.weight"
        );
        assert_eq!(
            hf_to_gguf_name("model.layers.2.self_attn.k_layernorm.weight"),
            "blk.2.attn_k_norm.weight"
        );
        assert_eq!(
            hf_to_gguf_name("model.layers.5.feed_forward.w2.weight"),
            "blk.5.ffn_down.weight"
        );
        // Pass-through for unknown names.
        assert_eq!(hf_to_gguf_name("something.else"), "something.else");
    }
}
