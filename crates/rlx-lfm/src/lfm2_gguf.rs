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
use rlx_core::standard_decoder::{Lfm2Spec, build_lfm2_decode, build_lfm2_prefill};
use rlx_core::weight_loader::WeightLoader;
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
    if let Some(rest) = name.strip_prefix("model.layers.") {
        if let Some((idx, suffix)) = rest.split_once('.') {
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

/// A compiled + weight-attached decode graph, cached and reused across
/// [`Lfm2GgufRunner::generate`] calls so repeated generations skip the
/// per-call graph build and the ~1.6 GB weight re-attach.
struct DecodeSession {
    compiled: rlx_runtime::CompiledGraph,
    names: Vec<String>,
    max_kv: usize,
}

/// LFM2 / LFM2.5 GGUF text runner (hybrid ShortConv + GQA attention).
///
/// [`generate`](Self::generate) uses **incremental O(n) decode**: one fixed-max
/// [`build_lfm2_decode`] graph stepped one token at a time, with a per-layer
/// conv-state cache (ShortConv) and KV cache (attention). Weights stay packed
/// (`DequantMatMul`). The compiled graph is cached and reused across calls (the
/// CPU dequant cache is process-global, so warmup is amortized too).
/// [`generate_prefill`](Self::generate_prefill) keeps the O(n²) re-prefill path
/// as a correctness reference.
pub struct Lfm2GgufRunner {
    weights: PathBuf,
    device: Device,
    spec: Lfm2Spec,
    eos_id: u32,
    decode: std::sync::Mutex<Option<DecodeSession>>,
}

impl Lfm2GgufRunner {
    /// Open a `.gguf` LFM2/LFM2.5 checkpoint and parse its config.
    pub fn open(weights: impl Into<PathBuf>, device: Device) -> Result<Self> {
        let weights = weights.into();
        let raw = GgufFile::from_path(&weights)
            .map_err(|e| anyhow!("lfm2: open GGUF {}: {e}", weights.display()))?;
        let spec = lfm2_spec_from_gguf(&raw)?;
        let eos_id = meta_u32(&raw, "tokenizer.ggml.eos_token_id").unwrap_or(u32::MAX);
        rlx_core::validate_standard_device("lfm", device)?;
        Ok(Self {
            weights,
            device,
            spec,
            eos_id,
            decode: std::sync::Mutex::new(None),
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
    fn compile_for(&self, seq: usize) -> Result<rlx_runtime::CompiledGraph> {
        let seq = seq.max(1);
        let inner = rlx_core::weight_registry::open_weight_loader(&self.weights)
            .map_err(|e| anyhow!("lfm2: open weights {}: {e}", self.weights.display()))?;
        let mut shim = GgufNameShim::new(inner);
        let exec_device = self.pick_exec_device(&shim);
        let mut packed: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
        let (graph, params) = build_lfm2_prefill(&self.spec, &mut shim, seq, &mut packed)?;
        let opts = packed_compile_opts(exec_device);
        let mut compiled = rlx_core::flow_bridge::packed_gguf_compile_guard(exec_device, || {
            Session::new(exec_device).compile_with(graph, &opts)
        });
        self.attach(&mut compiled, &params, &packed, &shim)?;
        Ok(compiled)
    }

    /// Attach dense params + packed (mmap-borrowed) weights to a compiled graph.
    fn attach(
        &self,
        compiled: &mut rlx_runtime::CompiledGraph,
        params: &HashMap<String, Vec<f32>>,
        packed: &HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
        shim: &GgufNameShim,
    ) -> Result<()> {
        for (n, d) in params {
            compiled.set_param(n, d);
        }
        for (n, (bytes, _scheme, _shape)) in packed {
            // Empty bytes = GGUF zero-copy marker: borrow the packed blob from
            // the loader's mmap (remapped to the GGUF name by the shim).
            let slice = if bytes.is_empty() {
                shim.tensor_bytes_borrowed(n)
                    .ok_or_else(|| anyhow!("lfm2: packed weight {n} bytes unavailable at attach"))?
            } else {
                bytes.as_slice()
            };
            compiled.set_param_typed(n, slice, rlx_ir::DType::U8);
        }
        Ok(())
    }

    /// Choose the device to actually compile/run the packed graph on.
    ///
    /// The scheme is fine on every backend (Q4_K), but native **wgpu / Vulkan**
    /// execution of the LFM2 **ShortConv** graph currently returns incorrect
    /// output — the depthwise causal conv routes through those backends' host
    /// round-trip path (an op qwen3 / llama don't use), which is not yet
    /// correct here; CPU / Metal / MLX are all correct. So run the packed
    /// prefill on the host CPU for wgpu / Vulkan, giving correct results when
    /// `--device gpu`/`vulkan` is selected. Metal / MLX / CUDA / ROCm / CoreML
    /// keep their own device. Override with `RLX_LFM_FORCE_NATIVE=1`.
    fn pick_exec_device(&self, _shim: &GgufNameShim) -> Device {
        let native = rlx_core::flow_bridge::packed_gguf_execution_device(self.device);
        if matches!(native, Device::Gpu | Device::Vulkan)
            && !rlx_ir::env::flag("RLX_LFM_FORCE_NATIVE")
        {
            eprintln!(
                "[rlx-lfm] {native:?}: native ShortConv execution is not yet correct — \
                 running the packed prefill on host CPU (set RLX_LFM_FORCE_NATIVE=1 to force native)"
            );
            return Device::Cpu;
        }
        native
    }

    /// Last-position logits `[vocab]` after a single prefill of `prompt`.
    pub fn predict_logits(&self, prompt: &[u32]) -> Result<Vec<f32>> {
        if prompt.is_empty() {
            return Err(anyhow!("lfm2: empty prompt"));
        }
        let seq = prompt.len();
        let mut compiled = self.compile_for(seq)?;
        let ids_f32: Vec<f32> = prompt.iter().map(|&t| t as f32).collect();
        let out = compiled.run(&[("input_ids", ids_f32.as_slice())]);
        let logits = out
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("lfm2: no logits"))?;
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
        let mut compiled = self.compile_for(total)?;
        let mut buf = vec![0u32; total];
        buf[..prompt.len()].copy_from_slice(prompt);
        let mut cur = prompt.len();
        let mut out = Vec::with_capacity(n_new);
        while cur < total {
            let ids_f32: Vec<f32> = buf.iter().map(|&t| t as f32).collect();
            let res = compiled.run(&[("input_ids", ids_f32.as_slice())]);
            let logits = res.first().ok_or_else(|| anyhow!("lfm2: no logits"))?;
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
        let inner = rlx_core::weight_registry::open_weight_loader(&self.weights)
            .map_err(|e| anyhow!("lfm2: open weights {}: {e}", self.weights.display()))?;
        let mut shim = GgufNameShim::new(inner);
        let exec_device = self.pick_exec_device(&shim);
        // CPU decode: default the wide-matmul int8 fast path ON (measured ~2×
        // decode TPS on LFM2.5-Q4_K_M). The dispatch only affects m==1 Q4_K GEMVs
        // whose output width ≥ this threshold (LM head + large FFN); smaller
        // matmuls stay on Accelerate/AMX. It quantizes the activation to Q8, so
        // it is NOT bit-identical to the pure-F32 path (can flip rare near-tie
        // greedy tokens). Only set when the user hasn't chosen a value — override
        // with `RLX_Q4K_FUSED_MIN_N=<n>` (set huge, e.g. that many 9s, to disable).
        if exec_device == Device::Cpu && rlx_ir::env::is_unset("RLX_Q4K_FUSED_MIN_N") {
            rlx_ir::env::set("RLX_Q4K_FUSED_MIN_N", "2048");
        }
        let mut packed: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
        let (graph, params, names) = build_lfm2_decode(&self.spec, &mut shim, max_kv, &mut packed)?;
        let opts = packed_compile_opts(exec_device);
        let mut compiled = rlx_core::flow_bridge::packed_gguf_compile_guard(exec_device, || {
            Session::new(exec_device).compile_with(graph, &opts)
        });
        self.attach(&mut compiled, &params, &packed, &shim)?;
        Ok(DecodeSession {
            compiled,
            names,
            max_kv,
        })
    }

    /// **O(n)** greedy generation via incremental decode. Steps one token at a
    /// time over a cached fixed-`max_kv` [`build_lfm2_decode`] graph: each
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
        let spec = &self.spec;
        let dh = spec.head_dim;
        let half = dh / 2;
        let kv_dim = spec.num_key_value_heads * spec.head_dim;
        let cdim = spec.conv_dim;
        let kc = spec.conv_kernel;
        let theta = spec.rope_theta;
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
        } = session;
        let max_kv = *max_kv;

        // Cache buffers keyed by graph *input* name (zeroed initial state).
        let mut state: HashMap<String, Vec<f32>> = HashMap::new();
        for il in 0..spec.num_hidden_layers {
            if spec.full_attn_layers.contains(&il) {
                state.insert(format!("k_in_{il}"), vec![0f32; max_kv * kv_dim]);
                state.insert(format!("v_in_{il}"), vec![0f32; max_kv * kv_dim]);
            } else {
                state.insert(format!("conv_state_in_{il}"), vec![0f32; (kc - 1) * cdim]);
            }
        }

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
            let s_iter = std::time::Instant::now();
            // Per-step inputs (kept alive across the borrow of `state`).
            let tok_f = [tok as f32];
            let mut cos = vec![0f32; half];
            let mut sin = vec![0f32; half];
            for i in 0..half {
                let fr = theta.powf(-(2.0 * i as f64) / dh as f64);
                let (s, c) = (pos as f64 * fr).sin_cos();
                cos[i] = c as f32;
                sin[i] = s as f32;
            }
            let mut key_mask = vec![0f32; max_kv];
            key_mask[..=pos].fill(1.0);
            let mut write_oh = vec![0f32; max_kv];
            write_oh[pos] = 1.0;
            let write_inv: Vec<f32> = write_oh.iter().map(|&x| 1.0 - x).collect();

            let fixed: [(&str, &[f32]); 6] = [
                ("token_id", &tok_f[..]),
                ("rope_cos", &cos),
                ("rope_sin", &sin),
                ("key_mask", &key_mask),
                ("kv_write_oh", &write_oh),
                ("kv_write_inv", &write_inv),
            ];

            let s_run = std::time::Instant::now();
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
            if profile {
                t_run += s_run.elapsed();
                t_host += s_run.saturating_duration_since(s_iter);
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

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
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
