// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// BitNet Qwen2-1.5B decoder wiring. The LM GGUF loads directly via a
// `WeightLoader` over the mmap'd file.
//
// Two paths:
//   * `generate` (dense) — the rlx-qwen3 flow: `inputs_embeds` prefill exporting
//     KV, then bucketed greedy KV-decode (same pattern as rlx-qwen3-asr). All
//     weights are dequantized to f32.
//   * `generate_packed` (default) — the BitNet path: the ternary `I2_S`
//     projections are transcoded to rlx's `Q2_0` scheme and served packed via
//     `take_packed`, so they run through `Op::DequantMatMul` (2-bit in the
//     arena). Built on `rlx_core::build_standard_decoder_packed` with the new
//     `embeds_input` mode (audio features spliced into the token stream);
//     full-sequence recompute (no KV cache), bucketed by power-of-two length.

use anyhow::{Result, anyhow, ensure};
use rlx_core::autoregressive::run_packed_prefill;
use rlx_core::autoregressive::{KvCacheState, kv_from_prefill_outputs, run_bucketed_kv_decode};
use rlx_core::build_standard_decoder_packed;
use rlx_core::flow_bridge::{WeightLoaderSource, compile_options_from_profile};
use rlx_core::flow_bridge::{
    compile_options_for_packed_gguf_prefill_with_profile, packed_gguf_compile_guard,
    packed_gguf_execution_device,
};
use rlx_core::flow_util::{compile_built, graph_from_built};
use rlx_core::weight_loader::WeightLoader;
use rlx_flow::blocks::{
    LmHeadStage, Qwen3DecoderSpec, RopeTablesStage, qwen3_prefill_layer_fused_kv,
};
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, ModelFlow, SideOutputs};
use rlx_gguf::GgufFile;
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_ir::{DType, Shape};
use rlx_qwen3::builder::qwen3_decoder_spec;
use rlx_qwen3::{Qwen3Config, Qwen3DecodeOpts, build_qwen3_decode_built};
use rlx_runtime::Device;
use rlx_runtime::attn_mask::bucket_decode_mask;
use rlx_runtime::compile_cache::{BucketedCompileCache, CacheRunInput};
use rlx_runtime::{CompiledGraph, Session};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::config::LmConfig;
use crate::embed::argmax;

/// Shared cache of I2_S→Q2_0-transcoded projection bytes (ggml name → packed
/// bytes), so the ternary repack happens once across all bucket rebuilds.
type PackedCache = Arc<Mutex<HashMap<String, Vec<u8>>>>;

/// Stop tokens: `<|im_end|>` and `<|endoftext|>`.
const EOS_IDS: [i64; 2] = [151645, 151643];

/// Build a Qwen2-mode [`Qwen3Config`] from the ASR LM config.
pub fn qwen3_config(lm: &LmConfig) -> Qwen3Config {
    Qwen3Config {
        vocab_size: lm.vocab_size,
        hidden_size: lm.hidden_size,
        intermediate_size: lm.intermediate_size,
        num_hidden_layers: lm.num_hidden_layers,
        num_attention_heads: lm.num_attention_heads,
        num_key_value_heads: lm.num_key_value_heads,
        head_dim: lm.head_dim,
        max_position_embeddings: lm.max_position_embeddings,
        rms_norm_eps: lm.rms_norm_eps as f64,
        rope_theta: lm.rope_theta as f64,
        hidden_act: "silu".to_string(),
        tie_word_embeddings: lm.tie_word_embeddings,
        attention_bias: lm.attention_bias, // Qwen2: true
        qk_norm: lm.qk_norm,               // Qwen2: false
        sliding_window: None,
        max_window_layers: lm.num_hidden_layers,
        use_sliding_window: false,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    }
}

/// Map an HF-style Qwen2 weight key to the GGUF (ggml) tensor name.
fn map_key(key: &str) -> Option<String> {
    match key {
        "lm_head.weight" => return Some("output.weight".into()),
        "model.norm.weight" => return Some("output_norm.weight".into()),
        "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        _ => {}
    }
    let rest = key.strip_prefix("model.layers.")?;
    let (idx, tail) = rest.split_once('.')?;
    let g = match tail {
        "self_attn.q_proj.weight" => format!("blk.{idx}.attn_q.weight"),
        "self_attn.q_proj.bias" => format!("blk.{idx}.attn_q.bias"),
        "self_attn.k_proj.weight" => format!("blk.{idx}.attn_k.weight"),
        "self_attn.k_proj.bias" => format!("blk.{idx}.attn_k.bias"),
        "self_attn.v_proj.weight" => format!("blk.{idx}.attn_v.weight"),
        "self_attn.v_proj.bias" => format!("blk.{idx}.attn_v.bias"),
        "self_attn.o_proj.weight" => format!("blk.{idx}.attn_output.weight"),
        "mlp.gate_proj.weight" => format!("blk.{idx}.ffn_gate.weight"),
        "mlp.up_proj.weight" => format!("blk.{idx}.ffn_up.weight"),
        "mlp.down_proj.weight" => format!("blk.{idx}.ffn_down.weight"),
        "input_layernorm.weight" => format!("blk.{idx}.attn_norm.weight"),
        "post_attention_layernorm.weight" => format!("blk.{idx}.ffn_norm.weight"),
        _ => return None,
    };
    Some(g)
}

/// Transpose a row-major `[r, c]` matrix to `[c, r]`.
fn transpose2d(data: &[f32], shape: &[usize]) -> (Vec<f32>, Vec<usize>) {
    if shape.len() != 2 {
        return (data.to_vec(), shape.to_vec());
    }
    let (r, c) = (shape[0], shape[1]);
    let mut out = vec![0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = data[i * c + j];
        }
    }
    (out, vec![c, r])
}

/// A [`WeightLoader`] over the LM GGUF. Dequantizes on demand (I2_S/Q6_K/F16/F32
/// → f32) and presents tensors in torch `[out, in]` order (GGUF stores them
/// innermost-first; the flat data is already `[out, in]` row-major, so only the
/// reported shape is reversed).
pub struct VibeLmLoader {
    g: Arc<GgufFile>,
    packed_cache: PackedCache,
    taken: HashSet<String>,
}

impl VibeLmLoader {
    pub fn new(g: Arc<GgufFile>, packed_cache: PackedCache) -> Self {
        Self {
            g,
            packed_cache,
            taken: HashSet::new(),
        }
    }
}

impl WeightLoader for VibeLmLoader {
    fn format_id(&self) -> &'static str {
        "gguf"
    }
    fn len(&self) -> usize {
        self.g.keys().count()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let gname = map_key(key).ok_or_else(|| anyhow!("unmapped LM weight key `{key}`"))?;
        ensure!(
            self.g.get(&gname).is_some(),
            "LM GGUF missing tensor `{gname}` (for `{key}`)"
        );
        let (data, shape) = self.g.dequant_f32(&gname)?;
        self.taken.insert(gname);
        let torch_shape: Vec<usize> = shape.iter().rev().copied().collect();
        Ok((data, torch_shape))
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.take(key)?;
        Ok(transpose2d(&data, &shape))
    }
    /// The BitNet ternary projections (GGUF `I2_S`) are transcoded to rlx's
    /// `Q2_0` ternary scheme and served packed → `Op::DequantMatMul` (weights
    /// stay 2-bit in the arena). Everything else (norms, Q/K/V biases, Q6_K
    /// embed, F16 `lm_head`) returns `None` and loads dense via `take`.
    fn take_packed(
        &mut self,
        key: &str,
    ) -> Result<Option<(Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)>> {
        let gname = match map_key(key) {
            Some(g) => g,
            None => return Ok(None),
        };
        let t = match self.g.get(&gname) {
            Some(t) => t.clone(),
            None => return Ok(None),
        };
        if t.dtype != rlx_gguf::GgmlType::I2_S {
            return Ok(None);
        }
        let n = t.n_elements();
        let (in_dim, out_dim) = (t.shape[0], t.shape[1]); // GGUF ne = [in, out]
        let bytes = {
            let mut cache = self.packed_cache.lock().expect("packed cache poisoned");
            if let Some(b) = cache.get(&gname) {
                b.clone()
            } else {
                let raw = self.g.tensor_bytes(&t)?;
                let f32w = rlx_gguf::i2s_dequant::dequant_i2_s(raw, n)?;
                let q2 = rlx_gguf::q2_dequant::quantize_q2_0(&f32w)?;
                cache.insert(gname.clone(), q2.clone());
                q2
            }
        };
        self.taken.insert(gname);
        Ok(Some((
            bytes,
            rlx_ir::quant::QuantScheme::GgufQ2_0,
            vec![out_dim, in_dim],
        )))
    }
    fn remaining_keys(&self) -> Vec<String> {
        Vec::new()
    }
    fn arch_hint(&self) -> Option<&str> {
        Some("qwen2")
    }
}

/// Load `token_embd.weight` (Q6_K → f32) as `(matrix[vocab*hidden], vocab, hidden)`.
/// GGUF shape is `[hidden, vocab]` (innermost-first), so the flat data is
/// `[vocab, hidden]` row-major — row `t` is token `t`'s embedding.
pub fn token_embed_matrix(g: &GgufFile) -> Result<(Vec<f32>, usize, usize)> {
    let t = g
        .get("token_embd.weight")
        .ok_or_else(|| anyhow!("LM GGUF missing token_embd.weight"))?;
    let hidden = t.shape[0];
    let vocab = t.shape[1];
    let (data, _) = g.dequant_f32("token_embd.weight")?;
    ensure!(
        data.len() == vocab * hidden,
        "token_embd len {} != vocab*hidden {}",
        data.len(),
        vocab * hidden
    );
    Ok((data, vocab, hidden))
}

/// The BitNet Qwen2 LM decoder.
pub struct VibeLm {
    pub cfg: Qwen3Config,
    g: Arc<GgufFile>,
    device: Device,
    packed_cache: PackedCache,
}

impl VibeLm {
    /// Open the LM GGUF (mmap-backed) and build the decoder config.
    pub fn load(lm_path: &Path, lm: &LmConfig, device: Device) -> Result<Self> {
        let g = Arc::new(GgufFile::from_path_mmap(lm_path)?);
        Ok(Self {
            cfg: qwen3_config(lm),
            g,
            device,
            packed_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Access the underlying GGUF (for `token_embed_matrix`).
    pub fn gguf(&self) -> &GgufFile {
        &self.g
    }

    /// Greedy generate up to `max_new` tokens from prefilled `inputs_embeds`
    /// `[seq, hidden]`. Returns the generated token ids (excluding the prompt).
    pub fn generate(&self, inputs_embeds: &[f32], seq: usize, max_new: usize) -> Result<Vec<i64>> {
        let batch = 1;
        let device = self.device;
        let skip_fusion = matches!(device, Device::Metal);
        let cfg = &self.cfg;
        let vocab = cfg.vocab_size;
        let layers = cfg.num_hidden_layers;
        let kv_dim = cfg.kv_proj_dim();

        ensure!(
            inputs_embeds.len() == seq * cfg.hidden_size,
            "inputs_embeds len {} != seq*hidden {}",
            inputs_embeds.len(),
            seq * cfg.hidden_size
        );

        // Prefill.
        let outs = {
            let mut loader = VibeLmLoader::new(self.g.clone(), self.packed_cache.clone());
            let built = build_prefill_built(cfg, &mut loader, batch, seq, skip_fusion)?;
            let params = built.params().clone();
            let mut prefill = compile_built(built, device)?;
            for (n, d) in &params {
                prefill.set_param(n, d);
            }
            prefill.run(&[("inputs_embeds", inputs_embeds)])
        };
        ensure!(
            outs[0].len() == batch * vocab,
            "prefill logits len {} != {}",
            outs[0].len(),
            batch * vocab
        );
        let (logits0, mut kv) = kv_from_prefill_outputs(outs, batch, seq, kv_dim, layers)?;
        let mut next = argmax(&logits0);
        if std::env::var("VIBEASR_DEBUG").is_ok() {
            let finite = logits0.iter().filter(|v| v.is_finite()).count();
            let (mn, mx) = logits0
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            eprintln!(
                "[dbg] prefill logits: finite {finite}/{}, range [{mn:.3},{mx:.3}], argmax={next}",
                logits0.len()
            );
        }

        let mut generated: Vec<i64> = Vec::new();

        // Bucketed greedy decode.
        let max_total = seq.saturating_add(max_new).max(1) as u64;
        let mut decode_cache = BucketedCompileCache::power_of_two_ladder(device, 1, max_total);
        let mut decode_profile = CompileProfile::llama32_decode();
        if skip_fusion {
            decode_profile.fusion.skip = true;
        }
        let options =
            compile_options_from_profile(&decode_profile, device, KernelDispatchConfig::default());

        for past_seq in seq..seq.saturating_add(max_new) {
            if EOS_IDS.contains(&next) {
                break;
            }
            generated.push(next);

            let upper = decode_cache
                .bucket_for(past_seq as u64)
                .and_then(|idx| {
                    decode_cache
                        .buckets()
                        .nth(idx)
                        .map(|r| (r.end - 1) as usize)
                })
                .unwrap_or(past_seq);
            let (cos, sin) = rope_slice(cfg, past_seq);
            let token_f = [next as f32];
            let mask = bucket_decode_mask(past_seq, upper);
            let fixed = [
                CacheRunInput {
                    name: "input_ids",
                    data: &token_f,
                    row_inner: None,
                },
                CacheRunInput {
                    name: "rope_cos",
                    data: &cos,
                    row_inner: None,
                },
                CacheRunInput {
                    name: "rope_sin",
                    data: &sin,
                    row_inner: None,
                },
                CacheRunInput {
                    name: "mask",
                    data: &mask,
                    row_inner: None,
                },
            ];
            let g = self.g.clone();
            let pc = self.packed_cache.clone();
            let cfg_c = cfg.clone();
            let (logits, new_k, new_v) = run_bucketed_kv_decode(
                &mut decode_cache,
                past_seq,
                &kv,
                kv_dim,
                layers,
                &fixed,
                |upper_u64| {
                    let mut loader = VibeLmLoader::new(g.clone(), pc.clone());
                    let built = build_decode_built(
                        &cfg_c,
                        &mut loader,
                        batch,
                        upper_u64 as usize,
                        skip_fusion,
                    )
                    .expect("build decode graph");
                    graph_from_built(built).expect("lower decode graph")
                },
                &options,
            )?;
            next = argmax(&logits);
            kv = KvCacheState {
                past_len: past_seq + 1,
                layers_kv_base: vec![0; new_k.len()],
                layers_k: new_k,
                layers_v: new_v,
            };
        }

        if std::env::var("VIBEASR_DEBUG").is_ok() {
            eprintln!(
                "[dbg] generated {} ids: {:?}",
                generated.len(),
                &generated[..generated.len().min(24)]
            );
        }
        Ok(generated)
    }

    /// Greedy generate using the **packed BitNet** path: the LM's ternary
    /// projections stay 2-bit (`Q2_0` `DequantMatMul`) in the arena (~7× smaller
    /// than dense f32). The full sequence is re-run each step (bucketed,
    /// no KV cache — `build_standard_decoder_packed` is full-sequence), with
    /// generated tokens embedded host-side into the growing `inputs_embeds`.
    pub fn generate_packed(
        &self,
        prompt_embeds: &[f32],
        seq_prompt: usize,
        token_embed: &[f32],
        vocab: usize,
        hidden: usize,
        max_new: usize,
    ) -> Result<Vec<i64>> {
        ensure!(
            prompt_embeds.len() == seq_prompt * hidden,
            "prompt_embeds len {} != seq*hidden {}",
            prompt_embeds.len(),
            seq_prompt * hidden
        );
        let exec = packed_gguf_execution_device(self.device);
        // This GGUF ships a separate F16 `output.weight` (not tied), so force
        // `tie=false` → the builder loads `lm_head.weight` (our `output.weight`).
        let mut spec_cfg = self.cfg.clone();
        spec_cfg.tie_word_embeddings = false;
        let spec = qwen3_decoder_spec(&spec_cfg);
        let opts = compile_options_for_packed_gguf_prefill_with_profile(
            &CompileProfile::qwen3_prefill(),
            exec,
        );

        let mut embeds = prompt_embeds.to_vec();
        let mut cur_len = seq_prompt;
        let mut generated: Vec<i64> = Vec::new();
        let mut bucket_size = 0usize;
        let mut compiled: Option<CompiledGraph> = None;

        loop {
            let bucket = cur_len.next_power_of_two().max(16);
            if bucket != bucket_size {
                let mut loader = VibeLmLoader::new(self.g.clone(), self.packed_cache.clone());
                let mut packed: HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)> =
                    HashMap::new();
                let (graph, params) = build_standard_decoder_packed(
                    &spec,
                    &mut loader,
                    1,
                    bucket,
                    /*with_lm_head*/ true,
                    /*last_token_from_input*/ true,
                    /*embeds_input*/ true,
                    &mut packed,
                )?;
                let mut c = packed_gguf_compile_guard(exec, || {
                    Session::new(exec).compile_with(graph, &opts)
                });
                for (n, d) in &params {
                    c.set_param(n, d);
                }
                for (n, (b, _, _)) in &packed {
                    c.set_param_typed(n, b, DType::U8);
                }
                compiled = Some(c);
                bucket_size = bucket;
            }
            let c = compiled.as_mut().expect("compiled bucket");

            let mut padded = embeds.clone();
            padded.resize(bucket * hidden, 0.0);
            let last = [(cur_len - 1) as f32];
            let inputs: Vec<(&str, &[f32])> = vec![
                ("inputs_embeds", padded.as_slice()),
                ("last_token_idx", last.as_slice()),
            ];
            let out = run_packed_prefill(c, exec, cur_len, bucket, &inputs);
            let logits = out.into_iter().next().ok_or_else(|| anyhow!("no logits"))?;
            let next = argmax(&logits);
            if EOS_IDS.contains(&next) || generated.len() >= max_new {
                break;
            }
            generated.push(next);
            let t = next as usize;
            ensure!(t < vocab, "generated token {t} >= vocab {vocab}");
            embeds.extend_from_slice(&token_embed[t * hidden..(t + 1) * hidden]);
            cur_len += 1;
        }
        Ok(generated)
    }
}

/// Prefill the Qwen2 trunk from fused `inputs_embeds` `[batch, seq, hidden]`,
/// exporting `[logits, k0, v0, …]`. Adapted from rlx-qwen3-asr::lm_flow.
fn build_prefill_built(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    skip_fusion: bool,
) -> Result<BuiltModel> {
    let mut profile = CompileProfile::llama32_prefill();
    if skip_fusion {
        profile.fusion.skip = true;
    }
    let f = DType::F32;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let dh = cfg.head_dim;
    let eps = cfg.rms_norm_eps as f32;

    let hidden_shape = Shape::new(&[batch, seq, h], f);
    let (cos_data, sin_data) = rope_tables(cfg);

    let spec = Qwen3DecoderSpec {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: dh,
        eps,
        hidden_shape: hidden_shape.clone(),
        batch,
        seq,
        qk_norm: cfg.qk_norm,
        attention_bias: cfg.attention_bias,
        mask: rlx_ir::op::MaskKind::Causal,
    };

    let kv_sink = SideOutputs::new();

    let mut flow = ModelFlow::new("vibeasr_prefill")
        .with_profile(profile)
        .input("inputs_embeds", hidden_shape)
        .rope_tables(RopeTablesStage::param(
            cfg.max_position_embeddings,
            dh / 2,
            cos_data,
            sin_data,
        ))
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", dh);

    flow = flow.repeat_layers(cfg.num_hidden_layers, {
        let spec = spec.clone();
        let sink = kv_sink.clone();
        move |i| qwen3_prefill_layer_fused_kv(i, spec.clone(), sink.inner())
    });

    flow = flow.gather_last_token_at(batch, seq).final_norm(eps);

    let built = flow
        .raw_stage(FlowStage::LmHead(LmHeadStage::separate(
            "lm_head.weight",
            cfg.vocab_size,
            h,
        )))
        .output("logits")
        .build(&mut WeightLoaderSource(weights))?
        .with_extra_hir_outputs(kv_sink.drain());

    Ok(built)
}

/// One token-id decode step with a bucketed KV cache (custom mask).
fn build_decode_built(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    skip_fusion: bool,
) -> Result<BuiltModel> {
    let profile = if skip_fusion {
        let mut p = CompileProfile::llama32_decode();
        p.fusion.skip = true;
        Some(p)
    } else {
        None
    };
    let opts = Qwen3DecodeOpts {
        batch,
        past_seq,
        use_custom_mask: true,
        profile,
        ..Default::default()
    };
    build_qwen3_decode_built(cfg, weights, &opts)
}

/// RoPE cos/sin tables `[max_pos, head_dim/2]`.
fn rope_tables(cfg: &Qwen3Config) -> (Vec<f32>, Vec<f32>) {
    let dh = cfg.head_dim;
    let half = dh / 2;
    let mut cos = vec![0f32; cfg.max_position_embeddings * half];
    let mut sin = vec![0f32; cfg.max_position_embeddings * half];
    for pos in 0..cfg.max_position_embeddings {
        for i in 0..half {
            let freq = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / dh as f64);
            let (s, c) = (pos as f64 * freq).sin_cos();
            cos[pos * half + i] = c as f32;
            sin[pos * half + i] = s as f32;
        }
    }
    (cos, sin)
}

/// RoPE cos/sin row for one position (`[head_dim/2]`).
fn rope_slice(cfg: &Qwen3Config, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let dh = cfg.head_dim;
    let half = dh / 2;
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    for i in 0..half {
        let freq = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / dh as f64);
        let (s, c) = (pos as f64 * freq).sin_cos();
        cos[i] = c as f32;
        sin[i] = s as f32;
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_mapping() {
        assert_eq!(map_key("lm_head.weight").unwrap(), "output.weight");
        assert_eq!(map_key("model.norm.weight").unwrap(), "output_norm.weight");
        assert_eq!(
            map_key("model.layers.7.self_attn.q_proj.weight").unwrap(),
            "blk.7.attn_q.weight"
        );
        assert_eq!(
            map_key("model.layers.0.self_attn.k_proj.bias").unwrap(),
            "blk.0.attn_k.bias"
        );
        assert_eq!(
            map_key("model.layers.3.mlp.down_proj.weight").unwrap(),
            "blk.3.ffn_down.weight"
        );
        assert_eq!(
            map_key("model.layers.5.post_attention_layernorm.weight").unwrap(),
            "blk.5.ffn_norm.weight"
        );
    }

    #[test]
    fn transpose_roundtrip() {
        let d = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
        let (t, sh) = transpose2d(&d, &[2, 3]);
        assert_eq!(sh, vec![3, 2]);
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn config_is_qwen2() {
        let c = qwen3_config(&LmConfig::default());
        assert!(!c.qk_norm);
        assert!(c.attention_bias);
        assert_eq!(c.num_key_value_heads, 2);
        assert_eq!(c.head_dim, 128);
    }
}
