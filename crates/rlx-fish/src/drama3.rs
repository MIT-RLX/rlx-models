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

//! Drama 3 (`model_type: fish_omni`).
//!
//! ```text
//!  row t ─ embed ─▶ 40 × [ input norm ─ Gated DeltaNet | gated attention ─ +
//!                          post norm ─ shared expert + 8 of 256 experts ─ + ]
//!                 ─ final norm ─▶ h_t ─▶ main token ~ lm_head[eos ∪ band]
//!  tower at t: 4 shared layers (c0, + the codes 2..9 of row t)
//!            ─ 9 residual layers, layer i reads code i and its head draws i+1
//!  row t+1 = (E[main] + C[4096 + code 1]) / √2
//! ```
//!
//! Every tower layer attends across frames with its own KV cache, so the
//! tower is teacher-forced over the prompt (row `p` labelled with frame
//! `p + 1`) before decoding starts.
//!
//! Execution: one compiled graph per backbone layer (mixer, post norm,
//! router logits, shared expert) with the routed experts in between on the
//! host ([`crate::moe`]); one graph per tower layer. Prefill runs layer-major
//! over the whole prompt, and frees each layer's prefill graph once used so
//! the CPU never holds two copies of the backbone.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{Activation, MaskKind};
use rlx_ir::{DType, NodeId, RopeStyle, Shape};
use rlx_runtime::{CompiledGraph, Device};
use serde_json::Value;

use crate::moe::ExpertBank;
use crate::nn::{self, AttnSpec, BlockSpec, Builder, Keys, TensorOp};
use crate::sampling::{self, Flavor, Rng, SampleParams};
use crate::store::ShardStore;

pub(crate) const LM: &str = "model.language_model.";
const LAYERS: &str = "model.language_model.layers.";
pub(crate) const SHARED: &str = "audio_decoder_mtp_shared.layers.";
pub(crate) const RESIDUAL: &str = "audio_decoder_mtp.layers.";
pub(crate) const CODEBOOK_EMBED: &str = "audio_decoder.codebook_embeddings.weight";

const PREFILL_BUCKET: usize = 64;
const DECODE_BUCKET: usize = 256;
/// `l2norm` epsilon of the reference Gated DeltaNet.
const L2_EPS: f64 = 1e-6;

/// Repetition-aware sampling: when the main token repeats one of the last
/// `RAS_WINDOW` main tokens (prompt rows included), redraw it at these settings.
const RAS_WINDOW: usize = 16;
const RAS: SampleParams = SampleParams {
    temperature: 1.0,
    top_p: 0.95,
    top_k: 30,
};

/// Frames per chunk by default (`DRAMA3_CHUNK_MAX_NEW_TOKENS`, ~27.9 s).
pub const DEFAULT_MAX_NEW_TOKENS: usize = 600;
/// Longest prompt the reference accepts (`DRAMA3_MAX_PROMPT_ROWS`).
pub const MAX_PROMPT_ROWS: usize = 8192;

/// A backbone layer's token mixer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mixer {
    Linear,
    Full,
}

/// The codebook tower (`audio_decoder_config`).
#[derive(Clone, Debug)]
pub struct TowerConfig {
    pub dim: usize,
    pub ffn: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub rope_base: f64,
    pub rotary_dim: usize,
    pub vocab: usize,
    pub num_codebooks: usize,
    pub shared_layers: usize,
    pub vq_backbone_codebooks: usize,
}

/// Drama 3 `config.json`.
#[derive(Clone, Debug)]
pub struct DramaConfig {
    pub vocab_size: usize,
    pub hidden: usize,
    pub layers: Vec<Mixer>,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
    pub rotary_dim: usize,
    pub eps: f32,
    pub conv_kernel: usize,
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub moe_ffn: usize,
    pub shared_ffn: usize,
    pub norm_topk: bool,
    pub tower: TowerConfig,
    pub semantic_begin: u32,
    pub semantic_end: u32,
    pub eos: u32,
}

fn int(v: &Value, k: &str) -> Result<usize> {
    v.get(k)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .with_context(|| format!("config: missing integer {k}"))
}

fn num(v: &Value, k: &str) -> Result<f64> {
    v.get(k)
        .and_then(Value::as_f64)
        .with_context(|| format!("config: missing number {k}"))
}

impl DramaConfig {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let raw = std::fs::read(dir.join("config.json"))
            .with_context(|| format!("read {}/config.json", dir.display()))?;
        Self::from_value(&serde_json::from_slice(&raw)?)
    }

    pub fn from_value(v: &Value) -> Result<Self> {
        let mt = v.get("model_type").and_then(Value::as_str).unwrap_or("");
        ensure!(
            mt == "fish_omni",
            "not a Drama 3 checkpoint (model_type {mt:?})"
        );
        let t = v.get("text_config").context("config: text_config")?;
        let a = v
            .get("audio_decoder_config")
            .context("config: audio_decoder_config")?;
        let n = int(t, "num_hidden_layers")?;
        let layers: Vec<Mixer> = match t.get("layer_types").and_then(Value::as_array) {
            Some(list) => list
                .iter()
                .map(|x| match x.as_str() {
                    Some("linear_attention") => Ok(Mixer::Linear),
                    Some("full_attention") => Ok(Mixer::Full),
                    other => bail!("unknown layer type {other:?}"),
                })
                .collect::<Result<_>>()?,
            None => {
                let every = int(t, "full_attention_interval")?;
                (0..n)
                    .map(|i| {
                        if (i + 1) % every == 0 {
                            Mixer::Full
                        } else {
                            Mixer::Linear
                        }
                    })
                    .collect()
            }
        };
        ensure!(
            layers.len() == n,
            "{} layer types for {n} layers",
            layers.len()
        );
        let rope = t.get("rope_parameters").unwrap_or(t);
        let head_dim = int(t, "head_dim")?;
        let partial = rope
            .get("partial_rotary_factor")
            .or_else(|| t.get("partial_rotary_factor"))
            .and_then(Value::as_f64)
            .unwrap_or(0.25);
        let a_head = int(a, "head_dim")?;
        let tower = TowerConfig {
            dim: int(a, "dim")?,
            ffn: int(a, "intermediate_size")?,
            n_head: int(a, "n_head")?,
            n_kv: int(a, "n_local_heads")?,
            head_dim: a_head,
            eps: num(a, "norm_eps")? as f32,
            rope_base: num(a, "rope_base")?,
            rotary_dim: (a_head as f64 * num(a, "rotary_percent")?) as usize,
            vocab: int(a, "vocab_size")?,
            num_codebooks: int(a, "num_codebooks")?,
            shared_layers: int(a, "shared_enrichment_layers")?,
            vq_backbone_codebooks: int(a, "vq_backbone_codebooks")?,
        };
        let cfg = Self {
            vocab_size: int(t, "vocab_size")?,
            hidden: int(t, "hidden_size")?,
            layers,
            n_head: int(t, "num_attention_heads")?,
            n_kv: int(t, "num_key_value_heads")?,
            head_dim,
            rope_theta: num(rope, "rope_theta")?,
            rotary_dim: (head_dim as f64 * partial) as usize,
            eps: num(t, "rms_norm_eps")? as f32,
            conv_kernel: int(t, "linear_conv_kernel_dim")?,
            lin_k_heads: int(t, "linear_num_key_heads")?,
            lin_v_heads: int(t, "linear_num_value_heads")?,
            lin_k_dim: int(t, "linear_key_head_dim")?,
            lin_v_dim: int(t, "linear_value_head_dim")?,
            num_experts: int(t, "num_experts")?,
            top_k: int(t, "num_experts_per_tok")?,
            moe_ffn: int(t, "moe_intermediate_size")?,
            shared_ffn: int(t, "shared_expert_intermediate_size")?,
            norm_topk: t
                .get("norm_topk_prob")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            tower,
            semantic_begin: int(v, "semantic_start_token_id")? as u32,
            semantic_end: int(v, "semantic_end_token_id")? as u32,
            eos: v
                .get("eos_token_id")
                .or_else(|| t.get("eos_token_id"))
                .and_then(Value::as_u64)
                .context("config: eos_token_id")? as u32,
        };
        ensure!(
            cfg.tower.dim == cfg.hidden,
            "tower dim {} != hidden {}",
            cfg.tower.dim,
            cfg.hidden
        );
        ensure!(
            (cfg.semantic_end - cfg.semantic_begin + 1) as usize == cfg.tower.vocab,
            "semantic band is not {} wide",
            cfg.tower.vocab
        );
        Ok(cfg)
    }

    pub fn is_semantic(&self, token: u32) -> bool {
        (self.semantic_begin..=self.semantic_end).contains(&token)
    }

    pub(crate) fn conv_dim(&self) -> usize {
        2 * self.lin_k_heads * self.lin_k_dim + self.lin_v_heads * self.lin_v_dim
    }

    pub(crate) fn kv_width(&self) -> usize {
        self.n_kv * self.head_dim
    }
}

/// How Drama 3 is loaded.
#[derive(Clone, Debug)]
pub struct DramaOptions {
    /// Run only the first `n` backbone layers (tests against a truncated reference).
    pub layers: Option<usize>,
    /// Bytes of dequantized f32 experts to cache (each is 12 MB).
    pub expert_cache_bytes: usize,
    /// Keep prefill graphs after use (free on backends that share weights
    /// between graphs; a second copy of every layer on the CPU).
    pub keep_prefill: bool,
    /// How the dense matmul weights are stored in the graphs. `None` reads
    /// `RLX_WEIGHT_STORAGE`, else `Auto` (keep the checkpoint's bf16).
    pub weight_storage: Option<rlx_flow::storage::WeightStorage>,
}

impl Default for DramaOptions {
    fn default() -> Self {
        Self {
            layers: None,
            expert_cache_bytes: 4 << 30,
            keep_prefill: false,
            weight_storage: None,
        }
    }
}

/// Graph shape: a prompt bucket, or one decode row against a KV capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Prefill(usize),
    Decode(usize),
}

impl Mode {
    fn seq(self) -> usize {
        match self {
            Mode::Prefill(s) => s,
            Mode::Decode(_) => 1,
        }
    }
}

/// A loaded Drama 3 checkpoint.
pub struct DramaModel {
    pub cfg: DramaConfig,
    dir: PathBuf,
    pub(crate) store: Arc<ShardStore>,
    experts: ExpertBank,
    device: Device,
    n_layers: usize,
    keep_prefill: bool,
    pub(crate) final_norm: Vec<f32>,
    /// LM head rows `[eos, band…]` (the reference's sorted candidate order), `[n, hidden]`.
    pub(crate) head: Vec<f32>,
    layer_graphs: Vec<Option<(Mode, CompiledGraph)>>,
    tower_graphs: Vec<Option<(Mode, CompiledGraph)>>,
    matrix_dtype: DType,
    /// Decode time spent in the routed experts (host), for `RLX_FISH_VERBOSE`.
    moe_time: std::time::Duration,
}

/// Backbone caches after some rows.
pub struct BackboneState {
    pub len: usize,
    cap: usize,
    /// Full-attention layers: K/V `[cap, kv]`; linear layers: empty.
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// Linear layers: the last `kernel - 1` conv inputs `[k-1, conv_dim]`.
    conv: Vec<Vec<f32>>,
    /// Linear layers: the recurrent state `[v_heads, k_dim, v_dim]`.
    ssm: Vec<Vec<f32>>,
}

/// Tower KV caches.
pub struct TowerState {
    cap: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

/// Per-step reference hooks: main-token logits and each head's logits.
pub trait StepObserver {
    fn main_logits(&mut self, _step: usize, _logits: &[f32]) {}
    fn head_logits(&mut self, _step: usize, _head: usize, _logits: &[f32]) {}
    fn hidden(&mut self, _step: usize, _hidden: &[f32]) {}
}

impl StepObserver for () {}

/// A prompt or generated row: `[main, c0, …, c9]`.
pub type Row = Vec<u32>;

impl DramaModel {
    pub fn open(dir: &Path, device: Device, opts: DramaOptions) -> Result<Self> {
        let cfg = DramaConfig::from_dir(dir)?;
        let store = Arc::new(ShardStore::open(dir)?);
        let n_layers = opts
            .layers
            .unwrap_or(cfg.layers.len())
            .min(cfg.layers.len());
        let final_norm: Vec<f32> =
            nn::TensorSource::tensor(store.as_ref(), &format!("{LM}norm.weight"))?
                .0
                .into_iter()
                .map(|w| w + 1.0)
                .collect();
        let mut ids = vec![cfg.eos];
        ids.extend(cfg.semantic_begin..=cfg.semantic_end);
        let head_key = if store.contains("lm_head.weight") {
            "lm_head.weight"
        } else {
            "model.language_model.embed_tokens.weight"
        };
        let head = store.rows_f32(head_key, &ids)?;
        let experts = ExpertBank::new(
            store.clone(),
            LAYERS,
            cfg.hidden,
            cfg.moe_ffn,
            cfg.top_k,
            cfg.norm_topk,
            opts.expert_cache_bytes,
        );
        let n_tower = cfg.tower.shared_layers + cfg.tower.num_codebooks - 1;
        let stored = match store
            .info(&format!("{LAYERS}0.mlp.shared_expert.up_proj.weight"))?
            .dtype
        {
            safetensors::Dtype::BF16 => Some(DType::BF16),
            safetensors::Dtype::F16 => Some(DType::F16),
            _ => None,
        };
        let matrix_dtype = opts
            .weight_storage
            .or_else(rlx_flow::storage::WeightStorage::from_env)
            .unwrap_or_default()
            .resolve(stored);
        Ok(Self {
            dir: dir.to_path_buf(),
            store,
            experts,
            device,
            n_layers,
            // Metal maps one copy of each weight into every graph of the model, so
            // keeping the prefill graph alive costs activations, not weights — and
            // keeps those pages live for the decode graph to map instead of re-upload.
            keep_prefill: opts.keep_prefill || device == Device::Metal,
            final_norm,
            head,
            layer_graphs: (0..n_layers).map(|_| None).collect(),
            tower_graphs: (0..n_tower).map(|_| None).collect(),
            moe_time: std::time::Duration::ZERO,
            matrix_dtype,
            cfg,
        })
    }

    pub fn experts(&self) -> &ExpertBank {
        &self.experts
    }

    /// Candidate logit index → token id (`0` is `<|im_end|>`).
    pub fn candidate_token(&self, idx: usize) -> u32 {
        if idx == 0 {
            self.cfg.eos
        } else {
            self.cfg.semantic_begin + idx as u32 - 1
        }
    }

    fn compile(
        &self,
        mut b: Builder,
        outs: Vec<NodeId>,
        profile: rlx_flow::CompileProfile,
    ) -> Result<CompiledGraph> {
        b.g.set_outputs(outs);
        let label = format!("fish-drama3:{}", self.dir.display());
        let device = self.device;
        let mut g = rlx_runtime::with_param_sharing_scope(&label, || {
            rlx_core::flow_bridge::compile_graph_with_profile(device, b.g, &profile)
        })?;
        nn::upload(&mut g, b.params, self.store.as_ref())?;
        Ok(g)
    }

    // ── backbone layer graph ────────────────────────────────────────────

    /// Outputs: `[h + shared expert, post-norm input to the experts, router
    /// logits, A, B]` with `(A, B)` = new `(K, V)` rows (attention) or the
    /// pre-conv projection and the updated recurrent state (DeltaNet).
    fn build_layer(&self, l: usize, mode: Mode) -> Result<CompiledGraph> {
        let (b, outs) = self.layer_builder(l, mode)?;
        let profile = match mode {
            Mode::Prefill(_) => rlx_flow::CompileProfile::llama32_prefill(),
            Mode::Decode(_) => rlx_flow::CompileProfile::llama32_decode(),
        };
        self.compile(b, outs, profile)
    }

    /// [`Self::build_layer`]'s graph, uncompiled.
    pub(crate) fn layer_builder(&self, l: usize, mode: Mode) -> Result<(Builder, Vec<NodeId>)> {
        let c = &self.cfg;
        let s = mode.seq();
        let p = format!("{LAYERS}{l}.");
        let h = c.hidden;
        let mut b = Builder::with_matrix_dtype(&format!("fish_drama3_layer{l}"), self.matrix_dtype);
        b.native_gqa = nn::native_gqa(self.device);
        let x = b.input("x", &[1, s, h]);
        let xn = b.rms_norm(x, &format!("{p}input_layernorm.weight"), h, c.eps, true);
        let (mix, aux_a, aux_b) = match c.layers[l] {
            Mixer::Full => self.emit_attention(&mut b, &p, xn, mode),
            Mixer::Linear => self.emit_delta_net(&mut b, &p, xn, s)?,
        };
        let hres = b.g.add(x, mix);
        let hn = b.rms_norm(
            hres,
            &format!("{p}post_attention_layernorm.weight"),
            h,
            c.eps,
            true,
        );
        let router = b.linear(hn, &format!("{p}mlp.gate.weight"), h, c.num_experts);
        let sg = b.linear(
            hn,
            &format!("{p}mlp.shared_expert.gate_proj.weight"),
            h,
            c.shared_ffn,
        );
        let su = b.linear(
            hn,
            &format!("{p}mlp.shared_expert.up_proj.weight"),
            h,
            c.shared_ffn,
        );
        let act = {
            let a = b.g.silu(sg);
            b.g.mul(a, su)
        };
        let shared = b.linear(
            act,
            &format!("{p}mlp.shared_expert.down_proj.weight"),
            c.shared_ffn,
            h,
        );
        let gate = b.linear(hn, &format!("{p}mlp.shared_expert_gate.weight"), h, 1);
        let gate = b.g.sigmoid(gate);
        let gate = b.expand(gate, &[1, s, h]);
        let shared = b.g.mul(shared, gate);
        let out = b.g.add(hres, shared);
        Ok((b, vec![out, hn, router, aux_a, aux_b]))
    }

    /// Gated softmax attention: `q_proj` rows are `[query, gate]` per head,
    /// q/k norm, NeoX RoPE on the first `rotary_dim` channels, output ×
    /// `sigmoid(gate)`.
    fn emit_attention(
        &self,
        b: &mut Builder,
        p: &str,
        xn: NodeId,
        mode: Mode,
    ) -> (NodeId, NodeId, NodeId) {
        let c = &self.cfg;
        let s = mode.seq();
        let (nh, nkv, d) = (c.n_head, c.n_kv, c.head_dim);
        let qg = b.linear(
            xn,
            &format!("{p}self_attn.q_proj.weight"),
            c.hidden,
            nh * d * 2,
        );
        let qg = b.g.reshape_(qg, vec![1, s as i64, nh as i64, 2 * d as i64]);
        let q = b.g.narrow_(qg, 3, 0, d);
        let gate = b.g.narrow_(qg, 3, d, d);
        let q = b.g.reshape_(q, vec![1, s as i64, (nh * d) as i64]);
        let gate = b.g.reshape_(gate, vec![1, s as i64, (nh * d) as i64]);
        let k = b.linear(
            xn,
            &format!("{p}self_attn.k_proj.weight"),
            c.hidden,
            nkv * d,
        );
        let v = b.linear(
            xn,
            &format!("{p}self_attn.v_proj.weight"),
            c.hidden,
            nkv * d,
        );
        let q = b.per_head_rms(
            q,
            &format!("{p}self_attn.q_norm.weight"),
            s,
            nh,
            d,
            c.eps,
            true,
        );
        let k = b.per_head_rms(
            k,
            &format!("{p}self_attn.k_norm.weight"),
            s,
            nkv,
            d,
            c.eps,
            true,
        );
        let half = c.rotary_dim / 2;
        let cos = b.input("rope_cos", &[s, half]);
        let sin = b.input("rope_sin", &[s, half]);
        let q =
            b.g.rope_n_styled(q, cos, sin, d, c.rotary_dim, RopeStyle::NeoX);
        let k =
            b.g.rope_n_styled(k, cos, sin, d, c.rotary_dim, RopeStyle::NeoX);
        let group = nh / nkv;
        let (keys, rows, mask) = match mode {
            Mode::Prefill(_) => ((k, v), s, None),
            Mode::Decode(cap) => {
                let mask = b.input("mask", &[1, cap + 1]);
                let pk = b.input("past_k", &[1, cap, nkv * d]);
                let pv = b.input("past_v", &[1, cap, nkv * d]);
                (
                    (b.g.concat_(vec![pk, k], 1), b.g.concat_(vec![pv, v], 1)),
                    cap + 1,
                    Some(mask),
                )
            }
        };
        // Native GQA reads each of the `nkv` heads for its query heads;
        // elsewhere the keys are expanded to `nh` heads first.
        let (kh, (ka, va)) = if b.native_gqa {
            (nkv, keys)
        } else {
            (
                nh,
                (
                    b.repeat_kv(keys.0, rows, nkv, d, group),
                    b.repeat_kv(keys.1, rows, nkv, d, group),
                ),
            )
        };
        let view = |b: &mut Builder, x: NodeId, r: usize, h: usize| {
            b.g.reshape_(x, vec![1, r as i64, h as i64, d as i64])
        };
        let q4 = view(b, q, s, nh);
        let (k4, v4) = (view(b, ka, rows, kh), view(b, va, rows, kh));
        let shape = Shape::new(&[1, s, nh, d], DType::F32);
        let attn = match mask {
            None => {
                b.g.attention_kind(q4, k4, v4, nh, d, MaskKind::Causal, shape)
            }
            Some(m) => b.g.attention(q4, k4, v4, m, nh, d, shape),
        };
        let attn = b.g.reshape_(attn, vec![1, s as i64, (nh * d) as i64]);
        let sg = b.g.sigmoid(gate);
        let attn = b.g.mul(attn, sg);
        let out = b.linear(
            attn,
            &format!("{p}self_attn.o_proj.weight"),
            nh * d,
            c.hidden,
        );
        (out, k, v)
    }

    /// Gated DeltaNet over `s` rows from a carried conv window and state.
    /// `valid [1, s, v_heads]` is 1 on real rows and 0 on bucket padding: it
    /// zeroes beta and the log-decay there, so padding leaves the state alone.
    fn emit_delta_net(
        &self,
        b: &mut Builder,
        p: &str,
        xn: NodeId,
        s: usize,
    ) -> Result<(NodeId, NodeId, NodeId)> {
        let c = &self.cfg;
        let (hk, hv, dk, dv) = (c.lin_k_heads, c.lin_v_heads, c.lin_k_dim, c.lin_v_dim);
        let conv_dim = c.conv_dim();
        let kern = c.conv_kernel;
        let hist = kern - 1;
        let la = format!("{p}linear_attn.");
        let mixed = b.linear(xn, &format!("{la}in_proj_qkv.weight"), c.hidden, conv_dim);
        let conv_state = b.input("conv_state", &[1, hist, conv_dim]);
        let xp = b.g.concat_(vec![conv_state, mixed], 1);
        let (w, shape) =
            nn::TensorSource::tensor(self.store.as_ref(), &format!("{la}conv1d.weight"))?;
        ensure!(shape == [conv_dim, 1, kern], "{la}conv1d: {shape:?}");
        let mut acc = None;
        for j in 0..kern {
            let tap: Vec<f32> = (0..conv_dim).map(|ch| w[ch * kern + j]).collect();
            let tw = b.data(&format!("{la}conv.tap{j}"), tap, &[conv_dim]);
            let sl = b.g.narrow_(xp, 1, j, s);
            let m = b.g.mul(sl, tw);
            acc = Some(match acc {
                None => m,
                Some(a) => b.g.add(a, m),
            });
        }
        let y = b.g.silu(acc.expect("conv taps"));
        let q = b.g.narrow_(y, 2, 0, hk * dk);
        let k = b.g.narrow_(y, 2, hk * dk, hk * dk);
        let v = b.g.narrow_(y, 2, 2 * hk * dk, hv * dv);
        let group = hv / hk;
        let q = l2norm_heads(b, q, s, hk, dk, group);
        let k = l2norm_heads(b, k, s, hk, dk, group);
        let v = b.g.reshape_(v, vec![1, s as i64, hv as i64, dv as i64]);

        let valid = b.input("valid", &[1, s, hv]);
        let beta = b.linear(xn, &format!("{la}in_proj_b.weight"), c.hidden, hv);
        let beta = b.g.sigmoid(beta);
        let beta = b.g.mul(beta, valid);
        let a = b.linear(xn, &format!("{la}in_proj_a.weight"), c.hidden, hv);
        let dt = b.weight(&format!("{la}dt_bias"), &[hv], TensorOp::Raw);
        let a = b.g.add(a, dt);
        let sp =
            b.g.activation(Activation::Softplus, a, Shape::new(&[1, s, hv], DType::F32));
        let (a_log, _) = nn::TensorSource::tensor(self.store.as_ref(), &format!("{la}A_log"))?;
        let neg_a: Vec<f32> = a_log.iter().map(|v| -v.exp()).collect();
        let neg_a = b.data(&format!("{la}neg_A"), neg_a, &[hv]);
        let g = b.g.mul(sp, neg_a);
        let g = b.g.mul(g, valid);

        let state = b.input("ssm_state", &[1, hv, dk, dv]);
        let core = b.g.gated_delta_net_carry(
            q,
            k,
            v,
            g,
            beta,
            state,
            dk,
            Shape::new(&[1, s, hv, dv], DType::F32),
        );
        // Read the state back through a node ordered after the scan (the
        // in-place carry is not observable as a plain output everywhere).
        let tot = b.g.sum(core, vec![0, 1, 2, 3], false);
        let zero = b.g.constant(0.0, DType::F32);
        let one = b.g.constant(1.0, DType::F32);
        let dep = b.g.mul(tot, zero);
        let scale = b.g.add(one, dep);
        let state_out = b.g.mul(state, scale);

        let z = b.linear(xn, &format!("{la}in_proj_z.weight"), c.hidden, hv * dv);
        let core = b.g.reshape_(core, vec![(s * hv) as i64, dv as i64]);
        let z = b.g.reshape_(z, vec![(s * hv) as i64, dv as i64]);
        let normed = b.rms_norm(core, &format!("{la}norm.weight"), dv, c.eps, false);
        let zs = b.g.silu(z);
        let gated = b.g.mul(normed, zs);
        let gated = b.g.reshape_(gated, vec![1, s as i64, (hv * dv) as i64]);
        let out = b.linear(gated, &format!("{la}out_proj.weight"), hv * dv, c.hidden);
        Ok((out, mixed, state_out))
    }

    fn layer_graph(&mut self, l: usize, mode: Mode) -> Result<&mut CompiledGraph> {
        if self.layer_graphs[l].as_ref().map(|g| g.0) != Some(mode) {
            self.layer_graphs[l] = None;
            let g = self.build_layer(l, mode)?;
            self.layer_graphs[l] = Some((mode, g));
        }
        Ok(&mut self.layer_graphs[l].as_mut().expect("layer graph").1)
    }

    pub(crate) fn rope_backbone(&self, positions: &[usize]) -> (Vec<f32>, Vec<f32>) {
        nn::rope_tables(positions, self.cfg.rotary_dim, self.cfg.rope_theta, false)
    }

    /// Backbone input embeddings: the token embedding; for audio rows plus
    /// codes `1..=K` (K = `vq_backbone_codebooks`), the sum over `√(1+K)`.
    pub fn embed_rows(&self, rows: &[Row]) -> Result<Vec<f32>> {
        let h = self.cfg.hidden;
        let kq = self.cfg.tower.vq_backbone_codebooks;
        let toks: Vec<u32> = rows.iter().map(|r| r[0]).collect();
        let mut out = self
            .store
            .rows_f32(&format!("{LM}embed_tokens.weight"), &toks)?;
        let vocab = self.cfg.tower.vocab as u32;
        let scale = 1.0 / ((1 + kq) as f32).sqrt();
        for (i, r) in rows.iter().enumerate() {
            if kq == 0 || !self.cfg.is_semantic(r[0]) {
                continue;
            }
            let ids: Vec<u32> = (1..=kq).map(|cb| cb as u32 * vocab + r[1 + cb]).collect();
            let extra = self.store.rows_f32(CODEBOOK_EMBED, &ids)?;
            let slot = &mut out[i * h..(i + 1) * h];
            for e in extra.chunks(h) {
                for (s, v) in slot.iter_mut().zip(e) {
                    *s += v;
                }
            }
            for s in slot.iter_mut() {
                *s *= scale;
            }
        }
        Ok(out)
    }

    /// Final norm of each row (`[rows, hidden]`).
    fn final_norm_rows(&self, x: &[f32]) -> Vec<f32> {
        let h = self.cfg.hidden;
        let mut out = vec![0f32; x.len()];
        for (o, r) in out.chunks_mut(h).zip(x.chunks(h)) {
            let ms = r.iter().map(|v| v * v).sum::<f32>() / h as f32;
            let inv = 1.0 / (ms + self.cfg.eps).sqrt();
            for ((oo, v), w) in o.iter_mut().zip(r).zip(&self.final_norm) {
                *oo = v * inv * w;
            }
        }
        out
    }

    /// Main-token logits over `[eos, band…]` for one hidden row.
    pub fn main_logits(&self, hidden: &[f32]) -> Vec<f32> {
        self.head
            .chunks(self.cfg.hidden)
            .map(|w| w.iter().zip(hidden).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// Run the backbone over `rows` from position 0. Returns the state (sized
    /// for `cap` rows) and the final-norm hidden state of every row.
    pub fn prefill(&mut self, rows: &[Row], cap: usize) -> Result<(BackboneState, Vec<f32>)> {
        let c = self.cfg.clone();
        let t = rows.len();
        ensure!(t > 0 && cap > t, "prompt {t} rows, capacity {cap}");
        let s = t.div_ceil(PREFILL_BUCKET) * PREFILL_BUCKET;
        let h = c.hidden;
        let mut x = self.embed_rows(rows)?;
        x.resize(s * h, 0.0);
        let positions: Vec<usize> = (0..s).collect();
        let (cos, sin) = self.rope_backbone(&positions);
        let mut valid = vec![0f32; s * c.lin_v_heads];
        valid[..t * c.lin_v_heads].fill(1.0);
        let (hist, conv_dim, kvw) = (c.conv_kernel - 1, c.conv_dim(), c.kv_width());
        let ssm_len = c.lin_v_heads * c.lin_k_dim * c.lin_v_dim;
        let mut st = BackboneState {
            len: t,
            cap,
            k: vec![Vec::new(); self.n_layers],
            v: vec![Vec::new(); self.n_layers],
            conv: vec![Vec::new(); self.n_layers],
            ssm: vec![Vec::new(); self.n_layers],
        };
        for l in 0..self.n_layers {
            let kind = c.layers[l];
            let zero_conv = vec![0f32; hist * conv_dim];
            let zero_ssm = vec![0f32; ssm_len];
            let outs = {
                let g = self.layer_graph(l, Mode::Prefill(s))?;
                let mut inputs: Vec<(&str, &[f32])> = vec![("x", &x)];
                match kind {
                    Mixer::Full => {
                        inputs.push(("rope_cos", &cos));
                        inputs.push(("rope_sin", &sin));
                    }
                    Mixer::Linear => {
                        inputs.push(("conv_state", &zero_conv));
                        inputs.push(("ssm_state", &zero_ssm));
                        inputs.push(("valid", &valid));
                    }
                }
                g.run(&inputs)
            };
            if !self.keep_prefill {
                self.layer_graphs[l] = None;
            }
            match kind {
                Mixer::Full => {
                    let mut kb = vec![0f32; cap * kvw];
                    let mut vb = vec![0f32; cap * kvw];
                    kb[..t * kvw].copy_from_slice(&outs[3][..t * kvw]);
                    vb[..t * kvw].copy_from_slice(&outs[4][..t * kvw]);
                    st.k[l] = kb;
                    st.v[l] = vb;
                }
                Mixer::Linear => {
                    // The window is the last `hist` real inputs, zeros before row 0.
                    let mut win = vec![0f32; hist * conv_dim];
                    for j in 0..hist {
                        let src = t as isize - hist as isize + j as isize;
                        if src >= 0 {
                            let src = src as usize;
                            win[j * conv_dim..(j + 1) * conv_dim]
                                .copy_from_slice(&outs[3][src * conv_dim..(src + 1) * conv_dim]);
                        }
                    }
                    st.conv[l] = win;
                    st.ssm[l] = outs[4].clone();
                }
            }
            let routed =
                self.experts
                    .forward(l, &outs[1][..t * h], &outs[2][..t * c.num_experts], t)?;
            let mut next = outs[0].clone();
            for (o, r) in next[..t * h].iter_mut().zip(&routed) {
                *o += r;
            }
            next[t * h..].fill(0.0);
            x = next;
        }
        x.truncate(t * h);
        let hidden = self.final_norm_rows(&x);
        Ok((st, hidden))
    }

    /// Append one row (given as its embedding); returns its final-norm hidden state.
    pub fn step(&mut self, st: &mut BackboneState, embed: &[f32]) -> Result<Vec<f32>> {
        let c = self.cfg.clone();
        ensure!(st.len < st.cap, "backbone cache full ({} rows)", st.cap);
        let pos = st.len;
        let (cos, sin) = self.rope_backbone(&[pos]);
        let mut mask = vec![0f32; st.cap + 1];
        mask[..pos].fill(1.0);
        mask[st.cap] = 1.0;
        let valid = vec![1f32; c.lin_v_heads];
        let (hist, conv_dim, kvw) = (c.conv_kernel - 1, c.conv_dim(), c.kv_width());
        let mut x = embed.to_vec();
        for l in 0..self.n_layers {
            let kind = c.layers[l];
            let cap = st.cap;
            let outs = {
                let (kc, vc, conv, ssm) = (&st.k[l], &st.v[l], &st.conv[l], &st.ssm[l]);
                let g = self.layer_graph(l, Mode::Decode(cap))?;
                let mut inputs: Vec<(&str, &[f32])> = vec![("x", &x)];
                match kind {
                    Mixer::Full => {
                        inputs.extend([
                            ("rope_cos", &cos[..]),
                            ("rope_sin", &sin[..]),
                            ("mask", &mask[..]),
                            ("past_k", &kc[..]),
                            ("past_v", &vc[..]),
                        ]);
                    }
                    Mixer::Linear => {
                        inputs.extend([
                            ("conv_state", &conv[..]),
                            ("ssm_state", &ssm[..]),
                            ("valid", &valid[..]),
                        ]);
                    }
                }
                g.run(&inputs)
            };
            match kind {
                Mixer::Full => {
                    st.k[l][pos * kvw..(pos + 1) * kvw].copy_from_slice(&outs[3]);
                    st.v[l][pos * kvw..(pos + 1) * kvw].copy_from_slice(&outs[4]);
                }
                Mixer::Linear => {
                    let w = &mut st.conv[l];
                    w.copy_within(conv_dim.., 0);
                    w[(hist - 1) * conv_dim..].copy_from_slice(&outs[3]);
                    st.ssm[l] = outs[4].clone();
                }
            }
            let t_moe = std::time::Instant::now();
            let routed = self.experts.forward(l, &outs[1], &outs[2], 1)?;
            self.moe_time += t_moe.elapsed();
            x = outs[0].iter().zip(&routed).map(|(a, b)| a + b).collect();
        }
        st.len += 1;
        Ok(self.final_norm_rows(&x))
    }

    // ── tower ───────────────────────────────────────────────────────────

    fn tower_prefix(&self, i: usize) -> String {
        let ns = self.cfg.tower.shared_layers;
        if i < ns {
            format!("{SHARED}{i}.")
        } else {
            format!("{RESIDUAL}{}.", i - ns)
        }
    }

    /// One tower layer: `final_norm(block(eh_proj([enorm(e) ‖ hnorm(h)])))`,
    /// plus that layer's codebook head for the residual layers in decode.
    fn build_tower(&self, i: usize, mode: Mode) -> Result<CompiledGraph> {
        let (b, outs) = self.tower_builder(i, mode);
        let profile = match mode {
            Mode::Prefill(_) => rlx_flow::CompileProfile::llama32_prefill(),
            Mode::Decode(_) => rlx_flow::CompileProfile::llama32_decode(),
        };
        self.compile(b, outs, profile)
    }

    /// [`Self::build_tower`]'s graph, uncompiled.
    pub(crate) fn tower_builder(&self, i: usize, mode: Mode) -> (Builder, Vec<NodeId>) {
        let t = &self.cfg.tower;
        let s = mode.seq();
        let d = t.dim;
        let p = self.tower_prefix(i);
        let mut b = Builder::with_matrix_dtype(&format!("fish_drama3_tower{i}"), self.matrix_dtype);
        b.native_gqa = nn::native_gqa(self.device);
        let hidden = b.input("hidden", &[1, s, d]);
        let e = b.input("e", &[1, s, d]);
        let half = t.rotary_dim / 2;
        let cos = b.input("rope_cos", &[s, half]);
        let sin = b.input("rope_sin", &[s, half]);
        let en = b.rms_norm(e, &format!("{p}enorm.weight"), d, t.eps, true);
        let hn = b.rms_norm(hidden, &format!("{p}hnorm.weight"), d, t.eps, true);
        let cat = b.g.concat_(vec![en, hn], 2);
        let mixed = b.linear(cat, &format!("{p}eh_proj.weight"), 2 * d, d);
        let spec = BlockSpec {
            dim: d,
            ffn: t.ffn,
            eps: t.eps,
            attn: AttnSpec {
                n_head: t.n_head,
                n_kv: t.n_kv,
                head_dim: t.head_dim,
                qk_norm: true,
                rope_dim: t.rotary_dim,
                rope_style: RopeStyle::GptJ,
                output_gate: true,
            },
            layer_scale: false,
            zero_centered: true,
        };
        let keys = match mode {
            Mode::Prefill(_) => Keys::Own(MaskKind::Causal),
            Mode::Decode(cap) => {
                let kvw = t.n_kv * t.head_dim;
                Keys::Cached {
                    past_k: b.input("past_k", &[1, cap, kvw]),
                    past_v: b.input("past_v", &[1, cap, kvw]),
                    past: cap,
                    mask: b.input("mask", &[1, cap + 1]),
                }
            }
        };
        let o = nn::block(
            &mut b,
            &spec,
            &format!("{p}transformer_layer."),
            mixed,
            s,
            cos,
            sin,
            keys,
        );
        let out = b.rms_norm(o.x, &format!("{p}final_layernorm.weight"), d, t.eps, true);
        let mut outs = vec![out];
        let ns = t.shared_layers;
        if i >= ns && matches!(mode, Mode::Decode(_)) {
            outs.push(b.linear(
                out,
                &format!("audio_decoder_mtp.heads.{}.weight", i - ns),
                d,
                t.vocab,
            ));
        }
        outs.push(o.k);
        outs.push(o.v);
        (b, outs)
    }

    fn tower_graph(&mut self, i: usize, mode: Mode) -> Result<&mut CompiledGraph> {
        if self.tower_graphs[i].as_ref().map(|g| g.0) != Some(mode) {
            self.tower_graphs[i] = None;
            let g = self.build_tower(i, mode)?;
            self.tower_graphs[i] = Some((mode, g));
        }
        Ok(&mut self.tower_graphs[i].as_mut().expect("tower graph").1)
    }

    /// Tower RoPE: `t · base^(-2i/rot)` in f32 (not `1 / base^(…)`).
    pub(crate) fn rope_tower(&self, positions: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let t = &self.cfg.tower;
        let half = t.rotary_dim / 2;
        let base = t.rope_base as f32;
        let f: Vec<f32> = (0..half)
            .map(|i| base.powf(-((2 * i) as f32) / t.rotary_dim as f32))
            .collect();
        let mut cos = Vec::with_capacity(positions.len() * half);
        let mut sin = Vec::with_capacity(positions.len() * half);
        for &p in positions {
            for &fi in &f {
                let a = p as f32 * fi;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        (cos, sin)
    }

    /// The code embedding tower layer `i` mixes in for a row labelled with
    /// frame `label`, given the row's own frame `cur` (for the unseen codes
    /// the first shared layer adds).
    fn tower_embed(&self, i: usize, label: &[u32], cur: &[u32]) -> Result<Vec<f32>> {
        let t = &self.cfg.tower;
        let d = t.dim;
        let ns = t.shared_layers;
        let mut e = vec![0f32; d];
        if self.cfg.is_semantic(label[0]) {
            let (key, code) = if i < ns {
                (format!("{SHARED}{i}.semantic_embedding.weight"), label[1])
            } else {
                let r = i - ns;
                (
                    format!("{RESIDUAL}{r}.codebook_embedding.weight"),
                    label[1 + r],
                )
            };
            let code = code.min(t.vocab as u32 - 1);
            e = self.store.rows_f32(&key, &[code])?;
        }
        if i == 0 && self.cfg.is_semantic(cur[0]) {
            let first = 1 + t.vq_backbone_codebooks;
            let ids: Vec<u32> = (first..t.num_codebooks)
                .map(|cb| cb as u32 * t.vocab as u32 + cur[1 + cb])
                .collect();
            if !ids.is_empty() {
                for row in self.store.rows_f32(CODEBOOK_EMBED, &ids)?.chunks(d) {
                    for (s, v) in e.iter_mut().zip(row) {
                        *s += v;
                    }
                }
            }
        }
        Ok(e)
    }

    /// Teacher-force the tower over prompt rows `0..t-1` (row `p` labelled
    /// with row `p + 1`), filling its caches.
    pub fn tower_prefill(
        &mut self,
        rows: &[Row],
        hidden: &[f32],
        cap: usize,
    ) -> Result<TowerState> {
        let t = self.cfg.tower.clone();
        let d = t.dim;
        let kvw = t.n_kv * t.head_dim;
        let n = self.tower_graphs.len();
        let mut st = TowerState {
            cap,
            k: vec![vec![0f32; cap * kvw]; n],
            v: vec![vec![0f32; cap * kvw]; n],
        };
        let m = rows.len() - 1;
        if m == 0 {
            return Ok(st);
        }
        let s = m.div_ceil(PREFILL_BUCKET) * PREFILL_BUCKET;
        let positions: Vec<usize> = (0..s).collect();
        let (cos, sin) = self.rope_tower(&positions);
        let mut h = hidden[..m * d].to_vec();
        h.resize(s * d, 0.0);
        for i in 0..n {
            let mut e = vec![0f32; s * d];
            for pidx in 0..m {
                let row = self.tower_embed(i, &rows[pidx + 1], &rows[pidx])?;
                e[pidx * d..(pidx + 1) * d].copy_from_slice(&row);
            }
            let outs = {
                let g = self.tower_graph(i, Mode::Prefill(s))?;
                g.run(&[
                    ("hidden", &h),
                    ("e", &e),
                    ("rope_cos", &cos),
                    ("rope_sin", &sin),
                ])
            };
            if !self.keep_prefill {
                self.tower_graphs[i] = None;
            }
            st.k[i][..m * kvw].copy_from_slice(&outs[1][..m * kvw]);
            st.v[i][..m * kvw].copy_from_slice(&outs[2][..m * kvw]);
            h = outs[0].clone();
        }
        Ok(st)
    }

    /// One tower step at `pos`: sample the frame after `cur` from the backbone
    /// state `hidden` (final-norm row `pos`). `history` is every main token so
    /// far (for repetition-aware sampling; empty disables it).
    #[allow(clippy::too_many_arguments)]
    pub fn tower_step(
        &mut self,
        st: &mut TowerState,
        pos: usize,
        hidden: &[f32],
        cur: &[u32],
        history: &[u32],
        params: SampleParams,
        rng: &mut Rng,
        step: usize,
        obs: &mut dyn StepObserver,
    ) -> Result<Row> {
        let t = self.cfg.tower.clone();
        let d = t.dim;
        let kvw = t.n_kv * t.head_dim;
        let n = self.tower_graphs.len();
        let ns = t.shared_layers;
        let logits = self.main_logits(hidden);
        obs.main_logits(step, &logits);
        let first = self.candidate_token(sampling::sample(&logits, params, Flavor::Drama3, rng));
        let retry = self.candidate_token(sampling::sample(&logits, RAS, Flavor::Drama3, rng));
        let window = &history[history.len().saturating_sub(RAS_WINDOW)..];
        let main = if self.cfg.is_semantic(first) && window.contains(&first) {
            retry
        } else {
            first
        };
        let c0 = main
            .saturating_sub(self.cfg.semantic_begin)
            .min(t.vocab as u32 - 1);
        let mut frame = vec![0u32; 1 + t.num_codebooks];
        frame[0] = main;
        frame[1] = c0;
        let (cos, sin) = self.rope_tower(&[pos]);
        let cap = st.cap;
        let mut mask = vec![0f32; cap + 1];
        mask[..pos].fill(1.0);
        mask[cap] = 1.0;
        let mut h = hidden.to_vec();
        for i in 0..n {
            let e = self.tower_embed(i, &frame, cur)?;
            let outs = {
                let (kc, vc) = (&st.k[i], &st.v[i]);
                let g = self.tower_graph(i, Mode::Decode(cap))?;
                g.run(&[
                    ("hidden", &h),
                    ("e", &e),
                    ("rope_cos", &cos),
                    ("rope_sin", &sin),
                    ("mask", &mask),
                    ("past_k", kc),
                    ("past_v", vc),
                ])
            };
            let kv = if i >= ns { 2 } else { 1 };
            st.k[i][pos * kvw..(pos + 1) * kvw].copy_from_slice(&outs[kv]);
            st.v[i][pos * kvw..(pos + 1) * kvw].copy_from_slice(&outs[kv + 1]);
            h = outs[0].clone();
            if i >= ns {
                let r = i - ns;
                obs.head_logits(step, r, &outs[1]);
                frame[2 + r] = sampling::sample(&outs[1], params, Flavor::Drama3, rng) as u32;
            }
        }
        debug_assert_eq!(h.len(), d);
        Ok(frame)
    }

    /// Sample frames after `rows` until `<|im_end|>` or `max_new` frames.
    /// `ras: false` turns repetition-aware sampling off (parity tests).
    pub fn generate(
        &mut self,
        rows: &[Row],
        max_new: usize,
        params: SampleParams,
        rng: &mut Rng,
        ras: bool,
        obs: &mut dyn StepObserver,
    ) -> Result<Vec<Row>> {
        let t = rows.len();
        ensure!(
            t <= MAX_PROMPT_ROWS,
            "prompt has {t} rows; at most {MAX_PROMPT_ROWS}"
        );
        let cap = (t + max_new).div_ceil(DECODE_BUCKET) * DECODE_BUCKET;
        let (mut bst, hidden) = self.prefill(rows, cap)?;
        let mut tst = self.tower_prefill(rows, &hidden, cap)?;
        let d = self.cfg.hidden;
        let mut h = hidden[(t - 1) * d..].to_vec();
        let mut history: Vec<u32> = rows.iter().map(|r| r[0]).collect();
        let mut cur = rows[t - 1].clone();
        let mut out = Vec::new();
        let verbose = rlx_ir::env::flag("RLX_FISH_VERBOSE");
        let (mut t_tower, mut t_backbone) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        for step in 0..max_new {
            obs.hidden(step, &h);
            let t0 = std::time::Instant::now();
            let pos = t - 1 + step;
            let hist = if ras { &history[..] } else { &[] };
            let frame = self.tower_step(&mut tst, pos, &h, &cur, hist, params, rng, step, obs)?;
            if frame[0] == self.cfg.eos {
                break;
            }
            t_tower += t0.elapsed();
            history.push(frame[0]);
            out.push(frame.clone());
            if step + 1 < max_new {
                let t1 = std::time::Instant::now();
                let embed = self.embed_rows(std::slice::from_ref(&frame))?;
                h = self.step(&mut bst, &embed)?;
                t_backbone += t1.elapsed();
            }
            if verbose && out.len() % 16 == 0 {
                let n = out.len() as f32;
                let (hits, misses) = self.experts.cache_stats();
                eprintln!(
                    "drama3: {} frames, backbone {:.0} ms/frame (experts {:.0}), tower {:.0} ms/frame, experts {hits} hit / {misses} loaded",
                    out.len(),
                    t_backbone.as_secs_f32() * 1e3 / n,
                    self.moe_time.as_secs_f32() * 1e3 / n,
                    t_tower.as_secs_f32() * 1e3 / n
                );
            }
            cur = frame;
        }
        Ok(out)
    }

    /// Drop every compiled graph.
    pub fn release(&mut self) {
        self.layer_graphs.iter_mut().for_each(|g| *g = None);
        self.tower_graphs.iter_mut().for_each(|g| *g = None);
    }
}

/// `l2norm` of each `dk`-wide head of `x: [1, s, heads·dk]`, then each head
/// repeated `group` times (`repeat_interleave`) → `[1, s, heads·group, dk]`.
fn l2norm_heads(
    b: &mut Builder,
    x: NodeId,
    s: usize,
    heads: usize,
    dk: usize,
    group: usize,
) -> NodeId {
    let r = b.g.reshape_(x, vec![1, s as i64, heads as i64, dk as i64]);
    let sq = b.g.mul(r, r);
    let ss = b.g.sum(sq, vec![3], true);
    let eps = b.g.constant(L2_EPS, DType::F32);
    let ss = b.g.add(ss, eps);
    let inv = b.g.rsqrt(ss);
    let inv = b.expand(inv, &[1, s, heads, dk]);
    let n = b.g.mul(r, inv);
    if group == 1 {
        return n;
    }
    let n =
        b.g.reshape_(n, vec![1, s as i64, heads as i64, 1, dk as i64]);
    let n = b.expand(n, &[1, s, heads, group, dk]);
    b.g.reshape_(n, vec![1, s as i64, (heads * group) as i64, dk as i64])
}

/// Settings of a long-form Drama 3 synthesis.
#[derive(Clone, Debug)]
pub struct SynthOptions {
    pub params: SampleParams,
    pub seed: u64,
    /// Frames per chunk.
    pub max_new_tokens: usize,
    /// Generate the whole text as one chunk.
    pub single_chunk: bool,
}

impl Default for SynthOptions {
    fn default() -> Self {
        Self {
            params: SampleParams::default(),
            seed: 0,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            single_chunk: false,
        }
    }
}

/// `generate_long_drama3`: chunk `text`, generate each chunk with the first
/// and the latest earlier chunk as history, and return every chunk's codes
/// concatenated (`[num_codebooks][frames]`).
pub fn synthesize_codes(
    model: &mut DramaModel,
    enc: &dyn crate::prompt::TextEncoder,
    text: &str,
    references: &[crate::drama3_text::Turn],
    opts: &SynthOptions,
) -> Result<Vec<Vec<u32>>> {
    use crate::drama3_text::{History, Turn};
    let c = model.cfg.clone();
    crate::drama3_text::check_tokenizer(
        enc,
        c.semantic_begin,
        c.semantic_end,
        c.eos,
        c.tower.vocab,
    )?;
    let chunks = if opts.single_chunk {
        vec![text.to_string()]
    } else {
        let v = crate::drama3_text::split_text(
            text,
            crate::drama3_text::CHUNK_BYTES,
            crate::drama3_text::CHUNK_MAX_SPEAKERS,
        );
        if v.is_empty() {
            vec![text.to_string()]
        } else {
            v
        }
    };
    let nc = c.tower.num_codebooks;
    let mut rng = Rng::new(opts.seed);
    let mut history = History::default();
    let mut all = vec![Vec::new(); nc];
    for chunk in &chunks {
        let mut rows = None;
        for turns in history.candidates() {
            let segs = crate::drama3_text::prompt_segments(chunk, references, &turns)?;
            let r = crate::prompt::rows(&segs, enc, nc, c.semantic_begin)?;
            if r.len() <= MAX_PROMPT_ROWS {
                rows = Some(r);
                break;
            }
        }
        let Some(rows) = rows else {
            bail!("chunk prompt exceeds {MAX_PROMPT_ROWS} rows even without history");
        };
        let frames = model.generate(
            &rows,
            opts.max_new_tokens,
            opts.params,
            &mut rng,
            true,
            &mut (),
        )?;
        if frames.is_empty() {
            continue;
        }
        let codes: Vec<Vec<u32>> = (0..nc)
            .map(|i| frames.iter().map(|f| f[1 + i]).collect())
            .collect();
        for (dst, src) in all.iter_mut().zip(&codes) {
            dst.extend(src);
        }
        history.push(Turn {
            text: chunk.clone(),
            codes,
        });
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_drama3_shape() {
        let v: Value = serde_json::from_str(
            r#"{"model_type":"fish_omni","eos_token_id":248046,"semantic_start_token_id":248099,
            "semantic_end_token_id":252194,
            "text_config":{"hidden_size":2048,"num_hidden_layers":4,"layer_types":["linear_attention",
            "linear_attention","linear_attention","full_attention"],"num_attention_heads":16,
            "num_key_value_heads":2,"head_dim":256,"rope_parameters":{"rope_theta":10000000.0,
            "partial_rotary_factor":0.25},"rms_norm_eps":1e-6,"linear_conv_kernel_dim":4,
            "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_num_key_heads":16,
            "linear_num_value_heads":32,"num_experts":256,"num_experts_per_tok":8,
            "moe_intermediate_size":512,"shared_expert_intermediate_size":512,"vocab_size":252288},
            "audio_decoder_config":{"dim":2048,"intermediate_size":6144,"n_head":16,"n_local_heads":2,
            "head_dim":256,"norm_eps":1e-6,"rope_base":10000000.0,"rotary_percent":0.25,
            "vocab_size":4096,"num_codebooks":10,"shared_enrichment_layers":4,"vq_backbone_codebooks":1}}"#,
        )
        .unwrap();
        let c = DramaConfig::from_value(&v).unwrap();
        assert_eq!(c.layers[3], Mixer::Full);
        assert_eq!(c.rotary_dim, 64);
        assert_eq!(c.tower.rotary_dim, 64);
        assert_eq!(c.conv_dim(), 8192);
    }
}
