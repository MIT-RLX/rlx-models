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

//! [`SalaRunner`]: the step graph plus the host-side state it threads — the
//! attention layers' K/V cache and the Lightning layers' recurrent state.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use rlx_core::SafetensorsCheckpoint;
use rlx_ir::DType;
use rlx_qwen3::sampling::{SampleOpts, sample_token_at};
use rlx_runtime::{CompiledGraph, Device};

use crate::config::SalaConfig;
use crate::graph::{
    DecayFeeds, IN_ATTN_BIAS, IN_DECAY_INTRA, IN_DECAY_K, IN_DECAY_Q, IN_DECAY_S, IN_EMBEDS,
    IN_LAST, IN_ROPE_COS, IN_ROPE_SIN, StepLayout, StepShape, attention_bias, build_step,
    past_k_input, past_v_input, state_input,
};
use crate::sparse::{SparseFeeds, SparseMode};

#[derive(Default)]
pub struct SalaRunnerBuilder {
    dir: Option<PathBuf>,
    device: Option<Device>,
    chunk: Option<usize>,
    max_seq: Option<usize>,
    layers: Option<usize>,
    f32_weights: bool,
    config: Option<SalaConfig>,
}

impl SalaRunnerBuilder {
    /// HF checkpoint directory (`config.json`, sharded safetensors,
    /// `tokenizer.json`).
    pub fn model_dir(mut self, p: impl Into<PathBuf>) -> Self {
        self.dir = Some(p.into());
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    /// Tokens per graph run (default 16). Prompts go through in chunks of
    /// this size. A decode step uses one row of it, except on Metal, where a
    /// second one-row graph reads the same weight copy.
    pub fn chunk(mut self, n: usize) -> Self {
        self.chunk = Some(n);
        self
    }
    /// Longest prompt + answer (default 2048). Capped at the checkpoint's
    /// `dense_len`: past it InfLLM-v2 switches to block-sparse attention,
    /// which this crate does not implement.
    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = Some(n);
        self
    }
    /// Run only the first `n` layers (layer-subset parity tests).
    pub fn layers(mut self, n: usize) -> Self {
        self.layers = Some(n);
        self
    }
    /// Use `cfg` instead of reading `config.json` (a trunk embedded in a
    /// larger checkpoint, see [`SalaConfig::from_llama_value`]).
    pub fn config(mut self, cfg: SalaConfig) -> Self {
        self.config = Some(cfg);
        self
    }
    /// Widen the matrices to f32 instead of keeping the checkpoint's 16-bit
    /// type (bf16 or f16, whichever it stores). Doubles the footprint; the
    /// products are the same.
    pub fn f32_weights(mut self, on: bool) -> Self {
        self.f32_weights = on;
        self
    }

    pub fn build(self) -> Result<SalaRunner> {
        let dir = self
            .dir
            .ok_or_else(|| anyhow!("SalaRunner: model_dir is required"))?;
        let mut cfg = match self.config {
            Some(c) => c,
            None => SalaConfig::from_dir(&dir)?,
        };
        let trained_layers = cfg.num_layers();
        if let Some(n) = self.layers {
            ensure!(
                n >= 1 && n <= trained_layers,
                "minicpm-sala: layers={n} out of 1..={trained_layers}"
            );
            cfg.truncate(n);
        }
        let device = self.device.unwrap_or(Device::Cpu);
        let chunk = self.chunk.unwrap_or(16).max(1);
        let mut cap = self.max_seq.unwrap_or(2048);
        // Past `dense_len` keys the attention layers switch to InfLLM-v2's
        // block-sparse attention; the graph then carries the block choice,
        // over whole 64-key blocks.
        let sparse = cfg.dense_len.is_some_and(|d| cap > d);
        if sparse {
            let sp = cfg.sparse.ok_or_else(|| {
                anyhow!(
                    "minicpm-sala: max_seq {cap} is past dense_len, but sparse_config has no \
                     geometry this crate implements"
                )
            })?;
            cap = cap.div_ceil(sp.block_size) * sp.block_size;
            ensure!(
                sp.blocks(cap) >= 1,
                "minicpm-sala: max_seq {cap} too small for sparse attention"
            );
        }
        let ckpt = SafetensorsCheckpoint::open(&dir)
            .with_context(|| format!("minicpm-sala: open safetensors in {dir:?}"))?;
        // Keep the matrices in the checkpoint's own 16-bit type: rounding an
        // f16 checkpoint to bf16 would change the weights.
        let stored = ckpt.stored_dtype(&cfg.weight("model.layers.0.mlp.down_proj.weight"));
        let dtype = if self.f32_weights {
            DType::F32
        } else if stored == Some(safetensors::Dtype::F16) {
            DType::F16
        } else {
            DType::BF16
        };
        let shape = StepShape { chunk, cap, sparse };
        let label = format!("minicpm-sala:{}", dir.display());
        let compile = |shape: StepShape, profile: rlx_flow::CompileProfile| {
            let (graph, params, layout) = build_step(&cfg, shape, dtype);
            let compiled = rlx_runtime::with_param_sharing_scope(&label, || {
                rlx_core::flow_bridge::compile_graph_with_profile(device, graph, &profile)
            })?;
            anyhow::Ok((compiled, params, layout))
        };
        let t = web_time::Instant::now();
        let (mut compiled, params, layout) =
            compile(shape, rlx_flow::CompileProfile::llama32_prefill())?;
        let t_compile = t.elapsed();
        let t = web_time::Instant::now();
        rlx_fish::nn::upload(&mut compiled, params, &ckpt)?;
        let t_weights = t.elapsed();
        // A chunk-row graph spends a decode step on `chunk` rows, and on Metal
        // its matmuls past four rows widen the bf16 weights first. A one-row
        // graph in the same sharing scope maps the weights already placed, so
        // it costs no second copy; where the backend does not share, it would,
        // and decode keeps the chunk graph.
        let decode = if chunk > 1 && matches!(device, Device::Metal) {
            let one = StepShape {
                chunk: 1,
                cap,
                sparse,
            };
            let (mut g, params, _) = compile(one, rlx_flow::CompileProfile::llama32_decode())?;
            // Small tensors (norm weights) stay in each graph's own
            // arena; what matters is that the matrices are shared.
            let elems = |p: &rlx_fish::nn::ParamSpec| p.shape.iter().product::<usize>();
            let total: usize = params.iter().map(elems).sum();
            let own: usize = params
                .iter()
                .filter(|p| !g.param_already_set(&p.name))
                .map(elems)
                .sum();
            if own * 100 < total {
                rlx_fish::nn::upload(&mut g, params, &ckpt)?;
                Some((g, one))
            } else {
                None
            }
        } else {
            None
        };
        eprintln!(
            "[minicpm-sala] {} layers, chunk {chunk}{}, cap {cap}: compile {:.1}s, weights {:.1}s",
            cfg.num_layers(),
            if decode.is_some() {
                " + 1-row decode"
            } else {
                ""
            },
            t_compile.as_secs_f64(),
            t_weights.as_secs_f64()
        );
        let kvw = cfg.num_key_value_heads * cfg.attn_head_dim();
        let sw = cfg.num_attention_heads * cfg.head_dim * cfg.head_dim;
        let tokenizer = dir.join("tokenizer.json");
        Ok(SalaRunner {
            log_decay: cfg.lightning_log_decay(),
            k_cache: vec![vec![0.0; cap * kvw]; layout.attn_layers.len()],
            v_cache: vec![vec![0.0; cap * kvw]; layout.attn_layers.len()],
            states: vec![vec![0.0; sw]; layout.lightning_layers.len()],
            cfg,
            ckpt,
            graph: compiled,
            shape,
            decode,
            layout,
            past: 0,
            prompt_len: None,
            last_selection: Vec::new(),
            tokenizer: tokenizer.is_file().then_some(tokenizer),
        })
    }
}

pub struct SalaRunner {
    cfg: SalaConfig,
    ckpt: SafetensorsCheckpoint,
    graph: CompiledGraph,
    shape: StepShape,
    /// One-row graph for decode steps, sharing `graph`'s weights.
    decode: Option<(CompiledGraph, StepShape)>,
    layout: StepLayout,
    log_decay: Vec<f32>,
    /// Tokens already folded into the caches.
    past: usize,
    /// While a prompt fed from position 0 is going through: its length (the
    /// sparse layers' prefill rule depends on it).
    prompt_len: Option<usize>,
    /// The first attention layer's block choice in the last sparse step,
    /// `[nkv, chunk, blocks]` (1 = kept).
    last_selection: Vec<f32>,
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    states: Vec<Vec<f32>>,
    tokenizer: Option<PathBuf>,
}

impl SalaRunner {
    pub fn builder() -> SalaRunnerBuilder {
        SalaRunnerBuilder::default()
    }

    pub fn config(&self) -> &SalaConfig {
        &self.cfg
    }

    /// Tokens in the context so far.
    pub fn position(&self) -> usize {
        self.past
    }

    /// Forget the context.
    pub fn reset(&mut self) {
        self.past = 0;
        self.k_cache
            .iter_mut()
            .chain(self.v_cache.iter_mut())
            .for_each(|c| c.fill(0.0));
        self.states.iter_mut().for_each(|s| s.fill(0.0));
    }

    /// [`Self::feed`], calling `on_step(start, n, selection)` after every
    /// graph run with the run's first position, its real rows and
    /// [`Self::last_selection`].
    pub fn feed_observed(
        &mut self,
        ids: &[u32],
        mut on_step: impl FnMut(usize, usize, &[f32]),
    ) -> Result<Vec<f32>> {
        ensure!(
            !ids.is_empty(),
            "minicpm-sala: feed needs at least one token"
        );
        self.prompt_len = (self.past == 0).then_some(ids.len());
        let mut logits = Vec::new();
        for chunk in ids.chunks(self.shape.chunk) {
            let start = self.past;
            logits = self.step(chunk)?;
            on_step(start, chunk.len(), &self.last_selection);
        }
        self.prompt_len = None;
        Ok(logits)
    }

    /// Append `ids` to the context and return the logits after the last one.
    pub fn feed(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        ensure!(
            !ids.is_empty(),
            "minicpm-sala: feed needs at least one token"
        );
        self.prompt_len = (self.past == 0).then_some(ids.len());
        let mut logits = Vec::new();
        for chunk in ids.chunks(self.shape.chunk) {
            logits = self.step(chunk)?;
        }
        self.prompt_len = None;
        Ok(logits)
    }

    /// Append rows given as input embeddings (`[n, hidden]`, already scaled
    /// as the model expects — e.g. text rows from [`Self::embed`] with image
    /// features spliced in) and return the logits after the last one.
    pub fn feed_embeds(&mut self, embeds: &[f32]) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size;
        ensure!(
            !embeds.is_empty() && embeds.len().is_multiple_of(h),
            "minicpm-sala: feed_embeds needs whole rows of {h}"
        );
        self.prompt_len = (self.past == 0).then_some(embeds.len() / h);
        let mut logits = Vec::new();
        for chunk in embeds.chunks(self.shape.chunk * h) {
            logits = self.step_rows(chunk.to_vec())?;
        }
        self.prompt_len = None;
        Ok(logits)
    }

    /// Input embeddings of `ids` (`[n, hidden]`, × `scale_emb`).
    pub fn embed(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let rows =
            self.ckpt
                .load_tensor_rows_f32(&cfg.weight("model.embed_tokens.weight"), ids, h)?;
        let mut out = Vec::with_capacity(ids.len() * h);
        for r in &rows {
            out.extend(r.iter().map(|v| v * cfg.scale_emb));
        }
        Ok(out)
    }

    /// The first attention layer's block choice in the last step,
    /// `[kv_heads, chunk, blocks]` (1 = the query attends to the block); empty
    /// unless `max_seq` is past `dense_len`.
    pub fn last_selection(&self) -> &[f32] {
        &self.last_selection
    }

    /// Prefill `embeds` (see [`Self::feed_embeds`]) and generate like
    /// [`Self::generate`].
    pub fn generate_from_embeds(
        &mut self,
        embeds: &[f32],
        max_new: usize,
        opts: SampleOpts,
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let logits = self.feed_embeds(embeds)?;
        self.decode_from(logits, max_new, opts, on_token)
    }

    fn step(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let rows = self.embed(ids)?;
        self.step_rows(rows)
    }

    /// One graph run over `rows` (`[n, hidden]`, n ≤ chunk).
    fn step_rows(&mut self, rows: Vec<f32>) -> Result<Vec<f32>> {
        let n = rows.len() / self.cfg.hidden_size;
        let (
            graph,
            StepShape {
                chunk: t,
                cap,
                sparse,
            },
        ) = match &mut self.decode {
            Some((g, shape)) if n == 1 && self.prompt_len.is_none() => (g, *shape),
            _ => (&mut self.graph, self.shape),
        };
        if self.past + n > cap {
            bail!(
                "minicpm-sala: context would reach {} tokens, max_seq is {cap}",
                self.past + n
            );
        }
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let mut embeds = rows;
        embeds.resize(t * h, 0.0);
        let positions: Vec<usize> = (self.past..self.past + t).collect();
        let (cos, sin) = cfg.rope_tables(&positions);
        let last = [(n - 1) as f32];

        let bias = (!self.layout.attn_layers.is_empty())
            .then(|| attention_bias(cfg.num_attention_heads, t, cap, self.past, n));
        let decay = (!self.layout.lightning_layers.is_empty())
            .then(|| DecayFeeds::new(&self.log_decay, t, n));
        let sparse_feeds = sparse.then(|| {
            let sp = cfg.sparse.expect("checked at build");
            let dense = cfg.dense_len.expect("checked at build");
            // The reference picks per call by the key count: the prompt's
            // length in prefill, the keys so far in decode.
            let mode = match self.prompt_len {
                Some(n_prompt) if n_prompt >= dense => SparseMode::Prefill { n_prompt },
                Some(_) => SparseMode::Dense,
                None if self.past + 1 >= dense => SparseMode::Decode,
                None => SparseMode::Dense,
            };
            SparseFeeds::new(&sp, t, cap, self.past, n, mode)
        });

        let mut inputs: Vec<(String, &[f32])> = vec![
            (IN_EMBEDS.into(), &embeds),
            (IN_ROPE_COS.into(), &cos),
            (IN_ROPE_SIN.into(), &sin),
            (IN_LAST.into(), &last),
        ];
        if let Some(b) = &bias {
            inputs.push((IN_ATTN_BIAS.into(), b));
        }
        if let Some(d) = &decay {
            inputs.push((IN_DECAY_INTRA.into(), &d.intra));
            inputs.push((IN_DECAY_Q.into(), &d.q));
            inputs.push((IN_DECAY_K.into(), &d.k));
            inputs.push((IN_DECAY_S.into(), &d.s));
        }
        for (i, &l) in self.layout.attn_layers.iter().enumerate() {
            inputs.push((past_k_input(l), &self.k_cache[i]));
            inputs.push((past_v_input(l), &self.v_cache[i]));
        }
        for (i, &l) in self.layout.lightning_layers.iter().enumerate() {
            inputs.push((state_input(l), &self.states[i]));
        }
        if let Some(f) = &sparse_feeds {
            for (k, v) in f.inputs() {
                inputs.push((k.into(), v));
            }
        }
        let feeds: Vec<(&str, &[f32])> = inputs.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let outs = graph.run(&feeds);
        let n_attn = self.layout.attn_layers.len();
        ensure!(
            outs.len() == 1 + 2 * n_attn + self.layout.lightning_layers.len() + usize::from(sparse),
            "minicpm-sala: step graph returned {} outputs",
            outs.len()
        );
        let mut it = outs.into_iter();
        let logits = it.next().expect("checked above");

        let kvw = cfg.num_key_value_heads * cfg.attn_head_dim();
        for i in 0..n_attn {
            let k = it.next().expect("checked above");
            let v = it.next().expect("checked above");
            let at = self.past * kvw;
            self.k_cache[i][at..at + n * kvw].copy_from_slice(&k[..n * kvw]);
            self.v_cache[i][at..at + n * kvw].copy_from_slice(&v[..n * kvw]);
        }
        for s in self.states.iter_mut() {
            *s = it.next().expect("checked above");
        }
        if sparse {
            self.last_selection = it.next().expect("checked above");
        }
        self.past += n;
        Ok(logits)
    }

    /// Feed `prompt` and generate up to `max_new` tokens; stops after an EOS
    /// (not returned) or when `on_token` returns `false`.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        opts: SampleOpts,
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let logits = self.feed(prompt)?;
        self.decode_from(logits, max_new, opts, on_token)
    }

    fn decode_from(
        &mut self,
        mut logits: Vec<f32>,
        max_new: usize,
        opts: SampleOpts,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        for step in 0..max_new {
            let tok = sample_token_at(&logits, opts, step as u64) as u32;
            if self.cfg.eos_token_ids.contains(&tok) {
                break;
            }
            out.push(tok);
            if !on_token(tok) || step + 1 == max_new {
                break;
            }
            logits = self.feed(&[tok])?;
        }
        Ok(out)
    }

    pub fn tokenizer(&self) -> Option<&Path> {
        self.tokenizer.as_deref()
    }
}
