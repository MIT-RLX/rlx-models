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

//! High-level [`ClinicalBertRunner`] — config + safetensors → forward + pool.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rlx_core::config::BertConfig;
use rlx_core::flow_util::compile_built;
use rlx_core::validate_standard_device;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::{CompiledGraph, Device};

use crate::builder::build_clinicalbert_built;
#[cfg(feature = "mlm")]
use crate::builder::build_clinicalbert_with_mlm_built;
use crate::config::{ClinicalBertConfig, ClinicalBertVariant, validate_hf_config};
#[cfg(feature = "mlm")]
use crate::heads::MlmHead;
#[cfg(feature = "pooler")]
use crate::heads::PoolerHead;

/// Pooling strategy for sentence-level embeddings (matches HF reference).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// `[CLS]` (first token) hidden state.
    Cls,
    /// Attention-mask-weighted mean over tokens.
    Mean,
    /// No pooling — return raw `[batch, seq, hidden]` hidden states.
    None,
}

impl Pooling {
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cls" => Some(Pooling::Cls),
            "mean" | "avg" | "average" => Some(Pooling::Mean),
            "none" | "raw" => Some(Pooling::None),
            _ => None,
        }
    }
}

/// Where the Masked-Language-Model head — `dense(H→H) + GeLU + LN +
/// tied-decoder(H→V) + bias` — runs. Both modes produce numerically
/// equivalent logits (cos > 0.999999, drift from GEMM reduction order).
///
/// Measured on Bio_ClinicalBERT (seq=32, RTX 4090 + Intel x86, total ms):
///
/// | Device  | B=1  | B=8     | B=32     | Winner @ B=32 |
/// |---------|------|---------|----------|---------------|
/// | CPU+MKL | 52.9 / 55.4 | 135.8 / **124.1** | 437.0 / **399.1** | InGraph |
/// | CUDA    | 9.4 / **6.1**  | 30.2 / **28.6**  | **79.8** / 96.6  | Cpu     |
///
/// (`Cpu / InGraph`; bold = faster.) CUDA crosses over near B=8 because the
/// host sgemm of the H×V decoder matmul stops being the bottleneck once
/// CUDA's encoder runs in single-digit ms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MlmExecMode {
    /// CPU post-process via `rlx_cpu::blas::sgemm_bias` (Accelerate / MKL /
    /// OpenBLAS depending on the rlx-cpu link).
    Cpu,
    /// Head appended to the encoder's IR graph as a second output — runs on
    /// the same backend as the encoder.
    InGraph,
    /// Resolve at build time: `Cpu` for CUDA with batch > 8, `InGraph`
    /// otherwise (the table above is the source of this rule).
    #[default]
    Auto,
}

impl MlmExecMode {
    /// Resolve [`MlmExecMode::Auto`] against `(device, batch)`. Returns
    /// `Cpu` or `InGraph` — non-`Auto` inputs pass through unchanged.
    pub fn resolve(self, device: Device, batch: usize) -> MlmExecMode {
        match self {
            MlmExecMode::Cpu | MlmExecMode::InGraph => self,
            MlmExecMode::Auto => match device {
                Device::Cuda if batch > 8 => MlmExecMode::Cpu,
                _ => MlmExecMode::InGraph,
            },
        }
    }

    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" | "post" | "host" => Some(MlmExecMode::Cpu),
            "ingraph" | "in-graph" | "in_graph" | "graph" | "fold" | "folded" => {
                Some(MlmExecMode::InGraph)
            }
            "auto" | "default" => Some(MlmExecMode::Auto),
            _ => None,
        }
    }
}

/// Compiled ClinicalBERT encoder.
///
/// Each `forward_*` call must match the compiled `(batch, seq)`. Call
/// [`ClinicalBertRunner::recompile`] to retarget the graph for a new shape
/// (cached: a no-op when dims are unchanged).
pub struct ClinicalBertRunner {
    config: ClinicalBertConfig,
    weights_path: PathBuf,
    compiled: CompiledGraph,
    compiled_bs: (usize, usize),
    device: Device,
    pooling: Pooling,
    #[cfg(feature = "pooler")]
    pooler_head: Option<PoolerHead>,
    #[cfg(feature = "mlm")]
    mlm_head: Option<MlmHead>,
    /// `true` when the compiled graph embeds the MLM head as a second
    /// output (`with_mlm_in_graph()`). In that mode `forward()` caches
    /// the head's `mlm_logits [B,S,V]` output so `mlm_logits()` /
    /// `mlm_logits_into()` return it directly without a CPU matmul.
    #[cfg(feature = "mlm")]
    mlm_in_graph: bool,
    /// Last forward's cached `mlm_logits` when `mlm_in_graph == true`.
    #[cfg(feature = "mlm")]
    cached_mlm_logits: Option<Vec<f32>>,
}

impl ClinicalBertRunner {
    pub fn builder() -> ClinicalBertRunnerBuilder {
        ClinicalBertRunnerBuilder::default()
    }

    pub fn config(&self) -> &ClinicalBertConfig {
        &self.config
    }

    pub fn hidden_size(&self) -> usize {
        self.config.bert.hidden_size
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn pooling(&self) -> Pooling {
        self.pooling
    }

    pub fn compiled_shape(&self) -> (usize, usize) {
        self.compiled_bs
    }

    /// `true` when the builder was called with `.with_pooler()` and the
    /// pooler weights were found in the checkpoint.
    #[cfg(feature = "pooler")]
    pub fn has_pooler(&self) -> bool {
        self.pooler_head.is_some()
    }

    /// `true` when the builder was called with `.with_mlm()` and the MLM head
    /// weights were found in the checkpoint.
    #[cfg(feature = "mlm")]
    pub fn has_mlm(&self) -> bool {
        self.mlm_head.is_some() || self.mlm_in_graph
    }

    /// `true` when the MLM head runs in [`MlmExecMode::InGraph`] mode.
    #[cfg(feature = "mlm")]
    pub fn mlm_in_graph(&self) -> bool {
        self.mlm_in_graph
    }

    /// The resolved [`MlmExecMode`] the runner is executing in (never
    /// `Auto`), or `None` when the MLM head is disabled.
    #[cfg(feature = "mlm")]
    pub fn mlm_mode(&self) -> Option<MlmExecMode> {
        if self.mlm_in_graph {
            Some(MlmExecMode::InGraph)
        } else if self.mlm_head.is_some() {
            Some(MlmExecMode::Cpu)
        } else {
            None
        }
    }

    /// Pooler output `[batch, hidden_size]` = `tanh(W · h_cls + b)`.
    ///
    /// `hidden` must be the encoder output for the same `(batch, seq)` the
    /// runner is compiled for. Returns an error if `.with_pooler()` wasn't
    /// called at build time.
    #[cfg(feature = "pooler")]
    pub fn pooler_output(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let head = self.pooler_head.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "rlx-clinicalbert: pooler not enabled — call .with_pooler() on the builder"
            )
        })?;
        let (b, s) = self.compiled_bs;
        head.apply(hidden, b, s)
    }

    /// MLM logits `[batch, seq, vocab_size]`. In [`MlmExecMode::InGraph`]
    /// mode this returns the cached output of the last [`Self::forward`]
    /// call (no compute); in [`MlmExecMode::Cpu`] mode it runs the CPU
    /// post-process head on `hidden`.
    #[cfg(feature = "mlm")]
    pub fn mlm_logits(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        if self.mlm_in_graph {
            return self.cached_mlm_logits.clone().ok_or_else(|| {
                anyhow::anyhow!(
                    "rlx-clinicalbert: call forward() first to populate the in-graph MLM logits"
                )
            });
        }
        let head = self
            .mlm_head
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("rlx-clinicalbert: MLM head not enabled — call .with_mlm() or .with_mlm_in_graph() on the builder"))?;
        let (b, s) = self.compiled_bs;
        head.apply(hidden, b, s)
    }

    /// MLM logits into a caller-provided buffer (zero-allocation hot path).
    /// `logits.len()` must equal `batch * seq * vocab_size`. Use
    /// [`Self::allocate_mlm_logits`] to size it.
    #[cfg(feature = "mlm")]
    pub fn mlm_logits_into(&self, hidden: &[f32], logits: &mut [f32]) -> Result<()> {
        if self.mlm_in_graph {
            let src = self.cached_mlm_logits.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "rlx-clinicalbert: call forward() first to populate the in-graph MLM logits"
                )
            })?;
            if logits.len() != src.len() {
                bail!(
                    "rlx-clinicalbert: mlm_logits_into expected buffer of {} floats, got {}",
                    src.len(),
                    logits.len()
                );
            }
            logits.copy_from_slice(src);
            return Ok(());
        }
        let head = self
            .mlm_head
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("rlx-clinicalbert: MLM head not enabled — call .with_mlm() or .with_mlm_in_graph() on the builder"))?;
        let (b, s) = self.compiled_bs;
        head.apply_into(hidden, b, s, logits)
    }

    /// Allocate a buffer sized for [`Self::mlm_logits_into`].
    #[cfg(feature = "mlm")]
    pub fn allocate_mlm_logits(&self) -> Result<Vec<f32>> {
        if self.mlm_in_graph {
            let (b, s) = self.compiled_bs;
            return Ok(vec![0f32; b * s * self.config.bert.vocab_size]);
        }
        let head = self
            .mlm_head
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("rlx-clinicalbert: MLM head not enabled"))?;
        let (b, s) = self.compiled_bs;
        Ok(head.allocate_logits_buffer(b, s))
    }

    /// Retarget the compiled graph for a new `(batch, seq)`.
    pub fn recompile(&mut self, batch: usize, seq: usize) -> Result<()> {
        if self.compiled_bs == (batch, seq) {
            return Ok(());
        }
        let mut wm = if self.weights_path.is_dir() {
            WeightMap::from_resolved_path(&self.weights_path)
        } else {
            WeightMap::from_file(self.weights_path.to_str().ok_or_else(|| {
                anyhow::anyhow!(
                    "rlx-clinicalbert: non-UTF8 weights path {:?}",
                    self.weights_path
                )
            })?)
        }?;
        let built = build_clinicalbert_built(&self.config.bert, &mut wm, batch, seq)?;
        self.compiled = compile_built(built, self.device)?;
        self.compiled_bs = (batch, seq);
        Ok(())
    }

    /// Raw forward — returns flat `[batch * seq * hidden]` F32 hidden states.
    /// All four inputs are flattened `[batch, seq]` F32 buffers.
    ///
    /// In [`MlmExecMode::InGraph`] mode `mlm_logits` is also computed and
    /// cached for [`Self::mlm_logits`] / [`Self::mlm_logits_into`].
    pub fn forward(
        &mut self,
        input_ids: &[f32],
        attention_mask: &[f32],
        token_type_ids: &[f32],
        position_ids: &[f32],
    ) -> Result<Vec<f32>> {
        let (b, s) = self.compiled_bs;
        let expected = b * s;
        if input_ids.len() != expected
            || attention_mask.len() != expected
            || token_type_ids.len() != expected
            || position_ids.len() != expected
        {
            bail!(
                "rlx-clinicalbert: forward expects each input of length {expected} \
                 (batch={b}, seq={s}); got {}, {}, {}, {}",
                input_ids.len(),
                attention_mask.len(),
                token_type_ids.len(),
                position_ids.len()
            );
        }
        let outputs = self.compiled.run(&[
            ("input_ids", input_ids),
            ("attention_mask", attention_mask),
            ("token_type_ids", token_type_ids),
            ("position_ids", position_ids),
        ]);
        if std::env::var("RLX_CLINICALBERT_DEBUG").is_ok() {
            let sizes: Vec<usize> = outputs.iter().map(|o| o.len()).collect();
            eprintln!("[rlx-clinicalbert] forward outputs: {sizes:?}");
        }
        // The encoder graph declares `hidden_states` as the FIRST output.
        // When `.with_mlm_in_graph()` is active, `mlm_logits` is the SECOND
        // output; cache it for [`Self::mlm_logits`].
        #[cfg(feature = "mlm")]
        if self.mlm_in_graph {
            if outputs.len() >= 2 {
                self.cached_mlm_logits = Some(outputs[1].clone());
            } else {
                bail!(
                    "rlx-clinicalbert: with_mlm_in_graph but compiled graph returned {} outputs",
                    outputs.len()
                );
            }
        }
        outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("rlx-clinicalbert: compiled graph returned no outputs"))
    }

    /// Forward + pool using the runner's configured pooling.
    ///
    /// Returns `[batch, hidden]` flattened. With [`Pooling::None`] this matches
    /// [`Self::forward`] (returns `[batch, seq, hidden]`).
    pub fn embed(
        &mut self,
        input_ids: &[f32],
        attention_mask: &[f32],
        token_type_ids: &[f32],
        position_ids: &[f32],
    ) -> Result<Vec<f32>> {
        let hidden = self.forward(input_ids, attention_mask, token_type_ids, position_ids)?;
        let (b, s) = self.compiled_bs;
        let h = self.hidden_size();
        Ok(match self.pooling {
            Pooling::None => hidden,
            Pooling::Cls => pool_cls(&hidden, b, s, h),
            Pooling::Mean => pool_mean(&hidden, attention_mask, b, s, h),
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct ClinicalBertRunnerBuilder {
    weights: Option<PathBuf>,
    config: Option<ClinicalBertConfig>,
    config_path: Option<PathBuf>,
    variant: Option<ClinicalBertVariant>,
    device: Option<Device>,
    batch: Option<usize>,
    seq: Option<usize>,
    pooling: Option<Pooling>,
    #[cfg(feature = "pooler")]
    enable_pooler: bool,
    #[cfg(feature = "mlm")]
    enable_mlm: bool,
    #[cfg(feature = "mlm")]
    enable_mlm_in_graph: bool,
    #[cfg(feature = "mlm")]
    mlm_mode: Option<MlmExecMode>,
}

impl ClinicalBertRunnerBuilder {
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        self.weights = Some(path.into());
        self
    }

    pub fn config(mut self, cfg: BertConfig) -> Self {
        self.config = Some(ClinicalBertConfig::new(cfg));
        self
    }

    pub fn config_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config_path = Some(path.into());
        self
    }

    pub fn variant(mut self, v: ClinicalBertVariant) -> Self {
        self.variant = Some(v);
        self
    }

    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    pub fn batch(mut self, b: usize) -> Self {
        self.batch = Some(b);
        self
    }

    pub fn max_seq(mut self, s: usize) -> Self {
        self.seq = Some(s);
        self
    }

    pub fn pooling(mut self, p: Pooling) -> Self {
        self.pooling = Some(p);
        self
    }

    /// Load the pre-trained pooler head (`bert.pooler.dense.*`) so
    /// [`ClinicalBertRunner::pooler_output`] becomes available.
    #[cfg(feature = "pooler")]
    pub fn with_pooler(mut self) -> Self {
        self.enable_pooler = true;
        self
    }

    /// Enable the MLM head in [`MlmExecMode::Cpu`] mode. Shortcut for
    /// `.mlm_mode(MlmExecMode::Cpu)`.
    #[cfg(feature = "mlm")]
    pub fn with_mlm(mut self) -> Self {
        self.enable_mlm = true;
        self
    }

    /// Enable the MLM head in [`MlmExecMode::InGraph`] mode. Shortcut for
    /// `.mlm_mode(MlmExecMode::InGraph)`. Mutually exclusive with
    /// [`Self::with_mlm`].
    #[cfg(feature = "mlm")]
    pub fn with_mlm_in_graph(mut self) -> Self {
        self.enable_mlm_in_graph = true;
        self
    }

    /// Enable the MLM head and pick where it runs. The explicit form of
    /// [`Self::with_mlm`] / [`Self::with_mlm_in_graph`]; pass
    /// [`MlmExecMode::Auto`] to let the runner pick at [`Self::build`] time
    /// based on the configured `(device, batch)`. Overrides any prior
    /// shortcut call.
    #[cfg(feature = "mlm")]
    pub fn mlm_mode(mut self, mode: MlmExecMode) -> Self {
        self.mlm_mode = Some(mode);
        self.enable_mlm = false;
        self.enable_mlm_in_graph = false;
        self
    }

    pub fn build(self) -> Result<ClinicalBertRunner> {
        let weights = self
            .weights
            .clone()
            .ok_or_else(|| anyhow::anyhow!("rlx-clinicalbert: weights path required"))?;
        let device = self.device.unwrap_or(Device::Cpu);
        validate_standard_device("clinicalbert", device)?;

        let mut config = if let Some(cfg) = self.config {
            cfg
        } else if let Some(variant) = self.variant {
            ClinicalBertConfig::new(variant.preset()).with_variant(variant)
        } else {
            let cfg_path = self
                .config_path
                .clone()
                .unwrap_or_else(|| ClinicalBertConfig::config_json_path(&weights));
            if cfg_path.is_file() {
                validate_hf_config(cfg_path.parent().unwrap_or(Path::new(".")))?;
                ClinicalBertConfig::from_file(&cfg_path)?
            } else {
                bail!(
                    "rlx-clinicalbert: no config supplied — call `.config(..)`, \
                     `.config_path(..)`, or `.variant(..)`, or place `config.json` next \
                     to {weights:?}"
                );
            }
        };

        if config.variant.is_none() {
            config.variant = self.variant;
        }

        let batch = self.batch.unwrap_or(1);
        let seq = self
            .seq
            .unwrap_or_else(|| config.bert.max_position_embeddings.min(512));

        let weights_str = weights.to_str().ok_or_else(|| {
            anyhow::anyhow!("rlx-clinicalbert: non-UTF8 weights path {weights:?}")
        })?;
        let mut wm = if weights.is_dir() {
            WeightMap::from_resolved_path(&weights)
        } else {
            WeightMap::from_file(weights_str)
        }
        .with_context(|| format!("rlx-clinicalbert: loading {weights_str}"))?;

        // Disallow combining the two MLM modes — they consume the same
        // checkpoint tensors but route the head differently.
        #[cfg(feature = "mlm")]
        if self.enable_mlm && self.enable_mlm_in_graph {
            bail!("rlx-clinicalbert: .with_mlm() and .with_mlm_in_graph() are mutually exclusive");
        }

        // Resolve the MLM execution mode. Three input paths:
        //   1. `.mlm_mode(MlmExecMode::*)`           — explicit, takes precedence.
        //   2. `.with_mlm()` / `.with_mlm_in_graph()` — legacy shortcuts.
        //   3. neither                                — MLM head disabled.
        // `Auto` is resolved against the configured (device, batch) here so
        // the downstream code only sees the two concrete modes.
        #[cfg(feature = "mlm")]
        let resolved_mlm: Option<MlmExecMode> = match self.mlm_mode {
            Some(MlmExecMode::Auto) => Some(MlmExecMode::Auto.resolve(device, batch)),
            Some(m) => Some(m),
            None => {
                if self.enable_mlm {
                    Some(MlmExecMode::Cpu)
                } else if self.enable_mlm_in_graph {
                    Some(MlmExecMode::InGraph)
                } else {
                    None
                }
            }
        };

        // Load heads BEFORE the encoder build — MLM head needs to clone the
        // embedding matrix while it's still in the WeightMap.
        #[cfg(feature = "mlm")]
        let mlm_head: Option<MlmHead> = if resolved_mlm == Some(MlmExecMode::Cpu) {
            Some(MlmHead::load(&config.bert, &mut wm)?)
        } else {
            None
        };
        #[cfg(feature = "pooler")]
        let pooler_head: Option<PoolerHead> = if self.enable_pooler {
            Some(PoolerHead::load(&config.bert, &mut wm)?)
        } else {
            None
        };

        // Pick the right encoder builder: with the head folded into the
        // graph we emit a two-output graph (hidden_states + mlm_logits) and
        // compile both in one pipeline. Otherwise the original encoder-only
        // builder — the head, if any, runs as a CPU post-process.
        #[cfg(feature = "mlm")]
        let built = if resolved_mlm == Some(MlmExecMode::InGraph) {
            build_clinicalbert_with_mlm_built(&config.bert, &mut wm, batch, seq)?
        } else {
            build_clinicalbert_built(&config.bert, &mut wm, batch, seq)?
        };
        #[cfg(not(feature = "mlm"))]
        let built = build_clinicalbert_built(&config.bert, &mut wm, batch, seq)?;
        let compiled = compile_built(built, device)?;

        Ok(ClinicalBertRunner {
            config,
            weights_path: weights,
            compiled,
            compiled_bs: (batch, seq),
            device,
            pooling: self.pooling.unwrap_or(Pooling::Cls),
            #[cfg(feature = "pooler")]
            pooler_head,
            #[cfg(feature = "mlm")]
            mlm_head,
            #[cfg(feature = "mlm")]
            mlm_in_graph: resolved_mlm == Some(MlmExecMode::InGraph),
            #[cfg(feature = "mlm")]
            cached_mlm_logits: None,
        })
    }
}

fn pool_cls(hidden: &[f32], batch: usize, seq: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; batch * h];
    for bi in 0..batch {
        let src = bi * seq * h;
        out[bi * h..(bi + 1) * h].copy_from_slice(&hidden[src..src + h]);
    }
    out
}

fn pool_mean(
    hidden: &[f32],
    attention_mask: &[f32],
    batch: usize,
    seq: usize,
    h: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; batch * h];
    for bi in 0..batch {
        let mut count = 0.0f32;
        for si in 0..seq {
            let m = attention_mask[bi * seq + si];
            if m > 0.0 {
                count += 1.0;
                let off = (bi * seq + si) * h;
                let dst = bi * h;
                for j in 0..h {
                    out[dst + j] += hidden[off + j];
                }
            }
        }
        let inv = 1.0 / count.max(1.0);
        for j in 0..h {
            out[bi * h + j] *= inv;
        }
    }
    out
}
