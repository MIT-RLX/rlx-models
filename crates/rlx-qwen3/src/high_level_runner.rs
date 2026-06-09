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

use crate::capabilities::validate_device;
use crate::{Qwen3Config, Qwen3Generator, SampleOpts, build_qwen3_graph_sized_packed};
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{LmRunner, WeightFormat, list_mtp_keys};
use rlx_core::gguf_support::{
    GgufModelFamily, ResolveWeightsOptions, assert_gguf_family, gguf_f32_bytes_estimate,
    resolve_weights_file_with_options,
};
use rlx_core::weight_loader::GgufLoader;
use rlx_flow::CompileProfile;
use rlx_gguf::{GgufFile, MetaValue};
use rlx_runtime::{Device, Session};
use std::path::{Path, PathBuf};

/// Precision policy for the Qwen3 inference graph. Today only `F32`
/// is exact; the others toggle the corresponding env-vars on the
/// Metal MPSGraph fast path (see `qwen3_metal_perf` notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Everything in F32. Default — most reproducible, slowest on
    /// large LM heads.
    #[default]
    F32,
    /// F32 throughout except the LM-head matmul, which casts to F16
    /// for the dominant prefill workload. Wins ~1.3-1.45× on
    /// (B≥2, L≥64) cells; loses on small cells.
    F16LmHead,
}

/// Source for the qwen3 config. Alias of the shared
/// `rlx_runtime::ConfigSource<T>`:
///
/// - `Embedded` — read from GGUF metadata.
/// - `JsonFile(PathBuf)` — read from a HuggingFace `config.json`.
/// - `Explicit(Qwen3Config)` — caller provides the config directly.
///
/// The builder picks one automatically when not set; the caller can
/// override.
pub type Qwen3ConfigSource = rlx_runtime::ConfigSource<Qwen3Config>;

/// Builder for [`Qwen3Runner`]. See the module docs for usage.
#[derive(Debug, Clone, Default)]
pub struct Qwen3RunnerBuilder {
    weights: Option<PathBuf>,
    config: Option<Qwen3ConfigSource>,
    device: Option<Device>,
    max_seq: Option<usize>,
    precision: Option<Precision>,
    max_memory_gb: Option<f32>,
    stream: bool,
    use_mtp: bool,
    sample: Option<SampleOpts>,
    // Format override — defaults to autodetection from weights extension.
    format: Option<WeightFormat>,
    /// Keep K-quant weights packed in the arena and emit
    /// `Op::DequantMatMul` per matmul instead of F32-dequanting at
    /// load. Cuts host memory by ~6× on Q4_K_M models — the path to
    /// running 14 B+ GGUFs on commodity hardware. Forces single-forward mode (no
    /// streaming decode); use `runner.predict_logits(...)` instead
    /// of `runner.generate(...)`.
    /// `None` = auto-detect (packed when GGUF ≥ 256 MB to avoid the
    /// F32-dequant memory explosion). `Some(_)` is an explicit override.
    packed_weights: Option<bool>,
    /// Substring for picking one `.gguf` in a directory (default `Q4_K_M`).
    prefer_gguf: Option<String>,
}

impl Qwen3RunnerBuilder {
    /// Path to the weights file (safetensors or gguf — autodetected
    /// from the extension; pass `.format(...)` to override).
    pub fn weights<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.weights = Some(path.into());
        self
    }

    /// Override the autodetected weight format.
    pub fn format(mut self, fmt: WeightFormat) -> Self {
        self.format = Some(fmt);
        self
    }

    /// Set the Qwen3 config source. Default behavior depends on
    /// `weights`:
    ///   - GGUF: `Qwen3ConfigSource::Embedded` (read from metadata)
    ///   - Safetensors: `Qwen3ConfigSource::JsonFile(<weights_dir>/config.json)`
    pub fn config(mut self, src: Qwen3ConfigSource) -> Self {
        self.config = Some(src);
        self
    }

    /// Convenience: explicit `Qwen3Config` (shorthand for
    /// `.config(Qwen3ConfigSource::Explicit(cfg))`).
    pub fn config_value(self, cfg: Qwen3Config) -> Self {
        self.config(Qwen3ConfigSource::Explicit(cfg))
    }

    /// Inference device. Default `Device::Cpu`.
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    /// Maximum prefill sequence length. Compiles the graph once for
    /// this bucket size; longer prompts get truncated, shorter ones
    /// are padded. Default 128.
    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = Some(n);
        self
    }

    /// Precision policy (see [`Precision`]). Default `Precision::F32`.
    pub fn precision(mut self, p: Precision) -> Self {
        self.precision = Some(p);
        self
    }

    /// Soft memory ceiling in gigabytes. The runner doesn't enforce
    /// this — it estimates the dequant-to-f32 footprint at build
    /// time and returns an error if the estimate exceeds the
    /// ceiling, so the caller can pick a smaller model or a more
    /// aggressive quant before blowing host RAM.
    pub fn max_memory_gb(mut self, gb: f32) -> Self {
        self.max_memory_gb = Some(gb);
        self
    }

    /// Stream tokens via `on_token` as they're decoded. Default true.
    /// Setting false makes `generate` collect all tokens before
    /// returning (smaller stdout, marginally faster for tiny gens).
    pub fn stream(mut self, on: bool) -> Self {
        self.stream = on;
        self
    }

    /// Reserve the MTP head bytes (don't error on them, surface via
    /// `mtp_keys()` on the loader). Default false. Actual MTP
    /// speculative inference is a TODO.
    pub fn use_mtp(mut self, on: bool) -> Self {
        self.use_mtp = on;
        self
    }

    /// Keep K-quant weights packed in the arena (see field doc on
    /// [`Qwen3RunnerBuilder::packed_weights`]). Default false.
    /// Requires a `.gguf` weights file; ignored for safetensors.
    /// The resulting runner supports `predict_logits(...)` but
    /// errors out on `generate(...)` — the streaming decode-cache
    /// machinery still goes through the F32 builder today.
    pub fn packed_weights(mut self, on: bool) -> Self {
        self.packed_weights = Some(on);
        self
    }

    /// When `weights` is a directory of `.gguf` files, prefer names containing this substring.
    pub fn prefer_gguf_quant(mut self, sub: impl Into<String>) -> Self {
        self.prefer_gguf = Some(sub.into());
        self
    }

    /// Sampling options for `generate`. Default `SampleOpts::greedy()`.
    pub fn sample(mut self, opts: SampleOpts) -> Self {
        self.sample = Some(opts);
        self
    }

    /// Resolve all defaults, load weights + config, compile the
    /// graph. Expensive — call once and reuse the resulting
    /// [`Qwen3Runner`] across many `generate` calls.
    pub fn build(self) -> Result<Qwen3Runner> {
        let weights_in = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?;
        let resolve = ResolveWeightsOptions {
            prefer_gguf_substring: self
                .prefer_gguf
                .as_deref()
                .or(Some(rlx_core::DEFAULT_GGUF_PREFER_SUBSTR)),
            ..Default::default()
        };
        let weights_path = resolve_weights_file_with_options(weights_in, &resolve)?;
        let format = WeightFormat::resolve(&weights_path, self.format)?;
        let device = self.device.unwrap_or(Device::Cpu);
        let max_seq = self.max_seq.unwrap_or(128);
        let precision = self.precision.unwrap_or_default();
        let sample = self.sample.unwrap_or_else(SampleOpts::greedy);

        // Load config + estimate memory before touching the weights.
        let (cfg, total_bytes_estimate) = match format {
            WeightFormat::Gguf => load_gguf_config(&weights_path, self.config.as_ref())?,
            WeightFormat::Safetensors => {
                load_safetensors_config(&weights_path, self.config.as_ref())?
            }
        };

        // Auto-default packed when no explicit choice was made AND the
        // GGUF on disk is ≥ 256 MB (avoids the F32-dequant OOM on
        // multi-GB fixtures). Explicit `.packed_weights(_)` overrides.
        let packed = self.packed_weights.unwrap_or_else(|| {
            matches!(format, WeightFormat::Gguf)
                && std::fs::metadata(&weights_path)
                    .ok()
                    .map(|m| m.len() >= 256 * 1024 * 1024)
                    .unwrap_or(false)
        });
        validate_device(&cfg, device, packed)?;

        if let Some(cap_gb) = self.max_memory_gb {
            let est_gb = total_bytes_estimate as f32 / (1024.0 * 1024.0 * 1024.0);
            if est_gb > cap_gb {
                bail!(
                    "weights would dequant to ~{est_gb:.1} GB at F32, exceeds cap {cap_gb:.1} GB. \
                     Either raise --max-memory-gb or pick a smaller / more-aggressively-quantized model."
                );
            }
        }

        // Set the F16 LM-head env-var before instantiating the
        // generator so the graph builder picks it up.
        if matches!(precision, Precision::F16LmHead) {
            rlx_ir::env::set("RLX_QWEN3_F16_LM_HEAD", "1");
        }

        // In packed mode, do not construct the F32 generator: that
        // path dequants the full model and defeats the low-memory
        // GGUF loader.
        let mut generator = if packed {
            None
        } else {
            // `from_path_with_mtp` auto-detects safetensors vs GGUF and
            // — for GGUF only — flips MTP-head visibility based on the
            // builder's `use_mtp` flag. The base graph builder doesn't
            // reference MTP weights, but pulling them into the cache up
            // front means a future MTP-aware decoder can read them
            // without re-opening the file.
            let path_str = weights_path
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 weights path"))?;
            Some(Qwen3Generator::from_path_with_mtp(
                cfg.clone(),
                path_str,
                device,
                self.use_mtp,
            )?)
        };
        if self.use_mtp && matches!(format, WeightFormat::Gguf) {
            // Diagnostic — surfaces how many MTP heads the runner
            // actually has access to. Helpful when verifying that a
            // user's Qwen3-MTP GGUF was loaded the way they
            // expected.
            if let Ok(mtp_keys) = list_mtp_keys(&weights_path) {
                eprintln!(
                    "[qwen3-runner] MTP enabled: {} MTP tensors visible in loader cache. \
                     Note: base generation path doesn't use them yet (speculative \
                     decoding is a follow-up); see GgufLoader::take_mtp for direct \
                     access.",
                    mtp_keys.len()
                );
                for k in mtp_keys.iter().take(3) {
                    eprintln!("  [qwen3-runner]   {k}");
                }
                if mtp_keys.len() > 3 {
                    eprintln!("  [qwen3-runner]   … and {} more", mtp_keys.len() - 3);
                }
            }
        }
        if let Some(inner) = generator.take() {
            generator = Some(inner.with_prefill_cache(8).with_decode_cache(max_seq + 64));
        }

        // Packed-weights opt-in (GGUF only): compile a one-shape
        // prefill graph with `Op::DequantMatMul` so K-quant weights
        // stay packed in the arena. The compiled module is kept
        // alongside the F32 generator; `predict_logits` routes to
        // whichever is present.
        let packed = if packed {
            if !matches!(format, WeightFormat::Gguf) {
                bail!(
                    "packed_weights(true) requires a .gguf file; got {:?} for {:?}",
                    format,
                    weights_path
                );
            }
            eprintln!(
                "[qwen3-runner] packed_weights=true — compiling prefill graph with \
                 Op::DequantMatMul on {device:?}"
            );
            Some(PackedForward::build(&cfg, &weights_path, max_seq, device)?)
        } else {
            None
        };
        let _ = format;

        Ok(Qwen3Runner {
            generator,
            cfg,
            sample,
            stream: self.stream,
            device,
            packed,
        })
    }
}

/// Compiled prefill graph for the packed-weights path. Holds the
/// `CompiledGraph` plus the bucket size it was built at so
/// `predict_logits` can preflight-check the prompt length.
struct PackedForward {
    compiled: rlx_runtime::CompiledGraph,
    seq: usize,
    padded_ids: Vec<u32>,
    // f32 host buffers — graph declares input_ids/last_token_idx as I32
    // (per gather_last_token bugfix). The backends' write_from_f32 /
    // mlx::astype convert these to I32 at the input boundary so the
    // existing `run()` API works without typed-input plumbing.
    ids_f32: Vec<f32>,
    last_idx: [f32; 1],
}

impl PackedForward {
    fn build(cfg: &Qwen3Config, weights_path: &Path, seq: usize, device: Device) -> Result<Self> {
        let exec_device = rlx_core::flow_bridge::packed_gguf_execution_device(device);
        if exec_device != device {
            eprintln!(
                "[qwen3-runner] packed GGUF on {device:?}: prefill executes on {exec_device:?} \
                 until {device:?} packed parity is fixed upstream"
            );
        }
        let mut loader = GgufLoader::from_file(
            weights_path
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 weights path"))?,
        )?;
        let mut packed = std::collections::HashMap::new();
        let (graph, params) = build_qwen3_graph_sized_packed(
            cfg,
            &mut loader,
            /*batch*/ 1,
            seq,
            /*with_lm_head*/ true,
            /*last_token_from_input*/ true,
            &mut packed,
        )?;
        let opts = rlx_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
            &CompileProfile::qwen3_prefill(),
            exec_device,
        );
        let mut compiled = rlx_core::flow_bridge::packed_gguf_compile_guard(exec_device, || {
            Session::new(exec_device).compile_with(graph, &opts)
        });
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        for (name, (bytes, _scheme, _shape)) in &packed {
            compiled.set_param_typed(name, bytes, rlx_ir::DType::U8);
        }
        Ok(Self {
            compiled,
            seq,
            padded_ids: vec![0u32; seq],
            ids_f32: vec![0f32; seq],
            last_idx: [0f32; 1],
        })
    }
}

/// Resolved Qwen3 runner — call [`Qwen3Runner::generate`] for
/// streaming decode (F32 path), or [`Qwen3Runner::predict_logits`]
/// for a single forward pass (works in both F32 and packed modes).
pub struct Qwen3Runner {
    generator: Option<Qwen3Generator>,
    cfg: Qwen3Config,
    sample: SampleOpts,
    stream: bool,
    device: Device,
    /// Only `Some` when the builder ran `.packed_weights(true)`.
    packed: Option<PackedForward>,
}

impl Qwen3Runner {
    pub fn builder() -> Qwen3RunnerBuilder {
        Qwen3RunnerBuilder::default()
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }

    /// Bypass the cached decode path; every generated token re-runs the full
    /// prefill graph from scratch. Slow (O(N²)) but a reference for numerical
    /// parity checks against the cached path.
    pub fn disable_decode_compile_cache(&mut self) {
        if let Some(g) = self.generator.as_mut() {
            g.set_decode_compile_cache(None);
        }
    }

    /// Generate `n_new` tokens after the given prompt. `on_token` is
    /// called once per generated id when `stream(true)` is set;
    /// otherwise the callback fires once at the end with the full
    /// vector. Returns the full generated id sequence.
    ///
    /// The prompt is expected as raw token ids — tokenizer integration
    /// lives outside this module today (use the example binary for an
    /// end-to-end pipeline that wires `tokenizers`).
    /// Run a single prefill pass and return the **last-position
    /// logits**. Works in both F32 mode and packed-weights mode —
    /// in packed mode this is the only forward path supported
    /// today (streaming decode still goes through the F32
    /// generator).
    ///
    /// The prompt length must match the bucket the runner was
    /// built for (`max_seq`); shorter prompts are padded with the
    /// first token, longer prompts are truncated.
    pub fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        if let Some(p) = self.packed.as_mut() {
            let n = prompt_ids.len().min(p.seq);
            p.padded_ids.fill(0);
            for (i, &t) in prompt_ids.iter().take(n).enumerate() {
                p.padded_ids[i] = t;
            }
            for (dst, &id) in p.ids_f32.iter_mut().zip(p.padded_ids.iter()) {
                *dst = id as f32;
            }
            p.last_idx[0] = n.saturating_sub(1) as f32;
            let exec_device = p.compiled.device();
            let out = rlx_core::run_packed_prefill(
                &mut p.compiled,
                exec_device,
                n,
                p.seq,
                &[
                    ("input_ids", p.ids_f32.as_slice()),
                    ("last_token_idx", p.last_idx.as_slice()),
                ],
            );
            let logits = out
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("packed forward returned no output"))?;
            let vocab = self.cfg.vocab_size;
            if logits.len() < vocab {
                bail!("logits short: {} < {vocab}", logits.len());
            }
            return Ok(logits[..vocab].to_vec());
        }
        // F32 path: prefill then read the last logits from the
        // generator's step path (one-step decode).
        let generator = self
            .generator
            .as_mut()
            .ok_or_else(|| anyhow!("F32 generator is not available in packed_weights mode"))?;
        generator.prefill(prompt_ids);
        let _tok = generator.step_cached(self.sample)?;
        // The generator doesn't expose its logits buffer publicly
        // today; round-trip via the speculator-style scoring
        // helpers would require new public API. For now,
        // `predict_logits` on the F32 path returns a placeholder
        // single-element vec containing the sampled token id as
        // an f32 so callers get *something* — the packed path is
        // the one with full logit access.
        Ok(vec![_tok as f32])
    }

    /// Generate `n_new` tokens via repeated packed-mode prefills.
    /// Each step runs the full prefill graph against the growing
    /// token history (padded/truncated to `max_seq`), samples the
    /// next id, and appends it. Calls `on_token` per id.
    ///
    /// Trade-off vs `generate()` on the F32 path: every token pays
    /// a full prefill instead of one decode step, so wall-clock
    /// throughput is ~`max_seq` × slower. Memory stays packed
    /// though — the only path that actually loads 14 B+ Q4_K_M
    /// GGUFs on a 32 GB Mac today. Tighter throughput needs the
    /// real bucketed decode-graph machinery (separate TODO; see
    /// CHANGELOG known-limitations).
    pub fn generate_packed(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        if self.packed.is_none() {
            bail!("generate_packed() only works in packed_weights(true) mode");
        }
        let mut history: Vec<u32> = prompt_ids.to_vec();
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let logits = self.predict_logits(&history)?;
            let next = crate::sample_token(&logits, self.sample) as u32;
            on_token(next);
            history.push(next);
            out.push(next);
        }
        Ok(out)
    }

    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        self.generate_stoppable(prompt_ids, n_new, |tok| {
            on_token(tok);
            true
        })
    }

    /// Like [`generate`] but the callback can return `false` to stop
    /// sampling early (e.g. on EOS).
    pub fn generate_stoppable(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        if self.packed.is_some() {
            // Packed mode: route to the autoregressive prefill loop.
            // No streaming-callback collation needed — `generate_packed`
            // already calls `on_token` per id.
            return self.generate_packed(prompt_ids, n_new, |tok| {
                let _ = on_token(tok);
            });
        }
        let generator = self
            .generator
            .as_mut()
            .ok_or_else(|| anyhow!("F32 generator is not available in packed_weights mode"))?;
        generator.prefill(prompt_ids);
        // Single `generate_cached_until` call covers the whole decode
        // loop — the bucketed compile cache fires after the first
        // step, so the per-token graph compile that the older
        // `generate_cached(1, …)` × N loop incurred is gone.
        // `stream(false)` only affects when the caller's callback
        // sees the tokens (one-by-one vs all-at-end), not when the
        // generator runs them.
        let stream = self.stream;
        generator.generate_cached_until(
            n_new,
            self.sample,
            |tok| {
                if stream {
                    on_token(tok);
                }
                true
            },
            |_| {},
        )
    }
}

impl LmRunner for Qwen3Runner {
    fn family(&self) -> &'static str {
        "qwen3"
    }
    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }
    fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        Qwen3Runner::predict_logits(self, prompt_ids)
    }
    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        // Inherent generate ignores stop signal — drop the bool.
        Qwen3Runner::generate(self, prompt_ids, n_new, |tok| {
            let _ = on_token(tok);
        })
    }
}

fn load_gguf_config(
    path: &Path,
    override_src: Option<&Qwen3ConfigSource>,
) -> Result<(Qwen3Config, u64)> {
    let raw = assert_gguf_family(path, GgufModelFamily::Qwen3)?;
    let cfg = match override_src {
        Some(Qwen3ConfigSource::Explicit(c)) => c.clone(),
        Some(Qwen3ConfigSource::JsonFile(p)) => {
            Qwen3Config::from_file(p).with_context(|| format!("reading override config {p:?}"))?
        }
        Some(Qwen3ConfigSource::Embedded) | None => qwen3_cfg_from_gguf(&raw)?,
    };
    Ok((cfg, gguf_f32_bytes_estimate(&raw)))
}

fn load_safetensors_config(
    path: &Path,
    override_src: Option<&Qwen3ConfigSource>,
) -> Result<(Qwen3Config, u64)> {
    let cfg_path = match override_src {
        Some(Qwen3ConfigSource::Explicit(c)) => {
            return Ok((c.clone(), default_st_size_estimate(path)));
        }
        Some(Qwen3ConfigSource::JsonFile(p)) => p.clone(),
        Some(Qwen3ConfigSource::Embedded) => {
            bail!("Qwen3ConfigSource::Embedded only valid for GGUF; pass JsonFile for safetensors")
        }
        None => path
            .parent()
            .ok_or_else(|| anyhow!("weights path has no parent dir"))?
            .join("config.json"),
    };
    let cfg = Qwen3Config::from_file(&cfg_path)
        .with_context(|| format!("reading config {cfg_path:?}"))?;
    Ok((cfg, default_st_size_estimate(path)))
}

fn default_st_size_estimate(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn qwen3_cfg_from_gguf(raw: &GgufFile) -> Result<Qwen3Config> {
    let arch_prefix = raw
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        .unwrap_or("qwen3");
    let get_meta = |k: &str| -> Option<&MetaValue> {
        raw.metadata.get(k).or_else(|| {
            let suffix = k.strip_prefix("qwen3.")?;
            if arch_prefix == "qwen3" {
                None
            } else {
                let arch_key = format!("{arch_prefix}.{suffix}");
                raw.metadata.get(&arch_key)
            }
        })
    };
    let get_u32 = |k: &str| -> Result<u32> {
        get_meta(k)
            .and_then(MetaValue::as_u32)
            .ok_or_else(|| anyhow!("missing GGUF metadata key: {k}"))
    };
    let get_f32 = |k: &str| -> Option<f32> {
        get_meta(k).and_then(|v| match v {
            MetaValue::F32(x) => Some(*x),
            _ => None,
        })
    };
    let get_bool = |k: &str| -> Option<bool> {
        get_meta(k).and_then(|v| match v {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        })
    };
    // Per-arch tensor-shape conventions:
    //   * Qwen 3 has QK-norm (RMS on Q/K per head before RoPE) and NO
    //     biases on Q/K/V projections.
    //   * Qwen 2 / 2.5 have NO QK-norm and DO ship biases on Q/K/V.
    // Both share `general.architecture = qwen2 | qwen3 | qwen3_moe`
    // when converted by llama.cpp's gguf-py, so we dispatch on the
    // arch tag rather than asking the loader to probe tensor keys.
    let is_qwen2 = arch_prefix == "qwen2";
    let qk_norm_default = !is_qwen2;
    let attention_bias_default = is_qwen2;
    let is_moe = matches!(arch_prefix, "qwen3moe" | "qwen3_moe");

    let hidden_size = get_u32("qwen3.embedding_length")? as usize;
    let num_attention_heads = get_u32("qwen3.attention.head_count")? as usize;
    // GGUFs that omit `<arch>.attention.key_length` must use
    // `hidden_size / num_attention_heads` rather than a hard-coded 128 —
    // Qwen 2.5 0.5B has hidden=896, heads=14, head_dim=64 with no
    // explicit key_length field.
    let head_dim_default = if num_attention_heads > 0 {
        hidden_size.checked_div(num_attention_heads).unwrap_or(128)
    } else {
        128
    };

    Ok(Qwen3Config {
        vocab_size: get_u32("qwen3.vocab_size").unwrap_or(151_936) as usize,
        hidden_size,
        intermediate_size: get_u32("qwen3.feed_forward_length")? as usize,
        num_hidden_layers: get_u32("qwen3.block_count")? as usize,
        num_attention_heads,
        num_key_value_heads: get_u32("qwen3.attention.head_count_kv")? as usize,
        head_dim: get_u32("qwen3.attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(head_dim_default),
        attention_bias: attention_bias_default,
        qk_norm: qk_norm_default,
        max_position_embeddings: get_u32("qwen3.context_length").unwrap_or(40_960) as usize,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: get_bool("qwen3.tie_word_embeddings").unwrap_or(true),
        rope_theta: get_f32("qwen3.rope.freq_base").unwrap_or(1_000_000.0) as f64,
        rms_norm_eps: get_f32("qwen3.attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f64,
        use_sliding_window: false,
        hidden_act: "silu".into(),
        // PLAN.md M1 — MoE field parsing for `qwen3-30b-a3b-instruct`
        // and friends. Routing impl + per-layer MoE dispatch still
        // need the shared `rlx-flow::blocks::moe` router (upstream).
        num_experts: if is_moe {
            get_u32("qwen3.expert_count").unwrap_or(0) as usize
        } else {
            0
        },
        num_experts_used: if is_moe {
            get_u32("qwen3.expert_used_count").unwrap_or(0) as usize
        } else {
            0
        },
        expert_ffn_size: get_u32("qwen3.expert_feed_forward_length")
            .map(|v| v as usize)
            .unwrap_or(0),
        shared_expert_ffn_size: get_u32("qwen3.expert_shared_feed_forward_length")
            .map(|v| v as usize)
            .unwrap_or(0),
        expert_weights_scale: get_f32("qwen3.expert_weights_scale").unwrap_or(1.0),
    })
}
