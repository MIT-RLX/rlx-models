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

use crate::prefill_mode::MetalGgufPrefillMode;
use crate::{Llama32Config, Llama32Generator, llama32_cfg_from_gguf};
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{LmRunner, WeightFormat};
use rlx_core::lm_builder::LmBuilderCommon;
use rlx_gguf::{GgufFile, MetaValue};
use rlx_qwen3::SampleOpts;
use rlx_runtime::Device;
use std::path::Path;

// ────────────────────────────────────────────────────────────────
// LLaMA-3.2 runner — Meta Llama 3.x small LMs (1B / 3B).
// ────────────────────────────────────────────────────────────────

/// Where to load the Llama 3.2 config from.
///
/// Type alias of the shared `rlx_runtime::ConfigSource<T>` so the
/// per-family `*ConfigSource` enums no longer duplicate the same
/// `Embedded | JsonFile | Explicit(T)` shape. The variant constructors
/// (`Llama32ConfigSource::Embedded`, etc.) keep working because
/// type-alias resolution expands the path through the generic enum.
pub type Llama32ConfigSource = rlx_runtime::ConfigSource<Llama32Config>;

#[derive(Debug, Clone)]
pub struct Llama32RunnerBuilder {
    common: LmBuilderCommon<Llama32Config>,
    /// Family-local: `rlx_qwen3::SampleOpts`, not the `rlx_runtime` one.
    sample: Option<SampleOpts>,
    /// Family-local: `rlx_cli::WeightFormat`, not the `rlx_runtime` one.
    format: Option<WeightFormat>,
    /// `None` = auto (packed when GGUF ≥ 256 MB). `Some(_)` is an explicit override.
    packed_weights: Option<bool>,
    /// When false, decode uses one-shot graphs (slower compile, but
    /// avoids bucketed-cache edge cases on some GPU backends).
    bucketed_decode_cache: bool,
}

impl Default for Llama32RunnerBuilder {
    fn default() -> Self {
        Self {
            common: LmBuilderCommon::default(),
            sample: None,
            format: None,
            packed_weights: None,
            bucketed_decode_cache: true,
        }
    }
}

impl Llama32RunnerBuilder {
    rlx_core::lm_builder_setters!(Llama32Config);

    /// Override the format inferred from the weights path.
    pub fn format(mut self, fmt: WeightFormat) -> Self {
        self.format = Some(fmt);
        self
    }

    /// Sampling options for `generate`. Default `SampleOpts::greedy()`.
    pub fn sample(mut self, opts: SampleOpts) -> Self {
        self.sample = Some(opts);
        self
    }

    /// Keep K-quant weights packed in the arena (`Op::DequantMatMul`).
    /// GGUF only. Supported on CPU, Metal, and MLX.
    ///
    /// When unset, large GGUF files (≥ 256 MB on disk) auto-enable packed
    /// prefill to avoid F32-dequant host memory blowups.
    pub fn packed_weights(mut self, on: bool) -> Self {
        self.packed_weights = Some(on);
        self
    }

    /// Enable bucketed decode compile cache (default: true).
    pub fn bucketed_decode_cache(mut self, on: bool) -> Self {
        self.bucketed_decode_cache = on;
        self
    }

    pub fn build(self) -> Result<Llama32Runner> {
        let weights_path = self.common.weights_path()?.to_path_buf();
        let format = match self.format {
            Some(f) => f,
            None => {
                if weights_path.is_dir() {
                    WeightFormat::detect(&weights_path)?
                } else {
                    WeightFormat::from_path(&weights_path)?
                }
            }
        };
        // `device_or`, not `device_or_auto`: this family has always fallen
        // back to the CPU rather than picking a device, and so have qwen3 and
        // qwen35. Only gemma auto-picks. Kept as-is here so a builder
        // cleanup does not quietly move three families onto the GPU.
        let device = self.common.device_or(Device::Cpu);
        let max_seq = self.common.max_seq_or(128);
        let stream = self.common.stream;
        let sample = self.sample.unwrap_or_else(SampleOpts::greedy);

        let (cfg, total_bytes_estimate) = match format {
            WeightFormat::Gguf => {
                load_llama32_gguf_config(&weights_path, self.common.config.as_ref())?
            }
            WeightFormat::Safetensors => {
                load_llama32_safetensors_config(&weights_path, self.common.config.as_ref())?
            }
        };
        self.common.check_memory_cap(total_bytes_estimate)?;

        let use_packed = self.packed_weights.unwrap_or_else(|| {
            matches!(format, WeightFormat::Gguf)
                && std::fs::metadata(&weights_path)
                    .map(|m| m.len() >= 256 * 1024 * 1024)
                    .unwrap_or(false)
        });

        crate::validate_device(&cfg, device, use_packed)?;

        if use_packed && !matches!(format, WeightFormat::Gguf) {
            bail!(
                "packed_weights(true) requires a .gguf file; got {:?} for {:?}",
                format,
                weights_path
            );
        }

        let prefill_mode = if use_packed {
            if matches!(
                device,
                Device::Metal | Device::Cuda | Device::Rocm | Device::Mlx
            ) {
                MetalGgufPrefillMode::PackedGguf
            } else {
                MetalGgufPrefillMode::CpuF32
            }
        } else {
            MetalGgufPrefillMode::Auto
        };

        if use_packed {
            eprintln!(
                "[llama32-runner] packed_weights=true — Q4 prefill + bucketed decode on {device:?}"
            );
        }

        let path_str = weights_path
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 weights path"))?;
        let mut loader = rlx_core::weight_loader::load_from_path(path_str)?;
        let mut generator = Llama32Generator::from_loader_at_mode(
            cfg.clone(),
            loader.as_mut(),
            device,
            &weights_path,
            prefill_mode,
        )?
        .with_compile_seq_cap(max_seq);
        // Large dense safetensors: keep at most one compiled graph's worth of
        // device weights (prefill+decode would ~2x Nanbeige 3B ~= 31 GiB).
        let large_dense = matches!(format, WeightFormat::Safetensors)
            && total_bytes_estimate >= 2 * 1024 * 1024 * 1024;
        if !large_dense {
            generator = generator.with_prefill_cache(8);
        }
        // Prefer a decode cache for large dense ST even if the caller left
        // bucketed_decode off — oneshot re-attaches the full F32 model each
        // token and usually trips the soft RAM gate after Compiled prefill.
        if self.bucketed_decode_cache || large_dense {
            generator = generator.with_decode_cache(max_seq.saturating_add(16).max(64));
        }

        Ok(Llama32Runner {
            generator,
            cfg,
            sample,
            stream,
            device,
            packed_weights: use_packed,
        })
    }
}

pub struct Llama32Runner {
    generator: Llama32Generator,
    cfg: Llama32Config,
    sample: SampleOpts,
    /// Retained for builder API compatibility. Every generate path now
    /// invokes the caller's per-token callback per token — it is the only
    /// token sink and carries the stop signal — so this no longer gates
    /// anything. Same resolution as `rlx-qwen3`'s and `rlx-gemma`'s.
    #[allow(dead_code)]
    stream: bool,
    device: Device,
    packed_weights: bool,
}

impl Llama32Runner {
    pub fn builder() -> Llama32RunnerBuilder {
        Llama32RunnerBuilder::default()
    }

    pub fn config(&self) -> &Llama32Config {
        &self.cfg
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Current sampling options used by [`generate`](Self::generate) and
    /// [`generate_until`](Self::generate_until).
    pub fn sample_opts(&self) -> &SampleOpts {
        &self.sample
    }

    /// Override the sampling options for subsequent generations. Lets a
    /// long-lived runner switch between greedy / temperature / top-p on a
    /// per-request basis without rebuilding (weights + compile caches stay
    /// warm).
    pub fn set_sample(&mut self, opts: SampleOpts) {
        self.sample = opts;
    }

    /// Single prefill forward; returns last-position logits `[vocab]`.
    pub fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.generator.prefill_get_last_logits(prompt_ids)
    }

    /// Incremental single-token decode: feed the previously-sampled token and get
    /// the next-position logits from the KV cache (O(1) per step). Requires a
    /// prior [`Self::predict_logits`] prefill to populate the cache. Pairs with it
    /// for O(n) autoregressive loops that need raw logits (e.g. the Orpheus SNAC
    /// decode that applies per-slot logit masking before sampling) — instead of
    /// re-prefilling the whole growing history each step (O(n²)).
    pub fn decode_get_logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.generator.decode_get_logits(token)
    }

    /// Register a one-shot multimodal embed splice for the next `generate`
    /// prefill — the vision soft tokens overwrite positions `start..start+rows`
    /// of the packed `input_embeddings`. See
    /// [`Llama32Generator::set_multimodal_embed_override`].
    pub fn set_multimodal_embed_override(&mut self, start: usize, embeds: Vec<f32>) {
        self.generator.set_multimodal_embed_override(start, embeds);
    }

    /// Whether a registered multimodal splice is still unconsumed (packed path
    /// not taken). See [`Llama32Generator::multimodal_override_pending`].
    pub fn multimodal_override_pending(&self) -> bool {
        self.generator.multimodal_override_pending()
    }

    /// Drop any unconsumed multimodal splice. See
    /// [`Llama32Generator::clear_multimodal_embed_override`].
    pub fn clear_multimodal_embed_override(&mut self) {
        self.generator.clear_multimodal_embed_override();
    }

    pub fn generate_packed(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        if !self.packed_weights {
            bail!("generate_packed() only works in packed_weights(true) mode");
        }
        self.generate(prompt_ids, n_new, on_token)
    }

    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        // One loop, through `generate_until`. The `stream(false)` branch this
        // replaces collected every token first and only then replayed the
        // callback: identical ids in identical order, except that a caller
        // asking to be told per token was told at the end instead — and a
        // stop could not take effect at all, which is what made the trait
        // impl's dropped bool invisible. qwen3 and gemma settled this the
        // same way.
        self.generate_until(prompt_ids, n_new, |tok| {
            on_token(tok);
            true
        })
    }

    /// KV-cached generation with an early stop. `keep_going` is called with
    /// each freshly sampled id and returns whether to continue; returning
    /// `false` halts generation *after* that token (which is included in the
    /// result). Use it to stop on an end-of-sequence id instead of always
    /// decoding the full `n_new` budget.
    pub fn generate_until(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        keep_going: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.generator.prefill(prompt_ids);
        self.generator
            .generate_cached_until(n_new, self.sample, keep_going)
    }
}

impl LmRunner for Llama32Runner {
    fn family(&self) -> &'static str {
        "llama32"
    }
    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }
    fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        Llama32Runner::predict_logits(self, prompt_ids)
    }

    /// The host-driven decode pair. Both already existed on the runner; not
    /// forwarding them here left every caller that samples for itself —
    /// anything applying a temperature, a stop string, a logit bias or
    /// log-probs — with "host-driven logits decode unsupported", and so
    /// unable to generate at all through this trait.
    fn prefill_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        Llama32Runner::predict_logits(self, prompt_ids)
    }

    fn decode_logits(&mut self, token: u32) -> Result<Vec<f32>> {
        Llama32Runner::decode_get_logits(self, token)
    }

    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        // `generate_until` honours the callback's `false`; the plain
        // `generate` cannot, and routing through it dropped every reason a
        // caller has to end a turn early — EOS, a stop string, a repetition
        // guard — so a turn always ran the full `n_new`. (Its non-streaming
        // branch is worse than it looks: it collects every token first and
        // only then replays the callback, so a stop could not have been acted
        // on even if the bool had been read.)
        Llama32Runner::generate_until(self, prompt_ids, n_new, on_token)
    }
}

fn load_llama32_gguf_config(
    path: &Path,
    override_src: Option<&Llama32ConfigSource>,
) -> Result<(Llama32Config, u64)> {
    let raw = GgufFile::from_path(path).with_context(|| format!("opening {path:?}"))?;
    let arch = raw
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        .unwrap_or("llama");
    // mistral3/4: Ministral / Mistral Medium — Llama-shaped GGUFs with
    // arch-prefixed metadata keys (parsed via llama32_cfg_from_gguf).
    // granite: IBM Granite dense — Llama-shaped + scalar multipliers
    // (embedding/residual/attention/logit), applied in the packed builder.
    // exaone: LG ExaOne 3.x — Llama-shaped, NeoX RoPE.
    // olmo2/olmo, nemotron, glm4/chatglm all reuse the Llama qkv+rope+attention
    // core with a small per-arch structural delta applied in the packed builder
    // (see `builder.rs` `emit_*` helpers). Each is validated coherent on a real
    // Q4_K_M GGUF. cohere/command-r/cohere2 are NOT routed: the parallel-residual
    // path is coded but cohere2 (command-r7b) still outputs garbage — it needs
    // its per-layer sliding/full-attention + NoPE-on-global-layers logic, which
    // is unverified; kept in `known_unimplemented_arch` until it's correct.
    // muse-glimmer IS routed: unlike cohere2 above, its per-layer
    // sliding-window/full-attention split and NoPE-on-global-layers logic is
    // implemented and parity-tested against a reference decoder on CPU, Metal,
    // MLX, CoreML, wgpu, CUDA and Vulkan (`tests/muse_glimmer_arch.rs`).
    const LLAMA_SHAPED_GGUF_ARCHES: &[&str] = &[
        "llama",
        "phi3",
        "phi4",
        "mistral3",
        "mistral4",
        "granite",
        "exaone",
        "olmo2",
        "olmo",
        "nemotron",
        "glm4",
        "chatglm",
        "muse-glimmer",
    ];
    if !LLAMA_SHAPED_GGUF_ARCHES.contains(&arch) {
        bail!(
            "{path:?} has architecture {arch:?}; Llama32Runner expects general.architecture ∈ {LLAMA_SHAPED_GGUF_ARCHES:?}"
        );
    }
    let cfg = match override_src {
        Some(Llama32ConfigSource::Explicit(c)) => c.clone(),
        Some(Llama32ConfigSource::JsonFile(p)) => {
            Llama32Config::from_file(p).with_context(|| format!("reading override config {p:?}"))?
        }
        Some(Llama32ConfigSource::Embedded) | None => llama32_cfg_from_gguf(&raw)?,
    };
    let bytes_est: u64 = raw
        .tensors
        .values()
        .map(|t| (t.n_elements() as u64) * 4)
        .sum();
    Ok((cfg, bytes_est))
}

fn load_llama32_safetensors_config(
    path: &Path,
    override_src: Option<&Llama32ConfigSource>,
) -> Result<(Llama32Config, u64)> {
    let cfg_path = match override_src {
        Some(Llama32ConfigSource::Explicit(c)) => {
            return Ok((c.clone(), default_st_size_estimate(path)));
        }
        Some(Llama32ConfigSource::JsonFile(p)) => p.clone(),
        Some(Llama32ConfigSource::Embedded) => {
            bail!("ConfigSource::Embedded only valid for GGUF; pass JsonFile for safetensors")
        }
        // Beside the weights — which means *inside* them when the path is a
        // model directory rather than a file. A server hands over whichever
        // the model library recorded, and looking in the parent of a
        // directory lands one level too high, with an error naming a path
        // the user never mentioned.
        None if path.is_dir() => path.join("config.json"),
        None => path
            .parent()
            .ok_or_else(|| anyhow!("weights path has no parent dir"))?
            .join("config.json"),
    };
    let cfg = Llama32Config::from_file(&cfg_path)
        .with_context(|| format!("reading config {cfg_path:?}"))?;
    Ok((cfg, default_st_size_estimate(path)))
}

fn default_st_size_estimate(path: &Path) -> u64 {
    if path.is_dir() {
        let mut sum = 0u64;
        if let Ok(rd) = std::fs::read_dir(path) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension()
                    .and_then(|s| s.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
                {
                    sum = sum.saturating_add(std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0));
                }
            }
        }
        return sum;
    }
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

#[cfg(test)]
mod builder_api_tests {
    use super::*;

    /// The chainable setter surface is public API, and it is now generated
    /// by `rlx_core::lm_builder_setters!` rather than written out here.
    ///
    /// Two things this pins that a compile does not: that every setter still
    /// exists under its old name and signature, and that the macro writes
    /// each value into the field the builder actually reads. A setter wired
    /// to the wrong field of the right type — `max_seq` into `max_memory_gb`
    /// — compiles, and then the model is built with someone else's number.
    #[test]
    fn every_setter_still_exists_and_lands_where_build_reads_it() {
        let b = Llama32Runner::builder()
            .weights("/tmp/does-not-need-to-exist.gguf")
            .config(Llama32ConfigSource::Embedded)
            .device(Device::Cpu)
            .max_seq(77)
            .max_memory_gb(1.5)
            .stream(false)
            .sample(SampleOpts::greedy())
            .format(WeightFormat::Gguf);

        assert_eq!(
            b.common.weights.as_deref(),
            Some(std::path::Path::new("/tmp/does-not-need-to-exist.gguf"))
        );
        assert!(matches!(
            b.common.config,
            Some(Llama32ConfigSource::Embedded)
        ));
        assert_eq!(b.common.device, Some(Device::Cpu));
        assert_eq!(b.common.max_seq, Some(77));
        assert_eq!(b.common.max_memory_gb, Some(1.5));
        assert!(!b.common.stream);
        assert!(b.sample.is_some());
        assert!(matches!(b.format, Some(WeightFormat::Gguf)));

        // Unset stays unset, which is what keeps auto-device reachable.
        let d = Llama32Runner::builder();
        assert!(d.common.device.is_none());
        assert!(d.common.max_seq.is_none());
    }
}
