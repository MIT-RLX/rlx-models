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
use rlx_gguf::{GgufFile, MetaValue};
use rlx_qwen3::SampleOpts;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

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
    weights: Option<PathBuf>,
    config: Option<Llama32ConfigSource>,
    device: Option<Device>,
    max_seq: Option<usize>,
    max_memory_gb: Option<f32>,
    stream: bool,
    sample: Option<SampleOpts>,
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
            weights: None,
            config: None,
            device: None,
            max_seq: None,
            max_memory_gb: None,
            stream: true,
            sample: None,
            format: None,
            packed_weights: None,
            bucketed_decode_cache: true,
        }
    }
}

impl Llama32RunnerBuilder {
    pub fn weights<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.weights = Some(path.into());
        self
    }

    pub fn format(mut self, fmt: WeightFormat) -> Self {
        self.format = Some(fmt);
        self
    }

    pub fn config(mut self, src: Llama32ConfigSource) -> Self {
        self.config = Some(src);
        self
    }

    pub fn config_value(self, cfg: Llama32Config) -> Self {
        self.config(Llama32ConfigSource::Explicit(cfg))
    }

    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = Some(n);
        self
    }

    pub fn max_memory_gb(mut self, gb: f32) -> Self {
        self.max_memory_gb = Some(gb);
        self
    }

    pub fn stream(mut self, on: bool) -> Self {
        self.stream = on;
        self
    }

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
        let weights_path = self
            .weights
            .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?;
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
        let device = self.device.unwrap_or(Device::Cpu);
        let max_seq = self.max_seq.unwrap_or(128);
        let stream = self.stream;
        let sample = self.sample.unwrap_or_else(SampleOpts::greedy);

        let (cfg, total_bytes_estimate) = match format {
            WeightFormat::Gguf => load_llama32_gguf_config(&weights_path, self.config.as_ref())?,
            WeightFormat::Safetensors => {
                load_llama32_safetensors_config(&weights_path, self.config.as_ref())?
            }
        };

        if let Some(cap_gb) = self.max_memory_gb {
            let est_gb = total_bytes_estimate as f32 / (1024.0 * 1024.0 * 1024.0);
            if est_gb > cap_gb {
                bail!(
                    "weights would dequant to ~{est_gb:.1} GB at F32, exceeds cap {cap_gb:.1} GB"
                );
            }
        }

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
        // device weights (prefill+decode would ~2× Nanbeige 3B ≈ 31 GiB).
        let large_dense_st = matches!(format, WeightFormat::Safetensors)
            && total_bytes_estimate >= 2 * 1024 * 1024 * 1024;
        if !large_dense_st {
            generator = generator.with_prefill_cache(8);
        }
        // Prefer a decode cache for large dense ST even if the caller left
        // bucketed_decode off — oneshot re-attaches the full F32 model each
        // token and usually trips the soft RAM gate after Compiled prefill.
        if self.bucketed_decode_cache || large_dense_st {
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
        self.generator.prefill(prompt_ids);
        let tokens = if self.stream {
            self.generator
                .generate_cached_with(n_new, self.sample, &mut on_token)?
        } else {
            let toks = self.generator.generate_cached(n_new, self.sample)?;
            for &t in &toks {
                on_token(t);
            }
            toks
        };
        Ok(tokens)
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
    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        Llama32Runner::generate(self, prompt_ids, n_new, |tok| {
            let _ = on_token(tok);
        })
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
    const LLAMA_SHAPED_GGUF_ARCHES: &[&str] = &["llama", "phi3", "phi4"];
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
