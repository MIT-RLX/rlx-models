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

//! High-level FLUX.2 runner (denoiser + text encoder + VAE caches).

use anyhow::{Context, Result, anyhow, bail, ensure};
use rlx_runtime::Device;
use std::path::PathBuf;
use std::sync::Mutex;

/// Noise prediction from [`Flux2Runner::forward`].
#[derive(Debug, Clone)]
pub struct Flux2Output {
    pub noise_pred: Vec<f32>,
    pub img_seq: usize,
    pub out_dim: usize,
}

/// Builder for [`Flux2Runner`].
#[derive(Debug, Clone, Default)]
pub struct Flux2RunnerBuilder {
    weights: Option<PathBuf>,
    config: Option<crate::Flux2Config>,
    config_path: Option<PathBuf>,
    text_encoder_dir: Option<PathBuf>,
    text_encoder_config_path: Option<PathBuf>,
    vae_dir: Option<PathBuf>,
    vae_config_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    batch: Option<usize>,
    img_seq: Option<usize>,
    txt_seq: Option<usize>,
    device: Option<Device>,
    /// Use HIR denoiser on CPU too (default: native on CPU, compiled on GPU backends).
    compiled_denoiser: bool,
    /// Use HIR text encoder on CPU too (default: native CPU; compiled on Metal/MLX only).
    compiled_text_encoder: bool,
    /// Use HIR VAE decoder on CPU too (default: native on CPU, compiled on GPU backends).
    compiled_vae: bool,
    /// Load NVFP4 packed linears from weights (`None` = auto-detect U8+F8 pairs).
    nvfp4: Option<bool>,
    /// Skip loading Qwen3 text encoder weights (saves ~8GB RAM; img2img/edit with empty prompt).
    skip_text_encoder: bool,
    /// Persist compiled LIR to disk (speeds up repeat runs).
    aot_cache_dir: Option<PathBuf>,
    /// After prompt encode, drop TE weights + compiled cache to free RAM before denoiser.
    drop_text_encoder_after_encode: Option<bool>,
    /// Optional LoRA safetensors (merged into base weights before extract).
    lora_path: Option<PathBuf>,
    lora_scale: f32,
    /// Compile denoiser via tier-0 [`crate::Flux2Flow`] API (AOT key suffix `_flow`).
    use_flow_api: bool,
    /// Second timestep embedder for flow-map dual-time (auto on when LoRA is set).
    dual_time_embedder: bool,
}

impl Flux2RunnerBuilder {
    pub fn weights<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.weights = Some(p.into());
        self
    }
    pub fn config(mut self, cfg: crate::Flux2Config) -> Self {
        self.config = Some(cfg);
        self
    }
    pub fn config_path<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.config_path = Some(p.into());
        self
    }
    pub fn batch(mut self, n: usize) -> Self {
        self.batch = Some(n);
        self
    }
    pub fn img_seq(mut self, n: usize) -> Self {
        self.img_seq = Some(n);
        self
    }
    pub fn txt_seq(mut self, n: usize) -> Self {
        self.txt_seq = Some(n);
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    /// Run the denoiser via compiled HIR on CPU as well (for parity / bench).
    pub fn compiled_denoiser(mut self, yes: bool) -> Self {
        self.compiled_denoiser = yes;
        self
    }

    /// Run the text encoder via compiled HIR on CPU as well (for parity / bench).
    pub fn compiled_text_encoder(mut self, yes: bool) -> Self {
        self.compiled_text_encoder = yes;
        self
    }

    /// Run the VAE decoder via compiled HIR on CPU as well (for parity / bench).
    pub fn compiled_vae(mut self, yes: bool) -> Self {
        self.compiled_vae = yes;
        self
    }

    pub fn text_encoder_dir<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.text_encoder_dir = Some(path.into());
        self
    }

    pub fn text_encoder_config_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.text_encoder_config_path = Some(path.into());
        self
    }

    pub fn tokenizer_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.tokenizer_path = Some(path.into());
        self
    }

    pub fn vae_dir<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.vae_dir = Some(path.into());
        self
    }

    pub fn vae_config_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.vae_config_path = Some(path.into());
        self
    }

    /// Force NVFP4 packed weights on/off (`None` = auto-detect in safetensors).
    pub fn nvfp4(mut self, enable: bool) -> Self {
        self.nvfp4 = Some(enable);
        self
    }

    /// Do not load `text_encoder/` even when present (saves RAM; use with empty prompt).
    pub fn skip_text_encoder(mut self, yes: bool) -> Self {
        self.skip_text_encoder = yes;
        self
    }

    /// Directory for AOT compile cache (denoiser / TE / VAE / CFG graphs).
    pub fn aot_cache_dir<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.aot_cache_dir = Some(path.into());
        self
    }

    /// Drop text-encoder weights after first encode (default: true on GPU compiled paths).
    pub fn drop_text_encoder_after_encode(mut self, yes: bool) -> Self {
        self.drop_text_encoder_after_encode = Some(yes);
        self
    }

    /// Merge LoRA adapter weights from `path` with strength `scale` before loading the denoiser.
    pub fn lora<P: Into<PathBuf>>(mut self, path: P, scale: f32) -> Self {
        self.lora_path = Some(path.into());
        self.lora_scale = scale;
        self.dual_time_embedder = true;
        self
    }

    /// Use separate (or cloned) timestep embedder weights for dual-time flow-map forwards.
    pub fn dual_time_embedder(mut self, yes: bool) -> Self {
        self.dual_time_embedder = yes;
        self
    }

    /// Build the denoiser via [`crate::Flux2Flow`] instead of direct HIR builder.
    pub fn use_flow_api(mut self, yes: bool) -> Self {
        self.use_flow_api = yes;
        self
    }

    /// Cache key for [`crate::Flux2SessionCache`].
    pub fn session_key(&self) -> Option<crate::Flux2SessionKey> {
        self.weights.as_ref().map(|w| crate::Flux2SessionKey {
            weights: w.clone(),
            device: self.device.unwrap_or(Device::Cpu),
            config_path: self.config_path.clone(),
            lora_path: self.lora_path.clone(),
            lora_scale_bits: self.lora_scale.to_bits(),
            nvfp4: self.nvfp4,
        })
    }

    pub fn build(self) -> Result<Flux2Runner> {
        use crate::Flux2VaeConfig;
        use crate::{
            ExtractFlux2Opts, Flux2Config, extract_flux2_weights_with_opts,
            load_flux2_nvfp4_from_file, load_flux2_vae_weights, load_flux2_weight_map,
            load_text_encoder_weights, load_typed_linears_from_file, prepare_weight_map,
            resolve_text_encoder_dir, resolve_transformer_config, resolve_vae_dir,
            safetensors_has_nvfp4,
        };
        use rlx_core::gguf_support::{ResolveWeightsOptions, resolve_weights_file_with_options};
        use rlx_gguf::GgufFile;
        use rlx_qwen3::Qwen3Config;

        let weights_path = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?;
        let weights_file = resolve_weights_file_with_options(
            weights_path,
            &ResolveWeightsOptions {
                prefer_gguf_substring: Some("Q4_K_M"),
                ..Default::default()
            },
        )?;
        let is_gguf = weights_file.extension().and_then(|s| s.to_str()) == Some("gguf");

        let cfg = match (self.config, self.config_path.clone()) {
            (Some(c), _) => c,
            (_, Some(p)) if !is_gguf => Flux2Config::from_file(&p)?,
            _ if is_gguf => {
                let raw = GgufFile::from_path(&weights_file)
                    .with_context(|| format!("opening GGUF {weights_file:?}"))?;
                Flux2Config::from_gguf(&raw)?
            }
            _ => {
                if let Some(p) = resolve_transformer_config(&weights_file, None) {
                    Flux2Config::from_file(&p)?
                } else {
                    Flux2Config::flux2_dev()
                }
            }
        };
        let path = weights_file
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 weights path"))?;

        let device = self.device.unwrap_or(Device::Cpu);
        use crate::{
            assert_flux2_device_available, flux2_prefers_compiled_hir, flux2_prefers_compiled_te,
        };
        rlx_core::validate_standard_device("flux2", device)?;
        assert_flux2_device_available(device)?;
        if self.compiled_text_encoder && !flux2_prefers_compiled_te(device) {
            anyhow::bail!(
                "compiled text encoder on {device:?} can take hours and exhaust VRAM; \
                 use native CPU TE (default on CUDA/ROCm/wgpu/Vulkan)"
            );
        }
        let compiled_denoiser = self.compiled_denoiser || flux2_prefers_compiled_hir(device);
        let compiled_text_encoder = self.compiled_text_encoder || flux2_prefers_compiled_te(device);
        let compiled_vae = self.compiled_vae || flux2_prefers_compiled_hir(device);

        let use_nvfp4 = if is_gguf {
            false
        } else {
            match self.nvfp4 {
                Some(yes) => yes,
                None => safetensors_has_nvfp4(&weights_file).unwrap_or(false),
            }
        };
        let use_gguf_packed = is_gguf
            && match self.nvfp4 {
                Some(false) => false,
                _ => crate::packed_gguf::gguf_has_packed_linears(&weights_file).unwrap_or(false),
            };
        let packed = if use_nvfp4 {
            Some(load_flux2_nvfp4_from_file(&weights_file)?)
        } else {
            None
        };

        let mut exclude_f32 = std::collections::HashSet::new();
        if let Some(p) = &packed {
            exclude_f32.extend(p.exclude_f32_keys());
        }
        let typed_linears = if is_gguf {
            None
        } else if compiled_denoiser && self.lora_path.is_none() {
            Some(load_typed_linears_from_file(&weights_file, &exclude_f32)?)
        } else {
            if self.lora_path.is_some() && compiled_denoiser {
                eprintln!(
                    "[flux2] LoRA active — using merged f32 weights (typed BF16 linears disabled)"
                );
            }
            None
        };
        if let Some(t) = &typed_linears {
            exclude_f32.extend(t.skip_keys());
        }

        let (mut wm, gguf_packed) = if use_gguf_packed {
            if self.lora_path.is_some() {
                bail!("LoRA merge is not supported on GGUF denoiser weights; use safetensors");
            }
            eprintln!("[flux2] loading denoiser GGUF with packed DequantMatMul {weights_file:?}");
            {
                let (wm, g) = crate::packed_gguf::load_flux2_from_gguf(&weights_file)?;
                (wm, Some(g))
            }
        } else if is_gguf {
            if self.lora_path.is_some() {
                bail!("LoRA merge is not supported on GGUF denoiser weights; use safetensors");
            }
            eprintln!("[flux2] loading denoiser from GGUF (F32 drain) {weights_file:?}");
            (load_flux2_weight_map(&weights_file)?, None)
        } else {
            use rlx_core::weight_map::WeightMap;
            (WeightMap::from_file_excluding(path, &exclude_f32)?, None)
        };
        let packed = match (packed, gguf_packed) {
            (Some(nv), None) => Some(nv),
            (None, Some(g)) => Some(g),
            (None, None) => None,
            (Some(_), Some(_)) => unreachable!("nvfp4 and gguf packed are mutually exclusive"),
        };
        if let Some(lora_path) = &self.lora_path {
            let n = if lora_path.is_dir() {
                crate::load_and_apply_flux2_lora_dir(&mut wm, lora_path, self.lora_scale)?
            } else {
                crate::load_and_apply_flux2_lora(&mut wm, lora_path, self.lora_scale)?
            };
            eprintln!(
                "[flux2] merged {n} LoRA layers from {:?} (scale={})",
                lora_path, self.lora_scale
            );
        }
        let extract_opts = ExtractFlux2Opts {
            typed_linears: typed_linears.as_ref(),
            packed_linears: packed.as_ref(),
            dual_time_embedder: self.dual_time_embedder || self.lora_path.is_some(),
        };
        if extract_opts.dual_time_embedder {
            eprintln!("[flux2] dual-time timestep embedder enabled (flow-map / Diamond Maps)");
        }
        let model = extract_flux2_weights_with_opts(prepare_weight_map(wm), &cfg, extract_opts)?;

        let te_dir = if self.skip_text_encoder {
            None
        } else {
            self.text_encoder_dir
                .or_else(|| resolve_text_encoder_dir(&weights_file))
        };
        let (text_encoder, text_encoder_cfg) = if let Some(dir) = te_dir {
            let te_cfg_path = self
                .text_encoder_config_path
                .unwrap_or_else(|| dir.join("config.json"));
            let te_cfg = Qwen3Config::from_file(&te_cfg_path)?;
            let te = load_text_encoder_weights(&dir, &te_cfg)?;
            (Some(te), Some(te_cfg))
        } else {
            (None, None)
        };

        let vae_dir = self.vae_dir.or_else(|| resolve_vae_dir(&weights_file));
        let (vae, vae_cfg) = if let Some(dir) = vae_dir {
            let vae_cfg_path = self
                .vae_config_path
                .unwrap_or_else(|| dir.join("config.json"));
            let vae_cfg = Flux2VaeConfig::from_file(&vae_cfg_path)?;
            let vae = load_flux2_vae_weights(&dir, &vae_cfg)?;
            (Some(vae), Some(vae_cfg))
        } else {
            (None, None)
        };

        let drop_text_encoder_after_encode = self
            .drop_text_encoder_after_encode
            .unwrap_or(!self.skip_text_encoder && compiled_denoiser);

        Ok(Flux2Runner {
            model,
            cfg,
            batch: self.batch.unwrap_or(1),
            img_seq: self.img_seq.unwrap_or(256),
            txt_seq: self.txt_seq.unwrap_or(128),
            device,
            compiled_denoiser,
            compiled_text_encoder,
            compiled_vae,
            packed,
            typed_linears,
            aot_cache_dir: self.aot_cache_dir,
            drop_text_encoder_after_encode,
            use_flow_api: self.use_flow_api,
            text_encoder: Mutex::new(text_encoder),
            text_encoder_cfg: Mutex::new(text_encoder_cfg),
            vae,
            vae_cfg,
            tokenizer_path: self.tokenizer_path,
            model_root: weights_file,
            denoiser: Mutex::new(None),
            text_encoder_compiled: Mutex::new(None),
            vae_compiled: Mutex::new(None),
            vae_encoder_compiled: Mutex::new(None),
            cfg_compiled: Mutex::new(None),
        })
    }
}

struct Flux2DenoiserCache {
    compiled: rlx_runtime::CompiledGraph,
    device: Device,
    batch: usize,
    img_seq: usize,
    txt_seq: usize,
    img_ids: Vec<f32>,
    txt_ids: Vec<f32>,
}

struct Flux2TextEncoderCache {
    compiled: rlx_runtime::CompiledGraph,
    device: Device,
    batch: usize,
    txt_seq: usize,
}

struct Flux2VaeCache {
    compiled: rlx_runtime::CompiledGraph,
    device: Device,
    batch: usize,
    h: usize,
    w: usize,
}

struct Flux2VaeEncoderCache {
    compiled: rlx_runtime::CompiledGraph,
    device: Device,
    batch: usize,
    h: usize,
    w: usize,
}

struct Flux2CfgCache {
    compiled: rlx_runtime::CompiledGraph,
    device: Device,
    batch: usize,
    img_seq: usize,
    out_dim: usize,
}

/// FLUX.2 denoiser runner — native CPU or compiled HIR on any [`Device`].
pub struct Flux2Runner {
    model: crate::Flux2Weights,
    cfg: crate::Flux2Config,
    batch: usize,
    img_seq: usize,
    txt_seq: usize,
    device: Device,
    /// When true, use HIR even on CPU; otherwise CPU uses native forward.
    compiled_denoiser: bool,
    /// When true, use HIR text encoder even on CPU.
    compiled_text_encoder: bool,
    compiled_vae: bool,
    packed: Option<crate::Flux2PackedParams>,
    typed_linears: Option<crate::TypedLinearStore>,
    aot_cache_dir: Option<PathBuf>,
    drop_text_encoder_after_encode: bool,
    use_flow_api: bool,
    text_encoder: Mutex<Option<crate::Flux2TextEncoderWeights>>,
    text_encoder_cfg: Mutex<Option<rlx_qwen3::Qwen3Config>>,
    vae: Option<crate::Flux2VaeWeights>,
    vae_cfg: Option<crate::Flux2VaeConfig>,
    tokenizer_path: Option<PathBuf>,
    model_root: PathBuf,
    denoiser: Mutex<Option<Flux2DenoiserCache>>,
    text_encoder_compiled: Mutex<Option<Flux2TextEncoderCache>>,
    vae_compiled: Mutex<Option<Flux2VaeCache>>,
    vae_encoder_compiled: Mutex<Option<Flux2VaeEncoderCache>>,
    cfg_compiled: Mutex<Option<Flux2CfgCache>>,
}

impl Flux2Runner {
    pub fn builder() -> Flux2RunnerBuilder {
        Flux2RunnerBuilder::default()
    }

    fn aot_cache(&self) -> Option<rlx_runtime::AotCache> {
        self.aot_cache_dir
            .as_ref()
            .map(|p| rlx_runtime::AotCache::new(p.clone()))
    }

    pub fn drop_text_encoder_weights(&self) -> Result<()> {
        if let Ok(mut te) = self.text_encoder.lock() {
            if te.is_some() {
                eprintln!("[flux2] dropping text encoder weights (~8GB RAM)");
                *te = None;
            }
        }
        if let Ok(mut cfg) = self.text_encoder_cfg.lock() {
            *cfg = None;
        }
        if let Ok(mut cache) = self.text_encoder_compiled.lock() {
            *cache = None;
        }
        Ok(())
    }
    pub fn config(&self) -> &crate::Flux2Config {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn img_seq(&self) -> usize {
        self.img_seq
    }

    pub fn txt_seq(&self) -> usize {
        self.txt_seq
    }

    pub fn uses_nvfp4(&self) -> bool {
        self.packed.is_some()
    }

    pub fn has_text_encoder(&self) -> bool {
        self.text_encoder
            .lock()
            .map(|g| g.is_some())
            .unwrap_or(false)
    }

    pub fn has_vae(&self) -> bool {
        self.vae.is_some()
    }

    /// True when denoiser forwards use compiled HIR ([`Self::device`]).
    pub fn uses_compiled_denoiser(&self) -> bool {
        self.compiled_denoiser
    }

    /// True when text encoding uses compiled HIR on [`Self::device`].
    pub fn uses_compiled_text_encoder(&self) -> bool {
        self.compiled_text_encoder
    }

    pub fn uses_compiled_vae(&self) -> bool {
        self.compiled_vae
    }

    /// Pre-compile the denoiser HIR for the given position ids (RoPE tables are baked in).
    pub fn warmup_denoiser(&self, img_ids: &[f32], txt_ids: &[f32]) -> Result<()> {
        if self.uses_compiled_denoiser() {
            self.ensure_denoiser_compiled(img_ids, txt_ids)?;
        }
        Ok(())
    }

    fn ensure_denoiser_compiled(&self, img_ids: &[f32], txt_ids: &[f32]) -> Result<()> {
        use crate::{compile_flux2_forward, compile_flux2_forward_via_flow};

        let mut guard = self
            .denoiser
            .lock()
            .map_err(|e| anyhow!("denoiser cache lock poisoned: {e}"))?;
        let img_seq = img_ids.len() / (self.batch * 4);
        let recompile = guard.as_ref().is_none_or(|c| {
            c.device != self.device
                || c.batch != self.batch
                || c.img_seq != img_seq
                || c.txt_seq != self.txt_seq
                || c.img_ids != img_ids
                || c.txt_ids != txt_ids
        });
        if recompile {
            eprintln!(
                "[flux2] compiling denoiser HIR on {:?} (img_seq={img_seq}, txt_seq={}, flow={})…",
                self.device, self.txt_seq, self.use_flow_api
            );
            let aot = self.aot_cache();
            let (compiled, _) = if self.use_flow_api {
                compile_flux2_forward_via_flow(
                    &self.cfg,
                    &self.model,
                    self.batch,
                    img_seq,
                    self.txt_seq,
                    img_ids,
                    txt_ids,
                    self.device,
                    self.packed.as_ref(),
                    self.typed_linears.as_ref(),
                    aot.as_ref(),
                )?
            } else {
                compile_flux2_forward(
                    &self.cfg,
                    &self.model,
                    self.batch,
                    img_seq,
                    self.txt_seq,
                    img_ids,
                    txt_ids,
                    self.device,
                    self.packed.as_ref(),
                    self.typed_linears.as_ref(),
                    aot.as_ref(),
                )?
            };
            *guard = Some(Flux2DenoiserCache {
                compiled,
                device: self.device,
                batch: self.batch,
                img_seq,
                txt_seq: self.txt_seq,
                img_ids: img_ids.to_vec(),
                txt_ids: txt_ids.to_vec(),
            });
        }
        Ok(())
    }

    /// Encode a text prompt into FLUX.2 `encoder_hidden_states` and `txt_ids`.
    ///
    /// Uses compiled HIR on Metal / MLX when [`Self::uses_compiled_text_encoder`];
    /// native CPU on CUDA / ROCm / wgpu / Vulkan and CPU otherwise.
    pub fn encode_prompt(&self, prompt: &str) -> Result<(Vec<f32>, Vec<f32>)> {
        if self.uses_compiled_text_encoder() {
            return self.encode_prompt_compiled(prompt);
        }
        eprintln!("[flux2] text encoder: native CPU forward");
        self.encode_prompt_native(prompt)
    }

    /// Native CPU text encoder (no IR compile).
    pub fn encode_prompt_native(&self, prompt: &str) -> Result<(Vec<f32>, Vec<f32>)> {
        use crate::{
            DEFAULT_TEXT_ENCODER_LAYERS, encode_flux2_prompt, encode_prompt_padded,
            resolve_tokenizer_path,
        };

        let tok_path = resolve_tokenizer_path(&self.model_root, self.tokenizer_path.as_deref())
            .ok_or_else(|| {
                anyhow!(
                    "no tokenizer found near {:?}; pass .tokenizer_path(...)",
                    self.model_root
                )
            })?;
        let input_ids = encode_prompt_padded(&tok_path, prompt, self.txt_seq)?;
        let (out, txt_ids) = {
            let te_guard = self
                .text_encoder
                .lock()
                .map_err(|e| anyhow!("text encoder lock poisoned: {e}"))?;
            let te = te_guard.as_ref().ok_or_else(|| {
                anyhow!("text encoder not loaded (pass .text_encoder_dir(...) on build)")
            })?;
            let cfg_guard = self
                .text_encoder_cfg
                .lock()
                .map_err(|e| anyhow!("text encoder cfg lock poisoned: {e}"))?;
            let te_cfg = cfg_guard
                .as_ref()
                .ok_or_else(|| anyhow!("text encoder config missing"))?;
            encode_flux2_prompt(
                te,
                te_cfg,
                &input_ids,
                self.batch,
                self.txt_seq,
                DEFAULT_TEXT_ENCODER_LAYERS,
            )?
        };
        ensure!(
            out.joint_dim == self.cfg.joint_attention_dim,
            "text encoder joint_dim {} != transformer joint_attention_dim {}",
            out.joint_dim,
            self.cfg.joint_attention_dim
        );
        Ok((out.prompt_embeds, txt_ids))
    }

    fn ensure_text_encoder_compiled(&self) -> Result<()> {
        use crate::{DEFAULT_TEXT_ENCODER_LAYERS, compile_flux2_text_encoder_hir};

        let te = {
            let guard = self
                .text_encoder
                .lock()
                .map_err(|e| anyhow!("text encoder lock poisoned: {e}"))?;
            guard
                .as_ref()
                .ok_or_else(|| anyhow!("text encoder not loaded"))?
                .clone()
        };
        let te_cfg = {
            let guard = self
                .text_encoder_cfg
                .lock()
                .map_err(|e| anyhow!("text encoder cfg lock poisoned: {e}"))?;
            guard
                .as_ref()
                .ok_or_else(|| anyhow!("text encoder config missing"))?
                .clone()
        };

        let mut guard = self
            .text_encoder_compiled
            .lock()
            .map_err(|e| anyhow!("text encoder cache lock poisoned: {e}"))?;
        let recompile = guard.as_ref().is_none_or(|c| {
            c.device != self.device || c.batch != self.batch || c.txt_seq != self.txt_seq
        });
        if recompile {
            eprintln!(
                "[flux2] compiling text encoder HIR on {:?} (txt_seq={})…",
                self.device, self.txt_seq
            );
            let aot = self.aot_cache();
            let (compiled, _) = compile_flux2_text_encoder_hir(
                &te_cfg,
                &te,
                self.batch,
                self.txt_seq,
                DEFAULT_TEXT_ENCODER_LAYERS,
                self.device,
                aot.as_ref(),
            )?;
            *guard = Some(Flux2TextEncoderCache {
                compiled,
                device: self.device,
                batch: self.batch,
                txt_seq: self.txt_seq,
            });
        }
        Ok(())
    }

    fn ensure_cfg_compiled(&self, img_seq: usize) -> Result<()> {
        use crate::compile_flux2_cfg_combine;

        let out_dim = self.cfg.proj_out_dim();
        let mut guard = self
            .cfg_compiled
            .lock()
            .map_err(|e| anyhow!("cfg cache lock poisoned: {e}"))?;
        let recompile = guard.as_ref().is_none_or(|c| {
            c.device != self.device
                || c.batch != self.batch
                || c.img_seq != img_seq
                || c.out_dim != out_dim
        });
        if recompile {
            let aot = self.aot_cache();
            let compiled =
                compile_flux2_cfg_combine(self.batch, img_seq, out_dim, self.device, aot.as_ref())?;
            *guard = Some(Flux2CfgCache {
                compiled,
                device: self.device,
                batch: self.batch,
                img_seq,
                out_dim,
            });
        }
        Ok(())
    }

    fn cfg_combine_compiled(
        &self,
        pos: &[f32],
        neg: &[f32],
        scale: f32,
        img_seq: usize,
    ) -> Result<Vec<f32>> {
        self.ensure_cfg_compiled(img_seq)?;
        let mut guard = self
            .cfg_compiled
            .lock()
            .map_err(|e| anyhow!("cfg cache lock poisoned: {e}"))?;
        let cache = guard
            .as_mut()
            .ok_or_else(|| anyhow!("cfg compile cache missing"))?;
        Ok(cache
            .compiled
            .run(&[("pos", pos), ("neg", neg), ("guidance_scale", &[scale])])
            .remove(0))
    }

    /// Encode via compiled text-encoder HIR on [`Self::device`].
    pub fn encode_prompt_compiled(&self, prompt: &str) -> Result<(Vec<f32>, Vec<f32>)> {
        use crate::{
            DEFAULT_TEXT_ENCODER_LAYERS, encode_prompt_padded, prepare_text_ids,
            resolve_tokenizer_path,
        };

        let te_cfg = self
            .text_encoder_cfg
            .lock()
            .map_err(|e| anyhow!("text encoder cfg lock poisoned: {e}"))?;
        let te_cfg = te_cfg
            .as_ref()
            .ok_or_else(|| anyhow!("text encoder config missing"))?;

        let tok_path = resolve_tokenizer_path(&self.model_root, self.tokenizer_path.as_deref())
            .ok_or_else(|| anyhow!("no tokenizer found near {:?}", self.model_root))?;
        let input_ids = encode_prompt_padded(&tok_path, prompt, self.txt_seq)?;
        let ids_f32: Vec<f32> = input_ids.iter().map(|&x| x as f32).collect();

        self.ensure_text_encoder_compiled()?;
        let mut guard = self
            .text_encoder_compiled
            .lock()
            .map_err(|e| anyhow!("text encoder cache lock poisoned: {e}"))?;
        let cache = guard
            .as_mut()
            .ok_or_else(|| anyhow!("text encoder compile cache missing"))?;
        let embeds = cache
            .compiled
            .run(&[("input_ids", ids_f32.as_slice())])
            .remove(0);
        let joint = te_cfg.hidden_size * DEFAULT_TEXT_ENCODER_LAYERS.len();
        ensure!(
            joint == self.cfg.joint_attention_dim,
            "text encoder joint_dim {joint} != transformer {}",
            self.cfg.joint_attention_dim
        );
        let txt_ids = prepare_text_ids(self.batch, self.txt_seq);
        Ok((embeds, txt_ids))
    }

    /// One denoiser forward: latents + text context → noise prediction.
    pub fn forward(
        &self,
        hidden_states: &[f32],
        encoder_hidden_states: &[f32],
        timestep: &[f32],
        guidance: Option<&[f32]>,
        img_ids: &[f32],
        txt_ids: &[f32],
    ) -> Result<Flux2Output> {
        let noise_pred = self.forward_noise(
            hidden_states,
            encoder_hidden_states,
            timestep,
            guidance,
            img_ids,
            txt_ids,
        )?;
        Ok(Flux2Output {
            noise_pred,
            img_seq: hidden_states.len() / (self.batch * self.cfg.in_channels),
            out_dim: self.cfg.proj_out_dim(),
        })
    }

    /// VAE encode RGB planar `[-1,1]` NCHW → latent (compiled on GPU when enabled).
    pub fn vae_encode_rgb(&self, rgb: &[f32], pixel_h: usize, pixel_w: usize) -> Result<Vec<f32>> {
        if self.uses_compiled_vae() {
            return self.vae_encode_rgb_compiled(rgb, pixel_h, pixel_w);
        }
        let vae = self.vae.as_ref().ok_or_else(|| anyhow!("VAE not loaded"))?;
        let vae_cfg = self
            .vae_cfg
            .as_ref()
            .ok_or_else(|| anyhow!("VAE config missing"))?;
        crate::flux2_vae_encode(vae, vae_cfg, rgb, self.batch, pixel_h, pixel_w)
    }

    fn vae_encode_rgb_compiled(
        &self,
        rgb: &[f32],
        pixel_h: usize,
        pixel_w: usize,
    ) -> Result<Vec<f32>> {
        let vae_cfg = self
            .vae_cfg
            .as_ref()
            .ok_or_else(|| anyhow!("VAE config missing"))?;

        self.ensure_vae_encoder_compiled(pixel_h, pixel_w)?;
        let mut guard = self
            .vae_encoder_compiled
            .lock()
            .map_err(|e| anyhow!("vae encoder cache lock poisoned: {e}"))?;
        let cache = guard
            .as_mut()
            .ok_or_else(|| anyhow!("vae encoder compile cache missing"))?;
        let mut latent = cache.compiled.run(&[("rgb", rgb)]).remove(0);
        if vae_cfg.scaling_factor != 1.0 || vae_cfg.shift_factor != 0.0 {
            for v in &mut latent {
                *v = (*v - vae_cfg.shift_factor) * vae_cfg.scaling_factor;
            }
        }
        Ok(latent)
    }

    /// Encode planar RGB `[-1,1]` NCHW to packed transformer latents.
    pub fn encode_rgb_to_packed(
        &self,
        rgb: &[f32],
        pixel_h: usize,
        pixel_w: usize,
        latent_h: usize,
        latent_w: usize,
        eff_h: usize,
        eff_w: usize,
    ) -> Result<Vec<f32>> {
        use crate::pack_encoded_latents;

        let vae = self
            .vae
            .as_ref()
            .ok_or_else(|| anyhow!("VAE not loaded (required for img2img / edit)"))?;
        let vae_cfg = self
            .vae_cfg
            .as_ref()
            .ok_or_else(|| anyhow!("VAE config missing"))?;
        let stride = vae_cfg.encode_spatial_stride();
        let enc_h = pixel_h / stride;
        let enc_w = pixel_w / stride;
        ensure!(
            enc_h > 0 && enc_w > 0,
            "encoded spatial dims too small for {pixel_h}x{pixel_w}"
        );
        let encoded = self.vae_encode_rgb(rgb, pixel_h, pixel_w)?;
        ensure!(
            encoded.len() == self.batch * vae_cfg.latent_channels * enc_h * enc_w,
            "encoded len {} != expected {}",
            encoded.len(),
            self.batch * vae_cfg.latent_channels * enc_h * enc_w
        );
        pack_encoded_latents(
            vae, vae_cfg, encoded, self.batch, enc_h, enc_w, eff_h, eff_w, latent_h, latent_w,
        )
    }

    pub fn has_vae_encoder(&self) -> bool {
        self.vae.is_some()
    }

    /// img2img: encode source RGB and blend with noise at the strength-derived sigma.
    pub fn prepare_img2img_packed(
        &self,
        rgb: &[f32],
        pixel_h: usize,
        pixel_w: usize,
        latent_h: usize,
        latent_w: usize,
        eff_h: usize,
        eff_w: usize,
        noise: &[f32],
        image_strength: f32,
        num_inference_steps: usize,
    ) -> Result<Vec<f32>> {
        use crate::latent_ops::blend_latents_with_noise;
        use crate::{flow_match_init_timestep, flow_match_sigmas};

        let clean =
            self.encode_rgb_to_packed(rgb, pixel_h, pixel_w, latent_h, latent_w, eff_h, eff_w)?;
        ensure!(clean.len() == noise.len());
        let sigmas = flow_match_sigmas(num_inference_steps);
        let init_step = flow_match_init_timestep(image_strength, num_inference_steps);
        let sigma = sigmas[init_step.min(sigmas.len() - 1)];
        Ok(blend_latents_with_noise(&clean, noise, sigma))
    }

    /// Edit mode: encode reference images into concat conditioning tokens.
    pub fn prepare_edit_conditioning(
        &self,
        images: &[(&[f32], usize, usize)],
        eff_h: usize,
        eff_w: usize,
        latent_h: usize,
        latent_w: usize,
    ) -> Result<crate::Flux2ReferenceConditioning> {
        use crate::{
            Flux2ReferenceConditioning, concat_latent_ids, concat_packed_latents,
            prepare_latent_ids_with_t,
        };

        ensure!(
            !images.is_empty(),
            "edit requires at least one reference image"
        );
        let vae_cfg = self
            .vae_cfg
            .as_ref()
            .ok_or_else(|| anyhow!("VAE config missing"))?;
        let channels = vae_cfg.bn_channels();
        let mut packed_acc: Option<Vec<f32>> = None;
        let mut ids_acc: Option<Vec<f32>> = None;
        let mut total_seq = 0usize;

        for (i, (rgb, ph, pw)) in images.iter().enumerate() {
            let packed =
                self.encode_rgb_to_packed(rgb, *ph, *pw, latent_h, latent_w, eff_h, eff_w)?;
            let seq = packed.len() / (self.batch * channels);
            total_seq += seq;
            let ids = prepare_latent_ids_with_t(self.batch, latent_h, latent_w, 10 + 10 * i as i32);
            packed_acc = Some(match packed_acc {
                Some(prev) => concat_packed_latents(&prev, &packed, self.batch, channels),
                None => packed,
            });
            ids_acc = Some(match ids_acc {
                Some(prev) => concat_latent_ids(&prev, &ids, self.batch),
                None => ids,
            });
        }

        Ok(Flux2ReferenceConditioning {
            packed: packed_acc.unwrap(),
            img_ids: ids_acc.unwrap(),
            seq: total_seq,
        })
    }

    /// Denoiser noise prediction (compiled on [`Self::device`] when not CPU-native).
    pub fn forward_noise(
        &self,
        hidden_states: &[f32],
        encoder_hidden_states: &[f32],
        timestep: &[f32],
        guidance: Option<&[f32]>,
        img_ids: &[f32],
        txt_ids: &[f32],
    ) -> Result<Vec<f32>> {
        if self.uses_compiled_denoiser() {
            self.forward_noise_compiled(
                hidden_states,
                encoder_hidden_states,
                timestep,
                guidance,
                img_ids,
                txt_ids,
            )
        } else {
            self.forward_noise_native(
                hidden_states,
                encoder_hidden_states,
                timestep,
                guidance,
                img_ids,
                txt_ids,
            )
        }
    }

    /// Native CPU reference forward (no IR compile).
    pub fn forward_noise_native(
        &self,
        hidden_states: &[f32],
        encoder_hidden_states: &[f32],
        timestep: &[f32],
        guidance: Option<&[f32]>,
        img_ids: &[f32],
        txt_ids: &[f32],
    ) -> Result<Vec<f32>> {
        use crate::{Flux2ForwardInput, flux2_transformer_forward};

        flux2_transformer_forward(
            &self.model,
            &self.cfg,
            Flux2ForwardInput {
                hidden_states,
                encoder_hidden_states,
                timestep,
                timestep_target: None,
                guidance,
                img_ids,
                txt_ids,
                batch: self.batch,
                img_seq: hidden_states.len() / (self.batch * self.cfg.in_channels),
                txt_seq: self.txt_seq,
            },
        )
    }

    /// Native forward with dual-time embedding (flow-map).
    pub fn forward_noise_dual_native(
        &self,
        hidden_states: &[f32],
        encoder_hidden_states: &[f32],
        timestep: &[f32],
        timestep_target: &[f32],
        guidance: Option<&[f32]>,
        img_ids: &[f32],
        txt_ids: &[f32],
    ) -> Result<Vec<f32>> {
        use crate::{Flux2ForwardInput, flux2_transformer_forward};

        flux2_transformer_forward(
            &self.model,
            &self.cfg,
            Flux2ForwardInput {
                hidden_states,
                encoder_hidden_states,
                timestep,
                timestep_target: Some(timestep_target),
                guidance,
                img_ids,
                txt_ids,
                batch: self.batch,
                img_seq: hidden_states.len() / (self.batch * self.cfg.in_channels),
                txt_seq: self.txt_seq,
            },
        )
    }

    /// Compiled HIR denoiser on [`Self::device`] (Metal / MLX / CUDA / CPU).
    pub fn forward_noise_compiled(
        &self,
        hidden_states: &[f32],
        encoder_hidden_states: &[f32],
        timestep: &[f32],
        guidance: Option<&[f32]>,
        img_ids: &[f32],
        txt_ids: &[f32],
    ) -> Result<Vec<f32>> {
        use crate::host_temb;

        self.ensure_denoiser_compiled(img_ids, txt_ids)?;
        let temb = host_temb(&self.model, &self.cfg, timestep, guidance)?;
        let mut guard = self
            .denoiser
            .lock()
            .map_err(|e| anyhow!("denoiser cache lock poisoned: {e}"))?;
        let cache = guard
            .as_mut()
            .ok_or_else(|| anyhow!("denoiser compile cache missing after ensure"))?;
        Ok(cache
            .compiled
            .run(&[
                ("hidden", hidden_states),
                ("encoder", encoder_hidden_states),
                ("temb", temb.as_slice()),
            ])
            .remove(0))
    }

    /// Compiled forward with dual-time temb (flow-map).
    pub fn forward_noise_dual_compiled(
        &self,
        hidden_states: &[f32],
        encoder_hidden_states: &[f32],
        timestep: &[f32],
        timestep_target: &[f32],
        guidance: Option<&[f32]>,
        img_ids: &[f32],
        txt_ids: &[f32],
    ) -> Result<Vec<f32>> {
        use crate::host_temb_dual;

        self.ensure_denoiser_compiled(img_ids, txt_ids)?;
        let temb = host_temb_dual(&self.model, &self.cfg, timestep, timestep_target, guidance)?;
        let mut guard = self
            .denoiser
            .lock()
            .map_err(|e| anyhow!("denoiser cache lock poisoned: {e}"))?;
        let cache = guard
            .as_mut()
            .ok_or_else(|| anyhow!("denoiser compile cache missing after ensure"))?;
        Ok(cache
            .compiled
            .run(&[
                ("hidden", hidden_states),
                ("encoder", encoder_hidden_states),
                ("temb", temb.as_slice()),
            ])
            .remove(0))
    }

    /// Classifier-free guidance: positive + negative text, then
    /// `neg + cfg_scale * (pos - neg)` on the noise prediction.
    pub fn forward_cfg(
        &self,
        hidden_states: &[f32],
        pos_encoder: &[f32],
        neg_encoder: &[f32],
        timestep: &[f32],
        guidance: Option<&[f32]>,
        img_ids: &[f32],
        pos_txt_ids: &[f32],
        neg_txt_ids: &[f32],
        cfg_scale: f32,
    ) -> Result<Flux2Output> {
        use crate::cfg_combine;

        let pos = self.forward_noise(
            hidden_states,
            pos_encoder,
            timestep,
            guidance,
            img_ids,
            pos_txt_ids,
        )?;
        if cfg_scale <= 1.0 {
            return Ok(Flux2Output {
                noise_pred: pos,
                img_seq: hidden_states.len() / (self.batch * self.cfg.in_channels),
                out_dim: self.cfg.proj_out_dim(),
            });
        }
        let neg = self.forward_noise(
            hidden_states,
            neg_encoder,
            timestep,
            guidance,
            img_ids,
            neg_txt_ids,
        )?;
        let img_seq = hidden_states.len() / (self.batch * self.cfg.in_channels);
        let noise_pred = if self.uses_compiled_denoiser() {
            self.cfg_combine_compiled(&pos, &neg, cfg_scale, img_seq)?
        } else {
            cfg_combine(&pos, &neg, cfg_scale)
        };
        Ok(Flux2Output {
            noise_pred,
            img_seq,
            out_dim: self.cfg.proj_out_dim(),
        })
    }

    /// Tokenize and encode positive + optional negative prompts.
    #[allow(clippy::type_complexity)]
    pub fn encode_prompt_pair(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
    ) -> Result<(Vec<f32>, Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>)> {
        let (pos, pos_ids) = self.encode_prompt(prompt)?;
        let (neg, neg_ids) = match negative_prompt {
            Some(n) => {
                let (e, ids) = self.encode_prompt(n)?;
                (Some(e), Some(ids))
            }
            None => (None, None),
        };
        if self.drop_text_encoder_after_encode {
            self.drop_text_encoder_weights()?;
        }
        Ok((pos, pos_ids, neg, neg_ids))
    }

    pub fn vae_config(&self) -> Option<&crate::Flux2VaeConfig> {
        self.vae_cfg.as_ref()
    }

    /// Decode denoised packed latents to interleaved RGB u8 (HWC) and pixel `(height, width)`.
    pub fn decode_to_rgb(
        &self,
        packed_latents: &[f32],
        img_ids: &[f32],
        latent_h: usize,
        latent_w: usize,
    ) -> Result<(Vec<u8>, u32, u32)> {
        if self.uses_compiled_vae() {
            return self.decode_to_rgb_compiled(packed_latents, img_ids, latent_h, latent_w);
        }
        self.decode_to_rgb_native(packed_latents, img_ids, latent_h, latent_w)
    }

    /// Native CPU decode (unpack / BN / unpatchify + VAE decoder).
    pub fn decode_to_rgb_native(
        &self,
        packed_latents: &[f32],
        img_ids: &[f32],
        latent_h: usize,
        latent_w: usize,
    ) -> Result<(Vec<u8>, u32, u32)> {
        use crate::{flux2_decode_packed_latents, flux2_rgb_to_u8};

        let vae = self
            .vae
            .as_ref()
            .ok_or_else(|| anyhow!("VAE not loaded (place vae/ next to weights)"))?;
        let vae_cfg = self
            .vae_cfg
            .as_ref()
            .ok_or_else(|| anyhow!("VAE config missing"))?;
        let packed_channels = self.cfg.in_channels;
        let img_seq = img_ids.len() / (self.batch * 4);
        let rgb = flux2_decode_packed_latents(
            vae,
            vae_cfg,
            packed_latents,
            img_ids,
            self.batch,
            img_seq,
            packed_channels,
            latent_h,
            latent_w,
        )?;
        let up_stages = vae_cfg.block_out_channels.len().saturating_sub(1);
        let scale = 2usize.pow(up_stages as u32 + 1);
        let h_px = latent_h * scale;
        let w_px = latent_w * scale;
        let u8 = flux2_rgb_to_u8(&rgb, self.batch, 3, h_px, w_px);
        Ok((u8, h_px as u32, w_px as u32))
    }

    /// Compiled VAE decoder on [`Self::device`] (unpack/BN/unpatchify stay on CPU).
    pub fn decode_to_rgb_compiled(
        &self,
        packed_latents: &[f32],
        img_ids: &[f32],
        latent_h: usize,
        latent_w: usize,
    ) -> Result<(Vec<u8>, u32, u32)> {
        use crate::{
            denorm_patchified_latents, flux2_rgb_to_u8, unpack_latents_with_ids, unpatchify_latents,
        };

        let vae_cfg = self
            .vae_cfg
            .as_ref()
            .ok_or_else(|| anyhow!("VAE config missing"))?;
        let vae = self.vae.as_ref().ok_or_else(|| anyhow!("VAE not loaded"))?;
        let packed_channels = self.cfg.in_channels;
        let img_seq = img_ids.len() / (self.batch * 4);
        let spatial = unpack_latents_with_ids(
            packed_latents,
            img_ids,
            self.batch,
            img_seq,
            packed_channels,
            latent_h,
            latent_w,
        )?;
        let denorm = denorm_patchified_latents(
            &spatial,
            &vae.bn_running_mean,
            &vae.bn_running_var,
            vae_cfg.batch_norm_eps,
        );
        let mut latents =
            unpatchify_latents(&denorm, self.batch, packed_channels, latent_h, latent_w);
        if vae_cfg.scaling_factor != 1.0 || vae_cfg.shift_factor != 0.0 {
            for v in &mut latents {
                *v = *v / vae_cfg.scaling_factor + vae_cfg.shift_factor;
            }
        }
        let h2 = latent_h * 2;
        let w2 = latent_w * 2;

        self.ensure_vae_compiled(h2, w2)?;
        let mut guard = self
            .vae_compiled
            .lock()
            .map_err(|e| anyhow!("vae cache lock poisoned: {e}"))?;
        let cache = guard
            .as_mut()
            .ok_or_else(|| anyhow!("vae compile cache missing"))?;
        let rgb = cache
            .compiled
            .run(&[("latents", latents.as_slice())])
            .remove(0);

        let up_stages = vae_cfg.block_out_channels.len().saturating_sub(1);
        let scale = 2usize.pow(up_stages as u32 + 1);
        let h_px = latent_h * scale;
        let w_px = latent_w * scale;
        let u8 = flux2_rgb_to_u8(&rgb, self.batch, 3, h_px, w_px);
        Ok((u8, h_px as u32, w_px as u32))
    }

    fn ensure_vae_compiled(&self, h: usize, w: usize) -> Result<()> {
        use crate::compile_flux2_vae_hir;

        let vae = self.vae.as_ref().ok_or_else(|| anyhow!("VAE not loaded"))?;
        let vae_cfg = self
            .vae_cfg
            .as_ref()
            .ok_or_else(|| anyhow!("VAE config missing"))?;

        let mut guard = self
            .vae_compiled
            .lock()
            .map_err(|e| anyhow!("vae cache lock poisoned: {e}"))?;
        let recompile = guard.as_ref().is_none_or(|c| {
            c.device != self.device || c.batch != self.batch || c.h != h || c.w != w
        });
        if recompile {
            let aot = self.aot_cache();
            let (compiled, _) =
                compile_flux2_vae_hir(vae_cfg, vae, self.batch, h, w, self.device, aot.as_ref())?;
            *guard = Some(Flux2VaeCache {
                compiled,
                device: self.device,
                batch: self.batch,
                h,
                w,
            });
        }
        Ok(())
    }

    fn ensure_vae_encoder_compiled(&self, h: usize, w: usize) -> Result<()> {
        use crate::compile_flux2_vae_encoder_hir;

        let vae = self.vae.as_ref().ok_or_else(|| anyhow!("VAE not loaded"))?;
        let vae_cfg = self
            .vae_cfg
            .as_ref()
            .ok_or_else(|| anyhow!("VAE config missing"))?;

        let mut guard = self
            .vae_encoder_compiled
            .lock()
            .map_err(|e| anyhow!("vae encoder cache lock poisoned: {e}"))?;
        let recompile = guard.as_ref().is_none_or(|c| {
            c.device != self.device || c.batch != self.batch || c.h != h || c.w != w
        });
        if recompile {
            let aot = self.aot_cache();
            let (compiled, _) = compile_flux2_vae_encoder_hir(
                vae_cfg,
                vae,
                self.batch,
                h,
                w,
                self.device,
                aot.as_ref(),
            )?;
            *guard = Some(Flux2VaeEncoderCache {
                compiled,
                device: self.device,
                batch: self.batch,
                h,
                w,
            });
        }
        Ok(())
    }
}
