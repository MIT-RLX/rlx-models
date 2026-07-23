// RLX models — OpenAI-compatible multi-model server.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Load chat LM engines for [`rlx_openai`](crate).

use anyhow::{Context, Result, bail};
use rlx_runtime::Device;
use rlx_serve::Engine;
use rlx_serve::engine::SingleEngine;
use rlx_text::{auto_chat_template, load_tokenizer};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

/// One `--engine …` block from the CLI.
#[derive(Debug, Clone)]
pub struct EngineSpec {
    pub kind: String,
    pub weights: PathBuf,
    pub tokenizer_dir: Option<PathBuf>,
    pub model_id: String,
    pub device: String,
    pub eos: Vec<u32>,
    pub prompt_cache_mb: usize,
    pub continuous: bool,
    pub fused: bool,
    pub max_batch: usize,
    pub batch_tokens: usize,
    pub kv_bits: u32,
}

impl EngineSpec {
    pub fn new(kind: impl Into<String>) -> Self {
        let kind = kind.into();
        let model_id = kind.clone();
        Self {
            kind,
            weights: PathBuf::new(),
            tokenizer_dir: None,
            model_id,
            device: "cpu".into(),
            eos: Vec::new(),
            prompt_cache_mb: 0,
            continuous: false,
            fused: false,
            max_batch: 8,
            batch_tokens: 2048,
            kv_bits: 0,
        }
    }

    fn tok_path(&self) -> PathBuf {
        let p = self
            .tokenizer_dir
            .clone()
            .unwrap_or_else(|| self.weights.join("tokenizer.json"));
        if p.is_dir() {
            p.join("tokenizer.json")
        } else {
            p
        }
    }

    fn template_dir(&self) -> PathBuf {
        self.tokenizer_dir
            .clone()
            .unwrap_or_else(|| self.weights.clone())
    }
}

/// Human-readable list of `--engine` kinds compiled into this binary.
pub fn enabled_engine_kinds() -> Vec<&'static str> {
    let mut v = vec!["qwen3"];
    if cfg!(feature = "laguna") {
        v.push("laguna");
    }
    if cfg!(feature = "qwen35") {
        v.push("qwen35");
    }
    if cfg!(feature = "gemma") {
        v.push("gemma");
    }
    if cfg!(feature = "llama32") {
        v.push("llama32");
    }
    if cfg!(feature = "lfm") {
        v.push("lfm");
    }
    v
}

pub fn load_engine(spec: &EngineSpec) -> Result<Arc<dyn Engine>> {
    if spec.weights.as_os_str().is_empty() {
        bail!("--engine {} requires --weights PATH", spec.kind);
    }
    match spec.kind.as_str() {
        "qwen3" => load_qwen3(spec),
        "laguna" => load_laguna(spec),
        "qwen35" => load_qwen35(spec),
        "gemma" => load_gemma(spec),
        "llama32" => load_llama32(spec),
        "lfm" => load_lfm(spec),
        other => bail!(
            "unknown --engine {other:?}; enabled in this build: {}",
            enabled_engine_kinds().join(", ")
        ),
    }
}

fn parse_device(s: &str) -> Result<Device> {
    Device::from_str(s).map_err(|e| anyhow::anyhow!("{e}"))
}

fn load_tokenizer_and_template(
    tok_path: &Path,
    template_dir: &Path,
) -> Result<(rlx_text::TokenizerHandle, Option<rlx_text::ChatTemplate>)> {
    let tokenizer =
        load_tokenizer(tok_path).with_context(|| format!("loading tokenizer {tok_path:?}"))?;
    let chat_template = auto_chat_template(template_dir).ok();
    if chat_template.is_none() {
        eprintln!("[rlx-openai] no chat template under {template_dir:?}; plain roles");
    }
    Ok((tokenizer, chat_template))
}

fn single_lm(
    runner: Box<dyn rlx_runtime::LmRunner>,
    spec: &EngineSpec,
    tokenizer: rlx_text::TokenizerHandle,
    chat_template: Option<rlx_text::ChatTemplate>,
) -> Arc<dyn Engine> {
    let mut engine = SingleEngine::new(
        runner,
        tokenizer,
        chat_template,
        spec.eos.clone(),
        spec.model_id.clone(),
    );
    if spec.prompt_cache_mb > 0 {
        engine = engine.with_prompt_cache(spec.prompt_cache_mb << 20);
        eprintln!(
            "[rlx-openai] {}: prompt cache {} MiB",
            spec.model_id, spec.prompt_cache_mb
        );
    }
    Arc::new(engine)
}

fn load_qwen3(spec: &EngineSpec) -> Result<Arc<dyn Engine>> {
    use rlx_qwen3::Qwen3Runner;
    use rlx_runtime::quantized_kv::KvQuant;
    use rlx_serve::{BatchedEngine, FusedBatchRunner, RunnerBatchRunner};

    let device = parse_device(&spec.device)?;
    eprintln!(
        "[rlx-openai] loading qwen3 {} on {device:?} as {}",
        spec.weights.display(),
        spec.model_id
    );
    let runner = Qwen3Runner::builder()
        .weights(spec.weights.clone())
        .device(device)
        .build()
        .context("building Qwen3Runner")?;
    let (tokenizer, chat_template) =
        load_tokenizer_and_template(&spec.tok_path(), &spec.template_dir())?;

    if spec.continuous && spec.fused {
        let generator = runner
            .into_generator()
            .context("--fused requires F32 weights (not packed/quantized)")?;
        let kv_quant = match spec.kv_bits {
            0 => None,
            16 => Some(KvQuant::F16),
            8 => Some(KvQuant::Q8_0),
            5 => Some(KvQuant::Q5_0),
            4 => Some(KvQuant::Q4_0),
            other => bail!("--kv-bits must be one of 0,4,5,8,16 (got {other})"),
        };
        eprintln!(
            "[rlx-openai] {}: continuous+fused max_batch={} batch_tokens={} kv_bits={}",
            spec.model_id, spec.max_batch, spec.batch_tokens, spec.kv_bits
        );
        return Ok(Arc::new(BatchedEngine::new(
            Box::new(FusedBatchRunner::with_kv_quant(generator, kv_quant)),
            tokenizer,
            chat_template,
            spec.eos.clone(),
            spec.model_id.clone(),
            spec.batch_tokens,
            spec.max_batch,
        )));
    }
    if spec.continuous {
        let br = RunnerBatchRunner::new(Box::new(runner)).context("RunnerBatchRunner")?;
        eprintln!(
            "[rlx-openai] {}: continuous batching max_batch={} batch_tokens={}",
            spec.model_id, spec.max_batch, spec.batch_tokens
        );
        return Ok(Arc::new(BatchedEngine::new(
            Box::new(br),
            tokenizer,
            chat_template,
            spec.eos.clone(),
            spec.model_id.clone(),
            spec.batch_tokens,
            spec.max_batch,
        )));
    }
    Ok(single_lm(
        Box::new(runner),
        spec,
        tokenizer,
        chat_template,
    ))
}

fn load_laguna(spec: &EngineSpec) -> Result<Arc<dyn Engine>> {
    #[cfg(not(feature = "laguna"))]
    {
        let _ = spec;
        bail!("rebuild with --features laguna for --engine laguna");
    }
    #[cfg(feature = "laguna")]
    {
        use rlx_laguna::{DeviceMatmul, LagunaChat, LagunaEngine, LagunaPackedRunner, parse_device};

        let tok_dir = spec.tokenizer_dir.clone().ok_or_else(|| {
            anyhow::anyhow!("--engine laguna requires --tokenizer-dir DIR")
        })?;
        eprintln!(
            "[rlx-openai] loading laguna {} as {} (device={})",
            spec.weights.display(),
            spec.model_id,
            spec.device
        );
        let runner = LagunaPackedRunner::from_gguf_packed(&spec.weights)?;
        let chat = LagunaChat::from_dir(&tok_dir)?;
        let accel = match parse_device(&spec.device)? {
            Some(d) => Some(DeviceMatmul::try_new(d)?),
            None => None,
        };
        Ok(Arc::new(LagunaEngine::new(
            runner,
            chat,
            accel,
            spec.model_id.clone(),
        )))
    }
}

fn load_qwen35(spec: &EngineSpec) -> Result<Arc<dyn Engine>> {
    #[cfg(not(feature = "qwen35"))]
    {
        let _ = spec;
        bail!("rebuild with --features qwen35 for --engine qwen35");
    }
    #[cfg(feature = "qwen35")]
    {
        let device = parse_device(&spec.device)?;
        eprintln!(
            "[rlx-openai] loading qwen35 {} on {device:?} as {}",
            spec.weights.display(),
            spec.model_id
        );
        let runner = rlx_qwen35::Qwen35Runner::builder()
            .weights(spec.weights.clone())
            .device(device)
            .build()
            .context("building Qwen35Runner")?;
        let (tokenizer, chat_template) =
            load_tokenizer_and_template(&spec.tok_path(), &spec.template_dir())?;
        Ok(single_lm(
            Box::new(runner),
            spec,
            tokenizer,
            chat_template,
        ))
    }
}

fn load_gemma(spec: &EngineSpec) -> Result<Arc<dyn Engine>> {
    #[cfg(not(feature = "gemma"))]
    {
        let _ = spec;
        bail!("rebuild with --features gemma for --engine gemma");
    }
    #[cfg(feature = "gemma")]
    {
        let device = parse_device(&spec.device)?;
        eprintln!(
            "[rlx-openai] loading gemma {} on {device:?} as {}",
            spec.weights.display(),
            spec.model_id
        );
        let runner = rlx_gemma::GemmaRunner::builder()
            .weights(spec.weights.clone())
            .device(device)
            .build()
            .context("building GemmaRunner")?;
        let (tokenizer, chat_template) =
            load_tokenizer_and_template(&spec.tok_path(), &spec.template_dir())?;
        Ok(single_lm(
            Box::new(runner),
            spec,
            tokenizer,
            chat_template,
        ))
    }
}

fn load_llama32(spec: &EngineSpec) -> Result<Arc<dyn Engine>> {
    #[cfg(not(feature = "llama32"))]
    {
        let _ = spec;
        bail!("rebuild with --features llama32 for --engine llama32");
    }
    #[cfg(feature = "llama32")]
    {
        let device = parse_device(&spec.device)?;
        eprintln!(
            "[rlx-openai] loading llama32 {} on {device:?} as {}",
            spec.weights.display(),
            spec.model_id
        );
        let runner = rlx_llama32::Llama32Runner::builder()
            .weights(spec.weights.clone())
            .device(device)
            .build()
            .context("building Llama32Runner")?;
        let (tokenizer, chat_template) =
            load_tokenizer_and_template(&spec.tok_path(), &spec.template_dir())?;
        Ok(single_lm(
            Box::new(runner),
            spec,
            tokenizer,
            chat_template,
        ))
    }
}

fn load_lfm(spec: &EngineSpec) -> Result<Arc<dyn Engine>> {
    #[cfg(not(feature = "lfm"))]
    {
        let _ = spec;
        bail!("rebuild with --features lfm for --engine lfm");
    }
    #[cfg(feature = "lfm")]
    {
        let device = parse_device(&spec.device)?;
        eprintln!(
            "[rlx-openai] loading lfm {} on {device:?} as {}",
            spec.weights.display(),
            spec.model_id
        );
        let runner = rlx_lfm::LfmRunner::builder()
            .weights(spec.weights.clone())
            .device(device)
            .build()
            .context("building LfmRunner")?;
        let (tokenizer, chat_template) =
            load_tokenizer_and_template(&spec.tok_path(), &spec.template_dir())?;
        Ok(single_lm(
            Box::new(runner),
            spec,
            tokenizer,
            chat_template,
        ))
    }
}
