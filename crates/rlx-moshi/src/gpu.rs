//! Candle / Kyutai `moshi` GPU LM backend (Metal, CUDA).

use crate::checkpoint::MoshiCheckpoint;
use crate::config::{GenerateConfig, LmConfig, MoshiVariant};
use crate::generate::UNGENERATED;
use crate::sampling::LogitsProcessor;
use anyhow::{Context, Result, ensure};
use rlx_runtime::Device;
use std::path::Path;
use std::sync::Arc;

pub struct GpuLm {
    model: Arc<moshi::lm::LmModel>,
    cfg: LmConfig,
    candle_device: candle::Device,
}

pub struct GpuGenerateState {
    inner: moshi::lm_generate_multistream::State,
    gen_cfg: GenerateConfig,
}

impl GpuLm {
    pub fn open(
        weights_path: &Path,
        variant: MoshiVariant,
        checkpoint: MoshiCheckpoint,
        device: Device,
    ) -> Result<Self> {
        ensure!(
            device != Device::Cpu,
            "gpu-lm backend requires metal/cuda device, got {device:?}"
        );
        ensure!(
            checkpoint.gpu_loadable(),
            "checkpoint {:?} is not supported on GPU LM",
            checkpoint
        );
        let candle_device = rlx_device_to_candle(device)?;
        let dtype = if candle_device.is_cuda() || candle_device.is_metal() {
            candle::DType::BF16
        } else {
            candle::DType::F32
        };
        let moshi_cfg = moshi_config(variant);
        let model = if checkpoint.is_mlx() {
            crate::mlx_weights::load_lm_model_mlx(
                moshi_cfg,
                weights_path,
                checkpoint,
                dtype,
                &candle_device,
            )
            .with_context(|| {
                format!(
                    "load mlx moshi LM on {device:?} from {}",
                    weights_path.display()
                )
            })?
        } else {
            moshi::lm::load_lm_model(moshi_cfg, weights_path, dtype, &candle_device).with_context(
                || {
                    format!(
                        "load moshi LM on {device:?} from {}",
                        weights_path.display()
                    )
                },
            )?
        };
        let cfg = variant.lm_config();
        Ok(Self {
            model: Arc::new(model),
            cfg,
            candle_device,
        })
    }

    pub fn config(&self) -> &LmConfig {
        &self.cfg
    }

    pub fn candle_device(&self) -> &candle::Device {
        &self.candle_device
    }

    pub fn new_generate_state(
        &self,
        max_steps: usize,
        text_lp: LogitsProcessor,
        audio_lp: LogitsProcessor,
        gen_cfg: &GenerateConfig,
    ) -> Result<GpuGenerateState> {
        let moshi_gen_cfg = moshi_gen_config(gen_cfg);
        let text_lp_c = to_candle_lp(&text_lp);
        let audio_lp_c = to_candle_lp(&audio_lp);
        let state = moshi::lm_generate_multistream::State::new(
            (*self.model).clone(),
            max_steps,
            audio_lp_c,
            text_lp_c,
            None,
            None,
            None,
            moshi_gen_cfg,
        );
        Ok(GpuGenerateState {
            inner: state,
            gen_cfg: gen_cfg.clone(),
        })
    }
}

impl GpuGenerateState {
    pub fn config(&self) -> &GenerateConfig {
        &self.gen_cfg
    }

    pub fn step_idx(&self) -> usize {
        self.inner.step_idx()
    }

    pub fn text_tokens(&self) -> &[u32] {
        self.inner.text_tokens(false)
    }

    pub fn step(&mut self, text_token: u32, input_audio: &[u32]) -> Result<u32> {
        let sampled = self
            .inner
            .step_(Some(text_token), input_audio, None, None, None)
            .context("moshi gpu step")?;
        Ok(sampled)
    }

    pub fn last_audio_frame(&self) -> Option<Vec<u32>> {
        let cfg = self.inner.config();
        let tokens = self.inner.last_audio_tokens()?;
        if tokens[..cfg.generated_audio_codebooks]
            .iter()
            .any(|&t| t == UNGENERATED || t >= cfg.audio_pad_token())
        {
            return None;
        }
        Some(tokens[..cfg.generated_audio_codebooks].to_vec())
    }

    pub fn reset(
        &mut self,
        lm: &GpuLm,
        max_steps: usize,
        text_lp: LogitsProcessor,
        audio_lp: LogitsProcessor,
    ) -> Result<()> {
        *self = lm.new_generate_state(max_steps, text_lp, audio_lp, &self.gen_cfg)?;
        Ok(())
    }
}

fn moshi_config(variant: MoshiVariant) -> moshi::lm::Config {
    match variant {
        MoshiVariant::Moshiko | MoshiVariant::Moshika => moshi::lm::Config::v0_1_streaming(8),
        MoshiVariant::MoshikoOneWay | MoshiVariant::MoshikaOneWay => moshi::lm::Config::v0_1(),
    }
}

fn moshi_gen_config(gen_cfg: &GenerateConfig) -> moshi::lm_generate_multistream::Config {
    moshi::lm_generate_multistream::Config {
        generated_audio_codebooks: gen_cfg.generated_audio_codebooks,
        input_audio_codebooks: gen_cfg.input_audio_codebooks,
        audio_vocab_size: gen_cfg.audio_vocab_size,
        acoustic_delay: gen_cfg.acoustic_delay,
        text_pad_token: gen_cfg.text_pad_token,
        text_eop_token: gen_cfg.text_eop_token,
        text_start_token: gen_cfg.text_start_token,
    }
}

fn to_candle_lp(lp: &LogitsProcessor) -> candle_transformers::generation::LogitsProcessor {
    let k = lp.top_k();
    let sampling = if k > 0 {
        candle_transformers::generation::Sampling::TopK {
            k,
            temperature: lp.temperature(),
        }
    } else {
        candle_transformers::generation::Sampling::All {
            temperature: lp.temperature(),
        }
    };
    candle_transformers::generation::LogitsProcessor::from_sampling(lp.seed(), sampling)
}

pub fn rlx_device_to_candle(device: Device) -> Result<candle::Device> {
    Ok(match device {
        Device::Cpu => candle::Device::Cpu,
        Device::Metal => candle::Device::new_metal(0).context("Metal device")?,
        Device::Cuda => candle::Device::new_cuda(0).context("CUDA device")?,
        Device::Mlx => candle::Device::new_metal(0).context("MLX tag → Candle Metal")?,
        other => anyhow::bail!("gpu-lm unsupported device {other:?}"),
    })
}

pub fn gpu_lm_available(device: Device) -> bool {
    match device {
        Device::Metal => candle::utils::metal_is_available(),
        Device::Cuda => candle::utils::cuda_is_available(),
        Device::Mlx => candle::utils::metal_is_available(),
        _ => false,
    }
}
