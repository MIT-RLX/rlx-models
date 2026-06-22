//! Unified Moshi LM backend — eager CPU, Candle GPU, optional compiled path.

use crate::checkpoint::MoshiCheckpoint;
use crate::config::{GenerateConfig, MoshiVariant};
use crate::generate::GenerateState;
use crate::lm::LmModel;
use crate::sampling::LogitsProcessor;
use crate::weights::{open_lm, open_lm_from_weights};
use anyhow::{Result, ensure};
use rlx_runtime::Device;
use std::path::Path;

#[cfg(feature = "gpu-lm")]
use crate::gpu::{GpuGenerateState, GpuLm, gpu_lm_available, rlx_device_to_candle};

#[cfg(feature = "compiled-lm")]
pub use crate::compiled_lm::{CompiledGenerateState, CompiledLm, compiled_lm_available};

/// How to load and run the Moshi temporal + DepFormer stack.
pub enum MoshiLm {
    Eager(LmModel),
    #[cfg(feature = "gpu-lm")]
    Gpu(GpuLm),
}

/// Per-generation autoregressive state (CPU or GPU).
pub enum MoshiGenState {
    Eager(GenerateState),
    #[cfg(feature = "gpu-lm")]
    Gpu(GpuGenerateState),
}

impl MoshiLm {
    pub fn open(
        model_dir: &Path,
        variant: MoshiVariant,
        checkpoint: MoshiCheckpoint,
        device: Device,
    ) -> Result<Self> {
        if checkpoint.is_mlx() && device == Device::Cpu {
            let weights_path = checkpoint.lm_weights_path(model_dir);
            let cfg = variant.lm_config();
            let weights =
                crate::mlx_weights::load_eager_weight_map(&weights_path, checkpoint, &cfg)?;
            return Ok(Self::Eager(open_lm_from_weights(cfg, weights)?));
        }
        if checkpoint.is_mlx() {
            #[cfg(not(feature = "gpu-lm"))]
            anyhow::bail!(
                "checkpoint {:?} requires `gpu-lm` or `--device cpu`",
                checkpoint
            );
        }
        let weights_path = checkpoint.lm_weights_path(model_dir);
        let cfg = variant.lm_config();

        #[cfg(feature = "compiled-lm")]
        if device != Device::Cpu && compiled_lm_available(device) && checkpoint.gpu_loadable() {
            return Ok(Self::Gpu(CompiledLm::open(
                &weights_path,
                variant,
                checkpoint,
                device,
            )?));
        }

        #[cfg(all(feature = "gpu-lm", not(feature = "compiled-lm")))]
        if device != Device::Cpu && gpu_lm_available(device) && checkpoint.gpu_loadable() {
            return Ok(Self::Gpu(GpuLm::open(
                &weights_path,
                variant,
                checkpoint,
                device,
            )?));
        }

        if device != Device::Cpu && checkpoint.gpu_loadable() {
            #[cfg(feature = "gpu-lm")]
            if gpu_lm_available(device) {
                return Ok(Self::Gpu(GpuLm::open(
                    &weights_path,
                    variant,
                    checkpoint,
                    device,
                )?));
            }
            #[cfg(not(feature = "gpu-lm"))]
            anyhow::bail!(
                "device {device:?} requested but crate built without `gpu-lm` — rebuild with --features gpu-lm,metal"
            );
            #[cfg(feature = "gpu-lm")]
            anyhow::bail!("device {device:?} not available for Moshi GPU LM");
        }

        let lm = if checkpoint.is_gguf() {
            let weights = crate::gguf::load_gguf_weight_map(&weights_path, &cfg)?;
            open_lm_from_weights(cfg, weights)?
        } else {
            open_lm(model_dir, cfg)?
        };
        Ok(Self::Eager(lm))
    }

    pub fn config(&self) -> &crate::config::LmConfig {
        match self {
            Self::Eager(m) => m.config(),
            #[cfg(feature = "gpu-lm")]
            Self::Gpu(m) => m.config(),
        }
    }

    pub fn reset_state(&mut self) {
        match self {
            Self::Eager(m) => m.reset_state(),
            #[cfg(feature = "gpu-lm")]
            Self::Gpu(_) => {}
        }
    }

    pub fn new_gen_state(
        &self,
        max_steps: usize,
        text_lp: LogitsProcessor,
        audio_lp: LogitsProcessor,
        gen_cfg: GenerateConfig,
    ) -> Result<MoshiGenState> {
        match self {
            Self::Eager(_) => Ok(MoshiGenState::Eager(GenerateState::new(
                max_steps, text_lp, audio_lp, gen_cfg,
            ))),
            #[cfg(feature = "gpu-lm")]
            Self::Gpu(g) => Ok(MoshiGenState::Gpu(
                g.new_generate_state(max_steps, text_lp, audio_lp, &gen_cfg)?,
            )),
        }
    }
}

impl MoshiGenState {
    pub fn config(&self) -> &GenerateConfig {
        match self {
            Self::Eager(s) => s.config(),
            #[cfg(feature = "gpu-lm")]
            Self::Gpu(s) => s.config(),
        }
    }

    pub fn step_idx(&self) -> usize {
        match self {
            Self::Eager(s) => s.step_idx(),
            #[cfg(feature = "gpu-lm")]
            Self::Gpu(s) => s.step_idx(),
        }
    }

    pub fn text_tokens(&self) -> &[u32] {
        match self {
            Self::Eager(s) => s.text_tokens(),
            #[cfg(feature = "gpu-lm")]
            Self::Gpu(s) => s.text_tokens(),
        }
    }

    pub fn step(&mut self, lm: &mut MoshiLm, text_token: u32, input_audio: &[u32]) -> Result<u32> {
        match (self, lm) {
            (Self::Eager(s), MoshiLm::Eager(m)) => s.step(m, text_token, input_audio),
            #[cfg(feature = "gpu-lm")]
            (Self::Gpu(s), MoshiLm::Gpu(_m)) => s.step(text_token, input_audio),
            #[allow(unreachable_patterns)]
            _ => {
                ensure!(false, "Moshi LM / generation backend mismatch");
                Ok(0)
            }
        }
    }

    pub fn last_audio_frame(&self) -> Option<Vec<u32>> {
        match self {
            Self::Eager(s) => s.last_audio_frame(),
            #[cfg(feature = "gpu-lm")]
            Self::Gpu(s) => s.last_audio_frame(),
        }
    }

    pub fn reset(
        &mut self,
        lm: &mut MoshiLm,
        max_steps: usize,
        text_lp: LogitsProcessor,
        audio_lp: LogitsProcessor,
    ) -> Result<()> {
        match (self, lm) {
            (Self::Eager(s), MoshiLm::Eager(m)) => {
                let _ = (max_steps, text_lp, audio_lp);
                s.reset(m);
                Ok(())
            }
            #[cfg(feature = "gpu-lm")]
            (Self::Gpu(s), MoshiLm::Gpu(g)) => s.reset(g, max_steps, text_lp, audio_lp),
            #[allow(unreachable_patterns)]
            _ => anyhow::bail!("Moshi LM / generation backend mismatch"),
        }
    }
}

/// Pick the LM device: honor CPU requests, fall back when GPU preset or backend is unavailable.
pub fn resolve_lm_device(requested: Device, checkpoint: MoshiCheckpoint) -> Device {
    if requested == Device::Cpu {
        return Device::Cpu;
    }
    if !checkpoint.gpu_loadable() {
        return Device::Cpu;
    }
    #[cfg(feature = "gpu-lm")]
    {
        if gpu_lm_available(requested) {
            return requested;
        }
    }
    Device::Cpu
}

#[cfg(feature = "gpu-lm")]
pub fn candle_device_for(device: Device) -> Result<candle::Device> {
    rlx_device_to_candle(device)
}
