//! Unified Moshi LM backend — eager CPU (ndarray) or native RLX graphs.

use crate::checkpoint::MoshiCheckpoint;
use crate::config::{GenerateConfig, MoshiVariant};
use crate::generate::GenerateState;
use crate::lm::LmModel;
use crate::rlx_gen::{RlxGenerateState, RlxLm};
use crate::sampling::LogitsProcessor;
use crate::weights::{open_lm, open_lm_from_weights};
use anyhow::{Result, ensure};
use rlx_runtime::Device;
use std::path::Path;

/// How to load and run the Moshi temporal + DepFormer stack.
pub enum MoshiLm {
    /// Eager CPU reference (ndarray).
    Eager(LmModel),
    /// Native RLX graphs (CPU or GPU).
    Rlx(RlxLm),
}

/// Per-generation autoregressive state.
pub enum MoshiGenState {
    Eager(GenerateState),
    Rlx(RlxGenerateState),
}

impl MoshiLm {
    pub fn open(
        model_dir: &Path,
        variant: MoshiVariant,
        checkpoint: MoshiCheckpoint,
        device: Device,
    ) -> Result<Self> {
        // Native RLX graph backend: explicit opt-in, or any non-CPU device.
        if std::env::var_os("RLX_MOSHI_NATIVE").is_some() || device != Device::Cpu {
            return Ok(Self::Rlx(RlxLm::open(
                model_dir, variant, checkpoint, device,
            )?));
        }

        // CPU eager (ndarray) path. All checkpoint formats load candle-free.
        let cfg = variant.lm_config();
        let weights_path = checkpoint.lm_weights_path(model_dir);
        if checkpoint.is_mlx() {
            let weights =
                crate::mlx_weights::load_eager_weight_map(&weights_path, checkpoint, &cfg)?;
            return Ok(Self::Eager(open_lm_from_weights(cfg, weights)?));
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
            Self::Rlx(m) => m.config(),
        }
    }

    pub fn reset_state(&mut self) {
        match self {
            Self::Eager(m) => m.reset_state(),
            Self::Rlx(_) => {}
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
            Self::Rlx(_) => Ok(MoshiGenState::Rlx(RlxGenerateState::new(
                max_steps, text_lp, audio_lp, gen_cfg,
            ))),
        }
    }
}

impl MoshiGenState {
    pub fn config(&self) -> &GenerateConfig {
        match self {
            Self::Eager(s) => s.config(),
            Self::Rlx(s) => s.config(),
        }
    }

    pub fn step_idx(&self) -> usize {
        match self {
            Self::Eager(s) => s.step_idx(),
            Self::Rlx(s) => s.step_idx(),
        }
    }

    pub fn text_tokens(&self) -> &[u32] {
        match self {
            Self::Eager(s) => s.text_tokens(),
            Self::Rlx(s) => s.text_tokens(),
        }
    }

    pub fn step(&mut self, lm: &mut MoshiLm, text_token: u32, input_audio: &[u32]) -> Result<u32> {
        match (self, lm) {
            (Self::Eager(s), MoshiLm::Eager(m)) => s.step(m, text_token, input_audio),
            (Self::Rlx(s), MoshiLm::Rlx(m)) => s.step(m, text_token, input_audio),
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
            Self::Rlx(s) => s.last_audio_frame(),
        }
    }

    pub fn reset(
        &mut self,
        lm: &mut MoshiLm,
        max_steps: usize,
        text_lp: LogitsProcessor,
        audio_lp: LogitsProcessor,
    ) -> Result<()> {
        let _ = (max_steps, text_lp, audio_lp);
        match (self, lm) {
            (Self::Eager(s), MoshiLm::Eager(m)) => {
                s.reset(m);
                Ok(())
            }
            (Self::Rlx(s), MoshiLm::Rlx(_m)) => {
                // lps are kept; just clear KV cache + token buffers.
                s.reset();
                Ok(())
            }
            #[allow(unreachable_patterns)]
            _ => anyhow::bail!("Moshi LM / generation backend mismatch"),
        }
    }
}

/// Pick the LM device: honor CPU, keep an available GPU, else fall back to CPU.
pub fn resolve_lm_device(requested: Device, _checkpoint: MoshiCheckpoint) -> Device {
    if requested == Device::Cpu {
        return Device::Cpu;
    }
    if rlx_runtime::is_available(requested) {
        requested
    } else {
        Device::Cpu
    }
}
