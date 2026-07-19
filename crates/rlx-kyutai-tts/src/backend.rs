//! Unified Kyutai TTS backend — eager CPU ndarray or native RLX graphs on GPU.

use crate::config::KyutaiTtsConfig;
use crate::generate::{GenerateConfig, generate_codes};
use crate::model::KyutaiTtsModel;
use crate::rlx_model::RlxKyutaiTtsModel;
use crate::tokenizer::KyutaiTokenizer;
use anyhow::Result;
use ndarray::Array2;
use rlx_runtime::Device;
use std::path::Path;

/// How to run the Kyutai TTS LM stack.
pub enum KyutaiTtsBackend {
    /// Eager CPU reference (ndarray).
    Eager(KyutaiTtsModel),
    /// Native RLX temporal backbone (GPU/CPU); DepFormer stays eager.
    Rlx(RlxKyutaiTtsModel),
}

impl KyutaiTtsBackend {
    pub fn open(model_dir: &Path, cfg: KyutaiTtsConfig, device: Device) -> Result<Self> {
        let force_eager = std::env::var_os("RLX_KYUTAI_TTS_EAGER").is_some();
        let force_native = std::env::var_os("RLX_KYUTAI_TTS_NATIVE").is_some();
        let use_rlx = !force_eager || force_native;
        if use_rlx {
            if device != Device::Cpu {
                eprintln!("kyutai-tts: RLX temporal backbone on {device:?} (DepFormer eager)");
            } else {
                eprintln!("kyutai-tts: RLX temporal backbone on CPU (DepFormer eager)");
            }
            let max_upper = cfg.context;
            let lm = RlxKyutaiTtsModel::open(model_dir, cfg, device, max_upper)?;
            Ok(Self::Rlx(lm))
        } else {
            let lm = KyutaiTtsModel::open(model_dir, cfg, device)?;
            Ok(Self::Eager(lm))
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Self::Eager(m) => m.device(),
            Self::Rlx(m) => m.device(),
        }
    }

    pub fn reset_state(&mut self) {
        match self {
            Self::Eager(m) => m.reset_state(),
            Self::Rlx(m) => m.reset_state(),
        }
    }

    pub fn synthesize_codes(
        &mut self,
        tokenizer: &KyutaiTokenizer,
        prompt: &str,
        cfg: GenerateConfig,
        speaker: Option<&Array2<f32>>,
    ) -> Result<(Vec<Vec<u32>>, Vec<u32>)> {
        match self {
            Self::Eager(m) => {
                let (frames, _, text) = generate_codes(m, tokenizer, prompt, cfg, speaker)?;
                Ok((frames, text))
            }
            Self::Rlx(m) => {
                let (frames, _, text) = generate_codes(m, tokenizer, prompt, cfg, speaker)?;
                Ok((frames, text))
            }
        }
    }
}

/// Resolve LM device (honours availability; falls back to CPU).
pub fn resolve_lm_device(requested: Device) -> Device {
    crate::device::resolve_lm_device(requested)
}
