//! RLX TTS adapter (private `.cache/rlx-tts` bundle).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_runtime::Device;
use rlx_tts::{RlxTts, VarianceControls, WaveRnnOpts};

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "rlx-tts",
        supports_clone: false,
        feature: "rlx-tts",
        hints: WeightHints {
            default_dir: PathBuf::from(rlx_tts::DEFAULT_BUNDLE_DIR),
            env_keys: vec!["RLX_TTS_BUNDLE"],
            marker_files: vec!["manifest.json", "wavernn.safetensors"],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    if !matches!(device, Device::Cpu) {
        anyhow::bail!(
            "rlx-tts product path is host CPU only (requested {:?})",
            device
        );
    }
    let dir = meta().hints.resolve_dir();
    let inner = if let Some(d) = dir {
        RlxTts::open(&d).with_context(|| format!("open RLX TTS bundle {}", d.display()))?
    } else {
        RlxTts::open_default().context("open RLX TTS default bundle")?
    };
    Ok(Box::new(RlxTtsAdapter {
        inner,
        ctrl: VarianceControls::default(),
        vocoder: WaveRnnOpts::product_default(),
        device,
    }))
}

struct RlxTtsAdapter {
    inner: RlxTts,
    ctrl: VarianceControls,
    vocoder: WaveRnnOpts,
    device: Device,
}

impl TtsAdapter for RlxTtsAdapter {
    fn id(&self) -> &'static str {
        "rlx-tts"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        false
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let mut vocoder = self.vocoder.clone();
        if req.seed != 0 {
            vocoder.seed = Some(req.seed);
        }
        let t0 = Instant::now();
        let audio = self
            .inner
            .synthesize_text(req.text, &self.ctrl, &vocoder)
            .with_context(|| format!("rlx-tts synthesize_text len={}", req.text.len()))?;
        Ok(SynthResult {
            pcm: audio.pcm,
            sample_rate: audio.sample_rate,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("host-cpu/{:?}", self.device),
        })
    }
}
