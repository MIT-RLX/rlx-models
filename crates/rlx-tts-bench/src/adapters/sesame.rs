use std::time::Instant;

use anyhow::{Context, Result};
use rlx_runtime::Device;
use rlx_sesame::{SesameCSM, load_wav_mono_24k};

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "sesame",
        supports_clone: true,
        feature: "matrix-ar",
        hints: WeightHints {
            default_dir: rlx_sesame::default_model_dir(),
            env_keys: vec!["RLX_SESAME_DIR"],
            marker_files: vec!["config.json", "model.safetensors"],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner = SesameCSM::open(&dir, device).context("load sesame")?;
    Ok(Box::new(SesameAdapter { inner }))
}

struct SesameAdapter {
    inner: SesameCSM,
}

impl TtsAdapter for SesameAdapter {
    fn id(&self) -> &'static str {
        "sesame"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        true
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let ctx = if let Some(c) = req.clone {
            Some(load_wav_mono_24k(c.ref_wav).context("sesame ref wav")?)
        } else {
            None
        };
        let t0 = Instant::now();
        let pcm = self.inner.synthesize(req.text, ctx.as_deref())?;
        Ok(SynthResult {
            pcm,
            sample_rate: 24_000,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}
