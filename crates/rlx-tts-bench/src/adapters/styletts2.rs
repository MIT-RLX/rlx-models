use std::time::Instant;

use anyhow::{Context, Result};
use rlx_runtime::Device;
use rlx_styletts2::{STYLETTS2_SAMPLE_RATE, StyleTTS2};

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "styletts2",
        supports_clone: false,
        feature: "matrix-onnx",
        hints: WeightHints {
            default_dir: rlx_styletts2::default_model_dir(),
            env_keys: vec!["RLX_STYLETTS2_DIR", "RLX_KOKORO_DIR"],
            marker_files: vec![],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner = StyleTTS2::load(&dir, device).context("load styletts2/kokoro")?;
    Ok(Box::new(StyleAdapter { inner }))
}

struct StyleAdapter {
    inner: StyleTTS2,
}

impl TtsAdapter for StyleAdapter {
    fn id(&self) -> &'static str {
        "styletts2"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        false
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let t0 = Instant::now();
        let pcm = self.inner.synthesize(req.text)?;
        Ok(SynthResult {
            pcm,
            sample_rate: STYLETTS2_SAMPLE_RATE,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}
