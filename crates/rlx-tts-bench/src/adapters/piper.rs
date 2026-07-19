use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_piper::{DEFAULT_LOCAL_DIR, NativeVits};
use rlx_runtime::Device;

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "piper",
        supports_clone: false,
        feature: "matrix-onnx",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_LOCAL_DIR),
            env_keys: vec!["RLX_PIPER_DIR"],
            marker_files: vec![],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner = NativeVits::load(&dir, device).context("load piper")?;
    Ok(Box::new(PiperAdapter { inner }))
}

struct PiperAdapter {
    inner: NativeVits,
}

impl TtsAdapter for PiperAdapter {
    fn id(&self) -> &'static str {
        "piper"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        false
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let t0 = Instant::now();
        let pcm = self.inner.synthesize(req.text, None)?;
        Ok(SynthResult {
            pcm,
            sample_rate: self.inner.sample_rate(),
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}
