use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_runtime::Device;
use rlx_soprano::{DEFAULT_LOCAL_DIR, InferOpts, NativeSoprano, SAMPLE_RATE};

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "soprano",
        supports_clone: false,
        feature: "matrix-onnx",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_LOCAL_DIR),
            env_keys: vec!["RLX_SOPRANO_DIR"],
            marker_files: vec!["tokenizer.json"],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner = NativeSoprano::open(&dir, device).context("load soprano")?;
    Ok(Box::new(SopranoAdapter { inner }))
}

struct SopranoAdapter {
    inner: NativeSoprano,
}

impl TtsAdapter for SopranoAdapter {
    fn id(&self) -> &'static str {
        "soprano"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        false
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let mut opts = InferOpts::default();
        opts.seed = req.seed;
        let t0 = Instant::now();
        let pcm = self.inner.synthesize(req.text, &opts)?;
        Ok(SynthResult {
            pcm,
            sample_rate: SAMPLE_RATE,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}
