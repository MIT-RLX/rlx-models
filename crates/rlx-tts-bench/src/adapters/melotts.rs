use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_melotts::{DEFAULT_LOCAL_DIR, InferOpts, MeloTts};
use rlx_runtime::Device;

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "melotts",
        supports_clone: false,
        feature: "matrix-onnx",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_LOCAL_DIR),
            env_keys: vec!["RLX_MELOTTS_DIR", "RLX_TINY_TTS_DIR"],
            marker_files: vec!["config.json", "onnx/decoder.onnx"],
        },
    }
}

pub fn make(_device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = rlx_melotts::resolve_bundle_dir().or_else(|_| {
        meta()
            .hints
            .resolve_dir()
            .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))
    })?;
    let inner = MeloTts::load(&dir).context("load melotts")?;
    Ok(Box::new(MeloAdapter { inner }))
}

struct MeloAdapter {
    inner: MeloTts,
}

impl TtsAdapter for MeloAdapter {
    fn id(&self) -> &'static str {
        "melotts"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        false
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let mut opts = InferOpts::from_config(self.inner.config());
        opts.seed = req.seed;
        let t0 = Instant::now();
        let wav = self.inner.synthesize_on(req.text, req.device, &opts)?;
        Ok(SynthResult {
            pcm: wav.samples,
            sample_rate: wav.sample_rate,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}
