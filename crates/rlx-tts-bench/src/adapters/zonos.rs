use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_runtime::Device;
use rlx_zonos::{DEFAULT_DAC_DIR, DEFAULT_LOCAL_DIR, InferOpts, NativeZonos, SAMPLE_RATE};

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "zonos",
        supports_clone: false,
        feature: "matrix-ar",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_LOCAL_DIR),
            env_keys: vec!["RLX_ZONOS_DIR"],
            marker_files: vec!["config.json"],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let dac = std::env::var("RLX_DAC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_DAC_DIR));
    let inner = NativeZonos::open(&dir, &dac, device).context("load zonos")?;
    Ok(Box::new(ZonosAdapter { inner }))
}

struct ZonosAdapter {
    inner: NativeZonos,
}

impl TtsAdapter for ZonosAdapter {
    fn id(&self) -> &'static str {
        "zonos"
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
