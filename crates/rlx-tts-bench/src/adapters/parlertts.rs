use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_parlertts::{
    DEFAULT_DAC_DIR, DEFAULT_DESCRIPTION, DEFAULT_LOCAL_DIR, InferOpts, NativeParler, SAMPLE_RATE,
};
use rlx_runtime::Device;

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "parlertts",
        supports_clone: false,
        feature: "matrix-onnx",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_LOCAL_DIR),
            env_keys: vec!["RLX_PARLERTTS_DIR"],
            marker_files: vec!["tokenizer.json"],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let dac = std::env::var("RLX_PARLER_DAC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_DAC_DIR));
    let inner = NativeParler::open(&dir, &dac, device).context("load parlertts")?;
    let description =
        std::env::var("RLX_PARLER_DESCRIPTION").unwrap_or_else(|_| DEFAULT_DESCRIPTION.to_string());
    Ok(Box::new(ParlerAdapter { inner, description }))
}

struct ParlerAdapter {
    inner: NativeParler,
    description: String,
}

impl TtsAdapter for ParlerAdapter {
    fn id(&self) -> &'static str {
        "parlertts"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        false
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        // Keep SAMPLING (not `req.deterministic`): parlertts greedy is degenerate
        // (repetitive — collapses to "the", fox 0/6), so sampling gives the real
        // fox 6/6 output. The MLX crash for pt≠19 transcripts is separately fixed
        // (AOT cache-key now folds pt/et), so sampling is robust on MLX too. The
        // remaining GPU waveform-cosine gap is benign AR-sample divergence.
        let opts = InferOpts {
            seed: req.seed,
            ..Default::default()
        };
        let t0 = Instant::now();
        let pcm = self.inner.synthesize(req.text, &self.description, &opts)?;
        Ok(SynthResult {
            pcm,
            sample_rate: SAMPLE_RATE,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}
