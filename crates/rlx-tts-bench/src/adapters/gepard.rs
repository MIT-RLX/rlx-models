use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_gepard::GepardSynthesizer;
use rlx_runtime::Device;

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};
use crate::devices::device_label;

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "gepard",
        supports_clone: false,
        feature: "matrix-ar",
        hints: WeightHints {
            default_dir: PathBuf::from("weights/tts/gepard"),
            env_keys: vec!["RLX_GEPARD_DIR"],
            marker_files: vec!["model.safetensors"],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner =
        GepardSynthesizer::with_device(&dir, device_label(device)).context("load gepard")?;
    Ok(Box::new(GepardAdapter { inner }))
}

struct GepardAdapter {
    inner: GepardSynthesizer,
}

impl TtsAdapter for GepardAdapter {
    fn id(&self) -> &'static str {
        "gepard"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        false
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        // gepard's default temperature sampling (0.4) free-runs into coherent
        // but WRONG words (fox 0/6 — "The love is a cure…"); greedy is faithful
        // (verified fox 6/6 "The quick brown fox jumps over the lazy dog.").
        // Honor the bench's deterministic request → greedy.
        self.inner.opts.greedy = req.deterministic;
        self.inner.opts.seed = req.seed;
        let t0 = Instant::now();
        let pcm = self.inner.synthesize(req.text, "")?;
        Ok(SynthResult {
            pcm,
            sample_rate: 22_050,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}
