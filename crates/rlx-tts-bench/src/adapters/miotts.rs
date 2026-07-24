use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_miotts::{GenerateOpts, MioSession, SAMPLE_RATE};
use rlx_runtime::Device;

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

const DEFAULT_MODEL_DIR: &str = "weights/tts/miotts";
const DEFAULT_CODEC_DIR: &str = "weights/tts/miocodec";

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "miotts",
        supports_clone: false,
        feature: "matrix-onnx",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_MODEL_DIR),
            env_keys: vec!["RLX_MIOTTS_DIR"],
            marker_files: vec!["model.safetensors"],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let model_dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let codec_dir = std::env::var("RLX_MIOCODEC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_CODEC_DIR));
    let inner = MioSession::open(&model_dir, &codec_dir, device).context("load miotts")?;
    let preset = std::env::var("RLX_PRESET").unwrap_or_else(|_| "en_female".into());
    Ok(Box::new(MioAdapter { inner, preset }))
}

struct MioAdapter {
    inner: MioSession,
    preset: String,
}

impl TtsAdapter for MioAdapter {
    fn id(&self) -> &'static str {
        "miotts"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        false
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let mut opts = GenerateOpts::default();
        opts.seed = req.seed;
        opts.preset = self.preset.clone();
        let t0 = Instant::now();
        let result = self.inner.synthesize(req.text, &opts)?;
        Ok(SynthResult {
            pcm: result.samples,
            sample_rate: SAMPLE_RATE,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}
