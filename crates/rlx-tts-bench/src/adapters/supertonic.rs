use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_runtime::Device;
use rlx_supertonic::{DEFAULT_LOCAL_DIR, InferOpts, Supertonic, Voice};

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "supertonic",
        supports_clone: false,
        feature: "matrix-onnx",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_LOCAL_DIR),
            env_keys: vec!["RLX_SUPERTONIC_DIR"],
            marker_files: vec!["onnx/tts.json"],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner = Supertonic::load_on(&dir, device).context("load supertonic")?;
    let voice_path = dir.join("voice_styles/F1.json");
    let voice = Voice::load(&voice_path).with_context(|| format!("{}", voice_path.display()))?;
    Ok(Box::new(SupertonicAdapter { inner, voice }))
}

struct SupertonicAdapter {
    inner: Supertonic,
    voice: Voice,
}

impl TtsAdapter for SupertonicAdapter {
    fn id(&self) -> &'static str {
        "supertonic"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        false
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let opts = InferOpts::default();
        let t0 = Instant::now();
        let pcm = self.inner.synthesize(req.text, "en", &self.voice, &opts)?;
        Ok(SynthResult {
            pcm,
            sample_rate: self.inner.sample_rate(),
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}
