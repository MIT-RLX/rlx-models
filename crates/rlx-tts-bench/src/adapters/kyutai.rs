use std::time::Instant;

use anyhow::{Context, Result};
use rlx_kyutai_tts::{
    GenerationConfig, KyutaiTtsSession, default_kyutai_tts_dir, default_mimi_dir,
};
use rlx_runtime::Device;

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "kyutai",
        supports_clone: false,
        feature: "lm-tts",
        hints: WeightHints {
            default_dir: default_kyutai_tts_dir(),
            env_keys: vec!["RLX_KYUTAI_TTS_DIR"],
            marker_files: vec![],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .unwrap_or_else(default_kyutai_tts_dir);
    let mimi = default_mimi_dir();
    let inner = KyutaiTtsSession::open_on(&dir, &mimi, device).context("load kyutai")?;
    Ok(Box::new(KyutaiAdapter { inner }))
}

struct KyutaiAdapter {
    inner: KyutaiTtsSession,
}

impl TtsAdapter for KyutaiAdapter {
    fn id(&self) -> &'static str {
        "kyutai"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        false
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let cfg = GenerationConfig::default();
        let t0 = Instant::now();
        let result = self.inner.generate(req.text, &cfg)?;
        Ok(SynthResult {
            pcm: result.samples,
            sample_rate: result.sample_rate,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}
