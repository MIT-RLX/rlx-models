use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_metavoice::{DEFAULT_LOCAL_DIR, InferOpts, MetaVoice, SAMPLE_RATE};
use rlx_runtime::Device;

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "metavoice",
        supports_clone: true,
        feature: "matrix-ar",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_LOCAL_DIR),
            env_keys: vec!["RLX_METAVOICE_DIR"],
            marker_files: vec![],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner = MetaVoice::open(&dir, device).context("load metavoice")?;
    Ok(Box::new(MetaAdapter { inner }))
}

struct MetaAdapter {
    inner: MetaVoice,
}

impl TtsAdapter for MetaAdapter {
    fn id(&self) -> &'static str {
        "metavoice"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        true
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let ref_wav = req.clone.map(|c| c.ref_wav);
        let opts = bench_infer_opts(req.text);
        let t0 = Instant::now();
        let pcm = self.inner.synthesize(req.text, ref_wav, &opts)?;
        Ok(SynthResult {
            pcm,
            sample_rate: SAMPLE_RATE,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}

/// Bench/short-phrase budget. Default `InferOpts` uses 864 (demo-length); the
/// first-stage GPT is **CPU eager** (~1–2 s/token), so CUDA/wgpu only accelerate
/// EnCodec. Override with `RLX_METAVOICE_MAX_TOKENS`.
fn bench_infer_opts(text: &str) -> InferOpts {
    let mut opts = InferOpts::default();
    let words = text.split_whitespace().count().max(1);
    // Keep fox-length cells under ~2 min on CPU AR (~1.5s/step).
    let scaled = (words * 8 + 32).clamp(48, 96);
    opts.max_new_tokens = std::env::var("RLX_METAVOICE_MAX_TOKENS")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(scaled);
    opts
}
