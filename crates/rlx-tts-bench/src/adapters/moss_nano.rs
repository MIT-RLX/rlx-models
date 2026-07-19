use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_moss_nano::{DEFAULT_LOCAL_DIR, MossNative, NativeOpts};
use rlx_runtime::Device;

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "moss-nano",
        supports_clone: false,
        feature: "matrix-onnx",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_LOCAL_DIR),
            env_keys: vec!["RLX_MOSS_NANO_DIR"],
            marker_files: vec![],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner = MossNative::load_on(&dir, device).context("load moss-nano")?;
    let voice = inner
        .voice_names()
        .into_iter()
        .next()
        .unwrap_or_else(|| "default".into());
    Ok(Box::new(MossAdapter { inner, voice }))
}

struct MossAdapter {
    inner: MossNative,
    voice: String,
}

impl TtsAdapter for MossAdapter {
    fn id(&self) -> &'static str {
        "moss-nano"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        false
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let opts = NativeOpts::default();
        let t0 = Instant::now();
        let stereo = self.inner.synthesize(req.text, &self.voice, &opts)?;
        let ch = self.inner.channels().max(1) as usize;
        let pcm: Vec<f32> = if ch <= 1 {
            stereo
        } else {
            stereo
                .chunks(ch)
                .map(|c| c.iter().sum::<f32>() / ch as f32)
                .collect()
        };
        Ok(SynthResult {
            pcm,
            sample_rate: self.inner.sample_rate(),
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}
