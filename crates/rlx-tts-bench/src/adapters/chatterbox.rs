use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_chatterbox::{DEFAULT_LOCAL_DIR, NativeChatterBox, SAMPLE_RATE, SynthOpts};
use rlx_runtime::Device;

use crate::adapter::{
    AdapterMeta, CloneRequest, SynthRequest, SynthResult, TtsAdapter, WeightHints,
};
use crate::wav::read_wav_mono;

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "chatterbox",
        supports_clone: true,
        feature: "matrix-onnx",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_LOCAL_DIR),
            env_keys: vec!["RLX_CHATTERBOX_DIR"],
            marker_files: vec![
                "weights.safetensors",
                "native/t3_lm.safetensors",
                "manifest.json",
            ],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner = NativeChatterBox::load_on(&dir, device).context("load chatterbox")?;
    Ok(Box::new(ChatterboxAdapter { inner, dir }))
}

struct ChatterboxAdapter {
    inner: NativeChatterBox,
    dir: PathBuf,
}

impl TtsAdapter for ChatterboxAdapter {
    fn id(&self) -> &'static str {
        "chatterbox"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        true
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let (ref_pcm, ref_sr) = resolve_ref(req.clone.as_ref(), &self.dir)?;
        let opts = SynthOpts {
            seed: req.seed,
            ..Default::default()
        };
        let t0 = Instant::now();
        let pcm = self.inner.synthesize(req.text, &ref_pcm, ref_sr, &opts)?;
        Ok(SynthResult {
            pcm,
            sample_rate: SAMPLE_RATE,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}

fn resolve_ref(clone: Option<&CloneRequest<'_>>, dir: &std::path::Path) -> Result<(Vec<f32>, u32)> {
    let path = clone
        .map(|c| c.ref_wav.to_path_buf())
        .unwrap_or_else(|| dir.join("default_voice.wav"));
    if path.is_file() {
        return read_wav_mono(&path);
    }
    // Fallback: short tone so non-clone list/run still exercises the path.
    let sr = SAMPLE_RATE;
    let pcm: Vec<f32> = (0..sr as usize)
        .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sr as f32).sin() * 0.1)
        .collect();
    Ok((pcm, sr))
}
