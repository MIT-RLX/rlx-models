//! Synthetic adapter for compile / CLI preflight without real weights.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use rlx_runtime::Device;

use crate::adapter::AdapterMeta;
use crate::adapter::{SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "fake",
        supports_clone: true,
        feature: "default",
        hints: WeightHints {
            default_dir: PathBuf::from("."),
            env_keys: vec![],
            marker_files: vec![],
        },
    }
}

pub fn make(_device: Device) -> Result<Box<dyn TtsAdapter>> {
    Ok(Box::new(FakeAdapter))
}

struct FakeAdapter;

impl TtsAdapter for FakeAdapter {
    fn id(&self) -> &'static str {
        "fake"
    }

    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }

    fn supports_clone(&self) -> bool {
        true
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let t0 = Instant::now();
        let sr = 16_000u32;
        let secs = if req.phrase_id == "long" { 2.0 } else { 0.8 };
        let n = (sr as f64 * secs) as usize;
        let freq = if req.clone.is_some() { 220.0 } else { 440.0 };
        let pcm: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin() * 0.2
            })
            .collect();
        Ok(SynthResult {
            pcm,
            sample_rate: sr,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}
