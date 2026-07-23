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
    // The FIRST builtin voice is Chinese ("Junhao"), which garbles English text
    // on every backend. Prefer an English voice ("Trump" is the validated one),
    // overridable via RLX_MOSS_VOICE.
    let voices = inner.voice_names();
    let voice = std::env::var("RLX_MOSS_VOICE")
        .ok()
        .filter(|v| voices.iter().any(|n| n == v))
        .or_else(|| {
            ["Trump", "Alice", "Bob", "en", "English"].iter().find_map(|w| {
                voices
                    .iter()
                    .find(|n| n.eq_ignore_ascii_case(w) || n.to_lowercase().contains(&w.to_lowercase()))
                    .cloned()
            })
        })
        .or_else(|| voices.first().cloned())
        .unwrap_or_else(|| "default".into());
    eprintln!("[moss-nano] voice={voice}");
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
        // Validated backend_matrix uses max_frames=64 (NativeOpts::default is 96,
        // which garbles). Override via RLX_MOSS_FRAMES.
        let opts = NativeOpts {
            max_frames: std::env::var("RLX_MOSS_FRAMES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(64),
            ..NativeOpts::default()
        };
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
