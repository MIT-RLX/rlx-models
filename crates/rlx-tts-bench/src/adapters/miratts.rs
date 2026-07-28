use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_miratts::{MiraTts, SAMPLE_RATE};
use rlx_runtime::Device;

use crate::adapter::{
    AdapterMeta, CloneRequest, SynthRequest, SynthResult, TtsAdapter, WeightHints,
};
use crate::wav::{read_wav_mono, resample_linear};

const DEFAULT_DIR: &str = "weights/tts/miratts";

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "miratts",
        supports_clone: true,
        feature: "matrix-onnx",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_DIR),
            env_keys: vec!["RLX_MIRATTS_DIR"],
            marker_files: vec!["tokenizer.json"],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner = MiraTts::load(&dir, device).context("load miratts")?;
    Ok(Box::new(MiraAdapter { inner, dir }))
}

struct MiraAdapter {
    inner: MiraTts,
    dir: PathBuf,
}

impl TtsAdapter for MiraAdapter {
    fn id(&self) -> &'static str {
        "miratts"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        true
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let ref_pcm = resolve_ref_pcm(req.clone.as_ref(), &self.dir)?;
        let t0 = Instant::now();
        let pcm = self
            .inner
            .synthesize_with_ref(req.text, &ref_pcm, req.seed)?;
        Ok(SynthResult {
            pcm,
            sample_rate: SAMPLE_RATE,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}

fn resolve_ref_pcm(clone: Option<&CloneRequest<'_>>, dir: &Path) -> Result<Vec<f32>> {
    let sr = SAMPLE_RATE;
    if let Some(c) = clone {
        let (pcm, ref_sr) = read_wav_mono(c.ref_wav)?;
        return Ok(if ref_sr == sr {
            pcm
        } else {
            resample_linear(&pcm, ref_sr, sr)
        });
    }
    for name in ["default_voice.wav", "ref.wav", "prompt.wav"] {
        let path = dir.join(name);
        if path.is_file() {
            let (pcm, ref_sr) = read_wav_mono(&path)?;
            return Ok(if ref_sr == sr {
                pcm
            } else {
                resample_linear(&pcm, ref_sr, sr)
            });
        }
    }
    let jfk = PathBuf::from("assets/jfk/jfk_voice_clone.wav");
    if jfk.is_file() {
        let (pcm, ref_sr) = read_wav_mono(&jfk)?;
        return Ok(if ref_sr == sr {
            pcm
        } else {
            resample_linear(&pcm, ref_sr, sr)
        });
    }
    // Fallback: short tone so plain bench runs still exercise the clone path.
    Ok((0..sr as usize)
        .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sr as f32).sin() * 0.1)
        .collect())
}
