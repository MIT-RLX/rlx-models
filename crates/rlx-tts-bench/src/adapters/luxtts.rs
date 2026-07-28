use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_luxtts::{DEFAULT_LOCAL_DIR, InferOpts, LuxTts};
use rlx_runtime::Device;

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};
use crate::wav::read_wav_mono;

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "luxtts",
        supports_clone: true,
        feature: "matrix-onnx",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_LOCAL_DIR),
            env_keys: vec!["RLX_LUXTTS_DIR"],
            marker_files: vec!["encoder_body.onnx", "fm_decoder.onnx"],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner = LuxTts::load_on(&dir, device).context("load luxtts")?;
    Ok(Box::new(LuxAdapter { inner, dir }))
}

struct LuxAdapter {
    inner: LuxTts,
    dir: PathBuf,
}

impl TtsAdapter for LuxAdapter {
    fn id(&self) -> &'static str {
        "luxtts"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        true
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let ref_path = req
            .clone
            .as_ref()
            .map(|c| c.ref_wav.to_path_buf())
            .or_else(|| find_prompt_wav(&self.dir))
            .ok_or_else(|| {
                anyhow::anyhow!("luxtts needs --clone ref wav or prompt under weights")
            })?;
        let (prompt, _) = read_wav_mono(&ref_path)?;
        let prompt_text = req
            .clone
            .as_ref()
            .and_then(|c| c.ref_text)
            .unwrap_or("The quick brown fox jumps over the lazy dog.");
        let opts = InferOpts::default();
        let t0 = Instant::now();
        // luxtts' onnx-imported fm_decoder emits a malformed matmul at long
        // num_frames (long text panics: rlx-cuda MatMul unsupported shapes);
        // short works. Sentence-chunk so each stays in the working range, then
        // concatenate. Override the char budget via RLX_LUXTTS_MAX_CHARS.
        // Two thresholds: text up to `whole_max` generates fine unchunked (the
        // short fox phrase ≈63 chars works), so don't split it; longer text is
        // packed into `chunk_max`-sized pieces safely under the fm_decoder
        // shape-bug threshold.
        let whole_max = std::env::var("RLX_LUXTTS_WHOLE_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(72usize);
        let chunk_max = std::env::var("RLX_LUXTTS_MAX_CHARS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(52usize);
        let chunks = sentence_chunks(req.text, whole_max, chunk_max);
        let mut pcm: Vec<f32> = Vec::new();
        for ch in &chunks {
            let seg = self.inner.synthesize(ch, &prompt, prompt_text, &opts)?;
            if pcm.is_empty() {
                pcm = seg;
            } else {
                let xf = (self.inner.sample_rate() as usize / 50)
                    .min(pcm.len())
                    .min(seg.len());
                let base = pcm.len() - xf;
                for i in 0..xf {
                    let a = 1.0 - (i as f32 / xf as f32);
                    let b = i as f32 / xf as f32;
                    pcm[base + i] = pcm[base + i] * a + seg[i] * b;
                }
                pcm.extend_from_slice(&seg[xf..]);
            }
        }
        Ok(SynthResult {
            pcm,
            sample_rate: self.inner.sample_rate(),
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: self.inner.ort_ep(),
        })
    }
}

/// Split `text` into chunks of at most ~`max_chars` by packing WORDS (so a long
/// sentence is broken too), preferring to end a chunk after sentence punctuation.
/// luxtts' num_frames scales with text length; each chunk must stay under the
/// fm_decoder shape-bug threshold (short≈63 chars works; ~104 fails).
fn sentence_chunks(text: &str, whole_max: usize, max_chars: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return vec![String::new()];
    }
    if text.chars().count() <= whole_max {
        return vec![text.to_string()];
    }
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        let cand = if cur.is_empty() {
            word.to_string()
        } else {
            format!("{cur} {word}")
        };
        if cand.chars().count() > max_chars && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
            cur = word.to_string();
        } else {
            cur = cand;
        }
        // Prefer a clean break after sentence-ending punctuation once the chunk
        // is reasonably full.
        if cur.chars().count() >= max_chars * 3 / 4 && cur.trim_end().ends_with(['.', '!', '?']) {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn find_prompt_wav(dir: &std::path::Path) -> Option<PathBuf> {
    for name in ["prompt.wav", "default_voice.wav", "ref.wav"] {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    let jfk = PathBuf::from("assets/jfk/jfk_voice_clone.wav");
    jfk.is_file().then_some(jfk)
}
