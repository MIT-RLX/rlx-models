//! ONNX Runtime vocoder path — runs the vocoder (mel → waveform) through ORT.
//! With the `coreml` feature on Apple it uses the **CoreML execution provider**
//! (CPU/GPU/ANE), the only route to CoreML in this workspace; otherwise CPU.
//!
//! CoreML's MIL requires bounded dims, so a dynamic-axis model silently falls
//! back to CPU. The CoreML path therefore loads a **static-shape** model
//! (`vocoder_static.onnx`, [1, mels, COREML_STATIC_FRAMES]) and chunks the mel
//! into fixed-length segments with overlap, trimmed so the result is identical
//! to a whole-utterance vocode. The CPU path uses the dynamic `vocoder.onnx`.

#![cfg(feature = "onnx")]

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::Array2;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

/// Fixed mel-frame length of the static CoreML model (keep in sync with the
/// `COREML_STATIC_FRAMES` constant in `scripts/export_inflect_nano.py`).
pub const COREML_STATIC_FRAMES: usize = 256;
/// Mel context kept on each side of a static chunk (≥ vocoder receptive field).
const OVERLAP: usize = 32;

pub struct OnnxVocoder {
    session: Session,
    hop: usize,
    /// `Some(L)` ⇒ static model: vocode in fixed `L`-frame chunks; `None` ⇒ dynamic.
    static_frames: Option<usize>,
}

impl OnnxVocoder {
    /// Load the vocoder. With `coreml` (and the `coreml` feature on macOS/iOS) the
    /// CoreML EP runs a static-shape model; otherwise the dynamic model on CPU.
    pub fn load(dir: &Path, hop: usize, coreml: bool) -> Result<Self> {
        let use_coreml = coreml
            && cfg!(all(
                feature = "coreml",
                any(target_os = "macos", target_os = "ios")
            ));
        let (model, static_frames) = if use_coreml {
            (dir.join("vocoder_static.onnx"), Some(COREML_STATIC_FRAMES))
        } else {
            (dir.join("vocoder.onnx"), None)
        };

        let mut eps = Vec::new();
        #[cfg(all(feature = "coreml", any(target_os = "macos", target_os = "ios")))]
        if coreml {
            eps.push(
                ort::ep::CoreML::default()
                    .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                    .build(),
            );
        }
        eps.push(ort::ep::CPU::default().build());

        let session = Session::builder()
            .context("ORT session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(eps)?
            .commit_from_file(&model)
            .with_context(|| format!("load {}", model.display()))?;
        Ok(Self {
            session,
            hop,
            static_frames,
        })
    }

    /// Run the session on one `[80, F]` mel slice → `F*hop` samples.
    fn run_slice(&mut self, mel: &Array2<f32>) -> Result<Vec<f32>> {
        let (c, t) = mel.dim();
        let data: Vec<f32> = mel.iter().copied().collect(); // [C,T] row-major == [1,C,T]
        let input = Tensor::<f32>::from_array(([1usize, c, t], data)).context("mel tensor")?;
        let outputs = self.session.run(ort::inputs![input]).context("ORT run")?;
        let (_shape, wav) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("extract wav")?;
        Ok(wav.to_vec())
    }

    /// `mel: [80, T]` → raw waveform `[T*hop]`.
    pub fn forward(&mut self, mel: &Array2<f32>) -> Result<Vec<f32>> {
        let Some(l) = self.static_frames else {
            return self.run_slice(mel); // dynamic model: whole utterance at once
        };

        // Static model: fixed L-frame chunks. Each segment of `seg` frames is
        // padded with `OVERLAP` context (zeros at the ends) to exactly L, run,
        // then the segment's samples are trimmed out and concatenated.
        let (c, total) = mel.dim();
        let seg = l - 2 * OVERLAP;
        let mut out = Vec::with_capacity(total * self.hop);
        let mut start = 0usize;
        while start < total {
            let end = (start + seg).min(total);
            let ctx_start = start.saturating_sub(OVERLAP);
            let mut buf = Array2::<f32>::zeros((c, l)); // zero-padded to L
            let copy_end = (end + OVERLAP).min(total);
            for (j, src) in (ctx_start..copy_end).enumerate() {
                for ch in 0..c {
                    buf[[ch, j]] = mel[[ch, src]];
                }
            }
            let wav = self.run_slice(&buf)?;
            let left = (start - ctx_start) * self.hop;
            let len = (end - start) * self.hop;
            out.extend_from_slice(&wav[left..(left + len).min(wav.len())]);
            start = end;
        }
        Ok(out)
    }
}
