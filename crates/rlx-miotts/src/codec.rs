//! MioCodec decode: native RLX import of `decoder_body.onnx` → mag/phase + host ISTFT.
//!
//! The wave-transformer body imports via `rlx-tiny-tts` / `rlx-onnx-import` and runs
//! on any RLX [`Device`]. ISTFT uses [`rlx_xcodec::istft::istft_same`].

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;
use rlx_xcodec::istft::istft_same;

use crate::tokens::{SPEECH_LEN, fit_speech_len};

pub const SAMPLE_RATE: u32 = 24_000;
pub const N_FFT: usize = 1920;
pub const HOP: usize = 480;
pub const N_FREQ: usize = N_FFT / 2 + 1;
pub const GLOBAL_DIM: usize = 128;

/// MioCodec decoder (native `decoder_body.onnx` + ISTFT).
pub struct MioCodec {
    model: TinyModel,
    device: Device,
    window: Vec<f32>,
    codec_dir: PathBuf,
}

impl MioCodec {
    pub fn load(codec_dir: &Path, device: Device) -> Result<Self> {
        let onnx = codec_dir.join("decoder_body.onnx");
        anyhow::ensure!(
            onnx.is_file(),
            "missing {} — run crates/rlx-miotts/scripts/export_miocodec_decode.py",
            onnx.display()
        );
        // The ISTFT analysis window. Prefer a shipped fixture if present, but fall
        // back to computing it: MioCodec's `istft_head.ISTFT` uses the standard
        // periodic Hann window (`torch.hann_window(win_length)`), so it is fully
        // determined by `N_FFT` and does not need to be a downloaded artifact.
        let window = load_f32_bin(&codec_dir.join("fixtures/hann_window.f32"))
            .or_else(|_| load_f32_bin(&codec_dir.join("hann_window.f32")))
            .unwrap_or_else(|_| hann_window_periodic(N_FFT));
        anyhow::ensure!(
            window.len() == N_FFT,
            "window len {} != {N_FFT}",
            window.len()
        );

        let cfg = BundleConfig {
            model: String::new(),
            sample_rate: SAMPLE_RATE,
            add_blank: false,
            language: "EN".into(),
            speakers: Default::default(),
            default_speaker: None,
            noise_scale: 0.0,
            noise_scale_w: 0.0,
            length_scale: 1.0,
            inter_channels: 0,
            gin_channels: 0,
        };
        Ok(Self {
            model: TinyModel::new(codec_dir.to_path_buf(), cfg),
            device,
            window,
            codec_dir: codec_dir.to_path_buf(),
        })
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Backend label for logs (`rlx-cpu`, `rlx-metal`, …).
    pub fn ep(&self) -> &'static str {
        match self.device {
            Device::Cpu => "rlx-cpu",
            Device::Metal => "rlx-metal",
            Device::Mlx => "rlx-mlx",
            Device::Cuda => "rlx-cuda",
            Device::Rocm => "rlx-rocm",
            Device::Gpu => "rlx-gpu",
            Device::Vulkan => "rlx-vulkan",
            Device::Ane => "rlx-ane",
            _ => "rlx",
        }
    }

    pub fn codec_dir(&self) -> &Path {
        &self.codec_dir
    }

    pub fn decode(&self, content_codes: &[u32], global_emb: &[f32]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            global_emb.len() == GLOBAL_DIM,
            "global emb len {} != {GLOBAL_DIM}",
            global_emb.len()
        );
        let codes = fit_speech_len(content_codes);
        let tokens: Vec<i64> = codes.iter().map(|&c| c as i64).collect();
        let tok_b: Vec<u8> = tokens.iter().flat_map(|x| x.to_le_bytes()).collect();
        let emb_b: Vec<u8> = global_emb.iter().flat_map(|x| x.to_le_bytes()).collect();

        let mut g = self
            .model
            .compile_named("decoder_body", self.device, SPEECH_LEN, &[])
            .map_err(|e| anyhow::anyhow!("compile decoder_body: {e:#}"))?;
        let outs = g.run_typed(&[
            ("content_token_indices", &tok_b, DType::I64),
            ("global_embedding", &emb_b, DType::F32),
        ]);
        anyhow::ensure!(
            outs.len() >= 2,
            "decoder_body expected mag+phase, got {}",
            outs.len()
        );
        let mag = as_f32(&outs[0]).context("mag")?;
        let phase = as_f32(&outs[1]).context("phase")?;
        // Parity bisection (dev): dump the exact decoder_body I/O so a Python
        // onnxruntime run can diff native mag/phase (isolates decoder_body vs ISTFT).
        if let Some(dir) = std::env::var_os("RLX_MIO_DUMP") {
            let dir = std::path::PathBuf::from(dir);
            let _ = std::fs::create_dir_all(&dir);
            let dump = |name: &str, b: &[u8]| {
                let _ = std::fs::write(dir.join(name), b);
            };
            dump("in_tokens.i64", &tok_b);
            dump("in_emb.f32", &emb_b);
            dump(
                "mag_native.f32",
                &mag.iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<u8>>(),
            );
            dump(
                "phase_native.f32",
                &phase
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<u8>>(),
            );
            // Extra outputs (from RLX_ONNX_TAP) land at outs[2..] — dump each for a
            // native-vs-ort numerical bisect of intermediate tensors.
            for (i, o) in outs.iter().enumerate().skip(2) {
                dump(&format!("tap_{i}.f32"), &o.0);
                eprintln!("[mio-dump] tap_{i} len={}", o.0.len() / 4);
            }
            eprintln!(
                "[mio-dump] tokens={} emb={} mag={} phase={} t={}",
                codes.len(),
                global_emb.len(),
                mag.len(),
                phase.len(),
                mag.len() / N_FREQ
            );
        }
        anyhow::ensure!(mag.len() == phase.len() && mag.len() % N_FREQ == 0);
        let t = mag.len() / N_FREQ;
        Ok(istft_same(
            &mag,
            &phase,
            N_FREQ,
            t,
            &self.window,
            N_FFT,
            HOP,
        ))
    }
}

fn as_f32((bytes, dt): &(Vec<u8>, DType)) -> Result<Vec<f32>> {
    anyhow::ensure!(*dt == DType::F32, "expected F32, got {dt:?}");
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Load a preset global embedding (`.f32` raw).
pub fn load_preset_embedding(presets_dir: &Path, preset_id: &str) -> Result<Vec<f32>> {
    let f32_path = presets_dir.join(format!("{preset_id}.f32"));
    if f32_path.is_file() {
        let v = load_f32_bin(&f32_path)?;
        anyhow::ensure!(v.len() == GLOBAL_DIM, "preset {preset_id} dim {}", v.len());
        return Ok(v);
    }
    anyhow::bail!(
        "preset '{preset_id}' not found under {} (need {preset_id}.f32)",
        presets_dir.display()
    )
}

fn load_f32_bin(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Periodic Hann window of length `n` — matches PyTorch `torch.hann_window(n)`
/// (`periodic=True`), which MioCodec's ISTFT head uses: `0.5·(1 − cos(2πk/n))`.
fn hann_window_periodic(n: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    (0..n)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n as f32).cos()))
        .collect()
}
