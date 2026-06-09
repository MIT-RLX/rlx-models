// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Map [`rlx_runtime::Device`] to ONNX Runtime execution providers.
#![cfg(feature = "onnx")]

use std::path::Path;

use std::num::NonZeroUsize;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use rlx_runtime::{Device, is_available};

const STANDARD_DEVICES: &[Device] = &[
    Device::Cpu,
    Device::Metal,
    Device::Mlx,
    Device::Cuda,
    Device::Rocm,
    Device::Gpu,
    Device::Vulkan,
];

/// Result of building an ONNX Runtime session.
pub struct OrtSession {
    pub session: Session,
    /// Human-readable ORT execution provider that actually loaded the model.
    pub ort_ep: String,
}

/// Fail fast on unsupported runtime devices.
pub fn validate_device(device: Device) -> Result<()> {
    if STANDARD_DEVICES.contains(&device) {
        Ok(())
    } else {
        bail!(
            "kittentts: device {device:?} is not supported \
             (use cpu|metal|mlx|cuda|rocm|gpu|vulkan)"
        )
    }
}

/// ORT execution providers for `device`, with CPU as the final fallback.
pub fn execution_providers_for(device: Device) -> Vec<ExecutionProviderDispatch> {
    let mut eps = primary_execution_providers(device);
    if !matches!(device, Device::Cpu) {
        eps.push(ort::ep::CPU::default().build());
    }
    eps
}

fn cpu_ep() -> ExecutionProviderDispatch {
    ort::ep::CPU::default().build()
}

#[cfg(all(feature = "ort-coreml", any(target_os = "macos", target_os = "ios")))]
fn coreml_ep_default() -> ExecutionProviderDispatch {
    ort::ep::CoreML::default().build()
}

#[cfg(all(feature = "ort-coreml", any(target_os = "macos", target_os = "ios")))]
fn coreml_ep_mlprogram_cpu() -> ExecutionProviderDispatch {
    ort::ep::CoreML::default()
        .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
        .with_compute_units(ort::ep::coreml::ComputeUnits::CPUOnly)
        .build()
}

#[cfg(feature = "ort-cuda")]
fn cuda_ep() -> ExecutionProviderDispatch {
    use ort::ep::cuda::{AttentionBackend, ConvAlgorithmSearch};

    let mut ep = ort::ep::CUDA::default()
        .with_device_id(0)
        .with_conv_algorithm_search(ConvAlgorithmSearch::Heuristic)
        .with_tf32(true)
        .with_attention_backend(
            AttentionBackend::FLASH_ATTENTION
                | AttentionBackend::EFFICIENT_ATTENTION
                | AttentionBackend::CUDNN_FLASH_ATTENTION,
        );
    if std::env::var_os("KITTENTTS_ORT_CUDA_GRAPH").is_some_and(|v| v == "1") {
        ep = ep.with_cuda_graph(true);
    }
    ep.build()
}

#[cfg(feature = "ort-rocm")]
fn rocm_ep() -> ExecutionProviderDispatch {
    ort::ep::ROCm::default().build()
}

#[cfg(all(feature = "ort-directml", target_os = "windows"))]
fn directml_ep() -> ExecutionProviderDispatch {
    ort::ep::DirectML::default().build()
}

fn primary_execution_providers(device: Device) -> Vec<ExecutionProviderDispatch> {
    match device {
        Device::Cpu => vec![cpu_ep()],
        Device::Metal | Device::Mlx => coreml_primary(device),
        Device::Cuda => cuda_primary(),
        Device::Rocm => rocm_primary(),
        Device::Gpu | Device::Vulkan => gpu_primary(device),
        other => {
            eprintln!("[kittentts] WARN: {other:?} has no ONNX Runtime EP mapping; using CPU");
            Vec::new()
        }
    }
}

fn coreml_primary(device: Device) -> Vec<ExecutionProviderDispatch> {
    #[cfg(all(feature = "ort-coreml", any(target_os = "macos", target_os = "ios")))]
    {
        eprintln!("[kittentts] ORT execution provider: CoreML ({device:?})");
        return vec![coreml_ep_default()];
    }
    #[cfg(not(all(feature = "ort-coreml", any(target_os = "macos", target_os = "ios"))))]
    {
        eprintln!(
            "[kittentts] WARN: {device:?} requested but CoreML EP unavailable on this target; using CPU"
        );
        Vec::new()
    }
}

fn cuda_primary() -> Vec<ExecutionProviderDispatch> {
    #[cfg(feature = "ort-cuda")]
    {
        eprintln!("[kittentts] ORT execution provider: CUDA");
        return vec![cuda_ep()];
    }
    #[cfg(not(feature = "ort-cuda"))]
    {
        eprintln!("[kittentts] WARN: CUDA requested but ort/cuda not enabled; using CPU");
        Vec::new()
    }
}

fn rocm_primary() -> Vec<ExecutionProviderDispatch> {
    #[cfg(feature = "ort-rocm")]
    {
        eprintln!("[kittentts] ORT execution provider: ROCm");
        return vec![rocm_ep()];
    }
    #[cfg(not(feature = "ort-rocm"))]
    {
        eprintln!("[kittentts] WARN: ROCm requested but ort/rocm not enabled; using CPU");
        Vec::new()
    }
}

fn gpu_primary(device: Device) -> Vec<ExecutionProviderDispatch> {
    #[cfg(all(feature = "ort-directml", target_os = "windows"))]
    {
        eprintln!("[kittentts] ORT execution provider: DirectML ({device:?})");
        return vec![directml_ep()];
    }
    #[cfg(all(feature = "ort-cuda", target_os = "linux"))]
    {
        eprintln!("[kittentts] ORT execution provider: CUDA ({device:?})");
        return vec![cuda_ep()];
    }
    #[cfg(all(feature = "ort-coreml", target_os = "macos"))]
    {
        eprintln!("[kittentts] ORT execution provider: CoreML ({device:?})");
        return vec![coreml_ep_default()];
    }
    #[cfg(not(any(
        all(feature = "ort-directml", target_os = "windows"),
        all(feature = "ort-cuda", target_os = "linux"),
        all(feature = "ort-coreml", target_os = "macos")
    )))]
    {
        eprintln!(
            "[kittentts] WARN: {device:?} requested but no matching ORT GPU EP on this target; using CPU"
        );
        Vec::new()
    }
}

/// Ordered EP attempts for `device`. Later entries are more conservative fallbacks.
fn ep_attempts(device: Device) -> Vec<(&'static str, Vec<ExecutionProviderDispatch>)> {
    let mut attempts = Vec::new();

    match device {
        Device::Cpu => {
            attempts.push(("cpu", vec![cpu_ep()]));
        }
        Device::Metal | Device::Mlx => {
            #[cfg(all(feature = "ort-coreml", any(target_os = "macos", target_os = "ios")))]
            {
                attempts.push(("coreml_default+cpu", vec![coreml_ep_default(), cpu_ep()]));
                attempts.push((
                    "coreml_mlprogram_cpu+cpu",
                    vec![coreml_ep_mlprogram_cpu(), cpu_ep()],
                ));
            }
            attempts.push(("cpu", vec![cpu_ep()]));
        }
        Device::Cuda => {
            #[cfg(feature = "ort-cuda")]
            attempts.push(("cuda+cpu", vec![cuda_ep(), cpu_ep()]));
            attempts.push(("cpu", vec![cpu_ep()]));
        }
        Device::Rocm => {
            #[cfg(feature = "ort-rocm")]
            attempts.push(("rocm+cpu", vec![rocm_ep(), cpu_ep()]));
            attempts.push(("cpu", vec![cpu_ep()]));
        }
        Device::Gpu | Device::Vulkan => {
            #[cfg(all(feature = "ort-directml", target_os = "windows"))]
            attempts.push(("directml+cpu", vec![directml_ep(), cpu_ep()]));
            #[cfg(all(feature = "ort-cuda", target_os = "linux"))]
            attempts.push(("cuda+cpu", vec![cuda_ep(), cpu_ep()]));
            #[cfg(all(feature = "ort-coreml", target_os = "macos"))]
            {
                attempts.push(("coreml_default+cpu", vec![coreml_ep_default(), cpu_ep()]));
                attempts.push((
                    "coreml_mlprogram_cpu+cpu",
                    vec![coreml_ep_mlprogram_cpu(), cpu_ep()],
                ));
            }
            attempts.push(("cpu", vec![cpu_ep()]));
        }
        _ => {
            attempts.push(("cpu", vec![cpu_ep()]));
        }
    }

    attempts
}

fn ort_intra_threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        if let Ok(s) = std::env::var("KITTENTTS_ORT_INTRA_THREADS") {
            if let Ok(n) = s.parse::<usize>() {
                return n.max(1);
            }
        }
        std::thread::available_parallelism()
            .map(NonZeroUsize::get)
            .unwrap_or(4)
            .max(1)
    })
}

fn try_build(
    model_path: &Path,
    label: &str,
    eps: Vec<ExecutionProviderDispatch>,
) -> Result<Session> {
    let mut builder = Session::builder().context("Failed to create ORT session builder")?;
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .context("Failed to set ORT graph optimization level")?
        .with_intra_threads(ort_intra_threads())
        .context("Failed to set ORT intra-op threads")?;
    if !eps.is_empty() {
        builder = builder
            .with_execution_providers(eps)
            .with_context(|| format!("Failed to register ORT execution providers ({label})"))?;
    }
    builder.commit_from_file(model_path).with_context(|| {
        format!(
            "Cannot load ONNX model with {label}: {}",
            model_path.display()
        )
    })
}

/// Build an ONNX Runtime session for `model_path` on `device`.
///
/// Tries the preferred EP chain for `device`, then falls back to CPU-only so
/// parity tests always have a loadable session.
pub fn build_onnx_session(model_path: &Path, device: Device) -> Result<OrtSession> {
    validate_device(device)?;
    if !is_available(device) {
        eprintln!(
            "[kittentts] WARN: RLX backend {device:?} not available in this build; ORT may fall back to CPU EP"
        );
    }

    let attempts = ep_attempts(device);
    let mut last_err = None;

    for (label, eps) in attempts {
        match try_build(model_path, label, eps) {
            Ok(session) => {
                if label != "cpu" && device != Device::Cpu {
                    eprintln!("[kittentts] loaded {device:?} via ORT EP chain '{label}'");
                } else if device != Device::Cpu {
                    eprintln!(
                        "[kittentts] loaded {device:?} via CPU EP fallback (GPU EP unavailable for this graph)"
                    );
                }
                let ort_ep = if label.starts_with("cuda") {
                    "cuda".to_string()
                } else if label.starts_with("rocm") {
                    "rocm".to_string()
                } else if label.starts_with("directml") {
                    "directml".to_string()
                } else if label.contains("coreml") {
                    "coreml".to_string()
                } else {
                    label.to_string()
                };
                return Ok(OrtSession { session, ort_ep });
            }
            Err(e) => {
                eprintln!("[kittentts] WARN: ORT {label} failed for {device:?}: {e:#}");
                last_err = Some(e);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no ORT EP attempts for {device:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_always_has_cpu_ep() {
        let eps = execution_providers_for(Device::Cpu);
        assert_eq!(eps.len(), 1);
    }

    #[test]
    fn non_cpu_appends_cpu_fallback() {
        let eps = execution_providers_for(Device::Metal);
        assert!(!eps.is_empty());
    }

    #[test]
    fn validate_rejects_exotic_devices() {
        assert!(validate_device(Device::Tpu).is_err());
    }

    #[test]
    fn ep_attempts_end_with_cpu_for_metal() {
        let attempts = ep_attempts(Device::Metal);
        assert_eq!(attempts.last().map(|(l, _)| *l), Some("cpu"));
    }
}
