//! RLX compiled GPU path for Moshi — Candle Metal/CUDA kernels via Kyutai `moshi`.
//!
//! Native RLX graph compile for the 7B Helium temporal stack is not wired yet; this module
//! routes generation through the published Candle implementation (same weights as Q8 GGUF / bf16).

#[cfg(feature = "gpu-lm")]
pub use crate::gpu::{
    GpuGenerateState as CompiledGenerateState, GpuLm as CompiledLm, gpu_lm_available,
    rlx_device_to_candle,
};

#[cfg(feature = "gpu-lm")]
pub fn compiled_lm_available(device: rlx_runtime::Device) -> bool {
    gpu_lm_available(device)
}

#[cfg(not(feature = "gpu-lm"))]
pub fn compiled_lm_available(_device: rlx_runtime::Device) -> bool {
    false
}
