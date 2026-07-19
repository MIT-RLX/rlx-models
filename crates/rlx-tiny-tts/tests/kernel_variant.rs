// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! `KernelVariant` drives the `../rlx` backend kernel selection through
//! `rlx_ir::env` code overrides (precedence over process env, read at dispatch)
//! — so a TTS model/run can request precise (parity) or fast (production)
//! kernels without touching raw `RLX_*` env vars. This checks that applying a
//! variant installs the expected overrides that the Metal/CPU/CUDA backends read.
//!
//! Single test on purpose: the overrides are process-global, so asserting them
//! must not race a parallel test mutating the same keys.
use rlx_tiny_tts::KernelVariant;

#[test]
fn kernel_variant_installs_backend_overrides() {
    assert_eq!(KernelVariant::default(), KernelVariant::Fast);

    // Precise → precision/parity knobs on every backend.
    KernelVariant::Precise.apply();
    assert_eq!(rlx_ir::env::var("RLX_METAL_PRECISE").as_deref(), Some("1"));
    assert_eq!(rlx_ir::env::var("RLX_FAST_CONV").as_deref(), Some("0"));
    assert_eq!(rlx_ir::env::var("RLX_CUDA_NO_TF32").as_deref(), Some("1"));
    assert_eq!(rlx_ir::env::var("RLX_CUDA_PARITY").as_deref(), Some("1"));

    // Fast → clears the precise override (must not leak) + throughput knobs on.
    KernelVariant::Fast.apply();
    assert_eq!(rlx_ir::env::var("RLX_METAL_PRECISE").as_deref(), Some("0"));
    assert_eq!(rlx_ir::env::var("RLX_FAST_CONV").as_deref(), Some("1"));
    assert_eq!(rlx_ir::env::var("RLX_CUDA_CONV_TF32").as_deref(), Some("1"));
    assert_eq!(rlx_ir::env::var("RLX_CUDA_PARITY").as_deref(), Some("0"));

    // Inherit → leaves the current overrides untouched.
    KernelVariant::Inherit.apply();
    assert_eq!(rlx_ir::env::var("RLX_METAL_PRECISE").as_deref(), Some("0"));
}
