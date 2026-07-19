//! CoreML / ANE helpers shared by every TinyModel-backed TTS crate.
//!
//! Upstream `rlx-coreml::default_compute_units` already maps **fp32** graphs to
//! CPU+GPU (Neural-Engine BNNS AOT SIGSEGVs on many large imported ONNX graphs).
//! TTS still pins `RLX_COREML_UNITS=gpu` when unset so **f16** edges (e.g. F5-TTS)
//! and older rlx checkouts do not silently take the ANE path.

use std::sync::Once;

use rlx_runtime::Device;

/// Ensure CoreML compute units are safe for large TTS graphs.
///
/// Call before compiling on [`Device::Ane`]. No-op when the user already set
/// `RLX_COREML_UNITS`. Idempotent.
pub fn ensure_coreml_units_for_tts() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        match std::env::var("RLX_COREML_UNITS") {
            Ok(v) if !v.is_empty() => {
                eprintln!(
                    "[tiny-tts] CoreML: honoring RLX_COREML_UNITS={v} \
                     (Neural-Engine BNNS can SIGSEGV on large TTS graphs — \
                     prefer gpu|all if you hit crashes)"
                );
            }
            _ => {
                // SAFETY: set once at first Ane compile, before CoreML loads models.
                unsafe {
                    std::env::set_var("RLX_COREML_UNITS", "gpu");
                }
                eprintln!(
                    "[tiny-tts] CoreML: set RLX_COREML_UNITS=gpu \
                     (pins f16 TTS graphs off Neural-Engine BNNS; \
                      fp32 already defaults to CPU+GPU in rlx-coreml)"
                );
            }
        }
    });
}

/// Map a requested TTS execution device.
///
/// Pins CoreML compute units for [`Device::Ane`]. Passes every other device
/// through unchanged (including [`Device::Vulkan`] — no silent remaps; Supertonic
/// is the only crate that remaps Vulkan behind an explicit env force-gate).
pub fn resolve_tts_device(requested: Device) -> Device {
    if matches!(requested, Device::Ane) {
        ensure_coreml_units_for_tts();
    }
    requested
}

/// Tag for AOT cache keys so a change of compute units does not reuse a package
/// finalized under a different policy.
pub fn coreml_units_cache_tag() -> String {
    match std::env::var("RLX_COREML_UNITS") {
        Ok(v) if !v.is_empty() => format!("_cml{v}"),
        _ => "_cmlgpu".into(), // matches the default we install above
    }
}
