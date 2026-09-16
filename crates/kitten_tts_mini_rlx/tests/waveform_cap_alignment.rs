//! The vocoder wave axis and the NSF sine source must agree on length.
//!
//! `explicit_vocoder_shape` pins some sine-chain nodes to `max_wave` and others to
//! `frame_cap(max_wave) = ceil(max_wave / 300)`, then the `f0_upsamp` nearest-×300 turns the
//! second back into the first. Those only reconcile when `max_wave` is a whole number of
//! frames. A round cap like the TTS bench's `200_000` is not: `ceil(200_000/300)*300 = 200_100`,
//! so the sine source came out 100 samples longer than the `[1, max_wave, 9]` axis it feeds.
//! MLX rejected the `Reshape`; CPU and Metal accepted it and read a misaligned harmonic source.

use kitten_tts_mini_rlx::bundle_patches::{
    SAMPLES_PER_ALIGNMENT_FRAME, WAVEFORM_CAP_ALIGNMENT, align_waveform_cap,
};
use kitten_tts_mini_rlx::compile_profile::compile_waveform_cap;

/// Mirrors the private `bundle_patches::frame_cap`, which drives the sine-chain shapes.
const MEL_DIV: usize = 300;
fn frame_cap(max_wave: usize) -> usize {
    max_wave.div_ceil(MEL_DIV).max(1)
}

/// The invariant every other assertion here is really about.
fn assert_sine_source_matches_wave_axis(max_wave: usize) {
    assert_eq!(
        frame_cap(max_wave) * MEL_DIV,
        max_wave,
        "sine source ({} samples) outruns the wave axis ({max_wave} samples)",
        frame_cap(max_wave) * MEL_DIV,
    );
    assert_eq!(
        max_wave % SAMPLES_PER_ALIGNMENT_FRAME,
        0,
        "wave-frame cap (max_wave / {SAMPLES_PER_ALIGNMENT_FRAME}) is not exact for {max_wave}",
    );
}

#[test]
fn align_waveform_cap_holds_the_invariant() {
    // The three caps that violated it: the TTS-bench request, the wgpu storage-bind ceiling,
    // and the Vulkan `maxStorageBufferRange` ceiling. Each was off by exactly 100 samples.
    for raw in [200_000usize, 32_000, 80_000] {
        let aligned = align_waveform_cap(raw);
        assert_ne!(frame_cap(raw) * MEL_DIV, raw, "{raw} was already aligned");
        assert_sine_source_matches_wave_axis(aligned);
        assert!(aligned <= raw, "alignment must round down, not up");
        assert!(raw - aligned < WAVEFORM_CAP_ALIGNMENT);
    }
}

#[test]
fn align_waveform_cap_is_idempotent_and_never_zero() {
    assert_eq!(align_waveform_cap(0), 0);
    // Below one frame still has to produce a usable frame, not an empty axis.
    assert_eq!(align_waveform_cap(1), WAVEFORM_CAP_ALIGNMENT);
    for raw in [1usize, 599, 600, 24_000, 48_000, 199_999, 200_000] {
        let once = align_waveform_cap(raw);
        assert_eq!(once, align_waveform_cap(once), "not idempotent at {raw}");
        assert_sine_source_matches_wave_axis(once);
    }
}

#[test]
fn compile_waveform_cap_is_frame_aligned_under_engine_clamp() {
    // Unclamped, the estimate is aligned by construction (tokens*4800 + 12000). The regression
    // was the `engine_cap` bound, which is whatever the caller passed at load.
    for engine_cap in [usize::MAX, 200_000, 32_000, 80_000, 47_111] {
        for tokens in [1usize, 8, 25, 74, 256] {
            assert_sine_source_matches_wave_axis(compile_waveform_cap(tokens, engine_cap));
        }
    }
}
