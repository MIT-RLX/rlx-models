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

//! Parity against TEN-VAD, checked stage by stage.
//!
//! The target is **the model TEN-framework publishes** — `src/onnx_model/
//! ten-vad.onnx` driven by the DSP in `src/*.cc` — because that is what defines
//! TEN-VAD and what a port can be held to. Fixtures (see
//! `tests/fixtures/README.md`):
//!
//! | fixture | produced by |
//! |---|---|
//! | `reference_features.f32` | the upstream C DSP, `src/*.cc` built `-ffp-contract=off` |
//! | `reference_probs_onnx.f32` | onnxruntime on `ten-vad.onnx`, fed those features |
//! | `shipped_library_probs.f32` | the prebuilt `libten_vad` TEN-framework ships |
//!
//! The prebuilt library is checked separately and loosely on purpose: it embeds
//! the DSP tables from `src/coeff.h` byte-for-byte, but **it does not contain
//! the weights of its own `ten-vad.onnx`** in any float layout, so it is running
//! a different build of the network. It sits ~9.6e-4 from the published model,
//! roughly 150× further than this port does. Decisions still agree everywhere.

use std::path::PathBuf;

use rlx_ten_vad::model::{LstmState, Shape, TenVadModel};
use rlx_ten_vad::session::{TenVad, TenVadBatch, TenVadConfig};
use rlx_ten_vad::{CONTEXT_FRAMES, DEFAULT_THRESHOLD, FEATURE_LEN, HOP_SIZE, TenVadWeights, synth};

/// Network alone, on bit-identical features. This is onnxruntime's `f32`
/// accumulation order versus rlx's, and nothing else — the floor for any
/// implementation that is not a copy of ORT's kernels.
const NET_MAX: f32 = 1e-6;
/// Whole pipeline. The frontend contributes exactly zero (it is bit-identical),
/// so this is the same floor as `NET_MAX`.
const PIPELINE_MAX: f32 = if cfg!(feature = "fast-pitch") {
    // `fast-pitch` replaces the pitch estimator's inverse FFT with the
    // equivalent matrix. Same linear map, different summation order, so the
    // pitch feature moves in its last few bits and the probability follows.
    // Measured 2.09e-6 on the fixture with **zero decision flips** and cosine
    // 1.000000000 — the bound is set just above the measurement, not loosened
    // to whatever passes.
    1e-5
} else {
    1e-6
};
const PIPELINE_MEAN: f32 = if cfg!(feature = "fast-pitch") {
    1e-6
} else {
    5e-7
};
/// The prebuilt library runs a different network build; this only tracks it.
const SHIPPED_MAX: f32 = 3e-3;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn read_f32(name: &str) -> Vec<f32> {
    std::fs::read(fixtures().join(name))
        .unwrap_or_else(|e| panic!("read fixture {name}: {e}"))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn fixture_pcm() -> Vec<i16> {
    std::fs::read(fixtures().join("synthetic_speech_16k.pcm"))
        .expect("pcm fixture")
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

/// `[frames, FEATURE_LEN]` — the current-frame feature row per frame.
fn reference_features() -> Vec<Vec<f32>> {
    read_f32("reference_features.f32")
        .chunks_exact(FEATURE_LEN)
        .map(<[f32]>::to_vec)
        .collect()
}

struct Gap {
    max: f32,
    mean: f32,
    flips: usize,
    /// Cosine similarity of the two probability tracks. Reported alongside
    /// `max` because they answer different questions: `max` catches a single
    /// bad frame, cosine catches a systematic tilt (a scale or bias error)
    /// that a small per-frame difference would hide.
    cos: f64,
}

fn gap(got: &[f32], want: &[f32], label: &str) -> Gap {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let diffs = || got.iter().zip(want).map(|(a, b)| (a - b).abs());
    let dot: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let na: f64 = got
        .iter()
        .map(|a| f64::from(*a) * f64::from(*a))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = want
        .iter()
        .map(|b| f64::from(*b) * f64::from(*b))
        .sum::<f64>()
        .sqrt();
    let g = Gap {
        max: diffs().fold(0.0f32, f32::max),
        mean: diffs().sum::<f32>() / got.len() as f32,
        flips: got
            .iter()
            .zip(want)
            .filter(|(a, b)| (**a > DEFAULT_THRESHOLD) != (**b > DEFAULT_THRESHOLD))
            .count(),
        cos: if na > 0.0 && nb > 0.0 {
            dot / (na * nb)
        } else {
            0.0
        },
    };
    eprintln!(
        "{label}: n={} max|Δ|={:.3e} mean|Δ|={:.3e} cos={:.9} (1-cos={:.3e}) decision flips={}",
        got.len(),
        g.max,
        g.mean,
        g.cos,
        1.0 - g.cos,
        g.flips
    );
    g
}

/// Score committed reference features through the streaming graph, rebuilding
/// the `[3, 41]` context exactly as the frontend does (zero-padded at the start).
fn score_reference_features(device: rlx_runtime::Device) -> Vec<f32> {
    let rows = reference_features();
    let mut model = TenVadModel::new(device, Shape::Streaming, TenVadWeights::embedded())
        .expect("compile streaming graph");
    let mut state = LstmState::default();
    let mut stack = vec![0.0f32; CONTEXT_FRAMES * FEATURE_LEN];
    rows.iter()
        .map(|row| {
            stack.copy_within(FEATURE_LEN.., 0);
            stack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..].copy_from_slice(row);
            model.step(&stack, &mut state).expect("scoring")
        })
        .collect()
}

/// The committed PCM must still be what `synth` produces, or every reference
/// fixture describes a different signal. `sin()` can differ by an ULP across
/// libm versions, so allow ±1 LSB on a few samples but nothing more.
#[test]
fn fixture_pcm_matches_the_generator() {
    let fixture = fixture_pcm();
    let generated = synth::speech_like_clip();
    assert_eq!(generated.len(), fixture.len(), "clip length changed");
    let mut differing = 0usize;
    for (a, b) in generated.iter().zip(&fixture) {
        let d = (*a as i32 - *b as i32).abs();
        assert!(d <= 1, "sample differs by {d}, not rounding");
        differing += usize::from(d != 0);
    }
    assert!(
        differing * 100 < fixture.len(),
        "{differing} samples differ — the generator changed, regenerate the fixtures"
    );
}

/// The frontend must reproduce the upstream C DSP to the bit — on the platform
/// the fixture was generated for — and to within a couple of ULPs anywhere.
///
/// `src/ooura.rs` is a verbatim transliteration of the reference's Ooura
/// split-radix FFT and `src/pitch.rs` follows its `f32` arithmetic operation for
/// operation, so on arm64 macOS every one of the 30 750 feature values matches
/// exactly. Anything less there means an expression got reassociated.
///
/// Two things bound how far that travels, and neither is this port's doing:
///
/// * **`libm` is not bit-portable.** The features go through `ln`, `log10`,
///   `powf` and `cos`; Apple's and glibc's differ in the last place. On x86_64
///   Linux 99.1% of values still match to the bit; the rest are a last-place
///   `ln` apart, except **feature 40 (pitch)**, where a libm difference enters
///   the band energies and compounds through Levinson and the Viterbi track —
///   measured 151 ULP, i.e. ~1.1e-5 in standardized feature units. ULP is the
///   wrong yardstick for a value that came down a chain that long, so the
///   portable bound below is absolute.
/// * **`-ffp-contract`.** The reference is `src/*.cc` built with `off`; clang
///   on arm64 contracts `a*b + c` into `fma` by default, and `off` / `on` /
///   `fast` each produce a different spectrum from the same source.
///
/// So "bit-exact" is a claim about a stated build on a stated platform. The ULP
/// bound below is what holds everywhere.
#[test]
fn frontend_is_bit_identical_to_the_reference_dsp() {
    use rlx_ten_vad::frontend::{Frontend, pre_emphasis};

    /// The fixture was generated here; only this platform is held to the bit.
    const EXACT_PLATFORM: bool = cfg!(all(target_arch = "aarch64", target_os = "macos"));
    /// Elsewhere: bounded in standardized feature units. A real structural
    /// error (a reassociated expression, a wrong bin edge) is orders of
    /// magnitude larger than this; libm drift is ~1e-5.
    const MAX_ABS: f32 = 1e-3;

    let want = reference_features();
    let pcm: Vec<f32> = fixture_pcm().into_iter().map(f32::from).collect();
    let mut frontend = Frontend::new(TenVadWeights::embedded().core());
    let mut emph = vec![0.0f32; HOP_SIZE];
    let mut prev = 0.0f32;

    // Which feature indices differ at all — `fast-pitch` should move exactly
    // one of them.
    let mut moved = [false; FEATURE_LEN];
    let (mut checked, mut exact, mut worst_ulp) = (0usize, 0usize, 0i64);
    let mut worst_abs = 0.0f32;
    let mut worst_at = (0usize, 0usize);
    for (t, (raw, row)) in pcm.chunks_exact(HOP_SIZE).zip(&want).enumerate() {
        pre_emphasis(raw, &mut prev, &mut emph);
        frontend.push(raw, &emph);
        let got = &frontend.context()[(CONTEXT_FRAMES - 1) * FEATURE_LEN..];
        for (j, (&g, &w)) in got.iter().zip(row).enumerate() {
            checked += 1;
            if g.to_bits() == w.to_bits() {
                exact += 1;
                continue;
            }
            // Same-sign floats compare as ordered integers, so the bit
            // difference *is* the ULP distance.
            moved[j] = true;
            worst_ulp = worst_ulp.max((g.to_bits() as i64 - w.to_bits() as i64).abs());
            let abs = (g - w).abs();
            if abs > worst_abs {
                worst_abs = abs;
                worst_at = (t, j);
            }
        }
    }
    eprintln!(
        "frontend vs upstream C DSP: {exact}/{checked} bit-identical, worst |Δ| {worst_abs:.3e} \
         ({worst_ulp} ULP) at frame {}, feature {}",
        worst_at.0, worst_at.1
    );
    assert_eq!(checked, want.len() * FEATURE_LEN);
    assert!(
        worst_abs < MAX_ABS,
        "frontend is {worst_abs:.3e} off the reference in feature units — that is \
         structural, not libm drift"
    );
    if EXACT_PLATFORM && !cfg!(feature = "fast-pitch") {
        assert_eq!(
            exact, checked,
            "the fixture's own platform must match to the bit; an expression was reassociated"
        );
    }
    if cfg!(feature = "fast-pitch") {
        // The matrix touches exactly one feature — the pitch, index 40. If any
        // other feature moved, the change is not what it claims to be.
        let moved: Vec<usize> = (0..FEATURE_LEN).filter(|&j| moved[j]).collect();
        assert_eq!(
            moved,
            vec![FEATURE_LEN - 1],
            "fast-pitch should perturb only the pitch feature, but features {moved:?} moved"
        );
    }
}

/// Network alone vs onnxruntime on `ten-vad.onnx`, fed identical features.
/// Nothing but `f32` accumulation order separates the two here.
#[test]
fn network_matches_onnxruntime() {
    let g = gap(
        &score_reference_features(rlx_runtime::Device::Cpu),
        &read_f32("reference_probs_onnx.f32"),
        "network/cpu vs onnxruntime",
    );
    assert!(g.max < NET_MAX, "network max|Δ| {:.3e}", g.max);
    assert_eq!(g.flips, 0);
}

/// Whole pipeline — this crate's DSP and network — vs the published model.
#[test]
fn pipeline_matches_published_model() {
    let pcm = fixture_pcm();
    let mut vad = TenVad::new(TenVadConfig::default()).expect("session");
    let g = gap(
        &vad.probabilities_i16(&pcm).expect("scoring"),
        &read_f32("reference_probs_onnx.f32"),
        "pipeline/cpu vs published model",
    );
    assert!(g.max < PIPELINE_MAX, "pipeline max|Δ| {:.3e}", g.max);
    assert!(g.mean < PIPELINE_MEAN, "pipeline mean|Δ| {:.3e}", g.mean);
    assert_eq!(g.flips, 0);
}

#[test]
fn batched_pipeline_matches_published_model() {
    let pcm = fixture_pcm();
    let mut batch = TenVadBatch::new(rlx_runtime::Device::Cpu).expect("session");
    let g = gap(
        &batch.probabilities_i16(&pcm).expect("scoring"),
        &read_f32("reference_probs_onnx.f32"),
        "batched/cpu vs published model",
    );
    assert!(g.max < PIPELINE_MAX, "batched max|Δ| {:.3e}", g.max);
    assert_eq!(g.flips, 0);
}

/// Tracks the prebuilt `libten_vad`, which runs a different build of the
/// network (its binary carries the `coeff.h` DSP tables byte-for-byte but not
/// the weights of its own `ten-vad.onnx`). Decisions must still agree; the
/// probability gap is recorded so a change in it is visible.
#[test]
fn shipped_library_decisions_agree() {
    let pcm = fixture_pcm();
    let mut vad = TenVad::new(TenVadConfig::default()).expect("session");
    let ours = vad.probabilities_i16(&pcm).expect("scoring");
    let g = gap(
        &ours,
        &read_f32("shipped_library_probs.f32"),
        "pipeline vs shipped libten_vad",
    );
    let baseline = gap(
        &read_f32("reference_probs_onnx.f32"),
        &read_f32("shipped_library_probs.f32"),
        "published model vs shipped libten_vad",
    );
    assert_eq!(
        g.flips, 0,
        "voice decisions differ from the shipped library"
    );
    assert!(
        g.max < SHIPPED_MAX,
        "gap to the shipped library grew: {:.3e}",
        g.max
    );
    assert!(
        g.max < baseline.max * 1.5,
        "this port ({:.3e}) should not sit further from the shipped library than \
         the published model itself does ({:.3e})",
        g.max,
        baseline.max
    );
}

/// A short LSTM reset period must land on the same frames in both paths.
#[test]
fn state_reset_lines_up_between_paths() {
    let pcm = fixture_pcm();
    let chunk = 37usize;
    let cfg = TenVadConfig {
        reset_frames: chunk,
        ..Default::default()
    };
    let mut stream = TenVad::new(cfg).expect("session");
    let streamed = stream.probabilities_i16(&pcm).expect("scoring");

    let weights = TenVadWeights::embedded().clone();
    // Dispatch chunk and reset period both 37, so resets land on chunk edges.
    let mut batch =
        TenVadBatch::with_weights(rlx_runtime::Device::Cpu, weights, chunk).expect("session");
    batch.set_reset_frames(chunk);
    let batched = batch.probabilities_i16(&pcm).expect("scoring");
    let g = gap(
        &streamed,
        &batched,
        "streaming vs batched at a 37-frame reset",
    );
    assert!(g.max < 1e-4, "reset boundaries diverge: {:.3e}", g.max);
}

#[test]
fn segments_cover_the_voiced_stretches() {
    let pcm = fixture_pcm();
    let mut vad = TenVad::new(TenVadConfig::default()).expect("session");
    let probs = vad.probabilities_i16(&pcm).expect("scoring");
    let params = rlx_ten_vad::SegmentParams::default();
    let segs = rlx_ten_vad::speech_segments(&probs, pcm.len(), &params);
    assert!(
        !segs.is_empty(),
        "no speech found in a clip with voiced stretches"
    );
    let voiced: usize = segs.iter().map(|s| s.len()).sum();
    // Roughly 2.6 s of the 4 s clip carries a harmonic source.
    assert!(
        voiced > pcm.len() / 4 && voiced < pcm.len(),
        "voiced span {voiced} of {} samples looks wrong: {segs:?}",
        pcm.len()
    );
}

/// Every backend compiled into this build must hit the same parity as the CPU.
#[test]
fn backends_match_the_published_model() {
    let pcm = fixture_pcm();
    let want = read_f32("reference_probs_onnx.f32");
    let devices: Vec<_> = rlx_ten_vad::available_devices()
        .into_iter()
        .filter(|&d| d != rlx_runtime::Device::Cpu)
        .collect();

    if devices.is_empty() {
        eprintln!("no GPU backends in this build — run with --features all-backends");
        return;
    }
    for dev in devices {
        let label = rlx_ten_vad::device_label(dev);
        let g = gap(
            &score_reference_features(dev),
            &want,
            &format!("network/{label} vs onnxruntime"),
        );
        assert!(g.max < NET_MAX, "{label} network max|Δ| {:.3e}", g.max);

        let cfg = TenVadConfig {
            device: dev,
            ..Default::default()
        };
        let mut vad = TenVad::new(cfg).expect("streaming session");
        let g = gap(
            &vad.probabilities_i16(&pcm).expect("scoring"),
            &want,
            &format!("pipeline/{label} vs published model"),
        );
        assert!(
            g.max < PIPELINE_MAX,
            "{label} pipeline max|Δ| {:.3e}",
            g.max
        );
        assert_eq!(g.flips, 0);

        let mut batch = TenVadBatch::new(dev).expect("batched session");
        let g = gap(
            &batch.probabilities_i16(&pcm).expect("scoring"),
            &want,
            &format!("batched/{label} vs published model"),
        );
        assert!(g.max < PIPELINE_MAX, "{label} batched max|Δ| {:.3e}", g.max);
    }
}

/// Chunk size is a latency dial, not a correctness one: carrying the LSTM
/// state across chunks must give the same sequence at every chunk size, and
/// the same one `TenVad` gives frame by frame.
#[test]
fn chunk_size_does_not_change_the_answer() {
    let pcm = fixture_pcm();
    let want = read_f32("reference_probs_onnx.f32");
    for chunk in [1usize, 3, 8, 32, 250, 400] {
        let mut b = TenVadBatch::with_chunk(rlx_runtime::Device::Cpu, chunk).expect("session");
        let g = gap(
            &b.probabilities_i16(&pcm).expect("scoring"),
            &want,
            &format!("chunk={chunk} vs published model"),
        );
        assert!(g.max < PIPELINE_MAX, "chunk {chunk}: max|Δ| {:.3e}", g.max);
        assert_eq!(g.flips, 0, "chunk {chunk} flipped decisions");
    }
}

/// The `no_std` scalar network must agree with the rlx graph.
///
/// `rlx_ten_vad_core::Net` is what the MCU firmware runs and what the FPGA
/// datapath is generated from, so it has to be the same function the accelerated
/// backends compute — not merely close. Both are f32 and structured identically
/// (concatenated gate matmul included), so the only gap is summation order
/// inside the matmuls.
#[test]
fn scalar_net_matches_the_rlx_graph() {
    use rlx_ten_vad::net::Net;

    let rows = reference_features();
    let mut scalar = Net::new(rlx_ten_vad_core::weights::embedded_net());
    let mut stack = vec![0.0f32; CONTEXT_FRAMES * FEATURE_LEN];
    let mut got = Vec::with_capacity(rows.len());
    for row in &rows {
        stack.copy_within(FEATURE_LEN.., 0);
        stack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..].copy_from_slice(row);
        got.push(scalar.forward(&stack));
    }

    let g = gap(
        &got,
        &score_reference_features(rlx_runtime::Device::Cpu),
        "scalar net vs rlx graph",
    );
    assert!(
        g.max < 1e-5,
        "scalar net diverges from the graph: {:.3e}",
        g.max
    );
    assert_eq!(g.flips, 0);

    // And therefore also against the published model.
    let g = gap(
        &got,
        &read_f32("reference_probs_onnx.f32"),
        "scalar net vs published model",
    );
    assert!(g.max < PIPELINE_MAX, "scalar net max|Δ| {:.3e}", g.max);
    assert_eq!(g.flips, 0);
}

#[test]
fn frame_grid_is_16ms() {
    assert_eq!(HOP_SIZE, 256);
    assert!((rlx_ten_vad::session::frame_seconds() - 0.016).abs() < 1e-9);
}

/// The integer-only datapath — what the MCU firmware runs and what the FPGA
/// RTL is checked against — has to stay close enough to the published model to
/// never flip a decision.
///
/// The budget is looser than [`PIPELINE_MAX`] on purpose: int16 weights and
/// Q15 activations cost ~3e-4. That is still below the 9.6e-4 that separates
/// the *shipped Agora binary* from this same model, so the quantised port is
/// nearer the reference than the vendor's own build.
#[test]
fn fixed_net_tracks_the_published_model() {
    use rlx_ten_vad_core::fixed::{FixedNet, ONE};

    const FIXED_MAX: f32 = 5e-4;

    let rows = reference_features();
    let mut net = FixedNet::embedded();
    let mut stack = vec![0i32; CONTEXT_FRAMES * FEATURE_LEN];
    let mut got = Vec::with_capacity(rows.len());
    for row in &rows {
        stack.copy_within(FEATURE_LEN.., 0);
        for (slot, &v) in stack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..]
            .iter_mut()
            .zip(row)
        {
            *slot = (v * ONE as f32).round() as i32;
        }
        got.push(net.forward(&stack) as f32 / ONE as f32);
    }

    let g = gap(&got, &read_f32("reference_probs_onnx.f32"), "fixed net");
    assert!(
        g.max < FIXED_MAX,
        "fixed net max|Δ| {:.3e} exceeds {FIXED_MAX:.0e}",
        g.max
    );
    assert_eq!(g.flips, 0, "fixed net flipped a decision");
    assert!(
        1.0 - g.cos < 1e-6,
        "fixed net cosine distance {:.3e} — a systematic tilt, not just noise",
        1.0 - g.cos
    );

    // And it must beat the shipped binary's own gap to the published model.
    let shipped = gap(
        &read_f32("shipped_library_probs.f32"),
        &read_f32("reference_probs_onnx.f32"),
        "shipped vs published",
    );
    assert!(
        g.max < shipped.max,
        "fixed net {:.3e} is further from the reference than the shipped binary {:.3e}",
        g.max,
        shipped.max
    );
}

/// The integer net must also agree with the f32 scalar net it was derived from,
/// so a regression in either shows up as a divergence rather than as two
/// independently drifting implementations.
#[test]
fn fixed_net_tracks_the_scalar_net() {
    use rlx_ten_vad::net::Net;
    use rlx_ten_vad_core::fixed::{FixedNet, ONE};

    let rows = reference_features();
    let mut fx = FixedNet::embedded();
    let mut fl = Net::new(rlx_ten_vad_core::weights::embedded_net());
    let mut qstack = vec![0i32; CONTEXT_FRAMES * FEATURE_LEN];
    let mut fstack = vec![0.0f32; CONTEXT_FRAMES * FEATURE_LEN];
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for row in &rows {
        qstack.copy_within(FEATURE_LEN.., 0);
        fstack.copy_within(FEATURE_LEN.., 0);
        fstack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..].copy_from_slice(row);
        for (slot, &v) in qstack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..]
            .iter_mut()
            .zip(row)
        {
            *slot = (v * ONE as f32).round() as i32;
        }
        a.push(fx.forward(&qstack) as f32 / ONE as f32);
        b.push(fl.forward(&fstack));
    }
    let g = gap(&a, &b, "fixed vs scalar");
    assert!(g.max < 5e-4, "fixed vs scalar max|Δ| {:.3e}", g.max);
    assert_eq!(g.flips, 0);
}
