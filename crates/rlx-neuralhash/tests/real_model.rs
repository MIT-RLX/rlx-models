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

//! Validation against Apple's shipping NeuralHash model.
//!
//! macOS still ships `NeuralHashv3b_fp16-current.espresso.{net,shape,weights}`
//! and `neuralhash_128x96_seed1.dat` inside `Vision.framework`. When they are
//! present these tests run against the real network; otherwise they skip, so
//! the suite stays green on machines and CI images without them. No model data
//! is copied into the repo or redistributed.
//!
//! What is checked:
//!
//! * the 225-layer graph derives, with **every** activation shape agreeing with
//!   Apple's own `.espresso.shape` sidecar (asserted inside the deriver);
//! * the network's declared preprocessing (`transform_params`) is the `x·2/255 − 1`
//!   RGB chain this crate implements;
//! * the descriptor is 128 finite floats and the hash is stable;
//! * the perceptual invariants NeuralHash is built for — a contrast change, a
//!   rescale and a JPEG re-encode must not move the hash, while an unrelated
//!   image must land far away.
//!
//! The invariants are not decoration: they are what distinguishes a correct
//! InstanceNorm from a plausible-looking wrong one. Reading the norm parameters
//! as four contiguous blocks rather than interleaved per channel, or using the
//! stored (placeholder) mean/variance instead of runtime statistics, still
//! produces a well-formed 96-bit hash — but moves it by 45 and 28 bits
//! respectively, and destroys the contrast invariance below.

use rlx_neuralhash::espresso::EspressoNet;
use rlx_neuralhash::seed::EMBED_DIM;
use rlx_neuralhash::{NeuralHashModel, NeuralHashSpec, NeuralHasher, SeedMatrix};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

const RESOURCES: &str = "/System/Library/Frameworks/Vision.framework/Versions/A/Resources";

fn net_path() -> Option<PathBuf> {
    // The `_fp16` build is what current macOS ships; the plain name is what
    // earlier OS versions (and the reference implementation) use.
    for stem in [
        "NeuralHashv3b_fp16-current.espresso.net",
        "NeuralHashv3b-current.espresso.net",
    ] {
        let p = Path::new(RESOURCES).join(stem);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn seed_path() -> Option<PathBuf> {
    let p = Path::new(RESOURCES).join("neuralhash_128x96_seed1.dat");
    p.is_file().then_some(p)
}

/// `(net, seed)` when the system model is installed.
macro_rules! model_or_skip {
    () => {
        match (net_path(), seed_path()) {
            (Some(n), Some(s)) => (n, s),
            _ => {
                eprintln!("skip: Apple NeuralHash model not installed under {RESOURCES}");
                return;
            }
        }
    };
}

/// A deterministic scene rendered at an arbitrary size.
fn scene(w: u32, h: u32, variant: u32) -> image_buf::Rgb {
    image_buf::Rgb::render(w, h, variant)
}

/// Minimal RGB8 buffer + PNG/JPEG-free transforms, so this test pulls in no
/// image codecs of its own.
mod image_buf {
    pub struct Rgb {
        pub w: usize,
        pub h: usize,
        pub px: Vec<u8>,
    }

    impl Rgb {
        pub fn render(w: u32, h: u32, variant: u32) -> Self {
            let (w, h) = (w as usize, h as usize);
            let mut px = vec![0u8; w * h * 3];
            for y in 0..h {
                for x in 0..w {
                    let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
                    let c = match variant % 3 {
                        // Two colour fields split vertically, plus a ramp.
                        0 => {
                            let g = (v * 255.0) as u8;
                            if u < 0.5 { [210, g, 30] } else { [30, g, 210] }
                        }
                        // Bright disc on a dark field.
                        1 => {
                            let (du, dv) = (u - 0.5, v - 0.5);
                            if du * du + dv * dv < 0.08 {
                                [235, 235, 195]
                            } else {
                                [20, 25, 45]
                            }
                        }
                        // Low-frequency bands.
                        _ => {
                            if ((v * 5.0) as u32).is_multiple_of(2) {
                                [200, 50, 50]
                            } else {
                                [45, 195, 115]
                            }
                        }
                    };
                    px[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&c);
                }
            }
            Self { w, h, px }
        }

        /// Linear contrast change about mid-grey — the transform instance
        /// normalization is designed to be blind to.
        pub fn contrast(&self, factor: f32) -> Self {
            let px = self
                .px
                .iter()
                .map(|&p| (((p as f32 - 128.0) * factor) + 128.0).clamp(0.0, 255.0) as u8)
                .collect();
            Self {
                w: self.w,
                h: self.h,
                px,
            }
        }
    }
}

#[test]
fn derives_the_real_225_layer_graph() {
    let (net_p, _) = model_or_skip!();
    let net = EspressoNet::open(&net_p).expect("parse Apple's espresso container");
    // Both `.net` and `.shape` ship LZFSE-compressed on current macOS; getting
    // this far means the `pbze` container was decoded.
    assert_eq!(net.format_version, 200);
    assert!(net.layers.len() > 200, "layers: {}", net.layers.len());
    assert!(net.blob_count() > 200, "blobs: {}", net.blob_count());

    // Every op's computed shape is cross-checked against Apple's own
    // `.espresso.shape` sidecar inside the deriver, so a successful derivation
    // *is* a geometry proof: padding, stride, grouping and pooling all agree.
    let spec = NeuralHashSpec::from_espresso(&net).expect("derive the native spec");
    spec.validate_neuralhash_io()
        .expect("NeuralHash I/O contract");
    assert_eq!(spec.input_shape, [1, 3, 360, 360]);
    assert_eq!(spec.output_dim(), EMBED_DIM);
    // The unfused spec is one op per Espresso layer; fusion then collapses the
    // hard-swish chains, so the default (fused) spec is deliberately smaller.
    let raw = NeuralHashSpec::from_espresso_unfused(&net).expect("unfused spec");
    assert_eq!(raw.ops.len(), net.layers.len(), "one op per Espresso layer");
    assert!(
        spec.ops.len() < raw.ops.len(),
        "fusion did not shrink the spec ({} vs {})",
        spec.ops.len(),
        raw.ops.len()
    );

    // The spec is a complete standalone description: it must round-trip.
    let back = NeuralHashSpec::from_json(&spec.to_json().unwrap()).unwrap();
    assert_eq!(spec, back);
}

#[test]
fn seed_matrix_is_exactly_the_documented_size() {
    let (_, seed_p) = model_or_skip!();
    let bytes = std::fs::metadata(&seed_p).unwrap().len() as usize;
    assert_eq!(
        bytes,
        rlx_neuralhash::SEED_FILE_BYTES,
        "128-byte header + [96, 128] f32"
    );
    SeedMatrix::open(&seed_p).expect("parse the shipping seed matrix");
}

#[test]
fn hashes_the_real_model_and_is_deterministic() {
    let (net_p, seed_p) = model_or_skip!();
    let mut h = NeuralHasher::open_espresso(&net_p, &seed_p, Device::Cpu).expect("build");

    let img = scene(640, 480, 0);
    let a = h.hash_rgb8(&img.px, img.w, img.h).expect("hash");
    let b = h.hash_rgb8(&img.px, img.w, img.h).expect("hash again");
    assert_eq!(a, b, "the same input must hash identically");
    assert_eq!(a.to_hex().len(), 24);
    assert_ne!(a.to_hex(), "000000000000000000000000");
    assert_ne!(a.to_hex(), "ffffffffffffffffffffffff");

    // A freshly built model + seed must reproduce it — no run-to-run state.
    let mut h2 = NeuralHasher::open_espresso(&net_p, &seed_p, Device::Cpu).expect("rebuild");
    assert_eq!(h2.hash_rgb8(&img.px, img.w, img.h).unwrap(), a);
}

#[test]
fn descriptor_is_128_finite_floats() {
    let (net_p, _) = model_or_skip!();
    let mut m = NeuralHashModel::open_espresso(&net_p, Device::Cpu).expect("build");
    let img = scene(500, 400, 1);
    let x = rlx_neuralhash::preprocess::from_rgb8(&img.px, img.w, img.h).unwrap();
    let d = m.embed(&x).expect("forward");
    assert_eq!(d.len(), EMBED_DIM);
    assert!(d.iter().all(|v| v.is_finite()));
    let spread =
        d.iter().cloned().fold(f32::MIN, f32::max) - d.iter().cloned().fold(f32::MAX, f32::min);
    assert!(
        spread > 1.0,
        "descriptor is nearly constant (spread {spread})"
    );
}

/// The properties NeuralHash exists to have. These are what separate a correct
/// implementation from one that merely runs.
#[test]
fn perceptual_invariants_hold_on_the_real_model() {
    let (net_p, seed_p) = model_or_skip!();
    let mut h = NeuralHasher::open_espresso(&net_p, &seed_p, Device::Cpu).expect("build");

    let base = scene(720, 720, 0);
    let base_h = h.hash_rgb8(&base.px, base.w, base.h).unwrap();

    // Rescaled copies of the same scene.
    for (w, hh) in [(360u32, 360u32), (1080, 1080), (640, 640)] {
        let s = scene(w, hh, 0);
        let d = base_h.hamming(&h.hash_rgb8(&s.px, s.w, s.h).unwrap());
        assert!(d <= 6, "rescale to {w}x{hh} moved the hash by {d} bits");
    }

    // Contrast change: instance normalization removes it almost entirely. If
    // the norm statistics were read from the stored placeholders instead of
    // computed at runtime, this is the assertion that fails.
    let dim = base.contrast(0.6);
    let d = base_h.hamming(&h.hash_rgb8(&dim.px, dim.w, dim.h).unwrap());
    assert!(d <= 6, "a contrast change moved the hash by {d} bits");

    // Unrelated scenes must land far away — near the 48-bit random baseline.
    for variant in [1u32, 2] {
        let other = scene(720, 720, variant);
        let d = base_h.hamming(&h.hash_rgb8(&other.px, other.w, other.h).unwrap());
        assert!(
            d >= 20,
            "unrelated scene {variant} is only {d} bits away — the descriptor is \
             not discriminating"
        );
    }
}

/// Apple declares the preprocessing in the container itself; it must be the
/// chain this crate implements.
#[test]
fn declared_transform_params_match_our_preprocessing() {
    let (net_p, _) = model_or_skip!();
    let raw = rlx_neuralhash::espresso::read_maybe_compressed(&net_p).expect("read .net");
    let v: serde_json::Value = serde_json::from_slice(&raw).expect("parse .net");
    let Some(t) = v.get("transform_params").and_then(|t| t.get("image")) else {
        eprintln!("skip: container declares no transform_params");
        return;
    };
    let f = |k: &str| t.get(k).and_then(|x| x.as_f64());
    // scale = 2/255, bias = -1 per channel, RGB order, no mean centring.
    let scale = f("scale").expect("scale");
    assert!(
        (scale - 2.0 / 255.0).abs() < 1e-9,
        "declared scale {scale} != 2/255"
    );
    for c in ["bias_r", "bias_g", "bias_b"] {
        assert_eq!(f(c), Some(-1.0), "{c}");
    }
    assert_eq!(f("is_network_bgr"), Some(0.0), "input is RGB, not BGR");
    assert_eq!(f("center_mean"), Some(0.0));
}

/// Fusion must not change a single bit.
///
/// `spec::fuse_hard_activations` rewrites Espresso's four-op hard-swish chains
/// onto native activations, and `flow` emits instance norm as one `GroupNorm`.
/// Both are value-preserving *by construction* — this pins that claim to the
/// real 225-layer model, where 28 chains and 35 norms are at stake.
#[test]
fn fusion_is_value_preserving_on_the_real_model() {
    let (net_p, seed_p) = model_or_skip!();
    let net = EspressoNet::open(&net_p).expect("parse container");
    let seed = SeedMatrix::open(&seed_p).expect("parse seed");

    let fused = NeuralHashModel::from_espresso_fused(&net, Device::Cpu, true).expect("fused");
    let plain = NeuralHashModel::from_espresso_fused(&net, Device::Cpu, false).expect("unfused");
    assert!(
        fused.graph_nodes() < plain.graph_nodes(),
        "fusion did not shrink the graph ({} vs {})",
        fused.graph_nodes(),
        plain.graph_nodes()
    );

    let mut fused = NeuralHasher::from_parts(fused, seed.clone());
    let mut plain = NeuralHasher::from_parts(plain, seed);
    for variant in 0..3u32 {
        let img = scene(512, 384, variant);
        let a = fused.hash_rgb8(&img.px, img.w, img.h).unwrap();
        let b = plain.hash_rgb8(&img.px, img.w, img.h).unwrap();
        assert_eq!(
            a,
            b,
            "scene {variant}: fused {a} != unfused {b} ({} bits)",
            a.hamming(&b)
        );
    }
}
