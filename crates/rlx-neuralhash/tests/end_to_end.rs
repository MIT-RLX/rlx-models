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

//! End-to-end coverage of the whole crate at real NeuralHash geometry, on a
//! synthetic Espresso container.
//!
//! Apple's `NeuralHashv3b` weights cannot be redistributed, so these tests
//! stand up a MobileNetV3-shaped network with the same input/output contract
//! (`[1, 3, 360, 360]` → 128 floats) and deterministic pseudo-random weights.
//! That exercises every seam the real model uses — strided conv, asymmetric
//! SAME padding, depthwise grouping, hard-swish, squeeze-excite broadcast,
//! average pooling, the inner-product head, the seed projection and the bit
//! packing — without any vendor file.

use half::f16;
use rlx_neuralhash::espresso::{BlobShape, EspressoNet};
use rlx_neuralhash::seed::EMBED_DIM;
use rlx_neuralhash::{
    HASH_BITS, NeuralHash, NeuralHashModel, NeuralHashSpec, NeuralHasher, SeedMatrix,
};
use rlx_runtime::Device;
use std::collections::HashMap;

/// Deterministic weights — an LCG, so a failure is always reproducible.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        // Centred on 0 so activations do not saturate, and small enough that
        // the f16 round-trip is lossless-ish.
        ((self.0 >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * scale).collect()
    }
}

fn f16b(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| f16::from_f32(*x).to_le_bytes())
        .collect()
}
fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn container(entries: &[(u64, Vec<u8>)]) -> Vec<u8> {
    let mut out = (entries.len() as u64).to_le_bytes().to_vec();
    for (idx, data) in entries {
        out.extend(idx.to_le_bytes());
        out.extend((data.len() as u64).to_le_bytes());
    }
    for (_, data) in entries {
        out.extend(data);
    }
    out
}

/// A NeuralHash-shaped MobileNetV3 stub: 360×360×3 in, 128 floats out.
const NET_JSON: &str = r#"{
  "storage": "synthetic.espresso.weights",
  "format_version": 200,
  "layers": [
    {"type":"input","name":"image","top":"image","bottom":""},

    {"type":"convolution","name":"stem","top":"s","bottom":"image",
     "C":8,"K":3,"Nx":3,"Ny":3,"n_groups":1,"stride_x":2,"stride_y":2,
     "pad_t":1,"pad_b":1,"pad_l":1,"pad_r":1,
     "has_biases":1,"fused_relu":1,"blob_weights_f16":0,"blob_biases":1},

    {"type":"convolution","name":"dw","top":"d","bottom":"s",
     "C":8,"K":8,"Nx":3,"Ny":3,"n_groups":8,"stride_x":2,"stride_y":2,
     "pad_mode":1,"has_biases":0,"blob_weights_f16":2},

    {"type":"convolution","name":"pw","top":"p","bottom":"d",
     "C":16,"K":8,"Nx":1,"Ny":1,"n_groups":1,
     "has_biases":0,"fused_hard_swish":1,"blob_weights_f16":3},

    {"type":"pool","name":"se_squeeze","top":"sq","bottom":"p",
     "avg_or_max":0,"is_global":1},
    {"type":"convolution","name":"se_fc1","top":"se1","bottom":"sq",
     "C":4,"K":16,"Nx":1,"Ny":1,"has_biases":1,"fused_relu":1,
     "blob_weights_f16":4,"blob_biases":5},
    {"type":"convolution","name":"se_fc2","top":"se2","bottom":"se1",
     "C":16,"K":4,"Nx":1,"Ny":1,"has_biases":1,
     "blob_weights_f16":6,"blob_biases":7},
    {"type":"hard_sigmoid","name":"se_gate","top":"g","bottom":"se2"},
    {"type":"elementwise","name":"se_scale","top":"e","bottom":"p,g","operation":1},

    {"type":"pool","name":"down","top":"pool","bottom":"e",
     "avg_or_max":0,"size_x":2,"size_y":2,"stride_x":2,"stride_y":2},
    {"type":"pool","name":"gap","top":"gp","bottom":"pool",
     "avg_or_max":0,"is_global":1},

    {"type":"inner_product","name":"head","top":"descriptor","bottom":"gp",
     "nB":16,"nC":128,"blob_weights_f16":8,"blob_biases":9}
  ]
}"#;

fn synthetic_net() -> EspressoNet {
    let mut r = Lcg::new(0x4e48_4153_4831_3233);
    let weights = container(&[
        (0, f16b(&r.vec(8 * 3 * 3 * 3, 0.4))), // stem w [8,3,3,3]
        (1, f32b(&r.vec(8, 0.1))),             // stem b
        (2, f16b(&r.vec(8 * 3 * 3, 0.5))),     // dw w [8,1,3,3] (K=1 per group)
        (3, f16b(&r.vec(16 * 8, 0.5))),        // pw w [16,8,1,1]
        (4, f16b(&r.vec(4 * 16, 0.5))),        // se_fc1 w [4,16,1,1]
        (5, f32b(&r.vec(4, 0.1))),
        (6, f16b(&r.vec(16 * 4, 0.5))), // se_fc2 w [16,4,1,1]
        (7, f32b(&r.vec(16, 0.1))),
        (8, f16b(&r.vec(128 * 16, 0.6))), // head w [nC=128, nB=16]
        (9, f32b(&r.vec(128, 0.2))),      // head b
    ]);
    let mut shapes = HashMap::new();
    shapes.insert(
        "image".to_string(),
        BlobShape {
            n: 1,
            c: 3,
            h: 360,
            w: 360,
        },
    );
    EspressoNet::from_parts(NET_JSON, &weights, shapes).expect("parse synthetic espresso net")
}

/// A synthetic but non-degenerate `[96, 128]` projection.
fn synthetic_seed() -> SeedMatrix {
    let mut r = Lcg::new(0x5345_4544_3100_0001);
    SeedMatrix::from_rows(r.vec(HASH_BITS * EMBED_DIM, 1.0)).unwrap()
}

/// A deterministic "scene" sampled in normalized `[0, 1)` coordinates, so the
/// same `variant` rendered at two resolutions is genuinely the same picture.
///
/// The scenes differ in *global* statistics (colour balance, coverage), not
/// just in phase. This stub network ends in a global average pool, so a
/// high-frequency texture and a phase-shifted copy of it are — correctly —
/// indistinguishable to it; only the full-depth model keeps that detail.
fn scene(u: f32, v: f32, variant: u32) -> [u8; 3] {
    match variant % 4 {
        // Red left / blue right, with a green vertical ramp.
        0 => {
            let g = (v * 255.0) as u8;
            if u < 0.5 { [220, g, 20] } else { [20, g, 220] }
        }
        // Bright disc on a dark field.
        1 => {
            let (du, dv) = (u - 0.5, v - 0.5);
            if du * du + dv * dv < 0.09 {
                [240, 240, 200]
            } else {
                [15, 20, 40]
            }
        }
        // Wide horizontal bands (low frequency: survives downsampling).
        2 => {
            if ((v * 4.0) as u32).is_multiple_of(2) {
                [200, 40, 40]
            } else {
                [40, 200, 120]
            }
        }
        // Smooth two-axis gradient.
        _ => [(u * 255.0) as u8, (v * 255.0) as u8, 128],
    }
}

/// Render `scene` at `w × h` as HWC RGB8.
fn image(w: usize, h: usize, variant: u32) -> Vec<u8> {
    let mut px = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let c = scene(x as f32 / w as f32, y as f32 / h as f32, variant);
            let i = (y * w + x) * 3;
            px[i..i + 3].copy_from_slice(&c);
        }
    }
    px
}

fn hasher() -> NeuralHasher {
    let net = synthetic_net();
    let model = NeuralHashModel::from_espresso(&net, Device::Cpu).expect("build native model");
    NeuralHasher::from_parts(model, synthetic_seed())
}

#[test]
fn derived_spec_matches_the_neuralhash_contract() {
    let net = synthetic_net();
    let spec = NeuralHashSpec::from_espresso(&net).unwrap();
    assert_eq!(spec.input_shape, [1, 3, 360, 360]);
    assert_eq!(spec.output_dim(), EMBED_DIM);
    spec.validate_neuralhash_io().unwrap();

    // Spot-check the geometry chain the espresso fields imply.
    let shape_of = |blob: &str| spec.ops.iter().find(|o| o.out == blob).unwrap().shape;
    assert_eq!(shape_of("s"), [1, 8, 180, 180], "stride-2 3x3 pad-1 stem");
    assert_eq!(shape_of("d"), [1, 8, 90, 90], "depthwise, SAME padding");
    assert_eq!(shape_of("sq"), [1, 16, 1, 1], "squeeze");
    assert_eq!(shape_of("e"), [1, 16, 90, 90], "excite broadcasts back up");
    assert_eq!(shape_of("pool"), [1, 16, 45, 45]);
    assert_eq!(shape_of("descriptor"), [1, 128, 1, 1]);
}

#[test]
fn full_pipeline_produces_a_96_bit_hash() {
    let mut h = hasher();
    let img = image(640, 480, 0);
    let hash = h.hash_rgb8(&img, 640, 480).expect("hash image");
    let hex = hash.to_hex();
    assert_eq!(hex.len(), 24, "96 bits = 24 hex chars: {hex}");
    assert!(hex.chars().all(|c| c.is_ascii_hexdigit()), "{hex}");
    assert_eq!(NeuralHash::from_hex(&hex).unwrap(), hash);
    // A hash of all-zeros or all-ones would mean the descriptor collapsed.
    assert_ne!(hex, "000000000000000000000000");
    assert_ne!(hex, "ffffffffffffffffffffffff");
}

#[test]
fn hashing_is_deterministic() {
    let mut h = hasher();
    let img = image(320, 200, 5);
    let a = h.hash_rgb8(&img, 320, 200).unwrap();
    let b = h.hash_rgb8(&img, 320, 200).unwrap();
    assert_eq!(a, b, "same input must give the same hash");
    assert_eq!(a.hamming(&b), 0);

    // A fresh model + seed must reproduce it too (no hidden run-to-run state).
    let mut h2 = hasher();
    assert_eq!(h2.hash_rgb8(&img, 320, 200).unwrap(), a);
}

#[test]
fn different_images_hash_differently() {
    let mut h = hasher();
    let hashes: Vec<NeuralHash> = (0..4)
        .map(|v| h.hash_rgb8(&image(256, 256, v), 256, 256).unwrap())
        .collect();
    for (i, a) in hashes.iter().enumerate() {
        for (j, b) in hashes.iter().enumerate().skip(i + 1) {
            assert_ne!(a, b, "scenes {i} and {j} collided ({a})");
        }
    }
}

#[test]
fn descriptor_is_finite_and_128_wide() {
    let net = synthetic_net();
    let mut model = NeuralHashModel::from_espresso(&net, Device::Cpu).unwrap();
    let input = rlx_neuralhash::preprocess::from_rgb8(&image(400, 300, 3), 400, 300).unwrap();
    let d = model.embed(&input).unwrap();
    assert_eq!(d.len(), EMBED_DIM);
    assert!(d.iter().all(|v| v.is_finite()));
    // A constant descriptor would make every image hash identically.
    let spread =
        d.iter().cloned().fold(f32::MIN, f32::max) - d.iter().cloned().fold(f32::MAX, f32::min);
    assert!(
        spread > 1e-3,
        "descriptor is nearly constant (spread {spread})"
    );
}

#[test]
fn exported_spec_rebuilds_the_same_model() {
    // The spec is meant to be a complete, standalone architecture description:
    // round-tripping it through JSON and rebuilding must not change a bit.
    let net = synthetic_net();
    let spec = NeuralHashSpec::from_espresso(&net).unwrap();
    let reloaded = NeuralHashSpec::from_json(&spec.to_json().unwrap()).unwrap();
    assert_eq!(spec, reloaded);

    let img = image(300, 300, 11);
    let mut direct = hasher();
    let expected = direct.hash_rgb8(&img, 300, 300).unwrap();

    let weights = rlx_neuralhash::weights::from_espresso(&net).unwrap();
    let model = NeuralHashModel::from_spec(reloaded, weights, Device::Cpu).unwrap();
    let mut rebuilt = NeuralHasher::from_parts(model, synthetic_seed());
    assert_eq!(rebuilt.hash_rgb8(&img, 300, 300).unwrap(), expected);
}

#[test]
fn resized_copies_stay_close_in_hamming_distance() {
    // The point of a perceptual hash: a rescaled copy of the same picture must
    // land far nearer than an unrelated image. Exact equality is not expected
    // even in the reference implementation.
    let mut h = hasher();
    let base = h.hash_rgb8(&image(720, 720, 0), 720, 720).unwrap();
    let scaled = h.hash_rgb8(&image(360, 360, 0), 360, 360).unwrap();
    let unrelated = h.hash_rgb8(&image(720, 720, 1), 720, 720).unwrap();

    let near = base.hamming(&scaled);
    let far = base.hamming(&unrelated);
    assert!(
        near < far,
        "a rescaled copy ({near} bits) should be closer than an unrelated image ({far} bits)"
    );
}

#[test]
fn descriptor_dot_seed_matches_the_reference_formula() {
    // Recompute `seed1.dot(embedding)` + binary step by hand and compare with
    // what the pipeline produced, pinning the two against each other.
    let net = synthetic_net();
    let mut model = NeuralHashModel::from_espresso(&net, Device::Cpu).unwrap();
    let seed = synthetic_seed();
    let input = rlx_neuralhash::preprocess::from_rgb8(&image(500, 400, 7), 500, 400).unwrap();
    let d = model.embed(&input).unwrap();

    let rows = seed.as_slice();
    let mut bits = String::with_capacity(HASH_BITS);
    for r in 0..HASH_BITS {
        let mut acc = 0f64;
        for c in 0..EMBED_DIM {
            acc += rows[r * EMBED_DIM + c] as f64 * d[c] as f64;
        }
        bits.push(if acc as f32 >= 0.0 { '1' } else { '0' });
    }
    let expected: Vec<bool> = bits.chars().map(|c| c == '1').collect();

    let h = NeuralHasher::from_parts(model, synthetic_seed());
    let got = h.hash_embedding(&d).unwrap();
    assert_eq!(got.bits().to_vec(), expected);
}

/// Fusion parity on the synthetic container — runs everywhere, including hosts
/// with no `Vision.framework`.
#[test]
fn fusion_is_value_preserving() {
    use rlx_neuralhash::synth;
    let net = synth::synthetic_net(0xF05E).unwrap();
    let seed = synth::synthetic_seed(0xF05E);
    let fused = NeuralHashModel::from_espresso_fused(&net, Device::Cpu, true).unwrap();
    let plain = NeuralHashModel::from_espresso_fused(&net, Device::Cpu, false).unwrap();
    assert!(
        fused.graph_nodes() < plain.graph_nodes(),
        "fusion did not shrink the graph ({} vs {})",
        fused.graph_nodes(),
        plain.graph_nodes()
    );
    let mut fused = NeuralHasher::from_parts(fused, seed.clone());
    let mut plain = NeuralHasher::from_parts(plain, seed);
    for variant in 0..4u32 {
        let px = image(400, 300, variant);
        assert_eq!(
            fused.hash_rgb8(&px, 400, 300).unwrap(),
            plain.hash_rgb8(&px, 400, 300).unwrap(),
            "variant {variant}"
        );
    }
}
