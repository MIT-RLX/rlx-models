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

//! Cross-backend agreement for the NeuralHash graph.
//!
//! Every backend compiled into this build runs the same network and must
//! produce the same 96-bit hash as the CPU reference. A hash is a brutal
//! comparison — one mis-lowered op flips bits — which is exactly what we want
//! from a backend test.
//!
//! Two fixtures:
//!
//! * [`rlx_neuralhash::synth`] — synthetic weights, real geometry, and every
//!   layer form the shipping model uses. Runs anywhere, so this is what
//!   validates CUDA and ROCm hosts, which have no `Vision.framework`.
//! * Apple's installed model, when present (macOS only).
//!
//! Enable backends with the matching cargo features, e.g.
//! `cargo test -p rlx-neuralhash --release --test backends --features all-backends`.
//! Backends that are not compiled in are reported as skipped rather than
//! silently passing.

use rlx_neuralhash::{NeuralHash, NeuralHashModel, NeuralHasher, SeedMatrix, synth};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

const RESOURCES: &str = "/System/Library/Frameworks/Vision.framework/Versions/A/Resources";

/// Backends compiled into this build *and* actually present on this host, CPU
/// first.
///
/// The two halves are separate: `--features all-backends` compiles CUDA and
/// ROCm in everywhere, including on a Mac that has neither driver. Filtering on
/// [`rlx_runtime::device_ext::is_available`] keeps the same command line
/// working on every host — the machine-specific part is which backends get
/// exercised, reported below rather than assumed.
fn backends() -> Vec<(&'static str, Device)> {
    // `mut` is only used when a backend feature is on; with default features
    // every push below is cfg'd out and the binding is never mutated. The
    // `allow` keeps both ends honest — without it this is either a hard error
    // under `--features metal` or an `unused_mut` warning without it.
    #[allow(unused_mut)]
    let mut all = vec![("cpu", Device::Cpu)];
    #[cfg(feature = "metal")]
    all.push(("metal", Device::Metal));
    #[cfg(feature = "mlx")]
    all.push(("mlx", Device::Mlx));
    #[cfg(feature = "cuda")]
    all.push(("cuda", Device::Cuda));
    #[cfg(feature = "rocm")]
    all.push(("rocm", Device::Rocm));
    #[cfg(feature = "gpu")]
    all.push(("gpu", Device::Gpu));
    #[cfg(feature = "vulkan")]
    all.push(("vulkan", Device::Vulkan));

    let (present, absent): (Vec<_>, Vec<_>) = all
        .into_iter()
        .partition(|(_, d)| rlx_runtime::device_ext::is_available(*d));
    if !absent.is_empty() {
        eprintln!(
            "backends compiled in but not present on this host: {}",
            absent
                .iter()
                .map(|(n, _)| *n)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    present
}

/// A deterministic test scene.
fn scene(w: usize, h: usize, variant: u32) -> Vec<u8> {
    let mut px = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
            let c = match variant % 3 {
                0 => {
                    let g = (v * 255.0) as u8;
                    if u < 0.5 { [210, g, 30] } else { [30, g, 210] }
                }
                1 => {
                    let (du, dv) = (u - 0.5, v - 0.5);
                    if du * du + dv * dv < 0.08 {
                        [235, 235, 195]
                    } else {
                        [20, 25, 45]
                    }
                }
                _ => [(u * 255.0) as u8, (v * 255.0) as u8, 130],
            };
            px[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&c);
        }
    }
    px
}

/// Run one image through `device`, returning the hash and descriptor.
fn run_on(
    net: &rlx_neuralhash::EspressoNet,
    seed: &SeedMatrix,
    device: Device,
    px: &[u8],
    w: usize,
    h: usize,
) -> (NeuralHash, Vec<f32>) {
    let model = NeuralHashModel::from_espresso(net, device)
        .unwrap_or_else(|e| panic!("build for {device:?}: {e:#}"));
    let mut hasher = NeuralHasher::from_parts(model, seed.clone());
    let input = rlx_neuralhash::preprocess::from_rgb8(px, w, h).unwrap();
    let d = hasher
        .model_mut()
        .embed(&input)
        .unwrap_or_else(|e| panic!("forward on {device:?}: {e:#}"));
    let hash = hasher.hash_embedding(&d).unwrap();
    (hash, d)
}

/// Largest absolute difference between two descriptors.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn every_compiled_backend_agrees_with_cpu() {
    let net = synth::synthetic_net(0x4e48).expect("synthetic container");
    let seed = synth::synthetic_seed(0x4e48);
    let devices = backends();
    assert!(!devices.is_empty());

    let (w, h) = (640usize, 480usize);
    // Several scenes: a single image can coincidentally agree.
    for variant in 0..3u32 {
        let px = scene(w, h, variant);
        let (want, want_d) = run_on(&net, &seed, Device::Cpu, &px, w, h);
        for (name, dev) in devices.iter().skip(1) {
            let (got, got_d) = run_on(&net, &seed, *dev, &px, w, h);
            let bits = want.hamming(&got);
            assert_eq!(
                got,
                want,
                "scene {variant}: {name} produced {got} but cpu produced {want} \
                 ({bits} bits differ; max|Δdescriptor| = {:.3e})",
                max_abs_diff(&want_d, &got_d)
            );
        }
    }

    eprintln!(
        "backends verified against cpu: {}",
        devices
            .iter()
            .map(|(n, _)| *n)
            .collect::<Vec<_>>()
            .join(", ")
    );
    // Make an all-CPU build visible rather than letting it look like coverage.
    if devices.len() == 1 {
        eprintln!(
            "note: only the cpu backend was compiled in — rebuild with \
             --features all-backends (or metal/mlx/cuda/rocm/gpu/vulkan) to widen this"
        );
    }
}

#[test]
fn every_compiled_backend_is_deterministic() {
    let net = synth::synthetic_net(0x1234).expect("synthetic container");
    let seed = synth::synthetic_seed(0x1234);
    let (w, h) = (512usize, 512usize);
    let px = scene(w, h, 1);
    for (name, dev) in backends() {
        let (a, _) = run_on(&net, &seed, dev, &px, w, h);
        let (b, _) = run_on(&net, &seed, dev, &px, w, h);
        assert_eq!(a, b, "{name} is not deterministic across runs");
    }
}

fn real_model() -> Option<(PathBuf, PathBuf)> {
    let seed = Path::new(RESOURCES).join("neuralhash_128x96_seed1.dat");
    for stem in [
        "NeuralHashv3b_fp16-current.espresso.net",
        "NeuralHashv3b-current.espresso.net",
    ] {
        let net = Path::new(RESOURCES).join(stem);
        if net.is_file() && seed.is_file() {
            return Some((net, seed));
        }
    }
    None
}

/// Same check against Apple's real weights, where they exist.
#[test]
fn every_compiled_backend_agrees_on_the_real_model() {
    let Some((net_p, seed_p)) = real_model() else {
        eprintln!("skip: Apple NeuralHash model not installed under {RESOURCES}");
        return;
    };
    let net = rlx_neuralhash::EspressoNet::open(&net_p).expect("parse container");
    let seed = SeedMatrix::open(&seed_p).expect("parse seed");

    let (w, h) = (640usize, 480usize);
    for variant in 0..2u32 {
        let px = scene(w, h, variant);
        let (want, want_d) = run_on(&net, &seed, Device::Cpu, &px, w, h);
        for (name, dev) in backends().iter().skip(1) {
            let (got, got_d) = run_on(&net, &seed, *dev, &px, w, h);
            assert_eq!(
                got,
                want,
                "real model, scene {variant}: {name} = {got}, cpu = {want} \
                 ({} bits differ; max|Δdescriptor| = {:.3e})",
                want.hamming(&got),
                max_abs_diff(&want_d, &got_d)
            );
        }
    }
}
