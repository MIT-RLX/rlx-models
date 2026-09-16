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

//! `rlx-neuralhash` command line: hash images, inspect the network, compare hashes.
//!
//! A single-image run mirrors `nnhash.py` — same inputs, same 24-character hex
//! on stdout — except the network is Apple's Espresso container read directly
//! rather than a converted ONNX file.

use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;

use crate::espresso::EspressoNet;
use crate::hash::NeuralHash;
use crate::model::NeuralHashModel;
use crate::spec::NeuralHashSpec;
use crate::{NeuralHasher, seed::SeedMatrix};

const HELP: &str = "\
rlx-neuralhash — Apple NeuralHash perceptual image hashing on RLX

Usage:
  rlx-neuralhash --net <NeuralHashv3b-current.espresso.net> --seed <seed.dat> \\
                 --image <img> [--image <img> ...]
  rlx-neuralhash --net <net> --inspect
  rlx-neuralhash --compare <hash-a> <hash-b>

Model:
  --net <path>       Apple's NeuralHashv3b espresso net. The sibling .shape and
                     .weights files are located automatically, and the LZFSE
                     `pbze` container current macOS uses is decompressed inline.
                     The architecture is read from this container and built as
                     a native rlx graph — no ONNX runtime is involved.
                     On macOS: /System/Library/Frameworks/Vision.framework/
                     Versions/A/Resources/NeuralHashv3b_fp16-current.espresso.net
  --seed <path>      neuralhash_128x96_seed1.dat ([96, 128] output projection),
                     alongside the model in the same Resources directory.

Hashing:
  --image <path>     Image to hash. Repeat for a batch; one line per image.
  --device <dev>     cpu | metal | mlx | cuda | rocm | gpu (wgpu) | vulkan
                     [default: cpu]
  --embedding        Also print the 128-float descriptor (pre-projection).
  --scores           Also print the 96 projected scores (pre-threshold). Values
                     near zero are the bits that may differ between backends.
  --dump <path>      Write the descriptors as little-endian f32, one [128]
                     block per image, in --image order.
  --json             Emit one JSON object per image instead of bare hex.
  --distances        Print the pairwise Hamming distance matrix over the batch.
  --bench <n>        Time n forward passes of the first image (after a warm-up)
                     and report graph size, compile time and per-image latency.
  --no-fuse          Emit Espresso's literal hard-swish chains instead of the
                     fused activations. Same hash, more ops — for A/B timing.

Inspection:
  --inspect          Print the Espresso layer-type histogram and the derived
                     native op list, then exit. Run this first against a model
                     you have not used before: unsupported layer types are
                     named here rather than silently skipped.
  --export-spec <p>  Write the derived architecture as JSON. The spec is the
                     crate's native description of the network and is readable
                     without the vendor container.

Other:
  --compare <a> <b>  Hamming distance between two hex hashes; no model needed.
  -h, --help         Show this help

NeuralHash is a perceptual hash: re-encoded or lightly edited copies of an
image land a few bits apart rather than matching exactly, and the same is true
across rlx backends. Compare with Hamming distance, not equality.";

/// Parse `args` and run the requested NeuralHash command.
pub fn run(args: &[String]) -> Result<()> {
    let mut net_path: Option<PathBuf> = None;
    let mut seed_path: Option<PathBuf> = None;
    let mut images: Vec<PathBuf> = Vec::new();
    let mut device = "cpu".to_string();
    let mut dump: Option<PathBuf> = None;
    let mut export_spec: Option<PathBuf> = None;
    let mut compare: Option<(String, String)> = None;
    let mut show_embedding = false;
    let mut show_scores = false;
    let mut json = false;
    let mut distances = false;
    let mut inspect = false;
    let mut bench: Option<usize> = None;
    let mut fuse = true;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                return Ok(());
            }
            "--net" | "--espresso" | "--model" => {
                net_path = Some(PathBuf::from(req(args, &mut i)?))
            }
            "--seed" => seed_path = Some(PathBuf::from(req(args, &mut i)?)),
            "--image" => images.push(PathBuf::from(req(args, &mut i)?)),
            "--device" => device = req(args, &mut i)?,
            "--dump" => dump = Some(PathBuf::from(req(args, &mut i)?)),
            "--export-spec" => export_spec = Some(PathBuf::from(req(args, &mut i)?)),
            "--bench" => {
                bench = Some(
                    req(args, &mut i)?
                        .parse()
                        .map_err(|e| anyhow!("--bench needs an iteration count: {e}"))?,
                )
            }
            "--embedding" => {
                show_embedding = true;
                i += 1;
            }
            "--scores" => {
                show_scores = true;
                i += 1;
            }
            "--json" => {
                json = true;
                i += 1;
            }
            "--distances" => {
                distances = true;
                i += 1;
            }
            "--inspect" => {
                inspect = true;
                i += 1;
            }
            "--no-fuse" => {
                fuse = false;
                i += 1;
            }
            "--compare" => {
                let a = req(args, &mut i)?;
                let b = args
                    .get(i)
                    .cloned()
                    .ok_or_else(|| anyhow!("--compare needs two hex hashes"))?;
                i += 1;
                compare = Some((a, b));
            }
            other => bail!("unknown argument {other:?}\n\n{HELP}"),
        }
    }

    if let Some((a, b)) = compare {
        let a = NeuralHash::from_hex(&a).context("parsing the first --compare hash")?;
        let b = NeuralHash::from_hex(&b).context("parsing the second --compare hash")?;
        let d = a.hamming(&b);
        if json {
            println!(
                "{}",
                serde_json::json!({ "a": a.to_hex(), "b": b.to_hex(), "hamming": d })
            );
        } else {
            println!("{d}");
        }
        return Ok(());
    }

    let net_path = net_path.ok_or_else(|| {
        anyhow!("--net <NeuralHashv3b-current.espresso.net> is required\n\n{HELP}")
    })?;

    if inspect || export_spec.is_some() {
        let net = EspressoNet::open(&net_path)?;
        if inspect {
            println!("espresso format_version: {}", net.format_version);
            println!(
                "layers: {}  weight blobs: {}",
                net.layers.len(),
                net.blob_count()
            );
            println!("\nlayer types:");
            for (kind, n) in net.op_histogram() {
                println!("  {n:>4}  {kind}");
            }
            println!("\nlayers:\n{}", net.layer_report());
        }
        let spec = NeuralHashSpec::from_espresso(&net)?;
        if inspect {
            println!(
                "derived native spec: {} ops, input {:?} {:?}, output {:?} ({} floats)",
                spec.ops.len(),
                spec.input,
                spec.input_shape,
                spec.output,
                spec.output_dim()
            );
            for (i, op) in spec.ops.iter().enumerate() {
                println!(
                    "  {i:>4}  {:<28} {:?} -> {} {:?}",
                    op.name, op.ins, op.out, op.shape
                );
            }
            match spec.validate_neuralhash_io() {
                Ok(()) => println!("\nI/O matches the NeuralHash pipeline."),
                Err(e) => println!("\nWARNING: {e:#}"),
            }
        }
        if let Some(p) = export_spec {
            std::fs::write(&p, spec.to_json()?)
                .with_context(|| format!("writing spec to {}", p.display()))?;
            eprintln!(
                "[neuralhash] wrote the derived architecture to {}",
                p.display()
            );
        }
        if images.is_empty() {
            return Ok(());
        }
    }

    let seed_path = seed_path.ok_or_else(|| anyhow!("--seed <seed.dat> is required\n\n{HELP}"))?;
    if images.is_empty() {
        bail!("at least one --image is required\n\n{HELP}");
    }

    let device = parse_standard_device("neuralhash", &device)?;
    let seed = SeedMatrix::open(&seed_path)?;
    eprintln!(
        "[neuralhash] building the native graph from {} for {device:?}",
        net_path.display()
    );
    let build_start = std::time::Instant::now();
    let model = NeuralHashModel::open_espresso_fused(&net_path, device, fuse)?;
    let build = build_start.elapsed();
    let mut hasher = NeuralHasher::from_parts(model, seed);

    if let Some(iters) = bench {
        return run_bench(&mut hasher, &images[0], iters, build, device);
    }

    let mut hashes: Vec<NeuralHash> = Vec::with_capacity(images.len());
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(images.len());
    for path in &images {
        let input = crate::preprocess::load_image(path)?;
        let embedding = hasher.model_mut().embed(&input)?;
        let scores = hasher.seed().project(&embedding)?;
        let hash = NeuralHash::from_scores(&scores)?;

        if json {
            let mut obj = serde_json::json!({
                "image": path.display().to_string(),
                "hash": hash.to_hex(),
                "device": format!("{device:?}"),
            });
            if show_embedding {
                obj["embedding"] = serde_json::json!(embedding);
            }
            if show_scores {
                obj["scores"] = serde_json::json!(scores);
            }
            println!("{obj}");
        } else {
            if images.len() == 1 {
                println!("{hash}");
            } else {
                println!("{hash}  {}", path.display());
            }
            if show_embedding {
                eprintln!("[neuralhash] embedding: {embedding:?}");
            }
            if show_scores {
                eprintln!("[neuralhash] scores: {scores:?}");
                let min_abs = scores.iter().fold(f32::INFINITY, |m, s| m.min(s.abs()));
                eprintln!(
                    "[neuralhash] closest score to the decision boundary: {min_abs:e} \
                     (small values = bits that may differ on another backend)"
                );
            }
        }
        hashes.push(hash);
        embeddings.push(embedding);
    }

    if distances && hashes.len() > 1 {
        eprintln!("[neuralhash] pairwise Hamming distances:");
        for (a, ha) in hashes.iter().enumerate() {
            let row: Vec<String> = hashes.iter().map(|hb| ha.hamming(hb).to_string()).collect();
            eprintln!("  {}  {}", row.join(" "), images[a].display());
        }
    }

    if let Some(path) = dump {
        let bytes: Vec<u8> = embeddings
            .iter()
            .flat_map(|e| e.iter().flat_map(|v| v.to_le_bytes()))
            .collect();
        std::fs::write(&path, &bytes)
            .with_context(|| format!("writing descriptors to {}", path.display()))?;
        eprintln!(
            "[neuralhash] wrote {} descriptor(s) to {}",
            embeddings.len(),
            path.display()
        );
    }

    Ok(())
}

/// Time `iters` forward passes of one image on an already-built model.
///
/// Reports the emitted rlx graph size alongside latency: the two move together,
/// and a regression in one usually explains the other.
fn run_bench(
    hasher: &mut NeuralHasher,
    image: &std::path::Path,
    iters: usize,
    build: std::time::Duration,
    device: rlx_runtime::Device,
) -> Result<()> {
    let iters = iters.max(1);
    let input = crate::preprocess::load_image(image)?;

    println!("device        {device:?}");
    println!("spec ops      {}", hasher.model_mut().spec().ops.len());
    println!("graph nodes   {}", hasher.model_mut().graph_nodes());
    println!(
        "build         {:.2} s  (parse + derive + compile + bind)",
        build.as_secs_f64()
    );

    // One untimed pass: first-run costs (lazy kernel compilation, allocator
    // warm-up) belong to `build`, not to steady-state latency.
    let warm = hasher.model_mut().embed(&input)?;
    std::hint::black_box(&warm);

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        // The hash, not just the descriptor: a backend that defers work until a
        // readback would otherwise be timed as infinitely fast.
        let d = hasher.model_mut().embed(&input)?;
        let h = hasher.hash_embedding(&d)?;
        std::hint::black_box(h);
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    println!(
        "forward       {:.2} ms/image   ({:.1} images/s over {iters} iters)",
        dt * 1e3,
        1.0 / dt
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| (*x).to_string()).collect()
    }

    #[test]
    fn compare_needs_no_model() {
        run(&s(&[
            "--compare",
            "000000000000000000000000",
            "800000000000000000000001",
        ]))
        .unwrap();
    }

    #[test]
    fn compare_rejects_a_bad_hash() {
        let e = run(&s(&["--compare", "nope", "800000000000000000000001"])).unwrap_err();
        assert!(format!("{e:#}").contains("first --compare hash"));
    }

    #[test]
    fn help_short_circuits() {
        run(&s(&["--help"])).unwrap();
    }

    #[test]
    fn missing_required_flags_are_reported() {
        let e = run(&s(&["--image", "x.png"])).unwrap_err();
        assert!(format!("{e:#}").contains("--net"), "{e:#}");
        let e = run(&s(&["--net", "m.espresso.net", "--image", "x.png"])).unwrap_err();
        assert!(format!("{e:#}").contains("--seed"), "{e:#}");
        let e = run(&s(&["--net", "m.espresso.net", "--seed", "s.dat"])).unwrap_err();
        assert!(format!("{e:#}").contains("--image"), "{e:#}");
    }

    #[test]
    fn unknown_flags_are_rejected() {
        assert!(run(&s(&["--nope"])).is_err());
    }
}
