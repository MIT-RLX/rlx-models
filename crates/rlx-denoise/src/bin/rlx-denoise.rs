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

//! Train and evaluate the denoiser.
//!
//! ```sh
//! rlx-denoise train --train train.bin --val val.bin --out weights.bin \
//!     --device cuda --epochs 60 --batch 8
//! rlx-denoise eval --val val.bin --weights weights.bin --device metal
//! ```
//!
//! Datasets come from a renderer — `cargo run --example denoise_dataset` in
//! threers writes the format this reads.

use anyhow::{Result, bail};
use rlx_denoise::{
    Batch, Dataset, DenoiseNet, Denoiser, TrainConfig, Trainer, Widths, checkpoint,
    model::{Arch, Head, Sampling},
    train::LOSS_EPSILON,
};
use rlx_runtime::Device;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("train") => train(Args::parse(&args[1..])?),
        Some("eval") => eval(Args::parse(&args[1..])?),
        Some("-h") | Some("--help") | None => {
            println!("{USAGE}");
            Ok(())
        }
        Some(other) => bail!("unknown command {other}\n\n{USAGE}"),
    }
}

const USAGE: &str = "\
rlx-denoise — train and evaluate the Monte-Carlo render denoiser

  rlx-denoise train --train FILE [--val FILE] --out FILE [options]
  rlx-denoise eval  --val FILE --weights FILE [options]

  --device NAME    cpu | cuda | metal | mlx | rocm | gpu | vulkan   [cpu]
  --widths NAME    default (32/48/64/80) | tiny | wide (64/96/128/160)
                     huge (96/160/224/288), about OIDN's RT filter size
  --arch NAME      direct | kpn | oidn | optix                      [direct]
  --core-guides    9 input planes (colour/albedo/normal), not 11
  --mirrored       eval: average four mirrored passes (4x cost, ~2-3% better)
  --augment        train: mirror each tile at random, 4x the effective set
  --radius N       kernel-head support, (2N+1) squared taps         [2]
                     direct  residual output, strided resampling
                     kpn     kernel-predicting output, strided
                     oidn    residual output, pooled resampling
                     optix   kernel-predicting output, pooled
  --epochs N       passes over the training set                     [40]
  --batch N        tiles per step                                   [4]
  --lr F           initial learning rate                            [1e-3]
  --constant-lr    hold it there instead of decaying to a twentieth
  --gradient F     add F x a multi-scale image-gradient term         [0]
  --dump FILE      eval: write the predicted tiles as raw f32 planes
  --seed N         shuffle seed                                     [1]
  --resume FILE    start from these weights instead of fresh ones";

/// Epochs averaged for the unbiased end-of-run figure.
const TAIL_EPOCHS: usize = 50;

/// Parsed flags, with the defaults already applied.
struct Args {
    train: Option<String>,
    val: Option<String>,
    out: Option<String>,
    weights: Option<String>,
    resume: Option<String>,
    device: Device,
    widths: Widths,
    arch: Option<String>,
    radius: Option<usize>,
    core_guides: bool,
    mirrored: bool,
    augment: bool,
    epochs: usize,
    batch: usize,
    lr: f32,
    constant_lr: bool,
    gradient_weight: f32,
    dump: Option<String>,
    seed: u64,
}

impl Args {
    fn parse(args: &[String]) -> Result<Self> {
        let mut parsed = Self {
            train: None,
            val: None,
            out: None,
            weights: None,
            resume: None,
            device: Device::Cpu,
            widths: Widths::default(),
            arch: None,
            radius: None,
            core_guides: false,
            mirrored: false,
            augment: false,
            epochs: 40,
            batch: 4,
            lr: 1e-3,
            constant_lr: false,
            gradient_weight: 0.0,
            dump: None,
            seed: 1,
        };
        let mut i = 0;
        while i < args.len() {
            let flag = args[i].as_str();
            let value = args.get(i + 1).cloned();
            let need = || -> Result<String> {
                value
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))
            };
            match flag {
                "--train" => parsed.train = Some(need()?),
                "--val" => parsed.val = Some(need()?),
                "--out" => parsed.out = Some(need()?),
                "--weights" => parsed.weights = Some(need()?),
                "--resume" => parsed.resume = Some(need()?),
                "--device" => parsed.device = device(&need()?)?,
                "--widths" => parsed.widths = widths(&need()?)?,
                "--arch" => parsed.arch = Some(need()?),
                "--radius" => parsed.radius = Some(need()?.parse()?),
                "--core-guides" => {
                    parsed.core_guides = true;
                    i -= 1;
                }
                "--mirrored" => {
                    parsed.mirrored = true;
                    i -= 1;
                }
                "--augment" => {
                    parsed.augment = true;
                    i -= 1;
                }
                "--epochs" => parsed.epochs = need()?.parse()?,
                "--batch" => parsed.batch = need()?.parse()?,
                "--lr" => parsed.lr = need()?.parse()?,
                "--constant-lr" => {
                    parsed.constant_lr = true;
                    i -= 1;
                }
                "--gradient" => parsed.gradient_weight = need()?.parse()?,
                "--dump" => parsed.dump = Some(need()?),
                "--seed" => parsed.seed = need()?.parse()?,
                other => bail!("unknown flag {other}\n\n{USAGE}"),
            }
            i += 2;
        }
        Ok(parsed)
    }
}

fn device(name: &str) -> Result<Device> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda,
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "rocm" => Device::Rocm,
        "gpu" | "wgpu" => Device::Gpu,
        "vulkan" => Device::Vulkan,
        other => bail!("unknown device {other}"),
    })
}

/// Resolve `--arch` against `--widths`.
///
/// `oidn` and `optix` are this crate's reading of those two denoisers'
/// published and observable shape, not their weights, which are respectively
/// large and proprietary.
fn arch(name: Option<&str>, widths: Widths, radius: Option<usize>) -> Result<Arch> {
    let mut resolved = match name.unwrap_or("direct").to_ascii_lowercase().as_str() {
        "direct" => Arch {
            widths,
            head: Head::Direct,
            sampling: Sampling::Strided,
            ..Arch::default()
        },
        "kpn" | "kernel" => Arch {
            widths,
            head: Head::Kernel { radius: 2 },
            sampling: Sampling::Strided,
            ..Arch::default()
        },
        "oidn" => Arch::oidn_like(widths),
        "optix" => Arch::optix_like(widths),
        other => bail!("unknown architecture {other}"),
    };
    if let Some(radius) = radius {
        match &mut resolved.head {
            Head::Kernel { radius: r } => *r = radius,
            Head::Direct => bail!("--radius needs a kernel head; try --arch kpn or optix"),
        }
    }
    Ok(resolved)
}

fn widths(name: &str) -> Result<Widths> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "default" | "full" => Widths::default(),
        "tiny" | "small" => Widths::tiny(),
        "wide" | "large" => Widths::wide(),
        "huge" => Widths::huge(),
        other => bail!("unknown width preset {other}"),
    })
}

fn train(args: Args) -> Result<()> {
    let Some(train_path) = args.train.clone() else {
        bail!("train needs --train\n\n{USAGE}");
    };
    let Some(out_path) = args.out.clone() else {
        bail!("train needs --out\n\n{USAGE}");
    };

    // Shape is checked once the architecture is resolved, below: `--arch` and
    // `--core-guides` both change what the right shape is, so checking against
    // the default here rejects sets the run could have used.
    let data = Dataset::open(&train_path)?;
    let tile = data.tile();
    println!(
        "train  {} tiles of {tile}x{tile}   unfiltered relative error {:.4}",
        data.len(),
        data.baseline_error(LOSS_EPSILON)
    );
    if data.len() < args.batch {
        bail!("{} tiles cannot fill a batch of {}", data.len(), args.batch);
    }

    let validation = match &args.val {
        Some(path) => {
            let set = Dataset::open(path)?;
            if set.tile() != tile {
                bail!(
                    "validation tiles are {0}x{0}, training tiles are {tile}x{tile}",
                    set.tile()
                );
            }
            println!(
                "val    {} tiles              unfiltered relative error {:.4}",
                set.len(),
                set.baseline_error(LOSS_EPSILON)
            );
            Some(set)
        }
        None => None,
    };

    let mut arch = arch(args.arch.as_deref(), args.widths, args.radius)?;
    if args.core_guides {
        arch.inputs = rlx_denoise::model::CORE_CHANNELS;
    }
    // Now that the shape is known, both sets have to match it.
    data.check_shape(arch.inputs, rlx_denoise::model::OUT_CHANNELS)?;
    if let Some(set) = &validation {
        set.check_shape(arch.inputs, rlx_denoise::model::OUT_CHANNELS)?;
    }
    let net = DenoiseNet::with_arch(arch);
    let parameters: usize = net.params().iter().map(|s| s.elems()).sum();
    println!(
        "net    {:?} {:?} {:?}  {} planes in, {parameters} parameters   device {:?}",
        args.widths, arch.head, arch.sampling, arch.inputs, args.device
    );

    let mut trainer = Trainer::new(
        DenoiseNet::with_arch(arch),
        args.batch,
        tile,
        tile,
        args.device,
        TrainConfig {
            learning_rate: args.lr,
            // The schedule needs to know how far it is going. Steps, not
            // epochs, because that is what the optimiser counts.
            total_steps: if args.constant_lr {
                0
            } else {
                ((data.len() / args.batch) * args.epochs) as u32
            },
            gradient_weight: args.gradient_weight,
            ..Default::default()
        },
    )?;
    if let Some(path) = &args.resume {
        let (from, params) = checkpoint::load(path)?;
        if from.widths() != args.widths {
            bail!(
                "{path} holds a {:?} network, not {:?}",
                from.widths(),
                args.widths
            );
        }
        trainer.set_params(params)?;
        println!("resume {path}");
    }

    // Evaluated on the host so validation never competes with training for
    // device memory; the network is small enough that this costs seconds.
    let mut evaluator = match &validation {
        Some(set) if !set.is_empty() => Some(Denoiser::new(
            DenoiseNet::with_arch(arch),
            trainer.params().to_vec(),
            1,
            tile,
            tile,
            args.device,
        )?),
        _ => None,
    };

    let batches = data.len() / args.batch;
    let mut order: Vec<usize> = (0..data.len()).collect();
    let mut rng = Lcg::new(args.seed);
    let mut best = f32::INFINITY;
    let mut history: Vec<f32> = Vec::with_capacity(args.epochs);

    for epoch in 0..args.epochs {
        rng.shuffle(&mut order);
        let mut sum = 0.0f64;
        let mut steps = 0usize;
        for b in 0..batches {
            let indices = &order[b * args.batch..(b + 1) * args.batch];
            let (mut input, mut target) = data.gather(indices)?;
            if args.augment {
                // A mirrored render is a render: the scene is a valid one seen
                // from a mirrored world, so the pair stays a true example. Four
                // orientations means the network sees each tile four ways and
                // has four times as much to memorise before it can memorise
                // anything — the cheapest regulariser available, since the
                // transform already existed for inference.
                //
                // Per tile rather than per batch, so a batch is not four copies
                // of one orientation.
                augment_batch(
                    &mut input,
                    &mut target,
                    args.batch,
                    data.inputs(),
                    rlx_denoise::model::OUT_CHANNELS,
                    tile,
                    &mut rng,
                );
            }
            let loss = trainer.step(&Batch {
                n: args.batch,
                h: tile,
                w: tile,
                input: &input,
                target: &target,
            })?;
            if !loss.is_finite() {
                bail!(
                    "loss became {loss} at epoch {epoch}, step {}",
                    trainer.steps_taken()
                );
            }
            sum += loss as f64;
            steps += 1;
        }
        let train_loss = if steps == 0 {
            f64::NAN
        } else {
            sum / steps as f64
        };

        let mut line = format!("epoch {:3}  train {:.5}", epoch + 1, train_loss);
        let mut score = train_loss as f32;
        if let (Some(set), Some(evaluator)) = (&validation, evaluator.as_mut()) {
            let error = validate(evaluator, set, trainer.params().to_vec())?;
            line.push_str(&format!("   val {error:.5}"));
            score = error;
        }
        // Keep the weights that generalise, not the last ones: a small dataset
        // starts overfitting well before the epoch budget runs out, and the
        // final epoch is usually not the best one.
        history.push(score);
        if score < best {
            best = score;
            checkpoint::save(&net, trainer.params(), &out_path)?;
            line.push_str("   saved");
        }
        println!("{line}");
    }

    // `best` is the minimum of every epoch's error on the set it is reported
    // against, so it is a selection on the evaluation data and reads better
    // than the network is — by 2 to 6% on the runs behind this crate's README.
    // The tail mean has no such bias, and its spread says whether a difference
    // between two runs is a difference at all.
    let tail = TAIL_EPOCHS.min(history.len());
    if tail > 0 {
        let window = &history[history.len() - tail..];
        let mean = window.iter().map(|v| *v as f64).sum::<f64>() / tail as f64;
        let variance = window
            .iter()
            .map(|v| (*v as f64 - mean).powi(2))
            .sum::<f64>()
            / tail as f64;
        println!(
            "last {tail} epochs: mean {mean:.5}  sd {:.5}   <- report this",
            variance.sqrt()
        );
    }
    println!("best {best:.5}  ->  {out_path}   <- selected on the evaluation set");
    Ok(())
}

/// Relative error over a whole set, one tile at a time.
///
/// The squared errors accumulate across tiles and the root is taken once, so
/// this is the same quantity as [`Dataset::baseline_error`] and the two can be
/// compared directly. Averaging per-tile roots instead would report a smaller
/// number for identical predictions.
fn validate(evaluator: &mut Denoiser, set: &Dataset, params: Vec<Vec<f32>>) -> Result<f32> {
    validate_with(evaluator, set, params, false)
}

/// Mirror each tile of a batch independently, input and target together.
///
/// The guides ride along with the colour: mirroring the whole tile keeps every
/// relation between a pixel and its neighbours, which is what the network reads
/// them for.
#[allow(clippy::too_many_arguments)]
fn augment_batch(
    input: &mut [f32],
    target: &mut [f32],
    batch: usize,
    inputs: usize,
    outputs: usize,
    tile: usize,
    rng: &mut Lcg,
) {
    let in_stride = inputs * tile * tile;
    let out_stride = outputs * tile * tile;
    for t in 0..batch {
        let choice = rng.next() % 4;
        let (fx, fy) = (choice & 1 == 1, choice & 2 == 2);
        if !fx && !fy {
            continue;
        }
        let i0 = t * in_stride;
        let o0 = t * out_stride;
        let mirrored_in = mirror(&input[i0..i0 + in_stride], inputs, tile, fx, fy);
        input[i0..i0 + in_stride].copy_from_slice(&mirrored_in);
        let mirrored_out = mirror(&target[o0..o0 + out_stride], outputs, tile, fx, fy);
        target[o0..o0 + out_stride].copy_from_slice(&mirrored_out);
    }
}

/// Reverse a planar `[planes, tile, tile]` buffer along one or both axes.
fn mirror(buffer: &[f32], planes: usize, tile: usize, flip_x: bool, flip_y: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; buffer.len()];
    for p in 0..planes {
        let base = p * tile * tile;
        for y in 0..tile {
            let sy = if flip_y { tile - 1 - y } else { y };
            for x in 0..tile {
                let sx = if flip_x { tile - 1 - x } else { x };
                out[base + y * tile + x] = buffer[base + sy * tile + sx];
            }
        }
    }
    out
}

fn validate_with(
    evaluator: &mut Denoiser,
    set: &Dataset,
    params: Vec<Vec<f32>>,
    mirrored: bool,
) -> Result<f32> {
    validate_dumping(evaluator, set, params, mirrored, None)
}

/// As [`validate_with`], and if `dump` is set, write every predicted tile to
/// that path as bare `f32` planes in the dataset's own order.
///
/// A scalar says how far off the filter is but not *how* — whether what is left
/// is grain, or a smeared edge, or invented low-frequency shading. Those are
/// different failures with different fixes and the number does not separate
/// them. Raw planes rather than an image format because the caller already has
/// to undo the compression to look at anything.
fn validate_dumping(
    evaluator: &mut Denoiser,
    set: &Dataset,
    params: Vec<Vec<f32>>,
    mirrored: bool,
    dump: Option<&str>,
) -> Result<f32> {
    evaluator.set_params(params)?;
    let tile = set.tile();
    let mut dumped: Vec<f32> = Vec::new();
    let transforms: &[(bool, bool)] = if mirrored {
        &[(false, false), (true, false), (false, true), (true, true)]
    } else {
        &[(false, false)]
    };
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for i in 0..set.len() {
        let (input, target) = set.gather(&[i])?;
        // Average in the network's own range, before anything inverts it: the
        // compression is convex, so averaging after would bias every pixel up.
        let mut acc = vec![0.0f32; target.len()];
        for &(fx, fy) in transforms {
            let fed = if fx || fy {
                mirror(&input, set.inputs(), tile, fx, fy)
            } else {
                input.clone()
            };
            let out = evaluator.run(&fed)?;
            let out = if fx || fy {
                mirror(&out, set.outputs(), tile, fx, fy)
            } else {
                out
            };
            for (a, v) in acc.iter_mut().zip(&out) {
                *a += v;
            }
        }
        let scale = transforms.len() as f32;
        let predicted: Vec<f32> = acc.iter().map(|v| v / scale).collect();
        if dump.is_some() {
            dumped.extend_from_slice(&predicted);
        }
        let (s, n) = Denoiser::relative_error_sum(&predicted, &target, LOSS_EPSILON);
        sum += s;
        count += n;
    }
    if let Some(path) = dump {
        let bytes: Vec<u8> = dumped.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;
        println!(
            "wrote {path}: {} tiles of {tile}, {} planes",
            set.len(),
            set.outputs()
        );
    }
    if count == 0 {
        return Ok(f32::NAN);
    }
    Ok((sum / count as f64).sqrt() as f32)
}

fn eval(args: Args) -> Result<()> {
    let Some(val_path) = args.val.clone() else {
        bail!("eval needs --val\n\n{USAGE}");
    };
    let Some(weights_path) = args.weights.clone() else {
        bail!("eval needs --weights\n\n{USAGE}");
    };

    let set = Dataset::open(&val_path)?;
    let tile = set.tile();
    let (net, params) = checkpoint::load(&weights_path)?;
    let loaded = net.arch();
    // Against the weights' own shape, not a compile-time constant: a network
    // trained on nine planes is a valid network, and checking before the file
    // is read rejects it for not being the default.
    set.check_shape(loaded.inputs, rlx_denoise::model::OUT_CHANNELS)?;
    let mut evaluator = Denoiser::new(net, params.clone(), 1, tile, tile, args.device)?;

    let baseline = set.baseline_error(LOSS_EPSILON);
    let denoised = validate_dumping(
        &mut evaluator,
        &set,
        params,
        args.mirrored,
        args.dump.as_deref(),
    )?;
    println!(
        "{} tiles of {tile}x{tile}, {} planes, {:?} {:?} on {:?}{}",
        set.len(),
        loaded.inputs,
        loaded.head,
        loaded.sampling,
        args.device,
        if args.mirrored { ", mirrored" } else { "" }
    );
    println!("unfiltered  {baseline:.5}");
    println!(
        "denoised    {denoised:.5}   {:.2}x",
        baseline / denoised.max(1e-9)
    );
    Ok(())
}

/// A shuffle needs randomness but not a dependency, and a fixed seed makes a
/// run reproducible.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }

    fn shuffle(&mut self, items: &mut [usize]) {
        for i in (1..items.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}
