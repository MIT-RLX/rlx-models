// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Train the student on distillation records with RLX: AdamW, warmup +
//! cosine, global-norm clipping. The last 2% of records are held out; their
//! loss is reported every `--eval-every` steps. The student is saved as a
//! converted Kroko directory (`--out`), ready for `rlx-kroko`'s
//! `mcu_check --dir`.
//!
//! **Devices are picked automatically.** By default every local backend this
//! binary was built with and that can train ([`rlx_tensor::training_devices`]:
//! one backend per chip, the CPU included, NPUs only when named) takes a share
//! of each batch; each device is timed alone and against the speed split, and
//! the fastest layout wins (a device too slow to help is dropped).
//! `--devices rocm,cpu,xdna` restricts the candidates; `--devices rocm:24,cpu:8`
//! fixes the shares.
//!
//! **Machines are found automatically.** `--cluster` makes this process the
//! coordinator: it announces itself on the LAN and trains with every
//! `--worker` that joins within `RENDEZVOUS_WINDOW` seconds (default 10), or
//! alone if none does. A worker needs only this binary — no data, no teacher:
//! the coordinator sends the job, the initial weights and each step's samples.
//! Every machine picks its own devices as above; the global `--batch` is split
//! across machines by their measured speed, gradients are summed sample-
//! weighted, and every machine applies the same update. Between syncs each
//! machine runs as many micro-batches as its latest measured speed allows in a
//! round about ten times the sync's cost, re-planned every sync, so a machine
//! that slows down takes less work (`--accumulate N` fixes N everywhere).
//! Samples travel as f16.
//!
//! ```text
//! zipformer-distill-train --teacher .cache/zipformer-m/native \
//!     --data DIR --out .cache/zipformer-s [--steps 3000] [--batch 32] [--lr 1e-3]
//!     [--devices LIST] [--calibrate 3] [--no-init] [--cluster [--rendezvous URL]]
//!     [--accumulate N] [--mse-only] [--mse-weight 0.1] [--temperature 1] [--resume]
//!     [--token-weight 1] [--keep-checkpoints]
//!     [--optimizer adamw|muon] [--init RESUME_FILE] [--qat 4|8] [--max-mem-gb G]
//! zipformer-distill-train --worker [--devices LIST] [--rendezvous URL]
//! ```
//! `--optimizer muon` trains the 2-D weight matrices with Muon and the rest
//! with AdamW, at the same learning rate (`rlx_optim::MuonAdamW`).
//!
//! `--init PATH` starts from the weights in another run's `resume.bin` with
//! a fresh optimizer and schedule (a fine-tune); `--resume` takes precedence.
//!
//! `--qat 4` trains against the int4 MCU export: each layer matrix is
//! rounded in the forward pass exactly as `rlx_kroko::mcu` stores it, and
//! gradients pass straight through to the float weights (best as a short
//! fine-tune from a float run, with `--init`).
//!
//! Progress lines report the process's private and shared memory where the OS
//! exposes them (Linux). `--max-mem-gb G` exits with status 75 after a save
//! once the two add up to more than `G`, so a run loop that restarts with
//! `--resume` caps the process's memory instead of the machine running out.
//!
//! `--data A,B` trains on several record directories as one set; the
//! held-out records come from the end of the last one.
//!
//! `--keep-checkpoints` also saves every checkpoint to `--out/step-N/`, to
//! compare runs at the same step.
//!
//! Every save also writes `resume.bin` (steps done, parameters, AdamW
//! state) to `--out`; `--resume` continues from it exactly where the run
//! stopped, with the same arguments.
//!
//! With `contexts.u16` in the data directory (`zipformer-distill-prep
//! --contexts`) the loss is the KL divergence from the teacher's per-frame
//! token distribution to the student's, through the teacher's predictor and
//! joiner, plus `--mse-weight` times the squared error below. Without it, or
//! with `--mse-only`, the loss is the mean squared error to the teacher's
//! encoder output over the teacher's mean square (1.0 = outputting zeros).

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use rlx_driver::{ProcessGroup, ReduceKind};
use rlx_optim::{MuonAdamW, Optimizer, OptimizerState};
use rlx_tensor::parallel::Feeds;
use rlx_tensor::{
    DataParallel, Device, Func, LrSchedule, Replica, apply_gradients, available_devices,
    device_label, split_batch, training_devices, tune,
};
use rlx_zipformer_distill::config::StudentConfig;
use rlx_zipformer_distill::data::{Records, feat_len, target_len};
use rlx_zipformer_distill::init::Teacher;
use rlx_zipformer_distill::joiner::{DIM, Joiner};
use rlx_zipformer_distill::model::Kd;
use rlx_zipformer_distill::{export, model};
use serde_json::json;

/// Coordinator → worker: the job description, then each step's samples.
const TAG_JOB: u32 = 7100;
const TAG_SAMPLES: u32 = 7101;
/// The teacher's joiner output weight and bias, for the token loss.
const TAG_JOINER: u32 = 7102;
/// Exit status after a save when `--max-mem-gb` is exceeded (`EX_TEMPFAIL`):
/// the caller restarts with `--resume`.
const EXIT_MEMORY: i32 = 75;

/// This process's private (anonymous) and shared resident memory in GB, where
/// the OS reports them (Linux `/proc/self/status`).
fn mem_gb() -> Option<(f64, f64)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb = |key: &str| -> Option<f64> {
        let line = status.lines().find(|l| l.starts_with(key))?;
        line.split_whitespace().nth(1)?.parse::<f64>().ok()
    };
    Some((kb("RssAnon:")? / 1048576.0, kb("RssShmem:")? / 1048576.0))
}

/// The optimizer state of a resumed run.
const TAG_OPT: u32 = 7103;
/// Resume point in `--out`.
const RESUME_FILE: &str = "resume.bin";
/// Steps between re-broadcasts of the coordinator's weights, which bound any
/// floating-point divergence between machines applying the same update.
const RESYNC_EVERY: usize = 100;
/// Most micro-batches a machine runs per gradient sync.
const MAX_MICRO: usize = 16;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Samples travel as f16: half the bytes, and an f16 fbank / teacher output
/// is far inside the loss's own noise.
fn to_f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::f16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

fn from_f16_bytes(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|&c| half::f16::from_bits(u16::from_le_bytes(c)).to_f32())
        .collect()
}

/// Steps done, named tensors and optimizer state, for `--resume` (and the
/// optimizer state alone over the wire).
fn encode_state(step: usize, params: &[(String, Vec<f32>)], opt: &OptimizerState) -> Vec<u8> {
    let mut out = b"RZDS".to_vec();
    out.extend((step as u64).to_le_bytes());
    out.extend(opt.step.to_le_bytes());
    for list in [params, &opt.buffers[..]] {
        out.extend((list.len() as u32).to_le_bytes());
        for (name, v) in list {
            out.extend((name.len() as u32).to_le_bytes());
            out.extend(name.as_bytes());
            out.extend((v.len() as u64).to_le_bytes());
            out.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        }
    }
    out
}

type NamedTensors = Vec<(String, Vec<f32>)>;

/// Inverse of [`encode_state`].
fn decode_state(raw: &[u8]) -> Result<(usize, NamedTensors, OptimizerState)> {
    ensure!(raw.starts_with(b"RZDS"), "not a training state");
    let mut at = 4;
    let mut take = |n: usize| -> Result<&[u8]> {
        ensure!(at + n <= raw.len(), "training state truncated");
        at += n;
        Ok(&raw[at - n..at])
    };
    let u64_at = |b: &[u8]| u64::from_le_bytes(b.try_into().expect("8 bytes"));
    let u32_at = |b: &[u8]| u32::from_le_bytes(b.try_into().expect("4 bytes"));
    let step = u64_at(take(8)?) as usize;
    let opt_step = u64_at(take(8)?);
    let mut lists = Vec::new();
    for _ in 0..2 {
        let n = u32_at(take(4)?) as usize;
        let mut list = Vec::with_capacity(n);
        for _ in 0..n {
            let len = u32_at(take(4)?) as usize;
            let name = String::from_utf8(take(len)?.to_vec())?;
            let count = u64_at(take(8)?) as usize;
            let v = take(count * 4)?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect();
            list.push((name, v));
        }
        lists.push(list);
    }
    let buffers = lists.pop().expect("two lists");
    let params = lists.pop().expect("two lists");
    Ok((
        step,
        params,
        OptimizerState {
            step: opt_step,
            buffers,
        },
    ))
}

fn flatten(m: &Func) -> Vec<f32> {
    m.param_names()
        .iter()
        .flat_map(|n| m.param_binding(n).unwrap_or_default().to_vec())
        .collect()
}

/// Inverse of [`flatten`], by name (same [`Func::param_names`] order).
fn unflatten(m: Func, flat: &[f32]) -> Func {
    let mut offsets = std::collections::HashMap::new();
    let mut at = 0;
    for n in m.param_names() {
        let len = m.param_binding(&n).map_or(0, <[f32]>::len);
        offsets.insert(n, (at, len));
        at += len;
    }
    m.init_params(|name, _| {
        let (o, n) = offsets[name];
        flat[o..o + n].to_vec()
    })
}

/// This machine's candidate devices and fixed shares (`None` = measure).
fn local_spec(arg: Option<&String>, probe: &Func) -> Result<Vec<(Device, Option<usize>)>> {
    Ok(match arg {
        Some(list) if list != "auto" => list
            .split(',')
            .map(|item| {
                let (name, share) = item.split_once(':').unwrap_or((item, ""));
                let dev = rlx_tensor::parse_device(name).map_err(|e| anyhow::anyhow!("{e}"))?;
                let share = (!share.is_empty()).then(|| share.parse()).transpose()?;
                Ok((dev, share))
            })
            .collect::<Result<_>>()?,
        _ => training_devices(&available_devices(), probe.graph())
            .into_iter()
            .map(|d| (d, None))
            .collect(),
    })
}

/// The local replicas for `batch` samples per step: fixed shares as given,
/// else the fastest measured layout. Timing uses zero inputs (time does not
/// depend on values), so a worker can tune before it has any data.
fn local_layout(
    spec: &[(Device, Option<usize>)],
    batch: usize,
    calibrate: usize,
    build: &dyn Fn(usize) -> Func,
    m: &Func,
    zeros: &dyn Fn(&[usize]) -> Vec<Feeds>,
) -> (DataParallel, Vec<Device>, Vec<usize>, f64) {
    let time = |dp: &DataParallel, shares: &[usize]| {
        dp.value_and_grad(m, zeros(shares)); // compiles
        let t = std::time::Instant::now();
        for _ in 0..calibrate.max(1) {
            dp.value_and_grad(m, zeros(shares));
        }
        t.elapsed().as_secs_f64() / calibrate.max(1) as f64
    };
    if spec.iter().all(|s| s.1.is_some()) {
        let (devs, shares): (Vec<Device>, Vec<usize>) =
            spec.iter().map(|s| (s.0, s.1.unwrap_or(0))).unzip();
        let dp = DataParallel::new(
            devs.iter()
                .zip(&shares)
                .map(|(&d, &b)| Replica::new(d, build(b), b))
                .collect(),
        );
        let t = time(&dp, &shares);
        return (dp, devs, shares, t);
    }
    let devs: Vec<Device> = spec.iter().map(|s| s.0).collect();
    let tuned = tune(&devs, batch, |_, b| build(b), time);
    for (layout, t) in &tuned.tried {
        let l: Vec<String> = layout
            .iter()
            .map(|(d, b)| format!("{}:{b}", device_label(*d)))
            .collect();
        eprintln!("calibrate: {} {:.0} ms/step", l.join(","), t * 1e3);
    }
    let best = tuned
        .tried
        .iter()
        .find(|(l, _)| *l == tuned.layout)
        .map_or(0.0, |(_, t)| *t);
    let (devs, shares) = tuned.layout.iter().copied().unzip();
    (tuned.dp, devs, shares, best)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
    };
    let flag = |k: &str| args.iter().any(|a| a == k);
    let num = |k: &str, d: f64| -> Result<f64> {
        Ok(get(k).map(|v| v.parse()).transpose()?.unwrap_or(d))
    };
    let calibrate = num("--calibrate", 3.0)? as usize;
    let worker = flag("--worker");
    let rendezvous = get("--rendezvous").map_or("auto", String::as_str);

    // The machines: none (local), or whoever joins the rendezvous.
    let group: Option<Arc<ProcessGroup>> = if worker {
        eprintln!("worker: looking for a coordinator ({rendezvous})");
        Some(rlx_driver::Node::auto().rendezvous(rendezvous).connect()?)
    } else if flag("--cluster") {
        eprintln!("coordinator: waiting for workers ({rendezvous})");
        Some(
            rlx_driver::Node::new(0, 0)
                .rendezvous(rendezvous)
                .connect()?,
        )
    } else {
        None
    };
    let (rank, world) = group
        .as_ref()
        .map_or((0, 1), |g| (g.rank() as usize, g.world_size() as usize));
    let send = |to: usize, tag: u32, b: &[u8]| -> Result<()> {
        let g = group.as_ref().context("no group")?;
        g.transport()
            .send_bytes(to as u32, tag, b)
            .map_err(|e| anyhow::anyhow!("send to {to}: {e:?}"))
    };
    let recv = |from: usize, tag: u32| -> Result<Vec<u8>> {
        let g = group.as_ref().context("no group")?;
        g.transport()
            .recv_bytes(from as u32, tag)
            .map_err(|e| anyhow::anyhow!("recv from {from}: {e:?}"))
    };

    // The job: from the command line on the coordinator, over the wire on a worker.
    let mut source: Option<(Records, Teacher, PathBuf, PathBuf)> = None;
    let mut joiner: Option<Joiner> = None;
    let keep_checkpoints = flag("--keep-checkpoints");
    let max_mem_gb: Option<f64> = get("--max-mem-gb").map(|v| v.parse()).transpose()?;
    let mut resumed: Option<(NamedTensors, OptimizerState)> = None;
    let mut initial: Option<NamedTensors> = None;
    let job = if rank == 0 {
        let teacher_dir = PathBuf::from(get("--teacher").context("--teacher")?);
        let dirs: Vec<PathBuf> = get("--data")
            .context("--data")?
            .split(',')
            .map(PathBuf::from)
            .collect();
        let data = Records::open_all(&dirs)?;
        let out = PathBuf::from(get("--out").context("--out")?);
        let job = json!({
            "k": data.k,
            "steps": num("--steps", 3000.0)? as usize,
            "batch": num("--batch", 32.0)? as usize,
            "lr": num("--lr", 1e-3)?,
            "eval_every": num("--eval-every", 200.0)? as usize,
            "save_every": num("--save-every", 1000.0)? as usize,
            "kd": data.has_contexts() && !flag("--mse-only"),
            "mse_weight": num("--mse-weight", 0.1)?,
            "temperature": num("--temperature", 1.0)?,
            "token_weight": num("--token-weight", 1.0)?,
            "optimizer": get("--optimizer").map_or("adamw", String::as_str),
            "qat_bits": get("--qat").map(|b| b.parse::<u8>()).transpose()?,
        });
        let mut job = job;
        if flag("--resume") {
            let path = out.join(RESUME_FILE);
            let (step, params, opt) = decode_state(
                &std::fs::read(&path).with_context(|| format!("--resume: {}", path.display()))?,
            )?;
            eprintln!("resuming after step {step} from {}", path.display());
            job["start"] = json!(step);
            resumed = Some((params, opt));
        } else if let Some(path) = get("--init") {
            let (step, params, _) =
                decode_state(&std::fs::read(path).with_context(|| format!("--init: {path}"))?)?;
            eprintln!("starting from the weights of {path} (its step {step}), fresh optimizer");
            initial = Some(params);
        }
        let teacher = Teacher::open(&teacher_dir)?;
        let mut messages = vec![(TAG_JOB, job.to_string().into_bytes())];
        if job["kd"] == true {
            let j = Joiner::open(&teacher, &teacher_dir)?;
            let wb: Vec<u8> = j
                .out_w
                .iter()
                .chain(&j.out_b)
                .flat_map(|v| v.to_le_bytes())
                .collect();
            messages.push((TAG_JOINER, wb));
            job["blank"] = json!(j.blank());
            messages[0].1 = job.to_string().into_bytes();
            joiner = Some(j);
        }
        if let Some((_, opt)) = &resumed {
            messages.push((TAG_OPT, encode_state(0, &[], opt)));
        }
        for r in 1..world {
            for (tag, bytes) in &messages {
                send(r, *tag, bytes)?;
            }
        }
        source = Some((data, teacher, teacher_dir, out));
        job
    } else {
        serde_json::from_slice(&recv(0, TAG_JOB)?)?
    };
    // The token loss: the teacher's joiner on every machine.
    let kd: Option<Arc<Kd>> = if job["kd"] == true {
        let (out_w, out_b) = match &joiner {
            Some(j) => (j.out_w.clone(), j.out_b.clone()),
            None => {
                let raw = recv(0, TAG_JOINER)?;
                let v: Vec<f32> = raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&b| f32::from_le_bytes(b))
                    .collect();
                let vocab = v.len() / (DIM + 1);
                (v[..vocab * DIM].to_vec(), v[vocab * DIM..].to_vec())
            }
        };
        Some(Arc::new(Kd {
            out_w,
            out_b,
            mse_weight: job["mse_weight"].as_f64().context("job mse_weight")? as f32,
            temperature: job["temperature"].as_f64().context("job temperature")? as f32,
            token_weight: job["token_weight"].as_f64().unwrap_or(1.0) as f32,
            blank: job["blank"].as_u64().unwrap_or(0) as usize,
        }))
    } else {
        None
    };
    let start = job["start"].as_u64().unwrap_or(0) as usize;
    if rank != 0 && start > 0 {
        let (_, _, opt) = decode_state(&recv(0, TAG_OPT)?)?;
        resumed = Some((Vec::new(), opt));
    }
    let field = |k: &str| job[k].as_f64().with_context(|| format!("job {k}"));
    let (k, steps, batch) = (
        field("k")? as usize,
        field("steps")? as usize,
        field("batch")? as usize,
    );
    let (lr, eval_every, save_every) = (
        field("lr")? as f32,
        field("eval_every")? as usize,
        field("save_every")? as usize,
    );
    ensure!(batch >= world, "--batch {batch} below the {world} machines");

    let cfg = StudentConfig {
        qat_bits: job["qat_bits"].as_u64().map(|b| b as u8),
        ..StudentConfig::small10m()
    };
    let mut m = model::build(&cfg, 1, k);
    if let Some((_, teacher, _, _)) = &source {
        let from_teacher = !flag("--no-init");
        let mut seed = Rng(0x5eed_1234_abcd);
        m = m.init_params(|name, dims| {
            if from_teacher {
                teacher
                    .init(&cfg, name, dims)
                    .unwrap_or_else(|e| panic!("init {name}: {e}"))
            } else {
                // Small random init (Kaiming-like), for comparison runs.
                let n: usize = dims.iter().product();
                let fan_in = dims
                    .get(1..)
                    .map_or(1, |d| d.iter().product::<usize>())
                    .max(1);
                let std = (1.0 / fan_in as f64).sqrt();
                (0..n)
                    .map(|_| {
                        let u = (seed.next() >> 11) as f64 / (1u64 << 53) as f64;
                        ((u * 2.0 - 1.0) * std * 3f64.sqrt()) as f32
                    })
                    .collect()
            }
        });
    } else {
        m = m.init_params(|_, dims| vec![0.0; dims.iter().product()]);
    }
    if let Some(params) = resumed.as_ref().map(|(p, _)| p).or(initial.as_ref())
        && rank == 0
    {
        let by_name: std::collections::HashMap<&str, &Vec<f32>> =
            params.iter().map(|(n, v)| (n.as_str(), v)).collect();
        m = m.init_params(|name, dims| {
            let v = by_name
                .get(name)
                .unwrap_or_else(|| panic!("{RESUME_FILE} has no {name}"));
            assert_eq!(v.len(), dims.iter().product::<usize>(), "{name} shape");
            v.to_vec()
        });
    }
    if let Some(g) = &group {
        let mut flat = flatten(&m);
        g.broadcast(0, &mut flat)
            .map_err(|e| anyhow::anyhow!("broadcast weights: {e:?}"))?;
        m = unflatten(m, &flat);
    }
    let n_params: usize = flatten(&m).len();

    // This machine's devices and its share of the global batch.
    let (fl, tl) = (
        feat_len(k),
        target_len(k) * if kd.is_some() { 2 } else { 1 },
    );
    let build = |b: usize| -> Func {
        match &kd {
            Some(kd) => model::build_kd(&cfg, b, k, kd.clone()),
            None => model::build(&cfg, b, k),
        }
        .init_params(|name, _| {
            m.param_binding(name)
                .unwrap_or_else(|| panic!("no parameter {name}"))
                .to_vec()
        })
    };
    let zeros = |shares: &[usize]| -> Vec<Feeds> {
        shares
            .iter()
            .map(|&b| {
                vec![
                    ("feats".to_string(), vec![0.0; b * fl]),
                    ("target".to_string(), vec![0.0; b * tl]),
                ]
            })
            .collect()
    };
    let spec = local_spec(get("--devices"), &m)?;
    let fixed_local = spec.iter().all(|s| s.1.is_some());
    let probe = if world == 1 {
        batch
    } else {
        (batch / world).max(1)
    };
    let probe = if fixed_local {
        spec.iter().filter_map(|s| s.1).sum()
    } else {
        probe
    };
    let (mut dp, mut devs, mut shares, mut t) =
        local_layout(&spec, probe, calibrate, &build, &m, &zeros);
    // Machines: all-gather samples/s, then split the global batch by speed.
    let machine_batch: Vec<usize> = if let Some(g) = &group {
        let speeds = g
            .all_gather(&[(probe as f64 / t.max(1e-9)) as f32])
            .map_err(|e| anyhow::anyhow!("all_gather speeds: {e:?}"))?;
        let mut split = split_batch(
            batch,
            &speeds
                .iter()
                .map(|s| 1.0 / f64::from(*s))
                .collect::<Vec<_>>(),
        );
        // The coordinator evaluates on its own devices, so it keeps a sample.
        if split[0] == 0 {
            let (big, _) = split
                .iter()
                .enumerate()
                .max_by_key(|(_, b)| **b)
                .unwrap_or((0, &0));
            split[big] -= 1;
            split[0] = 1;
        }
        if rank == 0 {
            let rows: Vec<String> = split
                .iter()
                .zip(&speeds)
                .enumerate()
                .map(|(r, (b, s))| format!("machine {r}: {s:.1} samples/s → {b}"))
                .collect();
            eprintln!("{} machines\n  {}", world, rows.join("\n  "));
        }
        split
    } else {
        vec![batch]
    };
    let mine = machine_batch[rank];
    if mine > 0 && mine != shares.iter().sum::<usize>() && !fixed_local {
        drop(dp);
        (dp, devs, shares, t) = local_layout(&spec, mine, calibrate, &build, &m, &zeros);
    }
    let names: Vec<&str> = devs.iter().map(|d| device_label(*d)).collect();
    let layout: Vec<String> = names
        .iter()
        .zip(&shares)
        .map(|(n, b)| format!("{n}:{b}"))
        .collect();
    eprintln!(
        "machine {rank}: {} ({} samples per step), student {:.2} M parameters",
        layout.join(","),
        mine,
        n_params as f64 / 1e6
    );

    // Micro-batches per sync, per machine. Each machine keeps the batch it
    // compiled for; how many micro-batches it runs before a sync follows its
    // measured speed, re-planned every sync from every machine's latest times
    // (all compute the same plan from the same numbers). A machine that slows
    // down (other jobs, thermal limits) then takes less work instead of
    // holding everyone back, with no recompile. A sync moves every parameter
    // across the network and does not overlap compute, so a round targets
    // about ten times the sync's cost. `--accumulate N` fixes N everywhere.
    let fixed_k: Option<usize> = get("--accumulate")
        .map(|n| n.parse::<usize>())
        .transpose()?
        .map(|n| n.max(1));
    let round_secs = match &group {
        None => 0.0,
        Some(g) => {
            let mut buf = vec![0f32; n_params + 1];
            g.all_reduce(&mut buf, ReduceKind::Sum)
                .map_err(|e| anyhow::anyhow!("all_reduce: {e:?}"))?;
            let t0 = std::time::Instant::now();
            for _ in 0..2 {
                g.all_reduce(&mut buf, ReduceKind::Sum)
                    .map_err(|e| anyhow::anyhow!("all_reduce: {e:?}"))?;
            }
            let comm = t0.elapsed().as_secs_f64() / 2.0;
            let mut target = [(10.0 * comm).max(t) as f32];
            g.all_reduce(&mut target, ReduceKind::Max)
                .map_err(|e| anyhow::anyhow!("all_reduce: {e:?}"))?;
            if rank == 0 {
                eprintln!(
                    "gradient sync {:.0} ms → rounds of about {:.1} s between syncs",
                    comm * 1e3,
                    target[0]
                );
            }
            f64::from(target[0])
        }
    };
    // Seconds per micro-batch on each machine (0 = takes no samples).
    let mut micro_secs: Vec<f64> = match &group {
        None => vec![t],
        Some(g) => g
            .all_gather(&[if mine > 0 { t as f32 } else { 0.0 }])
            .map_err(|e| anyhow::anyhow!("all_gather: {e:?}"))?
            .iter()
            .map(|&x| f64::from(x))
            .collect(),
    };
    let plan = |micro_secs: &[f64]| -> Vec<usize> {
        if let Some(k) = fixed_k {
            return machine_batch
                .iter()
                .map(|&b| if b > 0 { k } else { 0 })
                .collect();
        }
        if world == 1 {
            return vec![1];
        }
        let mut k: Vec<usize> = micro_secs
            .iter()
            .zip(&machine_batch)
            .map(|(&s, &b)| {
                if b == 0 || s <= 0.0 {
                    0
                } else {
                    ((round_secs / s).round() as usize).min(MAX_MICRO)
                }
            })
            .collect();
        if k.iter().all(|&x| x == 0) {
            let best = (0..world)
                .filter(|&r| machine_batch[r] > 0 && micro_secs[r] > 0.0)
                .min_by(|&a, &b| micro_secs[a].total_cmp(&micro_secs[b]))
                .unwrap_or(0);
            k[best] = 1;
        }
        k
    };

    // Local replicas' feeds from `b` samples of feats + targets, back to back.
    let split_local = |feats: &[f32], target: &[f32]| -> Vec<Feeds> {
        let mut at = 0;
        shares
            .iter()
            .map(|&b| {
                let f = feats[at * fl..(at + b) * fl].to_vec();
                let t = target[at * tl..(at + b) * tl].to_vec();
                at += b;
                vec![("feats".to_string(), f), ("target".to_string(), t)]
            })
            .collect()
    };

    let mut opt: Box<dyn Optimizer> = match job["optimizer"].as_str() {
        None | Some("adamw") => Box::new(rlx_optim::AdamW::new(lr).with_weight_decay(1e-4)),
        Some("muon") => Box::new(MuonAdamW::new(lr).with_weight_decay(1e-4)),
        Some(o) => anyhow::bail!("--optimizer {o}: adamw or muon"),
    };
    if let Some((_, state)) = &resumed {
        ensure!(opt.load_state_dict(state), "optimizer state does not load");
    }
    let sched = LrSchedule::WarmupCosine {
        base: lr,
        min: lr * 0.05,
        warmup: (steps / 20).max(10),
        total: steps,
    };
    let (n_train, n_val) = source.as_ref().map_or((0, 0), |(d, _, _, _)| {
        let n_val = (d.n / 50).max(batch);
        (d.n - n_val, n_val)
    });
    // Features and targets of `recs`; with the token loss each target frame
    // is the teacher's encoder output then its greedy context's predictor output.
    let read = |recs: &[usize]| -> (Vec<f32>, Vec<f32>) {
        let (mut f, mut t) = (Vec::new(), Vec::new());
        if let Some((d, _, _, _)) = &source {
            let mut enc = Vec::new();
            for &r in recs {
                match &joiner {
                    None => d.read_into(r, &mut f, &mut t),
                    Some(j) => {
                        enc.clear();
                        d.read_into(r, &mut f, &mut enc);
                        for (e, ctx) in enc.as_chunks::<DIM>().0.iter().zip(d.contexts(r)) {
                            t.extend_from_slice(e);
                            t.extend_from_slice(&j.predictor(ctx));
                        }
                    }
                }
            }
        }
        (f, t)
    };
    let eval = |dp: &DataParallel, m: &Func| -> f32 {
        let local = shares.iter().sum::<usize>();
        let batches = (n_val / local.max(1)).clamp(1, 8);
        let mut total = 0f32;
        for b in 0..batches {
            let recs: Vec<usize> = (0..local)
                .map(|i| n_train + (b * local + i) % n_val)
                .collect();
            let (f, t) = read(&recs);
            let outs = dp.run(m, split_local(&f, &t));
            total += outs
                .iter()
                .zip(&shares)
                .map(|(o, &s)| o[0][0] * s as f32)
                .sum::<f32>()
                / local as f32;
        }
        total / batches as f32
    };
    if let Some((d, _, _, _)) = &source {
        eprintln!(
            "{} records ({n_train} train, {n_val} held out), {k} chunks each; batch {batch}, {steps} steps, lr {lr}",
            d.n
        );
        eprintln!("optimizer: {}", job["optimizer"]);
        if let Some(b) = cfg.qat_bits {
            eprintln!("quantization-aware: layer matrices rounded to int{b} as exported");
        }
        eprintln!(
            "loss: {}",
            match &kd {
                Some(kd) => format!(
                    "teacher token distribution (KL, temperature {}, token frames x{}) + {} x encoder squared error",
                    kd.temperature, kd.token_weight, kd.mse_weight
                ),
                None => "encoder squared error".into(),
            }
        );
        eprintln!("step {start}: held-out loss {:.4}", eval(&dp, &m));
    }

    let mut order: Vec<usize> = (0..n_train).collect();
    // A resumed run shuffles anew rather than replaying the first epoch's order.
    let mut rng = Rng(42 + start as u64);
    let mut cursor = order.len();
    let t0 = std::time::Instant::now();
    let mut avg = 0f32;
    let mut busy = vec![0f64; shares.len()];
    let sizes: Vec<usize> = std::iter::once(0)
        .chain(
            m.param_names()
                .iter()
                .map(|n| m.param_binding(n).map_or(0, <[f32]>::len)),
        )
        .collect();
    let mut samples_done = 0usize;
    let mut draw = |n: usize| -> Vec<usize> {
        (0..n)
            .map(|_| {
                if cursor == order.len() {
                    for i in (1..order.len()).rev() {
                        order.swap(i, (rng.next() % (i as u64 + 1)) as usize);
                    }
                    cursor = 0;
                }
                cursor += 1;
                order[cursor - 1]
            })
            .collect()
    };
    for step in start..steps {
        let k = plan(&micro_secs);
        let total: usize = k.iter().zip(&machine_batch).map(|(a, b)| a * b).sum();
        // The coordinator draws the round's samples and sends each machine
        // its micro-batches first, so the workers start while it computes.
        let mut own: Vec<Vec<usize>> = Vec::new();
        if rank == 0 {
            for (r, (&kr, &br)) in k.iter().zip(&machine_batch).enumerate() {
                for _ in 0..kr {
                    let recs = draw(br);
                    if r == 0 {
                        own.push(recs);
                    } else {
                        let (f, t) = read(&recs);
                        send(r, TAG_SAMPLES, &to_f16_bytes(&[f, t].concat()))?;
                    }
                }
            }
        }
        // Gradients summed sample-weighted over this machine's micro-batches,
        // then across machines.
        let mut flat = vec![0f32; sizes.iter().sum::<usize>() + 1];
        let mut compute = 0f64;
        for j in 0..k[rank] {
            let (feats, target) = if rank == 0 {
                read(&own[j])
            } else {
                let all = from_f16_bytes(&recv(0, TAG_SAMPLES)?);
                let (f, t) = all.split_at(mine * fl);
                (f.to_vec(), t.to_vec())
            };
            let t_mb = std::time::Instant::now();
            let (outputs, secs) = dp.value_and_grad(&m, split_local(&feats, &target));
            compute += t_mb.elapsed().as_secs_f64();
            for (b, t) in busy.iter_mut().zip(&secs) {
                *b += t;
            }
            let w = mine as f32;
            for (dst, v) in flat.iter_mut().zip(outputs.iter().flatten()) {
                *dst += v * w;
            }
        }
        if let Some(g) = &group {
            g.all_reduce(&mut flat, ReduceKind::Sum)
                .map_err(|e| anyhow::anyhow!("all_reduce gradients: {e:?}"))?;
            // Latest seconds per micro-batch, smoothed, for the next plan.
            let own_secs = if k[rank] > 0 {
                compute / k[rank] as f64
            } else {
                micro_secs[rank]
            };
            let latest = g
                .all_gather(&[own_secs as f32])
                .map_err(|e| anyhow::anyhow!("all_gather: {e:?}"))?;
            for (s, &l) in micro_secs.iter_mut().zip(&latest) {
                if l > 0.0 {
                    *s = if *s > 0.0 {
                        0.5 * *s + 0.5 * f64::from(l)
                    } else {
                        f64::from(l)
                    };
                }
            }
        }
        samples_done += total;
        let mut outputs = Vec::with_capacity(sizes.len());
        let mut at = 0;
        for (i, &n) in sizes.iter().enumerate() {
            let n = if i == 0 { 1 } else { n };
            outputs.push(flat[at..at + n].iter().map(|x| x / total as f32).collect());
            at += n;
        }
        let (next, l, _) = apply_gradients(&m, &mut *opt, &sched, step, 1.0, outputs);
        m = next;
        ensure!(l.is_finite(), "loss is {l} at step {step}");
        if let Some(g) = &group
            && (step + 1) % RESYNC_EVERY == 0
        {
            let mut w = flatten(&m);
            g.broadcast(0, &mut w)
                .map_err(|e| anyhow::anyhow!("broadcast weights: {e:?}"))?;
            m = unflatten(m, &w);
        }
        avg = if step == start {
            l
        } else {
            0.95 * avg + 0.05 * l
        };
        if rank == 0 && step % 20 == 0 {
            let el = t0.elapsed().as_secs_f64();
            let sps = (step + 1 - start) as f64 / el;
            let rounds: Vec<String> = k.iter().map(|x| x.to_string()).collect();
            let per: Vec<String> = names
                .iter()
                .zip(&busy)
                .map(|(n, b)| format!("{n} {:.0}%", 100.0 * b / el))
                .collect();
            let mem = mem_gb().map_or(String::new(), |(a, s)| {
                format!(", memory {a:.1} GB private {s:.1} GB shared")
            });
            eprintln!(
                "step {step}: loss {l:.4} (avg {avg:.4}), lr {:.2e}, {sps:.2} steps/s, {:.1} samples/s, micro-batches {}, busy {}, {:.1} min{mem}",
                sched.lr_at(step),
                samples_done as f64 / el,
                rounds.join("/"),
                per.join(" "),
                el / 60.0
            );
        }
        if let Some((_, teacher, teacher_dir, out)) = &source {
            if (step + 1) % eval_every == 0 {
                eprintln!("step {}: held-out loss {:.4}", step + 1, eval(&dp, &m));
            }
            if (step + 1) % save_every == 0 || step + 1 == steps {
                export::save_dir(&m, &cfg, teacher, teacher_dir, out)?;
                if keep_checkpoints {
                    let dir = out.join(format!("step-{}", step + 1));
                    export::save_dir(&m, &cfg, teacher, teacher_dir, &dir)?;
                }
                let params: NamedTensors = m
                    .param_names()
                    .into_iter()
                    .map(|n| {
                        let v = m.param_binding(&n).unwrap_or_default().to_vec();
                        (n, v)
                    })
                    .collect();
                let state = opt.state_dict().context("AdamW state")?;
                let tmp = out.join(format!("{RESUME_FILE}.tmp"));
                std::fs::write(&tmp, encode_state(step + 1, &params, &state))?;
                std::fs::rename(&tmp, out.join(RESUME_FILE))?;
                eprintln!("saved {}", out.display());
                if let (Some(max), Some((a, s))) = (max_mem_gb, mem_gb())
                    && a + s > max
                    && step + 1 < steps
                {
                    eprintln!(
                        "memory {:.1} GB is above --max-mem-gb {max}: exiting after the save (`--resume` continues)",
                        a + s
                    );
                    std::process::exit(EXIT_MEMORY);
                }
            }
        }
    }
    if source.is_some() {
        println!(
            "done: {} steps in {:.1} min, held-out loss {:.4}",
            steps - start,
            t0.elapsed().as_secs_f64() / 60.0,
            eval(&dp, &m)
        );
    } else {
        println!(
            "worker done: {} steps in {:.1} min",
            steps - start,
            t0.elapsed().as_secs_f64() / 60.0
        );
    }
    // Exit before thread-local teardown (rlx-tensor's compiled-graph cache
    // and Metal objects drop in an order that panics at process exit).
    std::process::exit(0)
}
