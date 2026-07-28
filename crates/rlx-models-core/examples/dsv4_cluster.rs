// RLX — versatile ML compiler + runtime. GPLv3.
//! **Config-driven DeepSeek-V4 cluster runner.** One TOML describes the model +
//! nodes; the coordinator probes each node's hardware, plans a fitting layer
//! split, launches workers over SSH, drives one forward, and prints the
//! prediction plus a per-node monitor. Each node honours its own device /
//! precision / KV placement. See `dsv4_cluster.toml` for a documented config
//! template (fill in your own addresses, `~/.ssh/config` aliases, and paths).
//!
//! Coordinator (reads the config, orchestrates everything):
//!   dsv4_cluster --config dsv4_cluster.toml --model-dir <full-ckpt-on-this-host> \
//!                --remote-bin ~/rlx-models/target/release/examples/dsv4_cluster \
//!                --ids 0,671,6102,294,8760,344
//!
//! Worker (spawned by the coordinator over SSH — you rarely run this by hand):
//!   dsv4_cluster --role worker --index 1 --layers 15:31 --ckpt <dir> \
//!     --addr 0.0.0.0:9101 --seq 6 --device cuda --precision bf16 --rng 42 --kv host
//!
//! Probe (self-report hardware as JSON; used by --probe over SSH):
//!   dsv4_cluster --probe --addr <addr> --ckpt <dir>

use anyhow::{Context, Result};
use rlx_distributed::cluster::{Cluster, ClusterConfig, ModelCost, NodeReport, probe_local};
use rlx_distributed::graph::serve_stage;
use rlx_distributed::{NamedTensor, Stage};
use rlx_ir::op::Op;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{ScaleLayout, ScaledFormat};
use rlx_models_core::distributed_bridge::{ManifestParamSource, StructureLoader};
use rlx_models_core::standard_decoder::{DeepseekV4Spec, build_deepseek_v4_stage};
use rlx_models_core::weight_loader::MlxLoader;
use rlx_runtime::precision::Precision as RtPrecision;
use rlx_runtime::{CompileOptions, Device, ScaledQuantConfig, parse_device};
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn flag(a: &[String], k: &str) -> Option<String> {
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1).cloned())
}
fn has(a: &[String], k: &str) -> bool {
    a.iter().any(|x| x == k)
}

fn sq(fmt: ScaledFormat, layout: ScaleLayout) -> ScaledQuantConfig {
    ScaledQuantConfig {
        lhs_format: fmt,
        rhs_format: fmt,
        scale_layout: layout,
    }
}

/// Compile options honouring this node's precision + RNG seed. Precision is a
/// free-form string: float widths (`f32`/`f16`/`bf16`/`mixed`) set the activation
/// dtype; scaled-GEMM names (`fp8`, `fp8e5m2`, `mxfp8`, `nvfp4`, `mxfp4`) and ANY
/// `fNeXmY` minifloat (`f8e4m3`, `f6e3m2`, `f4e2m1`, `f4e3m0`, …) opt every dense
/// MatMul into a dynamically-quantized low-bit GEMM (changes numerics).
fn node_opts(precision: &str, rng_seed: u64) -> Result<CompileOptions> {
    use ScaleLayout::*;
    use ScaledFormat::*;
    let mut o = CompileOptions {
        rng: rlx_ir::RngOptions::philox(rng_seed),
        ..CompileOptions::default()
    };
    match precision {
        "f32" | "mixed" => o.precision = RtPrecision::F32,
        "f16" => o.precision = RtPrecision::F16,
        "bf16" => o.precision = RtPrecision::BF16,
        "fp8" | "fp8e4m3" => o.scaled_quant = Some(ScaledQuantConfig::fp8_e4m3()),
        "fp8e5m2" => o.scaled_quant = Some(sq(F8E5M2, PerTensor)),
        "mxfp8" => o.scaled_quant = Some(ScaledQuantConfig::mxfp8_e4m3()),
        "nvfp4" => o.scaled_quant = Some(sq(F4E2M1, Nvfp4 { group: 16 })),
        "mxfp4" => o.scaled_quant = Some(sq(F4E2M1, BlockMxE8M0 { block: 32 })),
        other => {
            let fmt: ScaledFormat = other
                .parse()
                .map_err(|_| anyhow::anyhow!("unknown precision `{other}` (f32/f16/bf16/fp8/fp8e5m2/mxfp8/nvfp4/mxfp4 or an fNeXmY minifloat)"))?;
            // Sub-byte formats need microscaling block scales; 8-bit uses per-tensor.
            let layout = if fmt.bit_width() <= 6 {
                BlockMxE8M0 { block: 32 }
            } else {
                PerTensor
            };
            o.scaled_quant = Some(sq(fmt, layout));
        }
    }
    Ok(o)
}

fn param_names(g: &rlx_ir::Graph) -> Vec<String> {
    g.nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Param { name } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();

    // ── Probe: self-report hardware as JSON ──
    if has(&a, "--probe") {
        let addr = flag(&a, "--addr").unwrap_or_else(|| "0.0.0.0:0".into());
        let ckpt = flag(&a, "--ckpt").unwrap_or_else(|| ".".into());
        println!(
            "{}",
            serde_json::to_string(&probe_local(&addr, &ckpt, true))?
        );
        return Ok(());
    }

    match flag(&a, "--role").as_deref() {
        Some("worker") => worker(&a),
        _ => coordinator(&a),
    }
}

// ─────────────────────────── worker ───────────────────────────
fn worker(a: &[String]) -> Result<()> {
    let index: usize = flag(a, "--index").context("--index")?.parse()?;
    let lr = flag(a, "--layers").context("--layers A:B")?;
    let (ls, le) = lr.split_once(':').context("--layers A:B")?;
    let (ls, le): (usize, usize) = (ls.parse()?, le.parse()?);
    let ckpt = flag(a, "--ckpt").context("--ckpt")?;
    let addr = flag(a, "--addr").context("--addr")?;
    let seq: usize = flag(a, "--seq").and_then(|s| s.parse().ok()).unwrap_or(8);
    let device = flag(a, "--device").unwrap_or_else(|| "cpu".into());
    let precision = flag(a, "--precision").unwrap_or_else(|| "bf16".into());
    let rng: u64 = flag(a, "--rng").and_then(|s| s.parse().ok()).unwrap_or(0);
    let kv = flag(a, "--kv").unwrap_or_else(|| "none".into());
    let (first, last) = (has(a, "--first"), has(a, "--last"));

    // Primary compute device (first of a "cuda+cpu"-style list).
    let dev = device
        .split('+')
        .next()
        .and_then(|d| parse_device(d).ok())
        .unwrap_or(Device::Cpu);

    // A CUDA stage here (~42 GB) dwarfs the 3080 Ti's 16 GB VRAM, so page it via
    // managed memory (RLX_CUDA_UNIFIED) rather than OOM on a resident cudaMalloc.
    // The operator can still force resident VRAM by exporting the flag to "0".
    if matches!(dev, Device::Cuda) && std::env::var_os("RLX_CUDA_UNIFIED").is_none() {
        // SAFETY: single-threaded at this point (set before the Session/arena or
        // any worker thread is created).
        unsafe { std::env::set_var("RLX_CUDA_UNIFIED", "1") };
    }
    // wgpu: this model's stacked MoE weights are large — one proj's 256-expert
    // codes tensor is ~2 GiB, above wgpu's default 2 GiB storage-binding cap, so
    // the sharded arena can't place it (a single tensor can't span shards).
    // Large-buffer mode unclamps to the device max (RADV supports 4 GiB bindings).
    if matches!(dev, Device::Gpu | Device::WebGpu)
        && std::env::var_os("RLX_WGPU_LARGE_BUFFERS").is_none()
    {
        unsafe { std::env::set_var("RLX_WGPU_LARGE_BUFFERS", "1") };
    }

    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{ckpt}/config.json"))?)?;
    let spec = DeepseekV4Spec::from_config(&cfg)?;
    let mut loader = MlxLoader::open_lazy(&ckpt).context("open_lazy")?;
    let mut packed = HashMap::<String, (Vec<u8>, QuantScheme, Vec<usize>)>::new();
    let t0 = Instant::now();
    // STRUCTURE build: size the graph + record a load manifest WITHOUT materializing
    // the big packed codes (empty in `packed`; only the small BF16 scales/biases are
    // retained). Keeps build RAM ≈ scales, so the source weights and the compiled
    // arena never coexist at 2× — the peak that OOM'd a stage ≈ the node's RAM.
    let (graph, params, manifest) = {
        let mut sloader = StructureLoader::new(&mut loader);
        let (g, p) =
            build_deepseek_v4_stage(&spec, &mut sloader, seq, ls..le, first, last, &mut packed)?;
        (g, p, sloader.manifest)
    };
    drop(loader); // structure pass done; a fresh loader streams codes at serve time
    let build_ms = t0.elapsed().as_millis() as u64;

    let resident: u64 = params.values().map(|v| (v.len() * 4) as u64).sum::<u64>()
        + packed.values().map(|(b, ..)| b.len() as u64).sum::<u64>();
    let out = graph.node(*graph.outputs.first().unwrap());
    let out_shape: Vec<usize> = out.shape.dims().iter().map(|d| d.unwrap_static()).collect();
    let names = param_names(&graph);

    // Report to the coordinator (parsed from stdout).
    let report = NodeReport {
        addr: addr.clone(),
        layers: ls..le,
        device: rlx_runtime::device_label(dev).to_string(),
        precision: precision.clone(),
        build_ms,
        resident_bytes: resident,
        n_params: names.len(),
        n_packed: packed.len(),
    };
    println!("NODEREPORT {}", serde_json::to_string(&report)?);
    eprintln!(
        "[worker {index}] layers {ls}..{le} device={} precision={precision} rng={rng} kv={kv} \
               built {:.1}s ~{:.1}GB out {out_shape:?}",
        rlx_runtime::device_label(dev),
        build_ms as f64 / 1000.0,
        resident as f64 / 1e9
    );

    let stage = Stage {
        index,
        graph,
        inputs: vec![if first { "input_ids" } else { "hidden_in" }.into()],
        outputs: vec![if last { "logits" } else { "hidden_in" }.into()],
        output_shapes: vec![out_shape],
        params: names,
    };
    // Retained packed bytes are the non-empty BF16 scales/biases; the deferred codes
    // are empty here and streamed from the checkpoint during compile via the manifest.
    let synth_packed: HashMap<String, Vec<u8>> = packed
        .into_iter()
        .filter(|(_, (b, _, _))| !b.is_empty())
        .map(|(k, (b, ..))| (k, b))
        .collect();
    // Fresh loader for streaming (the structure pass marked keys taken on the first).
    let mut loader2 = MlxLoader::open_lazy(&ckpt).context("open_lazy(serve)")?;
    let mut src = ManifestParamSource {
        loader: &mut loader2,
        manifest,
        synth: params,
        synth_packed,
    };

    // ── Isolated profile (RLX_DSV4_SELFTEST): compile + run ONE forward on this
    // node with a synthetic input, print the stage forward time, then exit. No
    // coordinator / network / other nodes — lets us profile a stage in isolation
    // (with RLX_HD_PROFILE for the dtoh/compute split) despite a flaky cluster.
    if std::env::var_os("RLX_DSV4_SELFTEST").is_some() {
        let in_name = if first { "input_ids" } else { "hidden_in" };
        // Boundary input shape: first stage takes input_ids [1,seq]; others take
        // hidden_in [seq, hc_mult, dim] (matches build_deepseek_v4_stage).
        let in_shape: Vec<usize> = if first {
            vec![1, seq]
        } else {
            vec![seq, spec.hc_mult, spec.dim]
        };
        let n_in: usize = in_shape.iter().product::<usize>().max(1);
        let mut runner = rlx_distributed::StageRunner::compile(
            stage,
            &mut src,
            dev,
            &node_opts(&precision, rng)?,
        );
        let pool: HashMap<String, NamedTensor> = std::iter::once((
            in_name.to_string(),
            NamedTensor::new(in_name, in_shape, vec![0.0f32; n_in]),
        ))
        .collect();
        let t = Instant::now();
        let _ = runner.run(&pool);
        eprintln!(
            "[SELFTEST] stage {ls}..{le} forward: {:.2}s",
            t.elapsed().as_secs_f64()
        );
        return Ok(());
    }
    // serve_stage prints "serving on …" itself, AFTER the arena is compiled/loaded,
    // so the coordinator waits for real readiness (slow paging loads included).
    // n_requests=0 → serve indefinitely (prefill + every decode step); the
    // coordinator kills us when the generation loop is done.
    serve_stage(&addr, stage, &mut src, dev, &node_opts(&precision, rng)?, 0)?;
    Ok(())
}

// ─────────────────────────── coordinator ───────────────────────────
fn coordinator(a: &[String]) -> Result<()> {
    let cfg_path = flag(a, "--config").context("--config cluster.toml")?;
    let remote_bin =
        flag(a, "--remote-bin").context("--remote-bin <path to this binary on nodes>")?;
    let model_dir = flag(a, "--model-dir")
        .context("--model-dir <full checkpoint on this host, for the cost model>")?;
    let ids: Vec<u32> = flag(a, "--ids")
        .context("--ids i,i,...")?
        .split(',')
        .map(|s| s.trim().parse())
        .collect::<Result<_, _>>()?;

    let cfg = ClusterConfig::from_path(&cfg_path)?;
    anyhow::ensure!(
        ids.len() == cfg.seq,
        "--ids count ({}) must equal config seq ({})",
        ids.len(),
        cfg.seq
    );
    let mut cx = Cluster::from_config(cfg);

    // 1) Probe hardware.
    println!("── probing {} nodes ──", cx.cfg.nodes.len());
    cx.probe(&remote_bin)?;
    for (n, caps) in cx.cfg.nodes.iter().zip(&cx.caps) {
        println!("  {:<22} {}", n.addr, caps.summary());
    }

    // 2) Plan placement from the model's resident cost.
    let model = model_cost(&model_dir)?;
    println!(
        "── planning: {} layers, ~{:.0} GB resident, policy {:?} ──",
        model.n_layers,
        model.total_bytes() as f64 / 1e9,
        cx.cfg.placement.policy
    );
    cx.plan(model)?;
    for a in &cx.plan {
        println!(
            "  {:<22} layers {:>2}..{:<2} on {:<6} ~{:.1}/{:.1} GB{}{}",
            a.addr,
            a.layers.start,
            a.layers.end,
            a.device,
            a.est_bytes as f64 / 1e9,
            a.budget_bytes as f64 / 1e9,
            if a.first { " +embed" } else { "" },
            if a.last { " +head" } else { "" }
        );
    }

    // 3) Distribute: ensure each node holds exactly the shards its layers need
    //    (idempotent rsync from the coordinator's full checkpoint over LAN).
    println!("── distributing weights ──");
    distribute(&cx, &model_dir)?;

    // 4) Launch workers + await readiness (reading their stdout — never port-probe,
    //    an empty connect kills a serve_stage worker).
    println!("── launching workers ──");
    let mut kids = cx.launch(&remote_bin)?;
    // Generous: a node paging a big stage in (managed CUDA / swap-backed Vulkan)
    // compiles its arena slowly, and readiness is now signalled post-compile.
    let ready = await_ready(&mut kids, cx.cfg.nodes.len(), Duration::from_secs(900))?;
    println!("  all {} workers serving", ready.len());

    // 4) Prefill forward (= TTFT), timed per stage.
    let n_gen: usize = flag(a, "--gen").and_then(|s| s.parse().ok()).unwrap_or(0);
    let vocab = 129280usize;
    let argmax = |o: &[f32]| -> (usize, f32) {
        let last = &o[o.len().saturating_sub(vocab)..];
        last.iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .map(|(i, &v)| (i, v))
            .unwrap()
    };
    println!("── prefill: {} tokens ──", ids.len());
    let mut window: Vec<f32> = ids.iter().map(|&x| x as f32).collect();
    let t_prefill = Instant::now();
    let mut run = cx.drive(vec![NamedTensor::new(
        "input_ids",
        vec![1, window.len()],
        window.clone(),
    )])?;
    let ttft_ms = t_prefill.elapsed().as_millis();
    let (mut tok, val) = argmax(&run.output);
    let mut generated = vec![tok];
    println!(
        "  TTFT {:.1}s → token {tok} (logit {val:.3})",
        ttft_ms as f64 / 1000.0
    );

    // 5) Decode loop. No KV-cache yet, so each token is a full sliding-window
    //    forward — this measures the pipeline's real per-token throughput.
    let mut decode_ms: Vec<u128> = Vec::new();
    for step in 0..n_gen {
        window.push(tok as f32);
        if window.len() > cx.cfg.seq {
            let drop = window.len() - cx.cfg.seq;
            window.drain(0..drop);
        }
        let t = Instant::now();
        let r = cx.drive(vec![NamedTensor::new(
            "input_ids",
            vec![1, window.len()],
            window.clone(),
        )])?;
        let dt = t.elapsed().as_millis();
        decode_ms.push(dt);
        let (a2, _) = argmax(&r.output);
        generated.push(a2);
        tok = a2;
        eprintln!(
            "  decode {}/{n_gen}: {:.1}s → token {a2}",
            step + 1,
            dt as f64 / 1000.0
        );
    }
    for k in kids.iter_mut() {
        let _ = k.kill();
    }

    // 6) Output + metrics.
    println!("\n✅ generated token ids: {generated:?}");
    println!(
        "   TTFT (prefill → 1st token): {:.2} s",
        ttft_ms as f64 / 1000.0
    );
    if decode_ms.is_empty() {
        println!("   decode: none (pass --gen N for sustained-TPS decode)");
    } else {
        let avg = decode_ms.iter().sum::<u128>() as f64 / decode_ms.len() as f64;
        println!(
            "   decode: {} tokens, avg {:.2} s/token → {:.3} tok/s sustained (no KV-cache; each token = full forward)",
            decode_ms.len(),
            avg / 1000.0,
            1000.0 / avg
        );
    }
    // Merge worker build stats into the monitor table (prefill stage timings).
    for t in run.timings.iter_mut() {
        if let Some(r) = ready
            .iter()
            .find(|r| r.addr == t.addr || t.addr.ends_with(&r.addr) || r.addr.ends_with(&t.addr))
        {
            t.build_ms = r.build_ms;
            t.resident_bytes = r.resident_bytes;
        }
    }
    println!("\n{}", run.table());
    Ok(())
}

/// Read each worker's piped stdout until it prints `serving on`, collecting its
/// `NODEREPORT` line. Returns the reports; errors if any node isn't ready in time.
fn await_ready(
    kids: &mut [std::process::Child],
    n: usize,
    timeout: Duration,
) -> Result<Vec<NodeReport>> {
    let reports = Arc::new(Mutex::new(Vec::<NodeReport>::new()));
    let ready = Arc::new(Mutex::new(0usize));
    let mut handles = Vec::new();
    for k in kids.iter_mut() {
        let Some(out) = k.stdout.take() else { continue };
        let reports = reports.clone();
        let ready = ready.clone();
        handles.push(std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                if let Some(js) = line.strip_prefix("NODEREPORT ") {
                    if let Ok(r) = serde_json::from_str::<NodeReport>(js) {
                        reports.lock().unwrap().push(r);
                    }
                } else if line.contains("serving on") {
                    *ready.lock().unwrap() += 1;
                    break;
                }
            }
        }));
    }
    let t0 = Instant::now();
    while *ready.lock().unwrap() < n {
        if t0.elapsed() > timeout {
            anyhow::bail!(
                "only {}/{n} workers became ready in {:?}",
                *ready.lock().unwrap(),
                timeout
            );
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(Arc::try_unwrap(reports).unwrap().into_inner().unwrap())
}

/// Which shard files a layer range (+ optional embed/head) touches, from the
/// `model.safetensors.index.json` weight_map.
fn shards_for(
    idx: &serde_json::Value,
    layers: &std::ops::Range<usize>,
    first: bool,
    last: bool,
) -> std::collections::BTreeSet<String> {
    let mut set = std::collections::BTreeSet::new();
    if let Some(wm) = idx.get("weight_map").and_then(|v| v.as_object()) {
        for (tensor, shard) in wm {
            let Some(shard) = shard.as_str() else {
                continue;
            };
            let keep = if let Some(l) = tensor
                .strip_prefix("model.layers.")
                .and_then(|r| r.split('.').next())
                .and_then(|n| n.parse::<usize>().ok())
            {
                layers.contains(&l)
            } else if tensor.contains("embed_tokens") {
                first
            } else if tensor.starts_with("lm_head") || tensor == "model.norm.weight" {
                last
            } else {
                false
            };
            if keep {
                set.insert(shard.to_string());
            }
        }
    }
    set
}

/// Push each remote node exactly the shards its planned layers need, plus its
/// config + index. rsync is idempotent (size/mtime) so re-runs skip present
/// files. The coordinator's own (ssh-less) node is assumed to hold the checkpoint.
fn distribute(cx: &Cluster, model_dir: &str) -> Result<()> {
    let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(format!(
        "{model_dir}/model.safetensors.index.json"
    ))?)?;
    for (i, a) in cx.plan.iter().enumerate() {
        let Some(ssh) = &a.ssh else {
            println!("  {:<22} local — full checkpoint present", a.addr);
            continue;
        };
        let dst = &cx.cfg.nodes[i].ckpt_dir;
        let shards = shards_for(&idx, &a.layers, a.first, a.last);
        let mut files: Vec<String> = vec![
            "config.json".into(),
            "model.safetensors.index.json".into(),
            "tokenizer.json".into(),
            "tokenizer_config.json".into(),
        ];
        files.extend(shards.iter().cloned());
        let sources: Vec<String> = files
            .iter()
            .map(|f| format!("{model_dir}/{f}"))
            .filter(|p| std::path::Path::new(p).exists())
            .collect();
        let status = std::process::Command::new("rsync")
            .arg("-a")
            .args(&sources)
            .arg(format!("{ssh}:{dst}/"))
            .status()?;
        anyhow::ensure!(status.success(), "rsync to {ssh} failed");
        println!(
            "  {:<22} layers {}..{} → {} shards on {ssh}",
            a.addr,
            a.layers.start,
            a.layers.end,
            shards.len()
        );
    }
    Ok(())
}

/// Estimate the model's resident cost from the local full checkpoint: sum the
/// shard bytes (≈ resident with bf16-kept scales), split into per-layer + embed
/// + head using the config's layer count.
fn model_cost(dir: &str) -> Result<ModelCost> {
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{dir}/config.json"))?)?;
    let n_layers = cfg
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .context("num_hidden_layers")? as usize;
    let mut disk = 0u64;
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.extension().and_then(|x| x.to_str()) == Some("safetensors") {
            disk += std::fs::metadata(&p)?.len();
        }
    }
    anyhow::ensure!(
        disk > 0,
        "no .safetensors in {dir} (need the full checkpoint for the cost model)"
    );
    // Resident ≈ disk (bf16 scales kept) + ~15% for f32 norms/attention scales.
    let resident = (disk as f64 * 1.15) as u64;
    // Embed + head are the bf16 vocab table (vocab × hidden × 2), computed exactly
    // from config — the old flat "5% of resident" over-charged the first/last nodes
    // ~5× (real ≈ 1 GB, not ~5.5 GB), which stole layer capacity from the
    // GTT-limited Vulkan node. Head is free when word embeddings are tied.
    let vocab = cfg.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(0);
    let hidden = cfg.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(0);
    let tied = cfg
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let embed = vocab * hidden * 2;
    let head = if tied { 0 } else { embed };
    let body = resident.saturating_sub(embed + head);
    let per_layer = (body as f64 / n_layers as f64) as u64;
    Ok(ModelCost {
        n_layers,
        per_layer_bytes: per_layer,
        embed_bytes: embed,
        head_bytes: head,
        per_layer_flops: 1.0,
    })
}
