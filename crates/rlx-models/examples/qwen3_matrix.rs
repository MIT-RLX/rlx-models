// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Machine-readable Qwen3-0.6B matrix: Candle reference vs RLX backends.
// Requires `--features parity-candle` plus any backend passthroughs you
// want to exercise.

use anyhow::Result;
use candle_core::{DType as CDType, Device as CDevice, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3 as candle_qwen3;
use rlx_models::qwen3::{
    Qwen3Config, Qwen3Generator, SampleOpts, build_qwen3_graph_sized,
    build_qwen3_graph_sized_last_logits,
};
use rlx_models::weight_map::WeightMap;
use rlx_runtime::Device;
use std::env;
use std::time::Instant;

const HIDDEN_TOL: f32 = 5e-3;
const LOGIT_TOL: f32 = 1e-1;

const TOKEN_POOL: &[u32] = &[
    1, 17, 42, 314, 2718, 9001, 27182, 8128, 65535, 12345, 256, 1024, 4096, 16384, 32768, 100, 200,
    300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
    11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000,
    24000, 25000, 26000, 27000, 28000, 29000, 30000, 31000, 32000, 33000, 34000, 35000, 36000,
    37000, 38000, 39000, 40000, 41000, 42000, 43000, 44000, 45000, 46000, 47000, 48000, 49000,
    50000, 51000, 52000, 53000, 54000, 55000, 56000, 57000, 58000, 59000, 60000, 61000, 62000,
    63000, 64000, 65000, 66000, 67000, 68000, 69000, 70000, 71000, 72000, 73000, 74000, 75000,
    76000, 77000, 78000, 79000, 80000, 81000, 82000, 83000, 84000, 85000, 86000, 87000, 88000,
    89000, 90000, 91000, 92000, 93000, 94000, 95000, 96000, 97000, 98000, 99000, 100000, 101000,
    102000, 103000, 104000, 105000, 106000, 107000, 108000, 109000, 110000,
];

fn main() -> Result<()> {
    let weights = rlx_ir::env::var("RLX_QWEN3_WEIGHTS")
        .ok_or_else(|| anyhow::anyhow!("set RLX_QWEN3_WEIGHTS"))?;
    let cfg_path = rlx_ir::env::var("RLX_QWEN3_CONFIG")
        .ok_or_else(|| anyhow::anyhow!("set RLX_QWEN3_CONFIG"))?;
    let reps = rlx_ir::env::var("RLX_QWEN3_MATRIX_REPS")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3)
        .max(1);

    let cfg = Qwen3Config::from_file(std::path::Path::new(&cfg_path))?;
    let candle_cfg = to_candle_cfg(&cfg);
    let devices = parse_devices(env::args().skip(1).collect())?;

    println!(
        "kind,impl,backend,mode,batch,seq,shape_ok,max_abs,mean_abs,cos_mean,cos_min,top1_match,top1_total,min_ms,median_ms,status,message"
    );

    for &batch in &[1usize, 2, 4] {
        for &seq in &[8usize, 32, 64, 128] {
            let ids = make_batched_ids(batch, seq);
            let (candle_hidden, candle_hidden_ms) =
                run_candle_hidden(&weights, &candle_cfg, batch, seq, &ids)?;
            let (candle_logits, candle_logits_ms) =
                run_candle_logits(&weights, &candle_cfg, batch, seq, &ids)?;

            print_reference_row("candle", "CPU", "hidden", batch, seq, candle_hidden_ms);
            print_reference_row("candle", "CPU", "last", batch, seq, candle_logits_ms);

            for &device in &devices {
                if device == Device::Gpu && batch == 4 && seq == 128 {
                    print_skip(device, "all", batch, seq, "wgpu WebGPU buffer cap");
                    continue;
                }

                let (hidden, hidden_times, hidden_shape_ok) =
                    run_rlx(&weights, &cfg, device, batch, seq, RlxMode::Hidden, reps)?;
                let hidden_metrics =
                    metrics(&hidden, &candle_hidden, batch * seq, cfg.hidden_size, None);
                print_metric_row(
                    "rlx",
                    device.name(),
                    "hidden",
                    batch,
                    seq,
                    hidden_shape_ok,
                    &hidden_metrics,
                    &hidden_times,
                    hidden_metrics.max_abs <= HIDDEN_TOL,
                    "",
                );

                for mode in [RlxMode::FullLogits, RlxMode::LastLogits] {
                    let (logits, times, shape_ok) =
                        run_rlx(&weights, &cfg, device, batch, seq, mode, reps)?;
                    let last = match mode {
                        RlxMode::FullLogits => {
                            extract_last_position(&logits, batch, seq, cfg.vocab_size)
                        }
                        RlxMode::LastLogits => logits,
                        RlxMode::Hidden => unreachable!(),
                    };
                    let m = metrics(
                        &last,
                        &candle_logits,
                        batch,
                        cfg.vocab_size,
                        Some(cfg.vocab_size),
                    );
                    print_metric_row(
                        "rlx",
                        device.name(),
                        mode.name(),
                        batch,
                        seq,
                        shape_ok,
                        &m,
                        &times,
                        m.max_abs <= LOGIT_TOL && m.top1_match == m.top1_total,
                        "",
                    );
                }

                if rlx_ir::env::flag("RLX_QWEN3_FUSION_REPORT") {
                    let report = fusion_report(&weights, &cfg, batch, seq)?;
                    println!(
                        "fusion,rlx,{},graph,{batch},{seq},true,,,,,,,,,ok,{}",
                        device.name(),
                        report
                    );
                }
            }
        }
    }

    if rlx_ir::env::flag("RLX_QWEN3_DECODE") {
        run_decode_matrix(&weights, &cfg, &devices)?;
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RlxMode {
    Hidden,
    FullLogits,
    LastLogits,
}

impl RlxMode {
    fn name(self) -> &'static str {
        match self {
            Self::Hidden => "hidden",
            Self::FullLogits => "full",
            Self::LastLogits => "last",
        }
    }
}

#[derive(Default)]
struct Metrics {
    max_abs: f32,
    mean_abs: f32,
    cos_mean: f32,
    cos_min: f32,
    top1_match: usize,
    top1_total: usize,
}

fn run_rlx(
    weights: &str,
    cfg: &Qwen3Config,
    device: Device,
    batch: usize,
    seq: usize,
    mode: RlxMode,
    reps: usize,
) -> Result<(Vec<f32>, Vec<f64>, bool)> {
    let mut wm = WeightMap::from_file(weights)?;
    let (graph, params) = match mode {
        RlxMode::Hidden => build_qwen3_graph_sized(
            cfg, &mut wm, batch, seq, /*with_lm_head*/ false, /*with_kv_outputs*/ false,
        )?,
        RlxMode::FullLogits => build_qwen3_graph_sized(
            cfg, &mut wm, batch, seq, /*with_lm_head*/ true, /*with_kv_outputs*/ false,
        )?,
        RlxMode::LastLogits => build_qwen3_graph_sized_last_logits(
            cfg, &mut wm, batch, seq, /*with_kv_outputs*/ false,
        )?,
    };

    let expected_len = match mode {
        RlxMode::Hidden => batch * seq * cfg.hidden_size,
        RlxMode::FullLogits => batch * seq * cfg.vocab_size,
        RlxMode::LastLogits => batch * cfg.vocab_size,
    };

    let mut compiled =
        rlx_models::flow_util::compile_graph_qwen3_prefill_with_params(device, graph, params)?;

    let ids = make_batched_ids(batch, seq);
    let ids_f32: Vec<f32> = ids.iter().map(|&x| x as f32).collect();
    let _ = compiled.run(&[("input_ids", ids_f32.as_slice())]);

    let mut times = Vec::with_capacity(reps);
    let mut last = Vec::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        let outputs = compiled.run(&[("input_ids", ids_f32.as_slice())]);
        times.push(t0.elapsed().as_secs_f64() * 1e3);
        last = outputs.into_iter().next().unwrap_or_default();
    }
    let shape_ok = last.len() == expected_len;
    Ok((last, times, shape_ok))
}

fn run_candle_logits(
    weights: &str,
    cfg: &candle_qwen3::Config,
    batch: usize,
    seq: usize,
    ids: &[u32],
) -> Result<(Vec<f32>, f64)> {
    let device = CDevice::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], CDType::F32, &device)? };
    let mut model = candle_qwen3::ModelForCausalLM::new(cfg, vb)?;
    let input = Tensor::from_vec(ids.to_vec(), (batch, seq), &device)?;
    let t0 = Instant::now();
    let logits = model.forward(&input, 0)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok((logits.flatten_all()?.to_vec1::<f32>()?, ms))
}

fn run_candle_hidden(
    weights: &str,
    cfg: &candle_qwen3::Config,
    batch: usize,
    seq: usize,
    ids: &[u32],
) -> Result<(Vec<f32>, f64)> {
    let device = CDevice::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], CDType::F32, &device)? };
    let mut model = candle_qwen3::Model::new(cfg, vb)?;
    let input = Tensor::from_vec(ids.to_vec(), (batch, seq), &device)?;
    let t0 = Instant::now();
    let hidden = model.forward(&input, 0)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok((hidden.flatten_all()?.to_vec1::<f32>()?, ms))
}

fn metrics(a: &[f32], b: &[f32], rows: usize, row_dim: usize, top1_dim: Option<usize>) -> Metrics {
    if a.len() != b.len() || a.len() != rows * row_dim {
        return Metrics::default();
    }
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut cos_sum = 0f64;
    let mut cos_min = f32::INFINITY;
    for r in 0..rows {
        let aa = &a[r * row_dim..(r + 1) * row_dim];
        let bb = &b[r * row_dim..(r + 1) * row_dim];
        let mut dot = 0f32;
        let mut na = 0f32;
        let mut nb = 0f32;
        for i in 0..row_dim {
            let d = (aa[i] - bb[i]).abs();
            max_abs = max_abs.max(d);
            sum_abs += d as f64;
            dot += aa[i] * bb[i];
            na += aa[i] * aa[i];
            nb += bb[i] * bb[i];
        }
        let cos = (dot / (na.sqrt() * nb.sqrt()).max(f32::MIN_POSITIVE)).clamp(-1.0, 1.0);
        cos_sum += cos as f64;
        cos_min = cos_min.min(cos);
    }

    let (top1_match, top1_total) = if let Some(vocab) = top1_dim {
        top1_agreement(a, b, vocab)
    } else {
        (0, 0)
    };

    Metrics {
        max_abs,
        mean_abs: (sum_abs / a.len() as f64) as f32,
        cos_mean: (cos_sum / rows as f64) as f32,
        cos_min,
        top1_match,
        top1_total,
    }
}

fn print_metric_row(
    imp: &str,
    backend: &str,
    mode: &str,
    batch: usize,
    seq: usize,
    shape_ok: bool,
    m: &Metrics,
    times: &[f64],
    pass: bool,
    message: &str,
) {
    let (min, median) = min_median(times);
    println!(
        "prefill,{imp},{backend},{mode},{batch},{seq},{shape_ok},{:.6},{:.6},{:.7},{:.7},{},{},{:.1},{:.1},{},{}",
        m.max_abs,
        m.mean_abs,
        m.cos_mean,
        m.cos_min,
        m.top1_match,
        m.top1_total,
        min,
        median,
        if pass && shape_ok { "ok" } else { "fail" },
        message
    );
}

fn print_reference_row(imp: &str, backend: &str, mode: &str, batch: usize, seq: usize, ms: f64) {
    println!(
        "prefill,{imp},{backend},{mode},{batch},{seq},true,0,0,1,1,0,0,{ms:.1},{ms:.1},ok,reference"
    );
}

fn print_skip(device: Device, mode: &str, batch: usize, seq: usize, message: &str) {
    println!(
        "prefill,rlx,{},{mode},{batch},{seq},false,,,,,,,,,skip,{}",
        device.name(),
        message
    );
}

fn min_median(times: &[f64]) -> (f64, f64) {
    let min = times.iter().copied().fold(f64::INFINITY, f64::min);
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    (min, sorted[sorted.len() / 2])
}

fn extract_last_position(logits: &[f32], batch: usize, seq: usize, vocab: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(batch * vocab);
    for b in 0..batch {
        let start = b * seq * vocab + (seq - 1) * vocab;
        out.extend_from_slice(&logits[start..start + vocab]);
    }
    out
}

fn top1_agreement(a: &[f32], b: &[f32], vocab: usize) -> (usize, usize) {
    let rows = a.len() / vocab;
    let mut matched = 0;
    for r in 0..rows {
        if argmax(&a[r * vocab..(r + 1) * vocab]) == argmax(&b[r * vocab..(r + 1) * vocab]) {
            matched += 1;
        }
    }
    (matched, rows)
}

fn argmax(xs: &[f32]) -> usize {
    xs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn make_batched_ids(batch: usize, seq: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(batch * seq);
    for b in 0..batch {
        let offset = (b * 7) % TOKEN_POOL.len();
        for i in 0..seq {
            out.push(TOKEN_POOL[(offset + i) % TOKEN_POOL.len()]);
        }
    }
    out
}

fn parse_devices(args: Vec<String>) -> Result<Vec<Device>> {
    if args.is_empty() {
        return Ok(vec![Device::Cpu, Device::Metal, Device::Mlx, Device::Gpu]);
    }
    args.iter()
        .map(|s| match s.as_str() {
            "cpu" => Ok(Device::Cpu),
            "metal" | "mps" => Ok(Device::Metal),
            "mlx" => Ok(Device::Mlx),
            "gpu" | "wgpu" => Ok(Device::Gpu),
            other => anyhow::bail!("unknown device {other}; use cpu|metal|mps|mlx|gpu|wgpu"),
        })
        .collect()
}

fn fusion_report(weights: &str, cfg: &Qwen3Config, batch: usize, seq: usize) -> Result<String> {
    use rlx_opt::{FusionOptions, FusionReport, FusionTarget, fusion_passes, run_passes};

    let mut wm = WeightMap::from_file(weights)?;
    let (graph, _) = build_qwen3_graph_sized_last_logits(
        cfg, &mut wm, batch, seq, /*with_kv_outputs*/ false,
    )?;
    let before = graph.clone();
    let passes = fusion_passes(FusionTarget::Cpu, FusionOptions::default());
    let after = run_passes(graph, &passes, false);
    Ok(FusionReport::analyze(&before, &after).summary_line())
}

fn run_decode_matrix(weights: &str, cfg: &Qwen3Config, devices: &[Device]) -> Result<()> {
    let prompt_len = rlx_ir::env::var("RLX_QWEN3_DECODE_PROMPT")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(128);
    let steps = rlx_ir::env::var("RLX_QWEN3_DECODE_STEPS")
        .map(|s| {
            s.split(',')
                .map(|p| p.trim().parse::<usize>())
                .collect::<std::result::Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_else(|| vec![1, 16, 128]);
    let prompt = make_batched_ids(1, prompt_len);

    for &device in devices {
        for &n in &steps {
            if device == Device::Gpu && prompt_len + n >= 128 {
                println!(
                    "decode,rlx,{},{},1,{prompt_len},false,,,,,,,,,skip,wgpu buffer cap risk",
                    device.name(),
                    n
                );
                continue;
            }

            let mut naive = Qwen3Generator::from_path(cfg.clone(), weights, device)?;
            naive.prefill(&prompt);
            let t0 = Instant::now();
            let naive_tokens = naive.generate(n, SampleOpts::greedy())?;
            let naive_ms = t0.elapsed().as_secs_f64() * 1e3;

            let mut cached = Qwen3Generator::from_path(cfg.clone(), weights, device)?
                .with_prefill_cache(4)
                .with_decode_cache(prompt_len + n + 8);
            cached.prefill(&prompt);
            let t0 = Instant::now();
            let cached_tokens = cached.generate_cached(n, SampleOpts::greedy())?;
            let cached_ms = t0.elapsed().as_secs_f64() * 1e3;

            let matched = naive_tokens == cached_tokens;
            println!(
                "decode,rlx,{},{n},1,{prompt_len},{matched},0,0,1,1,{},{},{:.1},{:.1},{},cached_vs_naive",
                device.name(),
                usize::from(matched),
                1,
                cached_ms,
                naive_ms,
                if matched { "ok" } else { "fail" }
            );
        }
    }
    Ok(())
}

fn to_candle_cfg(cfg: &Qwen3Config) -> candle_qwen3::Config {
    use candle_nn::Activation;
    let hidden_act = match cfg.hidden_act.as_str() {
        "silu" => Activation::Silu,
        "gelu" => Activation::Gelu,
        other => panic!("unsupported hidden_act for parity: {other}"),
    };
    candle_qwen3::Config {
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        head_dim: cfg.head_dim,
        attention_bias: cfg.attention_bias,
        num_key_value_heads: cfg.num_key_value_heads,
        max_position_embeddings: cfg.max_position_embeddings,
        sliding_window: cfg.sliding_window,
        max_window_layers: cfg.max_window_layers,
        tie_word_embeddings: cfg.tie_word_embeddings,
        rope_theta: cfg.rope_theta,
        rms_norm_eps: cfg.rms_norm_eps,
        use_sliding_window: cfg.use_sliding_window,
        hidden_act,
    }
}
