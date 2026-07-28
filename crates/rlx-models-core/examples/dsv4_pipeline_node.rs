// RLX — versatile ML compiler + runtime. GPLv3.
//! **Real cross-machine DeepSeek-V4 pipeline worker / coordinator.** Unlike
//! `pipeline_node.rs` (synthetic + whole-graph `partition`), each worker here
//! builds ONLY its layer range from ITS OWN checkpoint shards via
//! [`build_deepseek_v4_stage`] + [`MlxLoader::open_lazy`] — the only path that
//! works when no single machine holds the whole >RAM model. The boundary between
//! stages is the hidden state, carried as the tensor named `hidden_in`.
//!
//! Worker (one per machine; loads only its stage's shard):
//!   dsv4_pipeline_node --role worker --index 0 --layers 0:18 --first \
//!     --ckpt ~/DeepSeek-V4-Flash-2bit-DQ --addr 0.0.0.0:9101 --seq 8
//!   # middle stage: --index 1 --layers 18:35   (no --first/--last)
//!   # last stage:   --index 2 --layers 35:43 --last   (adds lm_head/norm)
//!
//! Coordinator (holds NO weights; relays the small hidden state):
//!   dsv4_pipeline_node --role coordinator --peers host-b:9101,host-c:9102,... \
//!     --ids 1,2,3,4,5,6,7,8 --vocab 129280
//!
//! `--seq` MUST equal the number of `--ids` and match across all workers (the
//! prefill graph is built for that fixed length).

use anyhow::{Context, Result};
use rlx_distributed::graph::{run_pipeline_tcp, serve_stage};
use rlx_distributed::{NamedTensor, Stage};
use rlx_ir::op::Op;
use rlx_ir::quant::QuantScheme;
use rlx_models_core::distributed_bridge::MapParamSource;
use rlx_models_core::standard_decoder::{DeepseekV4Spec, build_deepseek_v4_stage};
use rlx_models_core::weight_loader::MlxLoader;
use rlx_runtime::{CompileOptions, Device};
use std::collections::HashMap;

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}
fn has(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

/// `Op::Param` names in build order — the weights this worker must load.
fn param_names(g: &rlx_ir::Graph) -> Vec<String> {
    g.nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Param { name } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

/// Boundary names: model input on the first stage, hidden state between stages,
/// logits out of the last. Consistent name `hidden_in` = the graph's `Op::Input`.
fn io_names(first: bool, last: bool) -> (Vec<String>, Vec<String>) {
    let inp = if first { "input_ids" } else { "hidden_in" };
    let out = if last { "logits" } else { "hidden_in" };
    (vec![inp.into()], vec![out.into()])
}

/// Build this node's stage from its local shards and wrap it for the transport.
fn build_worker_stage(
    ckpt: &str,
    seq: usize,
    a: usize,
    b: usize,
    first: bool,
    last: bool,
    index: usize,
) -> Result<(Stage, MapParamSource)> {
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{ckpt}/config.json"))?)
            .context("parse config.json")?;
    let spec = DeepseekV4Spec::from_config(&cfg)?;
    let mut loader = MlxLoader::open_lazy(ckpt).context("open_lazy checkpoint")?;
    let mut packed = HashMap::<String, (Vec<u8>, QuantScheme, Vec<usize>)>::new();
    let t0 = std::time::Instant::now();
    let (graph, params) =
        build_deepseek_v4_stage(&spec, &mut loader, seq, a..b, first, last, &mut packed)?;
    let out = graph.node(*graph.outputs.first().unwrap());
    let out_shape: Vec<usize> = out.shape.dims().iter().map(|d| d.unwrap_static()).collect();
    let names = param_names(&graph);
    let f32_bytes: usize = params.values().map(|v| v.len() * 4).sum();
    let packed_bytes: usize = packed.values().map(|(b, ..)| b.len()).sum();
    eprintln!(
        "[worker {index}] layers {a}..{b} (first={first} last={last}) built in {:.1?}: {} nodes, \
         {} params ({} packed), ~{:.1} GB resident, out {out_shape:?}",
        t0.elapsed(),
        graph.len(),
        names.len(),
        packed.len(),
        (f32_bytes + packed_bytes) as f64 / 1e9,
    );
    let (inputs, outputs) = io_names(first, last);
    let stage = Stage {
        index,
        graph,
        inputs,
        outputs,
        output_shapes: vec![out_shape],
        params: names,
    };
    Ok((stage, MapParamSource::new(params, packed)))
}

/// Coordinator-side metadata stage: only the boundary NAMES matter (it holds no
/// weights and never compiles the graph).
fn meta_stage(index: usize, first: bool, last: bool) -> Stage {
    let (inputs, outputs) = io_names(first, last);
    Stage {
        index,
        graph: rlx_ir::Graph::new("meta"),
        inputs,
        outputs,
        output_shapes: vec![],
        params: vec![],
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let seq: usize = flag(&args, "--seq")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    match flag(&args, "--role").as_deref() {
        Some("worker") => {
            let index: usize = flag(&args, "--index").context("--index")?.parse()?;
            let lr = flag(&args, "--layers").context("--layers A:B")?;
            let (a, b) = lr.split_once(':').context("--layers A:B")?;
            let (a, b): (usize, usize) = (a.parse()?, b.parse()?);
            let ckpt = flag(&args, "--ckpt").context("--ckpt DIR")?;
            let addr = flag(&args, "--addr").context("--addr HOST:PORT")?;
            let (first, last) = (has(&args, "--first"), has(&args, "--last"));
            let (stage, mut src) = build_worker_stage(&ckpt, seq, a, b, first, last, index)?;
            eprintln!("[worker {index}] serving on {addr} — awaiting one forward");
            serve_stage(
                &addr,
                stage,
                &mut src,
                Device::Cpu,
                &CompileOptions::default(),
                1,
            )?;
            eprintln!("[worker {index}] done");
            Ok(())
        }
        Some("coordinator") => {
            let peers: Vec<String> = flag(&args, "--peers")
                .context("--peers a:p,b:p,...")?
                .split(',')
                .map(String::from)
                .collect();
            let ids: Vec<u32> = flag(&args, "--ids")
                .context("--ids i,i,...")?
                .split(',')
                .map(|s| s.trim().parse())
                .collect::<Result<_, _>>()?;
            anyhow::ensure!(
                ids.len() == seq,
                "--seq {seq} must equal number of --ids ({})",
                ids.len()
            );
            let vocab: usize = flag(&args, "--vocab")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let n = peers.len();
            let stages: Vec<Stage> = (0..n).map(|i| meta_stage(i, i == 0, i == n - 1)).collect();
            let ids_f: Vec<f32> = ids.iter().map(|&x| x as f32).collect();
            let input = NamedTensor::new("input_ids", vec![1, ids.len()], ids_f);
            eprintln!("[coordinator] driving {n} stages across {peers:?} for {seq} tokens");
            let t0 = std::time::Instant::now();
            let out = run_pipeline_tcp(&stages, &peers, vec![input])?;
            let logits = &out[0].data;
            eprintln!(
                "[coordinator] forward done in {:.1?}: {} logits",
                t0.elapsed(),
                logits.len()
            );
            // Last-row argmax = the model's next-token prediction.
            let v = if vocab > 0 {
                vocab
            } else {
                logits.len() / seq.max(1)
            };
            let last_row = &logits[logits.len().saturating_sub(v)..];
            let (arg, val) = last_row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, &x)| (i, x))
                .unwrap();
            let finite = logits.iter().all(|x| x.is_finite());
            println!(
                "✅ DeepSeek-V4 forward across {n} machines: {} finite logits (vocab {v}); \
                 next-token argmax = {arg} (logit {val:.4}); all finite = {finite}",
                logits.len()
            );
            anyhow::ensure!(finite, "non-finite logits");
            Ok(())
        }
        _ => {
            eprintln!("usage: --role worker|coordinator (see file header)");
            std::process::exit(2);
        }
    }
}
