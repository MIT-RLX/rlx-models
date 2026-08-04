// RLX — versatile ML compiler + runtime. GPLv3.
//! **Distributed DECODE launcher** for DeepSeek-V4 — the runnable wrapper around
//! the validated `serve_v4_decode_stage` (worker) / `run_v4_decode_pipelined_tcp`
//! (coordinator). Each worker builds ONE pipeline stage over its layer range
//! (holding only that range's KV cache) from an mlx-community/`deepseek-ai`
//! checkpoint via the lazy loader, and serves the per-token loop over TCP; the
//! coordinator relays each stage's hidden boundary and greedily samples.
//!
//! The forward math + the cross-node seam are unit-validated (a 2-stage TCP
//! decode == single-node, bit-exact). A real run just needs the checkpoint on
//! each node with a per-stage layer range that fits its RAM (the 167 GB GA MXFP4
//! doesn't fit the 3-node box — use the preview 2-bit or a smaller GA quant).
//!
//! Worker (one per node, layer range = this node's slice):
//!   dsv4_decode_cluster --role worker --ckpt <dir> --layers 0:15 --first \
//!                       --addr 0.0.0.0:9200 [--max-comp 512]
//!   dsv4_decode_cluster --role worker --ckpt <dir> --layers 15:31 --addr 0.0.0.0:9201
//!   dsv4_decode_cluster --role worker --ckpt <dir> --layers 31:43 --last \
//!                       --addr 0.0.0.0:9202
//! Coordinator (drives the token loop; --peers in stage order):
//!   dsv4_decode_cluster --role coordinator --ckpt <dir> \
//!       --peers 192.168.99.148:9200,192.168.99.76:9201,192.168.99.161:9202 \
//!       --ids 0,671,6102,294,8760,344 --gen 32
//!
//! Thunderbolt: the transport is interface-agnostic and auto-tuned (TCP_NODELAY +
//! large SO_SNDBUF/RCVBUF, `RLX_V4_SOCKBUF` bytes) for a high-bandwidth link — to
//! route the stage-boundary relay over a Thunderbolt bridge (~10-40 Gbps) instead
//! of gigabit Ethernet, just pass the TB-bridge interface IPs (macOS `bridge0` /
//! Linux `thunderbolt-net`, e.g. `ifconfig bridge0`) as --addr / --peers.

use anyhow::{Context, Result};
use rlx_models_core::standard_decoder::{
    DeepseekV4Spec, DsV4RefLoader, V4Decoder, run_v4_decode_pipelined_tcp, serve_v4_decode_stage,
};
use rlx_models_core::weight_loader::MlxLoader;
use rlx_runtime::Device;

fn flag(a: &[String], k: &str) -> Option<String> {
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1))
        .cloned()
}
fn has(a: &[String], k: &str) -> bool {
    a.iter().any(|x| x == k)
}

fn read_spec(ckpt: &str) -> Result<DeepseekV4Spec> {
    let bytes = std::fs::read(format!("{ckpt}/config.json")).context("read config.json")?;
    let cfg: serde_json::Value = serde_json::from_slice(&bytes).context("parse config.json")?;
    DeepseekV4Spec::from_config(&cfg)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let ckpt = flag(&a, "--ckpt").context("--ckpt <checkpoint dir>")?;
    let spec = read_spec(&ckpt)?;
    let max_win = spec.window_size.max(1) - 1;
    let max_comp: usize = flag(&a, "--max-comp")
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let device = match flag(&a, "--device").as_deref() {
        Some("metal") => Device::Metal,
        _ => Device::Cpu,
    };

    match flag(&a, "--role").as_deref() {
        Some("worker") => {
            let lr = flag(&a, "--layers").context("--layers A:B")?;
            let (lo, hi) = lr.split_once(':').context("--layers A:B")?;
            let layers = lo.parse::<usize>()?..hi.parse::<usize>()?;
            let (first, last) = (has(&a, "--first"), has(&a, "--last"));
            let addr = flag(&a, "--addr").context("--addr host:port")?;
            eprintln!(
                "[dsv4-decode] worker layers {:?} first={first} last={last} on {addr} (max_win {max_win}, max_comp {max_comp})",
                layers
            );
            // `--stacked`: mlx-community format (experts already stacked under
            // `ffn.switch_mlp.*`, e.g. the 2bit/3bit-DQ) → use the mmap loader
            // directly. Default: Vontra reference format (per-expert
            // `ffn.experts.{e}.w*`) → the DsV4RefLoader name-map + per-expert stack.
            let inner = MlxLoader::open_lazy(&ckpt).context("open mlx checkpoint")?;
            let t_load = std::time::Instant::now();
            let mut dec = if has(&a, "--stacked") {
                let mut loader = inner;
                V4Decoder::new_stage(
                    &spec,
                    &mut loader,
                    layers,
                    first,
                    last,
                    max_win,
                    max_comp,
                    device,
                )
                .context("build stage decoder")?
            } else {
                let mut loader = DsV4RefLoader::new(Box::new(inner), spec.n_routed_experts);
                V4Decoder::new_stage(
                    &spec,
                    &mut loader,
                    layers,
                    first,
                    last,
                    max_win,
                    max_comp,
                    device,
                )
                .context("build stage decoder")?
            };
            eprintln!(
                "[dsv4-decode] stage built + compiled in {:?}",
                t_load.elapsed()
            );
            if has(&a, "--build-only") {
                eprintln!("[dsv4-decode] --build-only: stage loads OK, exiting");
                return Ok(());
            }
            // --forward: run N decode steps on this stage's REAL weights and print the
            // output stats (proves execution, not just load/compile). A `first` stage
            // consumes token ids; the output is logits (if also `--last`) or the hidden
            // boundary. `--ids i,j,k` sets the tokens (default 0,1,2,3).
            if has(&a, "--forward") {
                let ids: Vec<u32> = flag(&a, "--ids")
                    .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
                    .unwrap_or_else(|| vec![0, 1, 2, 3]);
                for (i, &tok) in ids.iter().enumerate() {
                    let t0 = std::time::Instant::now();
                    let (logits, hidden) = dec.step_io(if first { Some(tok) } else { None }, None);
                    let out = if last {
                        logits.expect("last→logits")
                    } else {
                        hidden.expect("hidden")
                    };
                    let finite = out.iter().all(|v| v.is_finite());
                    let (mn, mx) = out
                        .iter()
                        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                    let mean = out.iter().sum::<f32>() / out.len().max(1) as f32;
                    let kind = if last { "logits" } else { "hidden" };
                    eprintln!(
                        "[dsv4-forward] step {i} tok {tok}: {kind} len {} finite={finite} min={mn:.4} max={mx:.4} mean={mean:.4} in {:?}",
                        out.len(),
                        t0.elapsed()
                    );
                    if last {
                        let am =
                            out.iter()
                                .enumerate()
                                .fold(
                                    (0usize, f32::MIN),
                                    |(bi, bv), (j, &v)| {
                                        if v > bv { (j, v) } else { (bi, bv) }
                                    },
                                );
                        eprintln!("    argmax token = {}", am.0);
                    }
                }
                return Ok(());
            }
            let listener =
                std::net::TcpListener::bind(&addr).with_context(|| format!("bind {addr}"))?;
            eprintln!("[dsv4-decode] serving on {addr}");
            serve_v4_decode_stage(&mut dec, listener)?;
            eprintln!("[dsv4-decode] worker done");
        }
        Some("coordinator") => {
            let peers: Vec<String> = flag(&a, "--peers")
                .context("--peers a,b,c (stage order)")?
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();
            let ids: Vec<u32> = flag(&a, "--ids")
                .context("--ids i,i,...")?
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            let n_gen: usize = flag(&a, "--gen").and_then(|s| s.parse().ok()).unwrap_or(32);
            eprintln!(
                "[dsv4-decode] coordinator: {} stages, prompt {} tok, gen {n_gen}",
                peers.len(),
                ids.len()
            );
            let t0 = std::time::Instant::now();
            let out = run_v4_decode_pipelined_tcp(&peers, spec.vocab_size, &ids, n_gen)?;
            let ms = t0.elapsed().as_millis();
            println!(
                "generated {} tokens in {ms} ms ({:.1} ms/tok): {out:?}",
                out.len(),
                ms as f64 / n_gen.max(1) as f64
            );
        }
        other => anyhow::bail!("--role worker|coordinator (got {other:?})"),
    }
    Ok(())
}
