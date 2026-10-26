// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Distributed **layer pipeline** for Kimi-K3 — inference and training.
//!
//! Each worker owns a contiguous layer range and reads it from its own local
//! checkpoint copy; the boundary state (hidden + every Attention-Residual
//! snapshot) moves node to node. The coordinator embeds, relays through the
//! workers in order, and applies the head. The backbone is thus read
//! concurrently from the nodes' disks instead of one.
//!
//! **One worker, every job** ([`serve`]): stateless prefix forwards, stateful
//! O(1)/token decode, and pipeline-parallel training of its range
//! ([`crate::train_stream::Stage`]) — the coordinator decides per request.
//!
//! **Wire protocol.** A request is `[op u8][len u64][payload]`. A reply is any
//! number of heartbeat frames `[2]` — sent every [`HEARTBEAT`] while the worker
//! computes — then `[0][len u64][payload]` or `[1][len u64][utf-8 error]`. The
//! heartbeat is what lets the coordinator bound every read: a node that loses
//! power or network keeps its socket "open" forever, and without a bound the
//! coordinator would look exactly like a model that is still thinking; with
//! it, a worker silent for [`DistOptions::timeout`] is an error naming that
//! worker, while a layer range that legitimately takes minutes (streaming
//! from disk) keeps the connection alive. Worker-side errors and panics are
//! reported in the reply and the worker keeps serving.
//!
//! **Agreement before work.** [`Cluster::connect`] pings every worker and
//! checks their configs match the coordinator's and that the stage ranges
//! tile `0..num_layers` exactly; a mismatch is otherwise invisible (every shape
//! still agrees and the model answers plausible nonsense).
//!
//! **Pipeline training** ([`PipelineTrainer`]): GPipe-style — one thread per
//! stage connection, micro-batches flow forward through the stages and back
//! while later ones are still going forward, gradients accumulate on the
//! worker that owns each layer, then one global-norm clip and one optimizer
//! step everywhere.

use crate::config::{KimiK3Config, KimiLinearConfig};
use crate::flow::FlowConfig;
use crate::layer_exec::{LayerSpec, head_loss_backward};
use crate::loader::CheckpointLoader;
use crate::runner::{
    DecodeState, HeadCache, LayerCache, apply_head_cached, argmax, decode_forward_range,
    run_layer_range_streaming,
};
use crate::source::{CheckpointSource, HeadWeights, WeightSource};
use crate::train::{AdamW, Batch, TrainMode, TrainSpec, global_norm};
use crate::train_stream::{Params, Stage, StreamOptions, StreamStep};
use anyhow::{Context, Result, bail, ensure};
use rlx_distributed::rpc::{Dec, Enc, RpcClient, RpcHandler, RpcOptions};
use rlx_runtime::Device;
use std::collections::HashMap;
use std::net::TcpListener;
use std::path::Path;

const EMB: &str = "language_model.model.embed_tokens.weight";

pub use rlx_distributed::rpc::HEARTBEAT;
const PROTOCOL: u32 = 3;

mod op {
    pub const PING: u8 = 1;
    pub const RANGE_FWD: u8 = 2;
    pub const DECODE: u8 = 3;
    pub const DECODE_DROP: u8 = 4;
    pub const TRAIN_INIT: u8 = 16;
    pub const TRAIN_FWD: u8 = 17;
    pub const TRAIN_BWD: u8 = 18;
    pub const TRAIN_GRAD_SQ: u8 = 19;
    pub const TRAIN_APPLY: u8 = 20;
    pub const TRAIN_SAVE: u8 = 21;
    pub const TRAIN_FETCH: u8 = 22;
    pub const TRAIN_LR: u8 = 23;
}

/// The boundary state on the wire: the hidden state, then every snapshot.
trait StateEnc {
    fn state(self, h: &[f32], snaps: &[Vec<f32>]) -> Self;
}

impl StateEnc for Enc {
    fn state(self, h: &[f32], snaps: &[Vec<f32>]) -> Self {
        self.vecs(std::iter::once(h).chain(snaps.iter().map(Vec::as_slice)))
    }
}

trait StateDec {
    fn state(&mut self) -> Result<(Vec<f32>, Vec<Vec<f32>>)>;
}

impl StateDec for Dec<'_> {
    fn state(&mut self) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let mut v = self.vecs()?;
        ensure!(!v.is_empty(), "state without a hidden vector");
        let snaps = v.split_off(1);
        Ok((v.pop().expect("non-empty"), snaps))
    }
}

// ── worker ─────────────────────────────────────────────────────────────────

/// How a worker runs.
pub struct ServeOptions<'a> {
    pub device: Device,
    /// Stop after this many requests (`0` = serve until a shutdown request).
    pub max_requests: usize,
    /// Inference shapes for a sequence length (default: the checkpoint config at
    /// batch 1).
    pub inference_cfg: Option<Box<dyn Fn(usize) -> FlowConfig + Send + Sync + 'a>>,
}

impl Default for ServeOptions<'_> {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            max_requests: 0,
            inference_cfg: None,
        }
    }
}

struct Worker<'a> {
    model_dir: String,
    config: KimiK3Config,
    device: Device,
    inference_cfg: Box<dyn Fn(usize) -> FlowConfig + Send + Sync + 'a>,
    ck: CheckpointLoader,
    /// Decode state per sequence id (concurrent sequences share the worker).
    decode: HashMap<u64, DecodeState>,
    cache: LayerCache,
    stage: Option<Stage>,
}

impl RpcHandler for Worker<'_> {
    fn handle(&mut self, op_code: u8, payload: &[u8]) -> Result<Vec<u8>> {
        self.handle_op(op_code, payload)
    }
}

impl Worker<'_> {
    fn tc(&self) -> &KimiLinearConfig {
        &self.config.text_config
    }

    fn handle_op(&mut self, op_code: u8, payload: &[u8]) -> Result<Vec<u8>> {
        let mut d = Dec(payload);
        match op_code {
            op::PING => {
                let tc = self.tc();
                let info = serde_json::json!({
                    "protocol": PROTOCOL,
                    "model_dir": self.model_dir,
                    "device": format!("{:?}", self.device),
                    "config": config_fingerprint(tc),
                    "num_layers": tc.num_hidden_layers,
                    "hidden": tc.hidden_size,
                    "vocab": tc.vocab_size,
                    "training": self.stage.as_ref().map(|s| (s.range.start, s.range.end)),
                    "sequences": self.decode.len(),
                });
                Ok(Enc::default().str(&info.to_string()).0)
            }
            op::RANGE_FWD => {
                let (start, end) = (d.u32()? as usize, d.u32()? as usize);
                let (h, snaps) = d.state()?;
                self.check_range(start, end)?;
                let seq = h.len() / self.tc().hidden_size.max(1);
                let cfg = (self.inference_cfg)(seq);
                let tc = self.config.text_config.clone();
                let (h, snaps) = run_layer_range_streaming(
                    &mut self.ck,
                    &tc,
                    &cfg,
                    h,
                    snaps,
                    start,
                    end,
                    self.device,
                )?;
                Ok(Enc::default().state(&h, &snaps).0)
            }
            op::DECODE => {
                let reset = d.u8()? != 0;
                let seq_id = d.u64()?;
                let (start, end) = (d.u32()? as usize, d.u32()? as usize);
                let (h, snaps) = d.state()?;
                self.check_range(start, end)?;
                let tc = self.config.text_config.clone();
                let cfg1 = (self.inference_cfg)(1);
                if reset || !self.decode.contains_key(&seq_id) {
                    self.decode.insert(seq_id, DecodeState::zeros(&tc, &cfg1));
                }
                let seq = h.len() / tc.hidden_size.max(1);
                let cfg = (self.inference_cfg)(seq);
                let state = self.decode.get_mut(&seq_id).expect("set above");
                let (h, snaps) = decode_forward_range(
                    &mut self.ck,
                    &tc,
                    &cfg,
                    h,
                    snaps,
                    state,
                    start,
                    end,
                    &mut self.cache,
                    self.device,
                )?;
                Ok(Enc::default().state(&h, &snaps).0)
            }
            op::DECODE_DROP => {
                let seq_id = d.u64()?;
                self.decode.remove(&seq_id);
                Ok(Vec::new())
            }
            op::TRAIN_INIT => {
                let init: StageInit = serde_json::from_str(&d.str()?)?;
                self.check_range(init.start, init.end)?;
                let src = CheckpointSource::open(&self.model_dir)?;
                let cfg = self.tc().flow_config(init.batch, init.seq);
                let mut stage = Stage::new(
                    Box::new(src),
                    cfg,
                    init.spec()?,
                    init.start..init.end,
                    StreamOptions {
                        lr: init.lr,
                        weight_decay: init.weight_decay,
                        clip: None,
                        device: self.device,
                        seed: init.seed,
                        cache_layers: init.cache_layers,
                    },
                );
                if let Some(dir) = &init.resume {
                    let d = Path::new(dir).join(stage_dir(init.start, init.end));
                    stage.store = crate::train_stream::load_params(&d.join("trained.safetensors"))?;
                    stage.opt.load(&d.join("optimizer.safetensors"))?;
                }
                self.stage = Some(stage);
                Ok(Vec::new())
            }
            op::TRAIN_FWD => {
                let mb = d.u64()?;
                let (h, snaps) = d.state()?;
                let (h, snaps) = self.stage()?.forward_mb(mb, h, snaps)?;
                Ok(Enc::default().state(&h, &snaps).0)
            }
            op::TRAIN_BWD => {
                let mb = d.u64()?;
                let (ct_h, ct_snaps) = d.state()?;
                let (dh, ds) = self.stage()?.backward_mb(mb, ct_h, ct_snaps)?;
                Ok(Enc::default().state(&dh, &ds).0)
            }
            op::TRAIN_GRAD_SQ => Ok(Enc::default().f64(self.stage()?.grad_sq_norm()).0),
            op::TRAIN_APPLY => {
                let scale = d.f32()?;
                self.stage()?.apply(scale);
                Ok(Vec::new())
            }
            op::TRAIN_LR => {
                let lr = d.f32()?;
                self.stage()?.opt.lr = lr;
                Ok(Vec::new())
            }
            op::TRAIN_SAVE => {
                let dir = d.str()?;
                let st = self.stage()?;
                let d = Path::new(&dir).join(stage_dir(st.range.start, st.range.end));
                std::fs::create_dir_all(&d)?;
                crate::train_stream::save_params(&d.join("trained.safetensors"), &st.store)?;
                st.opt.save(&d.join("optimizer.safetensors"))?;
                Ok(Vec::new())
            }
            op::TRAIN_FETCH => {
                let st = self.stage()?;
                let mut names: Vec<&String> = st.store.keys().collect();
                names.sort();
                let mut e = Enc::default().u32(names.len() as u32);
                for n in names {
                    let shape = st.shapes.get(n).cloned().unwrap_or_default();
                    e = e.str(n).u32(shape.len() as u32);
                    for s in shape {
                        e = e.u64(s as u64);
                    }
                    e = e.f32s(&st.store[n]);
                }
                Ok(e.0)
            }
            other => bail!("unknown request {other}"),
        }
    }

    fn stage(&mut self) -> Result<&mut Stage> {
        self.stage
            .as_mut()
            .context("training request before TRAIN_INIT on this worker")
    }

    fn check_range(&self, start: usize, end: usize) -> Result<()> {
        let n = self.tc().num_hidden_layers;
        ensure!(
            start < end && end <= n,
            "layer range {start}..{end} is not inside this checkpoint's 0..{n}"
        );
        Ok(())
    }
}

fn stage_dir(start: usize, end: usize) -> String {
    format!("stage-{start:03}-{end:03}")
}

/// A digest of the text config, compared across the cluster: two nodes with
/// different configs would agree on every shape and still compute different
/// models.
pub fn config_fingerprint(tc: &KimiLinearConfig) -> String {
    let s = format!("{tc:?}");
    let h = s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
    });
    format!("{h:016x}")
}

/// Serve the layer pipeline on `addr` from the checkpoint in `model_dir`, over
/// [`rlx_distributed::rpc`]: one thread per connection (a health probe is never
/// stuck behind an idle coordinator), requests run one at a time against the
/// shared state (decode caches, a training stage), which survives reconnects.
/// Returns after a shutdown request (or `max_requests`).
pub fn serve(addr: &str, model_dir: &str, opts: ServeOptions<'_>) -> Result<()> {
    let listener = TcpListener::bind(addr).with_context(|| format!("bind {addr}"))?;
    let config = KimiK3Config::load(Path::new(model_dir).join("config.json"))?;
    let tc = config.text_config.clone();
    let worker = Worker {
        model_dir: model_dir.to_string(),
        ck: CheckpointLoader::open(model_dir)?,
        inference_cfg: opts
            .inference_cfg
            .unwrap_or_else(|| Box::new(move |seq| tc.flow_config(1, seq))),
        config,
        device: opts.device,
        decode: HashMap::new(),
        cache: LayerCache::from_env(),
        stage: None,
    };
    eprintln!(
        "kimi worker on {addr} (checkpoint {model_dir}, {:?})",
        opts.device
    );
    rlx_distributed::rpc::serve(listener, worker, opts.max_requests)
}

// ── coordinator side ───────────────────────────────────────────────────────

/// Coordinator-side timeouts and retries ([`RpcOptions`]).
pub type DistOptions = RpcOptions;

/// A persistent connection to one worker.
pub struct Conn {
    pub addr: String,
    client: RpcClient,
}

impl Conn {
    pub fn connect(addr: &str, opts: &DistOptions) -> Result<Self> {
        Ok(Self {
            addr: addr.to_string(),
            client: RpcClient::connect(addr, opts)?,
        })
    }

    fn call(&mut self, op_code: u8, payload: &[u8]) -> Result<Vec<u8>> {
        self.client
            .call(op_code, payload)
            .map_err(|e| anyhow::anyhow!("worker {e:#}"))
    }

    pub fn ping(&mut self) -> Result<serde_json::Value> {
        let r = self.call(op::PING, &[])?;
        Ok(serde_json::from_str(&Dec(&r).str()?)?)
    }

    pub fn shutdown(&mut self) -> Result<()> {
        self.client.shutdown()
    }

    fn state_call(&mut self, op_code: u8, payload: Enc) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let r = self.call(op_code, &payload.0)?;
        Dec(&r).state()
    }
}

pub use rlx_distributed::stages::{StageSpec, parse_stages, plan_stages, validate_stages};

/// Connected, agreed-upon pipeline workers.
pub struct Cluster {
    pub stages: Vec<StageSpec>,
    pub conns: Vec<Conn>,
    /// Next decode sequence id to hand out (ids name per-sequence state on
    /// every worker).
    pub next_seq: u64,
}

impl Cluster {
    /// Connect to every stage, then check they agree: same protocol, same model
    /// config as `tc` (when given), and ranges that tile the model. Ranges may
    /// cover a prefix of the layers when `tc` is `None` (truncated runs).
    pub fn connect(
        stages: &[StageSpec],
        tc: Option<&KimiLinearConfig>,
        opts: &DistOptions,
    ) -> Result<Self> {
        let mut conns = Vec::with_capacity(stages.len());
        for s in stages {
            let mut c = Conn::connect(&s.addr, opts)?;
            let info = c.ping()?;
            ensure!(
                info["protocol"].as_u64() == Some(PROTOCOL as u64),
                "worker {} speaks protocol {:?}, this coordinator {PROTOCOL}",
                s.addr,
                info["protocol"]
            );
            if let Some(tc) = tc {
                let want = config_fingerprint(tc);
                ensure!(
                    info["config"].as_str() == Some(want.as_str()),
                    "worker {} has a different model config ({} vs {want}, checkpoint {}): \
                     every node must run the same model",
                    s.addr,
                    info["config"],
                    info["model_dir"]
                );
            }
            conns.push(c);
        }
        if let Some(tc) = tc {
            validate_stages(stages, tc.num_hidden_layers)?;
        }
        Ok(Self {
            stages: stages.to_vec(),
            conns,
            next_seq: 1,
        })
    }

    /// Stateless forward of `(h, snaps)` through every stage.
    pub fn prefix(
        &mut self,
        h: Vec<f32>,
        snaps: Vec<Vec<f32>>,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let (mut h, mut snaps) = (h, snaps);
        for (s, c) in self.stages.iter().zip(&mut self.conns) {
            let e = Enc::default()
                .u32(s.layers.start as u32)
                .u32(s.layers.end as u32)
                .state(&h, &snaps);
            (h, snaps) = c.state_call(op::RANGE_FWD, e)?;
        }
        Ok((h, snaps))
    }

    /// One decode step through every stage (`reset` starts a new sequence) for
    /// the default sequence (id 0).
    pub fn decode_step(
        &mut self,
        reset: bool,
        h: Vec<f32>,
        snaps: Vec<Vec<f32>>,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        self.decode_step_seq(0, reset, h, snaps)
    }

    /// One decode step of sequence `seq_id` through every stage.
    pub fn decode_step_seq(
        &mut self,
        seq_id: u64,
        reset: bool,
        h: Vec<f32>,
        snaps: Vec<Vec<f32>>,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let (mut h, mut snaps) = (h, snaps);
        for (s, c) in self.stages.iter().zip(&mut self.conns) {
            let e = Enc::default()
                .u8(reset as u8)
                .u64(seq_id)
                .u32(s.layers.start as u32)
                .u32(s.layers.end as u32)
                .state(&h, &snaps);
            (h, snaps) = c.state_call(op::DECODE, e)?;
        }
        Ok((h, snaps))
    }

    /// Free sequence `seq_id`'s decode state on every worker.
    pub fn drop_sequence(&mut self, seq_id: u64) -> Result<()> {
        for c in &mut self.conns {
            c.call(op::DECODE_DROP, &Enc::default().u64(seq_id).0)?;
        }
        Ok(())
    }

    pub fn shutdown(mut self) -> Result<()> {
        for c in &mut self.conns {
            c.shutdown()?;
        }
        Ok(())
    }
}

fn tuple_stages(stages: &[(String, usize, usize)]) -> Vec<StageSpec> {
    stages
        .iter()
        .map(|(a, s, e)| StageSpec {
            addr: a.clone(),
            layers: *s..*e,
        })
        .collect()
}

// ── compatibility entry points (the `kimi_k3_cluster` example) ────────────

/// Serve stateless layer-range forwards (and every other request) on `addr`.
/// `n_requests == 0` serves until shutdown.
pub fn serve_worker(
    addr: &str,
    model_dir: &str,
    _tc: &KimiLinearConfig,
    make_cfg: impl Fn(usize) -> FlowConfig + Send + Sync,
    device: Device,
    n_requests: usize,
) -> Result<()> {
    serve(
        addr,
        model_dir,
        ServeOptions {
            device,
            max_requests: n_requests,
            inference_cfg: Some(Box::new(make_cfg)),
        },
    )
}

/// Same server as [`serve_worker`]; decode state lives in it either way.
pub fn serve_decode_worker(
    addr: &str,
    model_dir: &str,
    tc: &KimiLinearConfig,
    make_cfg: impl Fn(usize) -> FlowConfig + Send + Sync,
    device: Device,
    n_requests: usize,
) -> Result<()> {
    serve_worker(addr, model_dir, tc, make_cfg, device, n_requests)
}

/// Embed `tokens` locally and relay the state through `stages`; returns the
/// final `(hidden, snapshots)` for the caller's head.
pub fn run_distributed_prefix(
    ck: &mut CheckpointLoader,
    cfg: &FlowConfig,
    tokens: &[u32],
    stages: &[(String, usize, usize)],
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let h = ck.gather_embed(EMB, tokens, cfg.hidden)?;
    let mut cluster = Cluster::connect(&tuple_stages(stages), None, &DistOptions::default())?;
    cluster.prefix(h, Vec::new())
}

/// Distributed greedy generation: prefill across the workers, then O(1) decode
/// steps; output identical to single-node [`crate::runner::run_generate`].
pub fn run_distributed_generate(
    ck: &mut CheckpointLoader,
    make_cfg: impl Fn(usize) -> FlowConfig,
    prompt: &[u32],
    n_gen: usize,
    stages: &[(String, usize, usize)],
    device: Device,
) -> Result<Vec<u32>> {
    let mut cluster = Cluster::connect(&tuple_stages(stages), None, &DistOptions::default())?;
    generate(&mut cluster, ck, make_cfg, prompt, n_gen, device)
}

/// Greedy generation over a connected [`Cluster`].
pub fn generate(
    cluster: &mut Cluster,
    ck: &mut CheckpointLoader,
    make_cfg: impl Fn(usize) -> FlowConfig,
    prompt: &[u32],
    n_gen: usize,
    device: Device,
) -> Result<Vec<u32>> {
    ensure!(!prompt.is_empty(), "empty prompt");
    let cfg1 = make_cfg(1);
    let hidden = cfg1.hidden;
    let mut head = HeadCache::from_env();
    let h0 = ck.gather_embed(EMB, prompt, hidden)?;
    let (h, snaps) = cluster.decode_step(true, h0, Vec::new())?;
    let last = prompt.len() - 1;
    let h_last = h[last * hidden..(last + 1) * hidden].to_vec();
    let snaps_last: Vec<Vec<f32>> = snaps
        .iter()
        .map(|s| s[last * hidden..(last + 1) * hidden].to_vec())
        .collect();
    let mut tok = argmax(&apply_head_cached(
        ck,
        &cfg1,
        &h_last,
        &snaps_last,
        &mut head,
        device,
    )?);
    let mut out = vec![tok];
    for _ in 1..n_gen {
        let hin = ck.gather_embed(EMB, &[tok], hidden)?;
        let (h, snaps) = cluster.decode_step(false, hin, Vec::new())?;
        tok = argmax(&apply_head_cached(
            ck, &cfg1, &h, &snaps, &mut head, device,
        )?);
        out.push(tok);
    }
    Ok(out)
}

/// Greedy generation of several prompts at once over a [`Cluster`], pipelined:
/// one thread per stage keeps every stage busy with a different sequence, so
/// `n` stages serve up to `n` sequences in the time one takes alone. Each
/// sequence's tokens equal what [`generate`] produces for it. Every sequence's
/// decode state (KDA state + MLA cache) lives on the workers until it finishes
/// — size the batch to their memory.
pub fn generate_batch(
    cluster: &mut Cluster,
    ck: &mut CheckpointLoader,
    make_cfg: impl Fn(usize) -> FlowConfig,
    prompts: &[Vec<u32>],
    n_gen: usize,
    device: Device,
) -> Result<Vec<Vec<u32>>> {
    ensure!(prompts.iter().all(|p| !p.is_empty()), "empty prompt");
    ensure!(n_gen > 0, "nothing to generate");
    let cfg1 = make_cfg(1);
    let hidden = cfg1.hidden;
    let mut head = HeadCache::from_env();
    let base = cluster.next_seq;
    cluster.next_seq += prompts.len() as u64;
    type Step = (usize, bool, Vec<f32>, Vec<Vec<f32>>);
    let n_stages = cluster.conns.len();
    let (to_coord, from_stages) = std::sync::mpsc::channel::<Result<Step>>();
    let mut inboxes = Vec::with_capacity(n_stages);
    let mut outboxes = Vec::with_capacity(n_stages);
    for _ in 0..n_stages {
        let (tx, rx) = std::sync::mpsc::channel::<Option<Step>>();
        inboxes.push(tx);
        outboxes.push(rx);
    }
    let conns = std::mem::take(&mut cluster.conns);
    let stages = cluster.stages.clone();
    let mut outs: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    let result = std::thread::scope(|scope| -> Result<Vec<Conn>> {
        let mut handles = Vec::new();
        for (s, (mut conn, rx)) in conns.into_iter().zip(outboxes).enumerate() {
            let next = (s + 1 < n_stages).then(|| inboxes[s + 1].clone());
            let coord = to_coord.clone();
            let layers = stages[s].layers.clone();
            handles.push(scope.spawn(move || -> Conn {
                while let Ok(Some((i, reset, h, snaps))) = rx.recv() {
                    let e = Enc::default()
                        .u8(reset as u8)
                        .u64(base + i as u64)
                        .u32(layers.start as u32)
                        .u32(layers.end as u32)
                        .state(&h, &snaps);
                    match conn.state_call(op::DECODE, e) {
                        Ok((h, snaps)) => match &next {
                            Some(n) => {
                                let _ = n.send(Some((i, reset, h, snaps)));
                            }
                            None => {
                                let _ = coord.send(Ok((i, reset, h, snaps)));
                            }
                        },
                        Err(e) => {
                            let _ = coord.send(Err(e));
                        }
                    }
                }
                conn
            }));
        }
        drop(to_coord);
        let drive = (|| -> Result<()> {
            for (i, p) in prompts.iter().enumerate() {
                let h0 = ck.gather_embed(EMB, p, hidden)?;
                inboxes[0]
                    .send(Some((i, true, h0, Vec::new())))
                    .map_err(|_| anyhow::anyhow!("first stage thread is gone"))?;
            }
            let mut open = prompts.len();
            while open > 0 {
                let (i, _, h, snaps) =
                    from_stages.recv().context("stage threads ended early")??;
                // A prefill returns every prompt position; the head reads the last.
                let row = if outs[i].is_empty() {
                    prompts[i].len() - 1
                } else {
                    0
                };
                let h_last = h[row * hidden..(row + 1) * hidden].to_vec();
                let snaps_last: Vec<Vec<f32>> = snaps
                    .iter()
                    .map(|s| s[row * hidden..(row + 1) * hidden].to_vec())
                    .collect();
                let tok = argmax(&apply_head_cached(
                    ck,
                    &cfg1,
                    &h_last,
                    &snaps_last,
                    &mut head,
                    device,
                )?);
                outs[i].push(tok);
                if outs[i].len() < n_gen {
                    let hin = ck.gather_embed(EMB, &[tok], hidden)?;
                    inboxes[0]
                        .send(Some((i, false, hin, Vec::new())))
                        .map_err(|_| anyhow::anyhow!("first stage thread is gone"))?;
                } else {
                    open -= 1;
                }
            }
            Ok(())
        })();
        for tx in &inboxes {
            let _ = tx.send(None);
        }
        let conns: Vec<Conn> = handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| anyhow::anyhow!("stage thread panicked"))
            })
            .collect::<Result<_>>()?;
        drive.map(|_| conns)
    });
    cluster.conns = result?;
    for i in 0..prompts.len() {
        cluster.drop_sequence(base + i as u64)?;
    }
    Ok(outs)
}

// ── pipeline-parallel training ─────────────────────────────────────────────

/// What a worker builds its training [`Stage`] from.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct StageInit {
    start: usize,
    end: usize,
    batch: usize,
    seq: usize,
    /// `full` or `lora`.
    mode: String,
    rank: usize,
    alpha: f32,
    targets: Vec<String>,
    freeze: Vec<String>,
    only: Vec<String>,
    lr: f32,
    weight_decay: f32,
    seed: u64,
    cache_layers: bool,
    resume: Option<String>,
}

impl StageInit {
    fn spec(&self) -> Result<TrainSpec> {
        let mut s = match self.mode.as_str() {
            "full" => TrainSpec::full(),
            "lora" => TrainSpec::lora(self.rank, self.alpha, &self.targets),
            m => bail!("unknown training mode {m:?}"),
        };
        s.freeze = self.freeze.clone();
        s.only = self.only.clone();
        Ok(s)
    }
}

/// Settings for a [`PipelineTrainer`].
#[derive(Debug, Clone)]
pub struct PipelineOptions {
    pub spec: TrainSpec,
    pub batch: usize,
    pub seq: usize,
    pub lr: f32,
    pub weight_decay: f32,
    pub clip: Option<f32>,
    pub seed: u64,
    /// Workers keep their layers' weights in RAM between steps.
    pub cache_layers: bool,
    /// Resume every stage (and the head) from a [`PipelineTrainer::save`] dir.
    pub resume: Option<String>,
    pub device: Device,
    pub dist: DistOptions,
}

enum StageMsg {
    Fwd(u64, Vec<f32>, Vec<Vec<f32>>),
    Bwd(u64, Vec<f32>, Vec<Vec<f32>>),
}

/// Pipeline-parallel streamed training: the coordinator holds the embedding and
/// head, each worker trains its layer range.
pub struct PipelineTrainer {
    pub cluster: Cluster,
    src: CheckpointSource,
    cfg: FlowConfig,
    spec: TrainSpec,
    head: HeadWeights,
    head_store: Params,
    head_values: Params,
    head_grads: Params,
    head_shapes: HashMap<String, Vec<usize>>,
    head_opt: AdamW,
    clip: Option<f32>,
    device: Device,
    seed: u64,
    head_graphs: Option<std::cell::RefCell<crate::layer_exec::GraphCache>>,
    pub step: usize,
}

impl PipelineTrainer {
    /// Connect to the stages, check agreement, and initialise each worker's
    /// training stage. The coordinator reads embedding and head from `model_dir`.
    pub fn new(model_dir: &Path, stages: &[StageSpec], opts: &PipelineOptions) -> Result<Self> {
        let mut src = CheckpointSource::open(model_dir)?;
        let tc = src.text_config().clone();
        let mut cluster = Cluster::connect(stages, Some(&tc), &opts.dist)?;
        let (mode, rank, alpha, targets) = match &opts.spec.mode {
            TrainMode::Full => ("full".to_string(), 0, 0.0, Vec::new()),
            TrainMode::Lora(l) => ("lora".into(), l.rank, l.alpha, l.target_modules.clone()),
        };
        for (s, c) in stages.iter().zip(&mut cluster.conns) {
            let init = StageInit {
                start: s.layers.start,
                end: s.layers.end,
                batch: opts.batch,
                seq: opts.seq,
                mode: mode.clone(),
                rank,
                alpha,
                targets: targets.clone(),
                freeze: opts.spec.freeze.clone(),
                only: opts.spec.only.clone(),
                lr: opts.lr,
                weight_decay: opts.weight_decay,
                seed: opts.seed,
                cache_layers: opts.cache_layers,
                resume: opts.resume.clone(),
            };
            c.call(
                op::TRAIN_INIT,
                &Enc::default().str(&serde_json::to_string(&init)?).0,
            )?;
        }
        let cfg = tc.flow_config(opts.batch, opts.seq);
        let head = src.head(&cfg)?;
        let mut head_opt = AdamW::new(opts.lr, opts.weight_decay);
        let mut head_store = Params::new();
        if let Some(r) = &opts.resume {
            let d = Path::new(r).join("head");
            head_store = crate::train_stream::load_params(&d.join("trained.safetensors"))?;
            head_opt.load(&d.join("optimizer.safetensors"))?;
        }
        Ok(Self {
            cluster,
            src,
            cfg,
            spec: opts.spec.clone(),
            head,
            head_store,
            head_values: Params::new(),
            head_grads: Params::new(),
            head_shapes: HashMap::new(),
            head_opt,
            clip: opts.clip,
            device: opts.device,
            seed: opts.seed,
            head_graphs: opts.cache_layers.then(Default::default),
            step: 0,
        })
    }

    pub fn set_lr(&mut self, lr: f32) -> Result<()> {
        self.head_opt.lr = lr;
        for c in &mut self.cluster.conns {
            c.call(op::TRAIN_LR, &Enc::default().f32(lr).0)?;
        }
        Ok(())
    }

    /// One optimizer step over `micro` micro-batches, pipelined across stages.
    pub fn train_step(&mut self, micro: &[Batch]) -> Result<StreamStep> {
        let t = std::time::Instant::now();
        ensure!(!micro.is_empty(), "no micro-batches");
        let n_stages = self.cluster.conns.len();
        let (to_coord_tx, to_coord_rx) = std::sync::mpsc::channel::<Result<StageMsg>>();
        let mut inboxes = Vec::with_capacity(n_stages);
        let mut outboxes = Vec::with_capacity(n_stages);
        for _ in 0..n_stages {
            let (tx, rx) = std::sync::mpsc::channel::<Option<StageMsg>>();
            inboxes.push(tx);
            outboxes.push(rx);
        }
        let mut embeds = Vec::with_capacity(micro.len());
        for b in micro {
            ensure!(
                b.tokens.len() == self.cfg.batch * self.cfg.seq,
                "micro-batch has {} tokens, expected {}",
                b.tokens.len(),
                self.cfg.batch * self.cfg.seq
            );
            embeds.push(match self.head_store.get("embed_tokens") {
                Some(t) => b
                    .tokens
                    .iter()
                    .flat_map(|&x| {
                        let r = x as usize * self.cfg.hidden;
                        t[r..r + self.cfg.hidden].to_vec()
                    })
                    .collect(),
                None => self.src.embed(&b.tokens, self.cfg.hidden)?,
            });
        }
        let conns = std::mem::take(&mut self.cluster.conns);
        let mut loss_sum = 0f32;
        let result = std::thread::scope(|scope| -> Result<Vec<Conn>> {
            // Stage s forwards to s+1 (or the coordinator) and backwards to s-1
            // (or the coordinator). Each thread owns its connection.
            let mut handles = Vec::new();
            for (s, (mut conn, rx)) in conns.into_iter().zip(outboxes).enumerate() {
                let next = if s + 1 < n_stages {
                    Some(inboxes[s + 1].clone())
                } else {
                    None
                };
                let prev = if s > 0 {
                    Some(inboxes[s - 1].clone())
                } else {
                    None
                };
                let coord = to_coord_tx.clone();
                handles.push(scope.spawn(move || -> Conn {
                    while let Ok(Some(msg)) = rx.recv() {
                        let r = match msg {
                            StageMsg::Fwd(mb, h, sn) => conn
                                .state_call(op::TRAIN_FWD, Enc::default().u64(mb).state(&h, &sn))
                                .map(|(h, sn)| StageMsg::Fwd(mb, h, sn)),
                            StageMsg::Bwd(mb, h, sn) => conn
                                .state_call(op::TRAIN_BWD, Enc::default().u64(mb).state(&h, &sn))
                                .map(|(h, sn)| StageMsg::Bwd(mb, h, sn)),
                        };
                        match r {
                            Ok(m @ StageMsg::Fwd(..)) => match &next {
                                Some(n) => {
                                    let _ = n.send(Some(m));
                                }
                                None => {
                                    let _ = coord.send(Ok(m));
                                }
                            },
                            Ok(m @ StageMsg::Bwd(..)) => match &prev {
                                Some(p) => {
                                    let _ = p.send(Some(m));
                                }
                                None => {
                                    let _ = coord.send(Ok(m));
                                }
                            },
                            Err(e) => {
                                let _ = coord.send(Err(e));
                            }
                        }
                    }
                    conn
                }));
            }
            drop(to_coord_tx);
            let feed = (|| -> Result<()> {
                for (mb, h) in embeds.into_iter().enumerate() {
                    inboxes[0]
                        .send(Some(StageMsg::Fwd(mb as u64, h, Vec::new())))
                        .map_err(|_| anyhow::anyhow!("first stage thread is gone"))?;
                }
                let mut done = 0;
                while done < micro.len() {
                    match to_coord_rx.recv().context("stage threads ended early")?? {
                        StageMsg::Fwd(mb, h, snaps) => {
                            // Head on the coordinator; its backward starts the
                            // micro-batch's trip back down the pipeline.
                            let b = &micro[mb as usize];
                            let spec_train = self.spec.clone();
                            let wrt = |n: &str| spec_train.is_trainable(n);
                            let ls = LayerSpec {
                                cfg: &self.cfg,
                                device: self.device,
                                lora: self.spec.lora_spec(),
                                overrides: &self.head_store,
                                seed: self.seed,
                                cache: self.head_graphs.as_ref(),
                            };
                            let hb = head_loss_backward(
                                &self.head, &h, &snaps, &b.labels, &b.mask, &wrt, &ls,
                            )?;
                            loss_sum += hb.loss;
                            for (k, g) in hb.grads {
                                crate::train_stream::accumulate(&mut self.head_grads, k, g);
                            }
                            for (k, v) in hb.values {
                                self.head_values.entry(k).or_insert(v);
                            }
                            self.head_shapes.extend(hb.shapes);
                            inboxes[n_stages - 1]
                                .send(Some(StageMsg::Bwd(mb, hb.d_h, hb.d_snaps)))
                                .map_err(|_| anyhow::anyhow!("last stage thread is gone"))?;
                        }
                        StageMsg::Bwd(mb, d_h0, _) => {
                            if self.spec.is_trainable("embed_tokens") {
                                let b = &micro[mb as usize];
                                let table = match self.head_store.get("embed_tokens") {
                                    Some(t) => t.clone(),
                                    None => self.src.embed_table()?,
                                };
                                let hid = self.cfg.hidden;
                                let mut g = vec![0f32; table.len()];
                                for (r, &tk) in b.tokens.iter().enumerate() {
                                    crate::layer_exec::add_into(
                                        &mut g[tk as usize * hid..(tk as usize + 1) * hid],
                                        &d_h0[r * hid..(r + 1) * hid],
                                    );
                                }
                                crate::train_stream::accumulate(
                                    &mut self.head_grads,
                                    "embed_tokens".into(),
                                    g,
                                );
                                self.head_values
                                    .entry("embed_tokens".into())
                                    .or_insert(table);
                                self.head_shapes
                                    .insert("embed_tokens".into(), vec![self.cfg.vocab, hid]);
                            }
                            done += 1;
                        }
                    }
                }
                Ok(())
            })();
            for tx in &inboxes {
                let _ = tx.send(None);
            }
            let conns: Vec<Conn> = handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .map_err(|_| anyhow::anyhow!("stage thread panicked"))
                })
                .collect::<Result<_>>()?;
            feed.map(|_| conns)
        });
        self.cluster.conns = result?;

        // Global-norm clip over every stage and the head, then step everywhere.
        let n = micro.len() as f32;
        let mut sq = 0f64;
        for c in &mut self.cluster.conns {
            let r = c.call(op::TRAIN_GRAD_SQ, &[])?;
            sq += Dec(&r).f64()?;
        }
        let hn = global_norm(self.head_grads.values()) as f64;
        let norm = ((sq + hn * hn).sqrt() as f32) / n;
        ensure!(norm.is_finite(), "non-finite gradient norm {norm}");
        let clip = match self.clip {
            Some(c) if norm > c => c / norm,
            _ => 1.0,
        };
        let scale = clip / n;
        for c in &mut self.cluster.conns {
            c.call(op::TRAIN_APPLY, &Enc::default().f32(scale).0)?;
        }
        let grads = std::mem::take(&mut self.head_grads);
        let mut names: Vec<String> = grads.keys().cloned().collect();
        names.sort();
        let mut grads = grads;
        for k in names {
            let mut g = grads.remove(&k).unwrap_or_default();
            g.iter_mut().for_each(|x| *x *= scale);
            if !self.head_store.contains_key(&k) {
                if let Some(v) = self.head_values.remove(&k) {
                    self.head_store.insert(k.clone(), v);
                } else {
                    continue;
                }
            }
            let p = self.head_store.get_mut(&k).expect("inserted");
            self.head_opt.step(&k, p, &g);
        }
        self.head_opt.end_step();
        self.step += 1;
        Ok(StreamStep {
            step: self.step,
            loss: loss_sum / n,
            grad_norm: norm,
            ms: t.elapsed().as_secs_f64() * 1e3,
        })
    }

    /// Save every stage's and the head's trained values + optimizer state under
    /// `dir` (resumable with [`PipelineOptions::resume`]). Worker paths are on the
    /// workers' file systems.
    pub fn save(&mut self, dir: &str) -> Result<()> {
        for c in &mut self.cluster.conns {
            c.call(op::TRAIN_SAVE, &Enc::default().str(dir).0)?;
        }
        let d = Path::new(dir).join("head");
        std::fs::create_dir_all(&d)?;
        crate::train_stream::save_params(&d.join("trained.safetensors"), &self.head_store)?;
        self.head_opt.save(&d.join("optimizer.safetensors"))
    }

    /// Every trained value in the cluster (stages + head) with its graph shape,
    /// gathered to the coordinator — what [`crate::export`] writes out.
    pub fn fetch_trained(&mut self) -> Result<(Params, HashMap<String, Vec<usize>>)> {
        let mut vals = self.head_store.clone();
        let mut shapes: HashMap<String, Vec<usize>> = self
            .head_shapes
            .iter()
            .filter(|(k, _)| vals.contains_key(*k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        for c in &mut self.cluster.conns {
            let r = c.call(op::TRAIN_FETCH, &[])?;
            let mut d = Dec(&r);
            for _ in 0..d.u32()? {
                let name = d.str()?;
                let rank = d.u32()? as usize;
                let shape = (0..rank)
                    .map(|_| d.u64().map(|x| x as usize))
                    .collect::<Result<Vec<_>>>()?;
                let v = d.f32s()?;
                shapes.insert(name.clone(), shape);
                vals.insert(name, v);
            }
        }
        Ok((vals, shapes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_round_trips() {
        let h = vec![1.0f32, 2.0, 3.0];
        let snaps = vec![vec![4.0f32, 5.0], vec![6.0f32]];
        let e = Enc::default().u64(7).state(&h, &snaps);
        let mut d = Dec(&e.0);
        assert_eq!(d.u64().unwrap(), 7);
        let (h2, s2) = d.state().unwrap();
        assert_eq!(h2, h);
        assert_eq!(s2, snaps);
        assert!(Dec(&e.0[..5]).state().is_err());
    }
}
