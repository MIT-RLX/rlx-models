// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! KDA state log (feature `state-log`): every KDA recurrent state of a greedy
//! generation, recoverable exactly, for a fraction of the storage.
//!
//! A KDA layer's state is its recurrent `scan` matrix (`heads × head_dim ×
//! head_dim`) plus three short-conv windows. Storing it per token costs ~465 MB
//! (f32) on the real model. But the next scan state is a pure function of the
//! previous one and four per-token inputs — the recurrence's `k`, `v`, `decay`
//! and `beta` — and the conv windows are just the last `conv_kernel - 1` raw
//! q/k/v projections. So the log keeps:
//!
//! * `kda_inputs.bin` — one fixed-size record per consumed token (token `t` at
//!   offset `t · record_bytes`): for every KDA layer in index order, `k`, `v`,
//!   `decay` (`heads·head_dim` each), `beta` (`heads`), then — with `conv` —
//!   `q_raw`, `k_raw`, `v_raw` (`heads·head_dim` each). ~10 MB/token on the
//!   real model, ~20 MB with conv windows.
//! * `snapshots/state_<p>.bin` — the full state after `p` tokens, every
//!   `snapshot_every` tokens and after any multi-token step (a chunked
//!   prefill): per KDA layer `conv_q`, `conv_k`, `conv_v`, `scan`.
//! * `manifest.json` — dims, record layout, snapshot positions, step sizes, the
//!   tokens, device, config fingerprint and the env flags that change numerics.
//!
//! [`StateLog::state_at`] rebuilds the state after any `p` tokens by stepping the
//! same `gated_delta_net_carry_pc` op from the nearest snapshot at or before `p`.
//! Rebuilt on the device that generated the log, one token per step as the
//! decode path ran them, it is bit-identical to the state the generation held
//! ([`StateLog::verify`] checks that against the snapshots). States inside a
//! chunked prefill never existed in the original run; rebuilding them steps one
//! token at a time — use `stepwise_prefill` when those must be the run's own.
//!
//! Capture is free when unused: the KDA decode step always names its inputs as
//! taps ([`crate::kda::DECODE_STATE_TAPS`]); only while a logger is active does
//! the decode graph expose them as outputs.

use crate::config::KimiLinearConfig;
use crate::flow::FlowConfig;
use crate::loader::CheckpointLoader;
use crate::runner::{
    AttnState, DecodeState, HeadCache, LayerCache, apply_head_cached, argmax, decode_forward_range,
};
use anyhow::{Context, Result, bail, ensure};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{SyncSender, sync_channel};

const FORMAT: u32 = 1;
const INPUTS: &str = "kda_inputs.bin";
const MANIFEST: &str = "manifest.json";

thread_local! {
    /// While `Some`, the decode path pushes each KDA layer's state-log outputs.
    static CAPTURE: RefCell<Option<Vec<(usize, Vec<Vec<f32>>)>>> = const { RefCell::new(None) };
}

/// Is a logged generation capturing on this thread?
pub fn capture_active() -> bool {
    CAPTURE.with(|c| c.borrow().is_some())
}

/// Called by the decode path with layer `i`'s outputs in `DECODE_STATE_TAPS` order.
pub(crate) fn push_capture(layer: usize, outputs: Vec<Vec<f32>>) {
    CAPTURE.with(|c| {
        if let Some(v) = c.borrow_mut().as_mut() {
            v.push((layer, outputs));
        }
    });
}

struct CaptureScope;

impl CaptureScope {
    fn begin() -> Self {
        CAPTURE.with(|c| *c.borrow_mut() = Some(Vec::new()));
        Self
    }
    fn take(&self) -> Vec<(usize, Vec<Vec<f32>>)> {
        CAPTURE.with(|c| {
            c.borrow_mut()
                .as_mut()
                .map(std::mem::take)
                .unwrap_or_default()
        })
    }
}

impl Drop for CaptureScope {
    fn drop(&mut self) {
        CAPTURE.with(|c| *c.borrow_mut() = None);
    }
}

/// Everything needed to read a log back.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format: u32,
    /// KDA layers in record order.
    pub kda_layers: Vec<usize>,
    pub heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    /// Records carry the raw projections (conv windows rebuildable everywhere).
    pub conv: bool,
    pub record_floats: usize,
    pub snapshot_every: usize,
    /// Positions (tokens consumed) with a full snapshot, ascending. Position 0
    /// (the zero state) is implicit.
    pub snapshots: Vec<usize>,
    /// Tokens consumed by each decode step, in order (a chunked prefill is one
    /// step of `prompt_len`).
    pub steps: Vec<usize>,
    /// Tokens consumed so far (= records written).
    pub positions: usize,
    /// Prompt then generated tokens.
    pub tokens: Vec<u32>,
    pub prompt_len: usize,
    pub device: String,
    /// [`crate::dist::config_fingerprint`] of the model.
    pub config: String,
    /// Env flags that change numerics, as they were set.
    pub env: BTreeMap<String, String>,
}

impl Manifest {
    fn proj(&self) -> usize {
        self.heads * self.head_dim
    }
    fn layer_floats(&self) -> usize {
        3 * self.proj() + self.heads + if self.conv { 3 * self.proj() } else { 0 }
    }
    fn snapshot_layer_floats(&self) -> usize {
        3 * (self.conv_kernel - 1) * self.proj() + self.heads * self.head_dim * self.head_dim
    }
    /// Bytes per token in `kda_inputs.bin`.
    pub fn record_bytes(&self) -> usize {
        self.record_floats * 4
    }
    /// Bytes per snapshot.
    pub fn snapshot_bytes(&self) -> usize {
        self.kda_layers.len() * self.snapshot_layer_floats() * 4
    }
}

/// How a logged generation runs.
#[derive(Debug, Clone)]
pub struct StateLogOptions {
    pub dir: PathBuf,
    /// Full snapshot every this many tokens (≥ 1).
    pub snapshot_every: usize,
    /// Also log raw q/k/v projections so conv windows rebuild at every position.
    pub conv: bool,
    /// Prefill one token per step, so every prompt-position state is the run's own.
    pub stepwise_prefill: bool,
}

enum Msg {
    Record(Vec<u8>),
    Snapshot(usize, Vec<u8>),
    Manifest(Vec<u8>),
}

/// Background writer + bookkeeping for one log.
struct Writer {
    tx: Option<SyncSender<Msg>>,
    handle: Option<std::thread::JoinHandle<Result<()>>>,
    m: Manifest,
}

fn f32_bytes(v: &[f32], out: &mut Vec<u8>) {
    out.reserve(v.len() * 4);
    for x in v {
        out.extend(x.to_le_bytes());
    }
}

const NUMERIC_ENV: &[&str] = &[
    "RLX_KIMI_BATCHED_MOE",
    "RLX_KIMI_HOST_ROUTER",
    "RLX_KIMI_BF16_BACKBONE",
    "RLX_KIMI_INT8_BACKBONE",
    "RLX_KIMI_QUANT",
    "RLX_KIMI_ACT_QUANT",
    "RLX_KIMI_PACKED_MOE",
    "RLX_KIMI_NO_PACKED_MOE",
    "RLX_MLA_VDIM",
    "RLX_KDA_CHUNK",
    "RLX_KDA_CONV_OLD",
    "RLX_NUMBER_FORMAT",
];

impl Writer {
    fn create(dir: &Path, m: Manifest) -> Result<Self> {
        std::fs::create_dir_all(dir.join("snapshots"))?;
        let mut inputs = BufWriter::with_capacity(8 << 20, File::create(dir.join(INPUTS))?);
        let dir = dir.to_path_buf();
        // A few tokens in flight: disk writes overlap the next tokens' compute.
        let (tx, rx) = sync_channel::<Msg>(4);
        let handle = std::thread::spawn(move || -> Result<()> {
            for msg in rx {
                match msg {
                    Msg::Record(b) => inputs.write_all(&b)?,
                    Msg::Snapshot(p, b) => {
                        let path = dir.join("snapshots").join(format!("state_{p:010}.bin"));
                        let mut f = File::create(&path)?;
                        f.write_all(&b)?;
                        f.sync_all()?;
                        inputs.flush()?;
                        inputs.get_ref().sync_data()?;
                    }
                    Msg::Manifest(b) => {
                        let tmp = dir.join("manifest.json.tmp");
                        std::fs::write(&tmp, &b)?;
                        std::fs::rename(&tmp, dir.join(MANIFEST))?;
                    }
                }
            }
            inputs.flush()?;
            inputs.get_ref().sync_all()?;
            Ok(())
        });
        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
            m,
        })
    }

    fn send(&mut self, msg: Msg) -> Result<()> {
        if self.tx.as_ref().map(|t| t.send(msg)).transpose().is_err() || self.tx.is_none() {
            return Err(self
                .join()
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("state-log writer stopped")));
        }
        Ok(())
    }

    fn join(&mut self) -> Result<()> {
        self.tx = None;
        match self.handle.take() {
            Some(h) => h
                .join()
                .map_err(|_| anyhow::anyhow!("state-log writer panicked"))?,
            None => Ok(()),
        }
    }

    fn write_manifest(&mut self) -> Result<()> {
        let b = serde_json::to_vec_pretty(&self.m)?;
        self.send(Msg::Manifest(b))
    }

    /// Will the step after `seq` more tokens write a snapshot?
    fn snapshot_due(&self, seq: usize) -> bool {
        (self.m.positions + seq).is_multiple_of(self.m.snapshot_every) || seq > 1
    }

    /// One decode step of `seq` tokens: its captures and the state after it.
    fn step(
        &mut self,
        seq: usize,
        state: &DecodeState,
        captures: Vec<(usize, Vec<Vec<f32>>)>,
    ) -> Result<()> {
        let layers: Vec<usize> = captures.iter().map(|(l, _)| *l).collect();
        ensure!(
            layers == self.m.kda_layers,
            "captured KDA layers {layers:?}, expected {:?}",
            self.m.kda_layers
        );
        let (proj, h) = (self.m.proj(), self.m.heads);
        let widths = [proj, proj, proj, h, proj, proj, proj];
        let used = if self.m.conv { 7 } else { 4 };
        for (layer, outs) in &captures {
            ensure!(
                outs.len() == 7,
                "layer {layer}: {} state-log outputs",
                outs.len()
            );
            for (j, w) in widths.iter().enumerate() {
                ensure!(
                    outs[j].len() == seq * w,
                    "layer {layer} output {j}: {} values for {seq} tokens",
                    outs[j].len()
                );
            }
        }
        for t in 0..seq {
            let mut rec = Vec::with_capacity(self.m.record_bytes());
            for (_, outs) in &captures {
                for (j, w) in widths.iter().enumerate().take(used) {
                    f32_bytes(&outs[j][t * w..(t + 1) * w], &mut rec);
                }
            }
            self.send(Msg::Record(rec))?;
        }
        self.m.positions += seq;
        self.m.steps.push(seq);
        if self.m.positions.is_multiple_of(self.m.snapshot_every) || seq > 1 {
            let mut b = Vec::with_capacity(self.m.snapshot_bytes());
            for &l in &self.m.kda_layers {
                let AttnState::Kda(ks) = &state.layers[l] else {
                    bail!("layer {l} carries no KDA state");
                };
                for v in [&ks.conv_q, &ks.conv_k, &ks.conv_v, &ks.scan] {
                    f32_bytes(v, &mut b);
                }
            }
            let p = self.m.positions;
            self.send(Msg::Snapshot(p, b))?;
            self.m.snapshots.push(p);
            self.write_manifest()?;
        }
        Ok(())
    }
}

/// Greedy generation (as [`crate::runner::run_generate`]) that writes a KDA state
/// log to `opts.dir`. Returns the generated tokens.
#[allow(clippy::too_many_arguments)]
pub fn generate_logged(
    ck: &mut CheckpointLoader,
    tc: &KimiLinearConfig,
    make_cfg: impl Fn(usize) -> FlowConfig,
    prompt: &[u32],
    n_gen: usize,
    n_layers: usize,
    device: Device,
    opts: &StateLogOptions,
) -> Result<Vec<u32>> {
    ensure!(
        !prompt.is_empty() && n_gen > 0,
        "need a prompt and at least one token"
    );
    ensure!(
        opts.snapshot_every >= 1,
        "snapshot_every must be at least 1"
    );
    let cfg1 = make_cfg(1);
    let hidden = cfg1.hidden;
    let kda = cfg1.kda;
    let kda_layers: Vec<usize> = (0..n_layers).filter(|&i| tc.is_kda_layer(i)).collect();
    let mut m = Manifest {
        format: FORMAT,
        kda_layers: kda_layers.clone(),
        heads: kda.num_heads,
        head_dim: kda.head_dim,
        conv_kernel: kda.conv_kernel,
        conv: opts.conv,
        record_floats: 0,
        snapshot_every: opts.snapshot_every,
        snapshots: Vec::new(),
        steps: Vec::new(),
        positions: 0,
        tokens: prompt.to_vec(),
        prompt_len: prompt.len(),
        device: format!("{device:?}"),
        config: crate::dist::config_fingerprint(tc),
        env: NUMERIC_ENV
            .iter()
            .filter_map(|k| rlx_ir::env::read(k).ok().map(|v| (k.to_string(), v)))
            .chain([(
                "dense_residency".to_string(),
                format!("{:?}", crate::common::dense_residency()),
            )])
            .collect(),
    };
    m.record_floats = kda_layers.len() * m.layer_floats();
    let mut w = Writer::create(&opts.dir, m)?;
    w.write_manifest()?;
    let scope = CaptureScope::begin();
    let mut state = DecodeState::zeros(tc, &cfg1);
    // Fresh caches: a graph compiled before capture began has no log outputs.
    let mut cache = LayerCache::from_env();
    let mut head = HeadCache::from_env();
    let emb = crate::source::EMBED;

    let run = |ck: &mut CheckpointLoader,
               cache: &mut LayerCache,
               state: &mut DecodeState,
               w: &mut Writer,
               toks: &[u32]|
     -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let cfg = make_cfg(toks.len());
        let h0 = ck.gather_embed(emb, toks, hidden)?;
        let out = decode_forward_range(
            ck,
            tc,
            &cfg,
            h0,
            Vec::new(),
            state,
            0,
            n_layers,
            cache,
            device,
        )?;
        if w.snapshot_due(toks.len()) {
            // Resident KDA state lives in the cached graphs; the snapshot reads
            // the host copy.
            cache.sync_kda_state(state)?;
        }
        w.step(toks.len(), state, scope.take())?;
        Ok(out)
    };

    let (h_last, snaps_last) = if opts.stepwise_prefill {
        let mut last = (Vec::new(), Vec::new());
        for &t in prompt {
            last = run(ck, &mut cache, &mut state, &mut w, &[t])?;
        }
        last
    } else {
        let (h, snaps) = run(ck, &mut cache, &mut state, &mut w, prompt)?;
        let p = prompt.len() - 1;
        (
            h[p * hidden..(p + 1) * hidden].to_vec(),
            snaps
                .iter()
                .map(|s| s[p * hidden..(p + 1) * hidden].to_vec())
                .collect(),
        )
    };
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
        let (h, snaps) = run(ck, &mut cache, &mut state, &mut w, &[tok])?;
        tok = argmax(&apply_head_cached(
            ck, &cfg1, &h, &snaps, &mut head, device,
        )?);
        out.push(tok);
    }
    drop(scope);
    w.m.tokens.extend(&out);
    w.write_manifest()?;
    w.join()?;
    Ok(out)
}

/// One KDA layer's full decode state.
#[derive(Debug, Clone, PartialEq)]
pub struct KdaLayerState {
    pub layer: usize,
    /// `[conv_kernel - 1, heads·head_dim]` each; empty when the log has no conv
    /// projections and the position is not a snapshot.
    pub conv_q: Vec<f32>,
    pub conv_k: Vec<f32>,
    pub conv_v: Vec<f32>,
    /// `[heads, head_dim, head_dim]`.
    pub scan: Vec<f32>,
}

/// A state log opened for reading.
pub struct StateLog {
    pub dir: PathBuf,
    pub manifest: Manifest,
    inputs: File,
}

/// Result of [`StateLog::verify`].
#[derive(Debug, Clone)]
pub struct VerifyReport {
    /// `(from, to)` snapshot pairs rebuilt.
    pub spans: Vec<(usize, usize)>,
    /// Largest absolute difference seen between a rebuilt and a stored state.
    pub max_abs_diff: f32,
    /// Every rebuilt state equal to its snapshot bit for bit.
    pub exact: bool,
}

/// The KDA recurrence for one token, compiled once and reused for every layer
/// and step: `(k, v, decay, beta, state) → state'` (`q` only shapes the output,
/// so it is fed zeros).
struct Stepper {
    c: CompiledGraph,
    zeros_q: Vec<f32>,
}

impl Stepper {
    fn new(m: &Manifest, device: Device) -> Result<Self> {
        let (h, hd) = (m.heads, m.head_dim);
        let mut hir = HirModule::new("kda_state_step");
        let mut g = HirMut::new(&mut hir);
        let bshd = Shape::new(&[1, 1, h, hd], DType::F32);
        let q = g.input("q", bshd.clone());
        let k = g.input("k", bshd.clone());
        let v = g.input("v", bshd.clone());
        let decay = g.input("decay", bshd.clone());
        let beta = g.input("beta", Shape::new(&[1, 1, h], DType::F32));
        let st = g.input("state", Shape::new(&[1, h, hd, hd], DType::F32));
        // Same op and operand order as the decode step; the state is updated in
        // place and read back as an output, exactly as the decode graph does.
        let y = g.gated_delta_net_carry_pc(q, k, v, decay, beta, st, hd, bshd);
        g.set_outputs(vec![y, st]);
        let built = rlx_core::flow_util::built_from_hir(hir, Default::default())?;
        Ok(Self {
            c: rlx_core::flow_util::compile_built(built, device)?,
            zeros_q: vec![0.0; h * hd],
        })
    }

    fn step(
        &mut self,
        k: &[f32],
        v: &[f32],
        decay: &[f32],
        beta: &[f32],
        state: &[f32],
    ) -> Result<Vec<f32>> {
        let mut out = self.c.run(&[
            ("q", self.zeros_q.as_slice()),
            ("k", k),
            ("v", v),
            ("decay", decay),
            ("beta", beta),
            ("state", state),
        ]);
        ensure!(out.len() == 2, "state step produced {} outputs", out.len());
        Ok(out.remove(1))
    }
}

fn read_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

impl StateLog {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let manifest: Manifest = serde_json::from_slice(
            &std::fs::read(dir.join(MANIFEST))
                .with_context(|| format!("reading {dir:?}/{MANIFEST}"))?,
        )?;
        ensure!(
            manifest.format == FORMAT,
            "state log format {} (expected {FORMAT})",
            manifest.format
        );
        let inputs = File::open(dir.join(INPUTS))?;
        let have = inputs.metadata()?.len() as usize / manifest.record_bytes().max(1);
        ensure!(
            have >= manifest.positions,
            "{INPUTS} holds {have} records, the manifest says {}",
            manifest.positions
        );
        Ok(Self {
            dir,
            manifest,
            inputs,
        })
    }

    /// Token `t`'s record (`record_floats` values).
    pub fn record(&self, t: usize) -> Result<Vec<f32>> {
        ensure!(
            t < self.manifest.positions,
            "no record {t} (log has {})",
            self.manifest.positions
        );
        let n = self.manifest.record_bytes();
        let mut f = &self.inputs;
        f.seek(SeekFrom::Start((t * n) as u64))?;
        let mut b = vec![0u8; n];
        f.read_exact(&mut b)?;
        Ok(read_f32s(&b))
    }

    /// The stored snapshot after `p` tokens (`p` must be a snapshot position or 0).
    pub fn snapshot(&self, p: usize) -> Result<Vec<KdaLayerState>> {
        let m = &self.manifest;
        let (proj, cw) = (m.proj(), (m.conv_kernel - 1) * m.proj());
        if p == 0 {
            return Ok(m
                .kda_layers
                .iter()
                .map(|&layer| KdaLayerState {
                    layer,
                    conv_q: vec![0.0; cw],
                    conv_k: vec![0.0; cw],
                    conv_v: vec![0.0; cw],
                    scan: vec![0.0; m.heads * m.head_dim * m.head_dim],
                })
                .collect());
        }
        ensure!(m.snapshots.contains(&p), "no snapshot at position {p}");
        let v = read_f32s(&std::fs::read(
            self.dir
                .join("snapshots")
                .join(format!("state_{p:010}.bin")),
        )?);
        ensure!(
            v.len() * 4 == m.snapshot_bytes(),
            "snapshot {p} is truncated"
        );
        let per = m.snapshot_layer_floats();
        let scan_n = m.heads * m.head_dim * m.head_dim;
        let _ = proj;
        Ok(m.kda_layers
            .iter()
            .enumerate()
            .map(|(li, &layer)| {
                let s = &v[li * per..(li + 1) * per];
                KdaLayerState {
                    layer,
                    conv_q: s[..cw].to_vec(),
                    conv_k: s[cw..2 * cw].to_vec(),
                    conv_v: s[2 * cw..3 * cw].to_vec(),
                    scan: s[3 * cw..3 * cw + scan_n].to_vec(),
                }
            })
            .collect())
    }

    /// The nearest snapshot position at or before `p` (0 = the zero state).
    pub fn base_for(&self, p: usize) -> usize {
        self.manifest
            .snapshots
            .iter()
            .copied()
            .filter(|&s| s <= p)
            .max()
            .unwrap_or(0)
    }

    /// The full KDA state after `p` tokens, rebuilt from the nearest snapshot.
    pub fn state_at(&self, p: usize, device: Device) -> Result<Vec<KdaLayerState>> {
        self.state_from(self.base_for(p), p, device)
    }

    /// The state after `p` tokens, rebuilt from the snapshot at `base` (≤ `p`).
    pub fn state_from(&self, base: usize, p: usize, device: Device) -> Result<Vec<KdaLayerState>> {
        let m = &self.manifest;
        ensure!(
            p <= m.positions,
            "position {p} is past the log ({} tokens)",
            m.positions
        );
        ensure!(base <= p, "base {base} after position {p}");
        let mut st = self.snapshot(base)?;
        if base == p {
            return Ok(st);
        }
        let (proj, h) = (m.proj(), m.heads);
        let lf = m.layer_floats();
        let mut stepper = Stepper::new(m, device)?;
        for t in base..p {
            let rec = self.record(t)?;
            for (li, ls) in st.iter_mut().enumerate() {
                let r = &rec[li * lf..(li + 1) * lf];
                let (k, v, d, b) = (
                    &r[..proj],
                    &r[proj..2 * proj],
                    &r[2 * proj..3 * proj],
                    &r[3 * proj..3 * proj + h],
                );
                ls.scan = stepper.step(k, v, d, b, &ls.scan)?;
            }
        }
        // Conv windows: the last `conv_kernel - 1` raw projections before `p`.
        let kw = m.conv_kernel - 1;
        for ls in st.iter_mut() {
            ls.conv_q.clear();
            ls.conv_k.clear();
            ls.conv_v.clear();
        }
        if m.conv {
            let off = 3 * proj + h;
            for j in 0..kw {
                let pos = p as isize - kw as isize + j as isize;
                let rec = if pos >= 0 {
                    Some(self.record(pos as usize)?)
                } else {
                    None
                };
                for (li, ls) in st.iter_mut().enumerate() {
                    for (which, dst) in [&mut ls.conv_q, &mut ls.conv_k, &mut ls.conv_v]
                        .into_iter()
                        .enumerate()
                    {
                        match &rec {
                            Some(r) => {
                                let s = li * lf + off + which * proj;
                                dst.extend_from_slice(&r[s..s + proj]);
                            }
                            None => dst.extend(std::iter::repeat_n(0.0, proj)),
                        }
                    }
                }
            }
        }
        Ok(st)
    }

    /// Rebuild every snapshot from the one before it and compare with what was
    /// stored. Snapshots that close a multi-token step (a chunked prefill) are
    /// used as bases but not checked: the run never held the states inside it.
    pub fn verify(&self, device: Device) -> Result<VerifyReport> {
        let m = &self.manifest;
        let mut chunk_ends = std::collections::HashSet::new();
        let mut pos = 0;
        for &s in &m.steps {
            pos += s;
            if s > 1 {
                chunk_ends.insert(pos);
            }
        }
        let mut prev = 0;
        let mut rep = VerifyReport {
            spans: Vec::new(),
            max_abs_diff: 0.0,
            exact: true,
        };
        for &s in &m.snapshots {
            if !chunk_ends.contains(&s) {
                let got = self.state_from(prev, s, device)?;
                let want = self.snapshot(s)?;
                for (a, b) in got.iter().zip(&want) {
                    let mut pairs = vec![(&a.scan, &b.scan)];
                    if m.conv {
                        pairs.extend([
                            (&a.conv_q, &b.conv_q),
                            (&a.conv_k, &b.conv_k),
                            (&a.conv_v, &b.conv_v),
                        ]);
                    }
                    for (x, y) in pairs {
                        ensure!(x.len() == y.len(), "layer {}: state sizes differ", a.layer);
                        for (u, w) in x.iter().zip(y) {
                            let d = (u - w).abs();
                            rep.max_abs_diff = rep.max_abs_diff.max(d);
                            if u.to_bits() != w.to_bits() {
                                rep.exact = false;
                            }
                        }
                    }
                }
                rep.spans.push((prev, s));
            }
            prev = s;
        }
        Ok(rep)
    }
}
