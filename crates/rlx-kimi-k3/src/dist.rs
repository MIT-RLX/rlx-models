// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Distributed **layer-pipeline** for the streaming Kimi-K3 forward: each node
//! owns a contiguous layer range (+ its checkpoint shards on FAST local NVMe) and
//! runs [`crate::runner::run_layer_range_streaming`] on its slice; the boundary
//! state (hidden + all AttnRes snapshots) is passed node→node over TCP. The
//! coordinator embeds (first node), relays through the workers, and applies the
//! head (last node). The whole 114 GB backbone is thus read CONCURRENTLY from the
//! nodes' fast NVMes instead of one slow disk — the point of the cluster.
//!
//! Wire format for the pipeline state (little-endian): `[n_vecs u32]` then, per
//! vec, `[len u32][len × f32]`. The first vec is the hidden state, the rest are
//! the snapshots.

use crate::config::KimiLinearConfig;
use crate::flow::FlowConfig;
use crate::loader::CheckpointLoader;
use crate::runner::{
    DecodeState, HeadCache, LayerCache, apply_head_cached, argmax, decode_forward_range,
    run_layer_range_streaming,
};
use anyhow::{Context, Result};
use rlx_runtime::Device;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

const EMB: &str = "language_model.model.embed_tokens.weight";

fn write_vecs(w: &mut impl Write, vecs: &[&[f32]]) -> Result<()> {
    w.write_all(&(vecs.len() as u32).to_le_bytes())?;
    for v in vecs {
        w.write_all(&(v.len() as u32).to_le_bytes())?;
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        w.write_all(&bytes)?;
    }
    w.flush()?;
    Ok(())
}

fn read_vecs(r: &mut impl Read) -> Result<Vec<Vec<f32>>> {
    let mut u = [0u8; 4];
    r.read_exact(&mut u)?;
    let n = u32::from_le_bytes(u) as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        r.read_exact(&mut u)?;
        let len = u32::from_le_bytes(u) as usize;
        let mut buf = vec![0u8; len * 4];
        r.read_exact(&mut buf)?;
        out.push(
            buf.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        );
    }
    Ok(out)
}

/// Pack `(hidden, snapshots)` for the wire (hidden first).
fn pack(h: &[f32], snaps: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut v = vec![h.to_vec()];
    v.extend(snaps.iter().cloned());
    v
}

/// Run a worker: bind `addr`, and for each connection read `[start u32][end u32]`
/// plus the incoming state, run that layer range from the LOCAL checkpoint, and send
/// the outgoing state back. `n_requests == 0` serves forever.
pub fn serve_worker(
    addr: &str,
    model_dir: &str,
    tc: &KimiLinearConfig,
    make_cfg: impl Fn(usize) -> FlowConfig,
    device: Device,
    n_requests: usize,
) -> Result<()> {
    let listener = TcpListener::bind(addr).with_context(|| format!("bind {addr}"))?;
    let mut ck = CheckpointLoader::open(model_dir)?;
    eprintln!("worker serving on {addr} (checkpoint {model_dir})");
    let mut served = 0;
    for conn in listener.incoming() {
        let mut s = conn?;
        let mut u = [0u8; 4];
        s.read_exact(&mut u)?;
        let start = u32::from_le_bytes(u) as usize;
        s.read_exact(&mut u)?;
        let end = u32::from_le_bytes(u) as usize;
        let mut vecs = read_vecs(&mut s)?;
        let snaps = vecs.split_off(1);
        let h = vecs.into_iter().next().context("no hidden")?;
        let seq = h.len() / tc.hidden_size.max(1);
        let cfg = make_cfg(seq); // graphs are seq-shaped; match the incoming state
        eprintln!(
            "  worker: layers {start}..{end}, seq {seq} ({} snapshots in)",
            snaps.len()
        );
        let (h_out, snaps_out) =
            run_layer_range_streaming(&mut ck, tc, &cfg, h, snaps, start, end, device)?;
        let packed = pack(&h_out, &snaps_out);
        let refs: Vec<&[f32]> = packed.iter().map(|v| v.as_slice()).collect();
        write_vecs(&mut s, &refs)?;
        served += 1;
        if n_requests != 0 && served >= n_requests {
            break;
        }
    }
    Ok(())
}

/// Send a layer range + state to a worker and get the resulting state back.
fn call_worker(
    addr: &str,
    start: usize,
    end: usize,
    h: &[f32],
    snaps: &[Vec<f32>],
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let mut s = TcpStream::connect(addr).with_context(|| format!("connect {addr}"))?;
    s.write_all(&(start as u32).to_le_bytes())?;
    s.write_all(&(end as u32).to_le_bytes())?;
    let packed = pack(h, snaps);
    let refs: Vec<&[f32]> = packed.iter().map(|v| v.as_slice()).collect();
    write_vecs(&mut s, &refs)?;
    let mut vecs = read_vecs(&mut s)?;
    let snaps_out = vecs.split_off(1);
    let h_out = vecs.into_iter().next().context("no hidden back")?;
    Ok((h_out, snaps_out))
}

/// Coordinator: embed the prompt (local checkpoint), relay the boundary state
/// through the `stages` (`(worker_addr, start, end)`, in pipeline order), then
/// return the final `(hidden, snapshots)`. The head is applied by the caller
/// ([`crate::runner::run_prefix_logits`]-style). Each worker reads its layer
/// range from its own FAST local NVMe → the backbone is read concurrently.
pub fn run_distributed_prefix(
    ck: &mut CheckpointLoader,
    cfg: &FlowConfig,
    tokens: &[u32],
    stages: &[(String, usize, usize)],
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let mut h = ck.gather_embed(
        "language_model.model.embed_tokens.weight",
        tokens,
        cfg.hidden,
    )?;
    let mut snaps: Vec<Vec<f32>> = Vec::new();
    for (addr, start, end) in stages {
        let (ho, so) = call_worker(addr, *start, *end, &h, &snaps)?;
        h = ho;
        snaps = so;
    }
    Ok((h, snaps))
}

// ── stateful distributed DECODE (O(1)/token, weights resident) ──────────────
//
// The prefill relay above re-streams a node's range per request. For generation
// we instead keep each node's cross-token KDA conv/scan + MLA KV **resident** and
// send it only the new token(s): O(1)/token, and the backbone stays hot on each
// node's fast NVMe. Per step the coordinator relays the same `(hidden, snaps)`
// boundary through the stages (AttnRes snapshots are per-token → empty into the
// first node, accumulate stage→stage) and applies the head. Opcode byte before
// the `[start][end]` header: `0` = reset state (new sequence, i.e. the prefill),
// `1` = continue.

/// Send one decode step to a resident worker: `op` (0 reset / 1 continue), the
/// node's layer range, and the incoming `(hidden, snaps)`; get the outgoing state.
fn call_decode_worker(
    addr: &str,
    op: u8,
    start: usize,
    end: usize,
    h: &[f32],
    snaps: &[Vec<f32>],
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let mut s = TcpStream::connect(addr).with_context(|| format!("connect {addr}"))?;
    s.write_all(&[op])?;
    s.write_all(&(start as u32).to_le_bytes())?;
    s.write_all(&(end as u32).to_le_bytes())?;
    let packed = pack(h, snaps);
    let refs: Vec<&[f32]> = packed.iter().map(|v| v.as_slice()).collect();
    write_vecs(&mut s, &refs)?;
    let mut vecs = read_vecs(&mut s)?;
    let snaps_out = vecs.split_off(1);
    let h_out = vecs.into_iter().next().context("no hidden back")?;
    Ok((h_out, snaps_out))
}

/// Relay `(h, snaps)` through the stages for one decode step (`op` on every hop).
fn relay_step(
    stages: &[(String, usize, usize)],
    op: u8,
    mut h: Vec<f32>,
    mut snaps: Vec<Vec<f32>>,
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    for (addr, start, end) in stages {
        let (ho, so) = call_decode_worker(addr, op, *start, *end, &h, &snaps)?;
        h = ho;
        snaps = so;
    }
    Ok((h, snaps))
}

/// Stateful decode worker: hold this node's layer-range [`DecodeState`] (+ open
/// checkpoint) RESIDENT across tokens and run [`decode_forward_range`] on each
/// incoming step. Opcode `0` zeroes the state first (new sequence). Serves
/// `n_requests` steps then returns (`0` = forever). The peer connection is
/// per-request but the state lives in this scope, so it persists across steps.
pub fn serve_decode_worker(
    addr: &str,
    model_dir: &str,
    tc: &KimiLinearConfig,
    make_cfg: impl Fn(usize) -> FlowConfig,
    device: Device,
    n_requests: usize,
) -> Result<()> {
    let listener = TcpListener::bind(addr).with_context(|| format!("bind {addr}"))?;
    let mut ck = CheckpointLoader::open(model_dir)?;
    let cfg1 = make_cfg(1);
    let mut state = DecodeState::zeros(tc, &cfg1);
    // this node's backbone stays RESIDENT across tokens (the point of the cluster):
    // load each of its layers once, reuse every step. Sized to fit the node's RAM.
    let mut cache = LayerCache::from_env();
    eprintln!("decode worker serving on {addr} (checkpoint {model_dir})");
    let mut served = 0;
    for conn in listener.incoming() {
        let mut s = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  decode worker: accept error: {e}");
                continue;
            }
        };
        // Read the WHOLE request before touching state — a bad/empty connection
        // (e.g. a health-check / port probe) must not kill the worker or corrupt
        // the resident decode state; just log it and wait for the next peer.
        let req = (|| -> Result<(u8, usize, usize, Vec<f32>, Vec<Vec<f32>>)> {
            let mut op = [0u8; 1];
            s.read_exact(&mut op)?;
            let mut u = [0u8; 4];
            s.read_exact(&mut u)?;
            let start = u32::from_le_bytes(u) as usize;
            s.read_exact(&mut u)?;
            let end = u32::from_le_bytes(u) as usize;
            let mut vecs = read_vecs(&mut s)?;
            let snaps_in = vecs.split_off(1);
            let h = vecs.into_iter().next().context("no hidden")?;
            Ok((op[0], start, end, h, snaps_in))
        })();
        let (op, start, end, h, snaps_in) = match req {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  decode worker: skip connection ({e})");
                continue;
            }
        };
        let seq = h.len() / tc.hidden_size.max(1);
        if op == 0 {
            state = DecodeState::zeros(tc, &cfg1); // new sequence
        }
        let cfg = make_cfg(seq);
        eprintln!(
            "  decode worker: layers {start}..{end}, seq {seq}, s_past {} (op {op})",
            state.s_past
        );
        let (h_out, snaps_out) = decode_forward_range(
            &mut ck, tc, &cfg, h, snaps_in, &mut state, start, end, &mut cache, device,
        )?;
        let packed = pack(&h_out, &snaps_out);
        let refs: Vec<&[f32]> = packed.iter().map(|v| v.as_slice()).collect();
        write_vecs(&mut s, &refs)?;
        served += 1;
        if n_requests != 0 && served >= n_requests {
            break;
        }
    }
    Ok(())
}

/// Distributed **generation**: prefill the prompt across the resident-state
/// workers (opcode `0`), take the head on the last position for the first token,
/// then generate `n_gen-1` more tokens — each an O(1) relay step (opcode `1`).
/// Output is IDENTICAL to single-node [`crate::runner::run_generate`]; the
/// backbone stays resident on each node so only the head/embed touch the
/// coordinator's disk. Returns the generated token ids.
#[allow(clippy::too_many_arguments)]
pub fn run_distributed_generate(
    ck: &mut CheckpointLoader,
    make_cfg: impl Fn(usize) -> FlowConfig,
    prompt: &[u32],
    n_gen: usize,
    stages: &[(String, usize, usize)],
    device: Device,
) -> Result<Vec<u32>> {
    let cfg1 = make_cfg(1);
    let hidden = cfg1.hidden;
    // resident head on the coordinator: the 4.7 GB lm_head is loaded/lowered once.
    let mut head = HeadCache::from_env();

    // ── prefill (opcode 0 resets each worker's state) ──
    let h0 = ck.gather_embed(EMB, prompt, hidden)?;
    let (h, snaps) = relay_step(stages, 0, h0, Vec::new())?;
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

    // ── decode the rest, one O(1) relay step at a time ──
    for _ in 1..n_gen {
        let hin = ck.gather_embed(EMB, &[tok], hidden)?;
        let (h, snaps) = relay_step(stages, 1, hin, Vec::new())?;
        tok = argmax(&apply_head_cached(
            ck, &cfg1, &h, &snaps, &mut head, device,
        )?);
        out.push(tok);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_roundtrips() {
        let h = vec![1.0f32, 2.0, 3.0];
        let snaps = vec![vec![4.0f32, 5.0], vec![6.0f32]];
        let packed = pack(&h, &snaps);
        let refs: Vec<&[f32]> = packed.iter().map(|v| v.as_slice()).collect();
        let mut buf = Vec::new();
        write_vecs(&mut buf, &refs).unwrap();
        let mut got = read_vecs(&mut &buf[..]).unwrap();
        let snaps2 = got.split_off(1);
        assert_eq!(got[0], h);
        assert_eq!(snaps2, snaps);
    }
}
