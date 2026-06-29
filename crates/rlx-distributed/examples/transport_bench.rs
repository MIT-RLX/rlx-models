// RLX models — distributed inference.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Transport microbenchmark — loopback TCP latency / bandwidth / collectives.
//!
//! Spins up a 2-rank mesh on `127.0.0.1` (a thread per rank, but real
//! kernel TCP sockets) and measures the cost the distribution layer adds:
//! point-to-point round-trip latency, streaming bandwidth across payload
//! sizes, and `all_reduce` latency for a hidden-state-sized vector.
//!
//! Run: `cargo run -p rlx-distributed --example transport_bench --release`
//!
//! Loopback is ~memcpy speed, so these numbers isolate framing/serialization
//! overhead; a real Ethernet/Thunderbolt link adds its own latency floor.

use rlx_distributed::{NetTransport, ProcessGroup, ReduceKind};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

fn main() {
    let world = 2u32;
    let listeners: Vec<TcpListener> = (0..world)
        .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
        .collect();
    let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();

    println!("transport: loopback TCP, world_size = {world}");
    let handles: Vec<_> = listeners
        .into_iter()
        .enumerate()
        .map(|(rank, listener)| {
            let addrs = addrs.clone();
            thread::spawn(move || {
                let t = NetTransport::from_listener(rank as u32, world, listener, addrs, 64 << 20)
                    .unwrap();
                bench(&ProcessGroup::new(Arc::new(t)));
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

fn bench(g: &ProcessGroup) {
    let rank = g.rank();
    g.barrier().unwrap();

    // ---- point-to-point round-trip latency (ping-pong) ----
    let iters = 5000;
    if rank == 0 {
        let p = [1.0f32];
        for _ in 0..200 {
            g.send_f32(1, 0, &p).unwrap();
            g.recv_f32(1, 0).unwrap();
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            g.send_f32(1, 0, &p).unwrap();
            g.recv_f32(1, 0).unwrap();
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!(
            "[latency]   ping-pong: {us:6.2} µs RTT   ({:.2} µs one-way)   [{iters} iters]",
            us / 2.0
        );
    } else {
        for _ in 0..(200 + iters) {
            let x = g.recv_f32(0, 0).unwrap();
            g.send_f32(0, 0, &x).unwrap();
        }
    }
    g.barrier().unwrap();

    // ---- streaming bandwidth across payload sizes ----
    for &elems in &[256usize, 16_384, 262_144, 4_194_304] {
        let bytes = elems * 4;
        let reps = (512 * 1024 * 1024 / bytes).clamp(8, 200_000);
        if rank == 0 {
            let buf = vec![0.5f32; elems];
            g.send_f32(1, 1, &[0.0]).unwrap(); // sync start
            let t0 = Instant::now();
            for _ in 0..reps {
                g.send_f32(1, 2, &buf).unwrap();
            }
            g.recv_f32(1, 3).unwrap(); // ack: all received
            let secs = t0.elapsed().as_secs_f64();
            let gbps = (reps * bytes) as f64 / secs / 1e9;
            println!(
                "[bandwidth] {:>9} B × {:>6} = {:>5.1} GB  ->  {gbps:6.2} GB/s",
                bytes,
                reps,
                (reps * bytes) as f64 / 1e9
            );
        } else {
            g.recv_f32(0, 1).unwrap();
            for _ in 0..reps {
                g.recv_f32(0, 2).unwrap();
            }
            g.send_f32(0, 3, &[1.0]).unwrap();
        }
        g.barrier().unwrap();
    }

    // ---- collective latency for a hidden-state-sized vector ----
    // e.g. batch1 × seq1 × d_model=4096 — the kind of all_reduce a
    // tensor-parallel layer issues, or a broadcast the pipeline issues.
    for &elems in &[4096usize, 4096 * 8] {
        let mut data = vec![1.0f32; elems];
        let iters = 2000;
        for _ in 0..50 {
            g.all_reduce(&mut data, ReduceKind::Sum).unwrap();
        }
        g.barrier().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            g.all_reduce(&mut data, ReduceKind::Sum).unwrap();
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        if rank == 0 {
            println!("[collective] all_reduce {:>6} f32: {us:6.2} µs/op", elems);
        }
        g.barrier().unwrap();
    }

    g.barrier().unwrap();
}
