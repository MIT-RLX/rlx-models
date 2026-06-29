// RLX models — distributed inference.
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

//! Local multi-process launcher — run an N-rank cluster on one host over
//! loopback TCP, with **no hand-written `hosts.json` and no N terminals**.
//!
//! A distributed binary has two roles. When launched with `--rank R` it is a
//! **worker**: it forms the process group and runs its block. With no
//! `--rank`, it is the **launcher**: it picks free loopback ports, generates
//! a hostfile, and re-spawns itself once per rank as a separate OS process.
//! The same shape scales to a real cluster — only the hostfile IPs change.
//!
//! ```no_run
//! use rlx_distributed::launch::{worker_args, LocalCluster};
//!
//! # fn run_worker(_rank: u32, _hostfile: &str) {}
//! fn main() -> anyhow::Result<()> {
//!     match worker_args() {
//!         Some(w) => run_worker(w.rank, &w.hostfile),  // spawned worker
//!         None => {                                     // top-level launcher
//!             let out = LocalCluster::new(3).arg("--device").arg("cpu").run()?;
//!             for line in out {
//!                 println!("{line}");
//!             }
//!         }
//!     }
//!     Ok(())
//! }
//! ```

use crate::config::{Hostfile, TransportBackend};
use anyhow::{Context, Result};
use std::io::{BufRead, BufReader};
use std::net::{Ipv4Addr, TcpListener};
use std::process::{Command, Stdio};

/// Reserve `n` free loopback TCP ports by binding ephemeral sockets and
/// immediately releasing them. There is an inherent (small) race: the OS
/// may hand a freed port to another process before the worker rebinds it.
/// For local launches this is negligible and matches what mlx-lm / torchrun
/// do for single-host runs.
pub fn free_loopback_ports(n: u32) -> Result<Vec<u16>> {
    (0..n)
        .map(|_| {
            let l = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
                .context("binding an ephemeral loopback port")?;
            // The listener drops at the end of this closure, freeing the port
            // for the worker process to bind.
            Ok(l.local_addr().context("reading local_addr")?.port())
        })
        .collect()
}

/// The launch-level arguments every spawned worker receives. Model-specific
/// flags (e.g. `--device`) stay in argv for the binary to parse itself.
#[derive(Debug, Clone)]
pub struct WorkerArgs {
    pub rank: u32,
    pub hostfile: String,
}

/// Parse this process's worker invocation: `--rank N --hostfile PATH` (rank
/// falls back to `RLX_RANK`). Returns `None` when neither `--rank` nor
/// `RLX_RANK` is present — meaning this is the top-level launcher, which
/// should build a [`LocalCluster`].
pub fn worker_args() -> Option<WorkerArgs> {
    let mut rank: Option<u32> = None;
    let mut hostfile: Option<String> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--rank" => rank = it.next().and_then(|v| v.parse().ok()),
            "--hostfile" => hostfile = it.next(),
            _ => {}
        }
    }
    let rank = rank.or_else(|| std::env::var("RLX_RANK").ok().and_then(|v| v.parse().ok()))?;
    Some(WorkerArgs {
        rank,
        hostfile: hostfile.unwrap_or_default(),
    })
}

/// A self-spawning local cluster: re-runs the current executable once per
/// rank as a separate OS process, each pointed at a generated loopback
/// hostfile, capturing one rank's stdout.
///
/// This exercises the real `DistConfig::load` / `connect` path over real TCP
/// sockets in separate address spaces — the genuine deployment shape minus
/// the physical wire.
pub struct LocalCluster {
    world: u32,
    backend: TransportBackend,
    extra_args: Vec<String>,
    capture_rank: u32,
    quiet: bool,
}

impl LocalCluster {
    /// A `world`-rank cluster over loopback TCP, capturing rank 0's stdout.
    pub fn new(world: u32) -> Self {
        Self {
            world,
            backend: TransportBackend::Tcp,
            extra_args: Vec::new(),
            capture_rank: 0,
            quiet: false,
        }
    }

    /// Override the transport backend written into the hostfile.
    pub fn backend(mut self, backend: TransportBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Forward an extra CLI argument to every spawned worker (e.g.
    /// `--device mlx`). Workers also receive `--rank i --hostfile <path>`.
    pub fn arg(mut self, a: impl Into<String>) -> Self {
        self.extra_args.push(a.into());
        self
    }

    /// Forward several extra arguments at once.
    pub fn args<I, S>(mut self, it: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.extra_args.extend(it.into_iter().map(Into::into));
        self
    }

    /// Which rank's stdout to capture and return from [`run`](Self::run).
    pub fn capture_rank(mut self, rank: u32) -> Self {
        self.capture_rank = rank;
        self
    }

    /// Suppress the one-line "spawning N ranks" banner on stderr.
    pub fn quiet(mut self, quiet: bool) -> Self {
        self.quiet = quiet;
        self
    }

    /// Spawn `world` workers, block until all exit, and return the captured
    /// rank's stdout line by line. Errors if any worker exits non-zero.
    pub fn run(self) -> Result<Vec<String>> {
        let exe = std::env::current_exe().context("resolving current_exe")?;
        let ports = free_loopback_ports(self.world)?;
        let hostfile = Hostfile::loopback(&ports, self.backend).write_temp("cluster")?;

        if !self.quiet {
            eprintln!(
                "LocalCluster: spawning {} ranks ({} backend), hostfile {}",
                self.world,
                self.backend.as_str(),
                hostfile.display()
            );
        }

        let mut children = Vec::with_capacity(self.world as usize);
        for rank in 0..self.world {
            let capture = rank == self.capture_rank;
            let child = Command::new(&exe)
                .arg("--rank")
                .arg(rank.to_string())
                .arg("--hostfile")
                .arg(&hostfile)
                .args(&self.extra_args)
                .stdout(if capture {
                    Stdio::piped()
                } else {
                    Stdio::null()
                })
                .stderr(Stdio::inherit())
                .spawn()
                .with_context(|| format!("spawning worker rank {rank}"))?;
            children.push((rank, child));
        }

        // Drain the captured rank's stdout to EOF (it closes when that worker
        // exits); the other ranks run concurrently meanwhile.
        let mut lines = Vec::new();
        if let Some((_, child)) = children.iter_mut().find(|(r, _)| *r == self.capture_rank) {
            if let Some(out) = child.stdout.take() {
                lines.extend(BufReader::new(out).lines().map_while(Result::ok));
            }
        }

        let mut failures = Vec::new();
        for (rank, mut child) in children {
            match child.wait() {
                Ok(status) if status.success() => {}
                Ok(status) => failures.push(format!("rank {rank} exited with {status}")),
                Err(e) => failures.push(format!("rank {rank} wait() failed: {e}")),
            }
        }
        let _ = std::fs::remove_file(&hostfile);

        if !failures.is_empty() {
            anyhow::bail!("LocalCluster worker failure: {}", failures.join("; "));
        }
        Ok(lines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_distinct_free_ports() {
        let ports = free_loopback_ports(4).unwrap();
        assert_eq!(ports.len(), 4);
        // Distinct and non-zero.
        let mut sorted = ports.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 4, "ports should be distinct: {ports:?}");
        assert!(ports.iter().all(|&p| p != 0));
    }

    #[test]
    fn worker_args_none_without_flag() {
        // The test harness runs with no `--rank` and no RLX_RANK.
        assert!(std::env::var("RLX_RANK").is_err());
        assert!(worker_args().is_none());
    }

    #[test]
    fn builder_collects_args() {
        let c = LocalCluster::new(2)
            .backend(TransportBackend::Thunderbolt)
            .arg("--device")
            .arg("mlx")
            .args(["--decode", "--max-tokens", "8"]);
        assert_eq!(c.world, 2);
        assert_eq!(c.backend, TransportBackend::Thunderbolt);
        assert_eq!(
            c.extra_args,
            ["--device", "mlx", "--decode", "--max-tokens", "8"]
        );
    }
}
