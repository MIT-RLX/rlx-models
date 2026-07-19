// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! `--nnodes N` self-spawning launcher for data-parallel training — one
//! command, no hostfile, no N terminals.
//!
//! A training binary has two roles. Launched with `--nnodes N` and no `RANK`
//! in its environment it is the **launcher**: it reserves `N` free loopback
//! ports and re-spawns itself once per rank as a separate OS process with
//! `RANK` / `WORLD` / `PEERS` set — exactly the env [`crate::from_env`] reads.
//! Each spawned copy sees `RANK` set, so it takes the **worker** path and
//! trains. The same shape scales to a real cluster: point `PEERS` at real
//! hosts (or set `DISCOVER=1`) and drop `--nnodes`.
//!
//! ```no_run
//! use rlx_tune::cluster::{launch_or_join, Role};
//!
//! fn main() -> anyhow::Result<()> {
//!     match launch_or_join()? {
//!         Role::Launcher => Ok(()), // parent: workers spawned + awaited
//!         Role::Worker { rank, world, comm } => {
//!             if rank == 0 {
//!                 eprintln!("training on {world} rank(s)");
//!             }
//!             // rlx_tune::train_dp(graph, &wrt, &mut params, &inputs,
//!             //     steps, comm.as_deref(), &cfg, |m| { /* log */ })?;
//!             let _ = comm;
//!             Ok(())
//!         }
//!     }
//! }
//! ```

use crate::distributed::{GradComm, from_env};
use anyhow::{Context, Result};
use std::net::{Ipv4Addr, TcpListener};
use std::process::Command;

/// The role this process plays after [`launch_or_join`].
pub enum Role {
    /// This process was the launcher: it spawned the workers and waited for
    /// them to exit. The caller should return / exit — the real work happened
    /// in the child processes.
    Launcher,
    /// This process is a training worker. `comm` is `None` for a single-rank
    /// (`world == 1`) run and `Some` otherwise; pass `comm.as_deref()` to
    /// [`crate::train_dp`].
    Worker {
        rank: u32,
        world: u32,
        comm: Option<Box<dyn GradComm>>,
    },
}

/// Parse `--nnodes N` from an argument list (typically `std::env::args()`).
/// Returns `None` if the flag is absent or its value doesn't parse.
pub fn parse_nnodes<I>(args: I) -> Option<u32>
where
    I: IntoIterator<Item = String>,
{
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--nnodes" => return it.next().and_then(|v| v.parse().ok()),
            _ => {
                if let Some(v) = a.strip_prefix("--nnodes=") {
                    return v.parse().ok();
                }
            }
        }
    }
    None
}

/// Reserve `n` free loopback TCP ports by binding ephemeral sockets and
/// releasing them. There is an inherent small race — the OS may hand a freed
/// port to another process before the worker rebinds it — but for local
/// launches it is negligible (this is what torchrun / mlx-lm do for
/// single-host runs).
pub fn free_loopback_ports(n: u32) -> Result<Vec<u16>> {
    (0..n)
        .map(|_| {
            let l = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
                .context("binding an ephemeral loopback port")?;
            Ok(l.local_addr().context("reading local_addr")?.port())
        })
        .collect()
}

/// Build the `PEERS` value (`127.0.0.1:p0,127.0.0.1:p1,…`) from a port list.
pub fn loopback_peers(ports: &[u16]) -> String {
    ports
        .iter()
        .map(|p| format!("127.0.0.1:{p}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// One-call cluster bring-up.
///
/// - **Worker** (this process has `RANK` set — either spawned by the launcher
///   or launched under an external launcher like torchrun): returns
///   [`Role::Worker`] with a [`from_env`] collective.
/// - **Launcher** (`--nnodes N` with `N > 1` and `RANK` unset): reserves ports,
///   spawns `N` copies of this executable with `RANK`/`WORLD`/`PEERS` set,
///   waits for all to exit (erroring if any fails), and returns
///   [`Role::Launcher`].
/// - **Single process** (no `--nnodes` / `--nnodes 1`, `RANK` unset): returns
///   [`Role::Worker`] with `world == 1` and `comm == None`.
pub fn launch_or_join() -> Result<Role> {
    // Already a rank (spawned worker, or an external launcher set the env)?
    if std::env::var("RANK").is_ok() {
        let world: u32 = std::env::var("WORLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let rank: u32 = std::env::var("RANK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let comm = from_env().context("joining the data-parallel group")?;
        return Ok(Role::Worker { rank, world, comm });
    }

    match parse_nnodes(std::env::args()) {
        Some(n) if n > 1 => {
            spawn_workers(n)?;
            Ok(Role::Launcher)
        }
        // No `--nnodes` (or `--nnodes 1`): plain single-process run.
        _ => Ok(Role::Worker {
            rank: 0,
            world: 1,
            comm: from_env().context("single-process bring-up")?,
        }),
    }
}

/// Spawn `world` copies of the current executable, one per rank, each with
/// `RANK`/`WORLD`/`PEERS` set to a fresh loopback mesh; wait for all.
fn spawn_workers(world: u32) -> Result<()> {
    let exe = std::env::current_exe().context("resolving current_exe")?;
    let ports = free_loopback_ports(world)?;
    let peers = loopback_peers(&ports);
    // Forward this process's own args (minus the program name) so workers get
    // the same flags; they ignore `--nnodes` because `RANK` is set.
    let fwd: Vec<String> = std::env::args().skip(1).collect();

    eprintln!("rlx-tune: launching {world} local ranks (loopback mesh {peers})");

    let mut children = Vec::with_capacity(world as usize);
    for rank in 0..world {
        let child = Command::new(&exe)
            .args(&fwd)
            .env("RANK", rank.to_string())
            .env("WORLD", world.to_string())
            .env("PEERS", &peers)
            .env("TOPOLOGY", "mesh")
            .spawn()
            .with_context(|| format!("spawning worker rank {rank}"))?;
        children.push((rank, child));
    }

    let mut failures = Vec::new();
    for (rank, mut child) in children {
        match child.wait() {
            Ok(status) if status.success() => {}
            Ok(status) => failures.push(format!("rank {rank} exited with {status}")),
            Err(e) => failures.push(format!("rank {rank} wait() failed: {e}")),
        }
    }
    if !failures.is_empty() {
        anyhow::bail!("worker failure: {}", failures.join("; "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nnodes_spaced_and_equals() {
        let spaced = ["prog", "--device", "cpu", "--nnodes", "4"].map(String::from);
        assert_eq!(parse_nnodes(spaced), Some(4));
        let eq = ["prog", "--nnodes=3"].map(String::from);
        assert_eq!(parse_nnodes(eq), Some(3));
        let none = ["prog", "--device", "cpu"].map(String::from);
        assert_eq!(parse_nnodes(none), None);
        let bad = ["prog", "--nnodes", "abc"].map(String::from);
        assert_eq!(parse_nnodes(bad), None);
    }

    #[test]
    fn allocates_distinct_free_ports() {
        let ports = free_loopback_ports(4).unwrap();
        assert_eq!(ports.len(), 4);
        let mut sorted = ports.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 4, "ports should be distinct: {ports:?}");
        assert!(ports.iter().all(|&p| p != 0));
    }

    #[test]
    fn peers_string_is_loopback_csv() {
        assert_eq!(
            loopback_peers(&[29500, 29501, 29502]),
            "127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502"
        );
    }
}
