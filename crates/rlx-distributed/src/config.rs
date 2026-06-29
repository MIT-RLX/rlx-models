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

//! Launch configuration: hostfile parsing + process-group construction.
//!
//! A run is described by a `hosts.json` (mirroring mlx-lm's) plus this
//! process's rank:
//!
//! ```json
//! {
//!   "backend": "tcp",
//!   "hosts": ["10.0.0.1:9000", "10.0.0.2:9000", "10.0.0.3:9000"]
//! }
//! ```
//!
//! `hosts[r]` is rank `r`'s `ip:port` listen address. Every rank reads the
//! same file and is told its own rank (CLI flag or env `RLX_RANK`).

use anyhow::{Context, Result, bail};
use rlx_driver::{DEFAULT_HEAP_BYTES, ProcessGroup, TcpTransport, ThunderboltTransport, Transport};
use serde::{Deserialize, Serialize};
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Which parallelism strategy the run uses. Pipeline splits the model by
/// layer; tensor splits within each layer. (Only pipeline is wired into
/// the coordinator today; tensor parallel is future work.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelMode {
    Pipeline,
    Tensor,
}

/// Network backend for the process group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportBackend {
    /// Portable TCP (Ethernet or the Thunderbolt Bridge IP link).
    Tcp,
    /// TCP pinned to the Thunderbolt interface (one-sided heap exposed).
    Thunderbolt,
    /// MLX's distributed backend (jaccl/ring). Construct `MlxTransport`
    /// from `rlx-mlx` directly and launch with `mlx.launch`; not built
    /// here because it needs the MLX runtime.
    MlxJaccl,
}

impl TransportBackend {
    fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "tcp" | "ring" => Ok(Self::Tcp),
            "thunderbolt" | "tb" => Ok(Self::Thunderbolt),
            "mlx" | "jaccl" | "mlx-jaccl" => Ok(Self::MlxJaccl),
            other => bail!("unknown transport backend {other:?} (tcp|thunderbolt|mlx-jaccl)"),
        }
    }

    /// Canonical lowercase name — the inverse of [`parse`](Self::parse),
    /// suitable for writing back into a `hosts.json`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tcp => "tcp",
            Self::Thunderbolt => "thunderbolt",
            Self::MlxJaccl => "mlx-jaccl",
        }
    }
}

/// Parsed `hosts.json`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hostfile {
    /// Backend name; defaults to "tcp" when absent.
    #[serde(default = "default_backend")]
    pub backend: String,
    /// `ip:port` per rank, in rank order.
    pub hosts: Vec<String>,
}

fn default_backend() -> String {
    "tcp".to_string()
}

impl Hostfile {
    pub fn from_json_str(s: &str) -> Result<Self> {
        let hf: Hostfile = serde_json::from_str(s).context("parsing hosts.json")?;
        if hf.hosts.is_empty() {
            bail!("hosts.json has an empty `hosts` list");
        }
        Ok(hf)
    }

    /// Build a loopback hostfile: rank `r` listens on `127.0.0.1:ports[r]`.
    /// This is the local-run analogue of hand-writing a `hosts.json` — used
    /// by [`crate::launch::LocalCluster`].
    pub fn loopback(ports: &[u16], backend: TransportBackend) -> Self {
        Self {
            backend: backend.as_str().to_string(),
            hosts: ports.iter().map(|p| format!("127.0.0.1:{p}")).collect(),
        }
    }

    /// Serialize back to the `hosts.json` wire form.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("Hostfile serializes")
    }

    /// Write this hostfile to a uniquely-named file in the temp dir and
    /// return its path. `tag` distinguishes concurrent runs.
    pub fn write_temp(&self, tag: &str) -> Result<PathBuf> {
        let path =
            std::env::temp_dir().join(format!("rlx_hosts_{tag}_{}.json", std::process::id()));
        std::fs::write(&path, self.to_json())
            .with_context(|| format!("writing hostfile {}", path.display()))?;
        Ok(path)
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading hostfile {}", path.display()))?;
        Self::from_json_str(&text)
    }

    pub fn world_size(&self) -> u32 {
        self.hosts.len() as u32
    }

    pub fn backend(&self) -> Result<TransportBackend> {
        TransportBackend::parse(&self.backend)
    }

    /// Resolve every `hosts[r]` entry to a single `SocketAddr`, in rank
    /// order.
    pub fn peers(&self) -> Result<Vec<SocketAddr>> {
        self.hosts
            .iter()
            .enumerate()
            .map(|(r, h)| {
                h.to_socket_addrs()
                    .with_context(|| format!("resolving host[{r}] = {h:?}"))?
                    .next()
                    .with_context(|| format!("host[{r}] = {h:?} resolved to no address"))
            })
            .collect()
    }
}

/// A fully-resolved launch configuration for one process.
#[derive(Debug, Clone)]
pub struct DistConfig {
    pub rank: u32,
    pub world_size: u32,
    pub mode: ParallelMode,
    pub backend: TransportBackend,
    pub peers: Vec<SocketAddr>,
}

impl DistConfig {
    /// Build a config from a hostfile + this process's rank.
    pub fn from_hostfile(hostfile: &Hostfile, rank: u32, mode: ParallelMode) -> Result<Self> {
        let world_size = hostfile.world_size();
        if rank >= world_size {
            bail!("rank {rank} >= world_size {world_size}");
        }
        Ok(Self {
            rank,
            world_size,
            mode,
            backend: hostfile.backend()?,
            peers: hostfile.peers()?,
        })
    }

    /// Load `hosts.json` and build the config in one step. `rank` falls
    /// back to the `RLX_RANK` env var when `None`.
    pub fn load(
        hostfile_path: impl AsRef<Path>,
        rank: Option<u32>,
        mode: ParallelMode,
    ) -> Result<Self> {
        let hostfile = Hostfile::from_path(hostfile_path)?;
        let rank = match rank {
            Some(r) => r,
            None => std::env::var("RLX_RANK")
                .context("rank not given and RLX_RANK unset")?
                .parse()
                .context("RLX_RANK is not a u32")?,
        };
        Self::from_hostfile(&hostfile, rank, mode)
    }

    /// Establish the network mesh and return the process group. Blocks
    /// until every peer has connected.
    pub fn connect(&self) -> Result<ProcessGroup> {
        let transport: Arc<dyn Transport> = match self.backend {
            TransportBackend::Tcp => Arc::new(
                TcpTransport::bind(
                    self.rank,
                    self.world_size,
                    self.peers.clone(),
                    DEFAULT_HEAP_BYTES,
                )
                .context("binding TCP transport")?,
            ),
            TransportBackend::Thunderbolt => Arc::new(
                ThunderboltTransport::bind(
                    self.rank,
                    self.world_size,
                    self.peers.clone(),
                    DEFAULT_HEAP_BYTES,
                )
                .context("binding Thunderbolt transport")?,
            ),
            TransportBackend::MlxJaccl => {
                bail!(
                    "mlx-jaccl backend is provided by rlx_mlx::MlxTransport \
                     (launch with mlx.launch); construct it directly and pass \
                     to PipelineCoordinator::new"
                )
            }
        };
        Ok(ProcessGroup::new(transport))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_hostfile() {
        let hf = Hostfile::from_json_str(r#"{ "hosts": ["127.0.0.1:9000", "127.0.0.1:9001"] }"#)
            .unwrap();
        assert_eq!(hf.world_size(), 2);
        assert_eq!(hf.backend().unwrap(), TransportBackend::Tcp); // default
        let peers = hf.peers().unwrap();
        assert_eq!(peers.len(), 2);
        assert_eq!(peers[1].port(), 9001);
    }

    #[test]
    fn parses_backend_field() {
        let hf = Hostfile::from_json_str(
            r#"{ "backend": "thunderbolt", "hosts": ["10.0.0.1:5000", "10.0.0.2:5000"] }"#,
        )
        .unwrap();
        assert_eq!(hf.backend().unwrap(), TransportBackend::Thunderbolt);
    }

    #[test]
    fn empty_hosts_rejected() {
        assert!(Hostfile::from_json_str(r#"{ "hosts": [] }"#).is_err());
    }

    #[test]
    fn rank_out_of_range_rejected() {
        let hf = Hostfile::from_json_str(r#"{ "hosts": ["127.0.0.1:9000"] }"#).unwrap();
        assert!(DistConfig::from_hostfile(&hf, 3, ParallelMode::Pipeline).is_err());
    }

    #[test]
    fn loopback_roundtrips_through_json() {
        let hf = Hostfile::loopback(&[9000, 9001, 9002], TransportBackend::Tcp);
        assert_eq!(hf.world_size(), 3);
        assert_eq!(hf.hosts[2], "127.0.0.1:9002");
        // to_json → from_json_str is a faithful round-trip.
        let back = Hostfile::from_json_str(&hf.to_json()).unwrap();
        assert_eq!(back.hosts, hf.hosts);
        assert_eq!(back.backend().unwrap(), TransportBackend::Tcp);
    }

    #[test]
    fn backend_as_str_inverts_parse() {
        for b in [
            TransportBackend::Tcp,
            TransportBackend::Thunderbolt,
            TransportBackend::MlxJaccl,
        ] {
            assert_eq!(TransportBackend::parse(b.as_str()).unwrap(), b);
        }
    }

    #[test]
    fn config_from_hostfile_resolves_peers() {
        let hf = Hostfile::from_json_str(
            r#"{ "backend": "tcp", "hosts": ["127.0.0.1:9000", "127.0.0.1:9001"] }"#,
        )
        .unwrap();
        let cfg = DistConfig::from_hostfile(&hf, 1, ParallelMode::Pipeline).unwrap();
        assert_eq!(cfg.rank, 1);
        assert_eq!(cfg.world_size, 2);
        assert_eq!(cfg.backend, TransportBackend::Tcp);
        assert_eq!(cfg.peers.len(), 2);
    }
}
