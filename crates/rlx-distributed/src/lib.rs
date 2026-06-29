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

//! Multi-node distributed inference for RLX models.
//!
//! Layered on top of `rlx-driver`'s transports (`TcpTransport`,
//! `ThunderboltTransport`, and `rlx-mlx`'s `MlxTransport`) and
//! `ProcessGroup`:
//!
//!   - [`config`] — `hosts.json` parsing + process-group construction
//!     ([`DistConfig::connect`]).
//!   - [`partition`] — pipeline-parallel layer assignment
//!     ([`pipeline_layer_range`], [`block_role`]).
//!   - [`pipeline`] — the model-agnostic relay ([`PipelineCoordinator`])
//!     driving per-rank [`BlockRunner`]s.
//!
//! A model family provides a [`BlockRunner`] for its layer block (e.g.
//! `rlx_qwen3`'s `Qwen3PipelineStage`); everything else here is reusable.
//!
//! ```no_run
//! use rlx_distributed::{DistConfig, ParallelMode, PipelineCoordinator};
//!
//! # fn demo(rank: u32, mut runner: impl rlx_distributed::BlockRunner) -> anyhow::Result<()> {
//! let cfg = DistConfig::load("hosts.json", Some(rank), ParallelMode::Pipeline)?;
//! let group = cfg.connect()?;                 // blocks until the mesh forms
//! let coord = PipelineCoordinator::new(group);
//!
//! let mut tokens = vec![/* prompt ids */];
//! for _ in 0..32 {
//!     let tok = coord.forward_step(&mut runner, &tokens, |logits| argmax(logits))?;
//!     tokens.push(tok);
//! }
//! coord.barrier()?;                           // before dropping the group
//! # Ok(()) }
//! # fn argmax(_l: &[f32]) -> u32 { 0 }
//! ```

pub mod config;
pub mod launch;
pub mod partition;
pub mod pipeline;

pub use config::{DistConfig, Hostfile, ParallelMode, TransportBackend};
pub use launch::{LocalCluster, WorkerArgs, free_loopback_ports, worker_args};
pub use partition::{BlockRole, block_role, pipeline_layer_range};
pub use pipeline::{BlockInput, BlockOutput, BlockRunner, PipelineCoordinator};

// Re-export the transport primitives so model crates depend only on
// `rlx-distributed`, not `rlx-driver` directly.
pub use rlx_driver::{
    NetTransport, ProcessGroup, ReduceKind, TcpTransport, ThunderboltTransport, Transport,
};
