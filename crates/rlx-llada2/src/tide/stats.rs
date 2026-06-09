// RLX — versatile ML compiler + runtime.
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

// RLX — TIDE offload_stats aggregation.

use rlx_runtime::{ExpertPool, ExpertPoolStats, MoeResidencyStats};

/// Cumulative counters aligned with TIDE `LLaDA2MoeSparseMoeBlock.offload_stats`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TideOffloadStats {
    pub cpu_tokens: u64,
    pub gpu_tokens: u64,
    pub cpu_calls: u64,
    pub gpu_calls: u64,
    pub cpu_compute_time: u64,
    pub gpu_compute_time: u64,
    pub cpu_tokens_move_time: u64,
    pub gpu_tokens_move_time: u64,
    pub experts_move_time: u64,
    pub promotions: u64,
    pub demotions: u64,
}

impl TideOffloadStats {
    pub fn merge(&mut self, other: &Self) {
        self.cpu_tokens += other.cpu_tokens;
        self.gpu_tokens += other.gpu_tokens;
        self.cpu_calls += other.cpu_calls;
        self.gpu_calls += other.gpu_calls;
        self.cpu_compute_time += other.cpu_compute_time;
        self.gpu_compute_time += other.gpu_compute_time;
        self.cpu_tokens_move_time += other.cpu_tokens_move_time;
        self.gpu_tokens_move_time += other.gpu_tokens_move_time;
        self.experts_move_time += other.experts_move_time;
        self.promotions += other.promotions;
        self.demotions += other.demotions;
    }

    /// Map to TIDE `get_offload_stats()` key names.
    pub fn as_tide_dict(&self) -> std::collections::HashMap<&'static str, u64> {
        let mut m = std::collections::HashMap::new();
        m.insert("cpu_tokens", self.cpu_tokens);
        m.insert("gpu_tokens", self.gpu_tokens);
        m.insert("cpu_calls", self.cpu_calls);
        m.insert("gpu_calls", self.gpu_calls);
        m.insert("cpu_compute_time", self.cpu_compute_time);
        m.insert("gpu_compute_time", self.gpu_compute_time);
        m.insert("cpu_tokens_move_time", self.cpu_tokens_move_time);
        m.insert("gpu_tokens_move_time", self.gpu_tokens_move_time);
        m.insert("experts_move_time", self.experts_move_time);
        m.insert("promotions", self.promotions);
        m.insert("demotions", self.demotions);
        m
    }
}

pub fn pool_stats_to_tide(stats: &ExpertPoolStats) -> TideOffloadStats {
    TideOffloadStats {
        promotions: stats.promotions,
        demotions: stats.demotions,
        ..Default::default()
    }
}

pub fn residency_stats_to_tide(stats: &MoeResidencyStats) -> TideOffloadStats {
    TideOffloadStats {
        cpu_tokens: stats.cpu_tokens,
        gpu_tokens: stats.gpu_tokens,
        cpu_calls: stats.cpu_expert_calls,
        gpu_calls: stats.gpu_expert_calls,
        ..Default::default()
    }
}

/// Sum per-layer pool stats + optional CPU residency accounting from last forward.
pub fn aggregate_offload_stats(
    pools: &[ExpertPool],
    residency: Option<&MoeResidencyStats>,
) -> TideOffloadStats {
    let mut out = TideOffloadStats::default();
    for pool in pools {
        out.merge(&pool_stats_to_tide(pool.stats()));
    }
    if let Some(r) = residency {
        out.merge(&residency_stats_to_tide(r));
    }
    out
}
