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

//! In-process seq cache with shared [`CompiledGraph`] handles (no clone per infer).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use rlx_runtime::CompiledGraph;

/// One token-length bucket: optional split graphs for optimized infer.
#[derive(Clone)]
pub struct CachedSeqGraphs {
    pub full: Arc<Mutex<CompiledGraph>>,
    pub duration_refine: Option<Arc<Mutex<CompiledGraph>>>,
    pub waveform_only: Option<Arc<Mutex<CompiledGraph>>>,
    /// True when `duration_refine` was compiled for CPU (safe to share via the
    /// process-wide duration parity cache). GPU-computed durations must not be
    /// reused by other backends.
    pub duration_on_cpu: bool,
}

impl CachedSeqGraphs {
    pub fn full(full: CompiledGraph) -> Self {
        Self {
            full: Arc::new(Mutex::new(full)),
            duration_refine: None,
            waveform_only: None,
            duration_on_cpu: false,
        }
    }

    pub fn with_split(
        full: CompiledGraph,
        duration_refine: CompiledGraph,
        waveform_only: CompiledGraph,
    ) -> Self {
        Self {
            full: Arc::new(Mutex::new(full)),
            duration_refine: Some(Arc::new(Mutex::new(duration_refine))),
            waveform_only: Some(Arc::new(Mutex::new(waveform_only))),
            duration_on_cpu: false,
        }
    }

    /// Lock the primary infer graph (waveform-only slice in production split mode).
    pub fn lock_infer_graph(&self) -> std::sync::MutexGuard<'_, CompiledGraph> {
        if crate::compile_profile::production_waveform_only_infer() {
            if let Some(w) = &self.waveform_only {
                return w.lock().expect("waveform graph");
            }
        }
        self.full.lock().expect("full graph")
    }
}

pub struct SeqGraphCache {
    entries: Mutex<HashMap<usize, CachedSeqGraphs>>,
    order: Mutex<Vec<usize>>,
    capacity: usize,
}

impl SeqGraphCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            order: Mutex::new(Vec::new()),
            capacity: capacity.max(1),
        }
    }

    pub fn get(&self, seq: usize) -> Option<CachedSeqGraphs> {
        let entries = self.entries.lock().expect("seq graph cache");
        entries.get(&seq).map(clone_entry)
    }

    pub fn insert(&self, seq: usize, graphs: CachedSeqGraphs) {
        let mut entries = self.entries.lock().expect("seq graph cache");
        let mut order = self.order.lock().expect("seq graph cache order");
        if entries.len() >= self.capacity && !entries.contains_key(&seq) {
            if let Some(evict) = order.first().copied() {
                entries.remove(&evict);
                order.retain(|&k| k != evict);
            }
        }
        entries.insert(seq, graphs);
        order.retain(|&k| k != seq);
        order.push(seq);
    }

    pub fn prewarm<F>(&self, buckets: &[usize], mut build: F) -> Result<()>
    where
        F: FnMut(usize) -> Result<CachedSeqGraphs>,
    {
        for &seq in buckets {
            if self.get(seq).is_some() {
                continue;
            }
            self.insert(seq, build(seq)?);
        }
        Ok(())
    }
}

fn clone_entry(entry: &CachedSeqGraphs) -> CachedSeqGraphs {
    CachedSeqGraphs {
        full: Arc::clone(&entry.full),
        duration_refine: entry.duration_refine.as_ref().map(Arc::clone),
        waveform_only: entry.waveform_only.as_ref().map(Arc::clone),
        duration_on_cpu: entry.duration_on_cpu,
    }
}
