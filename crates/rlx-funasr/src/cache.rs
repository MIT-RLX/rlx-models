// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! A tiny LRU cache of compiled graphs keyed by sequence length.
//!
//! Each model rebuilds its graph for a specific input length; without caching,
//! every call re-clones the (potentially ~1 GB) weights, re-lowers the HIR, and
//! re-compiles the kernels. On a cache hit none of that happens — the resident
//! [`CompiledGraph`] (with its parameters already attached) is simply re-run.
//! This is the dominant speedup for fixed-chunk / streaming / repeated calls.

use std::cell::RefCell;

use anyhow::Result;
use rlx_core::flow_util::compile_built;
use rlx_flow::BuiltModel;
use rlx_runtime::{CompiledGraph, Device};

/// LRU cache of `(key → CompiledGraph)`.
pub struct GraphCache {
    entries: RefCell<Vec<(u64, CompiledGraph)>>,
    cap: usize,
}

impl GraphCache {
    /// Create an empty cache holding up to `cap` compiled graphs.
    pub fn new(cap: usize) -> Self {
        Self {
            entries: RefCell::new(Vec::new()),
            cap: cap.max(1),
        }
    }

    /// Run the cached graph for `key`, building+compiling it (via `build`) and
    /// attaching its params on the first miss. `inputs` feeds the run.
    pub fn run(
        &self,
        key: u64,
        device: Device,
        build: impl FnOnce() -> Result<BuiltModel>,
        inputs: &[(&str, &[f32])],
    ) -> Result<Vec<Vec<f32>>> {
        // hit: move to the back (most-recently-used) and run
        {
            let mut e = self.entries.borrow_mut();
            if let Some(pos) = e.iter().position(|(k, _)| *k == key) {
                let mut entry = e.remove(pos);
                let out = entry.1.run(inputs);
                e.push(entry);
                return Ok(out);
            }
        }
        // miss: build, compile, attach params
        let built = build()?;
        let saved = built.params().clone();
        let mut cg = compile_built(built, device)?;
        for (n, d) in &saved {
            cg.set_param(n, d);
        }
        let out = cg.run(inputs);
        let mut e = self.entries.borrow_mut();
        e.push((key, cg));
        if e.len() > self.cap {
            e.remove(0); // evict least-recently-used
        }
        Ok(out)
    }
}

impl Default for GraphCache {
    fn default() -> Self {
        Self::new(8)
    }
}
