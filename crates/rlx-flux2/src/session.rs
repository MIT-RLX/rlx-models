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

//! Reusable FLUX.2 runner sessions — keep compiled graphs warm across requests.

use super::pipeline::{Flux2SampleOutput, Flux2SampleParams, generate_to_rgb};
use crate::runner::{Flux2Runner, Flux2RunnerBuilder};
use anyhow::Result;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Cache key for deduplicating loaded runners.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Flux2SessionKey {
    pub weights: PathBuf,
    pub device: Device,
    pub config_path: Option<PathBuf>,
    pub lora_path: Option<PathBuf>,
    pub lora_scale_bits: u32,
    pub nvfp4: Option<bool>,
}

/// One loaded FLUX.2 pipeline — cheap to clone via `Arc`.
#[derive(Clone)]
pub struct Flux2Session {
    inner: Arc<Flux2Runner>,
}

impl Flux2Session {
    pub fn open(builder: Flux2RunnerBuilder) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(builder.build()?),
        })
    }

    pub fn runner(&self) -> &Flux2Runner {
        &self.inner
    }

    pub fn sample(&self, params: &Flux2SampleParams<'_>) -> Result<Flux2SampleOutput> {
        super::pipeline::sample_rectified_flow(&self.inner, params)
    }

    pub fn generate_rgb(&self, params: &Flux2SampleParams<'_>) -> Result<(Vec<u8>, u32, u32)> {
        generate_to_rgb(&self.inner, params)
    }
}

/// Process-wide cache of [`Flux2Runner`] instances (CLI `--reuse-session` / serve mode).
#[derive(Default)]
pub struct Flux2SessionCache {
    sessions: Mutex<HashMap<Flux2SessionKey, Arc<Flux2Runner>>>,
}

impl Flux2SessionCache {
    pub fn global() -> &'static Flux2SessionCache {
        static CACHE: std::sync::OnceLock<Flux2SessionCache> = std::sync::OnceLock::new();
        CACHE.get_or_init(Flux2SessionCache::default)
    }

    pub fn get_or_open(&self, builder: Flux2RunnerBuilder) -> Result<Flux2Session> {
        let key = builder
            .session_key()
            .ok_or_else(|| anyhow::anyhow!("session cache requires .weights(...) on builder"))?;
        let mut guard = self
            .sessions
            .lock()
            .map_err(|e| anyhow::anyhow!("session cache lock poisoned: {e}"))?;
        if let Some(r) = guard.get(&key) {
            return Ok(Flux2Session {
                inner: Arc::clone(r),
            });
        }
        let runner = Arc::new(builder.build()?);
        guard.insert(key, Arc::clone(&runner));
        Ok(Flux2Session { inner: runner })
    }

    pub fn len(&self) -> Result<usize> {
        Ok(self
            .sessions
            .lock()
            .map_err(|e| anyhow::anyhow!("session cache lock poisoned: {e}"))?
            .len())
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
}
