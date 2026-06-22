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

//! Bridge to plug a Gemma-style verifier into
//! [`crate::speculator::Eagle3Speculator`].
//!
//! This module provides [`CallbackHiddenSource`] — a thin adapter
//! that implements [`VerifierHiddenSource`] from a user-supplied
//! closure that returns the verifier's most recent aux hidden
//! states. The intent is that the caller drives the rlx-gemma decode
//! loop, then hands us the aux outputs each speculative round.
//!
//! ## Wiring contract
//!
//! The verifier-side glue lives in two places:
//!
//! 1. **`rlx-gemma`**'s decode graph already supports the
//!    `with_aux_hidden_layer_ids(&[2, 30, 57])` knob
//!    (`crates/rlx-gemma/src/flow.rs`), and
//!    `rlx_models_core::autoregressive::split_decode_logits_kv_aux`
//!    parses the resulting `(logits, K, V, aux)` outputs. So a
//!    caller can run rlx-gemma's decode with EAGLE3-style aux
//!    layer ids and get back the per-layer hidden states.
//!
//! 2. **`rlx-eagle3`**'s `Eagle3Speculator::new(cfg, weights, source)`
//!    accepts any `H: VerifierHiddenSource`. The simplest way to
//!    plug rlx-gemma in is to instantiate
//!    [`CallbackHiddenSource`] with a closure that returns the
//!    last-decode-step's aux states.
//!
//! ## Sketch of the full end-to-end loop
//!
//! ```text
//! loop {
//!     // 1. Verifier decode emits (logits, KV, aux_hidden_states).
//!     let (logits, _, _, aux) = run_gemma_decode_with_aux(
//!         &cfg, prompt_so_far, &[2, 30, 57])?;
//!
//!     // 2. Stash aux for the speculator to read.
//!     last_aux.store(aux);
//!
//!     // 3. Speculator proposes n draft tokens.
//!     let proposal = speculator.propose(&prompt_so_far, n);
//!
//!     // 4. Verifier scores them; accept/reject loop runs.
//!     // …
//! }
//! ```
//!
//! The closure-based design keeps `rlx-eagle3` agnostic to the
//! verifier framework — the same adapter works for any model whose
//! decode emits aux hidden states (Llama, Gemma, Qwen, ...).

use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::speculator::VerifierHiddenSource;

/// `VerifierHiddenSource` backed by a user-supplied callback.
///
/// The callback returns `aux_layer_ids.len()` vectors of length
/// `target_hidden_size`, in `aux_layer_ids` order. The closure is
/// boxed for type erasure so the speculator's `H` type parameter
/// stays simple.
pub struct CallbackHiddenSource {
    target_hidden: usize,
    layers: usize,
    /// Returns one row per aux layer, each `target_hidden` long.
    /// Wrapped in `Arc<Mutex<_>>` so the consumer can swap the
    /// underlying closure between speculative rounds — e.g. capture
    /// the latest decode-step output into a buffer and have the
    /// closure read from that buffer.
    callback: Arc<Mutex<Box<dyn FnMut() -> Result<Vec<Vec<f32>>> + Send>>>,
}

impl CallbackHiddenSource {
    pub fn new<F>(target_hidden: usize, layers: usize, f: F) -> Self
    where
        F: FnMut() -> Result<Vec<Vec<f32>>> + Send + 'static,
    {
        Self {
            target_hidden,
            layers,
            callback: Arc::new(Mutex::new(Box::new(f))),
        }
    }

    /// Shared handle to the underlying closure — useful when the
    /// caller wants to swap closures across propose() calls without
    /// recreating the speculator.
    pub fn callback_handle(&self) -> Arc<Mutex<Box<dyn FnMut() -> Result<Vec<Vec<f32>>> + Send>>> {
        Arc::clone(&self.callback)
    }
}

impl VerifierHiddenSource for CallbackHiddenSource {
    fn aux_hidden_states(&self) -> Result<Vec<Vec<f32>>> {
        let mut guard = self
            .callback
            .lock()
            .map_err(|e| anyhow::anyhow!("CallbackHiddenSource lock poisoned: {e}"))?;
        let out = (*guard)()?;
        anyhow::ensure!(
            out.len() == self.layers,
            "CallbackHiddenSource: closure returned {} layers, expected {}",
            out.len(),
            self.layers,
        );
        for (i, row) in out.iter().enumerate() {
            anyhow::ensure!(
                row.len() == self.target_hidden,
                "CallbackHiddenSource: row {i} has len {} (expected {})",
                row.len(),
                self.target_hidden,
            );
        }
        Ok(out)
    }
    fn target_hidden_size(&self) -> usize {
        self.target_hidden
    }
    fn num_aux_layers(&self) -> usize {
        self.layers
    }
}

/// Shared buffer for the common pattern of "verifier writes the
/// latest aux states; speculator reads them". The verifier-side code
/// calls [`AuxStateBuffer::write`] after each decode step; the
/// speculator-side closure calls [`AuxStateBuffer::read`].
#[derive(Clone)]
pub struct AuxStateBuffer {
    inner: Arc<Mutex<Option<Vec<Vec<f32>>>>>,
}

impl Default for AuxStateBuffer {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
        }
    }
}

impl AuxStateBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store the most recent aux hidden states. Overwrites whatever
    /// was there before.
    pub fn write(&self, aux: Vec<Vec<f32>>) {
        if let Ok(mut g) = self.inner.lock() {
            *g = Some(aux);
        }
    }

    /// Read + remove the buffered aux states. Returns an error if
    /// nothing has been written since the last read.
    pub fn read(&self) -> Result<Vec<Vec<f32>>> {
        let mut g = self
            .inner
            .lock()
            .map_err(|e| anyhow::anyhow!("AuxStateBuffer lock poisoned: {e}"))?;
        g.take().ok_or_else(|| {
            anyhow::anyhow!(
                "AuxStateBuffer: no aux states available — \
                 did the verifier forget to write before propose()?"
            )
        })
    }

    /// Build a [`CallbackHiddenSource`] that reads from this buffer.
    /// The closure errors out cleanly if the verifier never wrote.
    pub fn into_hidden_source(self, target_hidden: usize, layers: usize) -> CallbackHiddenSource {
        CallbackHiddenSource::new(target_hidden, layers, move || self.read())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_hidden_source_validates_shape() {
        let calls = Arc::new(Mutex::new(0usize));
        let calls_2 = Arc::clone(&calls);
        let src = CallbackHiddenSource::new(4, 3, move || {
            *calls_2.lock().unwrap() += 1;
            Ok(vec![vec![1.0; 4]; 3])
        });
        let aux = src.aux_hidden_states().unwrap();
        assert_eq!(aux.len(), 3);
        for row in &aux {
            assert_eq!(row.len(), 4);
        }
        assert_eq!(*calls.lock().unwrap(), 1);
    }

    #[test]
    fn callback_rejects_wrong_layer_count() {
        let src = CallbackHiddenSource::new(4, 3, || Ok(vec![vec![0.0; 4]; 2]));
        let err = src.aux_hidden_states().unwrap_err();
        assert!(format!("{err}").contains("returned 2 layers"));
    }

    #[test]
    fn callback_rejects_wrong_row_size() {
        let src = CallbackHiddenSource::new(4, 3, || Ok(vec![vec![0.0; 5]; 3]));
        let err = src.aux_hidden_states().unwrap_err();
        assert!(format!("{err}").contains("row 0 has len 5"));
    }

    #[test]
    fn aux_state_buffer_round_trip() {
        let buf = AuxStateBuffer::new();
        buf.write(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let out = buf.read().unwrap();
        assert_eq!(out, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        // Second read errors — write/read is one-shot.
        let err = buf.read().unwrap_err();
        assert!(format!("{err}").contains("did the verifier forget"));
    }

    #[test]
    fn aux_state_buffer_into_hidden_source() {
        let buf = AuxStateBuffer::new();
        let writer = buf.clone();
        let src = buf.into_hidden_source(3, 2);
        writer.write(vec![vec![0.0; 3], vec![0.0; 3]]);
        let aux = src.aux_hidden_states().unwrap();
        assert_eq!(aux.len(), 2);
        assert_eq!(aux[0].len(), 3);
    }
}
