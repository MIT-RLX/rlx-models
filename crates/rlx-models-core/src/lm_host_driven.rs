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

//! Serving a model family that has no host-driven decode of its own.
//!
//! [`LmRunner`] has two decode shapes. `generate` owns the loop; the server
//! needs the other one — `prefill_logits` then `decode_logits` per token — so
//! that sampling, logit bias, log-probs and stop handling stay above the model.
//! `rlx_serve::run_generation` calls `prefill_logits` with `?`, and the trait's
//! default returns *unsupported*, so a family that has not implemented the pair
//! cannot be served at all. Of ten model families in this workspace, two have.
//!
//! The pair cannot be defaulted in the trait, because `decode_logits(token)` is
//! stateful and a trait default has nowhere to keep the context. [`HostDriven`]
//! keeps it: it tracks the tokens seen and answers each step by re-running
//! `predict_logits` over the whole context.
//!
//! ```no_run
//! use rlx_models_core::lm_host_driven::HostDriven;
//! use rlx_runtime::lm::LmRunner;
//! # fn get() -> Box<dyn LmRunner> { unimplemented!() }
//! // `Box<dyn LmRunner>` is what a caller holds, so that is the shape this
//! // takes: anything with `predict_logits` becomes servable.
//! let mut runner: HostDriven<dyn LmRunner> = HostDriven::new(get());
//! let logits = runner.prefill_logits(&[9707, 11])?;
//! # let _ = logits;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ## The cost, stated plainly
//!
//! This is **quadratic**: each step re-reads the whole context, so `n` tokens
//! cost `O(n²)` work where a cached decode costs `O(n)`. It is the same bargain
//! the trait already strikes for `generate`'s default, and it is the difference
//! between a family being slow and being unavailable — but a family that will
//! be served in earnest should implement the pair natively and skip this.
//! [`HostDriven::is_passthrough`] reports whether that has happened.

use anyhow::{Result, bail};
use rlx_runtime::lm::{LmRunner, SessionSnapshot};

/// Gives any [`LmRunner`] the host-driven decode pair, by re-prefilling.
///
/// Every other method delegates, so wrapping a runner that *does* implement the
/// pair natively is a no-op on correctness and a waste of a re-prefill — see
/// [`Self::is_passthrough`].
pub struct HostDriven<R: LmRunner + ?Sized> {
    inner: Box<R>,
    /// Prompt plus every token fed to [`LmRunner::decode_logits`] since.
    context: Vec<u32>,
    /// Whether the inner runner answers the pair itself. Unknown until the
    /// first prefill, which is what settles it.
    native: Option<bool>,
    steps: usize,
}

impl<R: LmRunner + ?Sized> HostDriven<R> {
    /// Wrap `inner`. Nothing is called on it here.
    ///
    /// Whether the wrapper is needed is settled by the **first** real
    /// `prefill_logits`: it is tried on the inner runner, and only an
    /// *unsupported* error falls back to re-prefilling. Deciding it that way
    /// rather than probing at construction means no synthetic call is ever made
    /// into a model — a runner asked to prefill an empty prompt before the
    /// caller has done anything is entitled to do something surprising.
    pub fn new(inner: Box<R>) -> Self {
        HostDriven {
            inner,
            context: Vec::new(),
            native: None,
            steps: 0,
        }
    }

    /// Whether calls pass straight through because the family implements the
    /// pair itself. `None` until the first prefill settles it; `Some(true)`
    /// means this wrapper is doing nothing for you.
    pub fn is_passthrough(&self) -> Option<bool> {
        self.native
    }

    /// Re-prefills performed so far — the quadratic term, made countable.
    pub fn reprefills(&self) -> usize {
        self.steps
    }

    /// Tokens the wrapper believes are in context.
    pub fn context(&self) -> &[u32] {
        &self.context
    }

    pub fn inner(&self) -> &R {
        &self.inner
    }

    pub fn into_inner(self) -> Box<R> {
        self.inner
    }
}

impl<R: LmRunner + ?Sized> LmRunner for HostDriven<R> {
    fn family(&self) -> &'static str {
        self.inner.family()
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.inner.predict_logits(prompt_ids)
    }

    fn prefill_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        if prompt_ids.is_empty() {
            bail!("{}: prefill needs at least one token", self.inner.family());
        }
        if self.native != Some(false) {
            match self.inner.prefill_logits(prompt_ids) {
                Ok(logits) => {
                    self.native = Some(true);
                    return Ok(logits);
                }
                // Only "unsupported" means wrap it; any other failure is the
                // model's real answer and must not be retried differently.
                Err(e) if format!("{e:#}").contains("unsupported") => {
                    self.native = Some(false);
                }
                Err(e) => return Err(e),
            }
        }
        self.context.clear();
        self.context.extend_from_slice(prompt_ids);
        self.steps += 1;
        self.inner.predict_logits(&self.context)
    }

    fn decode_logits(&mut self, token: u32) -> Result<Vec<f32>> {
        if self.native == Some(true) {
            return self.inner.decode_logits(token);
        }
        if self.context.is_empty() {
            bail!(
                "{}: decode_logits before prefill_logits — there is no context \
                 to extend",
                self.inner.family()
            );
        }
        self.context.push(token);
        self.steps += 1;
        self.inner.predict_logits(&self.context)
    }

    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.inner.generate(prompt_ids, n_new, on_token)
    }

    fn supports_multimodal(&self) -> bool {
        self.inner.supports_multimodal()
    }

    fn export_session(&self) -> Option<SessionSnapshot> {
        self.inner.export_session()
    }

    fn restore_session(&mut self, snap: &SessionSnapshot) -> bool {
        self.inner.restore_session(snap)
    }

    fn kv_store_stats(&self) -> Option<(usize, usize, usize)> {
        self.inner.kv_store_stats()
    }
}
