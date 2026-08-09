// RLX models — OpenAI-compatible server.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! The model-selection seam: how a request's `model` field maps to the
//! [`Engine`] that serves it.
//!
//! [`Engine`] is a *single* model's decode loop. A
//! server often wants to expose several models behind one OpenAI-compatible
//! endpoint and route each request by its `model` field — the same shape as
//! mesh-llm's `OpenAiBackend`. [`ModelBackend`] is that higher seam: it turns
//! a model name into an `Engine` and reports the aggregate `/v1/models` list.
//!
//! Two impls ship: [`SingleBackend`] (one engine answers for every model —
//! the back-compatible default that [`build_router`](crate::build_router)
//! wraps) and [`RegistryBackend`] (name → engine map with a fallback default).
//! Anything more elaborate — a mesh router, a rlx-protocol staged backend, an
//! upstream HTTP proxy — is just another `ModelBackend`.

use crate::engine::{Engine, ModelCard};
use std::collections::HashMap;
use std::sync::Arc;

/// Resolves a request's `model` name to the [`Engine`] that serves it, and
/// reports the models this server advertises.
pub trait ModelBackend: Send + Sync {
    /// The engine that should serve `model`, or `None` if this backend can't
    /// (the route turns `None` into a 400). Implementations are free to fall
    /// back to a default engine rather than fail on an unknown name.
    fn resolve(&self, model: &str) -> Option<Arc<dyn Engine>>;

    /// The union of every served model's `/v1/models` cards.
    fn model_cards(&self) -> Vec<ModelCard>;
}

/// One engine serves every request, regardless of the `model` field. This is
/// what [`build_router`](crate::build_router) wraps, so existing single-model
/// callers are unchanged.
pub struct SingleBackend(pub Arc<dyn Engine>);

impl SingleBackend {
    pub fn new(engine: Arc<dyn Engine>) -> Self {
        Self(engine)
    }
}

impl ModelBackend for SingleBackend {
    fn resolve(&self, _model: &str) -> Option<Arc<dyn Engine>> {
        Some(self.0.clone())
    }
    fn model_cards(&self) -> Vec<ModelCard> {
        self.0.model_cards()
    }
}

/// A name → engine registry. Each registered engine's own `model_cards()` ids
/// are indexed, so a request's `model` routes to the engine that advertises
/// it; unknown names fall back to the default (the first engine registered,
/// unless overridden). Multiple model ids can map to the same engine.
#[derive(Default)]
pub struct RegistryBackend {
    /// Insertion order preserved for a stable `/v1/models` listing.
    engines: Vec<Arc<dyn Engine>>,
    by_id: HashMap<String, Arc<dyn Engine>>,
    default: Option<Arc<dyn Engine>>,
}

impl RegistryBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an engine, indexing every model id it advertises. The first
    /// engine registered becomes the fallback default (override with
    /// [`with_default`](Self::with_default)).
    pub fn register(mut self, engine: Arc<dyn Engine>) -> Self {
        for card in engine.model_cards() {
            self.by_id.insert(card.id, engine.clone());
        }
        if self.default.is_none() {
            self.default = Some(engine.clone());
        }
        self.engines.push(engine);
        self
    }

    /// Set the fallback engine used when a request names an unknown model.
    pub fn with_default(mut self, engine: Arc<dyn Engine>) -> Self {
        self.default = Some(engine);
        self
    }

    /// Number of registered engines (not model ids).
    pub fn len(&self) -> usize {
        self.engines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.engines.is_empty()
    }
}

impl ModelBackend for RegistryBackend {
    fn resolve(&self, model: &str) -> Option<Arc<dyn Engine>> {
        self.by_id
            .get(model)
            .cloned()
            .or_else(|| self.default.clone())
    }
    fn model_cards(&self) -> Vec<ModelCard> {
        self.engines.iter().flat_map(|e| e.model_cards()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, GenRequest, ModelCard, StreamItem};
    use anyhow::Result;

    /// Minimal engine that only knows its id — enough to test routing.
    struct NamedEngine(&'static str);
    impl Engine for NamedEngine {
        fn model_cards(&self) -> Vec<ModelCard> {
            vec![ModelCard {
                id: self.0.to_string(),
            }]
        }
        fn encode_chat(&self, _t: &[crate::engine::ChatTurn]) -> Result<Vec<u32>> {
            Ok(vec![])
        }
        fn encode_text(&self, _t: &str) -> Result<Vec<u32>> {
            Ok(vec![])
        }
        fn eos_ids(&self) -> Vec<u32> {
            vec![]
        }
        fn decode_token(&self, _id: u32) -> String {
            String::new()
        }
        fn run(&self, _req: &GenRequest, _emit: &mut dyn FnMut(StreamItem) -> bool) {}
    }

    #[test]
    fn registry_routes_by_model_id() {
        let a: Arc<dyn Engine> = Arc::new(NamedEngine("qwen3"));
        let b: Arc<dyn Engine> = Arc::new(NamedEngine("gemma"));
        let reg = RegistryBackend::new().register(a).register(b);

        assert_eq!(reg.resolve("qwen3").unwrap().model_cards()[0].id, "qwen3");
        assert_eq!(reg.resolve("gemma").unwrap().model_cards()[0].id, "gemma");
        // Unknown name falls back to the first-registered default.
        assert_eq!(reg.resolve("mystery").unwrap().model_cards()[0].id, "qwen3");

        let ids: Vec<String> = reg.model_cards().into_iter().map(|c| c.id).collect();
        assert_eq!(ids, vec!["qwen3", "gemma"]);
    }

    #[test]
    fn single_backend_serves_any_model() {
        let e: Arc<dyn Engine> = Arc::new(NamedEngine("only"));
        let b = SingleBackend::new(e);
        assert_eq!(b.resolve("anything").unwrap().model_cards()[0].id, "only");
        assert_eq!(b.model_cards().len(), 1);
    }
}
