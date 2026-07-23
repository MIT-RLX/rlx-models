// RLX models — OpenAI-compatible server.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! OpenAI-compatible HTTP server for rlx language models.
//!
//! Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
//! `/health`. Streaming over SSE. Everything above the [`engine::Engine`]
//! seam is host-side and `Device`-agnostic — the model runs wherever its
//! `LmRunner` runs (CPU / Metal / MLX).

pub mod backend;
pub mod batch;
pub mod engine;
pub mod openai;
pub mod routes;
pub mod sampling_map;
pub mod stop;

pub use backend::{ModelBackend, RegistryBackend, SingleBackend};
pub use batch::{
    BatchRunner, BatchedEngine, ContinuousBatcher, FusedBatchRunner, RunnerBatchRunner,
};
pub use engine::{Engine, GenRequest, SingleEngine, StreamItem};

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tower_http::cors::CorsLayer;

/// Shared server state handed to every route. Holds a [`ModelBackend`] so one
/// server can route requests across several models by their `model` field.
#[derive(Clone)]
pub struct AppState {
    pub backend: Arc<dyn ModelBackend>,
    pub default_max_tokens: usize,
}

/// Build the axum router for a single engine — wraps it in a
/// [`SingleBackend`] so existing single-model callers are unchanged.
pub fn build_router(engine: Arc<dyn Engine>, default_max_tokens: usize) -> Router {
    build_router_backend(Arc::new(SingleBackend::new(engine)), default_max_tokens)
}

/// Build the axum router over a [`ModelBackend`] (multi-model routing). Each
/// request's `model` field selects the engine via [`ModelBackend::resolve`].
pub fn build_router_backend(backend: Arc<dyn ModelBackend>, default_max_tokens: usize) -> Router {
    let state = AppState {
        backend,
        default_max_tokens,
    };
    Router::new()
        .route("/health", get(routes::health))
        .route("/v1/models", get(routes::models))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .route("/v1/completions", post(routes::completions))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Seconds since the Unix epoch (0 if the clock is before it).
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A fresh completion id.
pub fn gen_id(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::new_v4().simple())
}

/// Bind and run an axum OpenAI router until the server exits.
pub async fn serve_http(app: Router, host: &str, port: u16) -> anyhow::Result<()> {
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("binding {addr}: {e}"))?;
    eprintln!("rlx-serve listening on http://{addr}");
    eprintln!("  GET  /health  /v1/models");
    eprintln!("  POST /v1/chat/completions  /v1/completions");
    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("axum serve: {e}"))?;
    Ok(())
}

/// An error rendered as an OpenAI-style error envelope.
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }
    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(serde_json::json!({
            "error": { "message": self.message, "type": "rlx_serve_error" }
        }));
        (self.status, body).into_response()
    }
}
