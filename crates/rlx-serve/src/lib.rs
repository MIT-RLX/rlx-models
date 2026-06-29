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

pub mod batch;
pub mod engine;
pub mod openai;
pub mod routes;
pub mod sampling_map;
pub mod stop;

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

/// Shared server state handed to every route.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<dyn Engine>,
    pub default_max_tokens: usize,
}

/// Build the axum router for an engine.
pub fn build_router(engine: Arc<dyn Engine>, default_max_tokens: usize) -> Router {
    let state = AppState {
        engine,
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
