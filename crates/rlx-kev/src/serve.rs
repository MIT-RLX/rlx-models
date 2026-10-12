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

//! TypeSafe-compatible HTTP sidecar: `POST /v1/systemone`.
//!
//! Deliberately **not** part of `rlx-serve`. That crate is an
//! OpenAI-compatible server built around a generative `Engine` — streaming,
//! sampling, stop sequences, continuous batching — and a decision model has
//! none of those. Putting this there would also invert the dependency
//! (`rlx-serve` → `rlx-kev`) and drag a decision model into every LM server
//! build. kev itself ships a separate FastAPI sidecar for the same reason.
//!
//! # Concurrency
//!
//! One model, one mutex. The forward pass owns a compiled graph and its
//! parameter arena, which is not shareable, and requests are single-shot
//! rather than streamed — so serializing is the honest design rather than a
//! placeholder. Requests queue; they do not race.
//!
//! # State-prefix caching
//!
//! Enabled with `--prefix-cache`: a repeated state is encoded once and only
//! the question branches are replayed. Measured on a 625-token state with
//! three questions, warm: **1461 ms -> 265 ms**. Off by default because it
//! costs extra compiled parameter arenas.
//!
//! The server binds loopback and has no authentication, matching kev.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tower_http::cors::CorsLayer;

use crate::api::{Answer, SystemOneRequest, answers_json, to_record};
use crate::encode::HfTokenizer;
use crate::json::{Json as KJson, obj};
use crate::model::DecisionModel;

/// Shared server state: the single loaded model plus its metadata.
pub struct ServerState {
    model: Mutex<DecisionModel<HfTokenizer>>,
    run: String,
    base: String,
    device: String,
    temperature: f32,
}

impl ServerState {
    /// Live prefix-cache counters for `/v1/models`.
    fn prefix_stats(&self) -> serde_json::Value {
        #[cfg(feature = "prefix-cache")]
        {
            let stats = self.model.lock().ok().and_then(|m| m.prefix_stats());
            if let Some((hits, misses, cached)) = stats {
                return serde_json::json!({
                    "enabled": true, "hits": hits, "misses": misses, "cached_states": cached
                });
            }
        }
        serde_json::json!({ "enabled": false })
    }

    pub fn new(
        model: DecisionModel<HfTokenizer>,
        run: String,
        base: String,
        device: String,
    ) -> Self {
        let temperature = model.head().temperature;
        Self {
            model: Mutex::new(model),
            run,
            base,
            device,
            temperature,
        }
    }
}

/// An error carried back to the client as JSON.
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": { "message": self.1 } });
        (self.0, Json(body)).into_response()
    }
}

fn unprocessable(e: impl std::fmt::Display) -> ApiError {
    ApiError(StatusCode::UNPROCESSABLE_ENTITY, e.to_string())
}

/// Build the router. Caller owns the listener.
pub fn router(state: Arc<ServerState>) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(models))
        .route("/v1/systemone", post(systemone))
        .route("/v1/systemone/separate", post(separate))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn models(State(st): State<Arc<ServerState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "models": [{
            "id": "kev-latest",
            "aliases": ["jev-latest"],
            "run": st.run,
            "base": st.base,
            "device": st.device,
            "temperature": st.temperature,
            "prefix_cache": st.prefix_stats(),
        }]
    }))
}

/// Answer one request. Returns the TypeSafe response body.
fn answer_one(
    st: &ServerState,
    req: &SystemOneRequest,
) -> std::result::Result<(Vec<(String, Answer)>, crate::model::Usage, f64), ApiError> {
    let (rec, _meta) = to_record(req).map_err(unprocessable)?;
    let mut model = st.model.lock().map_err(|e| {
        ApiError(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("model lock: {e}"),
        )
    })?;
    let t0 = std::time::Instant::now();
    let (answers, usage) = model.answer(req).map_err(unprocessable)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    drop(rec);
    Ok((answers, usage, (ms * 10.0).round() / 10.0))
}

fn body(
    model_name: &str,
    answers: &[(String, Answer)],
    usage: crate::model::Usage,
    latency_ms: f64,
) -> KJson {
    obj([
        ("model", KJson::from(model_name.to_string())),
        ("answers", answers_json(answers)),
        (
            "usage",
            obj([
                ("input_tokens", KJson::from(usage.input_tokens)),
                ("state_tokens", KJson::from(usage.state_tokens)),
            ]),
        ),
        ("latency_ms", KJson::Float(latency_ms)),
    ])
}

async fn systemone(
    State(st): State<Arc<ServerState>>,
    body_text: String,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    // Parse from the raw text, not from an already-decoded `serde_json::Value`
    // — object key order is part of the prompt and `Value` would sort it.
    let req = SystemOneRequest::parse(&body_text).map_err(unprocessable)?;
    let (answers, usage, ms) = answer_one(&st, &req)?;
    let out = body(&req.model, &answers, usage, ms);
    Ok(Json(serde_json::to_value(&out).map_err(|e| {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?))
}

/// Each question in its own pass, for the packed-vs-separate comparison.
///
/// In the row form these must agree exactly with `/v1/systemone`, since the
/// rows are independent computations either way — which makes this a live
/// isolation check rather than a demo.
async fn separate(
    State(st): State<Arc<ServerState>>,
    body_text: String,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    let req = SystemOneRequest::parse(&body_text).map_err(unprocessable)?;
    let mut all: Vec<(String, Answer)> = Vec::new();
    let (mut input_tokens, mut state_tokens, mut total_ms) = (0usize, 0usize, 0.0f64);
    for (id, q) in &req.questions.0 {
        let one = SystemOneRequest {
            state: req.state.clone(),
            model: req.model.clone(),
            questions: crate::api::Questions(vec![(id.clone(), q.clone())]),
        };
        let (answers, usage, ms) = answer_one(&st, &one)?;
        all.extend(answers);
        input_tokens += usage.input_tokens;
        state_tokens = usage.state_tokens;
        total_ms += ms;
    }
    let out = body(
        &req.model,
        &all,
        crate::model::Usage {
            input_tokens,
            state_tokens,
        },
        (total_ms * 10.0).round() / 10.0,
    );
    Ok(Json(serde_json::to_value(&out).map_err(|e| {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?))
}

/// Serve on `addr` until the process is interrupted.
pub fn serve_blocking(state: Arc<ServerState>, addr: SocketAddr) -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building the tokio runtime")?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .with_context(|| format!("binding {addr}"))?;
        eprintln!("rlx-kev serving on http://{addr}  (no auth; keep it local)");
        axum::serve(listener, router(state))
            .await
            .context("serving")?;
        Ok::<_, anyhow::Error>(())
    })
}
