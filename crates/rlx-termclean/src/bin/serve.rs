//! `rlx-termclean-serve` — HTTP serving binary for TUI frame cleaning.
//!
//! Exposes:
//!   - `GET  /health`        → `"ok"`
//!   - `POST /v1/termclean`  → clean submitted frames
//!
//! Request:  `{"frames": ["<raw frame>", ...]}`
//! Response: `{"clean": ["<clean text>", ...], "backend": "tagger"|"fastclean"}`
//!
//! With `--weights <dir>` (or `RLX_TERMCLEAN_WEIGHTS`) the ML tagger cleans the
//! frames; if the weights fail to load (or none are given) the server still
//! starts and falls back to the pure-rule [`rlx_termclean::fastclean`] path.
//!
//! ```sh
//! cargo run -p rlx-termclean --features infer --bin rlx-termclean-serve -- \
//!   --weights crates/rlx-termclean/weights/tagger --port 8081
//! curl localhost:8081/v1/termclean -H 'content-type: application/json' \
//!   -d '{"frames":["┌─ Files ─┐\n│ src/    │\n└─────────┘"]}'
//! ```

use std::sync::Arc;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;

use rlx_termclean::fastclean;
use rlx_termclean::tagger::Tagger;

/// Shared server state. `tagger` is `None` when running in fastclean fallback.
#[derive(Clone)]
struct AppState {
    tagger: Option<Arc<Tagger>>,
}

#[derive(Deserialize)]
struct Req {
    frames: Vec<String>,
}

#[derive(Serialize)]
struct Resp {
    clean: Vec<String>,
    backend: String,
}

async fn termclean(State(st): State<AppState>, Json(req): Json<Req>) -> Json<Resp> {
    if let Some(tagger) = st.tagger.clone() {
        // rlx-tensor's compile cache is thread-local, so we pin the whole batch
        // to ONE blocking op in v1 (no cross-thread contention on the cache).
        let clean = tokio::task::spawn_blocking(move || {
            let refs: Vec<&str> = req.frames.iter().map(|s| s.as_str()).collect();
            tagger.clean_batch(&refs)
        })
        .await
        .unwrap_or_default();
        Json(Resp {
            clean,
            backend: "tagger".to_string(),
        })
    } else {
        let refs: Vec<&str> = req.frames.iter().map(|s| s.as_str()).collect();
        Json(Resp {
            clean: fastclean::clean_batch(&refs),
            backend: "fastclean".to_string(),
        })
    }
}

#[tokio::main]
async fn main() {
    let mut weights: Option<String> = std::env::var("RLX_TERMCLEAN_WEIGHTS").ok();
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 8081;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => {
                i += 1;
                weights = args.get(i).cloned();
            }
            "--host" => {
                i += 1;
                if let Some(h) = args.get(i) {
                    host = h.clone();
                }
            }
            "--port" => {
                i += 1;
                if let Some(p) = args.get(i).and_then(|p| p.parse().ok()) {
                    port = p;
                }
            }
            other => {
                eprintln!("[rlx-termclean-serve] ignoring unknown arg {other}");
            }
        }
        i += 1;
    }

    // Try to load the tagger; on failure log a warning and run fastclean-only so
    // the server ALWAYS starts.
    let tagger = match &weights {
        Some(dir) => match Tagger::load(dir) {
            Ok(t) => {
                eprintln!("[rlx-termclean-serve] loaded tagger weights from {dir}");
                Some(Arc::new(t))
            }
            Err(e) => {
                eprintln!(
                    "[rlx-termclean-serve] WARNING: failed to load weights from {dir}: {e} — falling back to fastclean"
                );
                None
            }
        },
        None => {
            eprintln!("[rlx-termclean-serve] no --weights given — running fastclean (pure-rule)");
            None
        }
    };

    let state = AppState { tagger };
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/termclean", post(termclean))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{host}:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[rlx-termclean-serve] failed to bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("[rlx-termclean-serve] listening on http://{addr}");
    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("[rlx-termclean-serve] server error: {e}");
        std::process::exit(1);
    }
}
