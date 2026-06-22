//! WebSocket duplex Moshi server (Kyutai-style binary protocol subset).
//!
//! ```sh
//! cargo run -p rlx-moshi --example ws_server --features ws-server --release -- \
//!   --host 127.0.0.1 --port 8998
//! ```

use axum::{
    Router,
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::{Query, State},
    response::IntoResponse,
    routing::get,
};
use futures_util::{SinkExt, StreamExt};
use rlx_mimi;
use rlx_moshi::{
    GenerationConfig, MoshiSession, MoshiVariant, StreamCommand, StreamEvent, WsMsgType,
    decode_ws_message, encode_ws_audio, encode_ws_handshake, encode_ws_text, parse_moshi_device,
    spawn_duplex_tokio,
};
use rlx_runtime::Device;
use serde::Deserialize;
use std::net::SocketAddr;
use std::sync::Arc;

#[derive(Clone)]
struct AppState {
    moshi_dir: std::path::PathBuf,
    mimi_dir: std::path::PathBuf,
    device: Device,
    max_steps: usize,
}

#[derive(Debug, Deserialize)]
struct WsQuery {
    prompt: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut host = "127.0.0.1".to_string();
    let mut port = 8998u16;
    let mut device = Device::Cpu;
    let mut max_steps = 50usize;
    let mut moshi_dir = rlx_moshi::default_moshi_dir();
    let mut mimi_dir = rlx_moshi::default_mimi_dir();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--host" => {
                i += 1;
                host = args[i].clone();
            }
            "--port" => {
                i += 1;
                port = args[i].parse()?;
            }
            "--device" => {
                i += 1;
                device = parse_moshi_device(&args[i])?;
            }
            "--max-steps" => {
                i += 1;
                max_steps = args[i].parse()?;
            }
            "--model-dir" => {
                i += 1;
                moshi_dir = std::path::PathBuf::from(&args[i]);
            }
            "--mimi-dir" => {
                i += 1;
                mimi_dir = std::path::PathBuf::from(&args[i]);
            }
            _ => anyhow::bail!("unknown arg {}", args[i]),
        }
        i += 1;
    }

    rlx_moshi::ensure_weights(&moshi_dir)?;
    rlx_mimi::ensure_weights(&mimi_dir)?;

    let state = Arc::new(AppState {
        moshi_dir,
        mimi_dir,
        device,
        max_steps,
    });
    let app = Router::new()
        .route("/ws", get(ws_handler))
        .with_state(state);
    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    eprintln!("rlx-moshi ws listening on ws://{addr}/ws?prompt=Hello");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    Query(q): Query<WsQuery>,
    State(st): State<Arc<AppState>>,
) -> impl IntoResponse {
    let prompt = q.prompt.unwrap_or_else(|| "Hello.".into());
    ws.on_upgrade(move |socket| handle_socket(socket, st, prompt))
}

async fn handle_socket(socket: WebSocket, st: Arc<AppState>, prompt: String) {
    let (mut sender, mut receiver) = socket.split();
    let session = match MoshiSession::open_on(
        &st.moshi_dir,
        &st.mimi_dir,
        MoshiVariant::Moshiko,
        st.device,
    ) {
        Ok(s) => s,
        Err(e) => {
            let _ = sender
                .send(Message::Text(format!("error: {e:#}").into()))
                .await;
            return;
        }
    };
    let run_cfg = GenerationConfig {
        max_steps: st.max_steps,
        ..GenerationConfig::default()
    };
    let handle = match spawn_duplex_tokio(session, &prompt, run_cfg, 64) {
        Ok(h) => h,
        Err(e) => {
            let _ = sender
                .send(Message::Text(format!("error: {e:#}").into()))
                .await;
            return;
        }
    };

    let cmd_tx = handle.cmd_tx.clone();
    let mut event_rx = handle.event_rx;

    let pump = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            let msg = match msg {
                Ok(m) => m,
                Err(_) => break,
            };
            let data = match msg {
                Message::Binary(b) => b.to_vec(),
                Message::Text(t) if t == "finish" => {
                    let _ = cmd_tx.send(StreamCommand::Finish).await;
                    break;
                }
                Message::Close(_) => break,
                _ => continue,
            };
            if let Ok(Some((kind, pcm))) = decode_ws_message(&data) {
                match kind {
                    WsMsgType::Audio if !pcm.is_empty() => {
                        let _ = cmd_tx.send(StreamCommand::Pcm(pcm)).await;
                    }
                    WsMsgType::Control => {
                        let _ = cmd_tx.send(StreamCommand::Finish).await;
                        break;
                    }
                    _ => {}
                }
            }
        }
        let _ = cmd_tx.send(StreamCommand::Finish).await;
    });

    while let Some(ev) = event_rx.recv().await {
        match ev {
            StreamEvent::Ready => {
                let _ = sender
                    .send(Message::Binary(encode_ws_handshake().into()))
                    .await;
            }
            StreamEvent::OutputPcm { samples, .. } => {
                let _ = sender
                    .send(Message::Binary(encode_ws_audio(&samples).into()))
                    .await;
            }
            StreamEvent::Text { text, .. } => {
                let _ = sender
                    .send(Message::Binary(encode_ws_text(&text).into()))
                    .await;
            }
            StreamEvent::Finished(_) | StreamEvent::Error(_) => break,
            StreamEvent::Step(_) => {}
        }
    }
    let _ = pump.await;
    handle.stop();
}
