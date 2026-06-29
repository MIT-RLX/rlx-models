// RLX models — OpenAI-compatible server.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Axum handlers for the OpenAI-compatible endpoints.

use crate::engine::{GenRequest, StreamItem};
use crate::openai::*;
use crate::sampling_map::to_sample_opts;
use crate::{ApiError, AppState, gen_id, now_unix};
use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use std::convert::Infallible;
use tokio_stream::wrappers::ReceiverStream;

pub async fn health() -> &'static str {
    "ok"
}

pub async fn models(State(st): State<AppState>) -> Json<ModelList> {
    let created = now_unix();
    let data = st
        .engine
        .model_cards()
        .into_iter()
        .map(|c| ModelEntry {
            id: c.id,
            object: "model",
            created,
            owned_by: "rlx",
        })
        .collect();
    Json(ModelList {
        object: "list",
        data,
    })
}

/// Build the SSE chunk event for one delta.
fn chunk_event(id: &str, created: u64, model: &str, delta: Delta, finish: Option<String>) -> Event {
    let chunk = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta,
            finish_reason: finish,
        }],
    };
    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
}

pub async fn chat_completions(
    State(st): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let engine = st.engine.clone();
    let prompt_ids = engine
        .encode_chat(&req.turns())
        .map_err(|e| ApiError::bad_request(format!("encode chat: {e}")))?;
    let (opts, bias) = to_sample_opts(&req.sampling(), req.logit_bias.as_ref());
    let want_lp = req.want_logprobs();
    let genreq = GenRequest {
        prompt_ids,
        opts,
        bias,
        max_tokens: req.max_tokens.unwrap_or(st.default_max_tokens),
        stop_ids: engine.eos_ids(),
        stop_strings: req
            .stop
            .clone()
            .map(StopField::into_vec)
            .unwrap_or_default(),
        want_logprobs: want_lp,
    };
    let model = req.model.clone();

    if req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
        let eng = engine.clone();
        tokio::task::spawn_blocking(move || {
            let id = gen_id("chatcmpl");
            let created = now_unix();
            let _ = tx.blocking_send(Ok(chunk_event(
                &id,
                created,
                &model,
                Delta {
                    role: Some("assistant"),
                    content: None,
                },
                None,
            )));
            let mut emit = |item: StreamItem| -> bool {
                match item {
                    StreamItem::Token { text, .. } => tx
                        .blocking_send(Ok(chunk_event(
                            &id,
                            created,
                            &model,
                            Delta {
                                role: None,
                                content: Some(text),
                            },
                            None,
                        )))
                        .is_ok(),
                    StreamItem::Done { finish_reason, .. } => {
                        let _ = tx.blocking_send(Ok(chunk_event(
                            &id,
                            created,
                            &model,
                            Delta {
                                role: None,
                                content: None,
                            },
                            Some(finish_reason.as_str().to_string()),
                        )));
                        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                        true
                    }
                    StreamItem::Error(e) => {
                        let _ = tx
                            .blocking_send(Ok(Event::default()
                                .data(format!("{{\"error\":{{\"message\":{e:?}}}}}"))));
                        false
                    }
                }
            };
            eng.run(&genreq, &mut emit);
        });
        return Ok(Sse::new(ReceiverStream::new(rx)).into_response());
    }

    // Non-streaming: collect on a blocking worker, then assemble.
    let eng = engine.clone();
    let items = tokio::task::spawn_blocking(move || {
        let mut v = Vec::new();
        eng.run(&genreq, &mut |it| {
            v.push(it);
            true
        });
        v
    })
    .await
    .map_err(|e| ApiError::internal(format!("join: {e}")))?;

    let mut content = String::new();
    let mut lp_content = Vec::new();
    let mut finish = "stop".to_string();
    let mut usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };
    for it in items {
        match it {
            StreamItem::Token { text, logprob, .. } => {
                if let Some(lp) = &logprob {
                    lp_content.push(logprob_entry(lp, &text, &|id| engine.decode_token(id)));
                }
                content.push_str(&text);
            }
            StreamItem::Done {
                finish_reason,
                prompt_tokens,
                completion_tokens,
            } => {
                finish = finish_reason.as_str().to_string();
                usage = Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                };
            }
            StreamItem::Error(e) => return Err(ApiError::internal(e)),
        }
    }
    let logprobs = want_lp.map(|_| LogprobsBlock {
        content: lp_content,
    });

    // Detect tool calls in the completed text and surface them OpenAI-style.
    let parsed = rlx_text::tool_parse::detect_and_parse(&content);
    let (tool_calls, finish) = if parsed.is_empty() {
        (None, finish)
    } else {
        (
            Some(tool_calls_out(&parsed, "call")),
            "tool_calls".to_string(),
        )
    };

    Ok(Json(ChatCompletionResponse {
        id: gen_id("chatcmpl"),
        object: "chat.completion",
        created: now_unix(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: RespMessage {
                role: "assistant",
                content,
                tool_calls,
            },
            finish_reason: finish,
            logprobs,
        }],
        usage,
    })
    .into_response())
}

pub async fn completions(
    State(st): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, ApiError> {
    let engine = st.engine.clone();
    let prompt_ids = engine
        .encode_text(&req.prompt)
        .map_err(|e| ApiError::bad_request(format!("encode prompt: {e}")))?;
    let (opts, bias) = to_sample_opts(&req.sampling(), req.logit_bias.as_ref());
    let genreq = GenRequest {
        prompt_ids,
        opts,
        bias,
        max_tokens: req.max_tokens.unwrap_or(st.default_max_tokens),
        stop_ids: engine.eos_ids(),
        stop_strings: req
            .stop
            .clone()
            .map(StopField::into_vec)
            .unwrap_or_default(),
        want_logprobs: None,
    };
    let model = req.model.clone();

    let eng = engine.clone();
    let items = tokio::task::spawn_blocking(move || {
        let mut v = Vec::new();
        eng.run(&genreq, &mut |it| {
            v.push(it);
            true
        });
        v
    })
    .await
    .map_err(|e| ApiError::internal(format!("join: {e}")))?;

    let mut text = String::new();
    let mut finish = "stop".to_string();
    let mut usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };
    for it in items {
        match it {
            StreamItem::Token { text: t, .. } => text.push_str(&t),
            StreamItem::Done {
                finish_reason,
                prompt_tokens,
                completion_tokens,
            } => {
                finish = finish_reason.as_str().to_string();
                usage = Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                };
            }
            StreamItem::Error(e) => return Err(ApiError::internal(e)),
        }
    }

    Ok(Json(CompletionResponse {
        id: gen_id("cmpl"),
        object: "text_completion",
        created: now_unix(),
        model,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason: finish,
        }],
        usage,
    }))
}
