// RLX models — OpenAI-compatible server.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! HTTP-level smoke test of the OpenAI endpoints, driven by a mock engine
//! so it needs no model or tokenizer.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use rlx_serve::build_router;
use rlx_serve::engine::{ChatTurn, Engine, FinishReason, GenRequest, ModelCard, StreamItem};
use std::sync::Arc;
use tower::ServiceExt;

struct MockEngine;

impl Engine for MockEngine {
    fn model_cards(&self) -> Vec<ModelCard> {
        vec![ModelCard { id: "mock".into() }]
    }
    fn encode_chat(&self, _t: &[ChatTurn]) -> anyhow::Result<Vec<u32>> {
        Ok(vec![1, 2, 3])
    }
    fn encode_text(&self, _t: &str) -> anyhow::Result<Vec<u32>> {
        Ok(vec![1, 2, 3])
    }
    fn eos_ids(&self) -> Vec<u32> {
        vec![]
    }
    fn decode_token(&self, id: u32) -> String {
        format!("<{id}>")
    }
    fn run(&self, req: &GenRequest, emit: &mut dyn FnMut(StreamItem) -> bool) {
        let n = req.max_tokens.min(2);
        for i in 0..n {
            if !emit(StreamItem::Token {
                id: 100 + i as u32,
                text: "hi ".into(),
                logprob: None,
            }) {
                return;
            }
        }
        emit(StreamItem::Done {
            finish_reason: FinishReason::Length,
            prompt_tokens: 3,
            completion_tokens: n,
        });
    }
}

fn app() -> axum::Router {
    build_router(Arc::new(MockEngine), 8)
}

async fn body_json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn health_ok() {
    let resp = app()
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn models_lists_mock() {
    let resp = app()
        .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["id"], "mock");
}

#[tokio::test]
async fn chat_non_streaming_assembles_content() {
    let payload = r#"{"model":"mock","messages":[{"role":"user","content":"hi"}],"max_tokens":2}"#;
    let resp = app()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(payload))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["content"], "hi hi ");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["usage"]["completion_tokens"], 2);
}

#[tokio::test]
async fn chat_streaming_emits_sse_chunks() {
    let payload = r#"{"model":"mock","messages":[{"role":"user","content":"hi"}],"max_tokens":2,"stream":true}"#;
    let resp = app()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(payload))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    // SSE frames for the role delta, content deltas, and a terminal [DONE].
    assert!(text.contains("chat.completion.chunk"));
    assert!(text.contains("\"content\":\"hi \""));
    assert!(text.contains("[DONE]"));
}

/// Emits a Hermes-style tool call as the assistant's content.
struct ToolMockEngine;
impl Engine for ToolMockEngine {
    fn model_cards(&self) -> Vec<ModelCard> {
        vec![ModelCard { id: "tool".into() }]
    }
    fn encode_chat(&self, _t: &[ChatTurn]) -> anyhow::Result<Vec<u32>> {
        Ok(vec![1])
    }
    fn encode_text(&self, _t: &str) -> anyhow::Result<Vec<u32>> {
        Ok(vec![1])
    }
    fn eos_ids(&self) -> Vec<u32> {
        vec![]
    }
    fn decode_token(&self, id: u32) -> String {
        format!("<{id}>")
    }
    fn run(&self, _req: &GenRequest, emit: &mut dyn FnMut(StreamItem) -> bool) {
        emit(StreamItem::Token {
            id: 1,
            text: r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#
                .into(),
            logprob: None,
        });
        emit(StreamItem::Done {
            finish_reason: FinishReason::Stop,
            prompt_tokens: 1,
            completion_tokens: 1,
        });
    }
}

#[tokio::test]
async fn chat_surfaces_tool_calls() {
    let app = build_router(Arc::new(ToolMockEngine), 8);
    let payload = r#"{"model":"tool","messages":[{"role":"user","content":"weather?"}]}"#;
    let resp = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(payload))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    let tc = &v["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(tc["type"], "function");
    assert_eq!(tc["function"]["name"], "get_weather");
    // arguments is a JSON-encoded string.
    let args: serde_json::Value =
        serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["city"], "Paris");
}

#[tokio::test]
async fn completions_endpoint_returns_text() {
    let payload = r#"{"model":"mock","prompt":"once upon","max_tokens":2}"#;
    let resp = app()
        .oneshot(
            Request::post("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(payload))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["choices"][0]["text"], "hi hi ");
}
