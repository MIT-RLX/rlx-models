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

//! OpenAI-compatible server engine for Laguna packed generate.
//!
//! Greedy decode only for now (`temperature` / `top_p` on the wire are ignored).

use crate::chat::LagunaChat;
use crate::device_matmul::DeviceMatmul;
use crate::runner::LagunaPackedRunner;
use anyhow::Result;
use rlx_serve::engine::{ChatTurn, Engine, FinishReason, GenRequest, ModelCard, StreamItem};
use rlx_text::StreamingDetokenizer;
use rlx_text::chat::ChatMessage;
use std::sync::Mutex;

/// Host-driven Laguna engine behind [`rlx_serve::build_router`].
pub struct LagunaEngine {
    runner: Mutex<LagunaPackedRunner>,
    accel: Mutex<Option<DeviceMatmul>>,
    chat: LagunaChat,
    eos_ids: Vec<u32>,
    model_id: String,
}

impl LagunaEngine {
    pub fn new(
        runner: LagunaPackedRunner,
        chat: LagunaChat,
        accel: Option<DeviceMatmul>,
        model_id: impl Into<String>,
    ) -> Self {
        let eos = runner.config().eos_token_id;
        Self {
            runner: Mutex::new(runner),
            accel: Mutex::new(accel),
            chat,
            eos_ids: vec![eos],
            model_id: model_id.into(),
        }
    }
}

impl Engine for LagunaEngine {
    fn model_cards(&self) -> Vec<ModelCard> {
        vec![ModelCard {
            id: self.model_id.clone(),
        }]
    }

    fn encode_chat(&self, turns: &[ChatTurn]) -> Result<Vec<u32>> {
        let msgs: Vec<ChatMessage> = turns
            .iter()
            .map(|t| ChatMessage {
                role: t.role.clone(),
                content: t.content.clone(),
            })
            .collect();
        self.chat.encode_chat(&msgs, false)
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.chat.encode_text(text)
    }

    fn eos_ids(&self) -> Vec<u32> {
        self.eos_ids.clone()
    }

    fn decode_token(&self, id: u32) -> String {
        self.chat.decode_token(id)
    }

    fn run(&self, req: &GenRequest, emit: &mut dyn FnMut(StreamItem) -> bool) {
        let runner = match self.runner.lock() {
            Ok(g) => g,
            Err(e) => {
                emit(StreamItem::Error(format!("runner mutex: {e}")));
                return;
            }
        };
        let mut accel_guard = match self.accel.lock() {
            Ok(g) => g,
            Err(e) => {
                emit(StreamItem::Error(format!("accel mutex: {e}")));
                return;
            }
        };

        let prompt_len = req.prompt_ids.len();
        let mut detok = StreamingDetokenizer::new(&self.chat.tokenizer, true);
        let mut new_count = 0usize;
        let mut finish = FinishReason::Length;

        let gen_result = runner.generate_with_device(
            &req.prompt_ids,
            req.max_tokens,
            accel_guard.as_mut(),
            &mut |tok| {
                new_count += 1;
                let text = detok.push(tok).unwrap_or_default();
                let _ = emit(StreamItem::Token {
                    id: tok,
                    text,
                    logprob: None,
                });
                if req.stop_ids.contains(&tok) {
                    finish = FinishReason::Stop;
                }
            },
        );

        match gen_result {
            Ok(_) => {
                let _ = emit(StreamItem::Done {
                    finish_reason: finish,
                    prompt_tokens: prompt_len,
                    completion_tokens: new_count,
                });
            }
            Err(e) => {
                emit(StreamItem::Error(e.to_string()));
            }
        }
    }
}

/// Run the axum OpenAI server until Ctrl-C.
///
/// Prefer `rlx-openai --engine laguna …` for multi-model hosts.
pub async fn serve(
    engine: LagunaEngine,
    host: &str,
    port: u16,
    default_max_tokens: usize,
) -> Result<()> {
    let app = rlx_serve::build_router(std::sync::Arc::new(engine), default_max_tokens);
    eprintln!("  note: Laguna greedy decode only (temperature/top_p ignored)");
    rlx_serve::serve_http(app, host, port).await
}
