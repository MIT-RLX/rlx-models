// RLX models — reusable guardrails (tool-call validation + compaction).
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Reusable, framework-agnostic guardrails for RLX OpenAI-compatible paths.
//!
//! Ported from `mesh-llm-guardrails` so `rlx-serve` (and anything else that
//! speaks the OpenAI function-calling wire format) can validate model tool
//! calls and keep conversations inside a token budget. There is no coupling
//! to any HTTP layer, model runner, or `Device` — everything here operates on
//! plain [`ToolDef`] / [`ToolCall`] / [`Message`] values that (de)serialize to
//! the OpenAI JSON shapes, with dynamic schemas kept as [`serde_json::Value`].
//!
//! # Tool-call validation
//! ```
//! use rlx_guardrails::{ToolCall, ToolDef, validate_tool_call};
//! use serde_json::json;
//!
//! let tools = vec![ToolDef::function(
//!     "read_file",
//!     json!({
//!         "type": "object",
//!         "properties": { "path": { "type": "string" } },
//!         "required": ["path"],
//!     }),
//! )];
//! let call = ToolCall::new("read_file", json!({ "path": "README.md" }));
//! assert!(validate_tool_call(&call, &tools).is_ok());
//! ```
//!
//! # Compaction
//! ```
//! use rlx_guardrails::{Message, compact_messages};
//!
//! let messages = vec![
//!     Message::new("system", "You are helpful."),
//!     Message::new("user", "old and long ...".repeat(50)),
//!     Message::new("user", "hi"),
//! ];
//! let trimmed = compact_messages(messages, 40);
//! assert!(trimmed.iter().any(|m| m.is_system()));
//! ```

pub mod compact;
pub mod error;
pub mod tools;

pub use compact::{Message, compact_messages, estimate_messages_tokens, estimate_tokens};
pub use error::GuardrailError;
pub use tools::{
    FunctionDef, RESERVED_TOOL_NAMES, RESERVED_TOOL_PREFIX, RLX_EMIT_STRUCTURED, RLX_RESPOND,
    ToolCall, ToolDef, is_reserved_tool_name, is_reserved_tool_name_with_prefix,
    normalize_tool_arguments, rlx_emit_structured_tool_definition, rlx_respond_tool_definition,
    sanitize_tool_arguments, tool_arguments_wire_string, tools_collide_with_reserved,
    validate_tool_call, validate_tool_call_anyhow,
};
