// RLX models — guardrails: context compaction.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Cheap token estimation and context compaction.
//!
//! Ported from `mesh-llm-guardrails`'s `compact.rs`. The source operated on
//! raw `serde_json::Value` messages and drove compaction from a
//! percentage-of-context config; here we expose a strongly-typed [`Message`]
//! and a direct token *budget*, keeping the same core policy: never drop a
//! `system` message, and evict the oldest non-system messages (tool results
//! first) until the budget fits.

use serde::{Deserialize, Serialize};

/// A chat message in OpenAI shape (`role` + `content`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// `"system"`, `"user"`, `"assistant"`, `"tool"`, …
    pub role: String,
    /// The message text.
    #[serde(default)]
    pub content: String,
}

impl Message {
    /// Construct a message from a role and content.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    /// `true` for a `system` message (never dropped by compaction).
    pub fn is_system(&self) -> bool {
        self.role == "system"
    }

    /// `true` for a `tool` result message (dropped first).
    pub fn is_tool_result(&self) -> bool {
        self.role == "tool"
    }

    /// Rough token count for this message: the content estimate plus a small
    /// per-message overhead for role/formatting framing.
    pub fn estimate_tokens(&self) -> usize {
        estimate_tokens(&self.content) + estimate_tokens(&self.role) + 3
    }
}

/// Cheap token estimate for a string: roughly `chars / 4`, always at least 1
/// for non-empty input. Matches the source's `len() / 4 + 1` heuristic while
/// treating empty strings as zero tokens.
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    text.chars().count() / 4 + 1
}

/// Total estimated tokens across a message slice.
pub fn estimate_messages_tokens(messages: &[Message]) -> usize {
    messages.iter().map(Message::estimate_tokens).sum()
}

/// Trim `messages` so the total estimated token count fits within
/// `budget_tokens`.
///
/// Policy (ported from the source, ordered by aggressiveness):
/// 1. `system` messages are always retained;
/// 2. non-system messages are evicted oldest-first, `tool` results before
///    anything else, until the estimate fits the budget;
/// 3. if only system messages remain and they still exceed the budget, they
///    are kept anyway (dropping instructions is never safe).
pub fn compact_messages(messages: Vec<Message>, budget_tokens: usize) -> Vec<Message> {
    if estimate_messages_tokens(&messages) <= budget_tokens {
        return messages;
    }

    let mut messages = messages;

    // Pass 1: evict oldest tool-result messages first.
    evict_oldest_while_over(&mut messages, budget_tokens, Message::is_tool_result);
    if estimate_messages_tokens(&messages) <= budget_tokens {
        return messages;
    }

    // Pass 2: evict any oldest non-system message.
    evict_oldest_while_over(&mut messages, budget_tokens, |m| !m.is_system());

    messages
}

fn evict_oldest_while_over(
    messages: &mut Vec<Message>,
    budget_tokens: usize,
    mut evictable: impl FnMut(&Message) -> bool,
) {
    while estimate_messages_tokens(messages) > budget_tokens {
        let Some(index) = messages.iter().position(&mut evictable) else {
            break;
        };
        messages.remove(index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_tokens_is_chars_over_four() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 2);
        assert_eq!(estimate_tokens("the quick brown fox"), 19 / 4 + 1);
    }

    #[test]
    fn under_budget_passes_through_unchanged() {
        let messages = vec![
            Message::new("system", "You are helpful."),
            Message::new("user", "hi"),
        ];
        let out = compact_messages(messages.clone(), 10_000);
        assert_eq!(out, messages);
    }

    #[test]
    fn keeps_system_message_and_fits_budget() {
        let system = Message::new("system", "SYSTEM PROMPT");
        let messages = vec![
            system.clone(),
            Message::new("user", "a".repeat(400)),
            Message::new("assistant", "b".repeat(400)),
            Message::new("user", "short"),
        ];
        let budget = system.estimate_tokens() + Message::new("user", "short").estimate_tokens();
        let out = compact_messages(messages, budget);

        // System message survived.
        assert!(out.iter().any(|m| m.is_system()));
        assert_eq!(out[0].role, "system");
        // Result fits the budget.
        assert!(estimate_messages_tokens(&out) <= budget);
    }

    #[test]
    fn drops_tool_results_before_other_messages() {
        let system = Message::new("system", "sys");
        let user = Message::new("user", "keep me");
        let messages = vec![
            system.clone(),
            Message::new("tool", "x".repeat(400)),
            user.clone(),
        ];
        let budget = system.estimate_tokens() + user.estimate_tokens();
        let out = compact_messages(messages, budget);

        assert!(out.iter().any(|m| m.is_system()));
        assert!(out.iter().any(|m| m.role == "user"));
        assert!(!out.iter().any(|m| m.is_tool_result()));
        assert!(estimate_messages_tokens(&out) <= budget);
    }

    #[test]
    fn keeps_system_even_when_it_exceeds_budget() {
        let messages = vec![Message::new("system", "s".repeat(4000))];
        let out = compact_messages(messages.clone(), 1);
        assert_eq!(out, messages);
    }
}
