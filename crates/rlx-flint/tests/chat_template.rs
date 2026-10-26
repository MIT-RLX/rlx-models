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

//! The chat template against Flint's own `chat_template.jinja`, rendered by
//! transformers in `scripts/flint_chat_reference.py`: tool declarations,
//! calls, tool results and where the tool preamble lands.

use rlx_flint::chat::{Message, Role, render_with_tools};
use rlx_flint::tools::{Tool, ToolCall, parse_reply};

const FIXTURE: &str = include_str!("fixtures/chat_template.json");

fn role(s: &str) -> Role {
    match s {
        "system" => Role::System,
        "developer" => Role::Developer,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        other => panic!("unknown role {other}"),
    }
}

fn cases() -> Vec<(String, Vec<Tool>, Vec<Message>, bool, String)> {
    let v: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    v["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let tools = Tool::list_from_json(c["tools"].as_str().unwrap()).unwrap();
            let messages = c["messages"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| {
                    let mut msg = Message::new(
                        role(m["role"].as_str().unwrap()),
                        m["content"].as_str().unwrap(),
                    );
                    msg.reasoning = m["reasoning"].as_str().map(String::from);
                    if let Some(calls) = m["tool_calls"].as_array() {
                        msg.tool_calls = calls
                            .iter()
                            .map(|tc| {
                                ToolCall::from_json(
                                    tc["name"].as_str().unwrap(),
                                    tc["arguments"].as_str().unwrap(),
                                )
                                .unwrap()
                            })
                            .collect();
                    }
                    msg
                })
                .collect();
            (
                c["name"].as_str().unwrap().to_string(),
                tools,
                messages,
                c["add_generation_prompt"].as_bool().unwrap(),
                c["expected"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

#[test]
fn renders_exactly_what_the_jinja_template_renders() {
    let cases = cases();
    assert!(cases.len() >= 5);
    for (name, tools, messages, generation_prompt, expected) in cases {
        assert_eq!(
            render_with_tools(&messages, &tools, generation_prompt),
            expected,
            "case {name}"
        );
    }
}

#[test]
fn rendered_assistant_turns_parse_back_into_their_calls() {
    for (name, tools, messages, _, _) in cases() {
        for m in messages.iter().filter(|m| m.role == Role::Assistant) {
            let turn = render_with_tools(std::slice::from_ref(m), &[], false);
            let reply = parse_reply(turn.strip_prefix("<|assistant|>").unwrap(), &tools);
            assert_eq!(reply.reasoning, m.reasoning, "case {name}");
            assert_eq!(reply.content, m.content, "case {name}");
            assert_eq!(reply.tool_calls, m.tool_calls, "case {name}");
        }
    }
}
