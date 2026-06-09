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

//! Per-family chat-template fixture transcripts (PLAN.md M3).
//!
//! Each JSON file under `tests/chat_templates/` carries one
//! conversation: the Jinja `template`, optional `bos_token` /
//! `eos_token`, the `messages` array, the `add_generation_prompt`
//! flag, and the `expected` rendered string. This test loads every
//! fixture in the directory and asserts the renderer reproduces
//! `expected` byte-for-byte.
//!
//! Adding a family: drop another `<family>.json` into the directory
//! — no code change required.

use std::path::PathBuf;

use rlx_cli::{ChatMessage, ChatTemplate};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct FixtureMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatFixture {
    name: String,
    #[allow(dead_code)]
    family: String,
    #[allow(dead_code)]
    #[serde(default)]
    comment: Option<String>,
    template: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
    messages: Vec<FixtureMessage>,
    add_generation_prompt: bool,
    expected: String,
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/chat_templates")
}

fn load_fixtures() -> Vec<(PathBuf, ChatFixture)> {
    let dir = fixtures_dir();
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("read chat_templates dir") {
        let entry = entry.expect("entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {path:?}: {e}"));
        let fx: ChatFixture = serde_json::from_slice(&bytes)
            .unwrap_or_else(|e| panic!("parse fixture {path:?}: {e}"));
        out.push((path, fx));
    }
    out.sort_by(|a, b| a.0.file_name().cmp(&b.0.file_name()));
    out
}

fn to_chat_messages(fx: &ChatFixture) -> Vec<ChatMessage> {
    fx.messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect()
}

#[test]
fn every_family_fixture_renders_to_expected() {
    let fixtures = load_fixtures();
    assert!(
        !fixtures.is_empty(),
        "no fixtures found under {:?}",
        fixtures_dir()
    );

    let mut failures: Vec<String> = Vec::new();
    for (path, fx) in &fixtures {
        let tmpl = ChatTemplate::from_source(&fx.template)
            .expect("compile template")
            .with_tokens(fx.bos_token.clone(), fx.eos_token.clone());
        let rendered = match tmpl.render(&to_chat_messages(fx), fx.add_generation_prompt) {
            Ok(s) => s,
            Err(e) => {
                failures.push(format!("[{}] render error: {e:#} ({:?})", fx.name, path));
                continue;
            }
        };
        if rendered != fx.expected {
            failures.push(format!(
                "[{}] mismatch:\n  expected = {:?}\n  rendered = {:?}\n  fixture  = {:?}",
                fx.name, fx.expected, rendered, path
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} fixtures failed:\n{}",
        failures.len(),
        fixtures.len(),
        failures.join("\n")
    );
}

#[test]
fn fixture_dir_covers_each_listed_family_once() {
    // Catches the easy mistake of two fixtures claiming the same
    // `family` (e.g. two `qwen3` fixtures with no naming distinction).
    // Multiple fixtures per family are allowed when their file names
    // differ — the assertion is on the *file name*, not on `family`.
    let fixtures = load_fixtures();
    let mut names: Vec<&str> = fixtures.iter().map(|(_, fx)| fx.name.as_str()).collect();
    names.sort();
    let before = names.len();
    names.dedup();
    assert_eq!(
        before,
        names.len(),
        "duplicate `name` field across fixtures: {names:?}"
    );
}
