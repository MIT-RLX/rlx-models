#!/usr/bin/env python3
"""Render conversations with Flint's own chat_template.jinja (through
transformers, the way SGLang applies it) and write them as a fixture for
crates/rlx-flint/tests/chat_template.rs.

Tool declarations and call arguments are stored as JSON *text* so the Rust
side reads them with their key order intact.

    python3 scripts/flint_chat_reference.py weights/lm/flint \
        crates/rlx-flint/tests/fixtures/chat_template.json
"""

import json
import sys
from pathlib import Path

from transformers.utils.chat_template_utils import render_jinja_template

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"],
        },
    },
}
# A bare function schema (no "function" wrapper), unsorted keys, non-ASCII,
# escapes and floats — everything `tojson` has to reproduce.
SEARCH = {
    "name": "search",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Suchbegriff — \"exakt\"\n"},
            "limit": {"type": "number", "default": 0.5, "maximum": 1e20, "minimum": 1e-7},
            "filters": {"type": "object"},
        },
    },
    "description": "Search <|user|> docs",
}

CASES = [
    ("tools_only", [WEATHER], [], True),
    (
        "one_call_round_trip",
        [WEATHER],
        [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "use the tool",
                "tool_calls": [
                    {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}}
                ],
            },
            {"role": "tool", "content": "sunny <|assistant|> 21°C"},
        ],
        True,
    ),
    (
        "system_before_tools",
        [WEATHER, SEARCH],
        [
            {"role": "system", "content": "Be brief."},
            {"role": "system", "content": "Really."},
            {"role": "developer", "content": "dev note"},
            {"role": "user", "content": "Find docs"},
            {
                "role": "assistant",
                "content": "Searching.",
                "tool_calls": [
                    {
                        "name": "search",
                        "arguments": {
                            "query": "a \"b\"",
                            "limit": 0.25,
                            "filters": {"lang": "de", "tags": ["x", 1, True, None]},
                        },
                    },
                    {"name": "get_weather", "arguments": {"city": "Zürich"}},
                ],
            },
            {"role": "tool", "content": "[]"},
            {"role": "tool", "content": "rain"},
            {"role": "assistant", "content": "Nothing found; rain."},
        ],
        False,
    ),
    ("system_only_with_tools", [SEARCH], [{"role": "system", "content": "S"}], True),
    ("no_tools", [], [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}], True),
]


def main() -> None:
    model_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    template = (model_dir / "chat_template.jinja").read_text()
    cases = []
    for name, tools, messages, gen in CASES:
        rendered, _ = render_jinja_template(
            conversations=[messages],
            tools=tools or None,
            chat_template=template,
            add_generation_prompt=gen,
        )
        stored = []
        for m in messages:
            s = {"role": m["role"], "content": m["content"]}
            if m.get("reasoning_content"):
                s["reasoning"] = m["reasoning_content"]
            calls = []
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", tc)
                calls.append({"name": fn["name"], "arguments": json.dumps(fn["arguments"], ensure_ascii=False)})
            if calls:
                s["tool_calls"] = calls
            stored.append(s)
        cases.append(
            {
                "name": name,
                "tools": json.dumps(tools, ensure_ascii=False),
                "messages": stored,
                "add_generation_prompt": gen,
                "expected": rendered[0],
            }
        )
    out.write_text(json.dumps({"template": "chat_template.jinja", "cases": cases}, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
