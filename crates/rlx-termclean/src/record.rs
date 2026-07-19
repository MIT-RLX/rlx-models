//! Dataset record types and a minimal, dependency-free JSONL serializer.

use std::fmt::Write as _;

/// Per-character provenance tag for the auxiliary content/chrome head.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tag {
    /// Real content that must be kept (letters, code, punctuation, unicode…).
    Content,
    /// Terminal chrome to drop (borders, ANSI, padding, gutters, bars…).
    Chrome,
}

impl Tag {
    /// The single ASCII marker written into the `tags` string.
    pub fn marker(self) -> char {
        match self {
            Tag::Content => 'C',
            Tag::Chrome => 'X',
        }
    }
}

/// One training example.
///
/// Invariant: `input.chars().count() == tags.chars().count()`, and every char
/// of `tags` is `'C'` or `'X'`. `target` is the clean, reflowed text the model
/// should produce; it is *not* required to be char-aligned to `input` (reflow,
/// reordering, and padding-collapse break 1:1 alignment on purpose).
pub struct Sample {
    pub id: u64,
    /// Layout family, e.g. `"panel"`, `"table"`, `"split"`.
    pub kind: &'static str,
    /// Dominant content type, e.g. `"prose"`, `"code"`, `"log"`, `"table"`.
    pub content_type: &'static str,
    /// A coarse width metric for the rendered screen (stratification aid).
    pub width: usize,
    /// Human-readable descriptor of the visual style (box family, theme…).
    pub style: String,
    /// Whether the rendered screen carries ANSI escape sequences.
    pub ansi: bool,
    /// The raw rendered screen: chrome + content + ANSI, exactly as a terminal
    /// would emit it.
    pub input: String,
    /// The clean target text.
    pub target: String,
    /// One marker char per `input` char (`C` = content, `X` = chrome).
    pub tags: String,
}

/// Escape `s` into a JSON string literal (surrounding quotes included).
///
/// All inputs are valid UTF-8 `&str`; box-drawing glyphs pass through verbatim,
/// while control bytes (notably ESC `0x1b` from ANSI sequences) are `\uXXXX`
/// escaped so the JSONL stays single-line and portable.
pub fn json_escape(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Serialize one `Sample` as a single JSON object (no trailing newline).
pub fn write_record(s: &Sample, out: &mut String) {
    out.push_str("{\"id\":");
    let _ = write!(out, "{}", s.id);
    out.push_str(",\"kind\":");
    json_escape(s.kind, out);
    out.push_str(",\"content_type\":");
    json_escape(s.content_type, out);
    out.push_str(",\"width\":");
    let _ = write!(out, "{}", s.width);
    out.push_str(",\"ansi\":");
    out.push_str(if s.ansi { "true" } else { "false" });
    out.push_str(",\"style\":");
    json_escape(&s.style, out);
    out.push_str(",\"input\":");
    json_escape(&s.input, out);
    out.push_str(",\"target\":");
    json_escape(&s.target, out);
    out.push_str(",\"tags\":");
    json_escape(&s.tags, out);
    out.push('}');
}
