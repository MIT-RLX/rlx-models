//! Synthetic TUI renderers.
//!
//! Each renderer takes clean content, lays it out as a realistic terminal
//! screen, and records provenance as it goes (via [`Screen`]) so it can emit
//! the rendered `input`, a per-char `tags` string, and the clean `target`
//! together. The corruption function is known, so labels are exact — no human
//! annotation.
//!
//! Faithfulness rule: we never place content in `target` that isn't visible on
//! screen. Truncated (`…`) text keeps only the visible prefix, so the model is
//! never trained to hallucinate cut-off characters.

use crate::corpus;
use crate::record::{Sample, Tag};
use crate::rng::Rng;
use crate::symbols::*;

/// A screen under construction. Appends chars to `input` while pushing one
/// tag marker per char to `tags`, keeping them length-aligned by construction.
struct Screen {
    input: String,
    tags: String,
    ansi: bool,
}

impl Screen {
    fn new() -> Self {
        Screen {
            input: String::new(),
            tags: String::new(),
            ansi: false,
        }
    }

    fn raw(&mut self, s: &str, t: Tag) {
        let m = t.marker();
        for _ in s.chars() {
            self.tags.push(m);
        }
        self.input.push_str(s);
    }

    fn content(&mut self, s: &str) {
        self.raw(s, Tag::Content);
    }
    fn chrome(&mut self, s: &str) {
        self.raw(s, Tag::Chrome);
    }
    fn nl(&mut self) {
        self.raw("\n", Tag::Chrome);
    }
    fn pad(&mut self, n: usize) {
        for _ in 0..n {
            self.chrome(" ");
        }
    }
    fn ansi_open(&mut self, esc: &str) {
        self.ansi = true;
        self.chrome(esc);
    }
    fn ansi_reset(&mut self) {
        self.chrome("\x1b[0m");
    }

    fn finish(
        self,
        id: u64,
        kind: &'static str,
        content_type: &'static str,
        width: usize,
        style: String,
        target: String,
    ) -> Sample {
        debug_assert_eq!(
            self.input.chars().count(),
            self.tags.chars().count(),
            "input/tags char-length must match"
        );
        Sample {
            id,
            kind,
            content_type,
            width,
            style,
            ansi: self.ansi,
            input: self.input,
            target,
            tags: self.tags,
        }
    }
}

/// A styled run of text on one line (optional SGR params for coloring).
struct Run {
    text: String,
    style: Option<&'static str>,
}

fn line_width(l: &[Run]) -> usize {
    l.iter().map(|r| r.text.chars().count()).sum()
}

fn push_rich(scr: &mut Screen, l: &[Run]) {
    for r in l {
        match r.style {
            Some(p) => {
                scr.ansi_open(&format!("\x1b[{p}m"));
                scr.content(&r.text);
                scr.ansi_reset();
            }
            None => scr.content(&r.text),
        }
    }
}

/// Greedy word wrap to `width` columns. Each returned inner vec is one line's
/// words; rejoining all words with single spaces reproduces the source exactly.
fn wrap(words: &[String], width: usize) -> Vec<Vec<String>> {
    let mut lines: Vec<Vec<String>> = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    let mut cl = 0usize;
    for w in words {
        let wl = w.chars().count();
        if !cur.is_empty() && cl + 1 + wl > width {
            lines.push(std::mem::take(&mut cur));
            cl = 0;
        }
        if cur.is_empty() {
            cl = wl;
        } else {
            cl += 1 + wl;
        }
        cur.push(w.clone());
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    if lines.is_empty() {
        lines.push(Vec::new());
    }
    lines
}

/// Draw a horizontal table/panel border row for the given column inner widths.
fn border(scr: &mut Screen, widths: &[usize], left: &str, mid: &str, right: &str, h: &str) {
    scr.chrome(left);
    for (i, wc) in widths.iter().enumerate() {
        if i > 0 {
            scr.chrome(mid);
        }
        for _ in 0..(wc + 2) {
            scr.chrome(h);
        }
    }
    scr.chrome(right);
    scr.nl();
}

// ---------------------------------------------------------------------------
// Layout renderers
// ---------------------------------------------------------------------------

/// A bordered box around wrapped prose / code / log / key-value content.
fn panel(rng: &mut Rng, id: u64) -> Sample {
    let style = *rng.pick(BOX_STYLES);
    let styled = rng.chance(0.5);
    let mut scr = Screen::new();

    let lines: Vec<Vec<Run>>;
    let content_type: &'static str;
    let target: String;

    match rng.below(4) {
        0 => {
            let np = rng.range(2, 5);
            let (words, para) = corpus::paragraph(rng, np);
            let w = rng.range(28, 60);
            let mut ls = Vec::new();
            for lw in wrap(&words, w) {
                let text = lw.join(" ");
                let st = if styled {
                    Some(rng.pick_str(SGR_PARAMS))
                } else {
                    None
                };
                ls.push(vec![Run { text, style: st }]);
            }
            lines = ls;
            content_type = "prose";
            target = para;
        }
        1 => {
            let nl = rng.range(3, 8);
            let cl = corpus::code_lines(rng, nl);
            let mut ls = Vec::new();
            let mut t = String::new();
            for (i, c) in cl.iter().enumerate() {
                let st = if styled { Some("36") } else { None };
                ls.push(vec![Run {
                    text: c.clone(),
                    style: st,
                }]);
                if i > 0 {
                    t.push('\n');
                }
                t.push_str(c);
            }
            lines = ls;
            content_type = "code";
            target = t;
        }
        2 => {
            let nl = rng.range(3, 8);
            let ll = corpus::log_lines(rng, nl);
            let mut ls = Vec::new();
            let mut t = String::new();
            for (i, l) in ll.iter().enumerate() {
                ls.push(vec![Run {
                    text: l.clone(),
                    style: None,
                }]);
                if i > 0 {
                    t.push('\n');
                }
                t.push_str(l);
            }
            lines = ls;
            content_type = "log";
            target = t;
        }
        _ => {
            let nk = rng.range(3, 7);
            let kv = corpus::kv_pairs(rng, nk);
            let mut ls = Vec::new();
            let mut t = String::new();
            for (i, (k, v)) in kv.iter().enumerate() {
                let text = format!("{k}: {v}");
                if i > 0 {
                    t.push('\n');
                }
                t.push_str(&text);
                ls.push(vec![Run { text, style: None }]);
            }
            lines = ls;
            content_type = "kv";
            target = t;
        }
    }

    let inner_content = lines.iter().map(|l| line_width(l)).max().unwrap_or(0);
    let pad_x = 1usize;
    let inner = inner_content + pad_x * 2;

    scr.chrome(style.tl);
    for _ in 0..inner {
        scr.chrome(style.h);
    }
    scr.chrome(style.tr);
    scr.nl();
    for l in &lines {
        scr.chrome(style.v);
        scr.pad(pad_x);
        let w = line_width(l);
        push_rich(&mut scr, l);
        scr.pad(inner_content - w + pad_x);
        scr.chrome(style.v);
        scr.nl();
    }
    scr.chrome(style.bl);
    for _ in 0..inner {
        scr.chrome(style.h);
    }
    scr.chrome(style.br);
    scr.nl();

    scr.finish(
        id,
        "panel",
        content_type,
        inner + 2,
        format!("box={}", style.name),
        target,
    )
}

/// Borderless hard-wrapped prose, with an occasional hyphenated line break.
/// This is the core reflow case: the model must rejoin words split across the
/// chrome newline (and rejoin a hyphenated word across the chrome `-`).
fn wrap_layout(rng: &mut Rng, id: u64) -> Sample {
    let np = rng.range(2, 6);
    let (words, para) = corpus::paragraph(rng, np);
    let width = rng.range(24, 56);
    let mut lines = wrap(&words, width);

    // Optionally hyphenate one interior break: move a prefix of the first word
    // on the next line up to the current line, followed by a chrome '-'.
    let mut hyph_head: Option<(usize, String)> = None;
    if lines.len() > 1 && rng.chance(0.35) {
        let hb = rng.below(lines.len() - 1);
        if let Some(fw) = lines[hb + 1].first().cloned() {
            if fw.is_ascii() && fw.len() >= 4 {
                let k = rng.range(2, fw.len() - 1);
                let head = fw[..k].to_string();
                let tail = fw[k..].to_string();
                lines[hb + 1][0] = tail;
                hyph_head = Some((hb, head));
            }
        }
    }

    let mut scr = Screen::new();
    let n = lines.len();
    for (i, l) in lines.iter().enumerate() {
        for (j, w) in l.iter().enumerate() {
            if j > 0 {
                scr.content(" ");
            }
            scr.content(w);
        }
        if let Some((hb, head)) = &hyph_head {
            if *hb == i {
                scr.content(" ");
                scr.content(head);
                scr.chrome("-");
            }
        }
        if i + 1 < n {
            scr.nl();
        }
    }

    scr.finish(id, "wrap", "prose", width, "none".into(), para)
}

/// Lines truncated with `…` when they overflow the width. Target keeps only the
/// visible prefix (never the cut-off tail).
fn truncate_layout(rng: &mut Rng, id: u64) -> Sample {
    let width = rng.range(28, 52);
    let n = rng.range(3, 7);
    let mut scr = Screen::new();
    let mut t = String::new();
    for i in 0..n {
        let s = corpus::sentence(rng);
        let chars: Vec<char> = s.chars().collect();
        if i > 0 {
            t.push('\n');
        }
        if chars.len() > width {
            let keep: String = chars[..width - 1].iter().collect();
            let kt = keep.trim_end().to_string();
            scr.content(&keep);
            scr.chrome(ELLIPSIS);
            t.push_str(&kt);
        } else {
            scr.content(&s);
            t.push_str(&s);
        }
        scr.nl();
    }
    scr.finish(id, "truncate", "prose", width, "clip".into(), t)
}

/// A two-column side-by-side split (tmux/vim style). Reading order for the
/// target is left panel fully, then right — so the model must de-interleave the
/// row-interleaved input.
fn split_layout(rng: &mut Rng, id: u64) -> Sample {
    let cw = rng.range(16, 30);
    let style = *rng.pick(BOX_STYLES);
    let nl = rng.range(2, 4);
    let (lw, lp) = corpus::paragraph(rng, nl);
    let nr = rng.range(2, 4);
    let (rw, rp) = corpus::paragraph(rng, nr);
    let ll = wrap(&lw, cw);
    let rl = wrap(&rw, cw);
    let rows = ll.len().max(rl.len());
    let widths = [cw, cw];

    let mut scr = Screen::new();
    border(&mut scr, &widths, style.tl, style.ttee, style.tr, style.h);
    for r in 0..rows {
        // left cell
        scr.chrome(style.v);
        scr.chrome(" ");
        let lwid = push_cell(&mut scr, ll.get(r));
        scr.pad(cw - lwid);
        scr.chrome(" ");
        // divider
        scr.chrome(style.v);
        scr.chrome(" ");
        let rwid = push_cell(&mut scr, rl.get(r));
        scr.pad(cw - rwid);
        scr.chrome(" ");
        scr.chrome(style.v);
        scr.nl();
    }
    border(&mut scr, &widths, style.bl, style.btee, style.br, style.h);

    scr.finish(
        id,
        "split",
        "prose",
        cw * 2 + 7,
        format!("box={}", style.name),
        format!("{lp}\n{rp}"),
    )
}

/// Push one wrapped line's words as content, returning the visible width.
fn push_cell(scr: &mut Screen, line: Option<&Vec<String>>) -> usize {
    let mut w = 0;
    if let Some(words) = line {
        for (j, word) in words.iter().enumerate() {
            if j > 0 {
                scr.content(" ");
                w += 1;
            }
            scr.content(word);
            w += word.chars().count();
        }
    }
    w
}

/// A vim/nano-style screen: reverse-video status bar, content body, help bar.
/// Both bars (text included) are chrome; only the body survives.
fn statusbar_layout(rng: &mut Rng, id: u64) -> Sample {
    let width = rng.range(40, 72);
    let mut scr = Screen::new();

    let mode = rng.pick_str(&["NORMAL", "INSERT", "VISUAL", "COMMAND"]);
    let fname = corpus::filename(rng);
    let mut top = format!(" {mode}  {fname} ");
    fit_width(&mut top, width);
    scr.ansi_open("\x1b[7m");
    scr.chrome(&top);
    scr.ansi_reset();
    scr.nl();

    let np = rng.range(2, 4);
    let (words, para) = corpus::paragraph(rng, np);
    for l in wrap(&words, width) {
        for (j, w) in l.iter().enumerate() {
            if j > 0 {
                scr.content(" ");
            }
            scr.content(w);
        }
        scr.nl();
    }

    let mut help = String::from(" ^S Save   ^Q Quit   ^F Find   ^G Help ");
    fit_width(&mut help, width);
    scr.ansi_open("\x1b[7m");
    scr.chrome(&help);
    scr.ansi_reset();
    scr.nl();

    scr.finish(id, "statusbar", "prose", width, "editor".into(), para)
}

/// Pad with spaces or hard-truncate `s` to exactly `width` chars.
fn fit_width(s: &mut String, width: usize) {
    let n = s.chars().count();
    if n < width {
        s.push_str(&" ".repeat(width - n));
    } else if n > width {
        *s = s.chars().take(width).collect();
    }
}

static KEYWORDS: &[&str] = &[
    "let", "fn", "for", "if", "else", "const", "while", "match", "return", "pub", "use", "impl",
    "struct", "mut", "loop", "enum",
];

fn push_code(scr: &mut Screen, code: &str, highlight: bool) {
    if !highlight {
        scr.content(code);
        return;
    }
    // Keep indentation (content), then color a leading keyword if present.
    let indent_len = code.len() - code.trim_start().len();
    let (indent, rest) = code.split_at(indent_len);
    scr.content(indent);
    let fe = rest.find(' ').unwrap_or(rest.len());
    let (tok, tail) = rest.split_at(fe);
    if KEYWORDS.contains(&tok) {
        scr.ansi_open("\x1b[35m");
        scr.content(tok);
        scr.ansi_reset();
    } else {
        scr.content(tok);
    }
    scr.content(tail);
}

/// An editor code view with a right-aligned line-number gutter. The gutter
/// (numbers + separator + padding) is chrome; the code (indentation included)
/// is content.
fn code_layout(rng: &mut Rng, id: u64) -> Sample {
    let nl = rng.range(4, 11);
    let lines = corpus::code_lines(rng, nl);
    let num_w = format!("{}", lines.len()).len();
    let gutter = rng.pick_str(&["│", "┃", "|", "▏", "┆"]);
    let highlight = rng.chance(0.5);
    let mut scr = Screen::new();
    let mut t = String::new();
    for (i, c) in lines.iter().enumerate() {
        let num = format!("{:>num_w$}", i + 1);
        scr.chrome(" ");
        scr.chrome(&num);
        scr.chrome(" ");
        scr.chrome(gutter);
        scr.chrome(" ");
        push_code(&mut scr, c, highlight);
        scr.nl();
        if i > 0 {
            t.push('\n');
        }
        t.push_str(c);
    }
    scr.finish(id, "code", "code", num_w + 3, "gutter".into(), t)
}

/// An aligned table with box-drawing separators and padded cells.
fn table_layout(rng: &mut Rng, id: u64) -> Sample {
    let (header, rows) = corpus::table(rng);
    let ncol = header.len();
    let mut w = vec![0usize; ncol];
    for (c, wc) in w.iter_mut().enumerate() {
        *wc = header[c].chars().count();
        for r in &rows {
            *wc = (*wc).max(r[c].chars().count());
        }
    }
    let style = *rng.pick(BOX_STYLES);
    let bold = rng.chance(0.6);
    let mut scr = Screen::new();

    border(&mut scr, &w, style.tl, style.ttee, style.tr, style.h);
    draw_cells(&mut scr, &style, &w, &header, bold);
    border(&mut scr, &w, style.ltee, style.cross, style.rtee, style.h);
    for r in &rows {
        draw_cells(&mut scr, &style, &w, r, false);
    }
    border(&mut scr, &w, style.bl, style.btee, style.br, style.h);

    // Target: cells joined by tab (an unambiguous clean delimiter), rows by \n.
    let mut t = header.join("\t");
    for r in &rows {
        t.push('\n');
        t.push_str(&r.join("\t"));
    }
    let width = w.iter().sum::<usize>() + ncol * 3 + 1;
    scr.finish(
        id,
        "table",
        "table",
        width,
        format!("box={}", style.name),
        t,
    )
}

fn draw_cells(scr: &mut Screen, style: &BoxStyle, widths: &[usize], cells: &[String], bold: bool) {
    for (i, c) in cells.iter().enumerate() {
        scr.chrome(style.v);
        scr.chrome(" ");
        if bold {
            scr.ansi_open("\x1b[1m");
            scr.content(c);
            scr.ansi_reset();
        } else {
            scr.content(c);
        }
        scr.pad(widths[i] - c.chars().count());
        scr.chrome(" ");
    }
    scr.chrome(style.v);
    scr.nl();
}

/// Colon-aligned key/value pairs. The alignment padding before the colon is
/// chrome; the target collapses to a single `key: value` spacing.
fn keyvalue_layout(rng: &mut Rng, id: u64) -> Sample {
    let nk = rng.range(3, 8);
    let kv = corpus::kv_pairs(rng, nk);
    let kw = kv.iter().map(|(k, _)| k.chars().count()).max().unwrap_or(0);
    let styled = rng.chance(0.4);
    let mut scr = Screen::new();
    let mut t = String::new();
    for (i, (k, v)) in kv.iter().enumerate() {
        if styled {
            scr.ansi_open("\x1b[36m");
            scr.content(k);
            scr.ansi_reset();
        } else {
            scr.content(k);
        }
        // alignment pad + the space before the colon are chrome
        scr.chrome(&" ".repeat(kw - k.chars().count() + 1));
        scr.content(":");
        scr.content(" ");
        scr.content(v);
        scr.nl();
        if i > 0 {
            t.push('\n');
        }
        t.push_str(&format!("{k}: {v}"));
    }
    scr.finish(id, "keyvalue", "kv", kw + 2, "aligned".into(), t)
}

/// A bullet list. The bullet glyph + indent are chrome; the target normalizes
/// every item to a `- ` prefix.
fn list_layout(rng: &mut Rng, id: u64) -> Sample {
    let n = rng.range(3, 8);
    let items = corpus::list_items(rng, n);
    let bullet = rng.pick_str(BULLETS);
    let indent = rng.range(0, 4);
    let mut scr = Screen::new();
    let mut t = String::new();
    for (i, it) in items.iter().enumerate() {
        scr.chrome(&" ".repeat(indent));
        scr.chrome(bullet);
        scr.chrome(" ");
        scr.content(it);
        scr.nl();
        if i > 0 {
            t.push('\n');
        }
        t.push_str("- ");
        t.push_str(it);
    }
    scr.finish(id, "list", "list", indent, format!("bullet={bullet}"), t)
}

/// Progress bars and spinners with textual labels. Bars, blocks, spinner
/// frames, and percentages are chrome; the label text is content.
fn progress_layout(rng: &mut Rng, id: u64) -> Sample {
    let n = rng.range(1, 4);
    let mut scr = Screen::new();
    let mut t = String::new();
    for i in 0..n {
        let label = corpus::label(rng);
        if rng.chance(0.5) {
            let frames = *rng.pick(SPINNERS);
            scr.chrome(frames[rng.below(frames.len())]);
            scr.chrome(" ");
            scr.content(&label);
            if rng.chance(0.5) {
                scr.chrome(ELLIPSIS);
            }
        } else {
            scr.content(&label);
            scr.content(" ");
            scr.chrome("[");
            let total = rng.range(10, 26);
            let filled = rng.below(total + 1);
            for _ in 0..filled {
                scr.chrome(BAR_FULL);
            }
            for _ in 0..(total - filled) {
                scr.chrome(rng.pick_str(&["░", "▒", " "]));
            }
            scr.chrome("]");
            scr.chrome(&format!(" {}%", filled * 100 / total));
        }
        scr.nl();
        if i > 0 {
            t.push('\n');
        }
        t.push_str(&label);
    }
    scr.finish(id, "progress", "label", 0, "bar".into(), t)
}

/// A tab bar (one active tab bracketed) above body text — the "tabs" UI state.
/// Tab labels and body are content; brackets, separators and the rule are chrome.
fn tab_layout(rng: &mut Rng, id: u64) -> Sample {
    let ntabs = rng.range(2, 5);
    let labels: Vec<String> = (0..ntabs).map(|_| corpus::label(rng)).collect();
    let active = rng.below(ntabs);
    let mut scr = Screen::new();
    let mut t = String::new();
    for (i, lab) in labels.iter().enumerate() {
        scr.chrome(" ");
        scr.chrome(if i == active { "[" } else { " " });
        scr.content(lab);
        scr.chrome(if i == active { "]" } else { " " });
        if i + 1 < labels.len() {
            scr.chrome(rng.pick_str(&["│", "|", " "]));
        }
        if i > 0 {
            t.push(' ');
        }
        t.push_str(lab);
    }
    scr.nl();
    let rule = rng.range(20, 40);
    scr.chrome(&"─".repeat(rule));
    scr.nl();
    let nlines = rng.range(2, 5);
    let (lines, _) = corpus::paragraph(rng, nlines);
    for line in &lines {
        scr.content(line);
        scr.nl();
        t.push('\n');
        t.push_str(line);
    }
    scr.finish(id, "tab", "tab", active, format!("tabs={ntabs}"), t)
}

/// A scrollable list with a scrollbar gutter and a "N more" indicator — the
/// "scroll" UI state. Items are content; the scrollbar track/thumb and the
/// more-indicator are chrome.
fn scroll_layout(rng: &mut Rng, id: u64) -> Sample {
    let n = rng.range(4, 9);
    let items = corpus::list_items(rng, n);
    let thumb = rng.below(n);
    let gutter = rng.range(24, 40);
    let mut scr = Screen::new();
    let mut t = String::new();
    for (i, it) in items.iter().enumerate() {
        scr.content(it);
        let fill = gutter.saturating_sub(it.chars().count());
        scr.chrome(&" ".repeat(fill + 1));
        scr.chrome(if i == thumb {
            "█"
        } else {
            rng.pick_str(&["│", "░", "▏"])
        });
        scr.nl();
        if i > 0 {
            t.push('\n');
        }
        t.push_str(it);
    }
    let arrow = rng.pick_str(&["↓", "▼", "v"]);
    let more = rng.range(3, 40);
    scr.chrome(&format!("{arrow} {more} more "));
    scr.nl();
    scr.finish(id, "scroll", "list", thumb, "scroll".into(), t)
}

/// Pick a layout by weight and render one sample.
pub fn generate(rng: &mut Rng, id: u64) -> Sample {
    match rng.below(100) {
        0..=15 => panel(rng, id),
        16..=29 => wrap_layout(rng, id),
        30..=40 => table_layout(rng, id),
        41..=50 => keyvalue_layout(rng, id),
        51..=59 => list_layout(rng, id),
        60..=67 => code_layout(rng, id),
        68..=75 => statusbar_layout(rng, id),
        76..=81 => truncate_layout(rng, id),
        82..=86 => split_layout(rng, id),
        87..=90 => progress_layout(rng, id),
        91..=95 => tab_layout(rng, id), // "tabs" interaction state
        _ => scroll_layout(rng, id),    // "scroll" interaction state
    }
}
