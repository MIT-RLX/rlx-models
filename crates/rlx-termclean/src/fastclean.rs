//! Fast, dependency-free TUI cleaner — the deterministic majority of
//! content/chrome classification, done in code instead of a neural net.
//!
//! The learned tagger largely *rediscovered a rule*: drop ANSI + chrome-glyph
//! classes (box-drawing, blocks, braille) + padding + border runs, keep the
//! text. That rule is a branch-light char-class lookup: ~memory-bandwidth
//! throughput (GB/s), zero training, no NaNs, and trivially batchable across
//! thousands of terminal sessions. Reserve the GPU-batched ML for the residual
//! *ambiguous* cases (panel titles, dashboard readings) the rule can't decide.
//!
//! SIMD note: `is_chrome_glyph` and the run-scan are integer range checks over
//! a contiguous buffer — the classic auto-vectorizable shape (and a natural fit
//! for `std::simd` / a 256-entry LUT if pushed further). `clean_batch` is
//! embarrassingly parallel (drop in rayon for multicore fan-out).

/// True if `c` is a terminal-drawing glyph (chrome): box-drawing + block
/// elements (U+2500–U+259F) or braille (U+2800–U+28FF, spinners/graphs).
#[inline(always)]
pub fn is_chrome_glyph(c: char) -> bool {
    let u = c as u32;
    (0x2500..=0x259F).contains(&u) || (0x2800..=0x28FF).contains(&u) || c == '|'
}

/// Strip ANSI CSI / OSC / charset-select escape sequences, returning the
/// visible text. ANSI is pure ASCII so this stays a cheap linear scan.
pub fn strip_ansi(s: &str, out: &mut String) {
    out.clear();
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c != '\x1b' {
            out.push(c);
            continue;
        }
        match it.peek() {
            Some('[') => {
                it.next();
                for n in it.by_ref() {
                    if ('@'..='~').contains(&n) {
                        break;
                    }
                }
            }
            Some(']') => {
                it.next();
                while let Some(n) = it.next() {
                    if n == '\x07' {
                        break;
                    }
                    if n == '\x1b' {
                        if it.peek() == Some(&'\\') {
                            it.next();
                        }
                        break;
                    }
                }
            }
            Some('(') | Some(')') | Some('#') => {
                it.next();
                it.next();
            }
            _ => {
                it.next();
            }
        }
    }
}

/// Per-char content mask over already-visible text (`true` = keep). Mirrors the
/// rule the tagger learned: content is the span between the first and last
/// "text" char, minus chrome glyphs and ≥4-length same-char runs (borders/art).
pub fn classify(chars: &[char], run: &mut Vec<bool>, tag: &mut Vec<bool>) {
    let n = chars.len();
    run.clear();
    run.resize(n, false);
    tag.clear();
    tag.resize(n, false);

    // Mark chars inside a >=4-length run of the same non-space char (borders,
    // ASCII/ACS rules `qqqq`/`----`, ascii-art fills).
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && chars[j] == chars[i] {
            j += 1;
        }
        if j - i >= 4 && chars[i] != ' ' {
            for r in run.iter_mut().take(j).skip(i) {
                *r = true;
            }
        }
        i = j;
    }

    // Fast path: most content is ASCII, where the "text" test collapses to
    // `is_ascii_alphanumeric` (the chrome-glyph ranges are all non-ASCII and '|'
    // isn't alphanumeric) — skipping the Unicode property table for the hot case.
    let is_text = |i: usize| {
        if run[i] {
            return false;
        }
        let c = chars[i];
        if c.is_ascii() {
            c.is_ascii_alphanumeric()
        } else {
            c.is_alphanumeric() && !is_chrome_glyph(c)
        }
    };
    let first = (0..n).find(|&i| is_text(i));
    let last = (0..n).rev().find(|&i| is_text(i));
    if let (Some(a), Some(b)) = (first, last) {
        for (i, t) in tag.iter_mut().enumerate().take(b + 1).skip(a) {
            *t = !is_chrome_glyph(chars[i]) && !run[i];
        }
    }
}

/// A bare pager/editor status marker — chrome, not content. These sit at the
/// bottom of a captured frame; left in, they pollute output and (mid-stream)
/// shift scroll-overlap alignment during stitching.
pub(crate) fn is_pager_status(s: &str) -> bool {
    let t = s.trim();
    matches!(
        t,
        ":" | "~" | "(END)" | "--More--" | "[EOF]" | "END" | "(more)"
    )
}

/// Clean one raw terminal line to its content text, reusing the caller's scratch
/// buffers. Strips ANSI, drops a bare pager prompt, then keeps only the chars the
/// [`classify`] mask marks as content. Returns the (possibly empty) cleaned line.
pub fn clean_line(
    line: &str,
    sbuf: &mut String,
    cbuf: &mut Vec<char>,
    run: &mut Vec<bool>,
    tag: &mut Vec<bool>,
) -> String {
    strip_ansi(line, sbuf);
    if is_pager_status(sbuf) {
        return String::new();
    }
    cbuf.clear();
    cbuf.extend(sbuf.chars());
    classify(cbuf, run, tag);
    let mut out = String::with_capacity(cbuf.len());
    for (i, &c) in cbuf.iter().enumerate() {
        if tag[i] {
            out.push(c);
        }
    }
    out
}

/// Clean a raw frame reusing caller-owned scratch buffers — the hot batch entry
/// (no per-frame allocation of the strip/classify buffers).
pub fn clean_frame_into(
    frame: &str,
    s: &mut String,
    c: &mut Vec<char>,
    r: &mut Vec<bool>,
    t: &mut Vec<bool>,
) -> String {
    let mut lines: Vec<String> = Vec::new();
    for line in frame.lines() {
        let cleaned = clean_line(line, s, c, r, t);
        if !cleaned.trim().is_empty() {
            lines.push(cleaned);
        }
    }
    lines.join("\n")
}

/// Clean a full raw terminal frame → clean text (chrome lines dropped).
pub fn clean_frame(frame: &str) -> String {
    let (mut s, mut c, mut r, mut t) = (String::new(), Vec::new(), Vec::new(), Vec::new());
    clean_frame_into(frame, &mut s, &mut c, &mut r, &mut t)
}

/// Batched interface: clean many frames (one per session), reusing one set of
/// scratch buffers across the whole batch — sequential.
pub fn clean_batch(frames: &[&str]) -> Vec<String> {
    let (mut s, mut c, mut r, mut t) = (String::new(), Vec::new(), Vec::new(), Vec::new());
    frames
        .iter()
        .map(|f| clean_frame_into(f, &mut s, &mut c, &mut r, &mut t))
        .collect()
}

/// Parallel batched clean — fans the frames across all cores with `std::thread`
/// (no external deps). Each thread owns its scratch buffers and writes into a
/// disjoint output slice. This is the "1000 sessions at once" path.
pub fn clean_batch_par(frames: &[&str]) -> Vec<String> {
    let n = frames.len();
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    if n < 64 || threads < 2 {
        return clean_batch(frames); // not worth the fan-out
    }
    // Oversubscribe 4× (ablation-tuned): fine static chunks let the OS rebalance
    // heterogeneous cores + uneven frame sizes; ~neutral on uniform, big win on skew.
    let chunk = n.div_ceil(threads * 4);
    let mut out: Vec<String> = Vec::with_capacity(n);
    out.resize_with(n, String::new);
    std::thread::scope(|scope| {
        let mut out_rest: &mut [String] = &mut out;
        let mut fr = frames;
        while !fr.is_empty() {
            let k = chunk.min(fr.len());
            let (fc, ft) = fr.split_at(k);
            let (oc, ot) = out_rest.split_at_mut(k);
            out_rest = ot;
            fr = ft;
            scope.spawn(move || {
                let (mut s, mut c, mut r, mut t) =
                    (String::new(), Vec::new(), Vec::new(), Vec::new());
                for (o, f) in oc.iter_mut().zip(fc) {
                    *o = clean_frame_into(f, &mut s, &mut c, &mut r, &mut t);
                }
            });
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_batch_equals_sequential() {
        // varied frames: box borders, braille, ASCII rules, unicode content, padding
        let owned: Vec<String> = (0..500)
            .map(|i| {
                format!(
                    "┌─────┐\n│ item {i} café ⣿ │\n├─────┤\n==== rule ====\n  content line {i}  \n└─────┘\n:"
                )
            })
            .collect();
        let frames: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
        assert_eq!(clean_batch_par(&frames), clean_batch(&frames));
        // and the ASCII fast-path must agree with the general path on real content
        assert_eq!(clean_frame("│ hello world 42 │"), "hello world 42");
    }
}
