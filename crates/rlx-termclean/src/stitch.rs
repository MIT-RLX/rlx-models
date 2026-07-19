//! Whole-document reconstruction from overlapping scrolled frames.
//!
//! A scrolling TUI (a pager, a log tail, a file list) shows a moving *window*
//! over a larger document; consecutive captures overlap by all-but-a-few lines.
//! To recover the whole document you must **stitch**: align each new frame
//! against the tail of what you've accumulated by its scroll overlap, then append
//! only the newly-revealed lines — which **deduplicates** the repeated overlap.
//!
//! This is the counterpart to per-frame cleaning: `raw frames → clean_frame →
//! per-frame content lines → stitch → whole document`. It's rule-shaped and
//! pure-std (no ML, no GPU): integer line comparisons over the frame tail.

use crate::fastclean;

/// Trailing-whitespace-insensitive view of a line (captures pad to the border).
fn norm(s: &str) -> &str {
    s.trim_end()
}

/// Frames may carry a status bar (pager prompt, filename) at the bottom that
/// survives cleaning; allow a join to skip up to this many trailing status lines
/// (of `out` for a downward join, of `f` for an upward join), which are then
/// dropped rather than allowed to eat real content lines.
const SLACK: usize = 2;

/// A strong overlap: real content present and ≥80% of the window's lines equal
/// (so a single highlighted "current line" doesn't break the alignment).
fn strong(hits: i32, cmp: i32, nonblank: i32) -> bool {
    nonblank >= 1 && cmp > 0 && hits * 100 >= cmp * 80
}

/// (matching lines, matching non-blank lines) between two equal-length windows.
fn score_overlap(a: &[String], b: &[String]) -> (i32, i32) {
    let (mut hits, mut nonblank) = (0i32, 0i32);
    for (x, y) in a.iter().zip(b) {
        if norm(x) == norm(y) {
            hits += 1;
            if !norm(x).is_empty() {
                nonblank += 1;
            }
        }
    }
    (hits, nonblank)
}

/// FORWARD join (scrolled down): `f` extends `out` at the bottom. Longest suffix
/// of `out` (skipping ≤SLACK trailing status lines) matching a prefix of `f`.
/// Returns (overlap_len, out_keep_len, net_score); caller appends `f[overlap..]`.
/// Net score (hits − misses) prefers the exact alignment over a junk-shifted one.
fn forward(out: &[String], f: &[String]) -> Option<(usize, usize, i32)> {
    let mut best: Option<(usize, usize, i32)> = None;
    for s in 0..=SLACK.min(out.len().saturating_sub(1)) {
        let avail = out.len() - s;
        let max_m = avail.min(f.len());
        for m in 1..=max_m {
            let (hits, nb) = score_overlap(&out[avail - m..avail], &f[..m]);
            let net = 2 * hits - m as i32;
            if strong(hits, m as i32, nb) && best.is_none_or(|(_, _, bn)| net > bn) {
                best = Some((m, avail, net));
            }
        }
        // A clean long overlap at this slack won't be beaten by skipping more
        // trailing junk — stop paying for higher slack (the common no-junk case).
        if best.is_some_and(|(_, _, net)| net >= 8) {
            break;
        }
    }
    best
}

/// BACKWARD join (scrolled up): `f` extends `out` at the top. Longest prefix of
/// `out` matching a suffix of `f` (skipping ≤SLACK trailing status lines of `f`).
/// Returns (prepend_len, overlap_len, net_score); caller prepends `f[..prepend_len]`
/// and the overlap `f[prepend_len..prepend_len+overlap]` aligns with `out[..overlap]`.
fn backward(out: &[String], f: &[String]) -> Option<(usize, usize, i32)> {
    let mut best: Option<(usize, usize, i32)> = None;
    for s in 0..=SLACK.min(f.len().saturating_sub(1)) {
        let favail = f.len() - s;
        let max_k = out.len().min(favail);
        for k in 1..=max_k {
            let (hits, nb) = score_overlap(&out[..k], &f[favail - k..favail]);
            let net = 2 * hits - k as i32;
            if strong(hits, k as i32, nb) && best.is_none_or(|(_, _, bn)| net > bn) {
                best = Some((favail - k, k, net)); // f[..favail-k] are the new top lines
            }
        }
    }
    best
}

/// If `f` (minus ≤SLACK trailing status lines) is already a contiguous run inside
/// `out`, return where it matches (start, len) — a revisit of seen content. The
/// caller adds no new lines but can still use it to VOTE on the matched region.
fn contained_at(out: &[String], f: &[String]) -> Option<(usize, usize)> {
    for s in 0..=SLACK.min(f.len().saturating_sub(1)) {
        let fe = &f[..f.len() - s];
        if fe.len() < 2 || fe.len() > out.len() {
            continue;
        }
        for start in 0..=out.len() - fe.len() {
            let (hits, nb) = score_overlap(&out[start..start + fe.len()], fe);
            if strong(hits, fe.len() as i32, nb) {
                return Some((start, fe.len()));
            }
        }
    }
    None
}

/// The structured alignment decision — shared by the plain merge and the voting
/// stitcher so both agree on where a frame's lines map onto the document.
enum Merge {
    /// `f` extends the bottom: `out[keep-overlap..keep]` ↔ `f[..overlap]`,
    /// `f[overlap..]` are new; any `out[keep..]` (status junk) is dropped.
    Append { keep: usize, overlap: usize },
    /// `f` extends the top: `f[..new]` are new, `f[new..new+overlap]` ↔ `out[..overlap]`.
    Prepend { new: usize, overlap: usize },
    /// `f[..len]` revisits `out[at..at+len]` — no new lines.
    Contained { at: usize, len: usize },
    /// unrelated screen — append `f` whole.
    Jump,
}

/// Decide how frame `f` aligns onto `out` (pure — no mutation).
fn plan_merge(out: &[String], f: &[String]) -> Merge {
    let fwd = forward(out, f);
    // Fast path: a clearly-strong forward overlap means a downward scroll (common);
    // skip the backward scan. Only when forward is absent/weak do we pay for it.
    let bwd = if fwd.is_none_or(|(_, _, net)| net < 4) {
        backward(out, f)
    } else {
        None
    };
    match (fwd, bwd) {
        (None, None) => match contained_at(out, f) {
            Some((at, len)) => Merge::Contained { at, len },
            None => Merge::Jump,
        },
        _ => {
            let fs = fwd.map_or(i32::MIN, |(_, _, s)| s);
            let bs = bwd.map_or(i32::MIN, |(_, _, s)| s);
            if fs >= bs {
                let (overlap, keep, _) = fwd.unwrap();
                Merge::Append { keep, overlap }
            } else {
                let (new, overlap, _) = bwd.unwrap();
                Merge::Prepend { new, overlap }
            }
        }
    }
}

/// Merge frame `f` into the accumulated document `out`, in EITHER scroll
/// direction: append at the bottom (scrolled down), prepend at the top (scrolled
/// up), skip a revisit of already-seen lines, or append an unrelated jump. The
/// stronger-scoring of the forward/backward joins wins.
pub fn merge_frame(out: &mut Vec<String>, f: &[String]) {
    if out.is_empty() {
        out.extend(f.iter().cloned());
        return;
    }
    match plan_merge(out, f) {
        Merge::Append { keep, overlap } => {
            out.truncate(keep); // drop skipped trailing status lines
            out.extend(f[overlap..].iter().cloned());
        }
        Merge::Prepend { new, .. } => {
            let mut head: Vec<String> = f[..new].to_vec();
            head.append(out);
            *out = head;
        }
        Merge::Contained { .. } => {} // revisit — nothing new
        Merge::Jump => out.extend(f.iter().cloned()),
    }
}

/// Per-column majority across equal-length line versions — recovers the true line
/// even when no single copy is clean, as long as each column has a majority.
fn char_vote(rows: &[&str]) -> String {
    let cols: Vec<Vec<char>> = rows.iter().map(|s| s.chars().collect()).collect();
    let width = cols.iter().map(|c| c.len()).max().unwrap_or(0);
    let mut out = String::new();
    for col in 0..width {
        let mut tally: Vec<(char, u32)> = Vec::new();
        for c in &cols {
            if let Some(&ch) = c.get(col) {
                match tally.iter_mut().find(|(k, _)| *k == ch) {
                    Some(e) => e.1 += 1,
                    None => tally.push((ch, 1)),
                }
            }
        }
        if let Some(&(ch, _)) = tally.iter().max_by_key(|(_, n)| *n) {
            out.push(ch);
        }
    }
    out
}

/// Resolve the true line from all observed versions across overlapping frames.
/// A transient glitch (cursor artifact, partial redraw, a mis-cleaned char) sits
/// in a minority of copies: a strict line-level majority votes it out; failing
/// that, equal-length copies are voted column-by-column (per-char glitches). A
/// length mismatch signals an insertion/shift (e.g. a "> " cursor prefix), which
/// per-column voting would garble — so there we preserve the first (reference) copy.
fn consensus(votes: &[String]) -> String {
    if votes.len() == 1 {
        return votes[0].clone();
    }
    let normed: Vec<&str> = votes.iter().map(|v| norm(v)).collect();
    let (mut best, mut best_c) = (normed[0], 0usize);
    for &cand in &normed {
        let c = normed.iter().filter(|&&x| x == cand).count();
        if c > best_c {
            best_c = c;
            best = cand;
        }
    }
    if best_c * 2 > votes.len() {
        return best.to_string(); // strict line-level majority
    }
    let width = normed[0].chars().count();
    if normed.iter().all(|s| s.chars().count() == width) {
        char_vote(&normed) // aligned per-char glitches
    } else {
        normed[0].to_string() // shifted/inserted — don't garble; keep reference
    }
}

/// Back-compat alias; `merge_frame` handles both scroll directions.
pub fn append_frame(out: &mut Vec<String>, f: &[String]) {
    merge_frame(out, f);
}

/// Stitch a sequence of cleaned frames (in any scroll order) into one document,
/// with per-line majority voting across overlapping frames for error correction.
pub fn stitch(frames: &[Vec<String>]) -> Vec<String> {
    let mut st = Stitcher::new();
    for f in frames {
        st.push(f);
    }
    st.into_document()
}

/// Full pipeline: clean each raw frame (drop chrome) then stitch the results.
pub fn stitch_raw(raw_frames: &[&str]) -> Vec<String> {
    let cleaned: Vec<Vec<String>> = raw_frames
        .iter()
        .map(|r| {
            fastclean::clean_frame(r)
                .lines()
                .map(|l| l.to_string())
                .collect()
        })
        .collect();
    stitch(&cleaned)
}

/// Reconstruction stats: how many input lines collapsed to how many unique.
pub struct Stats {
    pub frames: usize,
    pub input_lines: usize,
    pub output_lines: usize,
}

impl Stats {
    pub fn dedup_ratio(&self) -> f32 {
        if self.input_lines == 0 {
            0.0
        } else {
            1.0 - self.output_lines as f32 / self.input_lines as f32
        }
    }
}

/// Stitch and also report how much redundancy the overlap removed.
pub fn stitch_with_stats(frames: &[Vec<String>]) -> (Vec<String>, Stats) {
    let input_lines = frames.iter().map(|f| f.len()).sum();
    let out = stitch(frames);
    let stats = Stats {
        frames: frames.len(),
        input_lines,
        output_lines: out.len(),
    };
    (out, stats)
}

/// Cap on how many versions of a line we keep for voting — a strict majority over
/// a dozen samples is already robust, and it bounds memory under heavy revisiting.
const VOTE_CAP: usize = 12;

/// Streaming whole-document reconstruction WITH ERROR CORRECTION. Feed frames one
/// at a time; each document position accumulates every version seen across the
/// overlapping frames that cover it, and the reported line is the per-column
/// majority (`consensus`) — so a transient glitch present in a minority of frames
/// is voted out. Each `push` is O(H²) in the frame height H (the overlap can't
/// exceed one frame), independent of document length. One `Stitcher` per session;
/// sessions are independent, so a fleet fans trivially across cores.
#[derive(Default)]
pub struct Stitcher {
    votes: Vec<Vec<String>>, // per doc position: observed frame versions (≤VOTE_CAP)
    doc: Vec<String>,        // consensus per position, kept in sync for alignment+output
}

impl Stitcher {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one more observed version at position `i` and re-resolve its line.
    fn vote(&mut self, i: usize, line: &str) {
        if self.votes[i].len() < VOTE_CAP {
            self.votes[i].push(line.to_string());
            self.doc[i] = consensus(&self.votes[i]);
        }
    }

    /// Merge one already-cleaned frame (content lines) into the document.
    pub fn push(&mut self, frame: &[String]) {
        if self.doc.is_empty() {
            self.votes = frame.iter().map(|l| vec![l.clone()]).collect();
            self.doc = frame.to_vec();
            return;
        }
        match plan_merge(&self.doc, frame) {
            Merge::Append { keep, overlap } => {
                self.votes.truncate(keep); // drop trailing status junk
                self.doc.truncate(keep);
                let base = keep - overlap;
                for i in 0..overlap {
                    self.vote(base + i, &frame[i]); // corrective vote on the overlap
                }
                for line in &frame[overlap..] {
                    self.votes.push(vec![line.clone()]); // newly-revealed lines
                    self.doc.push(line.clone());
                }
            }
            Merge::Prepend { new, overlap } => {
                for j in 0..overlap {
                    self.vote(j, &frame[new + j]); // corrective vote on the overlap
                }
                self.votes
                    .splice(0..0, frame[..new].iter().map(|l| vec![l.clone()]));
                self.doc.splice(0..0, frame[..new].iter().cloned());
            }
            Merge::Contained { at, len } => {
                for j in 0..len {
                    self.vote(at + j, &frame[j]); // a revisit still corrects
                }
            }
            Merge::Jump => {
                for line in frame {
                    self.votes.push(vec![line.clone()]);
                    self.doc.push(line.clone());
                }
            }
        }
    }

    /// Clean a raw terminal frame (drop chrome) then merge it — the full pipeline.
    pub fn push_raw(&mut self, raw: &str) {
        let f: Vec<String> = fastclean::clean_frame(raw)
            .lines()
            .map(|l| l.to_string())
            .collect();
        self.push(&f);
    }

    pub fn document(&self) -> &[String] {
        &self.doc
    }

    pub fn into_document(self) -> Vec<String> {
        self.doc
    }

    pub fn len(&self) -> usize {
        self.doc.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc.is_empty()
    }
}

/// Reconstruct many independent sessions from their raw frames — sequential.
/// Each session is its ordered list of raw terminal frames (one `String` each).
pub fn stitch_sessions(sessions: &[Vec<String>]) -> Vec<Vec<String>> {
    sessions
        .iter()
        .map(|frames| {
            let mut st = Stitcher::new();
            for f in frames {
                st.push_raw(f);
            }
            st.into_document()
        })
        .collect()
}

/// Parallel session reconstruction — fans sessions across all cores with
/// `std::thread` (no external deps). Sessions are fully independent, so this
/// scales near-linearly: the "1000 live TUI sessions at once" path.
pub fn stitch_sessions_par(sessions: &[Vec<String>]) -> Vec<Vec<String>> {
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    // Oversubscribe 4×: the ablation shows fine static chunks let the OS rebalance
    // BOTH heterogeneous P/E cores AND skewed session sizes — 2.0×→5.6× on clustered
    // heavy loads — at ~neutral cost on uniform loads, and it beats rayon on skew.
    stitch_sessions_par_cfg(sessions, threads * 4)
}

/// Parametrized fan-out (for benchmarking/ablation): split the sessions into
/// `nworkers` static chunks and reconstruct each chunk on its own thread. With
/// `nworkers` = core count this is one chunk/core; larger values oversubscribe.
pub fn stitch_sessions_par_cfg(sessions: &[Vec<String>], nworkers: usize) -> Vec<Vec<String>> {
    let n = sessions.len();
    if n < 8 || nworkers < 2 {
        return stitch_sessions(sessions);
    }
    let chunk = n.div_ceil(nworkers.min(n));
    let mut out: Vec<Vec<String>> = Vec::with_capacity(n);
    out.resize_with(n, Vec::new);
    std::thread::scope(|scope| {
        let mut out_rest: &mut [Vec<String>] = &mut out;
        let mut ss = sessions;
        while !ss.is_empty() {
            let k = chunk.min(ss.len());
            let (sc, stail) = ss.split_at(k);
            let (oc, ot) = out_rest.split_at_mut(k);
            out_rest = ot;
            ss = stail;
            scope.spawn(move || {
                for (o, frames) in oc.iter_mut().zip(sc) {
                    let mut st = Stitcher::new();
                    for f in frames {
                        st.push_raw(f);
                    }
                    *o = st.into_document();
                }
            });
        }
    });
    out
}

/// Rayon variant of [`stitch_sessions_par`] — a work-stealing thread pool that
/// rebalances dynamically (better on heterogeneous P/E cores and reused across
/// calls). Opt-in behind the `rayon` feature; the default path stays pure-std.
#[cfg(feature = "rayon")]
pub fn stitch_sessions_rayon(sessions: &[Vec<String>]) -> Vec<Vec<String>> {
    use rayon::prelude::*;
    sessions
        .par_iter()
        .map(|frames| {
            let mut st = Stitcher::new();
            for f in frames {
                st.push_raw(f);
            }
            st.into_document()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| format!("line {i}: the quick brown fox {i}"))
            .collect()
    }

    /// Simulate scrolling a window of height `h` down `doc` in steps of `step`.
    fn scroll(d: &[String], h: usize, step: usize) -> Vec<Vec<String>> {
        let mut frames = Vec::new();
        let mut top = 0;
        loop {
            let end = (top + h).min(d.len());
            frames.push(d[top..end].to_vec());
            if end == d.len() {
                break;
            }
            top += step;
        }
        frames
    }

    #[test]
    fn reconstructs_scrolled_doc_exactly() {
        let d = doc(43);
        for &(h, step) in &[(12usize, 5usize), (10, 1), (20, 9), (8, 7)] {
            let frames = scroll(&d, h, step);
            assert_eq!(stitch(&frames), d, "h={h} step={step}");
        }
    }

    #[test]
    fn dedups_repeated_frame_and_tolerates_trailing_space() {
        let d = doc(12);
        let f1 = d.clone();
        let mut f2 = d.clone(); // same window (no scroll happened)
        f2[5] = format!("{}    ", f2[5]); // capture padded to border
        let out = stitch(&[f1, f2]);
        assert_eq!(
            out, d,
            "a re-captured identical window must dedup to one copy"
        );
    }

    #[test]
    fn tolerates_a_changed_cursor_line_in_the_overlap() {
        let d = doc(30);
        let mut frames = scroll(&d, 12, 5);
        // simulate a highlighted "current line" differing in one frame's overlap
        if frames.len() > 1 && frames[1].len() > 2 {
            frames[1][1] = format!("> {}", frames[1][1]);
        }
        let out = stitch(&frames);
        // the doc is still fully covered (every original line present, in order)
        let joined = out.join("\n");
        for line in &d {
            assert!(joined.contains(line.trim_end()), "missing: {line}");
        }
    }

    #[test]
    fn line_vote_corrects_a_minority_glitch() {
        // the same doc scrolled 3 ways so most lines are seen ≥3×; inject a
        // single-frame glitch on a line and check the majority votes it out.
        let d = doc(30);
        let mut frames = Vec::new();
        for &(h, step) in &[(12usize, 4usize), (12, 5), (12, 6)] {
            frames.extend(scroll(&d, h, step));
        }
        // corrupt line "18" in exactly one frame that contains it
        let target = &d[18];
        let mut hit = false;
        for f in frames.iter_mut() {
            if let Some(p) = f.iter().position(|l| l == target) {
                f[p] = "line 18: the qujck brown fox 18".to_string(); // one-char glitch
                hit = true;
                break;
            }
        }
        assert!(hit);
        let out = stitch(&frames);
        assert!(
            out.contains(target),
            "majority must correct the minority glitch"
        );
        assert!(
            !out.iter().any(|l| l.contains("qujck")),
            "glitch must not survive"
        );
    }

    #[test]
    fn char_vote_corrects_when_no_frame_is_clean() {
        // three different single-char glitches on the same line, in three copies —
        // no copy is correct, but each COLUMN has a 2/3 majority → recovered.
        let truth = "the quick brown fox jumps".to_string();
        let a = "thX quick brown fox jumps".to_string();
        let b = "the quick brXwn fox jumps".to_string();
        let c = "the quick brown fox jumpX".to_string();
        assert_eq!(consensus(&[a, b, c]), truth);
    }

    #[test]
    fn parallel_sessions_equal_sequential() {
        let sessions: Vec<Vec<String>> = (0..40)
            .map(|k| {
                let d: Vec<String> = (0..25)
                    .map(|i| format!("session {k} line {i:02} payload"))
                    .collect();
                let mut raws = Vec::new();
                let mut top = 0;
                loop {
                    let end = (top + 10).min(d.len());
                    let mut fr = String::new();
                    for l in &d[top..end] {
                        fr.push_str(l);
                        fr.push_str("  \u{2502}\n"); // a chrome gutter to strip
                    }
                    fr.push(':'); // pager status
                    raws.push(fr);
                    if end == d.len() {
                        break;
                    }
                    top += 4;
                }
                raws
            })
            .collect();
        assert_eq!(stitch_sessions_par(&sessions), stitch_sessions(&sessions));
    }

    #[test]
    fn streaming_stitcher_equals_batch() {
        let d = doc(50);
        let frames = scroll(&d, 14, 6);
        let mut st = Stitcher::new();
        for f in &frames {
            st.push(f);
        }
        assert_eq!(
            st.into_document(),
            stitch(&frames),
            "streaming push must equal batch stitch"
        );
    }

    #[test]
    fn reconstructs_upward_scroll() {
        // start mid-document and scroll UP — each frame reveals earlier lines
        // that must be PREPENDED, and the frames arrive in reverse doc order.
        let d = doc(40);
        let h = 12;
        let mut frames = Vec::new();
        let mut top: i32 = 28;
        loop {
            let t = top.max(0) as usize;
            frames.push(d[t..(t + h).min(d.len())].to_vec());
            if top <= 0 {
                break;
            }
            top -= 5;
        }
        assert_eq!(
            stitch(&frames),
            d,
            "upward scroll must prepend earlier content in order"
        );
    }

    #[test]
    fn dedups_down_then_back_up() {
        // scroll all the way down, then scroll back up: the up frames revisit
        // already-seen content and must add nothing (no duplicated blocks).
        let d = doc(30);
        let (h, step) = (12usize, 6usize);
        let down = scroll(&d, h, step);
        let mut frames = down.clone();
        for f in down.iter().rev().skip(1) {
            frames.push(f.clone()); // walk the same windows back up
        }
        assert_eq!(
            stitch(&frames),
            d,
            "scrolling back up must not duplicate seen content"
        );
    }

    #[test]
    fn appends_a_jump_to_unrelated_content() {
        let a = doc(10);
        let b: Vec<String> = (0..10)
            .map(|i| format!("DIFFERENT screen row {i}"))
            .collect();
        let out = stitch(&[a.clone(), b.clone()]);
        assert_eq!(
            out.len(),
            a.len() + b.len(),
            "no false overlap between unrelated screens"
        );
    }
}
