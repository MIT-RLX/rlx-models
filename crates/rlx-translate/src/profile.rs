//! Per-op timing for the graph executor, off unless asked for.
//!
//! Phase timings (`examples/profile_decode.rs`) localised the cost to the
//! search rather than the model, which was the important split. What they
//! cannot say is *which op* the remaining 246 ms of encoding is spent in —
//! twenty blocks of `inner_product`, `batch_matmul`, `instancenorm_1d` and
//! `elementwise` all land in one number.
//!
//! Set `RLX_TRANSLATE_PROFILE=1` and every [`crate::exec::run`] accumulates
//! wall time per op kind in a thread-local; [`report`] renders it. Disabled it
//! costs one relaxed atomic load per layer.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::Duration;

thread_local! {
    static TALLY: RefCell<BTreeMap<String, (Duration, usize)>> =
        const { RefCell::new(BTreeMap::new()) };
    /// The same time, attributed to the *graph* rather than the op kind.
    ///
    /// "`inner_product` is 81%" does not say whether to optimise the encoder or
    /// the decoder, and the two want different things: the encoder runs one
    /// wide batch, the decoder thousands of single rows.
    static BY_GRAPH: RefCell<BTreeMap<String, (Duration, usize)>> =
        const { RefCell::new(BTreeMap::new()) };
}

/// Whether `RLX_TRANSLATE_PROFILE` asked for op timing.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("RLX_TRANSLATE_PROFILE")
            .map(|v| !matches!(v.as_str(), "" | "0" | "false" | "no" | "off"))
            .unwrap_or(false)
    })
}

/// Adds one op's wall time to this thread's tally, under both keys.
pub fn record(kind: &str, graph: &str, d: Duration) {
    let add = |m: &RefCell<BTreeMap<String, (Duration, usize)>>, k: &str| {
        let mut m = m.borrow_mut();
        let e = m.entry(k.to_string()).or_insert((Duration::ZERO, 0));
        e.0 += d;
        e.1 += 1;
    };
    TALLY.with(|t| add(t, kind));
    BY_GRAPH.with(|t| add(t, graph));
}

/// Clears this thread's tallies.
pub fn reset() {
    TALLY.with(|t| t.borrow_mut().clear());
    BY_GRAPH.with(|t| t.borrow_mut().clear());
}

fn drain(
    m: &'static std::thread::LocalKey<RefCell<BTreeMap<String, (Duration, usize)>>>,
) -> Vec<(String, Duration, usize)> {
    let mut v: Vec<(String, Duration, usize)> = m.with(|t| {
        t.borrow()
            .iter()
            .map(|(k, (d, n))| (k.clone(), *d, *n))
            .collect()
    });
    v.sort_by_key(|r| std::cmp::Reverse(r.1));
    v
}

/// `(kind, total, calls)` sorted by time spent, descending.
pub fn tally() -> Vec<(String, Duration, usize)> {
    drain(&TALLY)
}

/// The same, attributed to the graph each op ran in.
pub fn by_graph() -> Vec<(String, Duration, usize)> {
    drain(&BY_GRAPH)
}

/// Both tallies, op kinds first.
pub fn report() -> String {
    format!(
        "{}\n  by graph\n{}",
        table(tally(), "op"),
        table(by_graph(), "graph")
    )
}

/// One tally as a table, with each row's share of the total.
fn table(rows: Vec<(String, Duration, usize)>, header: &str) -> String {
    if rows.is_empty() {
        return "  (nothing recorded; set RLX_TRANSLATE_PROFILE=1)\n".to_string();
    }
    let total: Duration = rows.iter().map(|r| r.1).sum();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let mut out = format!(
        "  {:<20} {:>10} {:>8} {:>10} {:>7}\n",
        header, "ms", "calls", "us/call", "share"
    );
    for (kind, d, n) in &rows {
        out += &format!(
            "  {:<20} {:>10.1} {:>8} {:>10.1} {:>6.1}%\n",
            kind,
            ms(*d),
            n,
            ms(*d) * 1e3 / *n as f64,
            100.0 * d.as_secs_f64() / total.as_secs_f64().max(f64::MIN_POSITIVE)
        );
    }
    out += &format!("  {:<20} {:>10.1}\n", "total", ms(total));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tally_sorts_by_time_and_sums() {
        reset();
        record("inner_product", "decoder", Duration::from_millis(30));
        record("inner_product", "encoder", Duration::from_millis(10));
        record("softmax", "encoder", Duration::from_millis(5));
        let t = tally();
        assert_eq!(t[0].0, "inner_product");
        assert_eq!(t[0].2, 2);
        assert_eq!(t[0].1, Duration::from_millis(40));
        let g = by_graph();
        assert_eq!(g[0].0, "decoder");
        assert_eq!(g[0].1, Duration::from_millis(30));
        let r = report();
        assert!(r.contains("88.9%"), "{r}");
        assert!(r.contains("by graph"), "{r}");
        reset();
        assert!(tally().is_empty());
    }
}
