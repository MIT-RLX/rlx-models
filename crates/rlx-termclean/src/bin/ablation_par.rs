//! Ablation study for parallel session-reconstruction throughput.
//!
//! Axes: worker count (core scaling), chunk granularity (static vs oversubscribed),
//! strategy (pure-std thread fan-out vs rayon work-stealing), and workload skew
//! (uniform vs clustered-heavy). Best-of-N per cell to survive shared-machine
//! noise. Build with `--features rayon` to include the rayon rows.

use std::time::{Duration, Instant};

use rlx_termclean::stitch;

/// One content line rendered as a raw frame with a scrollbar gutter + pager status.
fn raw_frame(lines: &[&str], thumb: usize) -> String {
    let mut s = String::new();
    for (i, l) in lines.iter().enumerate() {
        s.push_str(l);
        s.push_str("    ");
        s.push(if i == thumb { '█' } else { '│' });
        s.push('\n');
    }
    s.push(':');
    s
}

/// Scroll a document into overlapping raw frames.
fn scrolled(doc: &[String], h: usize, step: usize) -> Vec<String> {
    let mut raws = Vec::new();
    let mut top = 0;
    loop {
        let end = (top + h).min(doc.len());
        let win: Vec<&str> = doc[top..end].iter().map(|s| s.as_str()).collect();
        raws.push(raw_frame(&win, (top / step) % h.max(1)));
        if end == doc.len() {
            break;
        }
        top += step;
    }
    raws
}

fn best_of(trials: usize, mut f: impl FnMut() -> Vec<Vec<String>>) -> Duration {
    std::hint::black_box(f()); // warmup
    let mut best = Duration::MAX;
    for _ in 0..trials {
        let t = Instant::now();
        let r = f();
        best = best.min(t.elapsed());
        std::hint::black_box(r);
    }
    best
}

fn main() {
    const DOC: usize = 400;
    const H: usize = 40;
    const STEP: usize = 18;
    const N: usize = 1000;
    const TRIALS: usize = 5;

    let doc: Vec<String> = (0..DOC)
        .map(|i| format!("Document line {i:03} - payload text here for stitching bench"))
        .collect();
    let light = scrolled(&doc, H, STEP);
    let heavy: Vec<String> = (0..8).flat_map(|_| light.iter().cloned()).collect(); // 8× frames
    let fpl = light.len();

    let uniform: Vec<Vec<String>> = std::iter::repeat_with(|| light.clone()).take(N).collect();
    let u_frames = N * fpl;
    let mut skew: Vec<Vec<String>> = std::iter::repeat_with(|| heavy.clone()).take(100).collect();
    skew.extend(std::iter::repeat_with(|| light.clone()).take(900));
    let s_frames: usize = skew.iter().map(|s| s.len()).sum();

    let cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    let fps = |d: Duration, f: usize| f as f64 / d.as_secs_f64();
    let seq_u = best_of(TRIALS, || stitch::stitch_sessions(&uniform));
    let seq_s = best_of(TRIALS, || stitch::stitch_sessions(&skew));

    println!(
        "cores={cores} (P+E) | uniform {N}×{fpl}={u_frames} frames | skewed {s_frames} frames | best-of-{TRIALS}"
    );
    println!(
        "baseline sequential: uniform {:.1} ms, skewed {:.1} ms",
        seq_u.as_secs_f64() * 1e3,
        seq_s.as_secs_f64() * 1e3
    );

    println!("\n=== A: worker-count scaling (uniform) ===");
    println!(
        "{:>8} | {:>7} | {:>9} | {:>7}",
        "workers", "ms", "fps", "speedup"
    );
    for &w in &[1usize, 2, 4, 7, 10, 14, 28, 56] {
        let d = best_of(TRIALS, || stitch::stitch_sessions_par_cfg(&uniform, w));
        println!(
            "{w:>8} | {:>7.1} | {:>9.0} | {:>6.1}x",
            d.as_secs_f64() * 1e3,
            fps(d, u_frames),
            seq_u.as_secs_f64() / d.as_secs_f64()
        );
    }

    println!("\n=== B: strategy × chunk granularity on SKEWED (100 heavy@front + 900 light) ===");
    println!(
        "{:>18} | {:>7} | {:>9} | {:>7}",
        "strategy", "ms", "fps", "speedup"
    );
    for (label, w) in [
        ("std::thread x1", cores),
        ("std::thread x2", cores * 2),
        ("std::thread x4", cores * 4),
        ("std::thread x8", cores * 8),
    ] {
        let d = best_of(TRIALS, || stitch::stitch_sessions_par_cfg(&skew, w));
        println!(
            "{label:>18} | {:>7.1} | {:>9.0} | {:>6.1}x",
            d.as_secs_f64() * 1e3,
            fps(d, s_frames),
            seq_s.as_secs_f64() / d.as_secs_f64()
        );
    }
    #[cfg(feature = "rayon")]
    {
        let d = best_of(TRIALS, || stitch::stitch_sessions_rayon(&skew));
        println!(
            "{:>18} | {:>7.1} | {:>9.0} | {:>6.1}x",
            "rayon (steal)",
            d.as_secs_f64() * 1e3,
            fps(d, s_frames),
            seq_s.as_secs_f64() / d.as_secs_f64()
        );
    }
    #[cfg(not(feature = "rayon"))]
    println!("{:>18} | (build with --features rayon)", "rayon");
}
