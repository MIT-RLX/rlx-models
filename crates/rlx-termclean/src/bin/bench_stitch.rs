//! Throughput benchmark for the streaming clean+stitch pipeline — the "1000
//! live sessions at once" target. Each session streams a scrolled document
//! through a `Stitcher` (chrome dropped per frame, overlaps deduped). Pure-std,
//! single core; sessions are independent so real deployments fan across cores.

use std::time::Instant;

use rlx_termclean::stitch::Stitcher;

/// Render content lines as a raw terminal frame: a scrollbar gutter column and a
/// pager status line (both chrome the pipeline must strip) around the content.
fn raw_frame(lines: &[&str], thumb: usize) -> String {
    let mut s = String::new();
    for (i, l) in lines.iter().enumerate() {
        s.push_str(l);
        s.push_str("    ");
        s.push(if i == thumb { '█' } else { '│' }); // scrollbar (chrome)
        s.push('\n');
    }
    s.push(':'); // pager prompt (chrome)
    s
}

fn main() {
    const DOC: usize = 500; // document lines
    const H: usize = 40; // frame height
    const STEP: usize = 18; // scroll step (overlap = H - STEP)
    const SESSIONS: usize = 1000;

    let doc: Vec<String> = (0..DOC)
        .map(|i| {
            format!("Document line {i:03} — lorem ipsum dolor sit amet consectetur adipiscing")
        })
        .collect();

    // one scrolled frame sequence (shared shape for every session)
    let mut raws: Vec<String> = Vec::new();
    let mut top = 0;
    loop {
        let end = (top + H).min(DOC);
        let win: Vec<&str> = doc[top..end].iter().map(|s| s.as_str()).collect();
        raws.push(raw_frame(&win, (top / STEP) % H.min(win.len().max(1))));
        if end == DOC {
            break;
        }
        top += STEP;
    }
    let fps = raws.len();

    // correctness check on one session
    let mut chk = Stitcher::new();
    for r in &raws {
        chk.push_raw(r);
    }
    let ok = chk.len() == DOC;

    // throughput over SESSIONS independent reconstructions
    let t = Instant::now();
    let mut sink = 0usize;
    for _ in 0..SESSIONS {
        let mut st = Stitcher::new();
        for r in &raws {
            st.push_raw(r);
        }
        sink = sink.wrapping_add(st.len());
    }
    let el = t.elapsed();
    let total = SESSIONS * fps;
    let us_per_frame = el.as_secs_f64() * 1e6 / total as f64;

    println!("=== streaming clean+stitch throughput ===");
    println!(
        "doc {DOC} lines, frame H={H}, step {STEP}, {fps} frames/session, {SESSIONS} sessions"
    );
    println!(
        "reconstruction: {} lines (expected {DOC}) — {}",
        chk.len(),
        if ok { "OK" } else { "MISMATCH" }
    );
    println!(
        "throughput: {total} frames in {:.1} ms → {:.2} µs/frame, {:.0} frames/sec (single core)",
        el.as_secs_f64() * 1e3,
        us_per_frame,
        total as f64 / el.as_secs_f64()
    );
    std::hint::black_box(sink);

    // parallel fan-out across cores: reconstruct all sessions at once
    let sessions: Vec<Vec<String>> = (0..SESSIONS).map(|_| raws.clone()).collect();
    let cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    // best-of-N to cut through machine-load noise on this shared box
    const TRIALS: usize = 7;
    let best_of =
        |f: &mut dyn FnMut() -> Vec<Vec<String>>| -> (std::time::Duration, Vec<Vec<String>>) {
            let mut best = std::time::Duration::MAX;
            let mut last = f();
            for _ in 0..TRIALS {
                let t = Instant::now();
                last = f();
                best = best.min(t.elapsed());
            }
            (best, last)
        };
    let fps = |d: std::time::Duration| total as f64 / d.as_secs_f64();

    let (seq_t, seq) = best_of(&mut || rlx_termclean::stitch::stitch_sessions(&sessions));
    let (par_t, par) = best_of(&mut || rlx_termclean::stitch::stitch_sessions_par(&sessions));
    let ok = seq == par && seq.iter().all(|d| d.len() == DOC);

    println!(
        "\n=== batched session reconstruction ({SESSIONS} sessions, {cores} cores, best of {TRIALS}) ==="
    );
    println!(
        "parallel == sequential: {}",
        if ok { "OK" } else { "MISMATCH" }
    );
    println!(
        "sequential  : {:6.1} ms  {:>8.0} fps  1.0x",
        seq_t.as_secs_f64() * 1e3,
        fps(seq_t)
    );
    println!(
        "std::thread : {:6.1} ms  {:>8.0} fps  {:.1}x",
        par_t.as_secs_f64() * 1e3,
        fps(par_t),
        seq_t.as_secs_f64() / par_t.as_secs_f64()
    );
    #[cfg(feature = "rayon")]
    {
        let (ray_t, ray) = best_of(&mut || rlx_termclean::stitch::stitch_sessions_rayon(&sessions));
        println!(
            "rayon       : {:6.1} ms  {:>8.0} fps  {:.1}x{}",
            ray_t.as_secs_f64() * 1e3,
            fps(ray_t),
            seq_t.as_secs_f64() / ray_t.as_secs_f64(),
            if ray == seq { "" } else { "  MISMATCH" }
        );
    }
    #[cfg(not(feature = "rayon"))]
    println!("rayon       : (build with --features rayon to compare)");

    // SKEWED workload: 100 heavy sessions (8x frames) clustered at the front + 900
    // light — the worst case for STATIC chunking (heavy work lands in a few chunks
    // while other threads idle). This is where work-stealing should pull ahead.
    let big: Vec<String> = (0..8).flat_map(|_| raws.iter().cloned()).collect();
    let mut skew: Vec<Vec<String>> = std::iter::repeat_with(|| big.clone()).take(100).collect();
    skew.extend(std::iter::repeat_with(|| raws.clone()).take(900));
    let sk_frames: usize = skew.iter().map(|s| s.len()).sum();
    let skfps = |d: std::time::Duration| sk_frames as f64 / d.as_secs_f64();
    let (sk_seq, _) = best_of(&mut || rlx_termclean::stitch::stitch_sessions(&skew));
    let (sk_par, _) = best_of(&mut || rlx_termclean::stitch::stitch_sessions_par(&skew));
    println!(
        "\n=== SKEWED (100 heavy@front + 900 light, {sk_frames} frames, best of {TRIALS}) ==="
    );
    println!(
        "std::thread : {:6.1} ms  {:>8.0} fps  {:.1}x",
        sk_par.as_secs_f64() * 1e3,
        skfps(sk_par),
        sk_seq.as_secs_f64() / sk_par.as_secs_f64()
    );
    #[cfg(feature = "rayon")]
    {
        let (sk_ray, _) = best_of(&mut || rlx_termclean::stitch::stitch_sessions_rayon(&skew));
        println!(
            "rayon       : {:6.1} ms  {:>8.0} fps  {:.1}x",
            sk_ray.as_secs_f64() * 1e3,
            skfps(sk_ray),
            sk_seq.as_secs_f64() / sk_ray.as_secs_f64()
        );
    }
}
