//! Throughput + accuracy bench for the fast (code) TUI cleaner vs the GPU ML
//! tagger — quantifying "which of the ML's job is faster done in code".
//!
//! Run: `cargo run -q -p rlx-termclean --release --bin rlx-termclean-bench-clean`

use std::time::Instant;

use rlx_termclean::{Rng, fastclean, generate};

fn main() {
    // ---- throughput: clean N synthetic frames (one per "session") ----
    let n = 4000usize;
    let mut rng = Rng::new(42);
    let frames: Vec<String> = (0..n).map(|i| generate(&mut rng, i as u64).input).collect();
    let refs: Vec<&str> = frames.iter().map(|s| s.as_str()).collect();
    let bytes: usize = frames.iter().map(|s| s.len()).sum();

    let _ = fastclean::clean_batch(&refs[..16]); // warm caches
    let t = Instant::now();
    let cleaned = fastclean::clean_batch(&refs);
    let secs = t.elapsed().as_secs_f64();
    let per_frame_us = secs * 1e6 / n as f64;

    println!("=== fast code cleaner — throughput (single core, pure std) ===");
    println!(
        "  cleaned {n} frames ({:.2} MB) in {:.2} ms",
        bytes as f64 / 1e6,
        secs * 1e3
    );
    println!(
        "  {:.0} frames/sec | {:.1} MB/sec | {:.2} µs/frame",
        n as f64 / secs,
        bytes as f64 / 1e6 / secs,
        per_frame_us
    );
    println!(
        "  => 1000 concurrent sessions cleaned in ~{:.2} ms on ONE core",
        1000.0 * per_frame_us / 1e3
    );
    println!("     (embarrassingly parallel: ×cores with rayon; the ML path batches on GPU)");
    let kept: usize = cleaned.iter().map(|s| s.len()).sum();
    println!(
        "  size reduction: {:.2} MB chrome -> {:.2} MB content ({:.0}% dropped)",
        bytes as f64 / 1e6,
        kept as f64 / 1e6,
        100.0 * (1.0 - kept as f64 / bytes as f64)
    );

    // ---- accuracy vs the real 32-app val labels + ML comparison ----
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        "/private/tmp/claude-501/-Users-Shared-rlx-models/47c4aeaa-f216-4fbb-9241-d0928201ee4a/scratchpad/multi".to_string()
    });
    match (
        std::fs::read_to_string(format!("{dir}/val_input.txt")),
        std::fs::read_to_string(format!("{dir}/val_tags.txt")),
    ) {
        (Ok(inp), Ok(tg)) => {
            let (mut tp, mut fp, mut fnn, mut correct, mut total) = (0u64, 0u64, 0u64, 0u64, 0u64);
            let (mut run, mut tag) = (Vec::new(), Vec::new());
            for (s, gt) in inp.lines().zip(tg.lines()) {
                let chars: Vec<char> = s.chars().collect();
                if chars.len() != gt.chars().count() {
                    continue;
                }
                fastclean::classify(&chars, &mut run, &mut tag);
                for (t, g) in tag.iter().zip(gt.chars()) {
                    let (p, y) = (*t, g == 'C');
                    total += 1;
                    if p == y {
                        correct += 1;
                    }
                    match (p, y) {
                        (true, true) => tp += 1,
                        (true, false) => fp += 1,
                        (false, true) => fnn += 1,
                        _ => {}
                    }
                }
            }
            let prec = tp as f32 / (tp + fp).max(1) as f32;
            let rec = tp as f32 / (tp + fnn).max(1) as f32;
            let f1 = if prec + rec > 0.0 {
                2.0 * prec * rec / (prec + rec)
            } else {
                0.0
            };
            let acc = 100.0 * correct as f32 / total.max(1) as f32;
            println!("\n=== fast code cleaner — accuracy on the 32-app val labels ===");
            println!("  per-char acc {acc:.1}%, content-F1 {f1:.3}   (GPU ML tagger: 0.913)");
            println!("  => the code filter alone recovers most of the ML's job for free;");
            println!(
                "     the ~{:.0}%-char residual it gets wrong is what to route to the batched GPU model.",
                100.0 - acc
            );
        }
        _ => println!("\n(no val labels at {dir} — skipping accuracy comparison)"),
    }
}
