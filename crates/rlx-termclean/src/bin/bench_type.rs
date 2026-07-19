//! Accuracy (confusion matrix) + throughput for the fast 5-way type classifier
//! (JSON / code / text / file / UI). Labeled lines from `gen_typed_line`.
//!
//! Run: `cargo run -q -p rlx-termclean --release --bin rlx-termclean-bench-type`

use std::time::Instant;

use rlx_termclean::Rng;
use rlx_termclean::typeclass::{CType, classify_type, gen_typed_line};

fn main() {
    let per = 4000usize;
    let mut rng = Rng::new(7);
    let mut data: Vec<(String, CType)> = Vec::new();
    for &t in &CType::ALL {
        for _ in 0..per {
            data.push((gen_typed_line(&mut rng, t), t));
        }
    }

    // confusion matrix (rows = true, cols = predicted)
    let mut conf = [[0u32; 5]; 5];
    for (line, t) in &data {
        conf[t.idx()][classify_type(line).idx()] += 1;
    }

    println!("=== fast type classifier — JSON / code / text / file / UI ===");
    let hdr: String = CType::ALL
        .iter()
        .map(|t| format!("{:>7}", t.name()))
        .collect();
    println!("  true\\pred {hdr}    recall");
    let mut correct = 0u32;
    for &t in &CType::ALL {
        let row = conf[t.idx()];
        correct += row[t.idx()];
        let cells: String = row.iter().map(|c| format!("{c:>7}")).collect();
        println!(
            "  {:>8}{cells}    {:.1}%",
            t.name(),
            100.0 * row[t.idx()] as f32 / per as f32
        );
    }
    println!(
        "  overall accuracy: {:.1}%",
        100.0 * correct as f32 / data.len() as f32
    );

    // throughput
    let refs: Vec<&str> = data.iter().map(|(s, _)| s.as_str()).collect();
    let bytes: usize = refs.iter().map(|s| s.len()).sum();
    let mut sink = 0usize;
    for s in refs.iter().take(32) {
        sink += classify_type(s).idx();
    }
    let t0 = Instant::now();
    for s in &refs {
        sink += classify_type(s).idx();
    }
    let secs = t0.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    println!(
        "\nthroughput: {} lines in {:.2} ms = {:.0} lines/sec | {:.1} MB/s | {:.3} µs/line",
        refs.len(),
        secs * 1e3,
        refs.len() as f64 / secs,
        bytes as f64 / 1e6 / secs,
        secs * 1e6 / refs.len() as f64
    );
    println!(
        "  => tag+route a full 1000-line screen in ~{:.3} ms; single core, no GPU, no NaN.",
        1000.0 * secs * 1e3 / refs.len() as f64
    );
}
