fn main() {
    for (o, i) in [
        (1280usize, 1280usize),
        (3840, 1280),
        (5120, 1280),
        (1280, 5120),
    ] {
        let s = rlx_whisper::fast_decoder::bench_linear(o, i, 200);
        let gb = (o * i * 4) as f64 / 1e9;
        println!(
            "{o}x{i}: {:.3} ms  ({:.1} GB/s, {:.1} GFLOP/s)",
            s * 1e3,
            gb / s,
            (o * i * 5 * 2) as f64 / s / 1e9
        );
    }
}
#[allow(dead_code)]
fn _unused() {}
