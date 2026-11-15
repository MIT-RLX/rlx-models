//! `rlx_cpu::blas::sgemm` at Whisper-small encoder shapes (550 frames, d=768, MLP 3072), to see
//! what the linked BLAS reaches on this machine. usage: sgemm_bench [reps]
fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    for (m, k, n) in [(550, 768, 768), (550, 768, 3072), (550, 3072, 768)] {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 13) as f32 * 0.01).collect();
        let mut c = vec![0f32; m * n];
        rlx_cpu::blas::sgemm(&a, &b, &mut c, m, k, n);
        let t = std::time::Instant::now();
        for _ in 0..reps {
            rlx_cpu::blas::sgemm(&a, &b, &mut c, m, k, n);
        }
        let s = t.elapsed().as_secs_f64() / reps as f64;
        println!("{m}x{k}x{n}: {:.2} ms, {:.1} GFLOP/s", s * 1e3, 2.0 * (m * k * n) as f64 / s / 1e9);
    }
}
