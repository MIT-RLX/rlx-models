//! How fast the int8 dot product and GEMV actually are.
//!
//! End-to-end timings on this machine vary by ±30% with whatever else is
//! running, which is wider than most effects worth measuring, so this takes the
//! best of several runs inside one process — the least-interrupted one, which
//! is the honest figure for a CPU kernel on a contended box.
//!
//! It exists because `inner_product` is ~80% of executor time and therefore
//! looks like the thing to optimise. It is not: the loop already runs near what
//! the core can do. Measure here before writing intrinsics; a NEON version was,
//! and lost.

use std::hint::black_box;
use std::time::Instant;

/// The same loop written out, as a check that `dot_i8` is not doing something
/// surprising.
fn scalar(xs: &[i8], ws: &[i8]) -> i32 {
    xs.iter()
        .zip(ws)
        .map(|(a, b)| i32::from(*a) * i32::from(*b))
        .sum()
}

/// The shape `inner_product` actually runs: one row against a whole weight
/// matrix.
fn gemv(n_in: usize, n_out: usize, x: &[i8], w: &[i8]) -> Vec<i32> {
    let mut out = vec![0i32; n_out];
    for (o, slot) in out.iter_mut().enumerate() {
        *slot = rlx_translate::exec::dot_i8(x, &w[o * n_in..(o + 1) * n_in]);
    }
    out
}

fn main() {
    // 512 is the model width, 2048 the FFN's; both are what `inner_product`
    // contracts over.
    for n in [512usize, 2048] {
        let mut s = 12345u64;
        let mk = |s: &mut u64| -> Vec<i8> {
            (0..n)
                .map(|_| {
                    *s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (*s >> 33) as i8
                })
                .collect()
        };
        let a = mk(&mut s);
        let b = mk(&mut s);
        assert_eq!(rlx_translate::exec::dot_i8(&a, &b), scalar(&a, &b));

        let iters = 200_000usize;
        let (mut t_scalar, mut t_neon) = (f64::MAX, f64::MAX);
        // Best-of, alternating: the minimum is the run least interrupted, which
        // is the honest figure for a CPU kernel on a contended box.
        for _ in 0..5 {
            let t = Instant::now();
            let mut acc = 0i32;
            for _ in 0..iters {
                acc = acc.wrapping_add(scalar(black_box(&a), black_box(&b)));
            }
            black_box(acc);
            t_scalar = t_scalar.min(t.elapsed().as_secs_f64());

            let t = Instant::now();
            let mut acc = 0i32;
            for _ in 0..iters {
                acc = acc.wrapping_add(rlx_translate::exec::dot_i8(black_box(&a), black_box(&b)));
            }
            black_box(acc);
            t_neon = t_neon.min(t.elapsed().as_secs_f64());
        }
        let macs = (iters * n) as f64;
        println!(
            "  dot n={n:<5} local {:>6.2} GMAC/s   exec::dot_i8 {:>6.2} GMAC/s   {:>4.1}x",
            macs / t_scalar / 1e9,
            macs / t_neon / 1e9,
            t_scalar / t_neon
        );
    }

    // The dot product in isolation says nothing about the call around it. A
    // decoder step is exactly this: one row against each of the four projection
    // matrices and the two FFN ones.
    println!();
    for (n_in, n_out) in [(512usize, 512usize), (512, 2048), (2048, 512)] {
        let mut s = 999u64;
        let mut mk = |n: usize| -> Vec<i8> {
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (s >> 33) as i8
                })
                .collect()
        };
        let x = mk(n_in);
        let w = mk(n_in * n_out);
        let iters = 2_000usize;
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..iters {
                black_box(gemv(n_in, n_out, black_box(&x), black_box(&w)));
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        let macs = (iters * n_in * n_out) as f64;
        println!(
            "  gemv {n_in:>4}x{n_out:<5} {:>7.1} us/call  {:>6.2} GMAC/s  ({:.1} MB of weights)",
            best / iters as f64 * 1e6,
            macs / best / 1e9,
            (n_in * n_out) as f64 / 1e6
        );
    }
}
