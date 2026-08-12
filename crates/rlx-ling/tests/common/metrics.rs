// Shared test metrics (included, not a test target).
//
// `include!`d rather than a `mod`, matching `common/model_weights.rs`: these are
// integration-test binaries, so each one gets its own copy and dead-code
// warnings are suppressed per item.

/// Backend under test. `RLX_TEST_DEVICE=metal|mlx|cuda|rocm|gpu|coreml`.
#[allow(dead_code)]
fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

/// Deterministic pseudo-random values in `±0.5·spread`.
#[allow(dead_code)]
fn fill_spread(n: usize, seed: u64, spread: f32) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * spread
        })
        .collect()
}

/// Cosine similarity, accumulated in f64.
#[allow(dead_code)]
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

/// Largest absolute deviation, relative to the reference's peak magnitude.
///
/// Preferred over [`cosine`] for quantization work: cosine is dominated by the
/// bulk of the vector and stays >0.9999 through errors that visibly move a
/// handful of logits (see `mxfp4_model.rs`).
#[allow(dead_code)]
fn max_rel_dev(want: &[f32], got: &[f32]) -> f32 {
    let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
    want.iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
        / scale
}
