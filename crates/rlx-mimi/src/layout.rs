use ndarray::{Array2, ArrayView2};

/// `[channels, time]` → `[time, channels]`.
pub fn ct_to_tc(x: ArrayView2<f32>) -> Array2<f32> {
    let (c, t) = x.dim();
    let mut out = Array2::<f32>::zeros((t, c));
    for ci in 0..c {
        for ti in 0..t {
            out[[ti, ci]] = x[[ci, ti]];
        }
    }
    out
}

/// `[time, channels]` → `[channels, time]`.
pub fn tc_to_ct(x: ArrayView2<f32>) -> Array2<f32> {
    let (t, c) = x.dim();
    let mut out = Array2::<f32>::zeros((c, t));
    for ti in 0..t {
        for ci in 0..c {
            out[[ci, ti]] = x[[ti, ci]];
        }
    }
    out
}
