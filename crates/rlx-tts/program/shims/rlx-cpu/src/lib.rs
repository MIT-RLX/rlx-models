//! `rlx_cpu::vmath` as `src/ops.rs` calls it, in plain Rust.

pub mod vmath {
    /// `x[i] = exp(x[i])`.
    pub fn vvexpf_inplace(x: &mut [f32]) {
        for v in x {
            *v = v.exp();
        }
    }
}
