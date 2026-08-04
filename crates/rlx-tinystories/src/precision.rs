//! Single source of truth for the low-precision *compute* formats the trainer
//! supports (the `--precision` selector). One macro row per float size drives
//! three things at once — CLI parsing, the help/error text, and the range clamp
//! the model's straight-through round uses — so adding a float size is a
//! one-line edit here, not a hunt through `bin/train.rs` and `model.rs`.
//!
//! These are the *native* rlx [`DType`]s the matmuls emulate on the forward
//! (with an f32 accumulate + f32 backward — see [`crate::model`]). Emulated
//! sub-byte formats used for weight-only QAT (nvf4, f8e4m3, …) are a separate
//! axis, declared by `rlx_tensor::lowp::float_format!` and selected with
//! `--fake-quant`.

use rlx_tensor::DType;

/// Declare the supported compute float sizes. Each row is
/// `"<cli-name>" => (<compute DType>, <largest finite magnitude>)`.
///
/// Generates [`SUPPORTED`] (names, for help text), [`parse`] (name → `DType`),
/// and [`max_finite`] (`DType` → saturation bound for the STE round).
macro_rules! float_sizes {
    ( $( $name:literal => ($dt:expr, $max:expr) ),* $(,)? ) => {
        /// CLI names accepted by `--precision`, in declaration order.
        pub const SUPPORTED: &[&str] = &[ $( $name ),* ];

        /// Human-readable list of accepted names, e.g. `"f32|bf16|f16"`.
        pub fn names() -> String {
            SUPPORTED.join("|")
        }

        /// Parse a `--precision` name into its compute [`DType`]. `None` for an
        /// unknown name (the caller reports it against [`names`]).
        pub fn parse(name: &str) -> Option<DType> {
            match name {
                $( $name => Some($dt), )*
                _ => None,
            }
        }

        /// Largest finite magnitude representable by `dt`. Sub-f32 formats
        /// saturate to this in the model's straight-through round, so an
        /// out-of-range value clamps (as saturating-rounded hardware f16 does)
        /// instead of casting to ±inf → NaN. Unlisted dtypes fall back to
        /// f32's range (an effective no-op clamp).
        pub fn max_finite(dt: DType) -> f64 {
            $( if dt == $dt { return $max; } )*
            f64::from(f32::MAX)
        }
    };
}

float_sizes! {
    // name     compute DType   max finite magnitude
    "f32"  => (DType::F32,  f64::from(f32::MAX)),
    "bf16" => (DType::BF16, f64::from(f32::MAX)), // 8-bit exponent ⇒ f32's range
    "f16"  => (DType::F16,  65504.0),             // IEEE binary16 max normal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_supported_name() {
        for name in SUPPORTED {
            assert!(parse(name).is_some(), "{name} should parse");
        }
        assert!(parse("f8").is_none());
    }

    #[test]
    fn f16_clamps_bf16_and_f32_do_not() {
        assert_eq!(max_finite(DType::F16), 65504.0);
        assert_eq!(max_finite(DType::BF16), f64::from(f32::MAX));
        assert_eq!(max_finite(DType::F32), f64::from(f32::MAX));
    }

    #[test]
    fn names_lists_all() {
        assert_eq!(names(), "f32|bf16|f16");
    }
}
