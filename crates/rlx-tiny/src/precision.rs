//! The `--precision` compute-format registry: the float sizes the trainer's
//! `--precision` flag accepts, as a single macro table so adding one is a
//! one-line edit (not a hunt through `bin/train.rs`).
//!
//! rlx-tiny's forward runs the codebook weight-synthesis path in **f32** (the
//! codebook *is* the compression), so `--precision` only labels the run — it is
//! kept for CLI compatibility and parity with the dense sibling. The knob that
//! actually trains at reduced precision here is `--fake-quant` (emulated
//! weight-only QAT via `rlx_tensor::lowp`), which is independent of this table.

use rlx_tensor::DType;

/// Declare the supported compute float sizes. Each row is `"<cli-name>" =>
/// <compute DType>`. Generates [`SUPPORTED`] (names, for help text), [`names`]
/// (a human-readable `f32|bf16|f16` list), and [`parse`] (name → `DType`).
macro_rules! float_sizes {
    ( $( $name:literal => $dt:expr ),* $(,)? ) => {
        /// CLI names accepted by `--precision`, in declaration order.
        pub const SUPPORTED: &[&str] = &[ $( $name ),* ];

        /// Human-readable list of accepted names, e.g. `"f32|bf16|f16"`.
        pub fn names() -> String {
            SUPPORTED.join("|")
        }

        /// Parse a `--precision` name into its [`DType`]. `None` for an unknown
        /// name (the caller reports it against [`names`]).
        pub fn parse(name: &str) -> Option<DType> {
            match name {
                $( $name => Some($dt), )*
                _ => None,
            }
        }
    };
}

float_sizes! {
    "f32"  => DType::F32,
    "bf16" => DType::BF16,
    "f16"  => DType::F16,
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
    fn names_lists_all() {
        assert_eq!(names(), "f32|bf16|f16");
    }
}
