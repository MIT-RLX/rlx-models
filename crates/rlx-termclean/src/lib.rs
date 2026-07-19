//! `rlx-termclean` — synthetic dataset engine for TUI text extraction/cleaning.
//!
//! This crate generates supervised examples for a model that reads a raw
//! terminal screen (borders, ANSI escapes, box-drawing, padding, wrapped and
//! truncated text) and produces the clean, reflowed text content.
//!
//! Because the corruption function (rendering clean text *into* a TUI) is
//! known, every example is perfectly labeled with no human annotation:
//!   - `input`  — the rendered screen (chrome + content + ANSI)
//!   - `target` — the clean, reflowed text the model should output
//!   - `tags`   — one `C`/`X` marker per `input` char (content/chrome head)
//!
//! See [`render::generate`] for the entry point and `bin/gen_data.rs` for the
//! dataset-writing CLI.

pub mod corpus;
pub mod fastclean;
pub mod record;
pub mod render;
pub mod rng;
pub mod stitch;
pub mod symbols;
pub mod typeclass;

/// Loadable-weights ML tagger (opt-in `infer` feature): loads the trained
/// bundle and tags/cleans frames via the ported forward pass, falling back to
/// the pure-rule [`fastclean`] path when no weights are available.
#[cfg(feature = "infer")]
pub mod tagger;

pub use record::{Sample, Tag};
pub use render::generate;
pub use rng::Rng;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invariants_hold_over_many_samples() {
        let mut rng = Rng::new(1234);
        for id in 0..5000u64 {
            let s = generate(&mut rng, id);
            // input and tags are char-length aligned
            assert_eq!(
                s.input.chars().count(),
                s.tags.chars().count(),
                "kind={} id={id}",
                s.kind
            );
            // tags only ever contain the two markers
            assert!(
                s.tags.chars().all(|c| c == 'C' || c == 'X'),
                "unexpected tag marker in kind={}",
                s.kind
            );
            // target is never empty
            assert!(!s.target.is_empty(), "empty target in kind={}", s.kind);
            // every content char is real screen content (sanity: at least one
            // content char exists in each sample)
            assert!(
                s.tags.chars().any(|c| c == 'C'),
                "no content chars in kind={}",
                s.kind
            );
        }
    }

    #[test]
    fn deterministic_from_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for id in 0..200u64 {
            let sa = generate(&mut a, id);
            let sb = generate(&mut b, id);
            assert_eq!(sa.input, sb.input);
            assert_eq!(sa.target, sb.target);
            assert_eq!(sa.tags, sb.tags);
        }
    }
}
