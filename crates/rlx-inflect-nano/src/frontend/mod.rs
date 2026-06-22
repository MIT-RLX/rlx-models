//! Standalone English text frontend: clean → normalize → grapheme-to-phoneme →
//! phoneme ids → blank insertion. Mirrors the Inflect-Nano Python pipeline.

pub mod clean;
pub mod cmudict;
pub mod english;
pub mod g2p;
pub mod normalize;
pub mod numbers;
pub mod pos;
pub mod symbols;
pub mod tokenize_bert;

pub use clean::clean_tinytts_text;
pub use english::English;
pub use normalize::normalize_text;
