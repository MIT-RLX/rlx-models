// RLX — versatile ML compiler + runtime.
// Native tiny rec graph (decomposed + spatial-parameterized).

pub mod graph;
pub mod weights;

pub use graph::{GraphOptions, build_hir};
pub use weights::{LoadedWeights, load_weights};

pub const REF_HEIGHT: usize = 48;
pub const REF_WIDTH: usize = 160;
