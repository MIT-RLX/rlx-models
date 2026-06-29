// RLX models — distributed inference.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Pipeline-parallel layer partitioning.
//!
//! Layers are split into `world_size` contiguous blocks and assigned to
//! ranks in **reverse**: rank 0 owns the *last* block (so it produces the
//! logits and samples), rank `world-1` owns the *first* block (so it does
//! the token embedding). This matches mlx-lm's `PipelineMixin` and means a
//! token's activations flow rank `world-1 → … → 1 → 0`.
//!
//! The split is exact (every layer assigned exactly once) and as even as
//! possible: the first `num_layers % world` forward-blocks get one extra
//! layer.

use std::ops::Range;

/// Where a rank sits in the pipeline. Drives whether it embeds tokens,
/// just transforms hidden states, or produces logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockRole {
    /// `world == 1`: this rank runs the whole model (embed → layers → head).
    Single,
    /// Highest rank: owns the first layers, embeds the input tokens, and
    /// sends hidden states toward rank 0.
    First,
    /// Interior rank: receives hidden states, runs its layers, forwards.
    Middle,
    /// Rank 0: owns the last layers, runs the final norm + LM head, and
    /// produces logits to sample from.
    Last,
}

/// Role of `rank` in a `world`-sized pipeline.
pub fn block_role(rank: u32, world: u32) -> BlockRole {
    if world <= 1 {
        BlockRole::Single
    } else if rank == world - 1 {
        BlockRole::First
    } else if rank == 0 {
        BlockRole::Last
    } else {
        BlockRole::Middle
    }
}

/// The contiguous, half-open range of layer indices `[start, end)` that
/// `rank` owns in a `world`-sized pipeline.
///
/// rank 0 → last block, rank `world-1` → first block. Panics if
/// `rank >= world`.
pub fn pipeline_layer_range(num_layers: usize, rank: u32, world: u32) -> Range<usize> {
    let world = world.max(1) as usize;
    let rank = rank as usize;
    assert!(rank < world, "rank {rank} out of range for world {world}");

    let per = num_layers / world;
    let extra = num_layers % world;
    // This rank owns forward-block `j` (reverse assignment).
    let j = world - 1 - rank;
    // Forward block i has `per + (i < extra)` layers; the first `extra`
    // blocks carry the remainder.
    let size_of = |i: usize| per + if i < extra { 1 } else { 0 };
    let start: usize = (0..j).map(size_of).sum();
    let len = size_of(j);
    start..(start + len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_map_correctly() {
        assert_eq!(block_role(0, 1), BlockRole::Single);
        assert_eq!(block_role(0, 4), BlockRole::Last);
        assert_eq!(block_role(3, 4), BlockRole::First);
        assert_eq!(block_role(1, 4), BlockRole::Middle);
        assert_eq!(block_role(2, 4), BlockRole::Middle);
    }

    #[test]
    fn even_split_is_reversed() {
        // 16 layers / 4 ranks: rank 0 = last 4, rank 3 = first 4.
        assert_eq!(pipeline_layer_range(16, 0, 4), 12..16);
        assert_eq!(pipeline_layer_range(16, 1, 4), 8..12);
        assert_eq!(pipeline_layer_range(16, 2, 4), 4..8);
        assert_eq!(pipeline_layer_range(16, 3, 4), 0..4);
    }

    #[test]
    fn uneven_split_tiles_exactly() {
        // 10 layers / 4 ranks: forward block sizes 3,3,2,2.
        // reverse → rank0=[8,10), rank1=[6,8), rank2=[3,6), rank3=[0,3).
        assert_eq!(pipeline_layer_range(10, 0, 4), 8..10);
        assert_eq!(pipeline_layer_range(10, 1, 4), 6..8);
        assert_eq!(pipeline_layer_range(10, 2, 4), 3..6);
        assert_eq!(pipeline_layer_range(10, 3, 4), 0..3);

        // Reassembled in rank order (world-1 down to 0) covers [0,10) once.
        let mut covered = vec![0u8; 10];
        for r in 0..4u32 {
            for i in pipeline_layer_range(10, r, 4) {
                covered[i] += 1;
            }
        }
        assert!(covered.iter().all(|&c| c == 1), "every layer covered once");
    }

    #[test]
    fn single_rank_owns_all() {
        assert_eq!(pipeline_layer_range(28, 0, 1), 0..28);
    }

    #[test]
    fn more_ranks_than_layers_is_safe() {
        // 2 layers, 4 ranks: two ranks get a layer, two get empty ranges.
        let total: usize = (0..4u32).map(|r| pipeline_layer_range(2, r, 4).len()).sum();
        assert_eq!(total, 2);
    }
}
