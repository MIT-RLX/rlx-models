// RLX — versatile ML compiler + runtime.
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

// RLX — TIDE refresh_experts policy (modeling_llada2_moe.py generate loop).

/// Whether to call `moe_infer_with_expert_refresh` on this forward.
///
/// Matches TIDE:
/// ```text
/// refresh_experts = (num_block == prefill_blocks)
///     or (predictive_offload_enabled and step % jump_steps == 0)
/// ```
pub fn refresh_experts(
    predictive_offload_enabled: bool,
    jump_steps: usize,
    num_block: usize,
    prefill_blocks: usize,
    denoise_step: usize,
) -> bool {
    if num_block == prefill_blocks {
        return true;
    }
    if !predictive_offload_enabled {
        return false;
    }
    let tau = jump_steps.max(1);
    denoise_step.is_multiple_of(tau)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_tide_eval_jump_steps_two() {
        for step in 0..8 {
            let expect = step % 2 == 0;
            assert_eq!(refresh_experts(true, 2, 1, 0, step), expect, "step {step}");
        }
    }

    #[test]
    fn prefill_block_always_refreshes() {
        assert!(refresh_experts(false, 100, 3, 3, 99));
        assert!(!refresh_experts(false, 100, 4, 3, 0));
    }
}
