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

//! Classifier-free guidance — the conditional/unconditional blend shared by every
//! guided diffusion / flow-matching generator (Stable-Audio, Seed-VC, ACE-Step, …).

/// Blend a conditional and unconditional model output:
/// `out = uncond + scale · (cond − uncond)`.
///
/// `scale = 0` → unconditional, `scale = 1` → fully conditional, `scale > 1`
/// extrapolates past the conditional (the usual CFG regime).
pub fn classifier_free_guidance(cond: &[f32], uncond: &[f32], scale: f32) -> Vec<f32> {
    assert_eq!(cond.len(), uncond.len(), "CFG operand length mismatch");
    cond.iter()
        .zip(uncond)
        .map(|(&c, &u)| u + scale * (c - u))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_and_extrapolation() {
        let cond = vec![2.0f32, 4.0];
        let uncond = vec![0.0f32, 0.0];
        assert_eq!(classifier_free_guidance(&cond, &uncond, 0.0), uncond);
        assert_eq!(classifier_free_guidance(&cond, &uncond, 1.0), cond);
        assert_eq!(
            classifier_free_guidance(&cond, &uncond, 0.5),
            vec![1.0, 2.0]
        );
        // scale > 1 extrapolates beyond the conditional.
        assert_eq!(
            classifier_free_guidance(&cond, &uncond, 2.0),
            vec![4.0, 8.0]
        );
    }
}
