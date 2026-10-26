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

//! CTC prefix beam search producing N-best label sequences (for rescoring).

/// Prefix beam search over `[seq, classes]` log-probs. Returns collapsed, blank-free
/// label sequences with their total log-probability, best first.
///
/// Only labels within 9 nats of each frame's best score are extended; the rest
/// contribute about `e^-9`. This shrinks the inner loop from ~439 to a handful.
pub fn ctc_beam_nbest(
    logp: &[f32],
    seq: usize,
    classes: usize,
    blank: usize,
    beam: usize,
) -> Vec<(Vec<usize>, f32)> {
    let mut s = rlx_search::PrefixBeam::new(blank as u32, beam).prune_margin(9.0);
    s.push_frames(&logp[..seq * classes], classes);
    s.nbest()
        .into_iter()
        .map(|p| (p.tokens.iter().map(|t| t.id as usize).collect(), p.score))
        .collect()
}
