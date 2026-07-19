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

//! Post-decode PCM polish for MOSS: trim lead/trail silence and compress long
//! internal pauses so the render doesn't drag between phrases.

/// Options for [`tighten_pauses`].
#[derive(Debug, Clone, Copy)]
pub struct TightenOpts {
    /// Keep at most this many ms of silence inside the utterance (default 90).
    /// Set to `0` to disable internal compression (still trims lead/trail).
    pub max_internal_pause_ms: u32,
    /// Pad kept before first / after last speech (default 25 ms).
    pub edge_pad_ms: u32,
    /// Crossfade when splicing out silence (default 8 ms).
    pub fade_ms: u32,
    /// Window used for RMS (default 10 ms).
    pub win_ms: u32,
}

impl Default for TightenOpts {
    fn default() -> Self {
        Self {
            max_internal_pause_ms: 100,
            edge_pad_ms: 30,
            fade_ms: 8,
            win_ms: 10,
        }
    }
}

/// Peak absolute amplitude over interleaved PCM.
fn peak(pcm: &[f32]) -> f32 {
    pcm.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}

/// Mono mix of interleaved `channels`-wide PCM.
fn mono_mix(pcm: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return pcm.to_vec();
    }
    pcm.chunks(channels)
        .map(|c| c.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Trim lead/trail silence and compress long internal pauses.
///
/// `pcm` is interleaved (`channels` wide). Returns a new buffer of the same
/// channel layout. Only **long** holes (≥ ~150 ms of near-silence) are clamped
/// to `max_internal_pause_ms` — short dips (soft consonants / word edges) are
/// left alone so intelligibility stays intact.
pub fn tighten_pauses(pcm: &[f32], sr: u32, channels: usize, opts: TightenOpts) -> Vec<f32> {
    let ch = channels.max(1);
    if pcm.len() < ch {
        return pcm.to_vec();
    }
    let mono = mono_mix(pcm, ch);
    let n_frames = mono.len();
    let pk = peak(&mono);
    if pk < 1e-4 {
        return pcm.to_vec();
    }

    let win = ((sr as usize) * opts.win_ms as usize / 1000).max(1);
    // Conservative floor: quiet speech (fricatives) sits ~0.01–0.03 of peak;
    // only treat clearly empty regions as silence.
    let sil_thresh = (0.025 * pk).max(0.004);
    let n_wins = n_frames / win;
    if n_wins == 0 {
        return pcm.to_vec();
    }
    let rms: Vec<f32> = (0..n_wins)
        .map(|i| {
            let s = &mono[i * win..(i + 1) * win];
            (s.iter().map(|x| x * x).sum::<f32>() / win as f32).sqrt()
        })
        .collect();

    let pad_w = ((opts.edge_pad_ms as usize) / opts.win_ms.max(1) as usize).max(1);
    let max_sil_w = if opts.max_internal_pause_ms == 0 {
        usize::MAX
    } else {
        ((opts.max_internal_pause_ms as usize) / opts.win_ms.max(1) as usize).max(1)
    };
    // Don't touch internal runs shorter than this — they're word-boundary dips,
    // not the multi-hundred-ms AR holes we want to kill.
    let min_compress_w = (150usize / opts.win_ms.max(1) as usize).max(2);

    let mut keep = vec![true; n_wins]; // default keep everything
    let mut i = 0usize;
    let mut seen_speech = false;
    while i < n_wins {
        if rms[i] > sil_thresh {
            seen_speech = true;
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n_wins && rms[j] <= sil_thresh {
            j += 1;
        }
        let run = j - i;
        let trailing = j == n_wins;
        let leading = !seen_speech;

        if leading || trailing {
            // Drop lead/trail silence except a small pad.
            for k in i..j {
                keep[k] = false;
            }
            let keep_n = pad_w.min(run);
            let start = if leading { j.saturating_sub(keep_n) } else { i };
            for k in start..start + keep_n {
                if k < n_wins {
                    keep[k] = true;
                }
            }
        } else if run >= min_compress_w && opts.max_internal_pause_ms > 0 {
            // Long internal hole → keep only max_sil_w at the start of the run.
            for k in i..j {
                keep[k] = false;
            }
            let keep_n = max_sil_w.min(run);
            for k in i..i + keep_n {
                keep[k] = true;
            }
        }
        // else: short internal dip — leave keep[i..j] as true (default)
        i = j;
    }

    if !keep.iter().any(|&k| k) {
        return pcm.to_vec();
    }

    let fade = ((sr as usize) * opts.fade_ms as usize / 1000).min(win);
    let mut out: Vec<f32> = Vec::with_capacity(pcm.len());
    let mut w = 0usize;
    while w < n_wins {
        if !keep[w] {
            w += 1;
            continue;
        }
        let mut w1 = w;
        while w1 < n_wins && keep[w1] {
            w1 += 1;
        }
        let s0 = w * win;
        let s1 = (w1 * win).min(n_frames);
        let chunk_frames = s1 - s0;
        let chunk = &pcm[s0 * ch..s1 * ch];

        if out.is_empty() || fade == 0 {
            out.extend_from_slice(chunk);
        } else {
            let fade_n = fade.min(chunk_frames).min(out.len() / ch);
            let out_start = out.len() - fade_n * ch;
            for f in 0..fade_n {
                let t = (f + 1) as f32 / (fade_n + 1) as f32;
                for c in 0..ch {
                    let oi = out_start + f * ch + c;
                    let ci = f * ch + c;
                    out[oi] = out[oi] * (1.0 - t) + chunk[ci] * t;
                }
            }
            out.extend_from_slice(&chunk[fade_n * ch..]);
        }
        w = w1;
    }

    let edge = ((sr as usize) * 5 / 1000).max(1);
    let n_fr = out.len() / ch;
    for f in 0..edge.min(n_fr) {
        let g = f as f32 / edge as f32;
        for c in 0..ch {
            out[f * ch + c] *= g;
            let t = n_fr - 1 - f;
            out[t * ch + c] *= g;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compresses_long_internal_gap() {
        let sr = 48_000u32;
        let ch = 2usize;
        // 100ms speech + 500ms silence + 100ms speech
        let speech = (sr as usize) / 10;
        let gap = (sr as usize) / 2;
        let n = speech + gap + speech;
        let mut pcm = vec![0.0f32; n * ch];
        for i in 0..speech {
            let v = 0.3 * ((i as f32) * 0.1).sin();
            pcm[i * ch] = v;
            pcm[i * ch + 1] = v;
        }
        for i in speech + gap..n {
            let v = 0.3 * ((i as f32) * 0.1).sin();
            pcm[i * ch] = v;
            pcm[i * ch + 1] = v;
        }
        let out = tighten_pauses(&pcm, sr, ch, TightenOpts::default());
        let out_frames = out.len() / ch;
        // Original 700ms; after compress internal 500→≤90 + pads, should be << 500ms silence.
        assert!(out_frames < n, "should shrink: {out_frames} vs {n}");
        assert!(
            out_frames < (sr as usize) * 45 / 100, // < 450ms (100+100 speech + ≤100 pause + pads)
            "still too long: {} ms",
            out_frames * 1000 / sr as usize
        );
        // Still audible.
        let pk = out.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        assert!(pk > 0.1, "peak {pk}");
    }
}
