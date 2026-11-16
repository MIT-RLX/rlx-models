// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Per-channel energy normalization (PCEN; Wang et al., 2017) over the mel front end's
//! features: each mel band divided by its own slowly tracked level, then compressed. Steady
//! sounds (a fan, a plane's rumble, the room) fade toward a constant, onsets stand out, and
//! the speaker's distance (overall gain) matters little.
//!
//! The input is [`crate::MelFrontend`]'s features, `ln(power) / 10 + 2`; the output replaces
//! them in place. The smoother is a state carried from frame to frame: a stream keeps it
//! across calls ([`crate::WakeCnn`] does), a training window starts it some frames earlier.

/// Smoother weight per 10 ms frame (time constant ≈ 0.4 s).
pub const PCEN_S: f32 = 0.025;
/// Gain normalization exponent.
pub const PCEN_ALPHA: f32 = 0.98;
/// Offset and root of the compression.
pub const PCEN_DELTA: f32 = 2.0;
pub const PCEN_R: f32 = 0.5;
pub const PCEN_EPS: f32 = 1e-6;

/// PCEN over frame-major `frames` (`n_mels` each), in place. `m` is the smoother (one level
/// per band): empty starts it at the first frame's energies.
pub fn pcen_inplace(frames: &mut [f32], n_mels: usize, m: &mut alloc::vec::Vec<f32>) {
    if n_mels == 0 {
        return;
    }
    let offset = PCEN_DELTA.powf(PCEN_R);
    for fr in frames.chunks_exact_mut(n_mels) {
        if m.len() != n_mels {
            m.clear();
            m.extend(fr.iter().map(|&v| energy(v)));
        }
        for (v, level) in fr.iter_mut().zip(m.iter_mut()) {
            let e = energy(*v);
            *level = (1.0 - PCEN_S) * *level + PCEN_S * e;
            *v = (e / (PCEN_EPS + *level).powf(PCEN_ALPHA) + PCEN_DELTA).powf(PCEN_R) - offset;
        }
    }
}

/// A feature's mel energy.
#[inline]
fn energy(v: f32) -> f32 {
    ((v - 2.0) * 10.0).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A steady band settles to the same value whatever its level; a jump stands out.
    #[test]
    fn steady_levels_settle_and_onsets_stand_out() {
        let feat = |p: f32| p.ln() / 10.0 + 2.0;
        let settle = |p: f32| {
            let mut f = alloc::vec![feat(p); 300];
            pcen_inplace(&mut f, 1, &mut alloc::vec::Vec::new());
            f[299]
        };
        let (quiet, loud) = (settle(1e-3), settle(1.0));
        assert!((quiet - loud).abs() < 0.05, "steady: {quiet} vs {loud}");
        let mut f = alloc::vec![feat(1e-3); 300];
        f[200] = feat(1e-1);
        pcen_inplace(&mut f, 1, &mut alloc::vec::Vec::new());
        assert!(f[200] > 3.0 * f[199], "onset {} after {}", f[200], f[199]);
    }
}
