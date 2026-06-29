// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use super::mask::VisionKeySpan;
use super::probe::AifProbe;

/// Modulated-decode configuration derived from a paper probe (Fig. 6 step c).
#[derive(Debug, Clone)]
pub struct AifConfig {
    pub probe: AifProbe,
}

impl AifConfig {
    pub fn from_probe(probe: AifProbe) -> Self {
        Self { probe }
    }

    /// Fixed-ratio ablation (Sec. 3.2) — masks lowest μ; not adaptive AIF.
    pub fn ablation_low_mu(probe: AifProbe, ratio: f32) -> Self {
        Self {
            probe: AifProbe {
                mask_ratio: ratio,
                ..probe
            },
        }
    }

    pub fn mask_ratio(&self) -> f32 {
        self.probe.mask_ratio
    }

    pub fn blocked_keys(&self, span: VisionKeySpan) -> Vec<usize> {
        self.probe.blocked_keys(span)
    }
}

impl From<&AifProbe> for AifConfig {
    fn from(probe: &AifProbe) -> Self {
        Self::from_probe(probe.clone())
    }
}

impl From<AifProbe> for AifConfig {
    fn from(probe: AifProbe) -> Self {
        Self::from_probe(probe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_probe_preserves_ratio() {
        let dynamics = vec![vec![0.25; 4]; 8];
        let probe = AifProbe::build(dynamics);
        let cfg = AifConfig::from(&probe);
        assert_eq!(cfg.mask_ratio(), probe.mask_ratio);
    }
}
