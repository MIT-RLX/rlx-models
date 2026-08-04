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

//! MOSS-TTS model variants. `Nano` is the compact hosted model; `Local` is the
//! offline / voice-control variant, which shares the same architecture and native
//! pipeline but is loaded from a local directory (no default hosted repo) and
//! exposes voice-control conditioning.

/// Which MOSS-TTS variant to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MossVariant {
    /// MOSS-TTS-Nano — compact, hosted (see [`crate::DEFAULT_HF_REPO`]).
    #[default]
    Nano,
    /// MOSS-TTS-Local — offline variant with voice control, run from a local dir.
    Local,
}

impl MossVariant {
    /// Hosted weights repo, if the variant has a default one (`Local` is offline).
    pub fn hf_repo(self) -> Option<&'static str> {
        match self {
            MossVariant::Nano => Some(crate::DEFAULT_HF_REPO),
            MossVariant::Local => None,
        }
    }

    /// Default local weights directory.
    pub fn default_local_dir(self) -> &'static str {
        match self {
            MossVariant::Nano => crate::DEFAULT_LOCAL_DIR,
            MossVariant::Local => "weights/tts/moss-local",
        }
    }

    /// Whether the variant exposes voice-control conditioning.
    pub fn supports_voice_control(self) -> bool {
        matches!(self, MossVariant::Local)
    }

    pub fn display_name(self) -> &'static str {
        match self {
            MossVariant::Nano => "MOSS-TTS-Nano",
            MossVariant::Local => "MOSS-TTS-Local",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nano_is_hosted_local_is_offline() {
        assert!(MossVariant::Nano.hf_repo().is_some());
        assert!(MossVariant::Local.hf_repo().is_none());
        assert_eq!(MossVariant::default(), MossVariant::Nano);
    }

    #[test]
    fn only_local_has_voice_control() {
        assert!(!MossVariant::Nano.supports_voice_control());
        assert!(MossVariant::Local.supports_voice_control());
        assert_ne!(
            MossVariant::Nano.default_local_dir(),
            MossVariant::Local.default_local_dir()
        );
    }
}
