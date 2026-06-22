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

use crate::cluster::cluster_embeddings;
use crate::embed::{embed_window, window_samples};
use anyhow::Result;

const SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpeakerTurn {
    pub speaker_id: usize,
    pub start: f32,
    pub end: f32,
}

#[derive(Debug, Clone)]
pub struct DiarizeConfig {
    pub window_sec: f32,
    pub hop_sec: f32,
    pub cluster_threshold: f32,
}

impl Default for DiarizeConfig {
    fn default() -> Self {
        Self {
            window_sec: 1.5,
            hop_sec: 0.75,
            cluster_threshold: 0.25,
        }
    }
}

pub struct DiarizeSession {
    cfg: DiarizeConfig,
}

impl DiarizeSession {
    pub fn new(cfg: DiarizeConfig) -> Self {
        Self { cfg }
    }

    pub fn diarize(&mut self, pcm: &[f32]) -> Result<Vec<SpeakerTurn>> {
        let win = window_samples(self.cfg.window_sec);
        let hop = window_samples(self.cfg.hop_sec);
        if pcm.len() < win / 2 {
            return Ok(vec![SpeakerTurn {
                speaker_id: 0,
                start: 0.0,
                end: pcm.len() as f32 / SAMPLE_RATE as f32,
            }]);
        }

        let mut embeddings = Vec::new();
        let mut times = Vec::new();
        let mut start = 0usize;
        while start + win <= pcm.len() {
            embeddings.push(embed_window(&pcm[start..start + win]));
            times.push((
                start as f32 / SAMPLE_RATE as f32,
                (start + win) as f32 / SAMPLE_RATE as f32,
            ));
            start += hop;
        }

        if embeddings.is_empty() {
            return Ok(vec![SpeakerTurn {
                speaker_id: 0,
                start: 0.0,
                end: pcm.len() as f32 / SAMPLE_RATE as f32,
            }]);
        }

        let labels = cluster_embeddings(&embeddings, self.cfg.cluster_threshold);
        merge_turns(&times, &labels)
    }
}

fn merge_turns(times: &[(f32, f32)], labels: &[usize]) -> Result<Vec<SpeakerTurn>> {
    let mut out = Vec::new();
    let mut cur = labels[0];
    let mut t0 = times[0].0;
    let mut t1 = times[0].1;
    for (i, &lab) in labels.iter().enumerate().skip(1) {
        if lab == cur {
            t1 = times[i].1;
        } else {
            out.push(SpeakerTurn {
                speaker_id: cur,
                start: t0,
                end: t1,
            });
            cur = lab;
            t0 = times[i].0;
            t1 = times[i].1;
        }
    }
    out.push(SpeakerTurn {
        speaker_id: cur,
        start: t0,
        end: t1,
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diarize_short_pcm_single_speaker() {
        let pcm = vec![0.01f32; 16_000 * 3];
        let mut session = DiarizeSession::new(DiarizeConfig::default());
        let turns = session.diarize(&pcm).unwrap();
        assert!(!turns.is_empty());
        assert_eq!(turns[0].speaker_id, 0);
    }
}
