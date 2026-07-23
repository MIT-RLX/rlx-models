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

//! LCNetV4 depth/width schedules from the PP-OCRv6 paper (inference-fused).

use crate::config::Tier;

/// One stage: `depth` LCNetV4 blocks at `width` channels.
#[derive(Debug, Clone, Copy)]
pub struct StageCfg {
    pub depth: usize,
    pub width: usize,
}

/// Full backbone schedule for a task + tier.
#[derive(Debug, Clone, Copy)]
pub struct LcNetV4Cfg {
    pub stem: usize,
    pub stages: [StageCfg; 4],
    /// Recognition uses asymmetric stride `(2,1)` at stages 3–4.
    pub asymmetric_stride: bool,
}

impl LcNetV4Cfg {
    pub fn detection(tier: Tier) -> Self {
        match tier {
            Tier::Tiny => Self {
                stem: 16,
                stages: [
                    StageCfg {
                        depth: 2,
                        width: 16,
                    },
                    StageCfg {
                        depth: 3,
                        width: 32,
                    },
                    StageCfg {
                        depth: 5,
                        width: 64,
                    },
                    StageCfg {
                        depth: 3,
                        width: 160,
                    },
                ],
                asymmetric_stride: false,
            },
            Tier::Small => Self {
                stem: 48,
                stages: [
                    StageCfg {
                        depth: 2,
                        width: 48,
                    },
                    StageCfg {
                        depth: 3,
                        width: 96,
                    },
                    StageCfg {
                        depth: 5,
                        width: 192,
                    },
                    StageCfg {
                        depth: 3,
                        width: 384,
                    },
                ],
                asymmetric_stride: false,
            },
        }
    }

    pub fn recognition(tier: Tier) -> Self {
        match tier {
            Tier::Tiny => Self {
                stem: 48,
                stages: [
                    StageCfg {
                        depth: 1,
                        width: 48,
                    },
                    StageCfg {
                        depth: 1,
                        width: 48,
                    },
                    StageCfg {
                        depth: 3,
                        width: 96,
                    },
                    StageCfg {
                        depth: 4,
                        width: 160,
                    },
                ],
                asymmetric_stride: true,
            },
            Tier::Small => Self {
                stem: 96,
                stages: [
                    StageCfg {
                        depth: 1,
                        width: 96,
                    },
                    StageCfg {
                        depth: 2,
                        width: 96,
                    },
                    StageCfg {
                        depth: 7,
                        width: 192,
                    },
                    StageCfg {
                        depth: 3,
                        width: 384,
                    },
                ],
                asymmetric_stride: true,
            },
        }
    }
}

pub mod blocks;
