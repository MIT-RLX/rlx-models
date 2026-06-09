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

//! stderr progress for compile + synthesis (disable with `RLX_QWEN3_TTS_NO_PROGRESS=1`).

use std::io::Write;

pub struct Progress {
    label: &'static str,
    total: usize,
    enabled: bool,
}

impl Progress {
    pub fn new(label: &'static str, total: usize) -> Self {
        let enabled = std::env::var("RLX_QWEN3_TTS_NO_PROGRESS").ok().as_deref() != Some("1");
        Self {
            label,
            total: total.max(1),
            enabled,
        }
    }

    pub fn set(&self, done: usize, detail: &str) {
        if !self.enabled {
            eprintln!("[{}] {}/{} — {detail}", self.label, done, self.total);
            return;
        }
        let done = done.min(self.total);
        const W: usize = 28;
        let filled = (done * W) / self.total;
        let bar: String = (0..W).map(|i| if i < filled { '=' } else { '-' }).collect();
        let pct = (100 * done) / self.total;
        let _ = write!(
            std::io::stderr(),
            "\r[{}] [{bar}] {done}/{total} ({pct}%) {detail}   ",
            self.label,
            total = self.total
        );
        let _ = std::io::stderr().flush();
    }

    pub fn finish(&self, msg: &str) {
        if self.enabled {
            eprintln!(
                "\r[{}] done — {msg}                              ",
                self.label
            );
        } else {
            eprintln!("[{}] {msg}", self.label);
        }
    }
}
