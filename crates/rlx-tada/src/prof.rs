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

//! Stage timings behind `RLX_TADA_PROFILE`.
//!
//! Synthesis has a long tail of setup — weight reads, graph builds, LIR
//! compiles — that dwarfs the decode loop on a short utterance and vanishes
//! next to it on a long one. Attributing time needs all of those stages named
//! individually, so the instrumentation is permanent rather than something
//! added and removed around each investigation. It costs an `Instant::now()`
//! per stage when the variable is unset.
//!
//! Traces go to stderr, two spaces of indent per level of nesting.

use std::sync::OnceLock;
use std::time::Instant;

/// Whether `RLX_TADA_PROFILE` is set. Read once.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("RLX_TADA_PROFILE").is_some())
}

/// Emit one trace line. The format string carries its own leading indent.
macro_rules! trace {
    ($($arg:tt)*) => {
        if $crate::prof::enabled() {
            eprintln!("[prof] {}", format_args!($($arg)*));
        }
    };
}
pub(crate) use trace;

/// Run `f`, reporting how long it took as a depth-1 stage.
pub fn stage<T>(name: &str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let out = f();
    trace!("  {name} {:?}", t.elapsed());
    out
}

/// Peak resident set size in MB.
pub fn rss_mb() -> u64 {
    #[cfg(unix)]
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) == 0 {
            let v = ru.ru_maxrss as u64;
            // macOS reports bytes here; Linux reports kibibytes.
            return if cfg!(target_os = "macos") {
                v / 1_048_576
            } else {
                v / 1024
            };
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_returns_the_inner_value() {
        assert_eq!(stage("t", || 7), 7);
    }

    #[test]
    fn rss_is_plausible() {
        // Any live test process is above zero and below a terabyte.
        let mb = rss_mb();
        assert!(mb > 0, "rss_mb() returned 0");
        assert!(mb < 1_048_576, "rss_mb() returned {mb} MB");
    }
}
