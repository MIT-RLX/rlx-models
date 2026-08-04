//! A tiny zero-dependency training progress bar. On a TTY it redraws in place
//! with `\r` (throttled); when output is piped/captured it prints a plain line
//! at ~5% increments so logs stay readable.

use std::io::{IsTerminal, Write};
use std::time::Instant;

pub struct Progress {
    total: usize,
    start: Instant,
    width: usize,
    tty: bool,
    last_draw: Instant,
    last_bucket: i64,
}

impl Progress {
    pub fn new(total: usize) -> Self {
        let now = Instant::now();
        Self {
            total,
            start: now,
            width: 28,
            tty: std::io::stderr().is_terminal(),
            last_draw: now,
            last_bucket: -1,
        }
    }

    /// Advance the bar to `step` (1-based) with the latest `loss`/`lr`.
    pub fn tick(&mut self, step: usize, loss: f32, lr: f32) {
        let done = step >= self.total;
        let frac = (step as f64 / self.total.max(1) as f64).min(1.0);
        let sps = step as f64 / self.start.elapsed().as_secs_f64().max(1e-9);
        let eta = if sps > 0.0 {
            ((self.total.saturating_sub(step)) as f64 / sps) as u64
        } else {
            0
        };

        if self.tty {
            let now = Instant::now();
            if !done && now.duration_since(self.last_draw).as_millis() < 80 {
                return;
            }
            self.last_draw = now;
            let filled = (frac * self.width as f64).round() as usize;
            let bar: String = "█".repeat(filled) + &"░".repeat(self.width - filled);
            eprint!(
                "\r\x1b[K[{bar}] {:>3.0}%  {step}/{}  loss {loss:.3}  lr {lr:.1e}  {sps:.1} it/s  eta {}",
                frac * 100.0,
                self.total,
                fmt_eta(eta),
            );
            let _ = std::io::stderr().flush();
        } else {
            // ~5% buckets so a captured log has ≲20 progress lines.
            let bucket = (frac * 20.0) as i64;
            if !done && bucket == self.last_bucket {
                return;
            }
            self.last_bucket = bucket;
            eprintln!(
                "[{:>3.0}%] {step}/{}  loss {loss:.3}  lr {lr:.1e}  {sps:.1} it/s  eta {}",
                frac * 100.0,
                self.total,
                fmt_eta(eta),
            );
        }
    }

    /// Print a message above the bar (an eval/sample line), keeping the bar
    /// intact on the next `tick`.
    pub fn note(&self, msg: &str) {
        if self.tty {
            eprint!("\r\x1b[K");
        }
        eprintln!("{msg}");
    }

    /// Finish: drop to a fresh line after the final bar.
    pub fn finish(&self) {
        if self.tty {
            eprintln!();
        }
    }
}

fn fmt_eta(secs: u64) -> String {
    if secs >= 3600 {
        format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
    } else if secs >= 60 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{secs}s")
    }
}
