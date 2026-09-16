//! Peak resident-set-size sampling for the memory column.
//!
//! The suite runs every `(model, device)` cell in its own worker subprocess (see
//! [`crate::isolate`]), so a process high-water mark is already scoped to one model. What it is
//! *not* scoped to is the harness itself: the Whisper scorer is loaded before the model loop and
//! can outweigh a small TTS model. Callers therefore take a baseline before constructing the
//! adapter and report the difference — see [`RssTracker`].

/// Process peak RSS in MB via `getrusage(RUSAGE_SELF)`; 0 where unavailable.
///
/// Mirrors `rlx_llm_bench::metrics::peak_rss_mb` and `rlx_core::asr_bench::peak_rss_mb` so the
/// TTS, LLM and ASR leaderboards all report the same column computed the same way.
pub fn peak_rss_mb() -> u64 {
    peak_rss_bytes() / (1024 * 1024)
}

#[cfg(unix)]
fn peak_rss_bytes() -> u64 {
    // SAFETY: getrusage only writes into the zeroed rusage we hand it.
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) != 0 {
            return 0;
        }
        let max = ru.ru_maxrss as u64;
        // macOS reports bytes; Linux/BSD report kilobytes.
        if cfg!(target_os = "macos") {
            max
        } else {
            max.saturating_mul(1024)
        }
    }
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> u64 {
    0
}

/// Baseline-relative peak RSS for one model.
///
/// `ru_maxrss` is a monotonic high-water mark, so it cannot be "reset" between models — the
/// only way to attribute memory is to subtract what the process had already reached before the
/// model existed. Construct this immediately before building the adapter.
#[derive(Debug, Clone, Copy)]
pub struct RssTracker {
    baseline_mb: u64,
}

impl RssTracker {
    /// Snapshot the process high-water mark as this model's floor.
    pub fn new() -> Self {
        Self {
            baseline_mb: peak_rss_mb(),
        }
    }

    /// Harness footprint at construction time (Whisper scorer, corpus, runtime).
    pub fn baseline_mb(&self) -> u64 {
        self.baseline_mb
    }

    /// Current process high-water mark, harness included.
    pub fn peak_mb(&self) -> u64 {
        peak_rss_mb()
    }

    /// Growth attributable to the model: `peak - baseline`.
    ///
    /// Saturating, because a run that allocates less than the harness already had leaves the
    /// high-water mark untouched and legitimately reports 0.
    pub fn model_mb(&self) -> u64 {
        self.peak_mb().saturating_sub(self.baseline_mb)
    }
}

impl Default for RssTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory column for one bench row.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct RssMetrics {
    /// Process high-water mark, harness included.
    pub peak_mb: u64,
    /// Harness footprint before the model was constructed (Whisper scorer, runtime, corpus).
    pub baseline_mb: u64,
    /// `peak_mb - baseline_mb` — what this model cost on top of the harness.
    pub model_mb: u64,
}

impl RssTracker {
    /// Snapshot all three numbers for a report row.
    pub fn metrics(&self) -> RssMetrics {
        let peak_mb = self.peak_mb();
        RssMetrics {
            peak_mb,
            baseline_mb: self.baseline_mb,
            model_mb: peak_mb.saturating_sub(self.baseline_mb),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_is_nonzero_on_unix() {
        if cfg!(unix) {
            assert!(peak_rss_mb() > 0, "getrusage returned no resident pages");
        }
    }

    #[test]
    fn tracker_attributes_a_large_allocation_to_the_model() {
        let t = RssTracker::new();
        // Touch every page so the pages are genuinely resident, not just reserved.
        let mut v = vec![0u8; 64 << 20];
        for i in (0..v.len()).step_by(4096) {
            v[i] = 1;
        }
        std::hint::black_box(&v);
        assert!(
            t.model_mb() >= 32,
            "64 MiB of touched pages should move the high-water mark, got {} MB (baseline {})",
            t.model_mb(),
            t.baseline_mb()
        );
    }

    #[test]
    fn tracker_reports_zero_when_nothing_grows() {
        let t = RssTracker::new();
        assert_eq!(t.model_mb(), 0);
    }
}
