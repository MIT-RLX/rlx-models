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

//! Epoch-level early stopping on eval (or train) metrics.

#[derive(Debug, Clone)]
pub struct EarlyStopState {
    pub patience: usize,
    pub min_delta: f64,
    best_metric: f64,
    stale_epochs: usize,
    pub stopped: bool,
    pub stop_reason: Option<String>,
    /// 1-based epoch index when training stopped (if early).
    pub stopped_epoch: Option<usize>,
}

impl EarlyStopState {
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_metric: f64::INFINITY,
            stale_epochs: 0,
            stopped: false,
            stop_reason: None,
            stopped_epoch: None,
        }
    }

    pub fn enabled(&self) -> bool {
        self.patience > 0
    }

    pub fn best_metric(&self) -> f64 {
        self.best_metric
    }

    /// Returns `true` when training should stop after this epoch.
    pub fn observe(&mut self, epoch_one_based: usize, metric: Option<f64>) -> bool {
        if !self.enabled() {
            return false;
        }
        let Some(m) = metric.filter(|v| v.is_finite()) else {
            self.stale_epochs += 1;
            if self.stale_epochs >= self.patience {
                self.stopped = true;
                self.stopped_epoch = Some(epoch_one_based);
                self.stop_reason = Some(format!(
                    "no finite metric for {} consecutive epochs",
                    self.patience
                ));
                return true;
            }
            return false;
        };

        if m < self.best_metric - self.min_delta {
            self.best_metric = m;
            self.stale_epochs = 0;
        } else {
            self.stale_epochs += 1;
        }

        if self.stale_epochs >= self.patience {
            self.stopped = true;
            self.stopped_epoch = Some(epoch_one_based);
            self.stop_reason = Some(format!(
                "no improvement > {delta:.2e} for {n} epochs (best={best:.6})",
                delta = self.min_delta,
                n = self.patience,
                best = self.best_metric
            ));
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_never_stops() {
        let mut es = EarlyStopState::new(0, 1e-6);
        assert!(!es.observe(1, Some(1.0)));
        assert!(!es.observe(2, Some(1.0)));
    }

    #[test]
    fn stops_after_patience_without_improvement() {
        let mut es = EarlyStopState::new(3, 0.01);
        assert!(!es.observe(1, Some(1.0)));
        assert!(!es.observe(2, Some(1.0)));
        assert!(!es.observe(3, Some(1.0)));
        assert!(es.observe(4, Some(1.0)));
        assert_eq!(es.stopped_epoch, Some(4));
    }

    #[test]
    fn improvement_resets_patience() {
        let mut es = EarlyStopState::new(3, 0.01);
        assert!(!es.observe(1, Some(1.0)));
        assert!(!es.observe(2, Some(1.0)));
        assert!(!es.observe(3, Some(0.95)));
        assert!(!es.observe(6, Some(0.95)));
        assert!(!es.observe(7, Some(0.95)));
        assert!(es.observe(8, Some(0.95)));
    }
}
