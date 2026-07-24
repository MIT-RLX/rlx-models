// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

#[derive(Debug, Clone)]
pub struct TrainReport {
    pub epochs: usize,
    pub final_loss: f32,
    pub initial_loss: f32,
    pub train_acc: f32,
    pub keyword: String,
}

impl TrainReport {
    pub fn improved(&self) -> bool {
        self.final_loss < self.initial_loss
    }
}
