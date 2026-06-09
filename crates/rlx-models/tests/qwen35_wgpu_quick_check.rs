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

// basic test: tiny synthetic qwen35 runner on WGPU (skips when unavailable).

#[cfg(feature = "gpu")]
mod wgpu_tests {
    use rlx_models::Qwen35RunnerBuilder;
    use rlx_models::qwen35::synth;
    use rlx_runtime::Device;

    #[test]
    fn qwen35_tiny_runner_runs_on_wgpu() {
        if !rlx_runtime::is_available(Device::Gpu) {
            eprintln!("skip: WGPU/GPU device not available");
            return;
        }
        let cfg = synth::tiny_cfg();
        let weights = synth::synth_weights(&cfg);
        let mut runner = Qwen35RunnerBuilder::default()
            .inline_weights(cfg.clone(), weights)
            .device(Device::Gpu)
            .max_seq(8)
            .last_logits_only(true)
            .build()
            .expect("wgpu runner");
        let logits = runner
            .prefill_get_last_logits(&[1, 2, 3])
            .expect("wgpu prefill");
        assert_eq!(logits.len(), cfg.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}
