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

//! MLX quick check: dynamic prefill + decode compile paths.

mod compile_support;

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn qwen35_dynamic_prefill_runs_on_mlx() {
    use rlx_models::Qwen35RunnerBuilder;
    use rlx_models::qwen35::synth;
    use rlx_runtime::Device;

    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let mut runner = Qwen35RunnerBuilder::default()
        .inline_weights(cfg.clone(), weights)
        .device(Device::Mlx)
        .max_seq(8)
        .dynamic_prefill(true)
        .dynamic_decode(true)
        .bucketed_decode(false)
        .last_logits_only(true)
        .build()
        .expect("mlx dynamic runner");

    let logits = runner
        .prefill_get_last_logits(&[1, 2, 3])
        .expect("dynamic prefill on mlx");
    assert_eq!(logits.len(), cfg.vocab_size);
    for v in &logits {
        assert!(v.is_finite());
    }

    let step = runner.decode_get_logits(4).expect("dynamic decode on mlx");
    assert_eq!(step.len(), cfg.vocab_size);
}
