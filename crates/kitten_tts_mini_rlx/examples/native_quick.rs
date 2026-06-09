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

//! Fast native forward pass (bundle path). Use instead of lldb on the full parity test.
//!
//! ```bash
//! export KITTEN_RLX_BUNDLE=weights/rlx_bundle
//! cargo run -p kitten_tts_mini_rlx --example native_quick --release
//! ```

use std::time::Instant;

use rlx_runtime::Device;

fn main() -> anyhow::Result<()> {
    let weights = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights");
    let bundle = weights.join("rlx_bundle");
    if !bundle.join("graph.json").is_file() {
        anyhow::bail!("missing bundle at {}", bundle.display());
    }
    kitten_tts_mini_rlx::set_env_var("KITTEN_RLX_BUNDLE", &bundle);

    let opts = kitten_tts_mini_rlx::GraphOptions {
        sequence_length: 128,
        max_waveform_samples: 24_000,
    };

    eprintln!("compiling native graph…");
    let t0 = Instant::now();
    let compiled = kitten_tts_mini_rlx::compile(Device::Cpu, &weights, &opts)?;
    eprintln!("compile ok in {:.1}s", t0.elapsed().as_secs_f64());

    let seq = 128usize;
    let ids: Vec<i64> = (0..seq as i64).collect();
    let style = vec![0.0f32; 256];
    let speed = 1.0f32;

    let ids_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let style_bytes: Vec<u8> = style.iter().flat_map(|v| v.to_le_bytes()).collect();
    let speed_bytes: Vec<u8> = speed.to_le_bytes().to_vec();

    eprintln!("running inference…");
    let t1 = Instant::now();
    let mut graph = compiled;
    let outs = graph.run_typed(&[
        ("input_ids", ids_bytes.as_slice(), rlx_ir::DType::I64),
        ("style", style_bytes.as_slice(), rlx_ir::DType::F32),
        ("speed", speed_bytes.as_slice(), rlx_ir::DType::F32),
    ]);
    eprintln!(
        "infer ok in {:.1}s, {} outputs",
        t1.elapsed().as_secs_f64(),
        outs.len()
    );
    if let Some((bytes, dt)) = outs.first() {
        let n = bytes.len() / dt.size_bytes().max(1);
        eprintln!("waveform: dtype={dt:?} elements≈{n}");
    }
    Ok(())
}
