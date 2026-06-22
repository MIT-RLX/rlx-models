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

//! Compare native duration vs ONNX for hello IPA with Jasper style row.

use std::process::Command;

use rlx_runtime::Device;

fn load_style_row() -> Vec<f32> {
    let script = r#"
import numpy as np, sys
p = sys.argv[1]
row = int(sys.argv[2])
z = np.load(p)
v = z['expr-voice-2-m'][row]
sys.stdout.buffer.write(v.astype(np.float32).tobytes())
"#;
    let voices = std::path::Path::new(".cache/kittentts-mini-0.8/voices.npz");
    if !voices.is_file() {
        eprintln!("voices.npz missing; using zero style");
        return vec![0.0; 256];
    }
    let out = Command::new("python3")
        .args(["-c", script, voices.to_str().unwrap(), "6"])
        .output()
        .expect("python");
    if !out.status.success() {
        eprintln!("python failed: {}", String::from_utf8_lossy(&out.stderr));
        return vec![0.0; 256];
    }
    out.stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() -> anyhow::Result<()> {
    let weights = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights");
    let bundle = weights.join("rlx_bundle");
    kitten_tts_mini_rlx::set_env_var("KITTEN_RLX_BUNDLE", &bundle);

    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 0];
    let style = load_style_row();

    let token_len = ids.len();
    let compile_seq = kitten_tts_mini_rlx::compile_profile::compile_slot_length(token_len);
    let opts = kitten_tts_mini_rlx::GraphOptions {
        sequence_length: compile_seq,
        max_waveform_samples: token_len.saturating_mul(600).saturating_add(12_000),
    };
    let mut graph =
        kitten_tts_mini_rlx::bundle_compile::compile_from_bundle(Device::Cpu, &bundle, &opts)?;

    let mut ids_padded = ids.clone();
    ids_padded.resize(compile_seq, 0);
    let ids_bytes: Vec<u8> = ids_padded.iter().flat_map(|v| v.to_le_bytes()).collect();
    let style_bytes: Vec<u8> = style.iter().flat_map(|v| v.to_le_bytes()).collect();
    let speed_bytes: Vec<u8> = 1.0f32.to_le_bytes().to_vec();

    kitten_tts_mini_rlx::opts::set_compile_sequence_length(compile_seq);
    kitten_tts_mini_rlx::bundle_compile::set_runtime_input_ids_shape(&mut graph, token_len)?;
    kitten_tts_mini_rlx::bundle_compile::set_runtime_active_sequence(
        &mut graph,
        token_len,
        compile_seq,
    );

    let outs = kitten_tts_mini_rlx::bundle_compile::run_with_duration_fixed_point(
        &mut graph,
        &[
            ("input_ids", ids_bytes.as_slice(), rlx_ir::DType::I64),
            ("style", style_bytes.as_slice(), rlx_ir::DType::F32),
            ("speed", speed_bytes.as_slice(), rlx_ir::DType::F32),
        ],
    );

    let wave_len = outs[0].0.len() / 4;
    if let Some((dur_bytes, _)) = outs.get(1) {
        let dur: Vec<i64> = dur_bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let sum: i64 = dur[..token_len.min(dur.len())]
            .iter()
            .copied()
            .filter(|&d| d > 0 && d < 10_000)
            .sum();
        eprintln!("native duration={dur:?} active_sum={sum}");
        eprintln!(
            "native waveform raw len={wave_len} expected_trim={}",
            sum as usize * 600
        );
    }

    let ort_script = r#"
import onnx, numpy as np, onnxruntime as ort
model = onnx.load('.cache/kittentts-mini-0.8/kitten_tts_mini_v0_8.onnx')
voices = np.load('.cache/kittentts-mini-0.8/voices.npz')
style = voices['expr-voice-2-m'][6:7].astype(np.float32)
ids = np.array([[0,50,83,156,54,57,135,0]], dtype=np.int64)
speed = np.array([1.0], dtype=np.float32)
sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
wave, dur = sess.run(None, {'input_ids': ids, 'style': style, 'speed': speed})
print('ort duration', dur.reshape(-1).tolist(), 'sum', int(dur.sum()), 'wave', len(wave.reshape(-1)))
"#;
    let ort = Command::new("python3")
        .arg("-c")
        .arg(ort_script)
        .current_dir("/Users/Shared/rlx-models")
        .output();
    if let Ok(out) = ort {
        if out.status.success() {
            eprintln!("{}", String::from_utf8_lossy(&out.stdout).trim());
        }
    }
    Ok(())
}
