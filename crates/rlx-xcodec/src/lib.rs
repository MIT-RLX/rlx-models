// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// XCodec2 (HKUSTAudio/xcodec2) — the RoFormer-Vocos decoder backbone ported to
// rlx-runtime graphs running natively on every backend; the ISTFT runs on the
// host. (FSQ dequant + w2v-BERT encoder are out of scope here.)

pub mod graph;
pub mod istft;
pub mod model;

pub use graph::XcodecDecoderGraph;
pub use model::XcodecWeights;

/// Head output `[T, 1282]` + window → waveform via ISTFT.
pub fn head_to_wav(head_out: &[f32], t: usize, window: &[f32]) -> Vec<f32> {
    let n_freq = model::N_FFT / 2 + 1;
    let mut mag = vec![0f32; n_freq * t];
    let mut phase = vec![0f32; n_freq * t];
    for ti in 0..t {
        for f in 0..n_freq {
            mag[f * t + ti] = head_out[ti * (2 * n_freq) + f].exp().min(1e2);
            phase[f * t + ti] = head_out[ti * (2 * n_freq) + n_freq + f];
        }
    }
    istft::istft_same(&mag, &phase, n_freq, t, window, model::N_FFT, model::HOP)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_runtime::Device;

    fn devices() -> Vec<Device> {
        // `mut` is only exercised by the backend-feature pushes below.
        #[allow(unused_mut)]
        let mut v = vec![Device::Cpu];
        #[cfg(feature = "metal")]
        if rlx_runtime::is_available(Device::Metal) {
            v.push(Device::Metal);
        }
        #[cfg(feature = "mlx")]
        if rlx_runtime::is_available(Device::Mlx) {
            v.push(Device::Mlx);
        }
        #[cfg(feature = "gpu")]
        if rlx_runtime::is_available(Device::Gpu) {
            v.push(Device::Gpu);
        }
        v
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn decoder_matches_official_real_weights() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let st = dir.join("xcodec_dec.safetensors");
        let rf = dir.join("xcodec_ref.json");
        if !st.is_file() || !rf.is_file() {
            eprintln!("skip: run scripts/gen_fixture.py first");
            return;
        }
        let w = XcodecWeights::from_safetensors(&std::fs::read(&st).unwrap()).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&rf).unwrap()).unwrap();
        let emb: Vec<f32> = v["emb"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let ref_wav: Vec<f32> = v["wav"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let t = emb.len() / model::DIM;

        for dev in devices() {
            let mut g = XcodecDecoderGraph::compile_for(dev, &w, t).unwrap();
            let head = g.run(&emb).unwrap();
            let wav = head_to_wav(&head, t, &w.window);
            assert_eq!(
                wav.len(),
                ref_wav.len(),
                "{dev:?} wav {} vs {}",
                wav.len(),
                ref_wav.len()
            );
            let err = max_abs(&wav, &ref_wav);
            eprintln!("xcodec decoder {dev:?} vs official: max|Δ| = {err:.2e}");
            assert!(err < 3e-3, "xcodec decoder on {dev:?}: max|Δ| = {err}");
        }
    }
}
