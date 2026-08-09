// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// WavTokenizer (novateur/WavTokenizer) — Vocos decoder ported to rlx-runtime
// graphs running natively on every backend; the ISTFT runs on the host.
// (Encoder = encodec_24khz — reuse rlx-encodec; WIP.)

pub mod encoder;
pub mod graph;
pub mod istft;
pub mod model;

pub use encoder::WavtokEncoder;
pub use graph::WavtokDecoderGraph;
pub use model::WavtokWeights;

/// Decode head output `[T, 1282]` (row-major) + window → waveform via ISTFT.
pub fn head_to_wav(head_out: &[f32], t: usize, window: &[f32]) -> Vec<f32> {
    let n_freq = model::N_FFT / 2 + 1; // 641
    let mut mag = vec![0f32; n_freq * t];
    let mut phase = vec![0f32; n_freq * t];
    for ti in 0..t {
        for f in 0..n_freq {
            let m = head_out[ti * (2 * n_freq) + f];
            mag[f * t + ti] = m.exp().min(1e2);
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
        let st = dir.join("wavtokenizer.safetensors");
        let rf = dir.join("wavtokenizer_ref.json");
        if !st.is_file() || !rf.is_file() {
            eprintln!("skip: run scripts/gen_fixture.py first");
            return;
        }
        let w = WavtokWeights::from_safetensors(&std::fs::read(&st).unwrap()).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&rf).unwrap()).unwrap();
        let feats: Vec<f32> = v["feats"]
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
        let t = feats.len() / model::INPUT_CH;

        for dev in devices() {
            let mut g = WavtokDecoderGraph::compile_for(dev, &w, t).unwrap();
            let head = g.run(&feats).unwrap();
            let wav = head_to_wav(&head, t, &w.head.window);
            assert_eq!(
                wav.len(),
                ref_wav.len(),
                "{dev:?} wav {} vs {}",
                wav.len(),
                ref_wav.len()
            );
            let err = max_abs(&wav, &ref_wav);
            eprintln!("wavtokenizer decoder {dev:?} vs official: max|Δ| = {err:.2e}");
            assert!(
                err < 3e-3,
                "wavtokenizer decoder on {dev:?}: max|Δ| = {err}"
            );
        }
    }

    #[test]
    fn encoder_matches_official_real_weights() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let st = dir.join("wavtokenizer.safetensors");
        let rf = dir.join("wavtokenizer_ref.json");
        if !st.is_file() || !rf.is_file() {
            eprintln!("skip: run scripts/gen_fixture.py first");
            return;
        }
        let bytes = std::fs::read(&st).unwrap();
        let ew = encoder::EncoderWeights::from_safetensors(&bytes).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&rf).unwrap()).unwrap();
        let arr = |k: &str| -> Vec<f32> {
            v[k].as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let Some(pcm) = v.get("pcm").map(|_| arr("pcm")) else {
            eprintln!("skip: fixture has no pcm (regenerate)");
            return;
        };
        let ref_emb = arr("emb");
        let ref_codes: Vec<u32> = v["codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_i64().unwrap() as u32)
            .collect();

        for dev in devices() {
            let enc = WavtokEncoder::new(ew.clone(), dev);
            let (emb, _t, codes) = enc.encode(&pcm).unwrap();
            let emb_err = max_abs(&emb, &ref_emb);
            let matches = codes.iter().zip(&ref_codes).filter(|(a, b)| a == b).count();
            eprintln!(
                "wavtokenizer encoder {dev:?}: emb max|Δ|={emb_err:.2e}, codes {matches}/{}",
                ref_codes.len()
            );
            assert!(emb_err < 5e-3, "encoder emb on {dev:?}: {emb_err}");
            assert_eq!(
                matches,
                ref_codes.len(),
                "encoder codes on {dev:?}: {matches}/{}",
                ref_codes.len()
            );
        }
    }
}
