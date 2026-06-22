// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// FACodec (amphion/naturalspeech3_facodec) — the factorized-codec decoder
// (FACodecDecoder) ported to rlx-runtime graphs running natively on every
// backend. The HiFi-GAN/BigVGAN generator (anti-aliased SnakeBeta + MRF) and
// the timbre AdaIN conditioning all run on-device; the per-speaker timbre
// affine is precomputed on the host. (The VQ encoder is out of scope here.)

pub mod graph;
pub mod model;

pub use graph::FacodecDecoderGraph;
pub use model::FacodecWeights;

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_runtime::Device;

    fn devices() -> Vec<Device> {
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
        let st = dir.join("facodec_dec.safetensors");
        let rf = dir.join("facodec_ref.json");
        if !st.is_file() || !rf.is_file() {
            eprintln!("skip: run scripts/gen_fixture.py first");
            return;
        }
        let w = FacodecWeights::from_safetensors(&std::fs::read(&st).unwrap()).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&rf).unwrap()).unwrap();
        let emb: Vec<f32> = v["emb"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let spk: Vec<f32> = v["spk"]
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
        let t = v["T"].as_u64().unwrap() as usize;

        let (gamma, beta) = w.timbre_affine(&spk);
        for dev in devices() {
            let mut g = FacodecDecoderGraph::compile_for(dev, &w, &gamma, &beta, t).unwrap();
            let wav = g.run(&emb).unwrap();
            assert_eq!(
                wav.len(),
                ref_wav.len(),
                "{dev:?} wav {} vs {}",
                wav.len(),
                ref_wav.len()
            );
            let err = max_abs(&wav, &ref_wav);
            eprintln!("facodec decoder {dev:?} vs official: max|Δ| = {err:.2e}");
            assert!(err < 3e-3, "facodec decoder on {dev:?}: max|Δ| = {err}");
        }
    }
}
