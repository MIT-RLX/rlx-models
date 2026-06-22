// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// VibeVoice (microsoft/VibeVoice-1.5B) — the acoustic σ-VAE tokenizer decoder
// (continuous-latent, 7.5 Hz) ported to rlx-runtime graphs running natively and
// bit-exactly on every backend. The ConvNeXt causal upsampler runs on-device.
// (The encoder, semantic tokenizer, and diffusion LM head are out of scope.)

pub mod graph;
pub mod model;

pub use graph::VibeDecoderGraph;
pub use model::VibeWeights;

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
        let st = dir.join("vv_dec.safetensors");
        let rf = dir.join("vv_ref.json");
        if !st.is_file() || !rf.is_file() {
            eprintln!("skip: run scripts/gen_fixture.py first");
            return;
        }
        let w = VibeWeights::from_safetensors(&std::fs::read(&st).unwrap()).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&rf).unwrap()).unwrap();
        let latent: Vec<f32> = v["latent"]
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

        for dev in devices() {
            let mut g = VibeDecoderGraph::compile_for(dev, &w, t).unwrap();
            let wav = g.run(&latent).unwrap();
            assert_eq!(
                wav.len(),
                ref_wav.len(),
                "{dev:?} wav {} vs {}",
                wav.len(),
                ref_wav.len()
            );
            let err = max_abs(&wav, &ref_wav);
            eprintln!("vibevoice decoder {dev:?} vs official: max|Δ| = {err:.2e}");
            assert!(err < 3e-3, "vibevoice decoder on {dev:?}: max|Δ| = {err}");
        }
    }
}
