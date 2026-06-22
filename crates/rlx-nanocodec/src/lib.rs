// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// NVIDIA NanoCodec (nvidia/nemo-nano-codec-22khz-*) — the CausalHiFiGAN decoder
// + Group-FSQ dequantizer ported to rlx-runtime graphs running natively and
// bit-exactly on every backend. FSQ dequant is pure host arithmetic; the
// causal HiFi-GAN generator runs on-device. (The encoder is out of scope here.)

pub mod graph;
pub mod model;

pub use graph::NanoDecoderGraph;
pub use model::{NanoWeights, fsq_decode};

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
        let st = dir.join("nano_dec.safetensors");
        let rf = dir.join("nano_ref.json");
        if !st.is_file() || !rf.is_file() {
            eprintln!("skip: run scripts/gen_fixture.py first");
            return;
        }
        let w = NanoWeights::from_safetensors(&std::fs::read(&st).unwrap()).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&rf).unwrap()).unwrap();
        let codes: Vec<Vec<i64>> = v["codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| {
                g.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap())
                    .collect()
            })
            .collect();
        let ref_wav: Vec<f32> = v["wav"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let t = v["T"].as_u64().unwrap() as usize;

        for dev in devices() {
            let mut g = NanoDecoderGraph::compile_for(dev, &w, t).unwrap();
            let wav = g.decode_codes(&codes).unwrap();
            assert_eq!(
                wav.len(),
                ref_wav.len(),
                "{dev:?} wav {} vs {}",
                wav.len(),
                ref_wav.len()
            );
            let err = max_abs(&wav, &ref_wav);
            eprintln!("nanocodec decoder {dev:?} vs official: max|Δ| = {err:.2e}");
            assert!(err < 3e-3, "nanocodec decoder on {dev:?}: max|Δ| = {err}");
        }
    }
}
