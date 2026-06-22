// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SNAC — multi-scale neural audio codec (hubertsiuzdak/snac), decoder ported to
// rlx-runtime graphs running natively on every backend. Encoder is WIP.

pub mod codec;
pub mod config;
pub mod eager;
pub mod graph;
pub mod model;

pub use codec::SnacDecoder;
pub use config::SnacConfig;
pub use graph::{SnacDecoderGraph, SnacEncoderGraph, build_decode_graph, build_encode_graph};
pub use model::SnacWeights;
pub use rlx_core::HierarchicalCodes;

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_runtime::Device;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
    }

    fn devices() -> Vec<Device> {
        let v = vec![Device::Cpu];
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
    fn decoder_graph_matches_eager_all_backends() {
        // Small config (same shape as snac_24khz, fewer channels) for a fast test.
        let cfg = SnacConfig {
            sampling_rate: 24_000,
            encoder_dim: 4,
            encoder_rates: vec![2, 4, 8, 8],
            decoder_dim: 64,
            decoder_rates: vec![8, 8, 4, 2],
            attn_window_size: None,
            codebook_size: 256,
            codebook_dim: 8,
            vq_strides: vec![4, 2, 1],
            noise: true,
            depthwise: true,
            latent_dim: None,
        };
        let w = SnacWeights::random(cfg.clone(), 0xC0FFEE);

        // Random codes: base_len coarse frames → levels at strides 4,2,1.
        let base_len = 3usize;
        let finest = cfg.vq_strides[0];
        let t_base = base_len * finest;
        let mut r = Lcg(1);
        let mut next_code = |n: usize| {
            (0..n)
                .map(|_| {
                    ((r.f().abs() * cfg.codebook_size as f32) as usize % cfg.codebook_size) as u32
                })
                .collect::<Vec<u32>>()
        };
        let levels: Vec<Vec<u32>> = cfg
            .vq_strides
            .iter()
            .map(|&s| next_code(t_base / s))
            .collect();
        let codes = HierarchicalCodes::new(levels);

        let (z_q, t_latent) = eager::from_codes(&w, &codes).unwrap();
        assert_eq!(t_latent, t_base);

        // Build one graph (CPU) to learn the per-block noise lengths, then make
        // identical noise planes for eager + every backend.
        let probe = SnacDecoderGraph::compile_for(Device::Cpu, &w, t_latent).unwrap();
        let noise_lens = probe.noise_lengths();
        assert_eq!(
            noise_lens,
            eager::noise_plane_lengths(&cfg.decoder_rates, t_latent)
        );
        let mut rn = Lcg(2);
        let noise: Vec<Vec<f32>> = noise_lens
            .iter()
            .map(|&t| (0..t).map(|_| rn.f()).collect())
            .collect();

        let reference = eager::decode(&w, &z_q, t_latent, &noise);

        for dev in devices() {
            let mut g = SnacDecoderGraph::compile_for(dev, &w, t_latent).unwrap();
            let got = g.run(&z_q, &noise).unwrap();
            assert_eq!(got.len(), reference.len());
            let err = max_abs(&got, &reference);
            assert!(err < 3e-3, "snac decode on {dev:?}: max|Δ| = {err}");
            eprintln!("snac decode {dev:?} ok: max|Δ| = {err:.2e}");
        }
    }

    /// Real-weight parity vs the official `snac` package (fixture from
    /// `scripts/gen_fixture.py`): same codes + same noise must reproduce the
    /// reference waveform on CPU and every GPU backend.
    #[test]
    fn decoder_matches_official_snac_real_weights() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let st_path = dir.join("snac24_decoder.safetensors");
        let ref_path = dir.join("snac24_ref.json");
        if !st_path.is_file() || !ref_path.is_file() {
            eprintln!("skip real-weight parity: run scripts/gen_fixture.py first");
            return;
        }

        let bytes = std::fs::read(&st_path).unwrap();
        let w = SnacWeights::from_safetensors(&bytes, SnacConfig::snac_24khz()).unwrap();

        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&ref_path).unwrap()).unwrap();
        let levels: Vec<Vec<u32>> = v["codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|lvl| {
                lvl.as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c.as_i64().unwrap() as u32)
                    .collect()
            })
            .collect();
        let noise: Vec<Vec<f32>> = v["noise"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                p.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
                    .collect()
            })
            .collect();
        let ref_wav: Vec<f32> = v["wav"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();

        let codes = HierarchicalCodes::new(levels);
        let (z_q, t_latent) = eager::from_codes(&w, &codes).unwrap();

        // CPU eager must reproduce the official decoder.
        let eager_wav = eager::decode(&w, &z_q, t_latent, &noise);
        assert_eq!(
            eager_wav.len(),
            ref_wav.len(),
            "wav length {} vs ref {}",
            eager_wav.len(),
            ref_wav.len()
        );
        let eager_err = max_abs(&eager_wav, &ref_wav);
        assert!(
            eager_err < 2e-3,
            "CPU eager vs official SNAC: max|Δ| = {eager_err}"
        );
        eprintln!("snac CPU eager vs official: max|Δ| = {eager_err:.2e}");

        // Every backend's graph must reproduce it too.
        for dev in devices() {
            let mut g = SnacDecoderGraph::compile_for(dev, &w, t_latent).unwrap();
            let got = g.run(&z_q, &noise).unwrap();
            let err = max_abs(&got, &ref_wav);
            assert!(
                err < 3e-3,
                "graph on {dev:?} vs official SNAC: max|Δ| = {err}"
            );
            eprintln!("snac {dev:?} graph vs official: max|Δ| = {err:.2e}");
        }
    }

    /// Encoder real-weight parity vs official SNAC: PCM → latent (graph, all
    /// backends) must match HF, and the host RVQ must reproduce HF's codes.
    #[test]
    fn encoder_matches_official_snac_real_weights() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let st_path = dir.join("snac24_decoder.safetensors");
        let ref_path = dir.join("snac24_encode_ref.json");
        if !st_path.is_file() || !ref_path.is_file() {
            eprintln!("skip encoder parity: run scripts/gen_fixture.py first");
            return;
        }
        let bytes = std::fs::read(&st_path).unwrap();
        let w = SnacWeights::from_safetensors(&bytes, SnacConfig::snac_24khz()).unwrap();
        assert!(
            w.encoder.is_some(),
            "checkpoint must include encoder weights"
        );

        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&ref_path).unwrap()).unwrap();
        let pcm: Vec<f32> = v["pcm"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let ref_latent: Vec<f32> = v["latent"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let ref_codes: Vec<Vec<u32>> = v["codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|lvl| {
                lvl.as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c.as_i64().unwrap() as u32)
                    .collect()
            })
            .collect();

        let mut cpu_latent: Vec<f32> = Vec::new();
        for dev in devices() {
            let mut e = SnacEncoderGraph::compile_for(dev, &w, pcm.len()).unwrap();
            let (latent, _ld, _t) = e.run(&pcm).unwrap();
            assert_eq!(latent.len(), ref_latent.len());
            let err = max_abs(&latent, &ref_latent);
            assert!(
                err < 5e-3,
                "encoder latent on {dev:?} vs official: max|Δ| = {err}"
            );
            eprintln!("snac encoder {dev:?} latent vs official: max|Δ| = {err:.2e}");
            if dev == Device::Cpu {
                cpu_latent = latent;
            }
        }

        // Host RVQ encode on the (bit-exact) latent must reproduce HF's codes.
        let t_latent = ref_latent.len() / w.latent();
        let codes = eager::rvq_encode(&w, &cpu_latent, t_latent).unwrap();
        assert_eq!(codes.num_levels(), ref_codes.len());
        for (lvl, (got, want)) in codes.levels.iter().zip(&ref_codes).enumerate() {
            let matches = got.iter().zip(want).filter(|(a, b)| a == b).count();
            assert_eq!(
                matches,
                want.len(),
                "level {lvl}: {matches}/{} codes match HF",
                want.len()
            );
        }
        eprintln!(
            "snac encoder codes match official exactly: {:?}",
            codes.level_lengths()
        );
    }
}
