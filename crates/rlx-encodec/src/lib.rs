// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// EnCodec (facebook/encodec_24khz) — encoder ported to rlx-runtime graphs
// running natively on every backend; the LSTM bottleneck + euclidean RVQ run on
// the host. Decoder is WIP.

pub mod codec;
pub mod config;
pub mod eager;
pub mod graph;
pub mod lstm;
pub mod model;

pub use codec::EncodecCodec;
pub use config::EncodecConfig;
pub use graph::{DecodePostLstmGraph, DecodePreLstmGraph, PostLstmGraph, PreLstmGraph};
pub use model::EncodecWeights;
pub use rlx_core::audio_codec::{AudioCodec, CodecInfo, RvqCodes};

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_runtime::Device;

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
    fn encoder_matches_official_encodec_real_weights() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let st_path = dir.join("encodec24.safetensors");
        let ref_path = dir.join("encodec24_ref.json");
        if !st_path.is_file() || !ref_path.is_file() {
            eprintln!("skip encodec parity: run scripts/gen_fixture.py first");
            return;
        }
        let bytes = std::fs::read(&st_path).unwrap();
        let w = EncodecWeights::from_safetensors(&bytes, EncodecConfig::encodec_24khz()).unwrap();

        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&ref_path).unwrap()).unwrap();
        let arr = |k: &str| -> Vec<f32> {
            v[k].as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let pcm = arr("pcm");
        let lstm_pre = arr("lstm_pre");
        let lstm_post = arr("lstm_post");
        let ref_codes: Vec<Vec<u32>> = v["codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                r.as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c.as_i64().unwrap() as u32)
                    .collect()
            })
            .collect();
        let n_q = ref_codes.len();
        let lstm_dim = w.config.lstm_dim();
        let t_lat = lstm_pre.len() / lstm_dim;

        // 1) host LSTM matches official (with residual skip).
        let lstm_out = lstm::forward(&w.encoder.lstm, &lstm_pre, lstm_dim, t_lat);
        let lstm_err = max_abs(&lstm_out, &lstm_post);
        assert!(
            lstm_err < 2e-3,
            "host LSTM vs official: max|Δ| = {lstm_err}"
        );
        eprintln!("encodec host LSTM vs official: max|Δ| = {lstm_err:.2e}");

        // 2) pre-LSTM conv stack + 3) post-LSTM + RVQ codes, on every backend.
        for dev in devices() {
            let mut pre = PreLstmGraph::compile_for(dev, &w.encoder, pcm.len()).unwrap();
            let pre_out = pre.run(&pcm).unwrap();
            let pre_err = max_abs(&pre_out, &lstm_pre);
            assert!(
                pre_err < 5e-3,
                "pre-LSTM on {dev:?} vs official: max|Δ| = {pre_err}"
            );

            let post_in = lstm::forward(&w.encoder.lstm, &pre_out, lstm_dim, t_lat);
            let mut post = PostLstmGraph::compile_for(dev, &w.encoder, lstm_dim, t_lat).unwrap();
            let latent = post.run(&post_in).unwrap();
            let codes = eager::rvq_encode(&w.codebooks, &latent, w.config.hidden_size, t_lat, n_q);

            assert_eq!(codes.len(), n_q);
            let mut total = 0usize;
            let mut ok = 0usize;
            for (got, want) in codes.iter().zip(&ref_codes) {
                for (a, b) in got.iter().zip(want) {
                    total += 1;
                    if a == b {
                        ok += 1;
                    }
                }
            }
            let frac = ok as f32 / total as f32;
            eprintln!("encodec {dev:?}: pre max|Δ|={pre_err:.2e}, codes {ok}/{total} ({frac:.3})");
            assert!(
                frac > 0.97,
                "encodec codes on {dev:?}: only {ok}/{total} match"
            );
        }
    }

    #[test]
    fn decoder_matches_official_encodec_real_weights() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let st_path = dir.join("encodec24.safetensors");
        let ref_path = dir.join("encodec24_ref.json");
        if !st_path.is_file() || !ref_path.is_file() {
            eprintln!("skip encodec decoder parity: run scripts/gen_fixture.py first");
            return;
        }
        let bytes = std::fs::read(&st_path).unwrap();
        let w = EncodecWeights::from_safetensors(&bytes, EncodecConfig::encodec_24khz()).unwrap();
        let dec = w.decoder.as_ref().expect("decoder weights");

        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&ref_path).unwrap()).unwrap();
        let ref_wav: Vec<f32> = v["wav"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let ref_codes: Vec<Vec<u32>> = v["codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                r.as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c.as_i64().unwrap() as u32)
                    .collect()
            })
            .collect();

        let hidden = w.config.hidden_size;
        let z_q = eager::rvq_decode(&w.codebooks, &ref_codes, hidden);
        let t = ref_codes[0].len();
        let lstm_dim = w.config.lstm_dim();

        for dev in devices() {
            let mut pre = DecodePreLstmGraph::compile_for(dev, dec, t).unwrap();
            let pre_out = pre.run(&z_q).unwrap();
            let post_in = lstm::forward(&dec.lstm, &pre_out, lstm_dim, t);
            let mut post = DecodePostLstmGraph::compile_for(dev, dec, lstm_dim, t).unwrap();
            let wav = post.run(&post_in).unwrap();
            assert_eq!(
                wav.len(),
                ref_wav.len(),
                "{dev:?} wav len {} vs {}",
                wav.len(),
                ref_wav.len()
            );
            let err = max_abs(&wav, &ref_wav);
            eprintln!("encodec decoder {dev:?} vs official: max|Δ| = {err:.2e}");
            assert!(
                err < 3e-3,
                "encodec decoder on {dev:?} vs official: max|Δ| = {err}"
            );
        }
    }
}
