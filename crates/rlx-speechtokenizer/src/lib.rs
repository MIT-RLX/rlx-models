// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SpeechTokenizer (fnlp/SpeechTokenizer) — SEANet+RVQ speech codec ported to
// rlx-runtime graphs running natively on every backend. Conv stacks run on the
// graph; the LSTM bottlenecks (encoder bidirectional, decoder unidirectional)
// and euclidean RVQ run on the host.

pub mod codec;
pub mod config;
pub mod eager;
pub mod graph;
pub mod lstm;
pub mod model;

pub use codec::SpeechTokenizerCodec;
pub use config::SpeechTokenizerConfig;
pub use model::StWeights;
pub use rlx_core::audio_codec::{AudioCodec, CodecInfo, RvqCodes};

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

    fn layers(ls: &[model::LstmLayerW]) -> Vec<lstm::LayerW<'_>> {
        ls.iter()
            .map(|l| lstm::LayerW {
                w_ih: &l.w_ih,
                w_hh: &l.w_hh,
                b_ih: &l.b_ih,
                b_hh: &l.b_hh,
            })
            .collect()
    }

    fn load() -> Option<(StWeights, serde_json::Value)> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let st = dir.join("speechtokenizer.safetensors");
        let rf = dir.join("speechtokenizer_ref.json");
        if !st.is_file() || !rf.is_file() {
            eprintln!("skip: run scripts/gen_fixture.py first");
            return None;
        }
        let w = StWeights::from_safetensors(
            &std::fs::read(&st).unwrap(),
            SpeechTokenizerConfig::default_16khz(),
        )
        .unwrap();
        let v = serde_json::from_slice(&std::fs::read(&rf).unwrap()).unwrap();
        Some((w, v))
    }

    fn arr(v: &serde_json::Value, k: &str) -> Vec<f32> {
        v[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    #[test]
    fn encoder_matches_official_real_weights() {
        let Some((w, v)) = load() else { return };
        let dim = w.config.dimension;
        let pcm = arr(&v, "pcm");
        let enc_pre = arr(&v, "enc_lstm_pre");
        let enc_post = arr(&v, "enc_lstm_post");
        let t = enc_pre.len() / dim;
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

        // bidirectional LSTM matches official
        let fwd = layers(&w.encoder.lstm.fwd);
        let rev = layers(&w.encoder.lstm.rev);
        let lstm_out = lstm::bilstm(&fwd, &rev, &enc_pre, dim, t);
        let lstm_err = max_abs(&lstm_out, &enc_post);
        assert!(
            lstm_err < 2e-3,
            "encoder bidi LSTM vs official: max|Δ| = {lstm_err}"
        );
        eprintln!("st encoder bidi LSTM vs official: max|Δ| = {lstm_err:.2e}");

        for dev in devices() {
            let (g, p, oc, ot) = graph::build_enc_pre(&w.encoder, pcm.len()).unwrap();
            let mut pre = graph::StGraph::new(dev, g, p, oc, ot, "pcm");
            let pre_out = pre.run(&pcm).unwrap();
            let pre_err = max_abs(&pre_out, &enc_pre);
            assert!(pre_err < 5e-3, "enc pre on {dev:?}: max|Δ| = {pre_err}");

            let two_dim = 2 * dim;
            let post_in = lstm::bilstm(&fwd, &rev, &pre_out, dim, t);
            let (g2, p2, oc2) = graph::build_enc_post(&w.encoder, two_dim, t).unwrap();
            let mut post = graph::StGraph::new(dev, g2, p2, oc2, t, "z");
            let latent = post.run(&post_in).unwrap();
            let codes = eager::rvq_encode(&w.codebooks, &latent, dim, t, n_q);

            let (mut ok, mut total) = (0usize, 0usize);
            for (got, want) in codes.iter().zip(&ref_codes) {
                for (a, b) in got.iter().zip(want) {
                    total += 1;
                    if a == b {
                        ok += 1;
                    }
                }
            }
            eprintln!("st encoder {dev:?}: pre max|Δ|={pre_err:.2e}, codes {ok}/{total}");
            assert!(
                ok as f32 / total as f32 > 0.97,
                "st codes on {dev:?}: {ok}/{total}"
            );
        }
    }

    #[test]
    fn decoder_matches_official_real_weights() {
        let Some((w, v)) = load() else { return };
        let dim = w.config.dimension;
        let ref_wav = arr(&v, "wav");
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
        let t = ref_codes[0].len();
        let z_q = eager::rvq_decode(&w.codebooks, &ref_codes, dim);
        let dl = layers(&w.decoder.lstm);

        for dev in devices() {
            let (g, p, oc) = graph::build_dec_pre(&w.decoder, t).unwrap();
            let mut pre = graph::StGraph::new(dev, g, p, oc, t, "zq");
            let pre_out = pre.run(&z_q).unwrap();
            let post_in = lstm::lstm(&dl, &pre_out, dim, t);
            let (g2, p2, out_t) = graph::build_dec_post(&w.decoder, dim, t).unwrap();
            let mut post = graph::StGraph::new(dev, g2, p2, 1, out_t, "z");
            let wav = post.run(&post_in).unwrap();
            assert_eq!(
                wav.len(),
                ref_wav.len(),
                "{dev:?} wav {} vs {}",
                wav.len(),
                ref_wav.len()
            );
            let err = max_abs(&wav, &ref_wav);
            eprintln!("st decoder {dev:?} vs official: max|Δ| = {err:.2e}");
            assert!(err < 3e-3, "st decoder on {dev:?}: max|Δ| = {err}");
        }
    }
}
