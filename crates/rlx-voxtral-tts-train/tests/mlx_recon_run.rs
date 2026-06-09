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

//! End-to-end recon forward on MLX with real codec weights (env-gated).

use rlx_runtime::{Device, Session, is_available};
use rlx_voxtral_tts::config::VoxtralTtsConfig;
use rlx_voxtral_tts_train::codec_graph::{CodecGraphLayout, build_codec_recon_graph};
use rlx_voxtral_tts_train::config::patch_count;
use rlx_voxtral_tts_train::dataset::WavDataset;
use rlx_voxtral_tts_train::weights::{fit_params_to_graph, load_codec_weights};

#[test]
fn codec_recon_forward_on_mlx_with_real_weights() {
    if !cfg!(feature = "mlx") {
        eprintln!("skip codec_recon_forward_on_mlx_with_real_weights (mlx feature off)");
        return;
    }
    let device = Device::Mlx;
    if !is_available(device) {
        eprintln!("skip codec_recon_forward_on_mlx_with_real_weights (MLX unavailable)");
        return;
    }
    let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.cache/voxtral/Voxtral-4B-TTS-2603");
    if !model_dir.join("consolidated.safetensors").exists() {
        eprintln!("skip codec_recon_forward_on_mlx_with_real_weights (weights missing)");
        return;
    }
    let cfg = VoxtralTtsConfig::from_model_dir(&model_dir).expect("model cfg");
    let codec = &cfg.audio_config.codec_args;
    let n_patches = patch_count(codec, 4.0);
    let layout = CodecGraphLayout::new(codec, n_patches);
    let fwd = build_codec_recon_graph(codec, &layout).expect("recon graph");
    let mut graph = fwd.graph;
    graph.set_outputs(vec![fwd.recon_wav]);
    let mut exec = Session::new(device).compile(graph);

    let (enc, dec) = load_codec_weights(&model_dir, true, codec).expect("weights");
    let mut weights = enc;
    weights.merge(&dec);
    fit_params_to_graph(&mut weights, &fwd.params).expect("fit params");
    for (name, data) in &weights.0 {
        exec.set_param(name, data);
    }

    let wav_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/voxtral/bench-wavs");
    let batch = WavDataset::from_dir(&wav_dir, codec, 4.0)
        .expect("wav dir")
        .sample_batch()
        .expect("batch");
    let audio = WavDataset::patches_to_ncl(&batch.pcm, codec.pretransform_patch_size);
    assert_eq!(audio.len(), layout.patch_size * layout.n_patches);

    let outs = exec.run(&[("audio", &audio)]);
    let recon = outs.first().expect("recon output");
    assert_eq!(
        recon.len(),
        layout.patch_size * layout.wav_t,
        "recon flat len"
    );
    let mean: f32 = recon.iter().copied().sum::<f32>() / recon.len().max(1) as f32;
    assert!(mean.is_finite(), "recon mean must be finite, got {mean}");
}
