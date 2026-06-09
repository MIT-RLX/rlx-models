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

//! MLX encoder backward step with real weights (env-gated).

use rlx_runtime::{Device, Session, is_available};
use rlx_voxtral_tts::config::VoxtralTtsConfig;
use rlx_voxtral_tts_train::codec_graph::CodecGraphLayout;
use rlx_voxtral_tts_train::compile::compile_train_backward;
use rlx_voxtral_tts_train::config::patch_count;
use rlx_voxtral_tts_train::dataset::WavDataset;
use rlx_voxtral_tts_train::encoder_loss::build_encoder_train_graph;
use rlx_voxtral_tts_train::weights::{fit_params_to_graph, load_codec_weights};

#[test]
fn encoder_backward_step_on_mlx_with_real_weights() {
    if !cfg!(feature = "mlx") {
        eprintln!("skip encoder_backward_step_on_mlx_with_real_weights (mlx feature off)");
        return;
    }
    let device = Device::Mlx;
    if !is_available(device) {
        eprintln!("skip encoder_backward_step_on_mlx_with_real_weights (MLX unavailable)");
        return;
    }
    let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.cache/voxtral/Voxtral-4B-TTS-2603");
    if !model_dir.join("consolidated.safetensors").exists() {
        eprintln!("skip encoder_backward_step_on_mlx_with_real_weights (weights missing)");
        return;
    }
    let cfg = VoxtralTtsConfig::from_model_dir(&model_dir).expect("model cfg");
    let codec = &cfg.audio_config.codec_args;
    let n_patches = patch_count(codec, 4.0);
    let layout = CodecGraphLayout::new(codec, n_patches);
    let train = build_encoder_train_graph(codec, &layout, 1.0, 1.0, 0.1, 0.1, 0.0, 0.0);

    let (enc, dec) = load_codec_weights(&model_dir, true, codec).expect("weights");
    let mut weights = enc;
    weights.merge(&dec);
    fit_params_to_graph(&mut weights, &train.fwd.params).expect("fit params");

    let wav_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/voxtral/bench-wavs");
    let batch = WavDataset::from_dir(&wav_dir, codec, 4.0)
        .expect("wav dir")
        .sample_batch()
        .expect("batch");
    let audio = WavDataset::patches_to_ncl(&batch.pcm, codec.pretransform_patch_size);
    let mut target = vec![0f32; layout.patch_size * layout.wav_t];
    let copy = (layout.patch_size * layout.wav_t).min(batch.pcm.len());
    target[..copy].copy_from_slice(&batch.pcm[..copy]);
    let mel = vec![0.001f32; 64 * layout.wav_t.max(1)];
    let stft = vec![0.001f32; 128 * layout.wav_t.max(1)];
    let inputs = [
        ("audio", audio.as_slice()),
        ("target_wav", target.as_slice()),
        ("mel_basis", mel.as_slice()),
        ("stft_basis", stft.as_slice()),
        ("d_fake", [0.0f32].as_slice()),
        ("asr_mse", [0.0f32].as_slice()),
    ];

    eprintln!("compiling loss forward...");
    let mut loss_forward = Session::new(device).compile(train.forward.clone());
    for (name, data) in &weights.0 {
        loss_forward.set_param(name, data);
    }
    eprintln!("running loss forward...");
    let loss_out = loss_forward.run(&inputs);
    eprintln!(
        "loss forward ok: {:?}",
        loss_out.first().and_then(|v| v.first())
    );

    eprintln!("compiling backward...");
    let (_, mut backward) =
        compile_train_backward(device, train.backward.clone(), "encoder").expect("compile");
    eprintln!("compiled backward");

    for (name, data) in &weights.0 {
        backward.set_param(name, data);
    }

    eprintln!("running backward...");
    let outs = backward.run(&[
        ("audio", &audio),
        ("target_wav", &target),
        ("mel_basis", &mel),
        ("stft_basis", &stft),
        ("d_fake", &[0.0f32]),
        ("asr_mse", &[0.0f32]),
        ("d_output", &[1.0f32]),
    ]);
    eprintln!("backward ok, {} outputs", outs.len());
    assert!(!outs.is_empty());
}
