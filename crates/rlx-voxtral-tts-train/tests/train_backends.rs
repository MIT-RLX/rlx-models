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

//! Compile training graphs on each enabled GPU backend.

use rlx_runtime::{Device, Session, is_available};
use rlx_voxtral_tts::config::TextConfig;
use rlx_voxtral_tts_train::codec_graph::{CodecGraphLayout, build_codec_forward_graph};
use rlx_voxtral_tts_train::compile::compile_train_session;
use rlx_voxtral_tts_train::lm_lora_graph::build_lora_train_graph;

fn sample_codec() -> rlx_voxtral_tts::config::CodecArgs {
    rlx_voxtral_tts::config::CodecArgs {
        channels: 1,
        sampling_rate: 24000,
        pretransform_patch_size: 240,
        patch_proj_kernel_size: 7,
        semantic_codebook_size: 128,
        semantic_dim: 256,
        acoustic_codebook_size: 21,
        acoustic_dim: 36,
        dim: 1024,
        hidden_dim: 4096,
        head_dim: 128,
        n_heads: 8,
        n_kv_heads: 8,
        attn_sliding_window_size: 16,
        encoder_transformer_lengths_str: "1,1".into(),
        encoder_convs_kernels_str: "4,3".into(),
        encoder_convs_strides_str: "2,1".into(),
        decoder_transformer_lengths_str: "1,1".into(),
        decoder_convs_kernels_str: "3,4".into(),
        decoder_convs_strides_str: "1,2".into(),
    }
}

fn tiny_text() -> TextConfig {
    TextConfig {
        hidden_size: 64,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        vocab_size: 1024,
        rms_norm_eps: 1e-5,
        max_position_embeddings: 128,
        rope_theta: 1_000_000.0,
        intermediate_size: Some(128),
    }
}

macro_rules! backend_lora_backward_compile {
    ($name:ident, $feature:literal, $device:expr) => {
        #[test]
        fn $name() {
            if !cfg!(feature = $feature) {
                eprintln!("skip {} (feature {} disabled)", stringify!($name), $feature);
                return;
            }
            let device = $device;
            if !is_available(device) {
                eprintln!("skip {} ({device:?} unavailable)", stringify!($name));
                return;
            }
            let lora = build_lora_train_graph(&tiny_text(), 16, 4, 1).expect("lora graph");
            let session = Session::new(device);
            let _ = session.compile(lora.backward.clone());
        }
    };
}

macro_rules! backend_codec_train_compile {
    ($name:ident, $feature:literal, $device:expr) => {
        #[test]
        fn $name() {
            if !cfg!(feature = $feature) {
                eprintln!("skip {} (feature {} disabled)", stringify!($name), $feature);
                return;
            }
            let device = $device;
            if !is_available(device) {
                eprintln!("skip {} ({device:?} unavailable)", stringify!($name));
                return;
            }
            let cfg = sample_codec();
            let layout = CodecGraphLayout::new(&cfg, 8);
            let fwd = build_codec_forward_graph(&cfg, &layout).expect("codec forward");
            let train = rlx_voxtral_tts_train::encoder_loss::build_encoder_train_graph(
                &cfg, &layout, 1.0, 1.0, 0.1, 0.1, 0.0, 0.0,
            );
            let compiled = compile_train_session(
                device,
                fwd.graph,
                train.backward,
                "encoder",
            )
            .expect("compile train session");
            assert_eq!(
                compiled.forward_device,
                device,
                "codec forward should compile on requested backend"
            );
            assert_eq!(
                compiled.backward_device,
                device,
                "codec backward should compile natively on {device:?} (set RLX_VOXTRAL_TTS_TRAIN_BACKWARD_CPU=1 for hybrid)"
            );
        }
    };
}

backend_lora_backward_compile!(lora_backward_compiles_cpu, "cpu", Device::Cpu);
backend_lora_backward_compile!(lora_backward_compiles_metal, "metal", Device::Metal);
backend_lora_backward_compile!(lora_backward_compiles_mlx, "mlx", Device::Mlx);
backend_lora_backward_compile!(lora_backward_compiles_cuda, "cuda", Device::Cuda);
backend_lora_backward_compile!(lora_backward_compiles_rocm, "rocm", Device::Rocm);
backend_lora_backward_compile!(lora_backward_compiles_gpu, "gpu", Device::Gpu);
backend_lora_backward_compile!(lora_backward_compiles_vulkan, "vulkan", Device::Vulkan);

backend_codec_train_compile!(codec_train_compiles_cpu, "cpu", Device::Cpu);
backend_codec_train_compile!(codec_train_compiles_metal, "metal", Device::Metal);
backend_codec_train_compile!(codec_train_compiles_mlx, "mlx", Device::Mlx);
backend_codec_train_compile!(codec_train_compiles_cuda, "cuda", Device::Cuda);
backend_codec_train_compile!(codec_train_compiles_rocm, "rocm", Device::Rocm);
backend_codec_train_compile!(codec_train_compiles_gpu, "gpu", Device::Gpu);
backend_codec_train_compile!(codec_train_compiles_vulkan, "vulkan", Device::Vulkan);

#[test]
fn lora_backward_step_on_auto_device() {
    if std::env::var("RLX_VOXTRAL_TTS_TRAIN_GPU_STEP")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("skip lora_backward_step_on_auto_device (set RLX_VOXTRAL_TTS_TRAIN_GPU_STEP=1)");
        return;
    }
    let device = rlx_voxtral_tts_train::resolve_train_device(Some("auto")).expect("device");
    if device == Device::Cpu {
        eprintln!("skip lora_backward_step_on_auto_device (no GPU backend)");
        return;
    }
    let text = tiny_text();
    let seq = 16;
    let graph = build_lora_train_graph(&text, seq, 4, 1).expect("lora graph");
    let compiled = compile_train_session(
        device,
        graph.forward.clone(),
        graph.backward.clone(),
        "lora",
    )
    .expect("compile");
    let mut backward = compiled.backward;
    let rank = 4;
    let h = text.hidden_size;
    let mut weights = rlx_voxtral_tts_train::weights::WeightStore::default();
    for slot in &graph.params {
        weights.0.insert(slot.name.clone(), vec![0.001; rank * h]);
    }
    for (name, data) in &weights.0 {
        backward.set_param(name, data);
    }
    let embed_len = seq * h;
    let inputs = vec![0.01f32; embed_len];
    let target = vec![0.011f32; embed_len];
    let outs = backward.run(&[
        ("inputs_embeds", &inputs),
        ("target_embeds", &target),
        ("d_output", &[1.0f32]),
    ]);
    assert!(!outs.is_empty());
}

#[test]
fn encoder_backward_step_on_auto_device() {
    if std::env::var("RLX_VOXTRAL_TTS_TRAIN_GPU_STEP")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!(
            "skip encoder_backward_step_on_auto_device (set RLX_VOXTRAL_TTS_TRAIN_GPU_STEP=1)"
        );
        return;
    }
    let device = rlx_voxtral_tts_train::resolve_train_device(Some("auto")).expect("device");
    if device == Device::Cpu {
        eprintln!("skip encoder_backward_step_on_auto_device (no GPU backend)");
        return;
    }
    let cfg = sample_codec();
    let layout = CodecGraphLayout::new(&cfg, 8);
    let fwd = build_codec_forward_graph(&cfg, &layout).expect("codec forward");
    let train = rlx_voxtral_tts_train::encoder_loss::build_encoder_train_graph(
        &cfg, &layout, 1.0, 1.0, 0.1, 0.1, 0.0, 0.0,
    );
    let compiled = compile_train_session(device, fwd.graph, train.backward, "encoder")
        .expect("compile encoder train session");
    assert_eq!(
        compiled.backward_device, device,
        "encoder backward should run on auto-selected GPU"
    );

    let mut backward = compiled.backward;
    let mut weights = rlx_voxtral_tts_train::weights::WeightStore::default();
    for slot in &train.fwd.params {
        let n = slot.name.len().max(8);
        weights.0.insert(slot.name.clone(), vec![0.001; n]);
    }
    for (name, data) in &weights.0 {
        backward.set_param(name, data);
    }

    let patch = cfg.pretransform_patch_size;
    let audio_len = layout.n_patches * patch;
    let wav_len = layout.wav_t.max(1);
    let audio = vec![0.01f32; audio_len];
    let target = vec![0.011f32; patch * wav_len];
    let mel = vec![0.001f32; 64 * wav_len];
    let stft = vec![0.001f32; 128 * wav_len];

    let outs = backward.run(&[
        ("audio", &audio),
        ("target_wav", &target),
        ("mel_basis", &mel),
        ("stft_basis", &stft),
        ("d_fake", &[0.0f32]),
        ("asr_mse", &[0.0f32]),
        ("d_output", &[1.0f32]),
    ]);
    assert!(outs.len() > train.params.len());
}
