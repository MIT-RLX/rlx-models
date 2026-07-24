//! Per-model TTS adapters (feature-gated).

mod fake;

// `matrix-onnx` is a deprecated alias of `matrix-native` (both enable the deps).
#[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
mod chatterbox;
#[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
mod f5tts;
#[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
mod luxtts;
#[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
mod melotts;
#[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
mod miotts;
#[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
mod miratts;
#[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
mod moss_nano;
#[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
mod parlertts;
#[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
mod piper;
#[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
mod pocket_tts;
#[cfg(feature = "rlx-tts")]
mod rlx_tts;
#[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
mod soprano;
#[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
mod styletts2;
#[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
mod supertonic;
#[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
mod voxtral_tts;

#[cfg(feature = "matrix-ar")]
mod gepard;
#[cfg(feature = "matrix-ar")]
mod metavoice;
#[cfg(feature = "matrix-ar")]
mod sesame;
#[cfg(feature = "matrix-ar")]
mod zonos;

#[cfg(feature = "lm-tts")]
mod kittentts;
// kyutai has its own feature: it pulls C++ sentencepiece, whose static protobuf
// collides at link with ort_sys (matrix-onnx). Keeping it out of `lm-tts` lets
// the ort adapters build without kyutai (e.g. Linux/CUDA, where the ODR clash is
// a hard link error). `all-models` still enables it.
#[cfg(feature = "kyutai")]
mod kyutai;
#[cfg(feature = "lm-tts")]
mod orpheus;
#[cfg(feature = "lm-tts")]
mod qwen3_tts;

use anyhow::Result;
use rlx_runtime::Device;

use crate::adapter::{AdapterFactory, AdapterMeta, TtsAdapter};

pub fn catalog() -> Vec<AdapterMeta> {
    let mut v = vec![fake::meta()];
    #[cfg(feature = "rlx-tts")]
    {
        v.push(rlx_tts::meta());
    }
    #[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
    {
        v.push(chatterbox::meta());
        v.push(supertonic::meta());
        v.push(piper::meta());
        v.push(luxtts::meta());
        v.push(styletts2::meta());
        v.push(f5tts::meta());
        v.push(moss_nano::meta());
        v.push(pocket_tts::meta());
        v.push(parlertts::meta());
        v.push(miotts::meta());
        v.push(miratts::meta());
        v.push(melotts::meta());
        v.push(voxtral_tts::meta());
        v.push(soprano::meta());
    }
    #[cfg(feature = "matrix-ar")]
    {
        v.push(sesame::meta());
        v.push(zonos::meta());
        v.push(gepard::meta());
        v.push(metavoice::meta());
    }
    #[cfg(feature = "lm-tts")]
    {
        v.push(orpheus::meta());
        v.push(kittentts::meta());
        v.push(qwen3_tts::meta());
    }
    #[cfg(feature = "kyutai")]
    {
        v.push(kyutai::meta());
    }
    v
}

pub fn factory_for(id: &str) -> Option<AdapterFactory> {
    match id {
        "fake" => Some(fake::make),
        #[cfg(feature = "rlx-tts")]
        "rlx-tts" => Some(rlx_tts::make),
        #[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
        "chatterbox" => Some(chatterbox::make),
        #[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
        "supertonic" => Some(supertonic::make),
        #[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
        "piper" => Some(piper::make),
        #[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
        "luxtts" => Some(luxtts::make),
        #[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
        "styletts2" => Some(styletts2::make),
        #[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
        "f5tts" => Some(f5tts::make),
        #[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
        "moss-nano" => Some(moss_nano::make),
        #[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
        "pocket-tts" => Some(pocket_tts::make),
        #[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
        "parlertts" => Some(parlertts::make),
        #[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
        "miotts" => Some(miotts::make),
        #[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
        "miratts" => Some(miratts::make),
        #[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
        "melotts" => Some(melotts::make),
        #[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
        "voxtral-tts" => Some(voxtral_tts::make),
        #[cfg(any(feature = "matrix-native", feature = "matrix-onnx"))]
        "soprano" => Some(soprano::make),
        #[cfg(feature = "matrix-ar")]
        "sesame" => Some(sesame::make),
        #[cfg(feature = "matrix-ar")]
        "zonos" => Some(zonos::make),
        #[cfg(feature = "matrix-ar")]
        "gepard" => Some(gepard::make),
        #[cfg(feature = "matrix-ar")]
        "metavoice" => Some(metavoice::make),
        #[cfg(feature = "lm-tts")]
        "orpheus" => Some(orpheus::make),
        #[cfg(feature = "lm-tts")]
        "kittentts" => Some(kittentts::make),
        #[cfg(feature = "lm-tts")]
        "qwen3-tts" => Some(qwen3_tts::make),
        #[cfg(feature = "kyutai")]
        "kyutai" => Some(kyutai::make),
        _ => None,
    }
}

pub fn make_adapter(id: &str, device: Device) -> Result<Box<dyn TtsAdapter>> {
    let f = factory_for(id).ok_or_else(|| anyhow::anyhow!("unknown or disabled model '{id}'"))?;
    f(device)
}

pub fn all_model_ids() -> Vec<&'static str> {
    catalog().into_iter().map(|m| m.id).collect()
}
