// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Fabrice Bellard TSAC audio codec — https://bellard.org/tsac/

pub mod audio;
pub mod cli;
pub mod codec;
pub mod device;
pub mod docker_ref;
pub mod download;
pub mod parity;
pub mod perf;
pub mod platform;

#[cfg(feature = "native-codec")]
pub mod native;

/// TSAC decode on RLX backends (cpu/metal/mlx/wgpu/…) via the rlx-dac graph.
#[cfg(feature = "native-codec")]
pub mod rlx_decode;

/// Correct TSAC (= Descript-DAC-44kHz) codec on all RLX backends via rlx-dac.
pub mod correct;

pub use audio::{PcmCompare, load_wav_mono_f32, prepare_tsac_wav};
#[cfg(feature = "cli")]
pub use cli::run;
pub use codec::{EncodeStats, RoundtripStats, TsacBackendKind, TsacCodec, TsacOptions};
pub use device::{device_ready, native_device_ready, parse_tsac_device, resolve_codec_device};
pub use docker_ref::{
    DEFAULT_REF_IMAGE, DEFAULT_REF_PLATFORM, DockerRefOptions, RefRoundtrip, docker_ref_available,
    run_docker_ref_roundtrip,
};
pub use download::{
    TSAC_TARBALL_URL, TSAC_VERSION_DIR, dac_model_path, default_tsac_dir, ensure_tsac, fetch_tsac,
    install_complete, resolve_install_dir, resolve_tsac_bin, weights_available,
};
pub use parity::{
    EngineRoundtrip, ParityBenchReport, ParityOptions, bench_bellard_parity,
    bench_bellard_parity_default,
};
pub use perf::{PerfBenchReport, PerfOptions, bench_perf};
pub use platform::{platform_note, tsac_binary_supported, tsac_supported};
pub use rlx_core::audio_codec::{CompressStats, FileCodec};

/// TSAC native sample rate (44.1 kHz).
pub const SAMPLE_RATE: u32 = 44_100;
