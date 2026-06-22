//! Parity + timing bench: native [`TsacCodec`] vs the public Bellard Linux `tsac` binary.

use crate::TsacBackendKind;
use crate::audio::{self, PcmCompare, prepare_tsac_wav};
use crate::codec::{EncodeStats, TsacCodec, TsacOptions};
use crate::download::{ensure_tsac, resolve_install_dir};
use crate::platform::tsac_binary_supported;
use anyhow::{Result, bail};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct ParityOptions {
    pub quality: u8,
    /// Bellard `-f` / native fast path (no transformer).
    pub fast: bool,
    pub native_device: Device,
    pub bellard_cuda: bool,
    pub min_correlation: f32,
    pub max_mse: f32,
}

impl Default for ParityOptions {
    fn default() -> Self {
        Self {
            quality: 9,
            fast: true,
            native_device: Device::Cpu,
            bellard_cuda: false,
            min_correlation: env_f32("RLX_TSAC_PARITY_MIN_CORR", 0.92),
            max_mse: env_f32("RLX_TSAC_PARITY_MAX_MSE", 0.0025),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EngineRoundtrip {
    pub encode: EncodeStats,
    pub decode_ms: f64,
    pub pcm: Vec<f32>,
    pub tsac_path: PathBuf,
    pub wav_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ParityBenchReport {
    pub input_wav: PathBuf,
    pub prepared_wav: PathBuf,
    pub channels: u16,
    pub native: EngineRoundtrip,
    pub bellard: EngineRoundtrip,
    pub roundtrip_pcm: PcmCompare,
    pub native_enc_bellard_dec: PcmCompare,
    pub bellard_enc_native_dec: PcmCompare,
}

impl ParityBenchReport {
    pub fn passes(&self, opts: &ParityOptions) -> bool {
        self.roundtrip_pcm
            .passes(opts.min_correlation, opts.max_mse)
            && self
                .native_enc_bellard_dec
                .passes(opts.min_correlation, opts.max_mse)
            && self
                .bellard_enc_native_dec
                .passes(opts.min_correlation, opts.max_mse)
    }

    pub fn print_summary(&self, opts: &ParityOptions) {
        let pass = self.passes(opts);
        eprintln!(
            "\n=== TSAC parity (native vs Bellard binary) ===\n\
             input: {}\n\
             prepared @ 44.1 kHz: {} ({} ch)\n\
             quality={} fast={}\n",
            self.input_wav.display(),
            self.prepared_wav.display(),
            self.channels,
            opts.quality,
            opts.fast
        );
        print_engine("native", &self.native);
        print_engine("bellard", &self.bellard);
        print_compare(
            "roundtrip PCM (native vs bellard)",
            &self.roundtrip_pcm,
            opts,
        );
        print_compare(
            "native encode → bellard decode",
            &self.native_enc_bellard_dec,
            opts,
        );
        print_compare(
            "bellard encode → native decode",
            &self.bellard_enc_native_dec,
            opts,
        );
        eprintln!("overall: {}", if pass { "PASS" } else { "FAIL" });
    }
}

pub fn bench_bellard_parity(
    install_dir: impl AsRef<Path>,
    in_wav: impl AsRef<Path>,
    opts: &ParityOptions,
) -> Result<ParityBenchReport> {
    if !tsac_binary_supported() {
        bail!(
            "Bellard parity bench requires Linux x86_64 (public tsac binary); \
             run on a Linux host or set RLX_TSAC_PARITY=1 in CI"
        );
    }
    let install_dir = install_dir.as_ref();
    ensure_tsac(install_dir)?;

    let in_wav = in_wav.as_ref().to_path_buf();
    let tag = std::process::id();
    let tmp = std::env::temp_dir().join(format!("rlx-tsac-parity-{tag}"));
    std::fs::create_dir_all(&tmp)?;

    let prepared = tmp.join("input_44100.wav");
    let channels = prepare_tsac_wav(&in_wav, &prepared)?;

    let base_opts = TsacOptions {
        quality: Some(opts.quality),
        fast: opts.fast,
        verbose: false,
        separate_stereo: false,
        channels: None,
        device: Device::Cpu,
        backend: TsacBackendKind::Auto,
    };

    let native_opts = TsacOptions {
        device: opts.native_device,
        backend: TsacBackendKind::Native,
        ..base_opts.clone()
    };
    let bellard_opts = TsacOptions {
        device: if opts.bellard_cuda {
            Device::Cuda
        } else {
            Device::Cpu
        },
        backend: TsacBackendKind::Bellard,
        ..base_opts
    };

    let native_codec = TsacCodec::open_with_options(install_dir, native_opts)?;
    let bellard_codec = TsacCodec::open_with_options(install_dir, bellard_opts)?;

    let native = run_engine_roundtrip(&native_codec, "native", &prepared, &tmp, channels)?;
    let bellard = run_engine_roundtrip(&bellard_codec, "bellard", &prepared, &tmp, channels)?;

    let native_enc_bellard_dec = cross_decode(
        &bellard_codec,
        &native.tsac_path,
        &tmp.join("native_enc_bellard_dec.wav"),
    )?;
    let bellard_enc_native_dec = cross_decode(
        &native_codec,
        &bellard.tsac_path,
        &tmp.join("bellard_enc_native_dec.wav"),
    )?;

    let roundtrip_pcm = PcmCompare::compare(&native.pcm, &bellard.pcm);
    let native_enc_bellard_dec_cmp = PcmCompare::compare(&native.pcm, &native_enc_bellard_dec);
    let bellard_enc_native_dec_cmp = PcmCompare::compare(&bellard.pcm, &bellard_enc_native_dec);

    Ok(ParityBenchReport {
        input_wav: in_wav,
        prepared_wav: prepared,
        channels,
        native,
        bellard,
        roundtrip_pcm,
        native_enc_bellard_dec: native_enc_bellard_dec_cmp,
        bellard_enc_native_dec: bellard_enc_native_dec_cmp,
    })
}

pub fn bench_bellard_parity_default(in_wav: impl AsRef<Path>) -> Result<ParityBenchReport> {
    let dir = resolve_install_dir(None);
    bench_bellard_parity(dir, in_wav, &ParityOptions::default())
}

fn run_engine_roundtrip(
    codec: &TsacCodec,
    label: &str,
    prepared_wav: &Path,
    tmp: &Path,
    _channels: u16,
) -> Result<EngineRoundtrip> {
    let tsac_path = tmp.join(format!("{label}.tsac"));
    let wav_path = tmp.join(format!("{label}_roundtrip.wav"));
    let encode = codec.encode(prepared_wav, &tsac_path)?;
    let t0 = Instant::now();
    codec.decode(&tsac_path, &wav_path)?;
    let decode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let pcm = audio::load_pcm_from_wav(&wav_path)?;
    Ok(EngineRoundtrip {
        encode,
        decode_ms,
        pcm,
        tsac_path,
        wav_path,
    })
}

fn cross_decode(codec: &TsacCodec, in_tsac: &Path, out_wav: &Path) -> Result<Vec<f32>> {
    codec.decode(in_tsac, out_wav)?;
    audio::load_pcm_from_wav(out_wav)
}

fn print_engine(label: &str, rt: &EngineRoundtrip) {
    eprintln!(
        "{label}: encode {:.1} ms, decode {:.1} ms, {} bytes compressed, {} samples",
        rt.encode.encode_ms,
        rt.decode_ms,
        rt.encode.output_bytes,
        rt.pcm.len()
    );
}

fn print_compare(label: &str, cmp: &PcmCompare, opts: &ParityOptions) {
    let ok = cmp.passes(opts.min_correlation, opts.max_mse);
    eprintln!(
        "{label}: corr={:.4} mse={:.6} max_abs={:.5} n={} [{}]",
        cmp.correlation,
        cmp.mse,
        cmp.max_abs,
        cmp.samples,
        if ok { "ok" } else { "over tolerance" }
    );
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
