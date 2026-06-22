//! Host RLX perf bench vs Docker reference binaries (Bellard + tsac-ng).

use crate::audio::{self, PcmCompare, prepare_tsac_wav};
use crate::codec::{TsacBackendKind, TsacCodec, TsacOptions};
use crate::docker_ref::{DockerRefOptions, RefRoundtrip, run_docker_ref_roundtrip};
use crate::download::weights_available;
use crate::parity::EngineRoundtrip;
use anyhow::{Result, bail};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct PerfOptions {
    pub quality: u8,
    pub fast: bool,
    pub native_device: Device,
    pub docker: DockerRefOptions,
    pub min_correlation: f32,
    pub max_mse: f32,
}

impl Default for PerfOptions {
    fn default() -> Self {
        Self {
            quality: 9,
            fast: true,
            native_device: Device::Cpu,
            docker: DockerRefOptions::default(),
            min_correlation: env_f32("RLX_TSAC_PARITY_MIN_CORR", 0.92),
            max_mse: env_f32("RLX_TSAC_PARITY_MAX_MSE", 0.0025),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerfBenchReport {
    pub input_wav: PathBuf,
    pub prepared_wav: PathBuf,
    pub channels: u16,
    pub rlx: EngineRoundtrip,
    pub bellard: RefRoundtrip,
    pub tsac_ng: RefRoundtrip,
    pub rlx_vs_bellard: PcmCompare,
    pub rlx_vs_tsac_ng: PcmCompare,
}

impl PerfBenchReport {
    pub fn print_summary(&self, opts: &PerfOptions) {
        eprintln!(
            "\n=== TSAC perf (RLX host vs Docker refs) ===\n\
             input: {}\n\
             prepared @ 44.1 kHz: {} ({} ch)\n\
             quality={} fast={} device={:?}\n",
            self.input_wav.display(),
            self.prepared_wav.display(),
            self.channels,
            opts.quality,
            opts.fast,
            opts.native_device,
        );
        print_engine("rlx (host)", &self.rlx);
        print_ref("bellard (docker)", &self.bellard);
        print_ref("tsac-ng (docker)", &self.tsac_ng);
        print_compare("RLX vs bellard PCM", &self.rlx_vs_bellard, opts);
        print_compare("RLX vs tsac-ng PCM", &self.rlx_vs_tsac_ng, opts);
    }
}

pub fn bench_perf(
    install_dir: impl AsRef<Path>,
    in_wav: impl AsRef<Path>,
    opts: &PerfOptions,
) -> Result<PerfBenchReport> {
    let install_dir = install_dir.as_ref();
    if !weights_available(install_dir) {
        bail!(
            "TSAC weights missing under {} — run `just fetch-tsac`",
            install_dir.display()
        );
    }
    let in_wav = in_wav.as_ref().to_path_buf();
    let tag = std::process::id();
    let work = std::env::temp_dir().join(format!("rlx-tsac-perf-{tag}"));
    std::fs::create_dir_all(&work)?;

    let prepared = work.join("input_44100.wav");
    let channels = prepare_tsac_wav(&in_wav, &prepared)?;

    let bellard = run_docker_ref_roundtrip(
        &opts.docker,
        "bellard",
        &prepared,
        &work,
        opts.quality,
        opts.fast,
    )?;
    let tsac_ng = run_docker_ref_roundtrip(
        &opts.docker,
        "tsac-ng",
        &prepared,
        &work,
        opts.quality,
        opts.fast,
    )?;

    let native_opts = TsacOptions {
        quality: Some(opts.quality),
        fast: opts.fast,
        verbose: false,
        separate_stereo: false,
        channels: None,
        device: opts.native_device,
        backend: TsacBackendKind::Native,
    };
    let codec = TsacCodec::open_with_options(install_dir, native_opts)?;
    let rlx = run_rlx_roundtrip(&codec, "rlx", &prepared, &work)?;

    let bellard_pcm = audio::load_pcm_from_wav(&bellard.wav_path)?;
    let tsac_ng_pcm = audio::load_pcm_from_wav(&tsac_ng.wav_path)?;
    let rlx_vs_bellard = PcmCompare::compare(&rlx.pcm, &bellard_pcm);
    let rlx_vs_tsac_ng = PcmCompare::compare(&rlx.pcm, &tsac_ng_pcm);

    Ok(PerfBenchReport {
        input_wav: in_wav,
        prepared_wav: prepared,
        channels,
        rlx,
        bellard,
        tsac_ng,
        rlx_vs_bellard,
        rlx_vs_tsac_ng,
    })
}

fn run_rlx_roundtrip(
    codec: &TsacCodec,
    label: &str,
    prepared_wav: &Path,
    work: &Path,
) -> Result<EngineRoundtrip> {
    let tsac_path = work.join(format!("{label}.tsac"));
    let wav_path = work.join(format!("{label}_roundtrip.wav"));
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

fn print_engine(label: &str, rt: &EngineRoundtrip) {
    eprintln!(
        "{label}: encode {:.1} ms, decode {:.1} ms, {} bytes compressed, {} samples",
        rt.encode.encode_ms,
        rt.decode_ms,
        rt.encode.output_bytes,
        rt.pcm.len()
    );
}

fn print_ref(label: &str, rt: &RefRoundtrip) {
    eprintln!(
        "{label}: encode {:.1} ms, decode {:.1} ms, {} bytes compressed",
        rt.encode_ms, rt.decode_ms, rt.output_bytes
    );
}

fn print_compare(label: &str, cmp: &PcmCompare, opts: &PerfOptions) {
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
