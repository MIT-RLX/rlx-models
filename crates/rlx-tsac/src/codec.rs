use crate::device::{resolve_codec_device, resolve_rlx_device};
use crate::download::resolve_tsac_bin;
use crate::platform::tsac_binary_supported;
use anyhow::{Context, Result, bail};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[cfg(feature = "native-codec")]
use crate::native::NativeCodec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TsacBackendKind {
    /// Prefer the correct codec (descript weights) when available, else native.
    #[default]
    Auto,
    /// Faithful tsac-ng C codec (decodes existing tsac-ng files bit-exact).
    Native,
    /// Original Bellard binary (Linux x86_64 only).
    Bellard,
    /// Correct TSAC = Descript-DAC-44kHz on RLX backends (recommended; the q8
    /// weights are an un-dequantizable libnc BF8 format, so this uses the real
    /// descript weights, which Bellard's tsac itself uses).
    Correct,
}

#[derive(Debug, Clone)]
pub struct TsacOptions {
    pub device: Device,
    pub backend: TsacBackendKind,
    pub quality: Option<u8>,
    pub fast: bool,
    pub separate_stereo: bool,
    pub channels: Option<u8>,
    pub verbose: bool,
}

impl Default for TsacOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            backend: TsacBackendKind::Auto,
            quality: None,
            fast: false,
            separate_stereo: false,
            channels: None,
            verbose: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EncodeStats {
    pub encode_ms: f64,
    pub output_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct RoundtripStats {
    pub encode_ms: f64,
    pub decode_ms: f64,
    pub tsac_bytes: u64,
}

enum CodecEngine {
    #[cfg(feature = "native-codec")]
    Native(NativeCodec),
    External(ExternalCodec),
    Correct(crate::correct::CorrectCodec),
}

struct ExternalCodec {
    install_dir: PathBuf,
    options: TsacOptions,
    device: Device,
}

pub struct TsacCodec {
    install_dir: PathBuf,
    options: TsacOptions,
    device: Device,
    engine: CodecEngine,
}

impl TsacCodec {
    pub fn open(install_dir: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_options(install_dir, TsacOptions::default())
    }

    pub fn open_native(install_dir: impl AsRef<Path>, options: TsacOptions) -> Result<Self> {
        let mut opts = options;
        opts.backend = TsacBackendKind::Native;
        Self::open_with_options(install_dir, opts)
    }

    pub fn open_bellard(install_dir: impl AsRef<Path>, options: TsacOptions) -> Result<Self> {
        let mut opts = options;
        opts.backend = TsacBackendKind::Bellard;
        Self::open_with_options(install_dir, opts)
    }

    pub fn open_with_options(install_dir: impl AsRef<Path>, options: TsacOptions) -> Result<Self> {
        let install_dir = install_dir.as_ref().to_path_buf();
        // Keep the originally-requested device in the options so the native engine
        // can route decode to the RLX graph backend (metal/mlx/wgpu/…); the C
        // codec resolves its own fallback internally. `device` is the resolved
        // device reported by the codec.
        // The `Correct` codec runs the DAC purely on an rlx backend (cpu/metal/
        // mlx/wgpu), so its device is the requested rlx device — not the
        // libnc-oriented cuda/vulkan resolution the native/Bellard engines use.
        let device = match options.backend {
            TsacBackendKind::Correct => resolve_rlx_device(options.device),
            _ => resolve_codec_device(options.device),
        };
        let engine = open_engine(&install_dir, &options, options.device)?;
        Ok(Self {
            install_dir,
            options,
            device,
            engine,
        })
    }

    pub fn with_options(mut self, options: TsacOptions) -> Self {
        self.options = options.clone();
        self.device = match options.backend {
            TsacBackendKind::Correct => resolve_rlx_device(options.device),
            _ => resolve_codec_device(options.device),
        };
        if let Ok(engine) = open_engine(&self.install_dir, &self.options, self.options.device) {
            self.engine = engine;
        }
        self
    }

    pub fn install_dir(&self) -> &Path {
        &self.install_dir
    }

    pub fn options(&self) -> &TsacOptions {
        &self.options
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn encode(
        &self,
        in_audio: impl AsRef<Path>,
        out_tsac: impl AsRef<Path>,
    ) -> Result<EncodeStats> {
        let in_audio = in_audio.as_ref();
        let out_tsac = out_tsac.as_ref();
        ensure_parent(out_tsac)?;
        let t0 = Instant::now();
        match &self.engine {
            #[cfg(feature = "native-codec")]
            CodecEngine::Native(native) => {
                native.encode_file(in_audio, out_tsac, &self.options)?;
            }
            CodecEngine::Correct(correct) => {
                correct.encode_file(in_audio, out_tsac)?;
            }
            CodecEngine::External(ext) => {
                ext.run_tsac(&["c", in_audio.to_str().unwrap(), out_tsac.to_str().unwrap()])?;
            }
        }
        let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let output_bytes = std::fs::metadata(out_tsac)
            .with_context(|| format!("stat {}", out_tsac.display()))?
            .len();
        Ok(EncodeStats {
            encode_ms,
            output_bytes,
        })
    }

    pub fn decode(&self, in_tsac: impl AsRef<Path>, out_wav: impl AsRef<Path>) -> Result<()> {
        let in_tsac = in_tsac.as_ref();
        let out_wav = out_wav.as_ref();
        ensure_parent(out_wav)?;
        match &self.engine {
            #[cfg(feature = "native-codec")]
            CodecEngine::Native(native) => native.decode_file(in_tsac, out_wav)?,
            CodecEngine::Correct(correct) => correct.decode_file(in_tsac, out_wav)?,
            CodecEngine::External(ext) => {
                ext.run_tsac(&["d", in_tsac.to_str().unwrap(), out_wav.to_str().unwrap()])?;
            }
        }
        Ok(())
    }

    pub fn roundtrip(
        &self,
        in_audio: impl AsRef<Path>,
        out_wav: impl AsRef<Path>,
    ) -> Result<RoundtripStats> {
        let in_audio = in_audio.as_ref();
        let out_wav = out_wav.as_ref();
        let tmp = std::env::temp_dir().join(format!(
            "rlx-tsac-{}-{}.tsac",
            std::process::id(),
            in_audio
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("audio")
        ));
        let encode = self.encode(in_audio, &tmp)?;
        let t0 = Instant::now();
        self.decode(&tmp, out_wav)?;
        let decode_ms = t0.elapsed().as_secs_f64() * 1000.0;
        std::fs::remove_file(&tmp).ok();
        Ok(RoundtripStats {
            encode_ms: encode.encode_ms,
            decode_ms,
            tsac_bytes: encode.output_bytes,
        })
    }
}

fn open_engine(install_dir: &Path, options: &TsacOptions, device: Device) -> Result<CodecEngine> {
    match options.backend {
        TsacBackendKind::Correct => {
            let correct = crate::correct::CorrectCodec::open(options.device, options.quality)?;
            return Ok(CodecEngine::Correct(correct));
        }
        TsacBackendKind::Bellard => {
            if !tsac_binary_supported() {
                bail!("Bellard backend requires Linux x86_64");
            }
            if !resolve_tsac_bin(install_dir).is_file() {
                bail!(
                    "missing Bellard tsac binary under {}",
                    install_dir.display()
                );
            }
            return Ok(CodecEngine::External(ExternalCodec {
                install_dir: install_dir.to_path_buf(),
                options: options.clone(),
                device,
            }));
        }
        TsacBackendKind::Native => {
            #[cfg(feature = "native-codec")]
            {
                let native = NativeCodec::open(install_dir, options)?;
                return Ok(CodecEngine::Native(native));
            }
            #[cfg(not(feature = "native-codec"))]
            {
                bail!("native codec requires `native-codec` feature");
            }
        }
        TsacBackendKind::Auto => {}
    }

    // Auto: prefer the correct codec (descript weights) when they're already
    // present — a drop-in, correct, all-backend encode+decode path. Falls back to
    // the faithful native codec otherwise (which can still decode tsac-ng files).
    if crate::correct::weights_available(&crate::correct::default_dir())
        && std::env::var("RLX_TSAC_ENGINE").ok().as_deref() != Some("native")
    {
        if let Ok(correct) = crate::correct::CorrectCodec::open(options.device, options.quality) {
            return Ok(CodecEngine::Correct(correct));
        }
    }

    if prefer_external(install_dir) {
        return Ok(CodecEngine::External(ExternalCodec {
            install_dir: install_dir.to_path_buf(),
            options: options.clone(),
            device,
        }));
    }

    #[cfg(feature = "native-codec")]
    {
        let native = NativeCodec::open(install_dir, options)?;
        if device != Device::Cpu
            && native.device() == Device::Cpu
            && resolve_codec_device(options.device) != device
        {
            eprintln!(
                "tsac: requested {device:?} — running CPU native codec (enable backend features for GPU)"
            );
        }
        Ok(CodecEngine::Native(native))
    }

    #[cfg(not(feature = "native-codec"))]
    {
        let _ = (install_dir, options, device);
        bail!(
            "no TSAC engine available — enable `native-codec` or use Linux x86_64 with Bellard binary"
        );
    }
}

fn prefer_external(install_dir: &Path) -> bool {
    if std::env::var("RLX_TSAC_ENGINE").ok().as_deref() == Some("bellard") {
        return tsac_binary_supported() && resolve_tsac_bin(install_dir).is_file();
    }
    #[cfg(not(feature = "native-codec"))]
    {
        return tsac_binary_supported() && resolve_tsac_bin(install_dir).is_file();
    }
    #[cfg(feature = "native-codec")]
    {
        false
    }
}

impl ExternalCodec {
    fn run_tsac(&self, subcommand_and_paths: &[&str]) -> Result<()> {
        let install_dir = &self.install_dir;
        let bin = resolve_tsac_bin(install_dir);
        if !bin.is_file() {
            bail!(
                "TSAC binary missing at {} — run `just fetch-tsac`",
                bin.display()
            );
        }
        let mut cmd = Command::new(&bin);
        cmd.current_dir(install_dir);
        cmd.env("LD_LIBRARY_PATH", ld_library_path(install_dir));
        if matches!(self.device, Device::Cuda) {
            cmd.arg("--cuda");
        }
        if self.options.verbose {
            cmd.arg("-v");
        }
        if let Some(q) = self.options.quality {
            cmd.arg("-q").arg(q.to_string());
        }
        if self.options.fast {
            cmd.arg("-f");
        }
        if self.options.separate_stereo {
            cmd.arg("-s");
        }
        if let Some(ch) = self.options.channels {
            cmd.arg("-c").arg(ch.to_string());
        }
        cmd.args(subcommand_and_paths);
        let output = cmd
            .output()
            .with_context(|| format!("run {}", bin.display()))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            bail!(
                "tsac failed (status {})\nstdout:\n{stdout}\nstderr:\n{stderr}",
                output.status
            );
        }
        Ok(())
    }
}

fn ld_library_path(install_dir: &Path) -> String {
    let dir = install_dir.to_string_lossy();
    match std::env::var("LD_LIBRARY_PATH") {
        Ok(existing) if !existing.is_empty() => format!("{dir}:{existing}"),
        _ => dir.into_owned(),
    }
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create dir {}", parent.display()))?;
        }
    }
    Ok(())
}

/// Unified [`rlx_core::FileCodec`] view of TSAC (file-bitstream compressor).
/// TSAC has no RVQ codes, so it implements `FileCodec` rather than the
/// frame/quantizer `rlx_core::AudioCodec` trait used by mimi/dac.
impl rlx_core::FileCodec for TsacCodec {
    fn device(&self) -> Device {
        self.device
    }

    fn sample_rate(&self) -> u32 {
        crate::SAMPLE_RATE
    }

    fn encode_file(
        &self,
        in_audio: &Path,
        out_compressed: &Path,
    ) -> Result<rlx_core::CompressStats> {
        let s = TsacCodec::encode(self, in_audio, out_compressed)?;
        Ok(rlx_core::CompressStats {
            compressed_bytes: s.output_bytes,
            encode_ms: s.encode_ms,
            decode_ms: 0.0,
        })
    }

    fn decode_file(&self, in_compressed: &Path, out_wav: &Path) -> Result<()> {
        TsacCodec::decode(self, in_compressed, out_wav)
    }

    fn roundtrip_file(&self, in_audio: &Path, out_wav: &Path) -> Result<rlx_core::CompressStats> {
        let s = TsacCodec::roundtrip(self, in_audio, out_wav)?;
        Ok(rlx_core::CompressStats {
            compressed_bytes: s.tsac_bytes,
            encode_ms: s.encode_ms,
            decode_ms: s.decode_ms,
        })
    }
}
