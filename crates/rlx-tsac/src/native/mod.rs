//! Native TSAC inference (vendored [tsac-ng](https://github.com/Hope2333/tsac-ng)).
//!
//! Maps RLX [`Device`] tags to tsac-ng backends (CPU / CUDA / HIP / Vulkan).

pub(crate) mod ffi;

use crate::TsacOptions;
use crate::device::resolve_codec_device;
use crate::download::{dac_model_path, weights_available};
use anyhow::{Context, Result, bail};
use rlx_runtime::Device;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::ptr;

pub use ffi::TsacBackend;

pub struct NativeCodec {
    install_dir: PathBuf,
    ctx: *mut ffi::TSACContext,
    device: Device,
    /// RLX-graph decoder (cpu/metal/mlx/wgpu/…). Present when a GPU backend was
    /// requested and is available; decode then runs natively on that backend,
    /// bit-exact with the vendored C decoder. Encode stays on the C path.
    rlx_decoder: Option<crate::rlx_decode::RlxDecoder>,
}

impl NativeCodec {
    pub fn open(install_dir: impl AsRef<Path>, options: &TsacOptions) -> Result<Self> {
        let install_dir = install_dir.as_ref().to_path_buf();
        if !weights_available(&install_dir) {
            bail!(
                "missing TSAC weights under {} — run `just fetch-tsac`",
                install_dir.display()
            );
        }
        let device = resolve_codec_device(options.device);
        let backend = rlx_device_to_tsac_backend(device);
        let model_dir = CString::new(install_dir.to_string_lossy().into_owned())
            .context("install dir utf-8")?;
        let ctx = unsafe { ffi::tsac_init(backend, 0, model_dir.as_ptr()) };
        if ctx.is_null() {
            bail!("tsac_init failed for backend {backend:?} on {device:?}");
        }
        // Build the RLX-graph decoder for GPU backends (weights are copied out of
        // ctx, so it stays valid for the lifetime of this codec).
        let rlx_decoder = match rlx_graph_device(options.device) {
            Some(dev) => crate::rlx_decode::RlxDecoder::from_ctx(ctx, dev).ok(),
            None => None,
        };
        Ok(Self {
            install_dir,
            ctx,
            device,
            rlx_decoder,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Raw context pointer — used by the RLX graph backend to pull bit-exact
    /// dequantized weights via the C `dequant_weights` bridge.
    pub(crate) fn ctx_raw(&self) -> *mut ffi::TSACContext {
        self.ctx
    }

    pub fn encode_file(&self, in_wav: &Path, out_tsac: &Path, options: &TsacOptions) -> Result<()> {
        let in_c = path_to_c(in_wav)?;
        let out_c = path_to_c(out_tsac)?;
        let n_codebooks = options.quality.unwrap_or(12) as i32;
        let rc = unsafe {
            ffi::tsac_compress_file(self.ctx, in_c.as_ptr(), out_c.as_ptr(), n_codebooks)
        };
        if rc != ffi::TSAC_OK {
            bail!("tsac_compress_file failed ({rc})");
        }
        Ok(())
    }

    pub fn decode_file(&self, in_tsac: &Path, out_wav: &Path) -> Result<()> {
        // GPU backends decode through the RLX graph (bit-exact with C).
        if let Some(rlx) = &self.rlx_decoder {
            return rlx.decode_file(in_tsac, out_wav);
        }
        let in_c = path_to_c(in_tsac)?;
        let out_c = path_to_c(out_wav)?;
        let rc = unsafe { ffi::tsac_decompress_file(self.ctx, in_c.as_ptr(), out_c.as_ptr()) };
        if rc != ffi::TSAC_OK {
            bail!("tsac_decompress_file failed ({rc})");
        }
        Ok(())
    }

    pub fn model_hint(&self) -> PathBuf {
        dac_model_path(&self.install_dir, false)
    }
}

impl Drop for NativeCodec {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { ffi::tsac_free(self.ctx) };
            self.ctx = ptr::null_mut();
        }
    }
}

unsafe impl Send for NativeCodec {}

/// The rlx-runtime device to run the graph decoder on for a requested device, or
/// `None` to keep the C decode path (CPU, or backend unavailable at runtime).
fn rlx_graph_device(requested: Device) -> Option<Device> {
    match requested {
        Device::Metal
        | Device::Mlx
        | Device::Gpu
        | Device::Cuda
        | Device::Rocm
        | Device::Vulkan
            if rlx_runtime::is_available(requested) =>
        {
            Some(requested)
        }
        _ => None,
    }
}

fn path_to_c(path: &Path) -> Result<CString> {
    CString::new(path.to_string_lossy().into_owned()).context("path utf-8")
}

fn rlx_device_to_tsac_backend(device: Device) -> TsacBackend {
    match device {
        Device::Cuda => TsacBackend::Cuda,
        Device::Rocm => TsacBackend::Hip,
        Device::Vulkan | Device::Gpu => TsacBackend::Vulkan,
        // Metal / MLX: MoltenVK when vulkan feature is on; otherwise CPU SIMD.
        Device::Metal | Device::Mlx => {
            if cfg!(feature = "vulkan") {
                TsacBackend::Vulkan
            } else {
                TsacBackend::Cpu
            }
        }
        _ => TsacBackend::Cpu,
    }
}
