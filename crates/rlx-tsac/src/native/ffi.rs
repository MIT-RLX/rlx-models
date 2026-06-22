#![allow(non_camel_case_types, dead_code)]

use std::os::raw::c_char;

pub const TSAC_OK: i32 = 0;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TsacBackend {
    Cpu = 0,
    Cuda = 1,
    Hip = 2,
    Vulkan = 3,
    Llvm = 4,
}

pub enum TSACContext {}

unsafe extern "C" {
    pub fn tsac_init(
        backend: TsacBackend,
        n_threads: i32,
        model_path: *const c_char,
    ) -> *mut TSACContext;
    pub fn tsac_free(ctx: *mut TSACContext);
    pub fn tsac_compress_file(
        ctx: *mut TSACContext,
        in_wav: *const c_char,
        out_txc: *const c_char,
        n_codebooks: i32,
    ) -> i32;
    pub fn tsac_decompress_file(
        ctx: *mut TSACContext,
        in_txc: *const c_char,
        out_wav: *const c_char,
    ) -> i32;
    pub fn tsac_version() -> *const c_char;

    // ---- RLX graph backend bridge (bit-exact dequantized weight export) ----

    /// Dequantize a conv/conv-transpose/projection layer addressed by `prefix`.
    /// Returns a malloc'd row-major `[Co][Ci][K]` f32 buffer (free with
    /// [`tsac_free_buffer`]); fills `Co`/`Ci`/`K` and `is_ct`. NULL if absent.
    pub fn tsac_rlx_conv_weight(
        ctx: *mut TSACContext,
        prefix: *const c_char,
        co: *mut i32,
        ci: *mut i32,
        k: *mut i32,
        is_ct: *mut i32,
    ) -> *mut f32;

    /// Copy a plain f32 tensor (snake alpha / bias / codebook) by exact name.
    /// Returns malloc'd buffer (free with [`tsac_free_buffer`]); `n` = count.
    pub fn tsac_rlx_f32(ctx: *mut TSACContext, name: *const c_char, n: *mut i32) -> *mut f32;

    /// Raw q8-dequantized `{prefix}.weight_v` in on-disk `[d0][d1][d2]` layout
    /// (no rearrange / weight_g / L2). For the standard-DAC weight-norm path.
    pub fn tsac_rlx_weight_v_raw(
        ctx: *mut TSACContext,
        prefix: *const c_char,
        d0: *mut i32,
        d1: *mut i32,
        d2: *mut i32,
    ) -> *mut f32;

    /// Read RVQ code indices from a `.txc`/`.tsac` file: malloc'd int buffer of
    /// length `n_frames * n_cb`, row-major `[frame][codebook]`.
    pub fn tsac_rlx_read_codes(
        in_txc: *const c_char,
        n_frames: *mut i32,
        n_cb: *mut i32,
    ) -> *mut i32;

    pub fn tsac_free_buffer(ptr: *mut std::os::raw::c_void);
}
