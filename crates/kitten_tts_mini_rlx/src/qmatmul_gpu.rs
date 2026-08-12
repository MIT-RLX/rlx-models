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

//! Metal GPU GEMM path for `onnx.QMatMul`: dequant → f32 sgemm → output.

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};
    use std::sync::{Mutex, OnceLock};

    use rlx_metal::blas::buffers_sgemm_sync;
    use rlx_metal::device::metal_device;
    use rlx_metal::mtl::Buffer;

    fn env_flag(name: &str) -> bool {
        std::env::var(name).is_ok_and(|v| {
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
        })
    }

    pub fn qmatmul_gpu_enabled() -> bool {
        if std::env::var("KITTEN_RLX_QMATMUL_GPU").is_ok_and(|v| {
            v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("no")
        }) {
            return false;
        }
        env_flag("KITTEN_RLX_QMATMUL_GPU")
    }

    fn gpu_min_flops() -> usize {
        std::env::var("KITTEN_RLX_QMATMUL_GPU_MIN_FLOPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2_097_152)
    }

    fn matmul_dims(act_shape: &[usize], w_shape: &[usize]) -> (usize, usize, usize) {
        let k = w_shape.first().copied().filter(|&d| d > 0).unwrap_or(1);
        let n = w_shape.get(1).copied().filter(|&d| d > 0).unwrap_or(1);
        let m = if act_shape.len() >= 3 {
            act_shape[act_shape.len() - 2].max(1)
        } else if act_shape.len() == 2 {
            act_shape[0].max(1)
        } else {
            1
        };
        (m, k, n)
    }

    struct GemmScratch {
        a_buf: Buffer,
        b_buf: Buffer,
        c_buf: Buffer,
        a_cap: usize,
        b_cap: usize,
        c_cap: usize,
    }

    impl GemmScratch {
        fn ensure(&mut self, m: usize, k: usize, n: usize) {
            let dev = metal_device().expect("metal device");
            let a_need = m * k;
            let b_need = k * n;
            let c_need = m * n;
            if a_need > self.a_cap {
                self.a_buf = dev.alloc_shared(a_need * 4);
                self.a_cap = a_need;
            }
            if b_need > self.b_cap {
                self.b_buf = dev.alloc_shared(b_need * 4);
                self.b_cap = b_need;
            }
            if c_need > self.c_cap {
                self.c_buf = dev.alloc_shared(c_need * 4);
                self.c_cap = c_need;
            }
        }
    }

    thread_local! {
        static SCRATCH: RefCell<Option<GemmScratch>> = const { RefCell::new(None) };
    }

    fn with_scratch<R>(m: usize, k: usize, n: usize, f: impl FnOnce(&GemmScratch) -> R) -> R {
        SCRATCH.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.is_none() {
                let dev = metal_device().expect("metal device");
                *slot = Some(GemmScratch {
                    a_buf: dev.alloc_shared(1),
                    b_buf: dev.alloc_shared(1),
                    c_buf: dev.alloc_shared(1),
                    a_cap: 0,
                    b_cap: 0,
                    c_cap: 0,
                });
            }
            let scratch = slot.as_mut().expect("scratch");
            scratch.ensure(m, k, n);
            f(scratch)
        })
    }

    fn weight_cache() -> &'static Mutex<HashMap<u64, Buffer>> {
        static CACHE: OnceLock<Mutex<HashMap<u64, Buffer>>> = OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn weight_cache_key(w: &[i8], k: usize, n: usize, w_zp: i32, w_scale: f32) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut h = DefaultHasher::new();
        for &b in w {
            b.hash(&mut h);
        }
        k.hash(&mut h);
        n.hash(&mut h);
        w_zp.hash(&mut h);
        w_scale.to_bits().hash(&mut h);
        h.finish()
    }

    fn dequant_act_f32(
        act_q: &[u8],
        m: usize,
        k: usize,
        act_zp: u8,
        act_scale: f32,
        dst: &mut [f32],
    ) {
        let az = act_zp as f32;
        let n = m * k;
        for i in 0..n {
            dst[i] = (act_q[i] as f32 - az) * act_scale;
        }
    }

    fn dequant_weight_f32(w: &[i8], k: usize, n: usize, w_zp: i32, w_scale: f32, dst: &mut [f32]) {
        let wz = w_zp as f32;
        let len = k * n;
        for i in 0..len {
            dst[i] = (w[i] as f32 - wz) * w_scale;
        }
    }

    fn cached_weight_buffer(w: &[i8], k: usize, n: usize, w_zp: i32, w_scale: f32) -> Buffer {
        let key = weight_cache_key(w, k, n, w_zp, w_scale);
        if let Some(hit) = weight_cache().lock().expect("w cache").get(&key) {
            return hit.clone();
        }
        let dev = metal_device().expect("metal device");
        let buf = dev.alloc_shared(k * n * 4);
        let dst = unsafe { std::slice::from_raw_parts_mut(buf.contents() as *mut f32, k * n) };
        dequant_weight_f32(w, k, n, w_zp, w_scale, dst);
        weight_cache()
            .lock()
            .expect("w cache")
            .insert(key, buf.clone());
        buf
    }

    pub fn try_qmatmul_uint8_gpu_into(
        act_q: &[u8],
        act_shape: &[usize],
        act_scale: f32,
        act_zp: u8,
        w: &[i8],
        w_shape: &[usize],
        w_scale: f32,
        w_zp: i32,
        out: &mut [f32],
    ) -> bool {
        if !qmatmul_gpu_enabled() || metal_device().is_none() {
            return false;
        }
        let (m, k, n) = matmul_dims(act_shape, w_shape);
        if m == 0 || k == 0 || n == 0 || act_q.len() < m * k || w.len() < k * n || out.len() < m * n
        {
            return false;
        }
        if m * k * n < gpu_min_flops() {
            return false;
        }

        let b_buf = cached_weight_buffer(w, k, n, w_zp, w_scale);
        with_scratch(m, k, n, |scratch| {
            let a_slice = unsafe {
                std::slice::from_raw_parts_mut(scratch.a_buf.contents() as *mut f32, m * k)
            };
            dequant_act_f32(act_q, m, k, act_zp, act_scale, a_slice);
            buffers_sgemm_sync(&scratch.a_buf, &b_buf, &scratch.c_buf, m, k, n);
            let c_slice = unsafe {
                std::slice::from_raw_parts(scratch.c_buf.contents() as *const f32, m * n)
            };
            out[..m * n].copy_from_slice(c_slice);
        });
        true
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal::{qmatmul_gpu_enabled, try_qmatmul_uint8_gpu_into};

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn qmatmul_gpu_enabled() -> bool {
    false
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn try_qmatmul_uint8_gpu_into(
    _act_q: &[u8],
    _act_shape: &[usize],
    _act_scale: f32,
    _act_zp: u8,
    _w: &[i8],
    _w_shape: &[usize],
    _w_scale: f32,
    _w_zp: i32,
    _out: &mut [f32],
) -> bool {
    false
}
