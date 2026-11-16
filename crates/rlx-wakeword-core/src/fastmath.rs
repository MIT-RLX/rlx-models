// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! `exp` and `ln` for f32 as short polynomials (Cephes' `expf` / `logf`, ~1–2 ulp), one value
//! at a time or four at once with NEON. Both forms run the same fused multiply-adds in the same
//! order, so they give the same bits: PCEN computed while training (on any CPU) and on the robot
//! (NEON) agree exactly. Several times faster than the library calls, and vectorizable.

const LOG2E: f32 = core::f32::consts::LOG2_E;
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;
const EXP_P: [f32; 6] = [
    1.987_569_1e-4,
    1.398_199_9e-3,
    8.333_452e-3,
    4.166_579_6e-2,
    0.166_666_65,
    0.5,
];
const LOG_P: [f32; 9] = [
    7.037_683_6e-2,
    -0.115_146_1,
    0.116_769_98,
    -0.124_201_41,
    0.142_493_23,
    -0.166_680_57,
    0.200_007_15,
    -0.249_999_94,
    0.333_333_3,
];
const SQRT_HALF: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// `e^x` (inputs clamped to f32's range).
#[inline]
pub fn exp(x: f32) -> f32 {
    let x = x.clamp(-87.3, 88.7);
    let n = (x * LOG2E + 0.5).floor();
    let r = n.mul_add(-LN2_HI, x);
    let r = n.mul_add(-LN2_LO, r);
    let mut p = EXP_P[0];
    for &c in &EXP_P[1..] {
        p = p.mul_add(r, c);
    }
    let r2 = r * r;
    let y = p.mul_add(r2, r) + 1.0;
    y * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// `ln x` for `x > 0` (normal numbers).
#[inline]
pub fn ln(x: f32) -> f32 {
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xff) as i32 - 126;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f00_0000); // [0.5, 1)
    if m < SQRT_HALF {
        e -= 1;
        m = m + m - 1.0;
    } else {
        m -= 1.0;
    }
    let ef = e as f32;
    let z = m * m;
    let mut p = LOG_P[0];
    for &c in &LOG_P[1..] {
        p = p.mul_add(m, c);
    }
    let y = (p * m) * z;
    let y = ef.mul_add(LN2_LO, y);
    let y = (-0.5f32).mul_add(z, y);
    ef.mul_add(LN2_HI, m + y)
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use super::*;
    use core::arch::aarch64::*;

    /// [`super::exp`] on four lanes.
    #[inline]
    pub fn exp4(x: float32x4_t) -> float32x4_t {
        // SAFETY: NEON is part of aarch64.
        unsafe {
            let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.3)), vdupq_n_f32(88.7));
            let n = vrndmq_f32(vaddq_f32(vmulq_f32(x, vdupq_n_f32(LOG2E)), vdupq_n_f32(0.5)));
            let r = vfmaq_f32(x, n, vdupq_n_f32(-LN2_HI));
            let r = vfmaq_f32(r, n, vdupq_n_f32(-LN2_LO));
            let mut p = vdupq_n_f32(EXP_P[0]);
            for &c in &EXP_P[1..] {
                p = vfmaq_f32(vdupq_n_f32(c), p, r);
            }
            let r2 = vmulq_f32(r, r);
            let y = vaddq_f32(vfmaq_f32(r, p, r2), vdupq_n_f32(1.0));
            let scale = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127))));
            vmulq_f32(y, scale)
        }
    }

    /// [`super::ln`] on four lanes.
    #[inline]
    pub fn ln4(x: float32x4_t) -> float32x4_t {
        // SAFETY: NEON is part of aarch64.
        unsafe {
            let bits = vreinterpretq_u32_f32(x);
            let e = vsubq_s32(
                vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32::<23>(bits), vdupq_n_u32(0xff))),
                vdupq_n_s32(126),
            );
            let m = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(bits, vdupq_n_u32(0x007f_ffff)), vdupq_n_u32(0x3f00_0000)));
            let small = vcltq_f32(m, vdupq_n_f32(SQRT_HALF));
            let e = vsubq_s32(e, vreinterpretq_s32_u32(vandq_u32(small, vdupq_n_u32(1))));
            // m + m − 1 where small, else m − 1.
            let m = vsubq_f32(vbslq_f32(small, vaddq_f32(m, m), m), vdupq_n_f32(1.0));
            let ef = vcvtq_f32_s32(e);
            let z = vmulq_f32(m, m);
            let mut p = vdupq_n_f32(LOG_P[0]);
            for &c in &LOG_P[1..] {
                p = vfmaq_f32(vdupq_n_f32(c), p, m);
            }
            let y = vmulq_f32(vmulq_f32(p, m), z);
            let y = vfmaq_f32(y, ef, vdupq_n_f32(LN2_LO));
            let y = vfmaq_f32(y, vdupq_n_f32(-0.5), z);
            vfmaq_f32(vaddq_f32(m, y), ef, vdupq_n_f32(LN2_HI))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_to_the_library() {
        for i in 0..20_000 {
            let x = -80.0 + i as f32 * 0.0083;
            let (a, b) = (exp(x), x.exp());
            assert!(((a - b) / b).abs() < 3e-7, "exp({x}) = {a}, want {b}");
            let y = 1e-30f32 * (1.0 + i as f32).powf(6.0);
            let (a, b) = (ln(y), y.ln());
            assert!((a - b).abs() <= 3e-7 * b.abs().max(1.0), "ln({y}) = {a}, want {b}");
        }
    }

    /// The NEON lanes give the scalar forms' bits.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_bit_for_bit() {
        use core::arch::aarch64::*;
        for i in 0..5_000 {
            let xs = [-60.0 + i as f32 * 0.024, 0.37 * i as f32 - 900.0, 1.5, -0.25 - i as f32 * 1e-3];
            let ys = xs.map(|x| x.abs() + 1e-6);
            let (mut e, mut l) = ([0.0f32; 4], [0.0f32; 4]);
            // SAFETY: NEON is part of aarch64; four-float arrays.
            unsafe {
                vst1q_f32(e.as_mut_ptr(), neon::exp4(vld1q_f32(xs.as_ptr())));
                vst1q_f32(l.as_mut_ptr(), neon::ln4(vld1q_f32(ys.as_ptr())));
            }
            for k in 0..4 {
                assert_eq!(e[k].to_bits(), exp(xs[k]).to_bits(), "exp({})", xs[k]);
                assert_eq!(l[k].to_bits(), ln(ys[k]).to_bits(), "ln({})", ys[k]);
            }
        }
    }
}
