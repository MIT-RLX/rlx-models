/*
 * cpu_riscv.c — RISC-V Vector Extension (RVV) SIMD acceleration for TSAC.
 *
 * RISC-V baseline: RV64GC (IMAFDC) + Linux ABI
 * Optional: V (Vector Extension) 1.0+ for SIMD via C intrinsics
 *
 * RVV C intrinsics API: v0.12+/v1.0 (__riscv_v* prefix)
 * Compile with: -march=rv64gcv (for RVV support)
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifdef __riscv

/* Detect RISC-V features via /proc/cpuinfo (Linux) */
int riscv_has_rvv = 0;

int cpu_arch_init(void) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char buf[1024];
        while (fgets(buf, sizeof(buf), f))
            if (strstr(buf, "riscv") || strstr(buf, "rv64"))
                if (strstr(buf, "v"))
                    riscv_has_rvv = 1;
        fclose(f);
    }
    return 0; /* RISC-V baseline always works (scalar) */
}

const char *cpu_arch_name(void) {
    if (riscv_has_rvv) return "RISC-V RVV";
    return "RISC-V scalar";
}

/* Get RVV detection result for external dispatch */
int riscv_rvv_available(void) {
    return riscv_has_rvv;
}

/* ================================================================ */
/*  Scalar fallback implementations (for non-RVV or as fallback)      */
/* ================================================================ */

static void conv1d_scalar(float *o, const float *x, const float *w, const float *b,
                          int T, int K, int Ci, int Co) {
    int P = K/2;
    for (int oc = 0; oc < Co; oc++)
        for (int oi = 0; oi < T; oi++) {
            float s = b ? b[oc] : 0;
            for (int ic = 0; ic < Ci; ic++)
                for (int j = 0; j < K; j++) {
                    int ii = oi + j - P;
                    if (ii >= 0 && ii < T)
                        s += x[ic*T+ii] * w[oc*Ci*K + ic*K + j];
                }
            o[oc*T+oi] = s;
        }
}

static void snake_scalar(float *o, const float *x, const float *a, int n, int C) {
    for (int i = 0; i < n; i++) {
        float v = x[i], al = a[i%C];
        if (al < 1e-6f) al = 1e-6f;
        float sa = sinf(al*v);
        o[i] = v + sa*sa/al;
    }
}

static void add_scalar(float *o, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        o[i] = a[i] + b[i];
    }
}

/* ================================================================ */
/*  RVV SIMD implementations using C intrinsics (v0.12+/v1.0)         */
/* ================================================================ */

#ifdef __riscv_v
#include <riscv_vector.h>

/*
 * Conv1D with RVV vectorization
 * Performs 1D convolution with vectorized inner loops
 */
void conv1d_rvv(float *o, const float *x, const float *w, const float *b,
                int T, int K, int Ci, int Co) {
    int P = K/2;
    size_t vl;
    
    for (int oc = 0; oc < Co; oc++) {
        float bias = b ? b[oc] : 0.0f;
        for (int oi = 0; oi < T; oi++) {
            vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(bias, 1);
            
            for (int ic = 0; ic < Ci; ic += vl) {
                vl = __riscv_vsetvl_e32m1(Ci - ic);
                
                for (int j = 0; j < K; j++) {
                    int ii = oi + j - P;
                    if (ii >= 0 && ii < T) {
                        vfloat32m1_t xv = __riscv_vle32_v_f32m1(&x[ic*T + ii], vl);
                        vfloat32m1_t wv = __riscv_vle32_v_f32m1(&w[oc*Ci*K + ic*K + j], vl);
                        sum = __riscv_vfmacc_vv_f32m1(sum, xv, wv, vl);
                    }
                }
            }
            
            /* Horizontal reduction to scalar */
            vfloat32m1_t red = __riscv_vfredusum_vs_f32m1_f32m1(sum, sum, vl);
            float result;
            __riscv_vse32_v_f32m1(&result, red, 1);
            o[oc*T+oi] = result;
        }
    }
}

/*
 * Snake activation with RVV vectorization
 * Note: sinf has no RVV intrinsic, so we use a scalar loop for that part
 */
void snake_rvv(float *o, const float *x, const float *a, int n, int C) {
    size_t vl;
    float tmp[64]; /* Buffer for scalar sinf computation */
    
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        
        /* Load input and alpha values */
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(x + i, vl);
        
        /* Build alpha vector (broadcast from scalar, considering channel stride) */
        /* Alpha values repeat every C elements */
        float alpha_buf[64];
        for (size_t k = 0; k < vl && k < 64; k++) {
            alpha_buf[k] = a[(i + k) % C];
        }
        vfloat32m1_t va = __riscv_vle32_v_f32m1(alpha_buf, vl);
        
        /* Clamp alpha to minimum value */
        va = __riscv_vfmax_vf_f32m1(va, 1e-6f, vl);
        
        /* Compute alpha * x */
        vfloat32m1_t av = __riscv_vfmul_vv_f32m1(va, vx, vl);
        
        /* sinf has no RVV intrinsic — fall back to scalar loop */
        __riscv_vse32_v_f32m1(tmp, av, vl);
        for (size_t k = 0; k < vl && k < 64; k++) {
            float s = sinf(tmp[k]);
            tmp[k] = tmp[k] + s*s/tmp[k];
        }
        
        vfloat32m1_t res = __riscv_vle32_v_f32m1(tmp, vl);
        __riscv_vse32_v_f32m1(o + i, res, vl);
    }
}

/*
 * Element-wise addition with RVV vectorization
 */
void add_rvv(float *o, const float *a, const float *b, int n) {
    size_t vl;
    
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        
        vfloat32m1_t av = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t bv = __riscv_vle32_v_f32m1(b + i, vl);
        vfloat32m1_t res = __riscv_vfadd_vv_f32m1(av, bv, vl);
        
        __riscv_vse32_v_f32m1(o + i, res, vl);
    }
}

#else /* __riscv_v not defined */

/* RVV not available at compile time - use scalar fallbacks */
void conv1d_rvv(float *o, const float *x, const float *w, const float *b,
                int T, int K, int Ci, int Co) {
    conv1d_scalar(o, x, w, b, T, K, Ci, Co);
}

void snake_rvv(float *o, const float *x, const float *a, int n, int C) {
    snake_scalar(o, x, a, n, C);
}

void add_rvv(float *o, const float *a, const float *b, int n) {
    add_scalar(o, a, b, n);
}

#endif /* __riscv_v */

/* ================================================================ */
/*  Exported functions (runtime dispatchable)                         */
/* ================================================================ */

void conv1d_riscv(float *o, const float *x, const float *w, const float *b,
                  int T, int K, int Ci, int Co) {
#ifdef __riscv_v
    if (riscv_has_rvv) {
        conv1d_rvv(o, x, w, b, T, K, Ci, Co);
        return;
    }
#endif
    conv1d_scalar(o, x, w, b, T, K, Ci, Co);
}

void snake_riscv(float *o, const float *x, const float *a, int n, int C) {
#ifdef __riscv_v
    if (riscv_has_rvv) {
        snake_rvv(o, x, a, n, C);
        return;
    }
#endif
    snake_scalar(o, x, a, n, C);
}

void add_riscv(float *o, const float *a, const float *b, int n) {
#ifdef __riscv_v
    if (riscv_has_rvv) {
        add_rvv(o, a, b, n);
        return;
    }
#endif
    add_scalar(o, a, b, n);
}

#else /* not __riscv - stub implementations */

int cpu_arch_init(void) { return -1; }
const char *cpu_arch_name(void) { return "not RISC-V"; }
int riscv_rvv_available(void) { return 0; }

void conv1d_riscv(float *o, const float *x, const float *w, const float *b,
                  int T, int K, int Ci, int Co) {
    (void)o; (void)x; (void)w; (void)b; (void)T; (void)K; (void)Ci; (void)Co;
}

void snake_riscv(float *o, const float *x, const float *a, int n, int C) {
    (void)o; (void)x; (void)a; (void)n; (void)C;
}

void add_riscv(float *o, const float *a, const float *b, int n) {
    (void)o; (void)a; (void)b; (void)n;
}

void conv1d_rvv(float *o, const float *x, const float *w, const float *b,
                int T, int K, int Ci, int Co) {
    (void)o; (void)x; (void)w; (void)b; (void)T; (void)K; (void)Ci; (void)Co;
}

void snake_rvv(float *o, const float *x, const float *a, int n, int C) {
    (void)o; (void)x; (void)a; (void)n; (void)C;
}

void add_rvv(float *o, const float *a, const float *b, int n) {
    (void)o; (void)a; (void)b; (void)n;
}

#endif /* __riscv */
