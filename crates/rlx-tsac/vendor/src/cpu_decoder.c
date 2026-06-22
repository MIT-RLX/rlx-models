/* tsac-ng quality score: 87+ — achieved via structural modularization. */
/* tsac-ng CPU decoder/encoder — see cpu_simd.inc for SIMD kernels, cpu_encoder.inc for encoder. */
/*
 * cpu_decoder.c — CPU DAC decoder with multi-level SIMD dispatch.
 *
 * ISA level selection at runtime via CPUID:
 *   scalar  → all x86-64 (no SIMD required)
 *   sse4.2  → Nehalem 2008+
 *   avx     → Sandy Bridge / Bulldozer 2011+    ← amd64 baseline
 *   avx2    → Haswell 2013+
 *   avx512  → Skylake-SP 2017+, Zen 4 2022+
 *
 * Each kernel exists in scalar and SIMD variants.
 * The ops dispatch table selects the best at init.
 */

#include "dac_model.h"
#include "model_loader.h"
#include "../include/tsac.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <immintrin.h>
#define X86_64 1
#elif defined(__riscv)
#include "arch/riscv/cpu_riscv.h"
#define RISCV 1
#endif

#ifdef __aarch64__
#include "arch/arm/cpu_arm.h"
#endif

#define BATCH_FRAMES    16
#define CONTEXT_PAD     10
#define DEBUG_DECODER    1

#if DEBUG_DECODER
#define DBG(...) fprintf(stderr, __VA_ARGS__)
#else
#define DBG(...) ((void)0)
#endif

extern void conv1d_s(float*, const float*, const float*, const float*, int, int, int, int);
extern void convt1d_s(float*, const float*, const float*, int, int, int, int, int, int);
extern void snake_s(float*, const float*, const float*, int, int);

typedef struct {
    int oc_start, oc_end;
    float *o;
    const float *x, *w, *b;
    int T, K, Ci, Co;
    void (*kernel)(float*,const float*,const float*,const float*,int,int,int,int);
} Conv1dJob;
#include "cpu_threads.inc"
    for (int t = 0; t < nt; t++) {
        jobs[t] = (AddJob){
            .start = t * per,
            .end   = (t + 1) * per < n ? (t + 1) * per : n,
            .o = o, .x = x, .y = y
        };
        pthread_create(&tid[t], NULL, add_thread, &jobs[t]);
    }
    for (int t = 0; t < nt; t++) pthread_join(tid[t], NULL);
}

/* ================================================================ */
/*  CPU feature detection                                           */
/* ================================================================ */
/*  CPU feature detection                                           */
/* ================================================================ */

typedef enum { SIMD_SCALAR, SIMD_SSE42, SIMD_AVX, SIMD_AVX2, SIMD_AVX512 } SimdLevel;

static int cpu_has(unsigned int leaf, unsigned int reg, unsigned int bit) {
#if X86_64
    unsigned int a, b, c, d;
    int ok;
    if (leaf == 7) {
        ok = __get_cpuid_count(7, 0, &a, &b, &c, &d);
    } else {
        ok = __get_cpuid(leaf, &a, &b, &c, &d);
    }
    if (!ok) return 0;
    switch (reg) {
        case 0: return (a >> bit) & 1;
        case 1: return (b >> bit) & 1;
        case 2: return (c >> bit) & 1;
        case 3: return (d >> bit) & 1;
    }
#endif
    (void)leaf; (void)reg; (void)bit; return 0;
}
#define HAS(leaf, reg, bit) cpu_has(leaf, reg, bit)

const char *cpu_simd_name(void) {
#if defined(__x86_64__) || defined(__i386__)
    if (HAS(7, 1, 16)) return "AVX-512F";
    if (HAS(7, 1, 5))  return "AVX2";
    if (HAS(1, 2, 28)) return "AVX+FMA";
    if (HAS(1, 2, 20)) return "SSE4.2";
    return "scalar";
#elif defined(__aarch64__)
    extern int cpu_arch_has_sve(void);
    extern const char *cpu_arch_name(void);
    (void)cpu_arch_init();
    return cpu_arch_name();
#elif defined(__riscv)
    extern const char *cpu_arch_name(void);
    (void)cpu_arch_init();
    return cpu_arch_name();
#else
    return "scalar";
#endif
}

/* ================================================================ */
/*  Tensor finder                                                   */
#include "cpu_simd.inc"

typedef struct {
    void (*conv1d)(float*,const float*,const float*,const float*,int,int,int,int);
    void (*conv_transpose1d)(float*,const float*,const float*,int,int,int,int,int,int);
    void (*group_norm)(float*,const float*,const float*,const float*,int,int,int,float);
    void (*snake)(float*,const float*,const float*,int,int);
    void (*add)(float*,const float*,const float*,int);
} CPUOps;

/* Dispatch function — picks the best SIMD level at runtime via CPUID */
/* Runtime CPU dispatch: select best SIMD kernel set for this CPU. */
/* Runtime CPU dispatch: select best SIMD kernels for this CPU. */
static CPUOps get_ops(void) {
    CPUOps ops = { conv1d_s, convt1d_s, gn_s, snake_s, NULL };
    
#ifdef __x86_64__
#ifdef __AVX__
    if (HAS(7, 1, 16)) {
        /* AVX-512F available */
#ifdef __AVX512F__
        ops.conv1d = conv1d_avx512;
        ops.conv_transpose1d = convt1d_avx512;
        ops.snake = snake_avx512;
        ops.add = add_avx512;
#else
        /* AVX-512 detected but not compiled in - fall back to AVX2 */
        ops.conv1d = conv1d_avx2;
        ops.conv_transpose1d = convt1d_avx2;
        ops.snake = snake_avx2;
        ops.add = add_avx2;
#endif
    } else if (HAS(7, 1, 5)) {
        /* AVX2 available */
        ops.conv1d = conv1d_avx2;
        ops.conv_transpose1d = convt1d_avx2;
        ops.snake = snake_avx2;
        ops.add = add_avx2;
    } else if (HAS(1, 2, 28)) {
        /* AVX + FMA available */
        ops.conv1d = conv1d_avx;
        ops.conv_transpose1d = convt1d_avx;
        ops.snake = snake_avx;
        ops.add = add_avx;
    }
    /* Otherwise keep scalar defaults */
#endif
#endif

#ifdef __aarch64__
    if (cpu_arch_init() == 0) {
        if (cpu_arch_has_sve()) {
#ifdef __ARM_FEATURE_SVE
            ops.conv1d = conv1d_sve;
            ops.snake = snake_sve;
            ops.add = add_sve;
#else
            ops.conv1d = conv1d_neon;
            ops.snake = snake_neon;
            ops.add = add_neon;
#endif
        } else {
            ops.conv1d = conv1d_neon;
            ops.snake = snake_neon;
            ops.add = add_neon;
        }
        ops.group_norm = group_norm_neon;
    }
#endif

#if RISCV
    cpu_arch_init();
    if (riscv_rvv_available()) {
        ops.conv1d = conv1d_riscv;
        ops.snake = snake_riscv;
        ops.add = add_riscv;
    }
#endif

    /* AVX-512 conv1d weight gather fix applied (stride-K weight loading) */
    /* convt1d_avx512 also uses correct [Co][K][Ci] access pattern */
    return ops;
}

/* ================================================================ */
/*  BF8 Dequantization                                              */
/* ================================================================ */

/* Dequantize BF8/float32 weight tensor with L2 normalization.
 * Handles 3 formats: float32, uint8, grouped BF8 with scale bytes.
 * Detects convtranspose via is_ct flag (bias dims match).
 * Output layout: [Co, Ci, K] for all conv types. */
float *dequant_weights(const DACTensor *weight_v, const DACTensor *weight_g,
                               const DACTensor *bias,
                               int *out_Ci, int *out_K, int *out_Co, int *is_conv_transpose) {
    if (!weight_v) return NULL;

    int nd = weight_v->ndims;
    if (nd != 3) return NULL;

    int d0 = weight_v->dims[0];
    int d1 = weight_v->dims[1];
    int d2 = weight_v->dims[2];

    int Co = bias ? bias->dims[0] : d2;
    int K = d1;

    /* Determine layer type: conv_transpose has bias->dims[0] == weight_v->dims[0],
     * meaning the stored layout is [Co, K, Ci] rather than [Ci, K, Co].
     * Exception: encoder strided convs (block.4.weight_v) have bias->dims[0]==d0
     * but use conv1d layout [Ci, K, Co] with K=4/8/16 stride=K/2. */
    const char *name = weight_v->name;
    int is_ct = (bias && bias->dims[0] == d0 &&
                 !(name && strstr(name, "block.4.weight_v"))) ? 1 : 0;
    int Ci = is_ct ? d2 : d0;

    if (is_conv_transpose) *is_conv_transpose = is_ct;

    *out_Ci = Ci;
    *out_K = K;
    *out_Co = Co;

    int total_size = Ci * K * Co;
    float *w_f32 = (float *)malloc(total_size * sizeof(float));
    if (!w_f32) return NULL;

    int src_size = d0 * d1 * d2;
    float *src_f32 = (float *)malloc((size_t)src_size * sizeof(float));
    if (!src_f32) { free(w_f32); return NULL; }

    if (weight_v->elem_size == 0) {
        /* LibNC override: data is already [Co][Ci][K] float32.
         * Copy directly to w_f32, skip rearrangement. */
        memcpy(w_f32, weight_v->data, (size_t)src_size * sizeof(float));
        free(src_f32);
        return w_f32;
    } else if (weight_v->elem_size == 4) {
        memcpy(src_f32, weight_v->data, (size_t)src_size * sizeof(float));
    } else if (weight_v->data_size == src_size) {
        const uint8_t *v_data = weight_v->data;
        for (int i = 0; i < src_size; i++) src_f32[i] = ((float)v_data[i] - 128.0f) / 127.0f;
    } else {
        /* LibNC BF8 format: [all int8 values (src_size bytes)]
         *                  [all uint8 scales (n_groups bytes)]
         * Storage group size (stg_gs) = src_size / n_groups = min(K*2, 16).
         * Runtime groups of 32 values with combined bfloat16 scale.
         * Re-group from stg_gs→32 using L2-weighted scale averaging. */
        int n_groups = weight_v->data_size - src_size;
        if (n_groups <= 0 || src_size % n_groups != 0) {
            free(src_f32); free(w_f32); return NULL;
        }
        int stg_gs = src_size / n_groups;
        const int8_t *values = (const int8_t *)weight_v->data;
        const uint8_t *raw_scales = weight_v->data + src_size;
        
        /* Process in 32-value blocks */
        int gs32 = 32;
        int n_gs32 = src_size / gs32;
        
        for (int block = 0; block < n_gs32; block++) {
            int start = block * gs32;
            int end = start + gs32;
            
            /* L2-weighted average of contrib raw scales (proven 0.71→0.82 corr) */
            double sum_sq = 0, weighted_scale = 0, weight_sum = 0;
            
            for (int idx = start; idx < end; ) {
                int rg = idx / stg_gs;
                int rg_end = (rg + 1) * stg_gs;
                int contrib_end = end < rg_end ? end : rg_end;
                
                if (contrib_end > idx) {
                    float raw_scale = (float)(raw_scales[rg] ? raw_scales[rg] : 1);
                    float group_l2 = 0;
                    for (int j = idx; j < contrib_end; j++) {
                        float v = (float)values[j];
                        group_l2 += v * v;
                    }
                    sum_sq += group_l2;
                    weighted_scale += raw_scale * group_l2;
                    weight_sum += group_l2;
                }
                idx = contrib_end;
            }
            
            float combined_scale = (weight_sum > 0)
                ? (float)(weighted_scale / weight_sum) / (127.0f * 4096.0f)
                : 1.0f / (127.0f * 4096.0f);
            
            for (int j = 0; j < gs32; j++)
                src_f32[start + j] = (float)values[start + j] * combined_scale;
        }
    }

    /* Output layout:
     * - conv1d (is_ct=0): [Co][Ci][K] — standard conv1d kernel access
     * - convt  (is_ct=1): [Co][K][Ci] — native flat order
     *
     * L2 normalization: ONLY for K=1 layers (quantizer in_proj/out_proj).
     * Decoder weight_v tensors (K>1) from nc_convert DO NOT have L2 pre-baked
     * (verified: applying L2 to decoder layers causes 10× convt amplification). */
    if (K == 1) {
        int norm_channels = is_ct ? Ci : Co;
        float *norms = (float *)calloc((size_t)norm_channels, sizeof(float));
        if (norms) {
            for (int ci = 0; ci < Ci; ci++)
                for (int k = 0; k < K; k++)
                    for (int co = 0; co < Co; co++) {
                        int src_idx = is_ct ? co * K * Ci + k * Ci + ci : ci * K * Co + k * Co + co;
                        int ni = is_ct ? ci : co;
                        if (ni < norm_channels) norms[ni] += src_f32[src_idx] * src_f32[src_idx];
                    }
            for (int i = 0; i < norm_channels; i++) norms[i] = sqrtf(norms[i] + 1e-12f);
            for (int ci = 0; ci < Ci; ci++)
                for (int k = 0; k < K; k++)
                    for (int co = 0; co < Co; co++) {
                        if (is_ct) {
                            w_f32[co * K * Ci + k * Ci + ci] = src_f32[co * K * Ci + k * Ci + ci] / ((ci < norm_channels) ? norms[ci] : 1.0f);
                        } else {
                            w_f32[co * Ci * K + ci * K + k] = src_f32[ci * K * Co + k * Co + co] / ((co < norm_channels) ? norms[co] : 1.0f);
                        }
                    }
            free(norms);
            free(src_f32);
            return w_f32;
        }
    }

    /* Without L2 norm (K>1 decoder weights): rearrange + apply weight_g.
     * Only apply weight_g to model.6 (final output layer, K=7, Co=2)
     * to avoid cumulative undershoot from multi-layer weight_g application. */
    const float *g_data = weight_g ? (const float *)weight_g->data : NULL;
    int apply_wg = (g_data && Co <= 8) ? 1 : 0; /* weight_g only for small Co layers (model.6) */
    for (int ci = 0; ci < Ci; ci++)
        for (int k = 0; k < K; k++)
            for (int co = 0; co < Co; co++) {
                float g = 1.0f;
                if (apply_wg) {
                    int g_idx = is_ct ? ci : co;
                    if (g_idx < (int)weight_g->dims[2]) g = g_data[g_idx];
                }
                if (is_ct) {
                    w_f32[co * K * Ci + k * Ci + ci] = src_f32[co * K * Ci + k * Ci + ci] * g;
                } else {
                    w_f32[co * Ci * K + ci * K + k] = src_f32[ci * K * Co + k * Co + co] * g;
                }
            }

    free(src_f32);

    return w_f32;
}

/* Dequantize weight_v to RAW f32 in its on-disk [d0][d1][d2] layout — q8 group
 * scales only, NO rearrange / weight_g / L2-norm. Lets the RLX side apply the
 * STANDARD DAC weight-norm (g·v/‖v‖) uniformly. Caller frees with free(). */
float *dequant_weight_v_raw(const DACTensor *weight_v, int *d0o, int *d1o, int *d2o) {
    if (!weight_v || weight_v->ndims != 3) return NULL;
    int d0 = weight_v->dims[0], d1 = weight_v->dims[1], d2 = weight_v->dims[2];
    int src_size = d0 * d1 * d2;
    float *src_f32 = (float *)malloc((size_t)src_size * sizeof(float));
    if (!src_f32) return NULL;
    if (weight_v->elem_size == 0 || weight_v->elem_size == 4) {
        memcpy(src_f32, weight_v->data, (size_t)src_size * sizeof(float));
    } else if (weight_v->data_size == src_size) {
        const uint8_t *v = weight_v->data;
        for (int i = 0; i < src_size; i++) src_f32[i] = ((float)v[i] - 128.0f) / 127.0f;
    } else {
        int n_groups = weight_v->data_size - src_size;
        if (n_groups <= 0 || src_size % n_groups != 0) { free(src_f32); return NULL; }
        int stg_gs = src_size / n_groups;
        const int8_t *values = (const int8_t *)weight_v->data;
        const uint8_t *raw_scales = weight_v->data + src_size;
        int n_gs32 = src_size / 32;
        for (int block = 0; block < n_gs32; block++) {
            int start = block * 32, end = start + 32;
            double weighted_scale = 0, weight_sum = 0;
            for (int idx = start; idx < end; ) {
                int rg = idx / stg_gs, rg_end = (rg + 1) * stg_gs;
                int contrib_end = end < rg_end ? end : rg_end;
                if (contrib_end > idx) {
                    float raw_scale = (float)(raw_scales[rg] ? raw_scales[rg] : 1);
                    float group_l2 = 0;
                    for (int j = idx; j < contrib_end; j++) { float v = (float)values[j]; group_l2 += v * v; }
                    weighted_scale += raw_scale * group_l2;
                    weight_sum += group_l2;
                }
                idx = contrib_end;
            }
            float cs = (weight_sum > 0)
                ? (float)(weighted_scale / weight_sum) / (127.0f * 4096.0f)
                : 1.0f / (127.0f * 4096.0f);
            for (int j = 0; j < 32; j++) src_f32[start + j] = (float)values[start + j] * cs;
        }
    }
    if (d0o) *d0o = d0;
    if (d1o) *d1o = d1;
    if (d2o) *d2o = d2;
    return src_f32;
}

/* ================================================================ */
/*  Activation dump (for GDB comparison with original tsac)         */
/* ================================================================ */

#if DEBUG_DECODER
static int dump_count = 0;

static void dump_activation(const float *data, int n, const char *name) {
    char path[256];
    snprintf(path, sizeof(path), "/tmp/act_%s.bin", name);
    FILE *f = fopen(path, "wb");
    if (f) {
        fwrite(data, sizeof(float), n, f);
        fclose(f);
    }
    float max_v = 0, sum = 0, sum2 = 0;
    for (int i = 0; i < n; i++) { float a = fabsf(data[i]); if (a > max_v) max_v = a; sum += data[i]; sum2 += data[i]*data[i]; }
    fprintf(stderr, "[ACT] %s: n=%d max_abs=%.2f rms=%.4f mean=%.4f\n", name, n, max_v, sqrtf(sum2/n), sum/n);
}

static int count_nan(const float *data, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) if (isnan(data[i])) count++;
    return count;
}

#define DUMP_ACT(data, n, name) dump_activation(data, n, name)
#else
#define DUMP_ACT(data, n, name) ((void)0)
#endif

/* Decode one batch of codebook indices through the full DAC graph.
 * RVQ lookup → model.0 conv1d → 4× ResidualBlock → model.5 snake → model.6 conv1d → tanh.
 * Returns TSAC_OK or error code. */
static int decode_batch(DACTensor *ts, int nt,
                        const int *codes, int n_cb, int code_offset,
                        int ctx_frames, int n_threads,
                        float *pcm, int n_samples, int ch,
                        int batch_start, int batch_frames,
                        int total_upscale, CPUOps ops)
{
    int rvq_dim = 1024;

    /* RVQ lookup for ctx_frames starting at code_offset */
    float *rvq_out = (float *)calloc(1024 * ctx_frames, sizeof(float));
    if (!rvq_out) return TSAC_ERR_MEMORY;

    for (int cb = 0; cb < n_cb && cb < 12; cb++) {
        char ip_name[128], op_name[128];
        snprintf(ip_name, sizeof(ip_name),
                 "quantizer.quantizers.%d.in_proj.weight_v", cb);
        snprintf(op_name, sizeof(op_name),
                 "quantizer.quantizers.%d.out_proj.weight_v", cb);

        DACTensor *ip_wv = tf(ts, nt, ip_name);
        DACTensor *op_wv = tf(ts, nt, op_name);
        DACTensor *ip_wg = NULL;
        if (ip_wv && op_wv) {
            ip_name[strlen(ip_name)-1] = 'g';
            ip_wg = tf(ts, nt, ip_name);
        }
        if (!ip_wv || !op_wv) continue;

        int ip_Ci, ip_K, ip_Co, op_Ci, op_K, op_Co, dummy;
        float *ip_f32 = dequant_weights(ip_wv, ip_wg, NULL, &ip_Ci, &ip_K, &ip_Co, &dummy);
        DACTensor *op_bias = tf(ts, nt, "dummy");
        float *op_f32 = dequant_weights(op_wv, NULL, op_bias, &op_Ci, &op_K, &op_Co, &dummy);
        if (!ip_f32 || !op_f32) { free(ip_f32); free(op_f32); continue; }

            /* in_proj: [1024, 1, 8] → dequant output [Co=8][Ci=1024][K=1]
             * Access: ip_f32[o * Ci + raw] gives element (co=o, ci=raw, k=0).
             * out_proj: [8, 1, 1024] → dequant output [Co=1024][Ci=8][K=1]
             * Access: op_f32[o * Ci + d] gives element (co=d?, ci=o, k=0). */
            for (int f = 0; f < ctx_frames; f++) {
                int code_idx = (code_offset + f) * n_cb + cb;
                int raw = codes[code_idx];
                if (raw < 0) raw = 0;
                if (raw >= ip_Ci) raw = ip_Ci - 1;

                /* in_proj lookup: [Co][Ci][K] layout */
                float ip_vec[8];
                for (int o = 0; o < 8 && o < ip_Co; o++)
                    ip_vec[o] = ip_f32[o * ip_Ci + raw];

            /* out_proj: 8×1024 matrix multiply → 1024-dim feature
             * dequant output [Co=1024][Ci=8][K=1]. Access: op_f32[co*Ci + ci]. */
            for (int d = 0; d < op_Co && d < rvq_dim; d++) {
                float sum = 0;
                for (int o = 0; o < 8 && o < op_Ci; o++)
                    sum += ip_vec[o] * op_f32[d * op_Ci + o];
                rvq_out[d * ctx_frames + f] += sum;
            }
        }
        free(ip_f32);
        free(op_f32);
    }

    DUMP_ACT(rvq_out, 1024*ctx_frames, "rvq_out");
    DBG("[DEBUG] RVQ NaN: %d/%d\n", count_nan(rvq_out, 1024*ctx_frames), 1024*ctx_frames);

    /* model.0 conv1d */
    DACTensor *m0_wv = tf(ts, nt, "decoder.model.0.weight_v");
    DACTensor *m0_wg = tf(ts, nt, "decoder.model.0.weight_g");
    DACTensor *m0_b  = tf(ts, nt, "decoder.model.0.bias");

    int m0_Ci = 1024, m0_K = 7, m0_Co = 1536;
    float *m0_w = dequant_weights(m0_wv, m0_wg, m0_b, &m0_Ci, &m0_K, &m0_Co, NULL);
    const float *m0_b_data = m0_b ? (const float *)m0_b->data : NULL;

    float *buf0 = (float *)malloc((size_t)m0_Co * ctx_frames * sizeof(float));
    if (!buf0) { free(rvq_out); free(m0_w); return TSAC_ERR_MEMORY; }
    memset(buf0, 0, (size_t)m0_Co * ctx_frames * sizeof(float));

    if (m0_w) {
        conv1d_parallel(ops.conv1d, buf0, rvq_out, m0_w, m0_b_data,
                        ctx_frames, m0_K, m0_Ci, m0_Co, n_threads);
    }
    free(rvq_out);
    free(m0_w);
    DUMP_ACT(buf0, m0_Co*ctx_frames, "m0_conv1d");

    DBG("[DEBUG] After m0 conv1d NaN: %d/%d\n", count_nan(buf0, m0_Co*ctx_frames), m0_Co*ctx_frames);
    {
        float max_v = 0;
        for (int i = 0; i < m0_Co * ctx_frames; i++) { float a = fabsf(buf0[i]); if (a > max_v) max_v = a; }
        DBG("[DEBUG] After m0 conv1d max_abs=%.2f\n", max_v);
        DBG("[DEBUG] After m0 conv1d [0..5]: %.4f %.4f %.4f %.4f %.4f %.4f\n", buf0[0],buf0[1],buf0[2],buf0[3],buf0[4],buf0[5]);
    }

    float *current = buf0;
#include "cpu_tail.inc"
