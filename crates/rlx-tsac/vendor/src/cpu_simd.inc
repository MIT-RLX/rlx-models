/* ================================================================ */

DACTensor *tf(DACTensor *ts, int nt, const char *name) {
    for (int i = 0; i < nt; i++) if (!strcmp(ts[i].name, name)) return &ts[i];
    return NULL;
}

/* ================================================================ */
/*  Scalar kernels (baseline, works on ALL x86-64)                  */
/* ================================================================ */

void conv1d_s(float *o, const float *x, const float *w, const float *b,
              int T, int K, int Ci, int Co) {
    int P = K/2;
    for (int oc = 0; oc < Co; oc++)
        for (int oi = 0; oi < T; oi++) {
            float s = b ? b[oc] : 0;
            for (int ic = 0; ic < Ci; ic++)
                for (int j = 0; j < K; j++) { int ii = oi + j - P;
                    if (ii >= 0 && ii < T) s += x[ic*T+ii] * w[oc*Ci*K + ic*K + j]; }
            o[oc*T+oi] = s;
        }
}

/* Scalar dilated conv1d: supports stride > 1 for temporal reduction. */
static void conv1d_dilated_s(float *o, const float *x, const float *w, const float *b,
                             int T, int K, int Ci, int Co, int dilation) {
    int P = (K/2) * dilation;
    for (int oc = 0; oc < Co; oc++)
        for (int oi = 0; oi < T; oi++) {
            float s = b ? b[oc] : 0;
            for (int ic = 0; ic < Ci; ic++)
                for (int j = 0; j < K; j++) {
                    int ii = oi + j*dilation - P;
                    if (ii >= 0 && ii < T) s += x[ic*T+ii] * w[oc*Ci*K + ic*K + j];
                }
            o[oc*T+oi] = s;
        }
}

/* Scalar strided conv1d (deprecated, use dilated_s for new code). */
static void conv1d_strided_s(float *o, const float *x, const float *w, const float *b,
                              int T_in, int T_out, int K, int Ci, int Co, int stride) {
    int P = K/2; memset(o, 0, (size_t)Co * T_out * sizeof(float));
    for (int oc = 0; oc < Co; oc++)
        for (int oi = 0; oi < T_out; oi++) {
            float s = b ? b[oc] : 0;
            for (int ic = 0; ic < Ci; ic++)
                for (int j = 0; j < K; j++) {
                    int ii = oi * stride + j - P;
                    if (ii >= 0 && ii < T_in)
                        s += x[ic * T_in + ii] * w[oc * Ci * K + ic * K + j];
                }
            o[oc * T_out + oi] = s;
        }
}

void convt1d_s(float *o, const float *x, const float *w,
               int Ti, int To, int K, int Ci, int Co, int stride) {
    /* Weight layout: [Co][K][Ci] — matches nc_conv_transpose_1d's native
     * flat order. Access: w[oc * K * Ci + j * Ci + ic]. */
    int P = K/2;
    memset(o, 0, (size_t)Co*To*sizeof(float));
    for (int ic = 0; ic < Ci; ic++) for (int ii = 0; ii < Ti; ii++) {
        float v = x[ic*Ti+ii]; if (v == 0) continue;
        for (int oc = 0; oc < Co; oc++) for (int j = 0; j < K; j++) {
            int oi = ii*stride + j - P;
            if (oi >= 0 && oi < To)
                o[oc*To+oi] += v * w[oc * K * Ci + j * Ci + ic];
        }
    }
}

void gn_s(float *o, const float *x, const float *w, const float *b,
          int C, int T, int G, float eps) {
    int Cg = C/G;
    int E = Cg*T;
    for (int g = 0; g < G; g++) {
        float s = 0, sq = 0;
        for (int i = 0; i < E; i++) { float v = x[g*E+i]; s += v; sq += v*v; }
        float mn = s/E, vr = sq/E - mn*mn, is = 1.0f/sqrtf(fmaxf(vr+eps, 1e-10f));
        for (int i = 0; i < E; i++) {
            int idx = g*E+i;
            int channel = g*Cg + i/T;
            o[idx] = (x[idx]-mn)*is*(w?w[channel]:1)+(b?b[channel]:0);
        }
    }
}

void snake_s(float *o, const float *x, const float *a, int n, int C) {
    for (int i = 0; i < n; i++) {
        float v = x[i], al = a[i%C];
        if (al < 1e-6f) al = 1e-6f;
        float sa = sinf(al*v);
        o[i] = v + sa*sa/al;
    }
}

/* ================================================================ */
/*  SIMD helper functions                                           */
/* ================================================================ */

#ifdef __AVX__

/* Horizontal sum of 8 floats in __m256 */
static inline float hsum256_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow  = _mm_add_ps(vlow, vhigh);
    vlow  = _mm_hadd_ps(vlow, vlow);
    vlow  = _mm_hadd_ps(vlow, vlow);
    return _mm_cvtss_f32(vlow);
}

#ifdef __AVX512F__
/* Horizontal sum of 16 floats in __m512 using AVX512F */
static inline float hsum512_ps(__m512 v) {
    return _mm512_reduce_add_ps(v);
}
#endif

/* Gather 8 non-contiguous floats from input tensor
 * x: input tensor [Ci, T] layout
 * ic_base: starting input channel (0, 8, 16, ...)
 * ii: time index
 * T: time dimension size
 * Returns: __m256 with x[ic_base* T + ii], x[(ic_base+1)*T + ii], ... */
static inline __m256 gather8(const float *x, int ic_base, int ii, int T, int Ci) {
    float tmp[8];
    for (int k = 0; k < 8 && (ic_base + k) < Ci; k++) {
        tmp[k] = x[(ic_base + k) * T + ii];
    }
    for (int k = (Ci - ic_base); k < 8; k++) {
        tmp[k] = 0.0f;
    }
    return _mm256_loadu_ps(tmp);
}

#ifdef __AVX512F__
/* Gather 16 non-contiguous floats from input tensor for AVX-512 */
static inline __m512 gather16(const float *x, int ic_base, int ii, int T, int Ci) {
    float tmp[16];
    for (int k = 0; k < 16 && (ic_base + k) < Ci; k++) {
        tmp[k] = x[(ic_base + k) * T + ii];
    }
    for (int k = (Ci - ic_base); k < 16; k++) {
        tmp[k] = 0.0f;
    }
    return _mm512_loadu_ps(tmp);
}
#endif

/* ================================================================ */
/*  AVX kernels (8-wide, requires AVX + FMA)                        */
/* ================================================================ */

__attribute__((target("avx,fma")))
void conv1d_avx(float *o, const float *x, const float *w, const float *b,
                int T, int K, int Ci, int Co) {
    int P = K/2;
    int Ci_block = Ci & ~7;  /* Round down to multiple of 8 */
    
    for (int oc = 0; oc < Co; oc++) {
        __m256 bias_vec = b ? _mm256_set1_ps(b[oc]) : _mm256_setzero_ps();
        
        for (int oi = 0; oi < T; oi++) {
            __m256 sum8 = bias_vec;
            
            /* Vectorized loop over input channels (8 at a time) */
            for (int icb = 0; icb < Ci_block; icb += 8) {
                for (int j = 0; j < K; j++) {
                    int ii = oi + j - P;
                    if (ii >= 0 && ii < T) {
                        /* Load 8 input values (non-contiguous across channels) */
                        __m256 x8 = gather8(x, icb, ii, T, Ci);
                        /* Load 8 weights (contiguous) */
                        int w_idx = oc * Ci * K + icb * K + j;
                        __m256 w8 = _mm256_loadu_ps(&w[w_idx]);
                        /* FMA: sum8 += x8 * w8 */
                        sum8 = _mm256_fmadd_ps(x8, w8, sum8);
                    }
                }
            }
            
            /* Horizontal reduce sum8 to scalar */
            float sum = hsum256_ps(sum8);
            
            /* Handle remaining channels (Ci % 8) with scalar fallback */
            for (int ic = Ci_block; ic < Ci; ic++) {
                for (int j = 0; j < K; j++) {
                    int ii = oi + j - P;
                    if (ii >= 0 && ii < T) {
                        sum += x[ic*T+ii] * w[oc*Ci*K + ic*K + j];
                    }
                }
            }
            
            o[oc*T+oi] = sum;
        }
    }
}

__attribute__((target("avx,fma")))
void convt1d_avx(float *o, const float *x, const float *w,
                 int Ti, int To, int K, int Ci, int Co, int stride) {
    int P = K/2;
    int Co_block = Co & ~7;
    memset(o, 0, Co*To*sizeof(float));
    for (int ic = 0; ic < Ci; ic++) {
        for (int ii = 0; ii < Ti; ii++) {
            float v = x[ic*Ti+ii];
            if (v == 0) continue;
            
            __m256 vb = _mm256_set1_ps(v);
            
            /* Vectorized loop over output channels (8 at a time) */
            for (int ocb = 0; ocb < Co_block; ocb += 8) {
                for (int j = 0; j < K; j++) {
                    int oi = ii*stride + j - P;
                    if (oi >= 0 && oi < To) {
                        /* Load 8 weights [Co][K][Ci]: w[ocb:ocb+8, j, ic] */
                        int w_idx = ocb * K * Ci + j * Ci + ic;
                        __m256 w8 = _mm256_loadu_ps(&w[w_idx]);
                        /* Load current output values */
                        __m256 ov = _mm256_loadu_ps(&o[ocb*To + oi]);
                        /* FMA: ov += vb * w8 */
                        ov = _mm256_fmadd_ps(vb, w8, ov);
                        /* Store back */
                        _mm256_storeu_ps(&o[ocb*To + oi], ov);
                    }
                }
            }
            
            /* Handle remaining channels with scalar fallback */
            for (int oc = Co_block; oc < Co; oc++) {
                for (int j = 0; j < K; j++) {
                    int oi = ii*stride + j - P;
                    if (oi >= 0 && oi < To) {
                        o[oc*To+oi] += v * w[oc*K*Ci + j*Ci + ic];
                    }
                }
            }
        }
    }
}

__attribute__((target("avx,fma")))
void snake_avx(float *o, const float *x, const float *a, int n, int C) {
    int n_block = n & ~7;
    
    for (int i = 0; i < n_block; i += 8) {
        /* Load 8 values from x */
        __m256 xv = _mm256_loadu_ps(x + i);
        
        /* Load corresponding alphas (need to handle modulo) */
        float alpha_vals[8];
        for (int k = 0; k < 8; k++) {
            float al = a[(i+k)%C];
            alpha_vals[k] = (al < 1e-6f) ? 1e-6f : al;
        }
        __m256 av = _mm256_loadu_ps(alpha_vals);
        
        /* Compute av * xv for sin input */
        __m256 av_xv = _mm256_mul_ps(av, xv);
        
        /* sinf has no SIMD intrinsic - use scalar fallback */
        float tmp[8];
        _mm256_storeu_ps(tmp, av_xv);
        for (int k = 0; k < 8; k++) {
            float s = sinf(tmp[k]);
            tmp[k] = tmp[k] + s*s / alpha_vals[k];
        }
        __m256 res = _mm256_loadu_ps(tmp);
        
        _mm256_storeu_ps(o + i, res);
    }
    
    /* Handle remaining elements with scalar fallback */
    for (int i = n_block; i < n; i++) {
        float v = x[i], al = a[i%C];
        if (al < 1e-6f) al = 1e-6f;
        float sa = sinf(al*v);
        o[i] = v + sa*sa/al;
    }
}

__attribute__((target("avx,fma")))
void add_avx(float *o, const float *x, const float *y, int n) {
    int n_block = n & ~7;
    
    for (int i = 0; i < n_block; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 yv = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(o + i, _mm256_add_ps(xv, yv));
    }
    
    /* Handle remaining elements */
    for (int i = n_block; i < n; i++) {
        o[i] = x[i] + y[i];
    }
}

/* ================================================================ */
/*  AVX2 kernels (8-wide, requires AVX2 + FMA)                      */
/* ================================================================ */

/* AVX2 kernels - same implementation as AVX but with AVX2 target for better
 * compiler optimizations. The main difference is AVX2 has better horizontal
 * operations and permute support. */

__attribute__((target("avx2,fma")))
void conv1d_avx2(float *o, const float *x, const float *w, const float *b,
                 int T, int K, int Ci, int Co) {
    /* Same as AVX version - AVX2 mainly adds integer ops, 
     * but we use the target attribute for potential compiler optimizations */
    conv1d_avx(o, x, w, b, T, K, Ci, Co);
}

__attribute__((target("avx2,fma")))
void convt1d_avx2(float *o, const float *x, const float *w,
                  int Ti, int To, int K, int Ci, int Co, int stride) {
    convt1d_avx(o, x, w, Ti, To, K, Ci, Co, stride);
}

__attribute__((target("avx2,fma")))
void snake_avx2(float *o, const float *x, const float *a, int n, int C) {
    snake_avx(o, x, a, n, C);
}

__attribute__((target("avx2,fma")))
void add_avx2(float *o, const float *x, const float *y, int n) {
    add_avx(o, x, y, n);
}

/* ================================================================ */
/*  AVX-512 kernels (16-wide, requires AVX-512F)                    */
/* ================================================================ */

#ifdef __AVX512F__

__attribute__((target("avx512f")))
void conv1d_avx512(float *o, const float *x, const float *w, const float *b,
                   int T, int K, int Ci, int Co) {
    int P = K/2;
    int Ci_block = Ci & ~15;  /* Round down to multiple of 16 */
    
    for (int oc = 0; oc < Co; oc++) {
        __m512 bias_vec = b ? _mm512_set1_ps(b[oc]) : _mm512_setzero_ps();
        
        for (int oi = 0; oi < T; oi++) {
            __m512 sum16 = bias_vec;
            
            /* Vectorized loop: for each kernel position j, process 16 input channels.
             * Weight layout [Co][Ci][K]: w[oc][icb+k][j] is at stride K apart.
             * Must gather weights with stride K, NOT contiguous load. */
            for (int j = 0; j < K; j++) {
                int ii = oi + j - P;
                if (ii >= 0 && ii < T) {
                    for (int icb = 0; icb < Ci_block; icb += 16) {
                        __m512 x16 = gather16(x, icb, ii, T, Ci);
                        /* Gather weights: w[oc][icb..icb+15][j] stride=K */
                        float wt[16];
                        for (int k = 0; k < 16; k++)
                            wt[k] = w[oc * Ci * K + (icb + k) * K + j];
                        __m512 w16 = _mm512_loadu_ps(wt);
                        sum16 = _mm512_fmadd_ps(x16, w16, sum16);
                    }
                }
            }
            
            /* Horizontal reduce sum16 to scalar. Bias was broadcast to 16 lanes,
             * so subtract the excess 15×b[oc] to get single bias count. */
            float sum = hsum512_ps(sum16) - (b ? 15.0f * b[oc] : 0.0f);
            
            /* Handle remaining channels (Ci % 16) with scalar fallback */
            for (int ic = Ci_block; ic < Ci; ic++) {
                for (int j = 0; j < K; j++) {
                    int ii = oi + j - P;
                    if (ii >= 0 && ii < T) {
                        sum += x[ic*T+ii] * w[oc*Ci*K + ic*K + j];
                    }
                }
            }
            
            o[oc*T+oi] = sum;
        }
    }
}

__attribute__((target("avx512f")))
void convt1d_avx512(float *o, const float *x, const float *w,
                    int Ti, int To, int K, int Ci, int Co, int stride) {
    int P = K/2;
    int Co_block = Co & ~15;
    memset(o, 0, Co*To*sizeof(float));
    for (int ic = 0; ic < Ci; ic++) {
        for (int ii = 0; ii < Ti; ii++) {
            float v = x[ic*Ti+ii];
            if (v == 0) continue;
            
            __m512 vb = _mm512_set1_ps(v);
            
            /* Vectorized loop over output channels (16 at a time) */
            for (int ocb = 0; ocb < Co_block; ocb += 16) {
                for (int j = 0; j < K; j++) {
                    int oi = ii*stride + j - P;
                    if (oi >= 0 && oi < To) {
                        /* Gather 16 weights for channels ocb..ocb+15.
                         * [Co][K][Ci] layout: stride = K*Ci between channels. */
                        float wt[16];
                        for (int k = 0; k < 16; k++)
                            wt[k] = w[(ocb + k) * K * Ci + j * Ci + ic];
                        __m512 w16 = _mm512_loadu_ps(wt);
                        /* Load current output values */
                        __m512 ov = _mm512_loadu_ps(&o[ocb*To + oi]);
                        /* FMA: ov += vb * w16 */
                        ov = _mm512_fmadd_ps(vb, w16, ov);
                        /* Store back */
                        _mm512_storeu_ps(&o[ocb*To + oi], ov);
                    }
                }
            }
            
            /* Handle remaining channels with scalar fallback */
            for (int oc = Co_block; oc < Co; oc++) {
                for (int j = 0; j < K; j++) {
                    int oi = ii*stride + j - P;
                    if (oi >= 0 && oi < To) {
                        o[oc*To+oi] += v * w[oc*K*Ci + j*Ci + ic];
                    }
                }
            }
        }
    }
}

__attribute__((target("avx512f")))
void snake_avx512(float *o, const float *x, const float *a, int n, int C) {
    int n_block = n & ~15;
    
    for (int i = 0; i < n_block; i += 16) {
        /* Load 16 values from x */
        __m512 xv = _mm512_loadu_ps(x + i);
        
        /* Load corresponding alphas */
        float alpha_vals[16];
        for (int k = 0; k < 16; k++) {
            float al = a[(i+k)%C];
            alpha_vals[k] = (al < 1e-6f) ? 1e-6f : al;
        }
        __m512 av = _mm512_loadu_ps(alpha_vals);
        
        /* Compute av * xv for sin input */
        __m512 av_xv = _mm512_mul_ps(av, xv);
        
        /* sinf has no SIMD intrinsic - use scalar fallback */
        float tmp[16];
        _mm512_storeu_ps(tmp, av_xv);
        for (int k = 0; k < 16; k++) {
            float s = sinf(tmp[k]);
            tmp[k] = tmp[k] + s*s / alpha_vals[k];
        }
        __m512 res = _mm512_loadu_ps(tmp);
        
        _mm512_storeu_ps(o + i, res);
    }
    
    /* Handle remaining elements with scalar fallback */
    for (int i = n_block; i < n; i++) {
        float v = x[i], al = a[i%C];
        if (al < 1e-6f) al = 1e-6f;
        float sa = sinf(al*v);
        o[i] = v + sa*sa/al;
    }
}

__attribute__((target("avx512f")))
void add_avx512(float *o, const float *x, const float *y, int n) {
    int n_block = n & ~15;
    
    for (int i = 0; i < n_block; i += 16) {
        __m512 xv = _mm512_loadu_ps(x + i);
        __m512 yv = _mm512_loadu_ps(y + i);
        _mm512_storeu_ps(o + i, _mm512_add_ps(xv, yv));
    }
    
    /* Handle remaining elements */
    for (int i = n_block; i < n; i++) {
        o[i] = x[i] + y[i];
    }
}

#endif /* __AVX512F__ */

#endif /* __x86_64__ */

/* ================================================================ */
/*  Dispatch table                                                  */
/* ================================================================ */
